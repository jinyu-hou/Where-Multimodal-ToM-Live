"""
Create a non-ToM augmented dataset from MMToM-QA questions.

For each sampled ToM question, produce a factual (non-ToM) counterpart that:
  - uses the same scene description and video (same episode/end_time)
  - replaces mental-state reasoning with factual questions about the physical scene

Type 1.x  (belief):  remove "X thinks that" → ask about actual object location
Type 2.1  (goal+belief cond): extract container from conditional → ask what's inside it
Type 2.2–2.4 (goal): find a container in the scene that disambiguates the two objects
"""

import json
import random
import re
import sys
from pathlib import Path

QUESTIONS_PATH = "/home/jinyuhou/MMToM-QA/Benchmark/questions.json"
OUTPUT_DIR = Path(__file__).resolve().parent / "data"
SAMPLED_TOM_PATH = OUTPUT_DIR / "sampled_tom_questions.jsonl"
NON_TOM_PATH = OUTPUT_DIR / "sampled_non_tom_questions.jsonl"

N_SAMPLES = 100
SEED = 42


# ---------------------------------------------------------------------------
# Scene parsing utilities
# ---------------------------------------------------------------------------

def extract_scene_text(full_text: str) -> str:
    """Return the apartment-description portion of the question text."""
    m = re.search(
        r"What's inside the apartment:(.*?)Actions taken by",
        full_text,
        re.DOTALL,
    )
    return m.group(1) if m else ""


def _parse_object_list(text: str) -> list[str]:
    """Split a natural-language list like 'a plate, two apples, and a wine glass'."""
    items = re.split(r",\s*(?:and\s+)?|\s+and\s+", text)
    objects = []
    for item in items:
        item = item.strip()
        item = re.sub(
            r"^(?:a |an |the |two |three |four |another |some )", "", item
        )
        if item:
            objects.append(item.lower().strip())
    return objects


def parse_container_objects(scene_text: str) -> dict[str, list[str]]:
    """Build a mapping  container_name → [object, …]  from the scene text."""
    containers: dict[str, list[str]] = {}
    lower = scene_text.lower()

    # "Inside the {container}, there is/are {list}."
    for m in re.finditer(
        r"inside the ([\w\s]+?),\s+there (?:is|are)\s+([\w\s,]+?)\.", lower
    ):
        key = m.group(1).strip()
        containers.setdefault(key, []).extend(_parse_object_list(m.group(2)))

    # "The {container} contains/holds/houses/has {list}."
    for m in re.finditer(
        r"the ([\w\s]+?) (?:contains|holds|houses|has)\s+([\w\s,]+?)\.", lower
    ):
        key = m.group(1).strip()
        containers.setdefault(key, []).extend(_parse_object_list(m.group(2)))

    # "{list} placed on / resting on the {container}."
    for m in re.finditer(
        r"(?:with |,\s*)([\w\s,]+?) (?:placed|resting) on the ([\w\s]+?)\.", lower
    ):
        key = m.group(2).strip()
        containers.setdefault(key, []).extend(_parse_object_list(m.group(1)))

    # "On the {container}, there is/are {list}." or "On the {container}, {list}."
    for m in re.finditer(
        r"on the ([\w\s]+?),\s+(?:there (?:is|are)\s+)?([\w\s,]+?)\.", lower
    ):
        key = m.group(1).strip()
        containers.setdefault(key, []).extend(_parse_object_list(m.group(2)))

    # "The {container} is adorned with {list}."
    for m in re.finditer(
        r"the ([\w\s]+?) is adorned with\s+([\w\s,]+?)\.", lower
    ):
        key = m.group(1).strip()
        containers.setdefault(key, []).extend(_parse_object_list(m.group(2)))

    # "The {container} is storing {list}."
    for m in re.finditer(
        r"the ([\w\s]+?) is storing\s+([\w\s,]+?)\.", lower
    ):
        key = m.group(1).strip()
        containers.setdefault(key, []).extend(_parse_object_list(m.group(2)))

    # "there is a {object} in the {container}"  (scattered occurrences)
    for m in re.finditer(
        r"there is (?:a |an )?([\w\s]+?) in the ([\w\s]+?)(?:\.|,)", lower
    ):
        key = m.group(2).strip()
        containers.setdefault(key, []).extend(_parse_object_list(m.group(1)))

    # "you'll find {list}" inside context of a container sentence
    for m in re.finditer(
        r"(?:inside the |in the )([\w\s]+?),\s+you(?:'ll| will) find\s+([\w\s,]+?)\.", lower
    ):
        key = m.group(1).strip()
        containers.setdefault(key, []).extend(_parse_object_list(m.group(2)))

    return containers


def _fix_article(text: str) -> str:
    """Fix 'A apple' → 'An apple' etc.  Avoids 'either a or b'."""
    return re.sub(
        r"\b(A|a) ([aeiouAEIOU]\w{2,})",
        lambda m: ("An" if m.group(1)[0].isupper() else "an") + " " + m.group(2),
        text,
    )


def _obj_match(obj_name: str, obj_list: list[str]) -> bool:
    """Fuzzy-check whether obj_name appears in a list of parsed object strings."""
    obj = obj_name.lower().strip()
    for o in obj_list:
        if obj in o or o in obj:
            return True
    return False


def find_disambiguating_container(
    containers: dict[str, list[str]], obj_a: str, obj_b: str
) -> tuple[str, str] | None:
    """Find a container where exactly one of (obj_a, obj_b) is present.

    Returns (container_name, which_object)  — "a" or "b".
    """
    for cname, objs in containers.items():
        a_in = _obj_match(obj_a, objs)
        b_in = _obj_match(obj_b, objs)
        if a_in and not b_in:
            return cname, "a"
        if b_in and not a_in:
            return cname, "b"
    return None


# ---------------------------------------------------------------------------
# Augmentation functions
# ---------------------------------------------------------------------------

def augment_belief(q: dict) -> dict:
    """Type 1.x: convert belief question → factual object-location question."""
    out = q.copy()
    text = q["question"]
    scene = extract_scene_text(text)

    # Split context from question part
    parts = text.rsplit("\nQuestion: ", 1)
    if len(parts) < 2:
        parts = text.rsplit("Question: ", 1)
    context = parts[0]
    qpart = parts[1]

    # Extract object and container from option (a)
    # Pattern: "(a) {Person} thinks that the {object} is [not] inside the {container}."
    m_a = re.search(
        r"\(a\)\s+\w+\s+thinks that the (.+?) is (not )?inside the (.+?)\.", qpart
    )
    m_b = re.search(
        r"\(b\)\s+\w+\s+thinks that the (.+?) is (not )?inside the (.+?)\.", qpart
    )
    if not m_a or not m_b:
        # Fallback: return as-is with original answer
        return out

    obj = m_a.group(1).strip()
    container = m_a.group(3).strip()
    a_positive = m_a.group(2) is None   # (a) says "is inside" (no "not")

    # Determine factual answer: is the object actually inside the container?
    containers = parse_container_objects(scene)
    actually_inside = False
    for cname, objs in containers.items():
        if container.lower() in cname or cname in container.lower():
            if _obj_match(obj, objs):
                actually_inside = True
                break

    # Build factual question
    new_q = (
        f"Based on the apartment description, which one of the following "
        f"statements is true? "
        f"(a) The {obj} is {'not ' if not a_positive else ''}inside the {container}. "
        f"(b) The {obj} is {'not ' if a_positive else ''}inside the {container}. "
        f"Please respond with either a or b."
    )

    out["question"] = context + "\nQuestion: " + new_q

    # Determine new answer
    if a_positive:
        # (a) = "is inside", (b) = "is not inside"
        out["answer"] = "a" if actually_inside else "b"
    else:
        # (a) = "is not inside", (b) = "is inside"
        out["answer"] = "b" if actually_inside else "a"

    return out


def augment_goal_with_conditional(q: dict) -> dict:
    """Type 2.1: extract container from 'If X thinks there isn't Y in Z' conditional."""
    out = q.copy()
    text = q["question"]
    scene = extract_scene_text(text)

    parts = text.rsplit("\nQuestion: ", 1)
    if len(parts) < 2:
        parts = text.rsplit("Question: ", 1)
    context = parts[0]
    qpart = parts[1]

    # Extract objects from options
    m_a = re.search(
        r"\(a\)\s+\w+\s+has been trying to get (?:a |an )?([\w\s]+?)\.", qpart
    )
    m_b = re.search(
        r"\(b\)\s+\w+\s+has been trying to get (?:a |an )?([\w\s]+?)\.", qpart
    )
    if not m_a or not m_b:
        return out

    obj_a = m_a.group(1).strip()
    obj_b = m_b.group(1).strip()

    # Extract container from the conditional: "If X think(s) there isn't a Y inside the Z"
    cond_match = re.search(
        r"If \w+ thinks? there isn't (?:a |an )?.+? inside the (.+?),", qpart
    )
    containers = parse_container_objects(scene)

    if cond_match:
        target_container = cond_match.group(1).strip()
        # Normalize ordinal references: "3rd kitchen cabinet" → search containers
        target_lower = target_container.lower()

        # Determine which option object is in this container
        a_in = False
        b_in = False
        for cname, objs in containers.items():
            if target_lower in cname or cname in target_lower:
                a_in = a_in or _obj_match(obj_a, objs)
                b_in = b_in or _obj_match(obj_b, objs)

        if a_in != b_in:
            new_q = (
                f"Based on the apartment description, which one of the following "
                f"objects is inside the {target_container}? "
                f"(a) A {obj_a}. (b) A {obj_b}. "
                f"Please respond with either a or b."
            )
            out["question"] = context + "\nQuestion: " + new_q
            out["answer"] = "a" if a_in else "b"
            return out

    # Fallback: find any disambiguating container
    return _augment_goal_by_container_search(out, context, obj_a, obj_b, containers)


def augment_goal_no_conditional(q: dict) -> dict:
    """Type 2.2–2.4: convert goal question → factual object-in-container question."""
    out = q.copy()
    text = q["question"]
    scene = extract_scene_text(text)

    parts = text.rsplit("\nQuestion: ", 1)
    if len(parts) < 2:
        parts = text.rsplit("Question: ", 1)
    context = parts[0]
    qpart = parts[1]

    m_a = re.search(
        r"\(a\)\s+\w+\s+has been trying to get (?:a |an )?([\w\s]+?)\.", qpart
    )
    m_b = re.search(
        r"\(b\)\s+\w+\s+has been trying to get (?:a |an )?([\w\s]+?)\.", qpart
    )
    if not m_a or not m_b:
        return out

    obj_a = m_a.group(1).strip()
    obj_b = m_b.group(1).strip()
    containers = parse_container_objects(scene)

    return _augment_goal_by_container_search(out, context, obj_a, obj_b, containers)


def _augment_goal_by_container_search(
    out: dict, context: str, obj_a: str, obj_b: str,
    containers: dict[str, list[str]],
) -> dict:
    """Shared fallback: find a disambiguating container and build a factual question."""
    result = find_disambiguating_container(containers, obj_a, obj_b)
    if result:
        cname, which = result
        new_q = (
            f"Based on the apartment description, which one of the following "
            f"objects is inside the {cname}? "
            f"(a) A {obj_a}. (b) A {obj_b}. "
            f"Please respond with either a or b."
        )
        out["question"] = context + "\nQuestion: " + new_q
        out["answer"] = which
    else:
        # Last resort: ask which object appears first in the description
        scene_lower = context.lower()
        pos_a = scene_lower.find(obj_a.lower())
        pos_b = scene_lower.find(obj_b.lower())
        if pos_a >= 0 and pos_b >= 0:
            first = "a" if pos_a < pos_b else "b"
        else:
            first = "a" if pos_a >= 0 else "b"
        new_q = (
            f"Based on the apartment description, which one of the following "
            f"objects is mentioned first? "
            f"(a) A {obj_a}. (b) A {obj_b}. "
            f"Please respond with either a or b."
        )
        out["question"] = context + "\nQuestion: " + new_q
        out["answer"] = first

    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def augment_question(q: dict) -> dict:
    qt = q["question_type"]
    if qt in (1.1, 1.2, 1.3):
        out = augment_belief(q)
    elif qt == 2.1:
        out = augment_goal_with_conditional(q)
    else:  # 2.2, 2.3, 2.4
        out = augment_goal_no_conditional(q)
    out["question"] = _fix_article(out["question"])
    return out


def main():
    # Load all questions
    questions = []
    with open(QUESTIONS_PATH) as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))
    print(f"Loaded {len(questions)} questions from {QUESTIONS_PATH}")

    # Sample deterministically
    rng = random.Random(SEED)
    sampled = rng.sample(questions, min(N_SAMPLES, len(questions)))
    print(f"Sampled {len(sampled)} questions (seed={SEED})")

    # Print type distribution
    from collections import Counter
    type_counts = Counter(q["question_type"] for q in sampled)
    print(f"Type distribution: {dict(sorted(type_counts.items()))}")

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save sampled ToM questions
    with open(SAMPLED_TOM_PATH, "w") as f:
        for q in sampled:
            f.write(json.dumps(q) + "\n")
    print(f"Saved sampled ToM questions to {SAMPLED_TOM_PATH}")

    # Augment to non-ToM
    augmented = []
    n_changed = 0
    for q in sampled:
        aq = augment_question(q)
        augmented.append(aq)
        if aq["question"] != q["question"]:
            n_changed += 1

    print(f"Augmented {n_changed}/{len(sampled)} questions successfully")

    # Save non-ToM questions
    with open(NON_TOM_PATH, "w") as f:
        for q in augmented:
            f.write(json.dumps(q) + "\n")
    print(f"Saved non-ToM questions to {NON_TOM_PATH}")

    # Show a few examples
    print("\n=== Sample augmentations ===")
    for i, (orig, aug) in enumerate(zip(sampled, augmented)):
        if orig["question"] != aug["question"]:
            qt = orig["question_type"]
            orig_q = orig["question"].rsplit("Question: ", 1)[-1]
            aug_q = aug["question"].rsplit("Question: ", 1)[-1]
            print(f"\n[Type {qt}] Original (ans={orig['answer']}):")
            print(f"  {orig_q[:200]}")
            print(f"Non-ToM (ans={aug['answer']}):")
            print(f"  {aug_q[:200]}")
            if i > 4:
                break


if __name__ == "__main__":
    main()
