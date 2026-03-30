#!/usr/bin/env python3
"""
VLM ToM evaluation with gradient-based masking.

Adapted from ToM_and_perplexity_evaluation.py for Vision-Language Models.

Pipeline per m-value:
  1. Load base VLM.
  2. Load chunked gradient files (ToM, non-ToM).
  3. Apply differential mask:  mask = (top-m% ToM) & ~(top-m% non-ToM).
  4. Zero out (replace with group-mean) masked weights.
  5. Evaluate on MMToM-QA (100 questions) → accuracy.
  6. Record result.

Outputs:
  <out_dir>/eval_results.csv   – columns: m, accuracy, n_correct, n_total
"""

import argparse, gc, json, os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from vlm_utils import (
    load_vlm, detect_family,
    load_questions, get_video_frames,
    prepare_input, generate_answer,
    get_all_linear_modules,
    _strip_scene_and_actions,
)


# =====================================================================
# Mask application (in-memory, no checkpoint saving)
# =====================================================================

def _walk_to_module(model, dotted_name: str):
    parts = dotted_name.split(".")
    mod = model
    for p in parts:
        mod = getattr(mod, p)
    return mod


def apply_mask_inplace(model, family: str,
                       grad_tom_dir: str, grad_nontom_dir: str,
                       m_value: float, scale: float = 0.0):
    """Apply the differential mask in-place to the model weights.

    For each chunk file, for each module inside:
      1. Load ToM and non-ToM squared-gradient tensors.
      2. Compute top-m% thresholds independently.
      3. mask = (in top-m% of ToM) AND (NOT in top-m% of non-ToM)
      4. Replace masked weights with group-mean (when scale=0).
    """
    manifest_path = os.path.join(grad_tom_dir, "manifest.json")
    with open(manifest_path) as f:
        manifest = json.load(f)

    for layer_key, info in manifest["layers"].items():
        tom_path = os.path.join(grad_tom_dir, f"{layer_key}.pt")
        nontom_path = os.path.join(grad_nontom_dir, f"{layer_key}.pt")
        if not (os.path.exists(tom_path) and os.path.exists(nontom_path)):
            continue

        g_tom = torch.load(tom_path, map_location="cpu")
        g_nontom = torch.load(nontom_path, map_location="cpu")

        for short_name in info["modules"]:
            if short_name not in g_tom or short_name not in g_nontom:
                continue

            gw_tom = g_tom[short_name].float()
            gw_nontom = g_nontom[short_name].float()

            num_outliers = int(gw_tom.numel() * m_value)
            if num_outliers <= 0:
                continue

            thr_tom = gw_tom.reshape(-1).topk(k=num_outliers).values[-1]
            thr_nontom = gw_nontom.reshape(-1).topk(k=num_outliers).values[-1]
            mask_tom = gw_tom > thr_tom
            mask_nontom = gw_nontom > thr_nontom
            mask = mask_tom & (~mask_nontom)

            if mask.sum() == 0:
                continue

            # Find the actual weight tensor in the model
            # We need to resolve short_name → full_name
            # Short names are like "self_attn.q_proj"; full names stored during chunking
            # Instead, we search all linear modules
            full_name = _resolve_full_name(model, layer_key, short_name, family)
            if full_name is None:
                continue

            mod = _walk_to_module(model, full_name)
            w = mod.weight.data
            dev = w.device
            mask = mask.to(dev)

            no_outlier = w * (~mask)
            mean_no = no_outlier.mean()
            new_diff = w - mean_no
            scaled = new_diff * scale
            adjusted = scaled + mean_no
            final = no_outlier + adjusted * mask
            w.copy_(final)


def _resolve_full_name(model, layer_key: str, short_name: str, family: str) -> str | None:
    """Resolve a (layer_key, short_name) pair back to a full dotted module name."""
    # Build lookup once if not cached
    if not hasattr(_resolve_full_name, "_cache"):
        _resolve_full_name._cache = {}

    cache_key = id(model)
    if cache_key not in _resolve_full_name._cache:
        from vlm_utils import get_all_linear_modules
        lookup = {}
        import re as _re
        for full_name in get_all_linear_modules(model):
            from chunk_gradient_vlm import _extract_layer_key
            lk = _extract_layer_key(full_name, family)
            parts = full_name.rsplit(".", 2)
            sn = ".".join(parts[-2:]) if len(parts) >= 2 else parts[-1]
            lookup[(lk, sn)] = full_name
        _resolve_full_name._cache[cache_key] = lookup

    return _resolve_full_name._cache[cache_key].get((layer_key, short_name))


# =====================================================================
# Evaluation
# =====================================================================

def evaluate_mmtomqa(model, processor, family: str,
                     questions: list, video_dir: str,
                     mode: str = "multimodal",
                     max_frames: int = 8) -> dict:
    """Run zero-shot evaluation on MMToM-QA questions.

    Returns dict with keys: accuracy, n_correct, n_total, per_type.
    """
    correct = 0
    per_type = {}

    for q in tqdm(questions, desc="Evaluating"):
        if mode == "text_only":
            images = None
            q_text = q["question"]
        elif mode == "video_only":
            images = get_video_frames(video_dir, q["episode"],
                                      q["end_time"], max_frames)
            q_text = _strip_scene_and_actions(q["question"])
        else:
            images = get_video_frames(video_dir, q["episode"],
                                      q["end_time"], max_frames)
            q_text = q["question"]

        pred = generate_answer(model, processor, family, q_text, images)
        is_correct = (pred == q["answer"])
        correct += int(is_correct)

        qt = q.get("question_type", "unknown")
        per_type.setdefault(qt, {"correct": 0, "total": 0})
        per_type[qt]["total"] += 1
        per_type[qt]["correct"] += int(is_correct)

    return {
        "accuracy": correct / len(questions) if questions else 0.0,
        "n_correct": correct,
        "n_total": len(questions),
        "per_type": per_type,
    }


# =====================================================================
# CLI
# =====================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Base VLM model id")
    ap.add_argument("--grad_tom_chunks", required=True)
    ap.add_argument("--grad_nontom_chunks", required=True)
    ap.add_argument("--questions", required=True, help="JSONL with MMToM-QA questions")
    ap.add_argument("--video_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--cache_dir", default=None)
    ap.add_argument("--mode", choices=["multimodal", "text_only", "video_only"],
                    default="multimodal")
    ap.add_argument("--max_frames", type=int, default=8)
    ap.add_argument("--scale", type=float, default=0.0)
    ap.add_argument("--m_list", type=str, default="0.0,5e-6,1e-5,2e-5,5e-5",
                    help="Comma-separated list of m values for masking sweep")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    questions = load_questions(args.questions)
    m_values = [float(x) for x in args.m_list.split(",")]
    family = detect_family(args.model)

    rows = []
    for m in m_values:
        print(f"\n=== m={m} ===")

        # Load fresh model for each m
        model, processor, _ = load_vlm(args.model, cache_dir=args.cache_dir)
        model.eval()

        if m > 0:
            print(f"[mask] Applying differential mask m={m} ...")
            apply_mask_inplace(model, family,
                               args.grad_tom_chunks, args.grad_nontom_chunks,
                               m, scale=args.scale)

        result = evaluate_mmtomqa(model, processor, family, questions,
                                  args.video_dir, mode=args.mode,
                                  max_frames=args.max_frames)

        print(f"[eval] m={m}  accuracy={result['accuracy']:.4f} "
              f"({result['n_correct']}/{result['n_total']})")
        rows.append({"m": m, "accuracy": result["accuracy"],
                     "n_correct": result["n_correct"],
                     "n_total": result["n_total"]})

        # save per-type breakdown
        pt_path = os.path.join(args.out_dir, f"per_type_m{m}.json")
        with open(pt_path, "w") as f:
            json.dump(result["per_type"], f, indent=2)

        del model, processor
        gc.collect()
        torch.cuda.empty_cache()

    # save sweep results
    csv_path = os.path.join(args.out_dir, "eval_results.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"\n[DONE] Results saved to {csv_path}")


if __name__ == "__main__":
    main()
