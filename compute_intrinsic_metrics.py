#!/usr/bin/env python3
"""
Compute 3 intrinsic metrics from gradient chunks for all VLM settings.

Metrics (from intrinsic_metrics.txt):
  1. ToM Sensitivity Score
     Mean empirical Fisher (squared-gradient) per module group.
  2. ToM Selectivity Score
     Ratio of ToM sensitivity to non-ToM sensitivity per module group,
     plus fraction of top-k parameters with selectivity ratio > threshold.
  3. Localization Concentration
     % of top-k ToM-selective parameters in each module group.

Module groups (3-way split):
  - vision_encoder : vision_layer_* keys
  - cross_modal    : connector_* keys
  - language       : text_layer_* / text_misc keys
"""

import argparse, json, os, sys
import numpy as np
import torch
from pathlib import Path


# =====================================================================
# 3-way group classification from layer key
# =====================================================================

def classify_layer_key(layer_key: str) -> str:
    """Map a chunk layer key to one of 3 groups."""
    if layer_key.startswith("vision_layer"):
        return "vision_encoder"
    if layer_key.startswith("connector"):
        return "cross_modal"
    return "language"  # text_layer_*, text_misc


# =====================================================================
# Load all gradient values grouped by module family
# =====================================================================

def load_gradients_by_group(chunk_dir: str):
    """Load all gradient chunks and return per-group flat tensors.

    Returns {group_name: 1-D torch.Tensor of all gradient values}.
    """
    manifest_path = os.path.join(chunk_dir, "manifest.json")
    with open(manifest_path) as f:
        manifest = json.load(f)

    group_tensors = {"vision_encoder": [], "cross_modal": [], "language": []}

    for layer_key, info in manifest["layers"].items():
        pt_path = os.path.join(chunk_dir, f"{layer_key}.pt")
        if not os.path.exists(pt_path):
            continue
        blob = torch.load(pt_path, map_location="cpu")
        group = classify_layer_key(layer_key)
        for mod_name in info["modules"]:
            if mod_name in blob:
                group_tensors[group].append(blob[mod_name].float().reshape(-1))

    # Concatenate
    result = {}
    for g, tensors in group_tensors.items():
        if tensors:
            result[g] = torch.cat(tensors)
        else:
            result[g] = torch.tensor([], dtype=torch.float32)
    return result


# =====================================================================
# Metric 1: ToM Sensitivity Score
# =====================================================================

def compute_sensitivity_scores(tom_grads: dict, n_samples: int = 100) -> dict:
    """Mean empirical Fisher per group.

    s_i = (1/n) * sum_j (dL/dtheta_i)^2
    The chunks already store the accumulated squared gradients (sum over samples),
    so mean = sum / n_samples.

    Returns {group: {'mean_sensitivity': float, 'mean_topk_sensitivity': float,
                     'n_params': int}}.
    """
    results = {}
    for group, t in tom_grads.items():
        n = t.numel()
        if n == 0:
            results[group] = {"mean_sensitivity": 0.0, "mean_topk_sensitivity": 0.0,
                              "n_params": 0}
            continue
        mean_sens = (t.sum() / (n * n_samples)).item()
        # Mean of top-1% parameters
        k = max(1, int(n * 0.01))
        topk_vals = t.topk(k).values
        mean_topk = (topk_vals.sum() / (k * n_samples)).item()
        results[group] = {"mean_sensitivity": mean_sens,
                          "mean_topk_sensitivity": mean_topk,
                          "n_params": n}
    return results


# =====================================================================
# Metric 2: ToM Selectivity Score
# =====================================================================

def compute_selectivity_scores(tom_grads: dict, nontom_grads: dict,
                                eps: float = 1e-10,
                                threshold: float = 2.0) -> dict:
    """Per-group selectivity: ToM / nonToM ratio and fraction above threshold.

    Returns {group: {'mean_selectivity': float, 'mean_topk_selectivity': float,
                     'frac_above_threshold': float, 'n_params': int}}.
    """
    results = {}
    for group in tom_grads:
        t = tom_grads[group]
        nt = nontom_grads.get(group, torch.tensor([]))
        n = t.numel()
        if n == 0 or nt.numel() != n:
            results[group] = {"mean_selectivity": float("nan"),
                              "mean_topk_selectivity": float("nan"),
                              "frac_above_threshold": float("nan"),
                              "n_params": n}
            continue

        # Per-parameter selectivity ratio
        sel = t / (nt + eps)

        mean_sel = sel.mean().item()

        # Mean selectivity of top-1% most ToM-sensitive parameters
        k = max(1, int(n * 0.01))
        topk_idx = t.topk(k).indices
        mean_topk_sel = sel[topk_idx].mean().item()

        # Fraction of parameters with selectivity > threshold
        frac = (sel > threshold).float().mean().item()

        results[group] = {"mean_selectivity": mean_sel,
                          "mean_topk_selectivity": mean_topk_sel,
                          "frac_above_threshold": frac,
                          "n_params": n}
    return results


# =====================================================================
# Metric 3: Localization Concentration
# =====================================================================

def compute_localization(tom_grads: dict, nontom_grads: dict,
                         k_frac: float = 0.001,
                         eps: float = 1e-10) -> dict:
    """Fraction of top-k ToM-selective parameters in each group.

    1. Concatenate all parameters across groups.
    2. Compute per-parameter selectivity = tom / (nontom + eps).
    3. Take top-k by selectivity.
    4. Count how many top-k params fall in each group.

    Returns {group: {'frac_topk': float, 'count_topk': int, 'n_params': int}}.
    Also returns 'concentration_index' (max fraction) and 'entropy'.
    """
    # Build global arrays with group labels
    all_sel = []
    all_group = []
    groups = sorted(tom_grads.keys())

    for g in groups:
        t = tom_grads[g]
        nt = nontom_grads.get(g, torch.zeros_like(t))
        if t.numel() == 0:
            continue
        sel = t / (nt + eps)
        all_sel.append(sel)
        all_group.extend([g] * t.numel())

    if not all_sel:
        return {g: {"frac_topk": 0.0, "count_topk": 0, "n_params": 0} for g in groups}

    all_sel_cat = torch.cat(all_sel)
    total_params = all_sel_cat.numel()
    k = max(1, int(total_params * k_frac))

    topk_indices = all_sel_cat.topk(k).indices.numpy()

    # Convert group labels to array for indexing
    group_arr = np.array(all_group)
    topk_groups = group_arr[topk_indices]

    results = {}
    for g in groups:
        count = int((topk_groups == g).sum())
        n = tom_grads[g].numel()
        results[g] = {"frac_topk": count / k, "count_topk": count, "n_params": n}

    # Also for groups with no params
    for g in ["vision_encoder", "cross_modal", "language"]:
        if g not in results:
            results[g] = {"frac_topk": 0.0, "count_topk": 0, "n_params": 0}

    # Concentration index and entropy
    fracs = [results[g]["frac_topk"] for g in ["vision_encoder", "cross_modal", "language"]]
    results["concentration_index"] = max(fracs)
    # Shannon entropy (bits)
    ent = 0.0
    for f in fracs:
        if f > 0:
            ent -= f * np.log2(f)
    results["entropy"] = ent

    return results


# =====================================================================
# Main
# =====================================================================

SETTINGS = [
    ("llava_multimodal",        "LLaVA 1.5 (text+video)"),
    ("instructblip_multimodal", "InstructBLIP (text+video)"),
    ("qwen3vl_multimodal",      "Qwen3-VL (text+video)"),
    ("qwen3vl_text_only",       "Qwen3-VL (text-only)"),
    ("qwen3vl_video_only",      "Qwen3-VL (video-only)"),
]

GROUP_LABELS = {
    "vision_encoder": "Vision Encoder",
    "cross_modal": "Cross-Modal",
    "language": "Language Backbone",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_root", required=True)
    ap.add_argument("--out", default=None, help="Output JSON path")
    ap.add_argument("--n_samples", type=int, default=100)
    ap.add_argument("--selectivity_threshold", type=float, default=2.0)
    ap.add_argument("--topk_frac", type=float, default=0.001,
                    help="Fraction for top-k localization (default 0.1%%)")
    args = ap.parse_args()

    root = Path(args.results_root)
    out_path = args.out or str(root / "intrinsic_metrics.json")

    all_results = {}

    for key, label in SETTINGS:
        tom_dir = root / key / "chunks_tom"
        nontom_dir = root / key / "chunks_nontom"
        if not (tom_dir.exists() and nontom_dir.exists()):
            print(f"[SKIP] {key}: chunks not found")
            continue

        print(f"\n=== {label} ===")
        tom_grads = load_gradients_by_group(str(tom_dir))
        nontom_grads = load_gradients_by_group(str(nontom_dir))

        for g in ["vision_encoder", "cross_modal", "language"]:
            print(f"  {g}: {tom_grads[g].numel():,} params")

        # Metric 1: Sensitivity
        sens = compute_sensitivity_scores(tom_grads, n_samples=args.n_samples)
        print(f"  Sensitivity: " + ", ".join(
            f"{g}={sens[g]['mean_topk_sensitivity']:.6e}" for g in GROUP_LABELS))

        # Metric 2: Selectivity
        sel = compute_selectivity_scores(tom_grads, nontom_grads,
                                          threshold=args.selectivity_threshold)
        print(f"  Selectivity: " + ", ".join(
            f"{g}={sel[g]['mean_topk_selectivity']:.4f}" for g in GROUP_LABELS))

        # Metric 3: Localization
        loc = compute_localization(tom_grads, nontom_grads,
                                   k_frac=args.topk_frac)
        print(f"  Localization: " + ", ".join(
            f"{g}={loc[g]['frac_topk']:.4f}" for g in GROUP_LABELS))
        print(f"  Concentration={loc['concentration_index']:.4f}, Entropy={loc['entropy']:.4f}")

        all_results[key] = {
            "label": label,
            "sensitivity": sens,
            "selectivity": sel,
            "localization": loc,
        }

    # Save
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n[DONE] Saved to {out_path}")


if __name__ == "__main__":
    main()
