#!/usr/bin/env python3
"""
Generate a final report for VLM Theory-of-Mind analysis.

Reads:
  - Evaluation CSVs (eval_results.csv per setting)
  - Gradient chunk files (to compute sensitivity metrics)

Outputs:
  - A text/markdown report with:
    * Final ToM accuracy per setting (m=0 baseline)
    * Masking-sweep table (accuracy vs m) per setting
    * Relative sensitivity: text vs vision (one number each per setting)
    * Cross-model comparison table
"""

import argparse, json, os, sys
import numpy as np
import pandas as pd
import torch
from pathlib import Path


# =====================================================================
# Sensitivity computation
# =====================================================================

def compute_sensitivity(grad_tom_dir: str, grad_nontom_dir: str) -> dict:
    """Compute per-group relative sensitivity from gradient chunks.

    Metric for group G:
        rs(G) = sum(G_tom[w] for w in G) / sum(G_nontom[w] for w in G)

    A ratio > 1 means the group is relatively more important for ToM
    than for non-ToM (factual) tasks.

    Returns {
        'text':  {'sum_tom': float, 'sum_nontom': float, 'n_params': int, 'rs': float},
        'vision': {'sum_tom': float, 'sum_nontom': float, 'n_params': int, 'rs': float},
    }
    """
    manifest_path = os.path.join(grad_tom_dir, "manifest.json")
    with open(manifest_path) as f:
        manifest = json.load(f)

    groups = {"text": {"sum_tom": 0.0, "sum_nontom": 0.0, "n_params": 0},
              "vision": {"sum_tom": 0.0, "sum_nontom": 0.0, "n_params": 0}}

    for layer_key, info in manifest["layers"].items():
        cat = info["category"]
        tom_path = os.path.join(grad_tom_dir, f"{layer_key}.pt")
        nontom_path = os.path.join(grad_nontom_dir, f"{layer_key}.pt")
        if not (os.path.exists(tom_path) and os.path.exists(nontom_path)):
            continue

        g_tom = torch.load(tom_path, map_location="cpu")
        g_nontom = torch.load(nontom_path, map_location="cpu")

        for mod_name in info["modules"]:
            if mod_name not in g_tom or mod_name not in g_nontom:
                continue
            t = g_tom[mod_name].float()
            nt = g_nontom[mod_name].float()
            groups[cat]["sum_tom"] += t.sum().item()
            groups[cat]["sum_nontom"] += nt.sum().item()
            groups[cat]["n_params"] += t.numel()

    # compute ratios
    for cat in groups:
        s_nt = groups[cat]["sum_nontom"]
        if s_nt > 0:
            groups[cat]["rs"] = groups[cat]["sum_tom"] / s_nt
        else:
            groups[cat]["rs"] = float("inf") if groups[cat]["sum_tom"] > 0 else float("nan")

    return groups


# =====================================================================
# Report generation
# =====================================================================

SETTINGS = [
    ("llava_multimodal",        "LLaVA 1.5 (text+video)"),
    ("instructblip_multimodal", "InstructBLIP (text+video)"),
    ("qwen3vl_multimodal",      "Qwen3-VL (text+video)"),
    ("qwen3vl_text_only",       "Qwen3-VL (text-only)"),
    ("qwen3vl_video_only",      "Qwen3-VL (video-only)"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_root", required=True,
                    help="Root dir containing per-setting subdirectories")
    ap.add_argument("--out", default=None,
                    help="Output report path (default: results_root/report.md)")
    args = ap.parse_args()

    root = Path(args.results_root)
    out_path = args.out or str(root / "report.md")

    lines = []
    lines.append("# VLM Theory-of-Mind Analysis Report\n")

    # ---- 1. Cross-model comparison table (m=0 baseline) ----
    lines.append("## 1. Final ToM Accuracy (Baseline, m=0)\n")
    lines.append("| Setting | Accuracy | Correct / Total |")
    lines.append("|---------|----------|-----------------|")

    baseline_accs = {}
    for key, label in SETTINGS:
        csv_path = root / key / "eval_results.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            row0 = df[df["m"] == 0.0]
            if len(row0):
                acc = row0.iloc[0]["accuracy"]
                nc = int(row0.iloc[0]["n_correct"])
                nt = int(row0.iloc[0]["n_total"])
                lines.append(f"| {label} | {acc:.2%} | {nc}/{nt} |")
                baseline_accs[key] = acc
            else:
                lines.append(f"| {label} | N/A | N/A |")
        else:
            lines.append(f"| {label} | (not found) | — |")

    lines.append("")

    # ---- 2. Masking sweep per setting ----
    lines.append("## 2. ToM Accuracy vs Masking Sparsity (m)\n")
    for key, label in SETTINGS:
        csv_path = root / key / "eval_results.csv"
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        lines.append(f"### {label}\n")
        lines.append("| m | Accuracy |")
        lines.append("|---|----------|")
        for _, row in df.iterrows():
            lines.append(f"| {row['m']:.1e} | {row['accuracy']:.2%} |")
        lines.append("")

    # ---- 3. Relative sensitivity (text vs vision) ----
    lines.append("## 3. Relative Sensitivity: Text vs Vision Modules\n")
    lines.append("Metric: `rs(group) = Σ grad²_ToM / Σ grad²_nonToM` — "
                 "ratio > 1 means the group is relatively more important for "
                 "ToM than for factual (non-ToM) tasks.\n")
    lines.append("| Setting | Text RS | Vision RS | More ToM-sensitive |")
    lines.append("|---------|---------|-----------|-------------------|")

    all_sens = {}
    for key, label in SETTINGS:
        tom_dir = root / key / "chunks_tom"
        nontom_dir = root / key / "chunks_nontom"
        precomputed = root / key / "sensitivity.json"
        if tom_dir.exists() and nontom_dir.exists():
            sens = compute_sensitivity(str(tom_dir), str(nontom_dir))
        elif precomputed.exists():
            with open(precomputed) as f:
                sens = json.load(f)
        else:
            lines.append(f"| {label} | (not available) | — | — |")
            continue
        all_sens[key] = sens
        t_rs = sens["text"]["rs"]
        v_rs = sens["vision"]["rs"]
        t_str = f"{t_rs:.4f}" if np.isfinite(t_rs) else "N/A"
        v_str = f"{v_rs:.4f}" if np.isfinite(v_rs) else "N/A"
        if np.isfinite(t_rs) and np.isfinite(v_rs):
            more = "Text" if t_rs > v_rs else "Vision"
        else:
            more = "N/A"
        lines.append(f"| {label} | {t_str} | {v_str} | {more} |")

    lines.append("")

    # ---- 4. Detailed sensitivity per setting ----
    lines.append("## 4. Detailed Sensitivity Breakdown\n")
    for key, label in SETTINGS:
        if key not in all_sens:
            continue
        sens = all_sens[key]
        lines.append(f"### {label}\n")
        lines.append("| Group | Σ grad²_ToM | Σ grad²_nonToM | # Params | RS |")
        lines.append("|-------|-------------|----------------|----------|-----|")
        for cat in ("text", "vision"):
            s = sens[cat]
            rs_str = f"{s['rs']:.4f}" if np.isfinite(s["rs"]) else "N/A"
            lines.append(
                f"| {cat.capitalize()} | {s['sum_tom']:.6e} | "
                f"{s['sum_nontom']:.6e} | {s['n_params']:,} | {rs_str} |")
        lines.append("")

    # ---- write ----
    report = "\n".join(lines)
    with open(out_path, "w") as f:
        f.write(report)
    print(report)
    print(f"\n[DONE] Report written to {out_path}")


if __name__ == "__main__":
    main()
