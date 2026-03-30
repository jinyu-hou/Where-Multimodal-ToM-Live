#!/usr/bin/env python3
"""
Squared-gradient saver for VLMs.

Adapted from create_gradient.py for Vision-Language Models.
- Loads a VLM (LLaVA / InstructBLIP / Qwen3-VL).
- Registers grad hooks (g → g²) on ALL nn.Linear weights.
- Accumulates squared gradients over MMToM-QA samples (multimodal input).
- Overwrites weights with accumulated squared gradients, then save_pretrained.

Supports three input modes via --mode:
  multimodal : text + video frames  (default)
  text_only  : text only, no images
  video_only : question part + video frames (scene description stripped)
"""

import argparse, gc, json, os, random
import numpy as np
import torch
from tqdm import tqdm

from vlm_utils import (
    load_vlm, detect_family,
    load_questions, get_video_frames,
    prepare_input, get_all_linear_modules,
    _strip_scene_and_actions,
)


def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF model id")
    ap.add_argument("--data_path", required=True, help="JSONL questions file")
    ap.add_argument("--video_dir", required=True, help="Video frames root")
    ap.add_argument("--out", required=True, help="Output dir for gradient checkpoint")
    ap.add_argument("--cache_dir", default=None)
    ap.add_argument("--mode", choices=["multimodal", "text_only", "video_only"],
                    default="multimodal")
    ap.add_argument("--max_frames", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    set_seed(args.seed)

    # ---- load model ----
    print(f"[INFO] Loading {args.model} ...")
    model, processor, family = load_vlm(args.model, cache_dir=args.cache_dir)
    model.eval()

    # ---- register grad² hooks on every nn.Linear weight ----
    linear_mods = get_all_linear_modules(model)
    print(f"[INFO] Registering grad² hooks on {len(linear_mods)} linear modules")
    for name, mod in linear_mods.items():
        mod.weight.register_hook(lambda g: g.square())

    # ---- load data ----
    questions = load_questions(args.data_path)
    print(f"[INFO] {len(questions)} samples, mode={args.mode}")

    # ---- accumulate squared gradients ----
    model.zero_grad(set_to_none=True)
    n_ok = 0

    for q in tqdm(questions, desc="Accumulating grad²"):
        # decide images
        if args.mode == "text_only":
            images = None
            q_text = q["question"]
        elif args.mode == "video_only":
            images = get_video_frames(args.video_dir, q["episode"],
                                      q["end_time"], args.max_frames)
            q_text = _strip_scene_and_actions(q["question"])
        else:  # multimodal
            images = get_video_frames(args.video_dir, q["episode"],
                                      q["end_time"], args.max_frames)
            q_text = q["question"]

        try:
            inputs = prepare_input(processor, family, q_text, images,
                                   answer=q["answer"])
            device = next(model.parameters()).device
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                      for k, v in inputs.items()}
            out = model(**inputs)
            out.loss.backward()
            n_ok += 1
        except Exception as e:
            print(f"[WARN] Skipping sample (episode={q.get('episode')}): {e}")
            # Don't zero_grad — accumulated squared grads from prior samples are
            # already stored (hooks replace grad with grad²). Just continue.

    print(f"[INFO] {n_ok}/{len(questions)} samples processed successfully")

    # ---- overwrite weights with accumulated squared gradients ----
    n_written = 0
    for name, mod in linear_mods.items():
        if mod.weight.grad is not None:
            mod.weight.data = mod.weight.grad
            n_written += 1
        else:
            mod.weight.data.zero_()
    print(f"[INFO] Wrote gradients for {n_written}/{len(linear_mods)} modules")

    # ---- save ----
    os.makedirs(args.out, exist_ok=True)
    print(f"[INFO] Saving to {args.out} ...")
    model.save_pretrained(args.out)
    print("[DONE]")


if __name__ == "__main__":
    main()
