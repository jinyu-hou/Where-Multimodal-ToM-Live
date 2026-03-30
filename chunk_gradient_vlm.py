#!/usr/bin/env python3
"""
Layer chunker for VLM squared-gradient checkpoints.

Adapted from chunk_gradient.py for Vision-Language Models.
- Loads a VLM gradient checkpoint (weights = squared gradients).
- Enumerates ALL nn.Linear modules, groups them by component layer.
- Writes one .pt file per component layer:
    text_layer_{i}.pt    – language-model decoder layers
    vision_layer_{i}.pt  – vision-encoder layers
    connector_{i}.pt     – projector / Q-Former layers
- Also writes manifest.json with module categorisation metadata.
"""

import argparse, json, os, re
from collections import OrderedDict
import torch
from tqdm import tqdm

from vlm_utils import detect_family, get_all_linear_modules, categorize_param


def _load_model_only(model_path: str, cache_dir=None, device_map="cpu", dtype=torch.float16):
    """Load just the model (no processor) from a gradient checkpoint or HF id.

    Works for local gradient checkpoints that lack preprocessor_config.json.
    """
    config_path = os.path.join(model_path, "config.json") if os.path.isdir(model_path) else None
    if config_path and os.path.exists(config_path):
        with open(config_path) as f:
            cfg = json.load(f)
        arch = (cfg.get("architectures") or [""])[0].lower()
        model_type = cfg.get("model_type", "").lower()
    else:
        arch = model_path.lower()
        model_type = ""

    if "llava" in arch or "llava" in model_type:
        from transformers import LlavaForConditionalGeneration
        model = LlavaForConditionalGeneration.from_pretrained(
            model_path, cache_dir=cache_dir, device_map=device_map,
            torch_dtype=dtype, trust_remote_code=True,
            ignore_mismatched_sizes=True)
        family = "llava"
    elif "instructblip" in arch or "instructblip" in model_type:
        from transformers import InstructBlipForConditionalGeneration
        model = InstructBlipForConditionalGeneration.from_pretrained(
            model_path, cache_dir=cache_dir, device_map=device_map,
            torch_dtype=dtype, trust_remote_code=True,
            ignore_mismatched_sizes=True)
        family = "instructblip"
    elif "qwen" in arch or "qwen" in model_type:
        from transformers import Qwen3VLForConditionalGeneration
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path, cache_dir=cache_dir, device_map=device_map,
            torch_dtype=dtype, trust_remote_code=True,
            ignore_mismatched_sizes=True)
        family = "qwen3vl"
    else:
        raise ValueError(f"Cannot detect model family from {model_path}")

    return model, family


# =====================================================================
# Grouping linear modules into numbered component layers
# =====================================================================

def _extract_layer_key(name: str, family: str) -> str:
    """Map a dotted parameter name to a canonical layer key.

    Examples (llava):
        language_model.model.layers.5.self_attn.q_proj → text_layer_5
        vision_tower.vision_model.encoder.layers.3.self_attn.out_proj → vision_layer_3
        multi_modal_projector.linear_1 → connector_0
    """
    cat = categorize_param(name, family)

    # Q-Former uses singular ".layer." – check BEFORE the generic ".layers?" regex
    m_qf = re.search(r'qformer.*\.layer\.(\d+)\.', name, re.IGNORECASE)
    if m_qf:
        return f"connector_{int(m_qf.group(1))}"

    # Generic transformer layers: ".layers.N." or ".layer.N." (vision encoders)
    m = re.search(r'\.layers?\.(\d+)\.', name)
    if m:
        idx = int(m.group(1))
        if cat == "vision":
            return f"vision_layer_{idx}"
        return f"text_layer_{idx}"

    # Qwen3-VL visual blocks: ".blocks.N."
    m_blk = re.search(r'\.blocks\.(\d+)\.', name)
    if m_blk:
        return f"vision_layer_{int(m_blk.group(1))}"

    # Remaining connector / projector layers without a numeric index
    if cat == "vision":
        return "connector_0"

    return "text_misc"


def group_linears(model, family: str) -> dict:
    """Returns {layer_key: {short_module_name: full_dotted_name}}."""
    groups: dict[str, dict[str, str]] = {}
    linears = get_all_linear_modules(model)
    for full_name in linears:
        key = _extract_layer_key(full_name, family)
        # short name = last two dotted parts (e.g. "self_attn.q_proj")
        parts = full_name.rsplit(".", 2)
        short = ".".join(parts[-2:]) if len(parts) >= 2 else parts[-1]
        groups.setdefault(key, {})[short] = full_name
    return groups


# =====================================================================
# Main
# =====================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help="Path to VLM gradient checkpoint (or HF id if weights ARE gradients)")
    ap.add_argument("--output_path", required=True)
    ap.add_argument("--cache_dir", default=None)
    ap.add_argument("--device_map", default="cpu")
    args = ap.parse_args()

    os.makedirs(args.output_path, exist_ok=True)

    print(f"[INFO] Loading model from {args.model} ...")
    model, family = _load_model_only(
        args.model, cache_dir=args.cache_dir, device_map=args.device_map)

    groups = group_linears(model, family)
    sorted_keys = sorted(groups.keys())
    print(f"[INFO] {len(sorted_keys)} component-layer groups found")

    manifest_layers = {}
    for key in tqdm(sorted_keys, desc="Chunking"):
        blob = {}
        mod_map = groups[key]
        for short_name, full_name in mod_map.items():
            # walk to the module
            parts = full_name.split(".")
            mod = model
            for p in parts:
                mod = getattr(mod, p)
            w = mod.weight.data.detach().to("cpu").contiguous()
            blob[short_name] = w

        torch.save(blob, os.path.join(args.output_path, f"{key}.pt"))
        manifest_layers[key] = {
            "category": "vision" if key.startswith(("vision_", "connector")) else "text",
            "modules": list(mod_map.keys()),
        }

    manifest = {
        "source_model": args.model,
        "family": family,
        "num_groups": len(sorted_keys),
        "layers": manifest_layers,
    }
    with open(os.path.join(args.output_path, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"[DONE] Wrote {len(sorted_keys)} chunk files to {args.output_path}")


if __name__ == "__main__":
    main()
