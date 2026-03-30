#!/usr/bin/env python3
"""
Shared utilities for VLM Theory-of-Mind pipeline.

Supports:
  - llava-hf/llava-1.5-7b-hf          (LLaVA 1.5)
  - Salesforce/instructblip-vicuna-7b   (InstructBLIP)
  - Qwen/Qwen3-VL-8B-Instruct          (Qwen3-VL)

Provides: model loading, video-frame extraction, prompt/input preparation,
          parameter categorization (text vs vision).
"""

import json, os, pickle, re, random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image


# =====================================================================
# Model family detection & loading
# =====================================================================

def detect_family(model_id: str) -> str:
    n = model_id.lower()
    if "llava" in n:
        return "llava"
    if "instructblip" in n or "instruct-blip" in n:
        return "instructblip"
    if "qwen" in n and "vl" in n:
        return "qwen3vl"
    raise ValueError(f"Unknown VLM family: {model_id}")


def load_vlm(model_id: str,
             cache_dir: Optional[str] = None,
             device_map: str = "auto",
             dtype=torch.float16):
    """Load a VLM and its processor.  Returns (model, processor, family)."""
    family = detect_family(model_id)

    if family == "llava":
        from transformers import LlavaForConditionalGeneration, AutoProcessor
        model = LlavaForConditionalGeneration.from_pretrained(
            model_id, cache_dir=cache_dir, device_map=device_map,
            torch_dtype=dtype, trust_remote_code=True)
        processor = AutoProcessor.from_pretrained(model_id, cache_dir=cache_dir)

    elif family == "instructblip":
        from transformers import InstructBlipForConditionalGeneration, InstructBlipProcessor
        model = InstructBlipForConditionalGeneration.from_pretrained(
            model_id, cache_dir=cache_dir, device_map=device_map,
            torch_dtype=dtype, trust_remote_code=True)
        processor = InstructBlipProcessor.from_pretrained(model_id, cache_dir=cache_dir)

    elif family == "qwen3vl":
        from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id, cache_dir=cache_dir, device_map=device_map,
            torch_dtype=dtype, trust_remote_code=True)
        processor = AutoProcessor.from_pretrained(model_id, cache_dir=cache_dir)

    return model, processor, family


# =====================================================================
# Video frame extraction
# =====================================================================

def get_video_frames(video_dir: str,
                     episode: int,
                     end_time: int,
                     max_frames: int = 8) -> List[Image.Image]:
    """Sample up to *max_frames* RGB frames from a task episode.

    Follows the same logic as MMToM-QA/BIP-ALM/testing_gpt.py:
      1. Read frame_intervals.pik → times = [interval[1] for each action]
      2. end_frame = times[end_time]
      3. Uniformly sample *max_frames* indices from 0 … end_frame
    """
    task_dir = os.path.join(video_dir, f"task_{episode}")
    fi_path = os.path.join(task_dir, "frame_intervals.pik")
    with open(fi_path, "rb") as f:
        intervals = pickle.load(f)

    times = [action[1] for action in intervals]
    end_frame = int(times[end_time])

    # uniform sample (same logic as testing_gpt.py "full" mode)
    if max_frames <= 1:
        selected = [end_frame]
    else:
        step = max(1, int(end_frame / (max_frames - 1)))
        selected = [min(i * step, end_frame) for i in range(max_frames)]

    frames_dir = os.path.join(task_dir, "script", "0")
    images = []
    for idx in selected:
        path = os.path.join(frames_dir, f"Action_{idx:04d}_0_normal.png")
        if os.path.exists(path):
            images.append(Image.open(path).convert("RGB"))
        else:
            images.append(Image.new("RGB", (224, 224), (128, 128, 128)))
    return images


def make_frame_grid(images: List[Image.Image], cols: int = 4) -> Image.Image:
    """Tile a list of images into a single grid image (for single-image VLMs)."""
    if not images:
        return Image.new("RGB", (224, 224))
    n = len(images)
    cols = min(cols, n)
    rows = (n + cols - 1) // cols
    w, h = images[0].size
    grid = Image.new("RGB", (cols * w, rows * h))
    for i, img in enumerate(images):
        r, c = divmod(i, cols)
        grid.paste(img.resize((w, h)), (c * w, r * h))
    return grid


# =====================================================================
# Input preparation
# =====================================================================

def _strip_scene_and_actions(question_text: str) -> str:
    """For video-only mode: keep only the 'Question: …' part."""
    if "\nQuestion:" in question_text:
        return "Question:" + question_text.split("\nQuestion:")[-1]
    if "Question:" in question_text:
        return "Question:" + question_text.split("Question:")[-1]
    return question_text


def prepare_input_llava(processor, question_text: str,
                        images: Optional[List[Image.Image]],
                        answer: Optional[str] = None):
    """Build LLaVA 1.5 inputs.  If *answer* given, build labels too."""
    if images:
        grid = make_frame_grid(images)
        prompt = f"USER: <image>\n{question_text}\nAnswer:"
        if answer:
            prompt += f" {answer}"
        inputs = processor(images=grid, text=prompt, return_tensors="pt")
    else:
        prompt = f"USER: {question_text}\nAnswer:"
        if answer:
            prompt += f" {answer}"
        inputs = processor(text=prompt, return_tensors="pt")

    if answer:
        labels = inputs["input_ids"].clone()
        # mask everything except the last token (the answer)
        labels[:, :-1] = -100
        inputs["labels"] = labels
    return inputs


def prepare_input_instructblip(processor, question_text: str,
                               images: Optional[List[Image.Image]],
                               answer: Optional[str] = None):
    """Build InstructBLIP inputs."""
    prompt = question_text + "\nAnswer:"
    if answer:
        prompt += f" {answer}"

    if images:
        grid = make_frame_grid(images)
        inputs = processor(images=grid, text=prompt, return_tensors="pt",
                           truncation=True, max_length=512)
    else:
        # text-only fallback: provide a blank image (InstructBLIP requires pixel_values)
        blank = Image.new("RGB", (224, 224), (255, 255, 255))
        inputs = processor(images=blank, text=prompt, return_tensors="pt",
                           truncation=True, max_length=512)

    if answer:
        labels = inputs["input_ids"].clone()
        labels[:, :-1] = -100
        inputs["labels"] = labels
    return inputs


def prepare_input_qwen3vl(processor, question_text: str,
                          images: Optional[List[Image.Image]],
                          answer: Optional[str] = None):
    """Build Qwen3-VL inputs using chat template."""
    content = []
    if images:
        for img in images:
            content.append({"type": "image", "image": img})
    content.append({"type": "text", "text": question_text + "\nAnswer:"})
    if answer:
        content[-1]["text"] += f" {answer}"

    messages = [{"role": "user", "content": content}]
    text_prompt = processor.apply_chat_template(messages, tokenize=False,
                                                add_generation_prompt=True)
    if images:
        inputs = processor(text=[text_prompt], images=images,
                           return_tensors="pt", padding=True)
    else:
        inputs = processor(text=[text_prompt], return_tensors="pt", padding=True)

    if answer:
        labels = inputs["input_ids"].clone()
        labels[:, :-1] = -100
        inputs["labels"] = labels
    return inputs


def prepare_input(processor, family: str, question_text: str,
                  images: Optional[List[Image.Image]] = None,
                  answer: Optional[str] = None):
    """Dispatch to model-specific input builder."""
    if family == "llava":
        return prepare_input_llava(processor, question_text, images, answer)
    if family == "instructblip":
        return prepare_input_instructblip(processor, question_text, images, answer)
    if family == "qwen3vl":
        return prepare_input_qwen3vl(processor, question_text, images, answer)
    raise ValueError(f"Unknown family {family}")


# =====================================================================
# Generation & answer extraction
# =====================================================================

def generate_answer(model, processor, family: str, question_text: str,
                    images: Optional[List[Image.Image]] = None,
                    max_new_tokens: int = 5) -> str:
    """Generate an answer and extract the first 'a' or 'b'."""
    inputs = prepare_input(processor, family, question_text, images, answer=None)
    device = next(model.parameters()).device
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
              for k, v in inputs.items()}

    with torch.no_grad():
        out_ids = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                 do_sample=False)
    # decode only new tokens
    gen_ids = out_ids[:, inputs["input_ids"].shape[1]:]
    text = processor.batch_decode(gen_ids, skip_special_tokens=True)[0].strip().lower()

    for ch in text:
        if ch in ("a", "b"):
            return ch
    return text[0] if text else "a"


# =====================================================================
# Data loading
# =====================================================================

def load_questions(path: str) -> List[dict]:
    questions = []
    with open(path) as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))
    return questions


# =====================================================================
# Parameter categorization (text vs vision)
# =====================================================================

def get_all_linear_modules(model) -> Dict[str, torch.nn.Linear]:
    """Return {full_dotted_name: Linear_module} for every nn.Linear in model."""
    out = {}
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear):
            out[name] = mod
    return out


def categorize_param(name: str, family: str) -> str:
    """Classify a parameter name as 'text' or 'vision'."""
    n = name.lower()
    if family == "llava":
        if "vision_tower" in n or "multi_modal_projector" in n:
            return "vision"
        return "text"  # language_model.*
    if family == "instructblip":
        if "vision_model" in n or "qformer" in n or "language_projection" in n:
            return "vision"
        return "text"  # language_model.*
    if family == "qwen3vl":
        if "visual" in n:
            return "vision"
        return "text"  # model.* / lm_head.*
    return "text"


def categorize_all_linears(model, family: str) -> Dict[str, List[str]]:
    """Group all nn.Linear names by category.  Returns {'text': [...], 'vision': [...]}."""
    groups: Dict[str, List[str]] = {"text": [], "vision": []}
    for name in get_all_linear_modules(model):
        cat = categorize_param(name, family)
        groups[cat].append(name)
    return groups
