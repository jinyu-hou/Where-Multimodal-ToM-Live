# How Vision-Language Models Encode Theory-of-Mind: A Sparse Parameter Analysis

[![NPJAI 2025](https://img.shields.io/badge/NPJAI-2025-green)](https://www.nature.com/articles/s44387-025-00031-9)
[![arXiv](https://img.shields.io/badge/arXiv-2504.04238-red)](https://arxiv.org/abs/2504.04238)

This repository extends the NPJ AI paper **"How Large Language Models Encode Theory-of-Mind"** to Vision-Language Models (VLMs). It uses gradient-based sparse parameter analysis to identify which components of multimodal models -- vision encoder, cross-modal connector, or language backbone -- are most involved in Theory-of-Mind reasoning.

## Models and Settings

| Setting | Model | Input mode |
|---------|-------|------------|
| `llava_multimodal` | `llava-hf/llava-1.5-7b-hf` | text + video frames |
| `instructblip_multimodal` | `Salesforce/instructblip-vicuna-7b` | text + video frames |
| `qwen3vl_multimodal` | `Qwen/Qwen3-VL-8B-Instruct` | text + video frames |
| `qwen3vl_text_only` | `Qwen/Qwen3-VL-8B-Instruct` | text only |
| `qwen3vl_video_only` | `Qwen/Qwen3-VL-8B-Instruct` | video frames only |

## Metrics

The pipeline produces two categories of metrics: one **extrinsic** (behavioral) and three **intrinsic** (mechanistic).

### Extrinsic: ToM Accuracy under Differential Masking

For each masking rate *m*, the top-m% of parameters by ToM gradient magnitude are identified, those also in the top-m% for non-ToM are excluded, and the remaining ToM-specific parameters are ablated. The model is then re-evaluated on 100 MMToM-QA questions. A drop in accuracy confirms that the masked parameters are causally important for ToM.

### Intrinsic Metric 1: ToM Sensitivity Score

Measures how strongly each module group responds to ToM inputs using the empirical Fisher information approximation:

```
s_i = (1/n) * Σ_j (dL(theta; x_j, y_j) / d theta_i)^2
```

Reported as the mean of the top-1% most sensitive parameters per group. Higher values mean the group's parameters are more engaged during ToM reasoning.

### Intrinsic Metric 2: ToM Selectivity Score

Measures whether a group is *disproportionately* engaged by ToM compared to non-ToM (factual) tasks:

```
selectivity(w) = grad²_ToM(w) / (grad²_nonToM(w) + eps)
```

Reported as the mean selectivity of the top-1% most ToM-sensitive parameters. A value of 1.0 means equal engagement; values > 1 indicate ToM specificity. This distinguishes parameters that are important *specifically* for ToM from those that are generally important for all tasks.

### Intrinsic Metric 3: Localization Concentration

Measures where in the architecture the most ToM-selective parameters are concentrated. Computed by taking the global top-0.1% parameters by selectivity score and counting what fraction falls in each module group:

- **Vision Encoder** -- ViT layers
- **Cross-Modal** -- projector/connector (LLaVA's linear projector, InstructBLIP's Q-Former, Qwen3-VL's merger)
- **Language Backbone** -- transformer decoder layers

Also reports a **Concentration Index** (max fraction; 1.0 = all in one group) and **Shannon Entropy** (0 = maximally concentrated, 1.58 = uniform across 3 groups).

## Reproducing the Results

### Prerequisites

- **Conda environment:** `tom` (Python 3.12, PyTorch 2.6+, Transformers 5.4+)
- **GPU:** 4x NVIDIA A6000 (or equivalent with ~50GB VRAM each)
- **Data:** MMToM-QA questions file and video frames directory

### Quick start (all 5 settings end-to-end)

```bash
conda activate tom
bash run_vlm_pipeline.sh [RESULTS_ROOT] [CACHE_DIR]
```

Default paths:
- `RESULTS_ROOT`: `./vlm_results`
- `CACHE_DIR`: `/data/user_data/$USER/hf_cache`
- `VIDEO_DIR`: `/data/user_data/jinyuhou/single_agent_partial_train_240_hp_test_highres`

The shell script runs all 5 settings through steps 1--4 below, then generates the report. Each step is skipped if its output already exists.

### Step-by-step reproduction

Replace `[...]` with your own paths.

#### Step 0: Create datasets

Sample 100 ToM questions from MMToM-QA and generate matched non-ToM (factual) versions:

```bash
python create_non_tom_dataset.py \
  --questions_path [/path/to/MMToM-QA/Benchmark/questions.json] \
  --out_dir data/ \
  --n_samples 100 \
  --seed 42
```

Outputs:
- `data/sampled_tom_questions.jsonl` -- 100 ToM questions
- `data/sampled_non_tom_questions.jsonl` -- 100 matched non-ToM questions

#### Step 1: Compute squared gradients

Run for both ToM and non-ToM data, for each model/mode combination:

```bash
python create_gradient_vlm.py \
  --model [MODEL_ID] \
  --data_path [data/sampled_tom_questions.jsonl] \
  --video_dir [VIDEO_DIR] \
  --out [OUT_DIR_FOR_TOM_GRAD] \
  --cache_dir [CACHE_DIR] \
  --mode [multimodal|text_only|video_only] \
  --max_frames 8
```

Registers `g -> g^2` hooks on all `nn.Linear` weights, runs forward+backward on each sample, accumulates squared gradients, and saves the result as a HuggingFace checkpoint (weights replaced with accumulated squared gradients).

#### Step 2: Chunk gradients by layer

```bash
python chunk_gradient_vlm.py \
  --model [OUT_DIR_FOR_TOM_GRAD] \
  --output_path [TOM_CHUNKS_DIR] \
  --cache_dir [CACHE_DIR] \
  --device_map cpu
```

Groups all `nn.Linear` modules by architectural component and saves one `.pt` file per component layer, plus a `manifest.json` with module categorization (vision / text).

#### Step 3: Evaluate with differential masking sweep

```bash
python vlm_evaluation.py \
  --model [MODEL_ID] \
  --grad_tom_chunks [TOM_CHUNKS_DIR] \
  --grad_nontom_chunks [NONTOM_CHUNKS_DIR] \
  --questions [data/sampled_tom_questions.jsonl] \
  --video_dir [VIDEO_DIR] \
  --out_dir [SETTING_DIR] \
  --cache_dir [CACHE_DIR] \
  --mode [multimodal|text_only|video_only] \
  --max_frames 8 \
  --m_list "0.0,5e-6,1e-5,2e-5,5e-5"
```

For each *m* value: loads a fresh model, applies the differential mask in-place, evaluates on 100 ToM questions (greedy decoding), and records accuracy. Outputs `eval_results.csv`.

#### Step 4: Compute intrinsic metrics

```bash
python compute_intrinsic_metrics.py \
  --results_root [RESULTS_ROOT] \
  --n_samples 100 \
  --selectivity_threshold 2.0 \
  --topk_frac 0.001
```

Reads gradient chunks for all 5 settings and computes the three intrinsic metrics (sensitivity, selectivity, localization). Outputs `intrinsic_metrics.json`.

#### Step 5: Generate report

```bash
python report_vlm.py \
  --results_root [RESULTS_ROOT] \
  --out [RESULTS_ROOT]/report.md
```

Aggregates evaluation CSVs, sensitivity data, and intrinsic metrics into a single markdown report.

### Output structure

```
vlm_results/
  llava_multimodal/
    chunks_tom/          # Per-layer gradient chunks (ToM)
    chunks_nontom/       # Per-layer gradient chunks (non-ToM)
    eval_results.csv     # Accuracy vs masking rate
    per_type_m*.json     # Per-question-type breakdown
    sensitivity.json     # Pre-computed 2-way sensitivity (text/vision)
  instructblip_multimodal/
    ...
  qwen3vl_multimodal/
    ...
  qwen3vl_text_only/
    ...
  qwen3vl_video_only/
    ...
  intrinsic_metrics.json # All 3 intrinsic metrics for all settings
  report.md              # Final report
```

### Key scripts

| Script | Purpose |
|--------|---------|
| `vlm_utils.py` | Shared utilities: model loading, frame extraction, input preparation, parameter categorization |
| `create_gradient_vlm.py` | Accumulate squared gradients over MMToM-QA samples |
| `chunk_gradient_vlm.py` | Split gradient checkpoint into per-layer chunks |
| `vlm_evaluation.py` | Differential masking sweep + ToM accuracy evaluation |
| `compute_intrinsic_metrics.py` | Sensitivity, selectivity, and localization metrics |
| `report_vlm.py` | Generate final markdown report |
| `create_non_tom_dataset.py` | Sample ToM questions and create matched non-ToM versions |
| `run_vlm_pipeline.sh` | End-to-end runner for all 5 settings |

---

## Original LLM Pipeline

The code below is from the original NPJ AI paper, designed for text-only LLMs (LLaMA, Mistral, OPT). The VLM pipeline above extends this methodology to multimodal models.

1. `create_gradient.py` -- compute squared gradients and save as model
2. `chunk_gradient.py` -- split gradient into per-layer chunks
3. `ToM_and_perplexity_evaluation.py` -- build masked models for each `m`, run ToM & perplexity eval
4. `summarize.py` -- aggregate ToM results

Replace every `[]` with your own paths.

### 1. Create Gradients

```bash
# TOM dataset (full sequence, last-token only supervision)
python create_gradient.py \
  --model [MODEL_ID] \
  --dataset tom \
  --data_path [/path/to/tom_training_data.json] \
  --nsamples 100 \
  --seqlen 0 \
  --seed 0 \
  --cache_dir [CACHE_DIR] \
  --out [OUT_DIR_FOR_TOM_GRAD]
```

```bash
# C4 dataset (random 128-token windows)
python create_gradient.py \
  --model [MODEL_ID] \
  --dataset c4 \
  --nsamples 100 \
  --seqlen 128 \
  --seed 0 \
  --cache_dir [CACHE_DIR] \
  --out [OUT_DIR_FOR_C4_GRAD]
```

### 2. Chunk Gradients

```bash
python chunk_gradient.py \
  --model [OUT_DIR_FOR_TOM_GRAD] \
  --output_path [TOM_CHUNKS_DIR] \
  --cache_dir [CACHE_DIR] \
  --device_map auto
```

```bash
python chunk_gradient.py \
  --model [OUT_DIR_FOR_C4_GRAD] \
  --output_path [C4_CHUNKS_DIR] \
  --cache_dir [CACHE_DIR] \
  --device_map auto
```

Parts of the gradient extraction and chunking code are adapted from [SqueezeLLM: Dense-and-Sparse Quantization (ICML 2024)](https://arxiv.org/abs/2306.07629).

### 3. ToM + Perplexity Evaluation

```bash
python ToM_and_perplexity_evaluation.py \
  --model [MODEL_ID] \
  --grad_tom_chunks [TOM_CHUNKS_DIR] \
  --grad_c4_chunks [C4_CHUNKS_DIR] \
  --tom_tasks [/path/to/tom_tasks.py] \
  --out_dir [EVAL_OUT_DIR] \
  --cache_dir [CACHE_DIR] \
  --tensor_parallel_size 1 \
  --max_model_len 1024 \
  --batch_size 64 \
  --reps 5 \
  --m_start 0.0 --m_end 5e-5 --m_step 2e-6
```

### 4. Summarize ToM Scores

```bash
python summarize.py \
  --root [EVAL_OUT_DIR]/tom \
  --reps 5 \
  --out_csv [EVAL_OUT_DIR]/tom_summary.csv
```

## Citation

If you find this work useful, please cite our paper:

```bibtex
@article{wu2025large,
  title={How large language models encode theory-of-mind: a study on sparse parameter patterns},
  author={Wu, Yuheng and Guo, Wentao and Liu, Zirui and Ji, Heng and Xu, Zhaozhuo and Zhang, Denghui},
  journal={npj Artificial Intelligence},
  volume={1},
  number={1},
  pages={20},
  year={2025},
  publisher={Nature Publishing Group UK London}
}
```

For questions or issues, please contact [Yuheng Wu](mailto:yuhengwu@stanford.edu) or open an issue in this repository.
