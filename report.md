# VLM Theory-of-Mind Analysis Report

## Overview

This report presents results from a gradient-based sparse parameter analysis of three Vision-Language Models (VLMs) on the MMToM-QA benchmark. The analysis follows the methodology from [How Large Language Models Encode Theory of Mind](https://github.com/jinyuhou/how-large-language-models-encode-theory-of-mind), adapted for multimodal models.

**Models tested:**
- **LLaVA 1.5 (7B)** -- vision encoder + linear projector + Vicuna language model
- **InstructBLIP (Vicuna-7B)** -- vision encoder + Q-Former connector + Vicuna language model
- **Qwen3-VL (8B-Instruct)** -- integrated vision-language architecture

**Experimental settings:**
- Three multimodal settings (text + 8 sampled video frames) -- one per model
- Two Qwen3-VL ablations: text-only (no video frames) and video-only (scene description stripped from text)

**Dataset:** 100 sampled ToM questions from MMToM-QA and 100 matched non-ToM (factual) questions derived from the same episodes. Each question is binary-choice (a/b), so random-chance accuracy is 50%.

---

## 1. Final ToM Accuracy (Baseline, m=0)

This table shows each model's zero-shot ToM accuracy on the 100 MMToM-QA questions **without any masking applied** (m=0). This is the unmodified model's ability to answer Theory-of-Mind questions.

| Setting | Accuracy | Correct / Total |
|---------|----------|-----------------|
| LLaVA 1.5 (text+video) | 47.00% | 47/100 |
| InstructBLIP (text+video) | 53.00% | 53/100 |
| Qwen3-VL (text+video) | 37.00% | 37/100 |
| Qwen3-VL (text-only) | 36.00% | 36/100 |
| Qwen3-VL (video-only) | 39.00% | 39/100 |

**How to interpret:** Higher accuracy means the model is better at ToM reasoning on MMToM-QA. Since questions are binary-choice, 50% is chance level. InstructBLIP performs best (53%), while Qwen3-VL settings perform below chance, suggesting it struggles with this task format.

---

## 2. ToM Accuracy vs Masking Sparsity (m)

This section shows how ToM accuracy changes as we progressively mask (zero out) the model's most ToM-specific parameters.

**Methodology:** For each masking rate m:
1. Identify the top-m% of parameters by squared-gradient magnitude on ToM data.
2. Among those, remove any that are also in the top-m% on non-ToM data (these are important for general reasoning, not ToM-specific).
3. The remaining parameters form the **differential mask** -- weights that are specifically important for ToM but not for factual reasoning.
4. Replace masked weights with the group mean (effectively ablating them).
5. Evaluate the masked model on the same 100 ToM questions.

**How to interpret:** If masking ToM-specific parameters causes accuracy to **drop**, it confirms those parameters are causally important for ToM reasoning. If accuracy is **unchanged**, the masked parameters may not be critical. If accuracy **increases**, the masking may be removing noise or the effect is within statistical noise (with 100 binary questions, ~5% fluctuation is expected by chance).

### LLaVA 1.5 (text+video)

| m | Accuracy |
|---|----------|
| 0.0e+00 | 47.00% |
| 5.0e-06 | 49.00% |
| 1.0e-05 | 42.00% |
| 2.0e-05 | 50.00% |
| 5.0e-05 | 47.00% |

### InstructBLIP (text+video)

| m | Accuracy |
|---|----------|
| 0.0e+00 | 53.00% |
| 5.0e-06 | 41.00% |
| 1.0e-05 | 48.00% |
| 2.0e-05 | 46.00% |
| 5.0e-05 | 53.00% |

### Qwen3-VL (text+video)

| m | Accuracy |
|---|----------|
| 0.0e+00 | 37.00% |
| 5.0e-06 | 33.00% |
| 1.0e-05 | 33.00% |
| 2.0e-05 | 36.00% |
| 5.0e-05 | 37.00% |

### Qwen3-VL (text-only)

| m | Accuracy |
|---|----------|
| 0.0e+00 | 36.00% |
| 5.0e-06 | 36.00% |
| 1.0e-05 | 38.00% |
| 2.0e-05 | 36.00% |
| 5.0e-05 | 36.00% |

### Qwen3-VL (video-only)

| m | Accuracy |
|---|----------|
| 0.0e+00 | 39.00% |
| 5.0e-06 | 38.00% |
| 1.0e-05 | 35.00% |
| 2.0e-05 | 45.00% |
| 5.0e-05 | 42.00% |

---

## 3. Relative Sensitivity: Text vs Vision Modules

This section answers: **Are the text (language) modules or the vision modules more important for Theory-of-Mind reasoning?**

### Metric definition

For each parameter group G (text or vision), we compute the **Relative Sensitivity (RS)**:

```
RS(G) = Σ_w grad²_ToM(w) / Σ_w grad²_nonToM(w)
```

where the sums are over all weights w in group G.

- **grad²_ToM(w)** is the squared gradient of weight w accumulated over all 100 ToM questions. This measures how much the loss changes when w is perturbed -- i.e., how "important" w is for the ToM task.
- **grad²_nonToM(w)** is the same quantity computed over 100 non-ToM (factual) questions about the same episodes.
- **RS** is the ratio of these two sums. It measures the *relative* importance of the parameter group for ToM vs. factual reasoning.

### How to interpret RS values

| RS value | Interpretation |
|----------|---------------|
| RS = 1.0 | The group is equally important for ToM and non-ToM tasks |
| RS > 1.0 | The group is relatively **more** important for ToM than for factual tasks |
| RS < 1.0 | The group is relatively **less** important for ToM (more important for factual tasks) |
| RS >> 1.0 | The group is disproportionately ToM-sensitive; caution: may indicate very small non-ToM gradients in the denominator rather than exceptionally large ToM gradients |
| N/A | The group received no gradients (e.g., vision modules in text-only mode) |

**"More ToM-sensitive" column:** Compares Text RS vs Vision RS. The group with the higher RS value is labeled as more ToM-sensitive, meaning its parameters are relatively more specialized for ToM reasoning compared to the other group.

### Parameter grouping

- **Text** includes all `nn.Linear` modules in the language model backbone (decoder layers, embeddings, LM head).
- **Vision** includes all `nn.Linear` modules in the vision encoder (ViT layers) **and** the vision-language connector/projector (e.g., LLaVA's linear projector, InstructBLIP's Q-Former, Qwen3-VL's merger).

### Results

| Setting | Text RS | Vision RS | More ToM-sensitive |
|---------|---------|-----------|-------------------|
| LLaVA 1.5 (text+video) | 0.9729 | 0.8709 | Text |
| InstructBLIP (text+video) | 1.4705 | 1.1386 | Text |
| Qwen3-VL (text+video) | 80.0557 | 1946.0683 | Vision |
| Qwen3-VL (text-only) | 1.8892 | N/A | N/A |
| Qwen3-VL (video-only) | 3.6713 | 3.3055 | Text |

### Interpretation notes

- **LLaVA 1.5:** Both RS values are close to 1.0, meaning ToM and factual tasks activate similar parameter regions. Text modules are slightly more ToM-sensitive than vision modules (0.97 vs 0.87).
- **InstructBLIP:** Both groups have RS > 1.0, indicating the model allocates more gradient mass to ToM than to factual tasks overall. Text modules (RS=1.47) are more ToM-sensitive than vision+Q-Former modules (RS=1.14).
- **Qwen3-VL (multimodal):** Extremely high RS values (80 for text, 1946 for vision). This is driven by very small non-ToM gradient sums in the denominator (see Section 4: non-ToM sums are 1.4e+04 and 2.2e+03 respectively, orders of magnitude smaller than other models). This suggests the non-ToM factual questions produce very weak gradients in Qwen3-VL's multimodal mode, making the RS ratio unreliable as a comparative measure for this setting.
- **Qwen3-VL (text-only):** Vision RS is N/A because no gradients flow through vision modules when no images are provided. Text RS of 1.89 indicates text modules are moderately more engaged by ToM than factual tasks.
- **Qwen3-VL (video-only):** Both groups have RS around 3.3--3.7, meaning ToM questions produce roughly 3x the gradient activity of factual questions. Text is slightly more ToM-sensitive than vision.

---

## 4. Detailed Sensitivity Breakdown

This section shows the raw numbers behind the RS computation. Each row reports:

- **Σ grad²_ToM**: Total squared gradient summed over all weights in the group, accumulated across 100 ToM samples. Larger values mean the group's parameters are more "active" (i.e., the loss is more sensitive to perturbations of these weights) during ToM reasoning.
- **Σ grad²_nonToM**: Same quantity for the 100 non-ToM (factual) samples.
- **# Params**: Total number of scalar weight parameters in the group.
- **RS**: The ratio Σ grad²_ToM / Σ grad²_nonToM (see Section 3 for interpretation).

### LLaVA 1.5 (text+video)

| Group | Σ grad²_ToM | Σ grad²_nonToM | # Params | RS |
|-------|-------------|----------------|----------|-----|
| Text | 1.252267e+07 | 1.287120e+07 | 6,607,339,520 | 0.9729 |
| Vision | 7.778698e+06 | 8.932035e+06 | 322,961,408 | 0.8709 |

### InstructBLIP (text+video)

| Group | Σ grad²_ToM | Σ grad²_nonToM | # Params | RS |
|-------|-------------|----------------|----------|-----|
| Text | 1.750092e+07 | 1.190164e+07 | 6,607,339,520 | 1.4705 |
| Vision | 3.472143e+06 | 3.049419e+06 | 1,127,546,880 | 1.1386 |

### Qwen3-VL (text+video)

| Group | Σ grad²_ToM | Σ grad²_nonToM | # Params | RS |
|-------|-------------|----------------|----------|-----|
| Text | 1.137086e+06 | 1.420368e+04 | 7,568,097,280 | 80.0557 |
| Vision | 4.299598e+06 | 2.209376e+03 | 571,502,592 | 1946.0683 |

**Note on Qwen3-VL (multimodal):** The non-ToM gradient sums are 2--3 orders of magnitude smaller than those of LLaVA and InstructBLIP. This asymmetry inflates the RS ratio and makes it difficult to compare directly with other models. The likely cause is that Qwen3-VL's multimodal processing produces very flat loss landscapes for factual questions, possibly due to the model being highly confident on non-ToM answers or the non-ToM questions being trivially easy for this architecture.

### Qwen3-VL (text-only)

| Group | Σ grad²_ToM | Σ grad²_nonToM | # Params | RS |
|-------|-------------|----------------|----------|-----|
| Text | 1.963775e+06 | 1.039464e+06 | 7,568,097,280 | 1.8892 |
| Vision | 0.000000e+00 | 0.000000e+00 | 571,502,592 | N/A |

**Note:** Vision gradients are exactly zero because no images were provided in text-only mode -- gradients cannot flow through the vision encoder without visual input.

### Qwen3-VL (video-only)

| Group | Σ grad²_ToM | Σ grad²_nonToM | # Params | RS |
|-------|-------------|----------------|----------|-----|
| Text | 1.405674e+05 | 3.828823e+04 | 7,568,097,280 | 3.6713 |
| Vision | 6.490706e+04 | 1.963612e+04 | 571,502,592 | 3.3055 |

---

## 5. Intrinsic Metrics (3-Way Module Grouping)

The metrics in this section use a **3-way** module grouping that separates the cross-modal connector from the vision encoder, providing finer-grained localization of ToM-relevant parameters:

- **Vision Encoder** -- ViT layers (e.g., `vision_tower.encoder.layers.*` in LLaVA, `vision_model.encoder.layers.*` in InstructBLIP, `visual.blocks.*` in Qwen3-VL)
- **Cross-Modal** -- connector/projector that bridges vision and language (e.g., LLaVA's `multi_modal_projector`, InstructBLIP's Q-Former, Qwen3-VL's visual merger)
- **Language Backbone** -- all transformer decoder layers in the language model

### Parameter counts by group

| Setting | Vision Encoder | Cross-Modal | Language Backbone |
|---------|---------------|-------------|-------------------|
| LLaVA 1.5 | 301,989,888 | 20,971,520 | 6,607,339,520 |
| InstructBLIP | 984,023,040 | 143,523,840 | 6,607,339,520 |
| Qwen3-VL | 411,070,464 | 160,432,128 | 7,568,097,280 |

---

### 5.1 ToM Sensitivity Score

Measures how strongly each module group's parameters respond to ToM inputs, using the empirical Fisher information approximation.

**Definition:** For a group G with parameters {w_1, ..., w_N}:
```
MeanSensitivity(G) = (1 / (N * n_samples)) * Σ_i grad²_ToM(w_i)
TopkSensitivity(G) = (1 / (k * n_samples)) * Σ_{i in top-1%} grad²_ToM(w_i)
```
where `n_samples = 100` and `top-1%` selects the 1% of parameters with the largest squared gradients.

**How to interpret:**
- Higher values mean the group's parameters are more sensitive to ToM inputs on average.
- `TopkSensitivity` focuses on the most active parameters, filtering out the majority of near-zero gradients. This is often more informative than the mean.
- Compare across groups within the same model to see which component is most engaged by ToM reasoning.

| Setting | Vision Encoder | Cross-Modal | Language | Most sensitive |
|---------|---------------|-------------|----------|---------------|
| LLaVA 1.5 | 1.54e-02 | 1.29e-04 | 1.42e-03 | Vision Encoder |
| InstructBLIP | 2.01e-03 | 3.05e-03 | 2.02e-03 | Cross-Modal |
| Qwen3-VL (multimodal) | 6.71e-03 | 2.38e-05 | 1.27e-04 | Vision Encoder |
| Qwen3-VL (text-only) | 0.00e+00 | 0.00e+00 | 2.23e-04 | Language |
| Qwen3-VL (video-only) | 9.91e-05 | 1.26e-06 | 1.54e-05 | Vision Encoder |

*Values shown are TopkSensitivity (mean of top-1% parameters).*

**Key findings:**
- In LLaVA, the vision encoder's top-1% parameters are ~10x more sensitive than the language backbone's, suggesting strong visual activation during ToM processing.
- In InstructBLIP, the cross-modal Q-Former is the most sensitive group, indicating the Q-Former plays a critical role in bridging visual and linguistic ToM signals.
- In Qwen3-VL (text-only), vision and cross-modal sensitivity are exactly zero (no visual input), confirming clean modality isolation.

---

### 5.2 ToM Selectivity Score

Measures whether a module group is **disproportionately** engaged by ToM vs. non-ToM (factual) tasks, on a per-parameter basis.

**Definition:** For each parameter w:
```
selectivity(w) = grad²_ToM(w) / (grad²_nonToM(w) + eps)
```

Then for group G:
```
MeanSelectivity(G) = mean of selectivity(w) for all w in G
TopkSelectivity(G) = mean of selectivity(w) for w in top-1% by grad²_ToM
FracAbove(G) = fraction of parameters with selectivity(w) > threshold
```
where `threshold = 2.0` (parameter is at least 2x more important for ToM than non-ToM).

**How to interpret:**
- `MeanSelectivity = 1.0` means the group responds equally to ToM and non-ToM. Values > 1 indicate ToM preference.
- `TopkSelectivity` shows whether the most ToM-active parameters are also ToM-*specific* (not just generally active).
- `FracAbove` gives a concrete count: what fraction of parameters in the group are at least 2x more ToM-engaged.
- Very large values (>1000) indicate near-zero non-ToM gradients rather than genuine ToM specialization.

| Setting | Vision Encoder | Cross-Modal | Language | Most selective |
|---------|---------------|-------------|----------|---------------|
| LLaVA 1.5 | 1.10 | 0.96 | 2.01 | Language |
| InstructBLIP | 1.26 | 2.41 | 32845.04 | Language |
| Qwen3-VL (multimodal) | 1.25e+08 | 4.23e+06 | 1.18e+05 | Vision Encoder |
| Qwen3-VL (text-only) | 0.00 | 0.00 | 8.87 | Language |
| Qwen3-VL (video-only) | 23.41 | 18.81 | 41.91 | Language |

*Values shown are TopkSelectivity (mean selectivity of top-1% ToM-sensitive parameters).*

**Key findings:**
- For LLaVA, the language backbone's top-1% parameters have selectivity ~2.0 (twice as important for ToM as for factual tasks), while vision encoder parameters are only weakly selective (1.1).
- InstructBLIP's language backbone shows extreme selectivity (32845), driven by near-zero non-ToM gradients for its most ToM-sensitive parameters. This suggests a small subset of language parameters are almost exclusively used for ToM.
- Qwen3-VL multimodal values are unreliable due to the denominator issue (see Section 3 notes). The video-only setting provides cleaner comparisons: all groups show selectivity > 18, with language highest at 41.9.

---

### 5.3 Localization Concentration

Measures **where in the architecture** the most ToM-selective parameters are concentrated.

**Definition:**
1. Compute per-parameter selectivity = grad²_ToM(w) / (grad²_nonToM(w) + eps) for all parameters across all groups.
2. Select the global top-0.1% of parameters by selectivity.
3. Count what fraction of these top-k parameters belong to each group.

Also reports:
- **Concentration Index** = max(fraction across groups). Ranges from 0.33 (uniform) to 1.0 (all in one group).
- **Entropy** (bits) = Shannon entropy of the distribution. 0 = all in one group, 1.58 = perfectly uniform across 3 groups.

**How to interpret:**
- If a group has `frac_topk = 0.80`, it means 80% of the most ToM-selective parameters in the entire model fall within that group.
- High concentration index (close to 1.0) means ToM-selective parameters are highly localized to one component.
- Low entropy (close to 0) means the same thing; higher entropy means ToM-selective parameters are spread across components.

| Setting | Vision Encoder | Cross-Modal | Language | Concentration | Entropy |
|---------|---------------|-------------|----------|--------------|---------|
| LLaVA 1.5 | 0.0% | 0.0% | **100.0%** | 1.00 | 0.00 |
| InstructBLIP | 0.0% | 1.3% | **98.7%** | 0.99 | 0.10 |
| Qwen3-VL (multimodal) | **98.7%** | 0.6% | 0.7% | 0.99 | 0.11 |
| Qwen3-VL (text-only) | 0.0% | 0.0% | **100.0%** | 1.00 | 0.00 |
| Qwen3-VL (video-only) | 9.2% | 3.7% | **87.2%** | 0.87 | 0.66 |

*Percentages show the fraction of global top-0.1% most ToM-selective parameters in each group.*

**Key findings:**
- **LLaVA and InstructBLIP** concentrate nearly all ToM-selective parameters in the **language backbone** (99-100%). Despite having high absolute sensitivity in the vision encoder (Section 5.1), those vision parameters respond similarly to both ToM and non-ToM inputs, making them non-selective. The parameters that are *specifically* ToM-engaged reside in the language model.
- **Qwen3-VL (multimodal)** shows the opposite pattern: 98.7% of ToM-selective parameters are in the **vision encoder**. However, this is influenced by the near-zero non-ToM gradients in the vision module (inflating selectivity ratios).
- **Qwen3-VL (video-only)** shows the most distributed pattern (entropy = 0.66), with 87% in language, 9% in vision, and 4% in the cross-modal connector.
- All settings show high concentration (index >= 0.87), meaning ToM-selective parameters are not uniformly spread but clustered in specific architectural components.

---

## Methodology Summary

1. **Gradient computation:** For each model and input mode, we run a forward+backward pass on each of the 100 ToM (or non-ToM) questions. A hook replaces each gradient g with g² before accumulation. The accumulated squared gradients measure per-parameter importance (Fisher information approximation).

2. **Chunking:** The accumulated squared gradients are grouped by architectural component (text decoder layers, vision encoder layers, connector/projector layers) and saved as per-layer chunk files.

3. **Differential masking:** For a given masking rate m, we identify parameters that are in the top-m% by ToM gradient magnitude but NOT in the top-m% by non-ToM gradient magnitude. These "differentially important" parameters are ablated (replaced with group mean).

4. **Evaluation:** The masked model generates answers (greedy decoding, max 5 tokens) for each ToM question. The first "a" or "b" token in the output is taken as the prediction.

5. **Sensitivity analysis:** RS = Σ grad²_ToM / Σ grad²_nonToM per parameter group, measuring whether a group is relatively more engaged by ToM or factual reasoning.

6. **Intrinsic metrics (Section 5):** Three additional metrics computed from the gradient chunks with a 3-way module grouping (vision encoder / cross-modal connector / language backbone):
   - **ToM Sensitivity Score** -- mean empirical Fisher of top-1% parameters per group, measuring absolute ToM engagement.
   - **ToM Selectivity Score** -- per-parameter ratio of ToM to non-ToM gradients, measuring ToM *specificity* (high sensitivity that is not shared with factual tasks).
   - **Localization Concentration** -- global top-0.1% most ToM-selective parameters, counted by group, measuring where in the architecture ToM knowledge is stored.
