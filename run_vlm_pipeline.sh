#!/usr/bin/env bash
# =========================================================================
# VLM Theory-of-Mind Pipeline Runner
#
# Runs the full pipeline for 5 settings:
#   1. LLaVA 1.5          (text + video)
#   2. InstructBLIP        (text + video)
#   3. Qwen3-VL            (text + video)
#   4. Qwen3-VL            (text-only)
#   5. Qwen3-VL            (video-only)
#
# Usage:
#   conda activate tom
#   bash run_vlm_pipeline.sh [RESULTS_ROOT] [CACHE_DIR]
# =========================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

RESULTS_ROOT="${1:-${SCRIPT_DIR}/vlm_results}"
CACHE_DIR="${2:-/data/user_data/${USER}/hf_cache}"

DATA_DIR="${SCRIPT_DIR}/data"
TOM_DATA="${DATA_DIR}/sampled_tom_questions.jsonl"
NONTOM_DATA="${DATA_DIR}/sampled_non_tom_questions.jsonl"
VIDEO_DIR="/data/user_data/jinyuhou/single_agent_partial_train_240_hp_test_highres"

MAX_FRAMES=8
M_LIST="0.0,5e-6,1e-5,2e-5,5e-5"

# Model IDs
LLAVA="llava-hf/llava-1.5-7b-hf"
IBLIP="Salesforce/instructblip-vicuna-7b"
QWEN="Qwen/Qwen3-VL-8B-Instruct"

mkdir -p "${RESULTS_ROOT}"

# =========================================================================
# Helper: run one setting end-to-end
# =========================================================================
run_setting() {
    local TAG="$1"       # e.g. llava_multimodal
    local MODEL="$2"     # HF model id
    local MODE="$3"      # multimodal | text_only | video_only

    local SDIR="${RESULTS_ROOT}/${TAG}"
    mkdir -p "${SDIR}"

    echo ""
    echo "============================================================"
    echo "  Setting: ${TAG}  (model=${MODEL}, mode=${MODE})"
    echo "============================================================"

    # ---- Step 1: Gradient computation (ToM) ----
    local GRAD_TOM="${SDIR}/grad_tom"
    if [ ! -f "${GRAD_TOM}/config.json" ]; then
        echo "[1/5] Computing ToM gradients ..."
        python "${SCRIPT_DIR}/create_gradient_vlm.py" \
            --model "${MODEL}" \
            --data_path "${TOM_DATA}" \
            --video_dir "${VIDEO_DIR}" \
            --out "${GRAD_TOM}" \
            --cache_dir "${CACHE_DIR}" \
            --mode "${MODE}" \
            --max_frames "${MAX_FRAMES}"
    else
        echo "[1/5] ToM gradients already exist, skipping."
    fi

    # ---- Step 2: Gradient computation (non-ToM) ----
    local GRAD_NONTOM="${SDIR}/grad_nontom"
    if [ ! -f "${GRAD_NONTOM}/config.json" ]; then
        echo "[2/5] Computing non-ToM gradients ..."
        python "${SCRIPT_DIR}/create_gradient_vlm.py" \
            --model "${MODEL}" \
            --data_path "${NONTOM_DATA}" \
            --video_dir "${VIDEO_DIR}" \
            --out "${GRAD_NONTOM}" \
            --cache_dir "${CACHE_DIR}" \
            --mode "${MODE}" \
            --max_frames "${MAX_FRAMES}"
    else
        echo "[2/5] Non-ToM gradients already exist, skipping."
    fi

    # ---- Step 3: Chunk gradients ----
    local CHUNKS_TOM="${SDIR}/chunks_tom"
    local CHUNKS_NONTOM="${SDIR}/chunks_nontom"
    if [ ! -f "${CHUNKS_TOM}/manifest.json" ]; then
        echo "[3/5] Chunking ToM gradients ..."
        python "${SCRIPT_DIR}/chunk_gradient_vlm.py" \
            --model "${GRAD_TOM}" \
            --output_path "${CHUNKS_TOM}" \
            --cache_dir "${CACHE_DIR}" \
            --device_map cpu
    else
        echo "[3/5] ToM chunks already exist, skipping."
    fi

    if [ ! -f "${CHUNKS_NONTOM}/manifest.json" ]; then
        echo "[3/5] Chunking non-ToM gradients ..."
        python "${SCRIPT_DIR}/chunk_gradient_vlm.py" \
            --model "${GRAD_NONTOM}" \
            --output_path "${CHUNKS_NONTOM}" \
            --cache_dir "${CACHE_DIR}" \
            --device_map cpu
    else
        echo "[3/5] Non-ToM chunks already exist, skipping."
    fi

    # ---- Step 4: Masking sweep + evaluation ----
    if [ ! -f "${SDIR}/eval_results.csv" ]; then
        echo "[4/5] Running masking sweep evaluation ..."
        python "${SCRIPT_DIR}/vlm_evaluation.py" \
            --model "${MODEL}" \
            --grad_tom_chunks "${CHUNKS_TOM}" \
            --grad_nontom_chunks "${CHUNKS_NONTOM}" \
            --questions "${TOM_DATA}" \
            --video_dir "${VIDEO_DIR}" \
            --out_dir "${SDIR}" \
            --cache_dir "${CACHE_DIR}" \
            --mode "${MODE}" \
            --max_frames "${MAX_FRAMES}" \
            --m_list "${M_LIST}"
    else
        echo "[4/5] Evaluation results already exist, skipping."
    fi

    echo "[5/5] Setting ${TAG} complete."
}

# =========================================================================
# Run all 5 settings
# =========================================================================
run_setting "llava_multimodal"        "${LLAVA}" "multimodal"
run_setting "instructblip_multimodal" "${IBLIP}" "multimodal"
run_setting "qwen3vl_multimodal"      "${QWEN}"  "multimodal"
run_setting "qwen3vl_text_only"       "${QWEN}"  "text_only"
run_setting "qwen3vl_video_only"      "${QWEN}"  "video_only"

# =========================================================================
# Generate final report
# =========================================================================
echo ""
echo "============================================================"
echo "  Generating final report"
echo "============================================================"
python "${SCRIPT_DIR}/report_vlm.py" \
    --results_root "${RESULTS_ROOT}" \
    --out "${RESULTS_ROOT}/report.md"

echo ""
echo "=== ALL DONE ==="
echo "Results root: ${RESULTS_ROOT}"
echo "Report:       ${RESULTS_ROOT}/report.md"
