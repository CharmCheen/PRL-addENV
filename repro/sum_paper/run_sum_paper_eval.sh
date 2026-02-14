#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

TASK="sum"
BASE_MODEL="${BASE_MODEL:-/qiuyeqing/.cache/modelscope/models/Qwen/Qwen2___5-7B-Instruct}"
MODEL_TYPE="${MODEL_TYPE:-qwen2_5}"
CHECKPOINT="${CHECKPOINT:-}"
TRAIN_FILE="${TRAIN_FILE:-datasets/original/sum_train.jsonl}"
VAL_FILE="${VAL_FILE:-datasets/original/sum_val.jsonl}"
TEST_FILE="${TEST_FILE:-datasets/original/sum_test.jsonl}"
NUMBER_OF_PROMPTS="${NUMBER_OF_PROMPTS:-3}"
TEMPERATURE="${TEMPERATURE:-0.9}"
TOP_P="${TOP_P:-0.9}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
BATCH_SIZE="${BATCH_SIZE:-8}"
SEED="${SEED:-42}"
LIMIT="${LIMIT:-}"
OUTPUT_JSON="${OUTPUT_JSON:-}"
FIXED_PROMPT="${FIXED_PROMPT:-}"

export HF_HOME="${HF_HOME:-${REPO_ROOT}/.cache/hf}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-${REPO_ROOT}/.cache/modelscope}"
export USE_HF="${USE_HF:-0}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONUNBUFFERED=1

mkdir -p "${REPO_ROOT}/repro/sum_paper/runs/verify_smoke" "${REPO_ROOT}/repro/sum_paper/runs/paper_align"

CMD=(
  python repro/sum_paper/run_sum_paper_eval.py
  --base-model "${BASE_MODEL}"
  --model-type "${MODEL_TYPE}"
  --train-file "${TRAIN_FILE}"
  --val-file "${VAL_FILE}"
  --test-file "${TEST_FILE}"
  --number-of-prompts "${NUMBER_OF_PROMPTS}"
  --temperature "${TEMPERATURE}"
  --top-p "${TOP_P}"
  --max-new-tokens "${MAX_NEW_TOKENS}"
  --batch-size "${BATCH_SIZE}"
  --seed "${SEED}"
)

if [[ -n "${CHECKPOINT}" ]]; then
  CMD+=(--checkpoint "${CHECKPOINT}")
fi
if [[ -n "${LIMIT}" ]]; then
  CMD+=(--limit "${LIMIT}")
fi
if [[ -n "${OUTPUT_JSON}" ]]; then
  CMD+=(--output-json "${OUTPUT_JSON}")
fi
if [[ -n "${FIXED_PROMPT}" ]]; then
  CMD+=(--fixed-prompt "${FIXED_PROMPT}")
fi

echo "=== SUM Paper Eval ==="
echo "BASE_MODEL=${BASE_MODEL}"
echo "CHECKPOINT=${CHECKPOINT:-none}"
echo "SEED=${SEED} LIMIT=${LIMIT:-none}"
echo "DECODE: temperature=${TEMPERATURE}, top_p=${TOP_P}, max_new_tokens=${MAX_NEW_TOKENS}"

"${CMD[@]}"
