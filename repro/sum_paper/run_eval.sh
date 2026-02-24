#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

CHECKPOINT="${CHECKPOINT:-${1:-}}"
SUM_RUN_DIR="${SUM_RUN_DIR:-}"
CHECKPOINT_CHOICE="${CHECKPOINT_CHOICE:-last}"  # last | best (used only when SUM_RUN_DIR is set)
EVAL_ONLY="${EVAL_ONLY:-0}"  # 1 = skip generation, only score existing predictions

if [[ -z "${CHECKPOINT}" && -z "${SUM_RUN_DIR}" ]]; then
  echo "Usage:"
  echo "  CHECKPOINT=/path/to/checkpoint bash repro/sum_paper/run_eval.sh"
  echo "  or"
  echo "  SUM_RUN_DIR=output/sum/sum-qwen-qwen-YYYY... bash repro/sum_paper/run_eval.sh"
  exit 1
fi

NUMBER_OF_PROMPTS="${NUMBER_OF_PROMPTS:-10}"
LIMIT="${LIMIT:-}"
TEMPERATURE="${TEMPERATURE:-0.9}"
TOP_P="${TOP_P:-0.9}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
BATCH_SIZE="${BATCH_SIZE:-8}"
SEED="${SEED:-42}"
BASE_MODEL="${BASE_MODEL:-}"
MODEL_TYPE="${MODEL_TYPE:-}"
FIXED_PROMPT="${FIXED_PROMPT:-}"
TRAIN_FILE="${TRAIN_FILE:-datasets/original/sum_train.jsonl}"
VAL_FILE="${VAL_FILE:-datasets/original/sum_val.jsonl}"
TEST_FILE="${TEST_FILE:-datasets/original/sum_test.jsonl}"

export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-$ROOT_DIR/.cache/modelscope}"
export HF_HOME="${HF_HOME:-$ROOT_DIR/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$ROOT_DIR/.cache/huggingface/transformers}"
mkdir -p "$MODELSCOPE_CACHE" "$HF_HOME" "$TRANSFORMERS_CACHE"

# Force single-GPU eval to avoid NCCL hang
if [[ "${EVAL_ONLY}" == "1" ]]; then
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
  export NPROC_PER_NODE=1
  export DATALOADER_NUM_WORKERS=0
fi

OUT_ROOT="repro/sum_paper/results"
TS="$(date +%Y%m%d-%H%M%S)"
OUT_DIR="$OUT_ROOT/sum_test_eval_${TS}"
mkdir -p "$OUT_DIR"

RUN_JSON="$OUT_DIR/run_summary.json"
METRICS_JSON="$OUT_DIR/metrics.json"
PRED_JSONL="$OUT_DIR/predictions.jsonl"

CMD=(
  python3 repro/sum_paper/run_sum_test_eval.py
  --checkpoint-choice "$CHECKPOINT_CHOICE"
  --train-file "$TRAIN_FILE"
  --val-file "$VAL_FILE"
  --test-file "$TEST_FILE"
  --number-of-prompts "$NUMBER_OF_PROMPTS"
  --temperature "$TEMPERATURE"
  --top-p "$TOP_P"
  --max-new-tokens "$MAX_NEW_TOKENS"
  --batch-size "$BATCH_SIZE"
  --seed "$SEED"
  --output-json "$RUN_JSON"
  --metrics-json "$METRICS_JSON"
  --pred-path "$PRED_JSONL"
)

if [[ -n "${CHECKPOINT}" ]]; then
  CMD+=(--checkpoint "$CHECKPOINT")
fi
if [[ -n "${SUM_RUN_DIR}" ]]; then
  CMD+=(--sum-run-dir "$SUM_RUN_DIR")
fi
if [[ -n "${BASE_MODEL}" ]]; then
  CMD+=(--base-model "$BASE_MODEL")
fi
if [[ -n "${MODEL_TYPE}" ]]; then
  CMD+=(--model-type "$MODEL_TYPE")
fi
if [[ -n "${LIMIT}" ]]; then
  CMD+=(--limit "$LIMIT")
fi
if [[ -n "${FIXED_PROMPT}" ]]; then
  CMD+=(--fixed-prompt "$FIXED_PROMPT")
fi
if [[ "${EVAL_ONLY}" == "1" ]]; then
  CMD+=(--eval-only)
fi

printf 'Running command:\n%s\n' "${CMD[*]}"
"${CMD[@]}"
