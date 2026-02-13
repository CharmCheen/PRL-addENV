#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

CHECKPOINT="${CHECKPOINT:-${1:-}}"
if [[ -z "${CHECKPOINT}" ]]; then
  echo "Usage: CHECKPOINT=/path/to/checkpoint bash repro/mr_paper/run_mr_test_eval.sh [checkpoint]"
  exit 1
fi

NUMBER_OF_PROMPTS="${NUMBER_OF_PROMPTS:-1}"
SELECTION_MODE="${SELECTION_MODE:-single}"   # single | best
LIMIT="${LIMIT:-}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-16}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_P="${TOP_P:-1.0}"
BATCH_SIZE="${BATCH_SIZE:-8}"
SEED="${SEED:-42}"
BASE_MODEL="${BASE_MODEL:-}"
MODEL_TYPE="${MODEL_TYPE:-}"

# Keep all runtime caches in writable workspace paths.
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-$ROOT_DIR/.cache/modelscope}"
export HF_HOME="${HF_HOME:-$ROOT_DIR/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$ROOT_DIR/.cache/huggingface/transformers}"
mkdir -p "$MODELSCOPE_CACHE" "$HF_HOME" "$TRANSFORMERS_CACHE"

OUT_DIR="repro/mr_paper/results"
mkdir -p "$OUT_DIR"
TS="$(date +%Y%m%d-%H%M%S)"
OUT_JSON="$OUT_DIR/mr_test_eval_${TS}.json"

CMD=(
  python repro/mr_paper/run_mr_test_eval.py
  --checkpoint "$CHECKPOINT"
  --selection-mode "$SELECTION_MODE"
  --number-of-prompts "$NUMBER_OF_PROMPTS"
  --max-new-tokens "$MAX_NEW_TOKENS"
  --temperature "$TEMPERATURE"
  --top-p "$TOP_P"
  --batch-size "$BATCH_SIZE"
  --seed "$SEED"
  --output-json "$OUT_JSON"
)

if [[ -n "${BASE_MODEL}" ]]; then
  CMD+=(--base-model "$BASE_MODEL")
fi

if [[ -n "${MODEL_TYPE}" ]]; then
  CMD+=(--model-type "$MODEL_TYPE")
fi

if [[ -n "${LIMIT}" ]]; then
  CMD+=(--limit "$LIMIT")
fi

printf 'Running command:\n%s\n' "${CMD[*]}"
"${CMD[@]}"
