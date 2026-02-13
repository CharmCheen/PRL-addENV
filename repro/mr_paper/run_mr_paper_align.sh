#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

CHECKPOINT="${CHECKPOINT:-${1:-}}"
if [[ -z "${CHECKPOINT}" ]]; then
  echo "Usage: CHECKPOINT=/path/to/checkpoint bash repro/mr_paper/run_mr_paper_align.sh [checkpoint]"
  exit 1
fi

# Paper-aligned defaults (best-of prompts + training-like decoding).
SELECTION_MODE="${SELECTION_MODE:-best}"
NUMBER_OF_PROMPTS="${NUMBER_OF_PROMPTS:-10}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-384}"    # align with checkpoint max_completion_length used by trainer eval
TEMPERATURE="${TEMPERATURE:-0.9}"
TOP_P="${TOP_P:-0.9}"
BATCH_SIZE="${BATCH_SIZE:-8}"
BASE_MODEL="${BASE_MODEL:-}"
MODEL_TYPE="${MODEL_TYPE:-}"
TEST_FILE="${TEST_FILE:-datasets/original/mr_test.jsonl}"
TRAIN_FILE="${TRAIN_FILE:-datasets/original/mr_train.jsonl}"
LIMIT="${LIMIT:-}"

# Seed policy:
# 1) If SEED is set, run exactly one seed.
# 2) Otherwise run SEEDS (default: "42 43 44").
if [[ -n "${SEED:-}" ]]; then
  SEED_LIST="${SEED}"
else
  SEED_LIST="${SEEDS:-42 43 44}"
fi

export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-$ROOT_DIR/.cache/modelscope}"
export HF_HOME="${HF_HOME:-$ROOT_DIR/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$ROOT_DIR/.cache/huggingface/transformers}"
mkdir -p "$MODELSCOPE_CACHE" "$HF_HOME" "$TRANSFORMERS_CACHE"

RESULT_DIR="repro/mr_paper/results"
mkdir -p "$RESULT_DIR"
TS="$(date +%Y%m%d-%H%M%S)"
SUMMARY_JSON="$RESULT_DIR/summary_mr_paper_align.json"
RUN_META_TXT="$RESULT_DIR/.mr_paper_align_runs_${TS}.txt"
: > "$RUN_META_TXT"

echo "=== MR Paper-Align Eval ==="
echo "checkpoint: $CHECKPOINT"
echo "selection_mode: $SELECTION_MODE"
echo "number_of_prompts: $NUMBER_OF_PROMPTS"
echo "temperature/top_p/max_new_tokens: $TEMPERATURE/$TOP_P/$MAX_NEW_TOKENS"
echo "seeds: $SEED_LIST"
echo

for CUR_SEED in $SEED_LIST; do
  OUT_JSON="$RESULT_DIR/mr_paper_align_seed${CUR_SEED}_${TS}.json"
  OUT_LOG="$RESULT_DIR/mr_paper_align_seed${CUR_SEED}_${TS}.log"

  CMD=(
    python repro/mr_paper/run_mr_test_eval.py
    --checkpoint "$CHECKPOINT"
    --selection-mode "$SELECTION_MODE"
    --number-of-prompts "$NUMBER_OF_PROMPTS"
    --max-new-tokens "$MAX_NEW_TOKENS"
    --temperature "$TEMPERATURE"
    --top-p "$TOP_P"
    --batch-size "$BATCH_SIZE"
    --seed "$CUR_SEED"
    --test-file "$TEST_FILE"
    --train-file "$TRAIN_FILE"
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

  echo "[run] seed=$CUR_SEED"
  echo "[cmd] ${CMD[*]}"
  "${CMD[@]}" 2>&1 | tee "$OUT_LOG"

  ACC="$(python -c "import json;print(json.load(open('$OUT_JSON'))['best']['accuracy'])")"
  CORRECT="$(python -c "import json;d=json.load(open('$OUT_JSON'));print(d['best']['correct'])")"
  TOTAL="$(python -c "import json;d=json.load(open('$OUT_JSON'));print(d['best']['total'])")"
  echo "[result] seed=$CUR_SEED accuracy=$ACC ($CORRECT/$TOTAL)"
  echo -e "${CUR_SEED}\t${ACC}\t${CORRECT}\t${TOTAL}\t${OUT_JSON}\t${OUT_LOG}" >> "$RUN_META_TXT"
  echo
done

python - "$RUN_META_TXT" "$SUMMARY_JSON" << 'PY'
import json
import statistics
import sys
from datetime import datetime, timezone

run_meta_txt, summary_json = sys.argv[1], sys.argv[2]
runs = []
with open(run_meta_txt, 'r', encoding='utf-8') as f:
    for line in f:
        seed, acc, correct, total, out_json, out_log = line.rstrip('\n').split('\t')
        runs.append({
            'seed': int(seed),
            'accuracy': float(acc),
            'correct': int(correct),
            'total': int(total),
            'result_json': out_json,
            'log_file': out_log,
        })

accs = [r['accuracy'] for r in runs]
mean_acc = statistics.mean(accs)
std_acc = statistics.stdev(accs) if len(accs) > 1 else 0.0

paper_mean = 0.9127
paper_std = 0.0005
summary = {
    'timestamp_utc': datetime.now(timezone.utc).isoformat(),
    'n_runs': len(runs),
    'runs': runs,
    'mean_accuracy': mean_acc,
    'std_accuracy': std_acc,
    'paper_target': {
        'mean_accuracy': paper_mean,
        'std_accuracy': paper_std,
        'display': '91.27±0.05',
    },
    'gap_to_paper': {
        'mean_gap': mean_acc - paper_mean,
        'abs_mean_gap': abs(mean_acc - paper_mean),
        'std_gap': std_acc - paper_std,
    }
}

with open(summary_json, 'w', encoding='utf-8') as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print('=== Aggregate Summary ===')
print(f'mean_accuracy: {mean_acc:.6f}')
print(f'std_accuracy: {std_acc:.6f}')
print(f'paper_target: 91.27±0.05')
print(f'gap(mean): {mean_acc - paper_mean:+.6f}')
print(f'summary_json: {summary_json}')
PY

rm -f "$RUN_META_TXT"
