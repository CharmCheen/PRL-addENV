#!/usr/bin/env bash
set -euo pipefail

cd /qiuyeqing/llama_prl/PRL-REDO/PRL-addENV

SUM_RUN_DIR="output/sum/sum-qwen-qwen-20260214-104456"
RESULTS_DIR="repro/sum_paper/results"
mkdir -p "$RESULTS_DIR"

echo "=== Starting 3-seed evaluation for paper alignment ==="
echo "SUM_RUN_DIR: $SUM_RUN_DIR"
echo "RESULTS_DIR: $RESULTS_DIR"
echo "Start time: $(date)"
echo ""

for SEED in 42 43 44; do
  echo ">>> Running seed ${SEED} at $(date)..."

  CUDA_VISIBLE_DEVICES=0 \
  NPROC_PER_NODE=1 \
  SEED=${SEED} \
  SUM_RUN_DIR="$SUM_RUN_DIR" \
  bash repro/sum_paper/run_eval.sh 2>&1 | tee "$RESULTS_DIR/seed${SEED}.log"

  EXIT_CODE=${PIPESTATUS[0]}
  if [[ $EXIT_CODE -ne 0 ]]; then
    echo "ERROR: Seed ${SEED} failed with exit code $EXIT_CODE"
    exit $EXIT_CODE
  fi

  echo ">>> Seed ${SEED} completed successfully at $(date)"
  echo ""
done

echo "=== All 3 seeds completed at $(date) ==="
echo ""
echo "Computing summary statistics..."

python3 - <<'EOF'
import json
import statistics
from pathlib import Path

results_dir = Path("repro/sum_paper/results")
runs = []

for seed in [42, 43, 44]:
    log_file = results_dir / f"seed{seed}.log"
    if not log_file.exists():
        print(f"Warning: {log_file} not found")
        continue

    # Extract run_summary.json path from log
    with open(log_file) as f:
        for line in f:
            if "run_summary_json:" in line:
                json_path = Path(line.split("run_summary_json:")[-1].strip())
                if json_path.exists():
                    data = json.load(open(json_path))
                    test_metrics = data.get("test_metrics", {})
                    runs.append({
                        "seed": seed,
                        "rouge1": test_metrics.get("rouge1", 0.0),
                        "rouge2": test_metrics.get("rouge2", 0.0),
                        "rougeL": test_metrics.get("rougeL", 0.0),
                        "rouge_avg": test_metrics.get("rouge_avg", 0.0),
                        "json_path": str(json_path),
                    })
                    break

if len(runs) < 3:
    print(f"Error: Only {len(runs)} runs completed")
    exit(1)

paper = {"rouge1": 42.47, "rouge2": 16.17, "rougeL": 37.73}
mean = {k: statistics.mean([r[k] for r in runs]) for k in ["rouge1", "rouge2", "rougeL", "rouge_avg"]}
std = {k: statistics.pstdev([r[k] for r in runs]) for k in ["rouge1", "rouge2", "rougeL", "rouge_avg"]}
delta = {k: mean[k] - paper[k] for k in ["rouge1", "rouge2", "rougeL"]}

summary = {
    "task": "sum",
    "n_runs": len(runs),
    "runs": runs,
    "mean": mean,
    "std": std,
    "paper_value": paper,
    "delta": delta,
}

summary_path = results_dir / "summary_3seeds.json"
json.dump(summary, open(summary_path, "w"), indent=2, ensure_ascii=False)

print("\n=== 3-Seed Summary Statistics ===")
print(f"Mean: R1={mean['rouge1']:.2f} R2={mean['rouge2']:.2f} RL={mean['rougeL']:.2f} Avg={mean['rouge_avg']:.2f}")
print(f"Std:  R1={std['rouge1']:.2f} R2={std['rouge2']:.2f} RL={std['rougeL']:.2f} Avg={std['rouge_avg']:.2f}")
print(f"Paper: R1={paper['rouge1']:.2f} R2={paper['rouge2']:.2f} RL={paper['rougeL']:.2f}")
print(f"Delta: R1={delta['rouge1']:+.2f} R2={delta['rouge2']:+.2f} RL={delta['rougeL']:+.2f}")
print(f"\nSummary saved to: {summary_path}")

# Print individual runs
print("\n=== Individual Runs ===")
for r in runs:
    print(f"Seed {r['seed']}: R1={r['rouge1']:.2f} R2={r['rouge2']:.2f} RL={r['rougeL']:.2f}")
EOF

echo ""
echo "=== Done! ==="
echo "Results saved to: $RESULTS_DIR"
echo "Summary: $RESULTS_DIR/summary_3seeds.json"
