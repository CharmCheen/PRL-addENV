# SUM Paper-Aligned Evaluation

This module implements a paper-aligned evaluation path for **summarization (`sum`)** with strict split separation:

- Prompt selection: **validation split** (`datasets/original/sum_val.jsonl`, 100 samples)
- Final report: **test split** (`datasets/original/sum_test.jsonl`)
- Metrics: **ROUGE-1 / ROUGE-2 / ROUGE-L**
- Statistics: **3 independent seeds** (`42/43/44`), report **mean ± std**

It does **not** modify core training scripts or trainer behavior.

## Protocol Alignment

- Paper protocol uses validation for prompt selection and test for reporting.
- Development size for summarization/simplification is 100 (paper appendix).
- This module avoids MR-specific `n_test` ablation logic.

## Metric Implementation

`run_sum_paper_eval.py` computes ROUGE via:

- `rouge` package (`Rouge().get_scores(..., avg=True)`)
- `mosestokenizer` English tokenization before ROUGE
- Returns percentage values (`*100`) for `rouge1/rouge2/rougeL`

## Outputs

Single run JSON schema:

```json
{
  "meta": {
    "timestamp": "...",
    "git_commit": "...",
    "checkpoint": "...",
    "base_model": "...",
    "model_type": "qwen2_5",
    "task": "sum",
    "split": "test",
    "selection_split": "val",
    "metric": "rouge1/rouge2/rougeL",
    "seed": 42,
    "batch_size": 8,
    "decode_params": {
      "temperature": 0.9,
      "top_p": 0.9,
      "max_new_tokens": 1024
    }
  },
  "validation": {
    "selected_prompt": "...",
    "score": {
      "rouge_avg": 0.0,
      "rouge1": 0.0,
      "rouge2": 0.0,
      "rougeL": 0.0
    }
  },
  "test": {
    "rouge1": 0.0,
    "rouge2": 0.0,
    "rougeL": 0.0
  }
}
```

## Offline Setup

`run_sum_paper_eval.sh` sets:

- `MODELSCOPE_CACHE=$REPO/.cache/modelscope`
- `HF_HOME=$REPO/.cache/hf`
- `TRANSFORMERS_CACHE=$HF_HOME/transformers`

So the run can use local cached models.

## Usage

Smoke (`LIMIT=20`):

```bash
LIMIT=20 SEED=42 OUTPUT_JSON=repro/sum_paper/runs/verify_smoke/seed42_limit20.json \
bash repro/sum_paper/run_sum_paper_eval.sh
```

Full test (single seed):

```bash
SEED=42 OUTPUT_JSON=repro/sum_paper/runs/paper_align/seed42.json \
bash repro/sum_paper/run_sum_paper_eval.sh 2>&1 | tee repro/sum_paper/runs/paper_align/seed42.log
```

Three seeds + summary:

```bash
mkdir -p repro/sum_paper/runs/paper_align
for SEED in 42 43 44; do
  OUTPUT_JSON=repro/sum_paper/runs/paper_align/seed${SEED}.json \
  SEED=${SEED} bash repro/sum_paper/run_sum_paper_eval.sh \
    2>&1 | tee repro/sum_paper/runs/paper_align/seed${SEED}.log
done
python - <<'PY'
import json, statistics
from pathlib import Path
out = Path("repro/sum_paper/runs/paper_align")
runs = []
for s in [42, 43, 44]:
    d = json.load(open(out / f"seed{s}.json"))
    runs.append({
        "seed": s,
        "rouge1": d["test"]["rouge1"],
        "rouge2": d["test"]["rouge2"],
        "rougeL": d["test"]["rougeL"],
        "rouge_avg": d["test"]["rouge_avg"],
        "json": str(out / f"seed{s}.json"),
        "log": str(out / f"seed{s}.log"),
    })
paper = {"rouge1": 42.47, "rouge2": 16.17, "rougeL": 37.73}
mean = {k: statistics.mean([r[k] for r in runs]) for k in ["rouge1","rouge2","rougeL","rouge_avg"]}
std = {k: statistics.pstdev([r[k] for r in runs]) for k in ["rouge1","rouge2","rougeL","rouge_avg"]}
delta = {k: mean[k] - paper[k] for k in ["rouge1","rouge2","rougeL"]}
summary = {"task":"sum","n_runs":3,"runs":runs,"mean":mean,"std":std,"paper_value":paper,"delta":delta}
json.dump(summary, open(out / "summary.json", "w"), indent=2, ensure_ascii=False)
with open(out / "summary.md","w",encoding="utf-8") as f:
    f.write("# SUM Paper-Align Summary\n\n")
    f.write("| seed | ROUGE-1 | ROUGE-2 | ROUGE-L | ROUGE-AVG | json | log |\n")
    f.write("| --- | ---: | ---: | ---: | ---: | --- | --- |\n")
    for r in runs:
        f.write(f"| {r['seed']} | {r['rouge1']:.4f} | {r['rouge2']:.4f} | {r['rougeL']:.4f} | {r['rouge_avg']:.4f} | {r['json']} | {r['log']} |\n")
    f.write("\n")
    f.write(f"- mean(ROUGE-1/2/L): {mean['rouge1']:.4f} / {mean['rouge2']:.4f} / {mean['rougeL']:.4f}\n")
    f.write(f"- std(ROUGE-1/2/L): {std['rouge1']:.4f} / {std['rouge2']:.4f} / {std['rougeL']:.4f}\n")
    f.write(f"- paper(PRL): {paper['rouge1']:.2f} / {paper['rouge2']:.2f} / {paper['rougeL']:.2f}\n")
    f.write(f"- delta: {delta['rouge1']:.4f} / {delta['rouge2']:.4f} / {delta['rougeL']:.4f}\n")
PY
```
