# SIM Paper-Aligned Evaluation

This module implements a paper-aligned evaluation path for **simplification (`sim`)** with strict split separation:

- Prompt selection: **validation split** (`datasets/original/sim_val.jsonl`, 100 samples)
- Final report: **test split** (`datasets/original/sim_test.jsonl`)
- Metric: **SARI**
- Statistics: **3 independent seeds** (`42/43/44`), report **mean ± std**

It does **not** modify core training scripts or trainer behavior.

## Protocol Alignment

- Validation is used for prompt selection; test is used for final reporting.
- Development size for summarization/simplification is 100 (paper appendix).
- MR-only `n_test` mechanism is not used.

## Metric Implementation

`run_sim_paper_eval.py` computes SARI with a minimal local implementation:

- n-gram order: 1 to 4
- operation terms: `ADD(F1)`, `KEEP(F1)`, `DELETE(precision)`
- final score: average across 1..4 n-grams, scaled to 0-100
- tokenization: `mosestokenizer` English tokenizer

This fallback is used because `easse` is not guaranteed in the runtime.

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
    "task": "sim",
    "split": "test",
    "selection_split": "val",
    "metric": "sari",
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
    "score": 0.0
  },
  "test": {
    "sari": 0.0
  }
}
```

## Offline Setup

`run_sim_paper_eval.sh` sets:

- `MODELSCOPE_CACHE=$REPO/.cache/modelscope`
- `HF_HOME=$REPO/.cache/hf`
- `TRANSFORMERS_CACHE=$HF_HOME/transformers`

So the run can use local cached models.

## Usage

Smoke (`LIMIT=20`):

```bash
LIMIT=20 SEED=42 OUTPUT_JSON=repro/sim_paper/runs/verify_smoke/seed42_limit20.json \
bash repro/sim_paper/run_sim_paper_eval.sh
```

Full test (single seed):

```bash
SEED=42 OUTPUT_JSON=repro/sim_paper/runs/paper_align/seed42.json \
bash repro/sim_paper/run_sim_paper_eval.sh 2>&1 | tee repro/sim_paper/runs/paper_align/seed42.log
```

Three seeds + summary:

```bash
mkdir -p repro/sim_paper/runs/paper_align
for SEED in 42 43 44; do
  OUTPUT_JSON=repro/sim_paper/runs/paper_align/seed${SEED}.json \
  SEED=${SEED} bash repro/sim_paper/run_sim_paper_eval.sh \
    2>&1 | tee repro/sim_paper/runs/paper_align/seed${SEED}.log
done
python - <<'PY'
import json, statistics
from pathlib import Path
out = Path("repro/sim_paper/runs/paper_align")
runs = []
for s in [42, 43, 44]:
    d = json.load(open(out / f"seed{s}.json"))
    runs.append({
        "seed": s,
        "sari": d["test"]["sari"],
        "json": str(out / f"seed{s}.json"),
        "log": str(out / f"seed{s}.log"),
    })
paper = 52.26
mean = statistics.mean([r["sari"] for r in runs])
std = statistics.pstdev([r["sari"] for r in runs])
delta = mean - paper
summary = {"task":"sim","n_runs":3,"runs":runs,"mean":mean,"std":std,"paper_value":paper,"delta":delta}
json.dump(summary, open(out / "summary.json", "w"), indent=2, ensure_ascii=False)
with open(out / "summary.md","w",encoding="utf-8") as f:
    f.write("# SIM Paper-Align Summary\n\n")
    f.write("| seed | SARI | json | log |\n")
    f.write("| --- | ---: | --- | --- |\n")
    for r in runs:
        f.write(f"| {r['seed']} | {r['sari']:.4f} | {r['json']} | {r['log']} |\n")
    f.write("\n")
    f.write(f"- mean(SARI): {mean:.4f}\n")
    f.write(f"- std(SARI): {std:.4f}\n")
    f.write(f"- paper(PRL): {paper:.2f}\n")
    f.write(f"- delta: {delta:.4f}\n")
PY
```
