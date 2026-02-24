# SUM Paper Alignment Notes

## 1) Paper-derived scoring definition (from `docs/paper/base_ref.pdf`)

Source extraction was done directly from the local PDF content stream (no network), focusing on the **summarization** experiment section and **Table 2**.

Paper-aligned points:

1. Task + dataset:
- Summarization task is reported on **SAMSUM** (paper text: summarization benchmark section).
- In this repo, data is already materialized as:
  - `datasets/original/sum_train.jsonl`
  - `datasets/original/sum_val.jsonl`
  - `datasets/original/sum_test.jsonl`
- So implementation keeps repo datasets, while preserving paper metric definition.

2. Metrics:
- Paper explicitly uses **ROUGE-1 / ROUGE-2 / ROUGE-L** for summarization (see Table 2 description and ROUGE bullet list).
- We implement **ROUGE F1** (`rouge-1.f`, `rouge-2.f`, `rouge-l.f`) to match existing repo usage (`swift/plugin/orm.py::cal_rouge`) and the paper-aligned implementation style already used in this fork.

3. Aggregation:
- Paper text states summarization alignment reward is computed from the **average of the three ROUGE metrics**.
- Implemented as:
  - `rouge_avg = (rouge1 + rouge2 + rougeL) / 3`

4. `r_format`:
- Paper summarization section states `r_format = 0` because summarization is not fixed-label output.
- This is documented and reflected in evaluation logic (no label-format constraint used in scoring script).

5. Reported paper values (Table 2):
- PRL (ours): ROUGE-1 `42.47`, ROUGE-2 `16.17`, ROUGE-L `37.73` (± shown in paper table).
- These are reference targets only; this repo computes metrics on current local artifacts.

## 2) `repro/mr_paper` reusable pattern mapping

`repro/mr_paper` structure pattern:

1. Entry evaluator:
- `repro/mr_paper/run_mr_test_eval.py`

2. Shell entry:
- `repro/mr_paper/run_mr_test_eval.sh`

3. Optional multi-seed align driver:
- `repro/mr_paper/run_mr_paper_align.sh`

4. Output convention:
- default output under `repro/mr_paper/results/`
- JSON artifact includes `meta` + final metrics

SUM implementation is aligned to that pattern:

1. Evaluator:
- `repro/sum_paper/run_sum_test_eval.py`

2. Shell entry:
- `repro/sum_paper/run_eval.sh`

3. Scorer:
- `repro/sum_paper/score_predictions.py` (reads predictions JSONL and writes metrics JSON)

4. Output convention:
- default output under `repro/sum_paper/results/sum_test_eval_<timestamp>/`
- artifacts:
  - `predictions.jsonl`
  - `metrics.json`
  - `run_summary.json`

## 3) Sum pipeline “final output” location and field mapping

### 3.1 Current pipeline status (before this alignment)

Observed from run artifacts:

- `output/sum/<run>/v*/completions.jsonl` top-level keys are:
  - `step`, `messages`, `completion`, `reward`
- This file stores **Prompt Generator candidates**, not stable final test summaries for scoring.
- There is no stable, single “final summary predictions file” in current training output.

### 3.2 Stable final output added in `sum_paper`

The aligned evaluation now writes a canonical final output file:

- `repro/sum_paper/results/sum_test_eval_<timestamp>/predictions.jsonl`

Row schema:

```json
{"id": 0, "source": "...", "pred": "...", "gold": "..."}
```

Field mapping:

1. `gold`:
- from dataset row `solution` (fallback: `gold`/`reference`/`summary`)

2. `pred`:
- model generated summary from evaluator inference output text

3. scoring input:
- `score_predictions.py` reads `pred` + `gold`

This file is the explicit **“sum task final output”** used for scoring.

## 4) Running summary

1. From checkpoint directly:

```bash
CHECKPOINT=output/sum/.../checkpoint-1700 \
bash repro/sum_paper/run_eval.sh
```

2. From sum run dir (auto-resolve checkpoint from logging):

```bash
SUM_RUN_DIR=output/sum/sum-qwen-qwen-20260214-104456 \
CHECKPOINT_CHOICE=last \
bash repro/sum_paper/run_eval.sh
```

3. Score an existing predictions file:

```bash
python repro/sum_paper/score_predictions.py \
  --pred-path repro/sum_paper/results/.../predictions.jsonl
```

## 5) Reproduction judgement (2026-02-24, 3 seeds)

Paper-side setting cues (from extracted `base_ref.pdf` sentences):
- 3-run average reporting
- zero-shot setting
- entire response scoring
- ROUGE average aggregation
- dev size = 100 for prompt/development selection

Current run summary (`repro/sum_paper/results/summary_3seeds.json`):
- mean: R1=`42.4382`, R2=`15.5147`, RL=`37.3646`
- std: R1=`0.7670`, R2=`0.8390`, RL=`0.5365`
- paper(Table 2): R1=`42.47`, R2=`16.17`, RL=`37.73`
- delta(mean-paper): R1=`-0.0318`, R2=`-0.6553`, RL=`-0.3654`
- `|delta|/std`: R1=`0.041`, R2=`0.781`, RL=`0.681` (all < 1)

Decision:
- Metric-level reproduction: supported.
- Exact setting-level replication: not fully claimable without more paper-side low-level details.

Missing details that may change strict replication claim:
- exact decode configuration (sampling/stop/length penalties and implementation-level defaults)
- exact checkpoint choice rule and checkpoint step
- exact dataset snapshot/version and normalization details
- exact effective test sample count after filtering
- exact scoring stack versions (tokenizer/ROUGE and preprocessing behavior)

Suggested reporting sentence:
- "We reproduce the SUM result at the metric level under paper-aligned protocol (3-run average, zero-shot, entire-response ROUGE scoring, ROUGE-avg aggregation, dev-size=100), with all deltas within 1 run std; we do not claim strict exact setting-level replication due to missing low-level details in the paper."
