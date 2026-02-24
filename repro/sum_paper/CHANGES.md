# Changes

## 0) Documentation refresh aligned to `repro/mr_paper` (2026-02-24)

What:
- Reorganized `repro/sum_paper/README.md` to mirror MR task doc structure:
  - Evaluation Scope
  - Difference vs Default Repo Training Eval
  - Parameters
  - Outputs
  - Artifact Layout
  - Quick Start
- Added a clear reproduction-status section using current 3-seed summary statistics.
- Added explicit metric-level vs setting-level judgement language.
- Added `|delta|/std` checks for R1/R2/RL from `summary_3seeds.json`.

Why:
- Keep SUM documentation consistent with MR documentation style.
- Make "can we call this reproduced?" auditable and explicit.

Paper alignment reference:
- 3-run average, zero-shot, entire response scoring, ROUGE avg aggregation, dev size=100.

## 1) Added `repro/sum_paper/NOTES.md`

What:
- Added paper-alignment notes for summarization metric definition, aggregation, and repo mapping.

Why:
- To make the scoring definition reproducible and explicitly tied to paper text.

Paper alignment reference:
- Experiments section (`summarization`), ROUGE bullet definitions, Table 2.
- Reward-function wording: summarization uses ROUGE-based alignment reward and sets `r_format=0`.

---

## 2) Added `repro/sum_paper/score_predictions.py`

What:
- New standalone scorer for batch JSONL predictions.
- Input schema: each row must include prediction/reference fields, normalized to `{id,pred,gold}`.
- Computes `rouge1`, `rouge2`, `rougeL` (F1) and `rouge_avg = (r1+r2+rL)/3`.
- Writes `metrics.json`.

Why:
- Needed a stable, reusable, paper-aligned scoring module that can be called independently.

Paper alignment reference:
- Summarization metric: ROUGE-1/2/L.
- Aggregation: average of three ROUGE metrics for summarization reward/scoring.

---

## 3) Added `repro/sum_paper/run_sum_test_eval.py`

What:
- New test-eval driver following `mr_paper` style:
  - resolves checkpoint from either `--checkpoint` or `--sum-run-dir` + `--checkpoint-choice`
  - generates prompt candidates
  - selects best prompt on validation split by `rouge_avg`
  - runs selected prompt on test split
  - writes canonical prediction artifact `predictions.jsonl`
  - calls scoring logic and writes `metrics.json`
  - writes `run_summary.json` with metadata and prompt-selection traces

Why:
- Current training pipeline does not provide a stable final test-summary artifact for scoring.
- This script closes that gap with explicit, deterministic I/O.

Paper alignment reference:
- Prompt Selection on validation, final report on test.
- Summarization metric via ROUGE-1/2/L and average aggregation.

---

## 4) Added `repro/sum_paper/run_eval.sh`

What:
- Shell entry aligned with `mr_paper` invocation style.
- Supports both:
  - `CHECKPOINT=/.../checkpoint-xxxx`
  - `SUM_RUN_DIR=output/sum/sum-qwen-qwen-...` (auto checkpoint resolution from logging artifact)
- Writes outputs under:
  - `repro/sum_paper/results/sum_test_eval_<timestamp>/`

Why:
- Provide a single reproducible entrypoint with explicit environment and output paths.

Paper alignment reference:
- Evaluation path consistency and reproducible reporting.

---

## 5) Added `repro/sum_paper/TREE.md` and `repro/sum_paper/CHANGES.md`

What:
- Added tree and change log artifacts requested for delivery.

Why:
- Make audit/review straightforward.

---

## Final-output location (explicit)

### Current training artifact status
- `output/sum/<run>/v*/completions.jsonl` contains Prompt Generator candidates:
  - keys: `step`, `messages`, `completion`, `reward`
- This is **not** stable final summary prediction output for test scoring.

### Canonical final output introduced by this change
- `repro/sum_paper/results/sum_test_eval_<timestamp>/predictions.jsonl`
- Row fields:
  - `id`: sample id/index
  - `pred`: model generated summary
  - `gold`: ground-truth summary
  - `source`: source input text

Scoring reader:
- `repro/sum_paper/score_predictions.py` reads `pred` and `gold` from that file.

Gold source:
- from dataset row `solution` (fallbacks documented in `run_sum_test_eval.py`).
