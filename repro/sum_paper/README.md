# SUM Paper-Aligned Test Evaluation

This directory provides a paper-aligned evaluation path for summarization (`sum`) using
validation-based prompt selection and test-only reporting.

## Evaluation Scope (Paper Alignment)
- Task: summarization (`sum`)
- Split protocol: select prompt on `datasets/original/sum_val.jsonl` (dev size=100), report on `datasets/original/sum_test.jsonl`
- Metric: ROUGE-1 / ROUGE-2 / ROUGE-L (F1), and `rouge_avg=(R1+R2+RL)/3`
- Statistics: 3 independent runs (`SEED=42/43/44`), report mean and std

## Difference vs Default Repo Training Eval
- This path is an explicit paper-style eval pipeline:
  - prompt generation and selection on validation split
  - final metric reporting on test split
  - canonical output artifacts under `repro/sum_paper/results/`
- It does not change core training scripts.

## Parameters
`run_eval.sh` supports:
- checkpoint source: `CHECKPOINT` or `SUM_RUN_DIR` + `CHECKPOINT_CHOICE={last|best}`
- decoding: `NUMBER_OF_PROMPTS` (default `10`), `TEMPERATURE` (`0.9`), `TOP_P` (`0.9`), `MAX_NEW_TOKENS` (`1024`)
- execution: `BATCH_SIZE` (`8`), `SEED` (`42`), optional `LIMIT`, `FIXED_PROMPT`
- model override: `BASE_MODEL`, `MODEL_TYPE`
- data override: `TRAIN_FILE`, `VAL_FILE`, `TEST_FILE`
- `--eval-only` (Python entrypoint) to skip generation and score existing `predictions.jsonl`

## Outputs
- Per run directory: `repro/sum_paper/results/sum_test_eval_YYYYmmdd-HHMMSS/`
- Artifacts:
  - `run_summary.json`
  - `metrics.json`
  - `predictions.jsonl`
- 3-seed wrapper output:
  - `repro/sum_paper/results/seed42.log`
  - `repro/sum_paper/results/seed43.log`
  - `repro/sum_paper/results/seed44.log`
  - `repro/sum_paper/results/summary_3seeds.json`

## Artifact Layout
- `repro/sum_paper/results/` is the execution output root.
- `repro/sum_paper/runs/paper_align/` is recommended for long-term archived paper-align runs.
- Commit policy: avoid committing large raw run artifacts unless explicitly needed.

## Quick Start
Single run from SUM run dir:
```bash
CUDA_VISIBLE_DEVICES=0 \
NPROC_PER_NODE=1 \
SUM_RUN_DIR=output/sum/sum-qwen-qwen-20260214-104456 \
bash repro/sum_paper/run_eval.sh
```

3-seed run + summary:
```bash
CUDA_VISIBLE_DEVICES=0 \
NPROC_PER_NODE=1 \
bash repro/sum_paper/run_3seeds.sh
```

Eval-only scoring on existing predictions:
```bash
python3 repro/sum_paper/run_sum_test_eval.py \
  --eval-only \
  --pred-path repro/sum_paper/results/sum_test_eval_20260224-112118/predictions.jsonl \
  --metrics-json repro/sum_paper/results/sum_test_eval_20260224-112118/metrics_eval_only.json
```

## Reproduction Status (2026-02-24)
From `repro/sum_paper/results/summary_3seeds.json`:
- mean: R1=`42.4382`, R2=`15.5147`, RL=`37.3646`
- std: R1=`0.7670`, R2=`0.8390`, RL=`0.5365`
- paper(Table 2): R1=`42.47`, R2=`16.17`, RL=`37.73`
- delta(mean-paper): R1=`-0.0318`, R2=`-0.6553`, RL=`-0.3654`
- normalized by run std (`|delta|/std`): R1=`0.041`, R2=`0.781`, RL=`0.681` (<1 for all)

Interpretation:
- Metric-level reproduction: yes (all three deltas within run variance).
- Exact setting-level replication: not fully claimable unless missing low-level setup details are confirmed.
