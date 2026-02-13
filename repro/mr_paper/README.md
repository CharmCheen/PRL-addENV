# MR Paper-Aligned Test-Only Evaluation

This directory provides a **test-only** evaluation path for Movie Review (MR) that aligns with paper-style reporting on the MR **test split**.

## Evaluation Scope (Paper Alignment)
- Metric: **MR test accuracy (exact match)**
- Definition: `pred.strip() == solution.strip()`
- Dataset: `datasets/original/mr_test.jsonl` (2000 samples)
- Checkpoint: user-provided adapter checkpoint (e.g. `output/.../checkpoint-XXXX`)

## Difference vs Default Repo Training Eval
- Default training script path (`scripts/mr/mr_qwen_qwen.sh`) uses:
  - `--val_dataset datasets/original/mr_val.jsonl` (200 samples)
  - trainer `evaluation_loop` that performs best-of prompt selection over `NUMBER_OF_PROMPTS`
- This delivery switches evaluation data to:
  - **`datasets/original/mr_test.jsonl` (2000 samples)**
- Prompt selection policy in this delivery:
  - `SELECTION_MODE=single` (default): single fixed prompt (empty prefix)
  - `SELECTION_MODE=best`: generate `NUMBER_OF_PROMPTS` prompt candidates and report best test accuracy among them

Notes:
- `best` mode is useful for reproducing repo-style best-of behavior.
- `single` mode avoids test-label-driven prompt selection leakage and is often preferred for strict test reporting.

## Parameters
The shell entry (`run_mr_test_eval.sh`) supports:
- `CHECKPOINT` (required): adapter checkpoint dir
- `NUMBER_OF_PROMPTS` (default `1`)
- `SELECTION_MODE` (`single` or `best`, default `single`)
- `LIMIT` (optional): evaluate first N samples for smoke test
- `MAX_NEW_TOKENS` (default `16`)
- `TEMPERATURE` (default `0.0`)
- `TOP_P` (default `1.0`)
- `BATCH_SIZE` (default `8`)
- `SEED` (default `42`)
- `BASE_MODEL` (optional): local base model path for offline eval
- `MODEL_TYPE` (optional): override model type if needed

These decoding parameters affect generated predictions and thus final accuracy.

## Outputs
Results are saved to:
- JSON: `repro/mr_paper/results/mr_test_eval_YYYYmmdd-HHMMSS.json`
- Recommended log file via `tee` (example command below)

JSON includes:
- metadata (`git_commit`, timestamp, checkpoint, dataset, prompt policy, decoding params)
- per-prompt accuracy list
- best prompt index and final accuracy

## Artifact Layout
- Source files stay under `repro/mr_paper/` (scripts + this README).
- `results/` is temporary staging for ad-hoc eval outputs.
- `runs/` is long-term archive for reproducible runs.
- Paper-aligned 3-seed outputs are centralized under `runs/paper_align/`.
- Recommended commit policy: do not commit large run artifacts (`.log`, `.json`).

## Quick Start

### 1) Smoke test (LIMIT=20)
```bash
CHECKPOINT=output/v31-20260210-185134/checkpoint-2000 \
BASE_MODEL=/qiuyeqing/.cache/modelscope/models/Qwen/Qwen2___5-7B-Instruct \
LIMIT=20 \
SELECTION_MODE=single \
NUMBER_OF_PROMPTS=1 \
bash repro/mr_paper/run_mr_test_eval.sh
```

### 2) Full test (2000 samples)
```bash
CHECKPOINT=output/v31-20260210-185134/checkpoint-2000 \
BASE_MODEL=/qiuyeqing/.cache/modelscope/models/Qwen/Qwen2___5-7B-Instruct \
SELECTION_MODE=single \
NUMBER_OF_PROMPTS=1 \
bash repro/mr_paper/run_mr_test_eval.sh
```

### 3) Repo-style best-of prompt selection on test
```bash
CHECKPOINT=output/v31-20260210-185134/checkpoint-2000 \
BASE_MODEL=/qiuyeqing/.cache/modelscope/models/Qwen/Qwen2___5-7B-Instruct \
SELECTION_MODE=best \
NUMBER_OF_PROMPTS=10 \
TEMPERATURE=0.9 \
bash repro/mr_paper/run_mr_test_eval.sh
```

## Multi-run Template
Use this to run 3 independent seeds:
```bash
for s in 42 43 44; do
  CHECKPOINT=output/v31-20260210-185134/checkpoint-2000 \
  SELECTION_MODE=single NUMBER_OF_PROMPTS=1 SEED=$s \
  bash repro/mr_paper/run_mr_test_eval.sh
done
```

## Paper-Align 3-Run Command
`run_mr_paper_align.sh` runs 3 seeds and writes aggregate mean/std.

Defaults:
- `SELECTION_MODE=best`
- `NUMBER_OF_PROMPTS=10`
- `TEMPERATURE=0.9`
- `TOP_P=0.9`
- `MAX_NEW_TOKENS=384` (to align with checkpoint `max_completion_length=384`; can override to `1024` for original script setting)

Run:
```bash
CHECKPOINT=output/v31-20260210-185134/checkpoint-2000 \
BASE_MODEL=/qiuyeqing/.cache/modelscope/models/Qwen/Qwen2___5-7B-Instruct \
bash repro/mr_paper/run_mr_paper_align.sh
```

Output:
- script默认输出：`repro/mr_paper/results/*`
- 规范归档：`repro/mr_paper/runs/paper_align/`（见下节）

## 产物目录规范
- 固定目录：`repro/mr_paper/runs/paper_align/`
- 归档文件命名：
  - `seed42.log`, `seed42.json`
  - `seed43.log`, `seed43.json`
  - `seed44.log`, `seed44.json`
  - `summary.json`, `summary.md`
- `repro/mr_paper/results/` 只用于临时中间产物，不作为最终归档。
- 复现实验提交时，优先引用 `runs/paper_align/summary.*`。
