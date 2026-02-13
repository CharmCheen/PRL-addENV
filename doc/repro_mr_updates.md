# MR Repro Updates

## What was added
- Added a test-only MR evaluation delivery under `repro/mr_paper/`.
- Main scripts: `run_mr_test_eval.py`, `run_mr_test_eval.sh`, `run_mr_paper_align.sh`.

## Why
- Align evaluation with paper-style MR reporting on test split.
- Use exact-match accuracy (`pred.strip() == solution.strip()`) on `datasets/original/mr_test.jsonl`.

## How to run (example)
```bash
cd /qiuyeqing/llama_prl/PRL-REDO/PRL-addENV; source /qiuyeqing/tools/miniconda3/etc/profile.d/conda.sh; conda activate prl_clean; CHECKPOINT=output/v31-20260210-185134/checkpoint-2000 BASE_MODEL=/qiuyeqing/.cache/modelscope/models/Qwen/Qwen2___5-7B-Instruct SELECTION_MODE=single NUMBER_OF_PROMPTS=1 bash repro/mr_paper/run_mr_test_eval.sh 2>&1 | tee repro/mr_paper/runs/<timestamp>/run.log
```

## Current result files
- `repro/mr_paper/runs/20260213-063553/mr_test_eval_20260213-054629.json`
- `repro/mr_paper/runs/20260213-063553/full_test2000_single.log`
- `repro/mr_paper/runs/20260213-063553/run_mr_paper_align_3seeds.log`

## Notes
- Use `BASE_MODEL` to avoid online model download in offline environments.
- Runtime cache is placed under workspace `.cache/` by default.
- Large run artifacts are ignored by git (see `.gitignore`).
