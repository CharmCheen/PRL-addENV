# Repro Entry Points and Artifact Layout

This repo contains completed reproduction scripts for `mr`, `sim`, and `sum`.

## Canonical Script Entrypoints

### sum
- Base train script: `scripts/sum/sum_qwen_qwen.sh`
- Smoke wrapper (10-step validation): `scripts/sum/sum_qwen_qwen_smoke10.sh`
- Full-train wrapper: `scripts/sum/sum_qwen_qwen_train.sh`

### sim
- Base train script: `scripts/sim/sim_qwen_qwen.sh`
- Smoke wrapper (10-step validation): `scripts/sim/sim_qwen_qwen_smoke10_prl_clean.sh`
- Full-train wrapper: `scripts/sim/sim_qwen_qwen_train_prl_clean.sh`

### mr
- Base train script: `scripts/mr/mr_qwen_qwen.sh`
- Multi-GPU variant: `scripts/mr/mr_qwen_qwen_4gpu.sh`

## Output Directory Rules

- `sum` output root (fixed): `/qiuyeqing/llama_prl/PRL-REDO/PRL-addENV/output/sum/`
- `sim` output root: `/qiuyeqing/llama_prl/PRL-REDO/PRL-addENV/output/sim/`
- `mr` output root: `/qiuyeqing/llama_prl/PRL-REDO/PRL-addENV/output/` (script-specific subdirs)

Common run directory pattern:
- `<output_root>/<task>-qwen-qwen-<timestamp>/`
- run log file: `<run_dir>/<task>.log`

## Example Commands

### sum full training (4 GPUs)
```bash
cd /qiuyeqing/llama_prl/PRL-REDO/PRL-addENV
source /qiuyeqing/tools/miniconda3/etc/profile.d/conda.sh
conda activate prl_clean
CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC_PER_NODE=4 MASTER_PORT=29511 USE_LMDEPLOY=0 WANDB_MODE=offline \
  bash scripts/sum/sum_qwen_qwen_train.sh
```

### sim smoke10
```bash
cd /qiuyeqing/llama_prl/PRL-REDO/PRL-addENV
source /qiuyeqing/tools/miniconda3/etc/profile.d/conda.sh
conda activate prl_clean
DATASET=sim CUDA_VISIBLE_DEVICES=0,1 NPROC_PER_NODE=2 SMOKE_TEST=1 SMOKE_STEPS=10 USE_LMDEPLOY=0 \
  bash scripts/sim/sim_qwen_qwen_smoke10_prl_clean.sh
```

## Repo Artifact Policy

- `output/`, `wandb/`, `.cache/` are local runtime artifacts and ignored by git.
- `logs/**/*.log` is ignored to avoid large log churn in commits.
- Root-level `PATCH_*.md` is archived under `docs/repro/patches/`.
- Legacy literal directory `$OUT/` is normalized to `repro/legacy/OUT_literal/`.
