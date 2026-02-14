# PATCH: `sim_qwen_qwen` 本地可运行封装（仅新增文件）

## 目标
在不改任何既有 Python 训练/推理逻辑、也不改任何已有 `.sh` 的前提下，补一个可在当前仓库使用的 `sim_qwen_qwen` 启动封装，固定输出到 `output/sim`，并支持单卡/多卡分支。

## 上游对齐说明
- 新增脚本：`scripts/sim/sim_qwen_qwen_prl.sh`
- 对齐对象：上游/既有 `scripts/sim/sim_qwen_qwen.sh` 的调用方式（若存在）
- 兼容回退：若未找到 `scripts/sim/sim_qwen_qwen.sh`，自动尝试 `scripts/mr/mr_qwen_qwen.sh`

由于本次要求是“不运行任何命令、不做试跑”，这里无法直接读取并核验上游脚本全文内容；因此该脚本采用“语义等价 wrapper”策略：
- 复用仓库内既有 qwen/qwen 脚本作为底层入口
- 抽取其首个 `python/python3/swift/torchrun` 启动命令
- 仅做启动器层面的单卡/多卡分流与输出目录固定，不改训练参数语义

## 强约束实现点
- 不触碰任何已有 `.py` / 既有 `.sh`
- Conda 检查严格要求 `CONDA_DEFAULT_ENV=prl_clean`，否则直接 `exit 1`
- 输出基目录固定写死：`output/sim`
- 每次运行创建独立子目录：`output/sim/<exp_name>-<timestamp>`
- 脚本开头打印：
  - `BASE_OUTPUT_DIR=output/sim`
  - `RUN_DIR=...`
  - `LOG_FILE=.../sim.log`
- 日志保存：使用 `2>&1 | tee "$LOG_FILE"`

## 多 GPU 约定
- 可选环境变量：
  - `CUDA_VISIBLE_DEVICES`
  - `NPROC_PER_NODE`（未设默认=可见 GPU 数）
  - `MASTER_PORT`（未设默认 `29500`）
- 启动规则：
  - `NPROC_PER_NODE > 1`：走 `torchrun --nproc_per_node ... --master_port ...`
  - `NPROC_PER_NODE == 1`：直接单进程调用原入口

## 最小可运行示例
```bash
bash scripts/sim/sim_qwen_qwen_prl.sh
```

```bash
CUDA_VISIBLE_DEVICES=0,1 NPROC_PER_NODE=2 MASTER_PORT=29501 \
bash scripts/sim/sim_qwen_qwen_prl.sh
```

若要显式指定底层脚本：
```bash
UNDERLYING_SCRIPT=scripts/sim/sim_qwen_qwen.sh \
bash scripts/sim/sim_qwen_qwen_prl.sh
```

## 已知坑与规避（仅文档建议，不改 Python）
- `ParallelismConfig` 相关报错：
  - 先用单卡验证：`NPROC_PER_NODE=1`
  - 再逐步放大卡数；必要时沿用既有脚本中稳定的并行参数组合
- `lmdeploy` 依赖/导入问题：
  - 若任务路径不需要部署推理，优先使用训练/评测路径的既有参数，避免触发 lmdeploy 代码分支
  - 若确需该分支，请在 `prl_clean` 环境中补齐其已知依赖版本后再运行
- `trl` logging/wandb 相关噪声或初始化失败：
  - 可通过环境变量减少外部日志依赖，例如 `WANDB_MODE=offline` 或 `WANDB_DISABLED=true`
  - 保持与既有脚本一致的日志参数，先保证主流程可跑通

## 变更清单
- 新增：`scripts/sim/sim_qwen_qwen_prl.sh`
- 新增：`PATCH_sim_qwen_qwen.md`
- 其余文件未改动

