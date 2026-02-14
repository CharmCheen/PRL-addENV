# Patch: `sim_qwen_qwen_smoke10_prl_clean.sh`

## 根因分析

失败根因是嵌套 `torchrun`：

1. 外层命令使用了 `torchrun ... swift rlhf ...`。  
2. `swift rlhf` 内部又会触发 `python -m torch.distributed.run ... swift/cli/rlhf.py ...`。  
3. 两层 launcher 共用同一组分布式端口/会话，触发 `TCPStore` 端口抢占与组网错配。  
4. 典型报错就是：
   - `TCPStore ... failed to bind 29501`
   - `ncclRemoteError ... socketPollConnect ... Connection refused`

另外日志中的 `datasets/original/_train.jsonl` 显示 `DATASET` 为空，属于并发失败点（数据路径被拼坏）。

## 修复策略（仅脚本层）

新增脚本：`scripts/sim/sim_qwen_qwen_smoke10_prl_clean.sh`

1. 仅保留一层 launcher：
   - 不再外层 `torchrun`
   - 直接 `swift rlhf ...`（或 fallback `python -m swift rlhf ...`）
2. 强制 conda 非交互激活：
   - `source /qiuyeqing/tools/miniconda3/etc/profile.d/conda.sh`
   - `conda activate prl_clean`
   - 打印 `which python` / `python -V` / `CONDA_DEFAULT_ENV`
   - 若环境不是 `prl_clean`，立即退出
3. 分布式稳健变量（仅环境变量）：
   - `NPROC_PER_NODE` 默认=可见 GPU 数
   - `MASTER_ADDR=127.0.0.1`
   - `MASTER_PORT=${MASTER_PORT:-29501}`
   - `TORCHELASTIC_USE_AGENT_STORE=0`
   - `NCCL_SOCKET_IFNAME`：优先 `eth0`，否则 `^lo,docker0`
   - `NCCL_ASYNC_ERROR_HANDLING=1`
   - `NCCL_DEBUG=${NCCL_DEBUG:-WARN}`
4. 输出与日志：
   - `RUN_DIR=output/sim/<timestamp>`
   - `LOG_FILE=$RUN_DIR/sim.log`
   - 执行命令固定 `2>&1 | tee "$LOG_FILE"`
5. smoke test 默认开启（10 steps）：
   - 默认追加 `--max_steps 10 --eval_steps 1000000 --save_steps 1000000`
   - 默认 `SMOKE_TEST=1`

## 如何判断“>=10 steps 成功”

脚本在训练命令返回后执行日志判定：

1. 读取 `sim.log`，排除参数回显行（`FINAL_CMD`、`max_steps`、`eval_steps`、`save_steps`、`logging_steps`）。  
2. 用正则提取运行时 step 相关片段（`global_step` 或 `step` + 数字）。  
3. 计算最大 step 值 `MAX_STEP`。  
4. 若 `MAX_STEP >= 10`，打印：
   - `[OK] Smoke test reached step>=10...`
   否则退出失败并提示日志路径。

## 使用示例

```bash
DATASET=sim \
CUDA_VISIBLE_DEVICES=0,1 \
NPROC_PER_NODE=2 \
bash scripts/sim/sim_qwen_qwen_smoke10_prl_clean.sh
```

可选关闭 smoke：

```bash
DATASET=sim \
SMOKE_TEST=0 \
bash scripts/sim/sim_qwen_qwen_smoke10_prl_clean.sh
```

