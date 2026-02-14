# Patch: `sim_qwen_qwen_smoke10.sh`

## 1) 失败根因分析

根因是“嵌套 launcher（外层 `torchrun` + 内层 `torchrun`）”。

- 外层命令是：
  - `torchrun --nproc_per_node 4 --master_port 29501 ... swift rlhf ...`
- 但 `swift rlhf` 内部又触发了：
  - `python -m torch.distributed.run --nproc_per_node 4 --master_port 29501 ... swift/cli/rlhf.py ...`

这样同一作业内出现两层分布式启动器，且端口复用冲突，导致：

- `TCPStore ... failed to bind 29501`（端口已被另一层占用）
- 后续 worker 无法正确 rendezvous，触发 `ncclRemoteError` / `socketPollConnect ... Connection refused`

此外日志出现 `datasets/original/_train.jsonl`，说明 `DATASET` 为空，数据路径被拼坏，属于第二故障点。

## 2) 修复策略（仅脚本层）

新增脚本：`scripts/sim/sim_qwen_qwen_smoke10.sh`

修复点：

1. 去掉外层 `torchrun`，只保留一层 launcher：
   - 直接调用 `swift rlhf ...`
   - 通过环境变量 `NPROC_PER_NODE` 让 `swift` 内部按多卡启动（不再外包一层 `torchrun`）
2. 固定/增强分布式环境变量（不改 Python）：
   - `MASTER_ADDR=127.0.0.1`
   - `MASTER_PORT=${MASTER_PORT:-29501}`
   - `TORCHELASTIC_USE_AGENT_STORE=0`
   - `NCCL_SOCKET_IFNAME` 自动策略：
     - 有 `eth0` 时默认 `eth0`
     - 否则默认 `^lo,docker0`
     - 可由用户显式覆盖
3. 健壮性检查：
   - `CONDA_DEFAULT_ENV` 必须是 `prl_clean`，否则失败退出
   - `DATASET` 必须非空，否则失败退出并给用法
4. 输出与日志规范：
   - `BASE_OUTPUT_DIR=output/sim`
   - 每次新建 `RUN_DIR=output/sim/sim-qwen-qwen-<timestamp>`
   - 日志固定 `LOG_FILE=$RUN_DIR/sim.log`
   - 训练执行固定 `2>&1 | tee "$LOG_FILE"`
   - 启动前打印：
     - `CUDA_VISIBLE_DEVICES`
     - `NPROC_PER_NODE`
     - `MASTER_ADDR/MASTER_PORT`
     - `RUN_DIR/LOG_FILE`
     - `FINAL_CMD`
5. 默认 smoke test（10 steps）：
   - 默认 `SMOKE_TEST=1`，自动追加 `--max_steps 10`
   - 可通过 `SMOKE_TEST=0` 关闭

## 3) 使用示例

### 3.1 默认 smoke10（推荐）

```bash
DATASET=sim \
CUDA_VISIBLE_DEVICES=0,1 \
NPROC_PER_NODE=2 \
bash scripts/sim/sim_qwen_qwen_smoke10.sh
```

### 3.2 指定端口

```bash
DATASET=sim \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NPROC_PER_NODE=4 \
MASTER_PORT=29511 \
bash scripts/sim/sim_qwen_qwen_smoke10.sh
```

### 3.3 覆盖网卡选择（可选）

```bash
DATASET=sim \
CUDA_VISIBLE_DEVICES=0,1 \
NPROC_PER_NODE=2 \
NCCL_SOCKET_IFNAME=eth0 \
bash scripts/sim/sim_qwen_qwen_smoke10.sh
```

### 3.4 关闭 smoke10（跑常规训练）

```bash
DATASET=sim \
CUDA_VISIBLE_DEVICES=0,1 \
NPROC_PER_NODE=2 \
SMOKE_TEST=0 \
bash scripts/sim/sim_qwen_qwen_smoke10.sh
```

