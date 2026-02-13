# 平台无交互多卡训练脚本说明

## 1. 修改文件清单

**✅ 仅新增文件，无修改现有代码：**
- `scripts/mr/mr_qwen_qwen_4gpu.sh` - 新增多卡训练脚本（192行）

**✅ 单卡脚本完全未修改：**
- `scripts/mr/mr_qwen_qwen.sh` - 保持原样，单卡流程可继续使用

---

## 2. 新脚本完整特性

### 核心设计
- **平台无交互友好**：所有参数通过 `PRL_*` 环境变量配置
- **Fail-fast 验证**：启动前自动检查参数合法性
- **无 device_map 固定**：支持真正的分布式训练
- **保守默认值**：`dataloader_num_workers=0` 避免多进程问题

### 可配置参数（全部支持环境变量覆盖）

| 环境变量 | 默认值 | 说明 |
|---------|--------|------|
| `PRL_CUDA_VISIBLE_DEVICES` | `0,1,2,3` | 可见GPU列表 |
| `PRL_NPROC_PER_NODE` | `4` | 每节点进程数 |
| `PRL_MASTER_PORT` | `29500` | 分布式通信端口 |
| `PRL_PER_DEVICE_TRAIN_BATCH_SIZE` | `4` | 每设备训练batch size |
| `PRL_PER_DEVICE_EVAL_BATCH_SIZE` | `4` | 每设备评估batch size |
| `PRL_GRADIENT_ACCUMULATION_STEPS` | `1` | 梯度累积步数 |
| `PRL_NUM_GENERATIONS` | `2` | 每batch生成数 |
| `PRL_NUM_INFER_WORKERS` | `4` | 推理worker数 |
| `PRL_VLLM_GPU_MEMORY_UTILIZATION` | `0.40` | vLLM显存占用比例 |
| `PRL_DATALOADER_NUM_WORKERS` | `0` | 数据加载worker数（0最稳定） |
| `PRL_DATASET_NUM_PROC` | `8` | 数据集预处理进程数 |
| `PRL_MAX_COMPLETION_LENGTH` | `768` | 最大生成长度 |
| `PRL_EVAL_STEPS` | `200` | 评估间隔 |
| `PRL_SAVE_STEPS` | `200` | 保存间隔 |
| `PRL_LOGGING_STEPS` | `10` | 日志间隔 |

### 自动验证规则
- **必须满足**：`global_batch_size % num_generations == 0`
- **计算公式**：`global_batch_size = per_device_train_batch_size * nproc_per_node`
- **失败行为**：打印详细错误信息并退出（exit 1）

---

## 3. 无交互启动命令

### 2卡训练（本地/测试）

```bash
cd /qiuyeqing/llama_prl/PRL-REDO/PRL-addENV; \
source /qiuyeqing/tools/miniconda3/etc/profile.d/conda.sh; \
conda activate prl_clean; \
export WANDB_MODE=offline; \
PRL_CUDA_VISIBLE_DEVICES=0,1 \
PRL_NPROC_PER_NODE=2 \
PRL_PER_DEVICE_TRAIN_BATCH_SIZE=4 \
PRL_NUM_GENERATIONS=2 \
PRL_NUM_INFER_WORKERS=2 \
bash scripts/mr/mr_qwen_qwen_4gpu.sh
```

**配置说明：**
- 2卡并行，global_batch_size = 4 × 2 = 8
- num_generations = 2，满足 8 % 2 = 0 ✓
- num_infer_workers = 2（与GPU数匹配）

---

### 4卡训练（平台提交）

```bash
cd /qiuyeqing/llama_prl/PRL-REDO/PRL-addENV; \
source /qiuyeqing/tools/miniconda3/etc/profile.d/conda.sh; \
conda activate prl_clean; \
export WANDB_MODE=offline; \
PRL_CUDA_VISIBLE_DEVICES=0,1,2,3 \
PRL_NPROC_PER_NODE=4 \
PRL_PER_DEVICE_TRAIN_BATCH_SIZE=4 \
PRL_NUM_GENERATIONS=2 \
PRL_NUM_INFER_WORKERS=4 \
bash scripts/mr/mr_qwen_qwen_4gpu.sh
```

**配置说明：**
- 4卡并行，global_batch_size = 4 × 4 = 16
- num_generations = 2，满足 16 % 2 = 0 ✓
- num_infer_workers = 4（与GPU数匹配）

---

### 4卡训练（使用默认值，最简命令）

```bash
cd /qiuyeqing/llama_prl/PRL-REDO/PRL-addENV; \
source /qiuyeqing/tools/miniconda3/etc/profile.d/conda.sh; \
conda activate prl_clean; \
export WANDB_MODE=offline; \
bash scripts/mr/mr_qwen_qwen_4gpu.sh
```

**说明：** 脚本默认值已配置为4卡，可直接运行。

---

## 4. 单卡流程保持不变

### ✅ 验证命令
```bash
# 检查单卡脚本是否被修改
git diff scripts/mr/mr_qwen_qwen.sh
# 输出为空 → 确认未修改
```

### ✅ 单卡训练仍可正常运行
```bash
cd /qiuyeqing/llama_prl/PRL-REDO/PRL-addENV; \
source /qiuyeqing/tools/miniconda3/etc/profile.d/conda.sh; \
conda activate prl_clean; \
export WANDB_MODE=offline; \
bash scripts/mr/mr_qwen_qwen.sh
```

**明确声明：**
- ✅ 单卡脚本 `mr_qwen_qwen.sh` **完全未修改**
- ✅ 单卡流程 **保持可跑通**
- ✅ 多卡脚本为 **独立新增**，不影响现有流程

---

## 5. 参数调优建议

### 常见配置组合

| 场景 | GPU数 | batch_size | num_generations | global_batch | 说明 |
|------|-------|------------|-----------------|--------------|------|
| 快速测试 | 2 | 2 | 2 | 4 | 最小配置 |
| 标准2卡 | 2 | 4 | 2 | 8 | 推荐 |
| 标准4卡 | 4 | 4 | 2 | 16 | 默认配置 |
| 高吞吐4卡 | 4 | 4 | 4 | 16 | 更多生成 |
| 大batch4卡 | 4 | 8 | 4 | 32 | 需要更多显存 |

### 显存优化
- **显存不足**：降低 `PRL_VLLM_GPU_MEMORY_UTILIZATION`（0.40 → 0.30）
- **显存充足**：提高 `PRL_PER_DEVICE_TRAIN_BATCH_SIZE`（4 → 8）

### 速度优化
- **加速数据加载**：设置 `PRL_DATALOADER_NUM_WORKERS=4`（需平台支持）
- **减少checkpoint**：提高 `PRL_EVAL_STEPS` 和 `PRL_SAVE_STEPS`

---

## 6. 故障排查

### 验证失败：global_batch_size 不能被 num_generations 整除
```bash
# 错误示例
PRL_NPROC_PER_NODE=4 PRL_PER_DEVICE_TRAIN_BATCH_SIZE=4 PRL_NUM_GENERATIONS=3
# global_batch_size = 16, 16 % 3 = 1 ❌

# 修复方案1：调整 num_generations
PRL_NUM_GENERATIONS=2  # 16 % 2 = 0 ✓

# 修复方案2：调整 batch_size
PRL_PER_DEVICE_TRAIN_BATCH_SIZE=3  # global=12, 12 % 3 = 0 ✓
```

### 多进程问题
如果遇到 dataloader 相关错误，设置：
```bash
PRL_DATALOADER_NUM_WORKERS=0
```

### 端口冲突
如果 29500 端口被占用：
```bash
PRL_MASTER_PORT=29501 bash scripts/mr/mr_qwen_qwen_4gpu.sh
```

---

## 7. 与单卡脚本的差异

| 特性 | 单卡脚本 | 多卡脚本 |
|------|---------|---------|
| 文件名 | `mr_qwen_qwen.sh` | `mr_qwen_qwen_4gpu.sh` |
| 启动方式 | `CUDA_VISIBLE_DEVICES=0 swift rlhf` | `torchrun --nproc_per_node=N -m swift.cli.rlhf` |
| GPU数量 | 1 | 2/4/8（可配置） |
| 参数配置 | 硬编码 | 全部支持环境变量 |
| 参数验证 | 无 | 有（fail-fast） |
| 配置打印 | 无 | 有（详细日志） |
| device_map | 默认 | 无固定（支持分布式） |
| dataloader_workers | 0 | 0（可覆盖为4） |
| num_generations | 4 | 2（可配置） |
| vllm显存比例 | 0.5 | 0.40（可配置） |

---

## 8. 文件路径

- **多卡脚本**：`/qiuyeqing/llama_prl/PRL-REDO/PRL-addENV/scripts/mr/mr_qwen_qwen_4gpu.sh`
- **单卡脚本**：`/qiuyeqing/llama_prl/PRL-REDO/PRL-addENV/scripts/mr/mr_qwen_qwen.sh`（未修改）
- **本文档**：`/qiuyeqing/llama_prl/PRL-REDO/PRL-addENV/docs/multi_gpu_training_guide.md`
