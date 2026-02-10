# 多卡训练快速参考

## ✅ 单卡脚本未修改，单卡流程保持可跑通

---

## 📋 修改文件清单

**仅新增文件：**
- ✅ `scripts/mr/mr_qwen_qwen_4gpu.sh` - 多卡训练脚本（192行）
- ✅ `docs/multi_gpu_training_guide.md` - 完整文档
- ✅ `docs/multi_gpu_quick_reference.md` - 本文件

**未修改文件：**
- ✅ `scripts/mr/mr_qwen_qwen.sh` - 单卡脚本保持原样

---

## 🚀 无交互启动命令

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

**配置：** 2卡，global_batch=8，满足 8%2=0 ✓

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

**配置：** 4卡，global_batch=16，满足 16%2=0 ✓

---

### 4卡训练（最简命令，使用默认值）

```bash
cd /qiuyeqing/llama_prl/PRL-REDO/PRL-addENV; \
source /qiuyeqing/tools/miniconda3/etc/profile.d/conda.sh; \
conda activate prl_clean; \
export WANDB_MODE=offline; \
bash scripts/mr/mr_qwen_qwen_4gpu.sh
```

**说明：** 脚本默认值已配置为4卡最优参数

---

## 🔧 关键环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `PRL_CUDA_VISIBLE_DEVICES` | `0,1,2,3` | 可见GPU |
| `PRL_NPROC_PER_NODE` | `4` | 进程数 |
| `PRL_PER_DEVICE_TRAIN_BATCH_SIZE` | `4` | 每卡batch |
| `PRL_NUM_GENERATIONS` | `2` | 生成数 |
| `PRL_NUM_INFER_WORKERS` | `4` | 推理worker |
| `PRL_VLLM_GPU_MEMORY_UTILIZATION` | `0.40` | vLLM显存 |
| `PRL_DATALOADER_NUM_WORKERS` | `0` | 数据worker |

**完整参数列表：** 见 `docs/multi_gpu_training_guide.md`

---

## ⚠️ 验证规则

**必须满足：** `global_batch_size % num_generations == 0`

**计算公式：** `global_batch_size = per_device_train_batch_size × nproc_per_node`

**有效组合示例：**
- 4卡：batch=4, gen=2 → global=16 ✓
- 4卡：batch=4, gen=4 → global=16 ✓
- 2卡：batch=4, gen=2 → global=8 ✓
- 4卡：batch=4, gen=3 → global=16 ❌ (16%3=1)

---

## 📖 详细文档

- **完整指南：** `docs/multi_gpu_training_guide.md`
- **今日日志：** `docs/daily_log_2026-02-10.md`

---

## ✅ 明确声明

1. ✅ 单卡脚本 `scripts/mr/mr_qwen_qwen.sh` **完全未修改**
2. ✅ 单卡流程 **保持可跑通**
3. ✅ 多卡脚本为 **独立新增**，不影响现有流程
4. ✅ 所有参数 **可通过环境变量调整**，适合平台无交互提交
5. ✅ 启动前 **自动验证参数**，fail-fast 设计
