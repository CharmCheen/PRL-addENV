# SUM 评分修复：EVAL_ONLY 模式

## 问题诊断

### 根本原因

1. **swift/cli/main.py:117** - `infer` 命令自动触发 torchrun：
   ```python
   if torchrun_args is None or method_name not in {'pt', 'sft', 'rlhf', 'infer'}:
       args = [python_cmd, file_path, *argv]
   else:
       args = [python_cmd, '-m', 'torch.distributed.run', *torchrun_args, file_path, *argv]
   ```

2. **swift/llm/argument/infer_args.py:153-166** - InferArguments 自动初始化分布式：
   ```python
   def _init_ddp(self):
       if not is_dist():  # 检查 NPROC_PER_NODE 环境变量
           return
       ...
       dist.init_process_group(backend=self.ddp_backend)  # 初始化 NCCL
   ```

3. **历史错误**：
   - 设置 `CUDA_VISIBLE_DEVICES=0,1 NPROC_PER_NODE=2`
   - 但评分脚本是单进程 Python 调用（非 torchrun）
   - InferArguments 检测到 `NPROC_PER_NODE=2`，尝试初始化分布式
   - 单进程无法完成 NCCL collective，导致 timeout

---

## 修复方案

### 修改 1：run_eval.sh

增加 `EVAL_ONLY` 模式，强制单 GPU 评分：

```bash
# 第 10 行：增加 EVAL_ONLY 变量
EVAL_ONLY="${EVAL_ONLY:-0}"  # 1 = skip generation, only score existing predictions

# 第 39-44 行：EVAL_ONLY 模式下强制单 GPU
if [[ "${EVAL_ONLY}" == "1" ]]; then
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
  export NPROC_PER_NODE=1
  export DATALOADER_NUM_WORKERS=0
fi

# 第 90-92 行：传递 --eval-only 参数
if [[ "${EVAL_ONLY}" == "1" ]]; then
  CMD+=(--eval-only)
fi
```

### 修改 2：run_sum_test_eval.py

增加 `--eval-only` 参数，跳过生成直接评分：

```python
# 第 13 行：导入 load_predictions
from score_predictions import compute_rouge_metrics, score_prediction_rows, write_metrics_json, load_predictions

# 第 214 行：增加 --eval-only 参数
parser.add_argument("--eval-only", action="store_true", help="Skip generation, only score existing predictions.")

# 第 217-222 行：禁用分布式环境变量
if args.eval_only:
    os.environ.pop("NPROC_PER_NODE", None)
    os.environ.pop("WORLD_SIZE", None)
    os.environ.pop("LOCAL_RANK", None)
    os.environ.pop("RANK", None)

# 第 227-248 行：EVAL_ONLY 模式逻辑
if args.eval_only:
    print("=== EVAL_ONLY mode: skipping generation, scoring existing predictions ===")
    if not args.pred_path:
        raise ValueError("--eval-only requires --pred-path to be specified")
    pred_path = Path(args.pred_path)
    if not pred_path.exists():
        raise FileNotFoundError(f"Prediction file not found: {pred_path}")

    rows = load_predictions(str(pred_path))
    test_metrics = score_prediction_rows(rows)

    metrics_json = Path(args.metrics_json) if args.metrics_json else pred_path.with_name("metrics.json")
    write_metrics_json(test_metrics, str(metrics_json))

    print("=== SUM Eval-Only Summary ===")
    print(f"pred_path: {pred_path}")
    print(f"num_samples: {test_metrics['num_samples']}")
    print(
        "test_rouge1/2/L/avg: "
        f"{test_metrics['rouge1']:.4f}/{test_metrics['rouge2']:.4f}/"
        f"{test_metrics['rougeL']:.4f}/{test_metrics['rouge_avg']:.4f}"
    )
    print(f"metrics_json: {metrics_json}")
    return
```

---

## 论文对齐

根据 `repro/sum_paper/README.md`：

### 评测协议

- **Prompt 选择**：validation split（100 samples）
- **最终报告**：test split
- **指标**：ROUGE-1 / ROUGE-2 / ROUGE-L（F1）
- **统计**：3 个独立种子（42/43/44），报告 mean ± std

### 指标实现

`score_predictions.py` 使用：
- `rouge` 包（`Rouge().get_scores(..., avg=True)`）
- `mosestokenizer` 英文分词
- 返回百分比值（`*100`）

### 论文基准值

```python
paper = {"rouge1": 42.47, "rouge2": 16.17, "rougeL": 37.73}
```

**当前实现已对齐论文协议，无需修改。**

---

## 使用方法

### 场景 1：已有生成结果，只需评分

```bash
# 找到已有的 predictions.jsonl
PRED_PATH="output/sum/sum-qwen-qwen-20260214-104456/v0-20260214-104536/predictions.jsonl"

# 只评分模式
EVAL_ONLY=1 \
CUDA_VISIBLE_DEVICES=0 \
python repro/sum_paper/run_sum_test_eval.py \
  --eval-only \
  --pred-path "$PRED_PATH" \
  --metrics-json "repro/sum_paper/results/metrics_eval_only.json"
```

### 场景 2：从 SUM_RUN_DIR 重新生成并评分

```bash
# 完整流程（生成 + 评分）
CUDA_VISIBLE_DEVICES=0 \
NPROC_PER_NODE=1 \
SUM_RUN_DIR=output/sum/sum-qwen-qwen-20260214-104456 \
bash repro/sum_paper/run_eval.sh
```

### 场景 3：三种子论文对齐评测

```bash
mkdir -p repro/sum_paper/results/paper_align

for SEED in 42 43 44; do
  CUDA_VISIBLE_DEVICES=0 \
  NPROC_PER_NODE=1 \
  SEED=${SEED} \
  SUM_RUN_DIR=output/sum/sum-qwen-qwen-20260214-104456 \
  bash repro/sum_paper/run_eval.sh \
    2>&1 | tee repro/sum_paper/results/paper_align/seed${SEED}.log
done

# 汇总统计
python - <<'PY'
import json, statistics
from pathlib import Path
out = Path("repro/sum_paper/results/paper_align")
runs = []
for s in [42, 43, 44]:
    d = json.load(open(out / f"seed{s}.json"))
    runs.append({
        "seed": s,
        "rouge1": d["test"]["rouge1"],
        "rouge2": d["test"]["rouge2"],
        "rougeL": d["test"]["rougeL"],
    })
paper = {"rouge1": 42.47, "rouge2": 16.17, "rougeL": 37.73}
mean = {k: statistics.mean([r[k] for r in runs]) for k in ["rouge1","rouge2","rougeL"]}
std = {k: statistics.pstdev([r[k] for r in runs]) for k in ["rouge1","rouge2","rougeL"]}
delta = {k: mean[k] - paper[k] for k in ["rouge1","rouge2","rougeL"]}
print(f"Mean: R1={mean['rouge1']:.2f} R2={mean['rouge2']:.2f} RL={mean['rougeL']:.2f}")
print(f"Std:  R1={std['rouge1']:.2f} R2={std['rouge2']:.2f} RL={std['rougeL']:.2f}")
print(f"Paper: R1={paper['rouge1']:.2f} R2={paper['rouge2']:.2f} RL={paper['rougeL']:.2f}")
print(f"Delta: R1={delta['rouge1']:.2f} R2={delta['rouge2']:.2f} RL={delta['rougeL']:.2f}")
PY
```

---

## 资源配置推荐

### 训练阶段（sum）

```bash
# 2×A100 40GB
CUDA_VISIBLE_DEVICES=0,1
NPROC_PER_NODE=2
DATALOADER_NUM_WORKERS=4
```

### 评分阶段

```bash
# 单 GPU（避免 NCCL hang）
CUDA_VISIBLE_DEVICES=0
NPROC_PER_NODE=1
DATALOADER_NUM_WORKERS=0
```

### CPU 核数

- 训练：`--dataloader_num_workers 4`（2 GPU × 2 workers）
- 评分：`--dataloader_num_workers 0`（单进程，无需多线程）

---

## 验证清单

- [x] 修复 NCCL hang（禁用分布式环境变量）
- [x] 增加 EVAL_ONLY 模式（跳过生成）
- [x] 论文对齐（ROUGE-1/2/L，validation 选 prompt，test 报告）
- [x] 单 GPU 评分（CUDA_VISIBLE_DEVICES=0）
- [x] 最小侵入式修改（仅修改 2 个文件）

---

## Git Diff

完整 diff 见上方输出。

关键修改点：
1. `run_eval.sh` +10 行（EVAL_ONLY 逻辑）
2. `run_sum_test_eval.py` +35 行（--eval-only 参数 + 环境变量清理）

---

## 注意事项

1. **不要在评分阶段设置 NPROC_PER_NODE > 1**
2. **EVAL_ONLY=1 时必须提供 --pred-path**
3. **论文对齐需要 3 个种子（42/43/44）**
4. **评分使用 mosestokenizer + rouge 包，确保依赖已安装**

---

## 故障排查

### 问题：仍然出现 NCCL timeout

**检查**：
```bash
echo $NPROC_PER_NODE  # 应该是 1
echo $CUDA_VISIBLE_DEVICES  # 应该是 0
```

**解决**：
```bash
unset NPROC_PER_NODE
unset WORLD_SIZE
unset LOCAL_RANK
unset RANK
export CUDA_VISIBLE_DEVICES=0
```

### 问题：找不到 predictions.jsonl

**检查**：
```bash
find output/sum -name "predictions.jsonl"
```

**解决**：
使用完整路径或从 logging.jsonl 解析 checkpoint 路径。
