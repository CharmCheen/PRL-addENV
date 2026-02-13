# MR 复现对齐分析报告（仅基于现有结果，不重训）

## 1) 论文协议摘要

### 1.1 来自论文 PDF 的可核验证据（短摘录）
- 指标（classification）：`"... used in all classification tasks is accuracy."`  
  证据：`docs/paper/base_ref.pdf`（提取证据文件：`docs/evidence/base_ref_protocol_snippets.txt`，Snippet A）
- `ntest` 设定：`"... number of test prompts to n_test = 10."`  
  证据：`docs/paper/base_ref.pdf` / `docs/evidence/base_ref_protocol_snippets.txt`（Snippet A）
- `ntest` 含义：`"n_test: number of prompts during training/Prompt Selection"`  
  证据：`docs/paper/base_ref.pdf` / `docs/evidence/base_ref_protocol_snippets.txt`（Snippet D）
- MR 上对比口径：`"... varying ... n_test = 1, 5, 10, and 15."`  
  证据：`docs/paper/base_ref.pdf` / `docs/evidence/base_ref_protocol_snippets.txt`（Snippet B）
- 多次运行统计：`"Each configuration is run three times ... average accuracy."` 与 `"... averaged over three runs."`  
  证据：`docs/paper/base_ref.pdf` / `docs/evidence/base_ref_protocol_snippets.txt`（Snippet B/C）

### 1.2 协议最小必要参数集合（用于 MR 对齐）
- `split`：MR 测试集（test split）。
- `ntest`：候选 prompt 数（至少要覆盖 `1/5/10/15` 对比；主报告推荐 `10`）。
- `seeds`：每个配置 3 次运行（例如 42/43/44）。
- `metric`：accuracy（classification）。
- `decode params`：论文摘录中未检出显式 `temperature/top_p` 强制值；需在复现报告中固定并披露。

注：当前 PDF 可提取文本中未发现“exact-match”字样；仓库 test-only 实现将 accuracy 具体化为 `pred.strip()==solution.strip()`。

## 2) 当前结果摘要（0.879 的口径）

目标结果文件：`repro/mr_paper/runs/20260213-063553/mr_test_eval_20260213-054629.json`。  
对应日志：`repro/mr_paper/runs/20260213-063553/full_test2000_single.log`。

从结果 JSON 读取到：
- `accuracy=0.879`（`1758/2000`）
- `selection_mode=single`
- `number_of_prompts=1`
- `test_file=datasets/original/mr_test.jsonl`
- `temperature=0.0`, `top_p=1.0`, `max_new_tokens=16`
- `metric=exact_match_accuracy`
- `definition=pred.strip() == solution.strip()`
- `seed=42`

从日志读取到同口径命令：
- `--selection-mode single --number-of-prompts 1 --temperature 0.0 --top-p 1.0 --max-new-tokens 16 --seed 42`
- 输出 `best_accuracy: 0.879000 (1758/2000)`

补充数据规模证据：
- `datasets/original/mr_test.jsonl` 为 2000 条。
- `datasets/original/mr_val.jsonl` 为 200 条。

## 3) 对齐检查表（参数逐项）

| 项目 | 论文协议 | 当前 0.879 口径 | 对齐 |
|---|---|---|---|
| split | MR test split | `mr_test.jsonl` (2000) | ✅ |
| metric 名称 | accuracy | `exact_match_accuracy` | ⚠️（语义接近，但命名不同） |
| metric 定义 | 分类正确率 | 严格字符串匹配 `pred==solution` | ✅（可视作 accuracy 实现） |
| ntest | 主设定 `n_test=10`；并比较 `1/5/10/15` | `number_of_prompts=1` | ❌ |
| 运行次数 | 每配置 3 次取平均 | 单次（seed=42） | ❌ |
| decode 参数 | 论文摘录未给硬性值（需固定披露） | `temp=0.0 top_p=1.0 max_new_tokens=16` | ⚠️（已披露，但与训练/仓库 paper-align 默认不同） |
| 选择方式 | Prompt Selection（best-of-n） | `single` 非 best-of | ❌ |

## 4) 仓库默认评测路径对照（必须项）

- 训练脚本默认验证集路径：
  - `scripts/mr/mr_qwen_qwen.sh` 使用 `--val_dataset datasets/original/mr_val.jsonl`（200 条）。
  - `scripts/mr/mr_qwen_qwen_4gpu.sh` 同样是 `mr_val.jsonl`。
- test-only 路径：
  - `repro/mr_paper/run_mr_test_eval.py` 默认 `--test-file datasets/original/mr_test.jsonl`（2000 条）。
  - 因此 test-only 与训练内 eval 的数据口径不同（`val200` vs `test2000`）。
- `grpo_trainer.evaluation_loop` 是否 best-of：
  - `swift/trainers/rlhf_trainer/grpo_trainer.py` 中循环 `for _ in range(int(os.environ["NUMBER_OF_PROMPTS"]))` 采样多个 prompt，随后以准确率更新 `best_accuracy`（adversarial=0 时取最大）。
  - 即训练内 eval 是 best-of-`NUMBER_OF_PROMPTS`。
- 当前 0.879 是否等价该逻辑：
  - 不等价。当前是 `single + number_of_prompts=1`，不是 best-of-10。

## 5) 结论：是否完成论文复现

结论：**未完成严格论文协议复现**。

主要差距（按影响从大到小）：
1. **`ntest` 不符**：论文主设定建议 `n_test=10`，当前为 1。
2. **缺少 3 次运行均值/方差**：当前仅 seed=42 单次。
3. **选择机制不符**：论文/训练内逻辑是 prompt selection(best-of-n)，当前是 single。
4. **decode 参数口径未与论文主结果严格对齐声明**：当前 `temp=0, top_p=1, max_new_tokens=16`，与仓库 `paper-align` 默认（`0.9/0.9/384`）不同。

## 6) 下一步一键命令（严格论文协议对齐）

以下都为单行分号命令流，输出集中到 `repro/mr_paper/runs/paper_align/`。

### 6.1 best-of-10 单次（seed=42）
```bash
mkdir -p repro/mr_paper/runs/paper_align; CHECKPOINT=output/v31-20260210-185134/checkpoint-2000; OUT_DIR=repro/mr_paper/runs/paper_align; SEED=42; python repro/mr_paper/run_mr_test_eval.py --checkpoint "$CHECKPOINT" --selection-mode best --number-of-prompts 10 --test-file datasets/original/mr_test.jsonl --train-file datasets/original/mr_train.jsonl --temperature 0.9 --top-p 0.9 --max-new-tokens 384 --batch-size 8 --seed "$SEED" --output-json "$OUT_DIR/paper_align_seed${SEED}.json" 2>&1 | tee "$OUT_DIR/paper_align_seed${SEED}.log"
```

### 6.2 best-of-10 三次 seeds（42/43/44）并汇总 mean±std
```bash
mkdir -p repro/mr_paper/runs/paper_align; CHECKPOINT=output/v31-20260210-185134/checkpoint-2000; OUT_DIR=repro/mr_paper/runs/paper_align; for SEED in 42 43 44; do python repro/mr_paper/run_mr_test_eval.py --checkpoint "$CHECKPOINT" --selection-mode best --number-of-prompts 10 --test-file datasets/original/mr_test.jsonl --train-file datasets/original/mr_train.jsonl --temperature 0.9 --top-p 0.9 --max-new-tokens 384 --batch-size 8 --seed "$SEED" --output-json "$OUT_DIR/paper_align_seed${SEED}.json" 2>&1 | tee "$OUT_DIR/paper_align_seed${SEED}.log"; done; python - <<'PY'
import json, statistics, pathlib, datetime
out_dir = pathlib.Path('repro/mr_paper/runs/paper_align')
seeds = [42,43,44]
runs = []
for s in seeds:
    p = out_dir / f'paper_align_seed{s}.json'
    d = json.loads(p.read_text())
    runs.append({'seed': s, 'accuracy': d['best']['accuracy'], 'correct': d['best']['correct'], 'total': d['best']['total'], 'result_json': str(p), 'log_file': str(out_dir / f'paper_align_seed{s}.log')})
accs = [r['accuracy'] for r in runs]
summary = {'timestamp_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'n_runs': len(runs), 'runs': runs, 'mean_accuracy': statistics.mean(accs), 'std_accuracy': statistics.stdev(accs) if len(accs) > 1 else 0.0}
(out_dir / 'paper_align_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY
```
