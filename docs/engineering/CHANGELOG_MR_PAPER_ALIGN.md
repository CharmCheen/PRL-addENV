# CHANGELOG: MR Paper-Align Module

## 背景
- 原仓库训练入口以 `val_dataset` 为主（`mr_val.jsonl`），不直接对应论文 test-only 报告口径。
- 为了可重复的论文对齐评测，增加了 `repro/mr_paper/` 下的 test-only / paper-align 评测链路。

## 本次整理做了什么
- 文档目录合并：`doc/` 内容迁移到 `docs/` 分类目录。
- 新文档结构：
  - `docs/paper/` 论文与参考材料
  - `docs/repro/` 复现分析报告
  - `docs/evidence/` 论文协议提取证据
  - `docs/engineering/` 工程说明与提交记录
- 3-seed paper-align 产物归档统一到：
  - `repro/mr_paper/runs/paper_align/seed{42,43,44}.{log,json}`
  - `repro/mr_paper/runs/paper_align/summary.{json,md}`
- `repro/mr_paper/README.md` 增加“产物目录规范”章节。

## 产物路径规范
- 最终归档目录：`repro/mr_paper/runs/paper_align/`
- 临时目录：`repro/mr_paper/results/`（允许保留，非最终引用）
- 训练入口不变：`scripts/mr/mr_qwen_qwen*.sh` 无修改。

## 标准复现命令（best-of-10 + 3 seeds）
```bash
mkdir -p repro/mr_paper/runs/paper_align; CHECKPOINT=output/v31-20260210-185134/checkpoint-2000; for SEED in 42 43 44; do python repro/mr_paper/run_mr_test_eval.py --checkpoint "$CHECKPOINT" --selection-mode best --number-of-prompts 10 --test-file datasets/original/mr_test.jsonl --train-file datasets/original/mr_train.jsonl --temperature 0.9 --top-p 0.9 --max-new-tokens 384 --batch-size 8 --seed "$SEED" --output-json "repro/mr_paper/runs/paper_align/seed${SEED}.json" 2>&1 | tee "repro/mr_paper/runs/paper_align/seed${SEED}.log"; done
```
