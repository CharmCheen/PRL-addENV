# `repro/sum_paper` Tree

```text
repro/sum_paper/
├── CHANGES.md
├── EVAL_ONLY_PATCH.md
├── NOTES.md
├── README.md
├── TREE.md
├── run_3seeds.sh
├── run_eval.sh
├── run_sum_paper_eval.py
├── run_sum_paper_eval.sh
├── run_sum_test_eval.py
├── score_predictions.py
├── results/
│   ├── seed42.log
│   ├── seed43.log
│   ├── seed44.log
│   └── summary_3seeds.json
└── runs/
    ├── paper_align/
    └── verify_smoke/
        └── seed42_limit20.log
```

Notes:
- `__pycache__/` is omitted from the tree view.
- New paper-aligned entry is `run_eval.sh` -> `run_sum_test_eval.py`.
- `run_3seeds.sh` is the default 3-run aggregation wrapper.
