#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from mosestokenizer import MosesTokenizer
from rouge import Rouge


def _clean_text(text: str) -> str:
    # MosesTokenizer asserts that inputs contain no newlines.
    if "\n" in text or "\r" in text:
        text = text.replace("\r", " ").replace("\n", " ")
    return text.strip()


def _to_text(value: Any) -> str:
    if value is None:
        return ""
    return _clean_text(str(value))


def _pick_first(row: Dict[str, Any], keys: List[str]) -> Any:
    for key in keys:
        if key in row:
            return row[key]
    return None


def load_predictions(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            pred = _to_text(_pick_first(data, ["pred", "prediction", "output", "completion"]))
            gold = _to_text(_pick_first(data, ["gold", "reference", "solution", "label"]))
            row_id = _pick_first(data, ["id", "row_id", "index"])
            if row_id is None:
                row_id = idx
            rows.append({"id": row_id, "pred": pred, "gold": gold})
    return rows


def compute_rouge_metrics(preds: List[str], golds: List[str]) -> Dict[str, float]:
    if not preds:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0, "rouge_avg": 0.0}

    with MosesTokenizer("en") as tokenize:
        preds_tok = [" ".join(tokenize(p)) for p in preds]
        golds_tok = [" ".join(tokenize(g)) for g in golds]

    scores = Rouge().get_scores(preds_tok, golds_tok, avg=True)
    rouge1 = float(scores["rouge-1"]["f"] * 100.0)
    rouge2 = float(scores["rouge-2"]["f"] * 100.0)
    rougeL = float(scores["rouge-l"]["f"] * 100.0)
    rouge_avg = (rouge1 + rouge2 + rougeL) / 3.0
    return {
        "rouge1": rouge1,
        "rouge2": rouge2,
        "rougeL": rougeL,
        "rouge_avg": rouge_avg,
    }


def score_prediction_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    preds = [_to_text(r.get("pred")) for r in rows]
    golds = [_to_text(r.get("gold")) for r in rows]
    metrics = compute_rouge_metrics(preds, golds)
    metrics.update({
        "metric": "rouge_f1",
        "num_samples": len(rows),
    })
    return metrics


def write_metrics_json(metrics: Dict[str, Any], output_json: str) -> None:
    out = Path(output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Score summarization predictions with ROUGE-1/2/L (F1).")
    parser.add_argument("--pred-path", required=True, help="JSONL path. Each row should contain pred + gold fields.")
    parser.add_argument("--output-json", default="", help="Output metrics JSON path.")
    args = parser.parse_args()

    rows = load_predictions(args.pred_path)
    metrics = score_prediction_rows(rows)

    if args.output_json:
        output_json = args.output_json
    else:
        output_json = str(Path(args.pred_path).with_name("metrics.json"))
    write_metrics_json(metrics, output_json)

    print("=== Sum Score Summary ===")
    print(f"pred_path: {args.pred_path}")
    print(f"num_samples: {metrics['num_samples']}")
    print(f"rouge1: {metrics['rouge1']:.4f}")
    print(f"rouge2: {metrics['rouge2']:.4f}")
    print(f"rougeL: {metrics['rougeL']:.4f}")
    print(f"rouge_avg: {metrics['rouge_avg']:.4f}")
    print(f"output_json: {output_json}")


if __name__ == "__main__":
    main()
