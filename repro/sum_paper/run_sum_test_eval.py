#!/usr/bin/env python3
import argparse
import json
import os
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from score_predictions import compute_rouge_metrics, score_prediction_rows, write_metrics_json, load_predictions
from swift.llm import InferArguments, PtEngine, RequestConfig
from swift.llm.infer.utils import prepare_model_template
from swift.plugin.orm import extract_xml_answer


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def batched(seq: List[Any], n: int):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def safe_git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def load_checkpoint_base_info(checkpoint_dir: str) -> Tuple[str, str]:
    args_path = Path(checkpoint_dir) / "args.json"
    if not args_path.exists():
        raise FileNotFoundError(f"Missing checkpoint args.json: {args_path}")
    with open(args_path, "r", encoding="utf-8") as f:
        ckpt_args = json.load(f)
    model = ckpt_args.get("model")
    model_type = ckpt_args.get("model_type")
    if not model or not model_type:
        raise ValueError(f"checkpoint args.json missing model/model_type: {args_path}")
    return str(model), str(model_type)


def resolve_checkpoint_from_sum_run(sum_run_dir: str, checkpoint_choice: str) -> Tuple[str, str]:
    run_dir = Path(sum_run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"sum run dir not found: {run_dir}")

    version_dirs = sorted([p for p in run_dir.iterdir() if p.is_dir() and p.name.startswith("v")], key=lambda x: x.name)
    if not version_dirs:
        raise FileNotFoundError(f"No v* directory in: {run_dir}")
    version_dir = version_dirs[-1]

    logging_path = version_dir / "logging.jsonl"
    if not logging_path.exists():
        raise FileNotFoundError(f"Missing logging.jsonl: {logging_path}")

    last_obj = None
    with open(logging_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                last_obj = json.loads(line)
    if last_obj is None:
        raise RuntimeError(f"logging.jsonl is empty: {logging_path}")

    if checkpoint_choice == "best":
        ckpt = last_obj.get("best_model_checkpoint")
    else:
        ckpt = last_obj.get("last_model_checkpoint")
    if not ckpt:
        raise RuntimeError(f"Cannot resolve {checkpoint_choice}_model_checkpoint from {logging_path}")
    return os.path.abspath(ckpt), str(version_dir.resolve())


def build_engine(base_model: str, model_type: str, checkpoint_dir: str, max_batch_size: int) -> PtEngine:
    infer_args = InferArguments(
        model=base_model,
        model_type=model_type,
        adapters=[checkpoint_dir],
        infer_backend="pt",
        max_batch_size=max_batch_size,
    )
    model, template = prepare_model_template(infer_args)
    return PtEngine.from_model_template(model, template, max_batch_size=max_batch_size)


def extract_source_and_gold(row: Dict[str, Any]) -> Tuple[str, str]:
    messages = row.get("messages")
    source = ""
    if isinstance(messages, list) and len(messages) > 1 and isinstance(messages[1], dict):
        source = str(messages[1].get("content", ""))
    if not source:
        source = str(row.get("source", row.get("input", "")))

    gold = row.get("solution")
    if gold is None:
        gold = row.get("gold", row.get("reference", row.get("summary", "")))
    return str(source).strip(), str(gold).strip()


def run_model_on_rows(
    engine: PtEngine,
    request_config: RequestConfig,
    rows: List[Dict[str, Any]],
    prompt_text: str,
    batch_size: int,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for batch_start, batch in enumerate(batched(rows, batch_size)):
        requests = []
        batch_ids: List[Any] = []
        batch_golds: List[str] = []
        batch_sources: List[str] = []
        for i, row in enumerate(batch):
            row_id = row.get("id", batch_start * batch_size + i)
            source, gold = extract_source_and_gold(row)
            content = f"{prompt_text}\n{source}" if prompt_text else source
            requests.append({
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": content},
                ]
            })
            batch_ids.append(row_id)
            batch_golds.append(gold)
            batch_sources.append(source)

        responses = engine.infer(requests, request_config, use_tqdm=False)
        for row_id, source, gold, resp in zip(batch_ids, batch_sources, batch_golds, responses):
            pred = resp.choices[0].message.content.strip()
            results.append({
                "id": row_id,
                "source": source,
                "pred": pred,
                "gold": gold,
            })
    return results


def generate_prompt_candidates(
    engine: PtEngine,
    request_config: RequestConfig,
    train_rows: List[Dict[str, Any]],
    number_of_prompts: int,
    fixed_prompt: str,
) -> List[str]:
    if fixed_prompt:
        return [fixed_prompt]

    if not train_rows:
        return [""]
    messages = train_rows[0].get("messages", [])
    if not isinstance(messages, list) or len(messages) < 2:
        return [""]

    reasoning_system = messages[0].get("content", "")
    reasoning_prompt = messages[1].get("content", "")

    prompts: List[str] = []
    for _ in range(number_of_prompts):
        req = {
            "messages": [
                {"role": "system", "content": reasoning_system},
                {"role": "user", "content": reasoning_prompt},
            ]
        }
        resp = engine.infer([req], request_config, use_tqdm=False)[0].choices[0].message.content.strip()
        extracted = extract_xml_answer(resp)
        prompts.append(extracted if extracted else resp)
    return prompts


def write_predictions_jsonl(rows: List[Dict[str, Any]], path: str) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="SUM test-only evaluator aligned with paper metric definition.")
    parser.add_argument("--checkpoint", default="", help="Adapter checkpoint directory.")
    parser.add_argument("--sum-run-dir", default="", help="Path like output/sum/sum-qwen-qwen-YYYY... .")
    parser.add_argument("--checkpoint-choice", choices=["last", "best"], default="last")
    parser.add_argument("--base-model", default="", help="If empty, read from checkpoint args.json.")
    parser.add_argument("--model-type", default="", help="If empty, read from checkpoint args.json.")
    parser.add_argument("--train-file", default="datasets/original/sum_train.jsonl")
    parser.add_argument("--val-file", default="datasets/original/sum_val.jsonl")
    parser.add_argument("--test-file", default="datasets/original/sum_test.jsonl")
    parser.add_argument("--number-of-prompts", type=int, default=int(os.environ.get("NUMBER_OF_PROMPTS", "10")))
    parser.add_argument("--fixed-prompt", default="")
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--pred-path", default="")
    parser.add_argument("--metrics-json", default="")
    parser.add_argument("--output-json", default="")
    parser.add_argument("--eval-only", action="store_true", help="Skip generation, only score existing predictions.")
    args = parser.parse_args()

    # Disable distributed for eval-only mode to avoid NCCL hang
    if args.eval_only:
        os.environ.pop("NPROC_PER_NODE", None)
        os.environ.pop("WORLD_SIZE", None)
        os.environ.pop("LOCAL_RANK", None)
        os.environ.pop("RANK", None)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

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

    checkpoint = args.checkpoint
    resolved_version_dir = ""
    if not checkpoint:
        if not args.sum_run_dir:
            raise ValueError("Either --checkpoint or --sum-run-dir must be provided.")
        checkpoint, resolved_version_dir = resolve_checkpoint_from_sum_run(args.sum_run_dir, args.checkpoint_choice)
    checkpoint = os.path.abspath(checkpoint)

    ckpt_base_model, ckpt_model_type = load_checkpoint_base_info(checkpoint)
    base_model = args.base_model if args.base_model else ckpt_base_model
    model_type = args.model_type if args.model_type else ckpt_model_type

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    if args.output_json:
        output_json = Path(args.output_json)
        out_dir = output_json.parent
    else:
        out_dir = Path("repro/sum_paper/results") / f"sum_test_eval_{timestamp}"
        output_json = out_dir / "run_summary.json"
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_path = Path(args.pred_path) if args.pred_path else out_dir / "predictions.jsonl"
    metrics_json = Path(args.metrics_json) if args.metrics_json else out_dir / "metrics.json"

    train_rows = read_jsonl(args.train_file)
    val_rows = read_jsonl(args.val_file)
    test_rows = read_jsonl(args.test_file)
    if args.limit is not None:
        val_rows = val_rows[:args.limit]
        test_rows = test_rows[:args.limit]

    engine = build_engine(
        base_model=base_model,
        model_type=model_type,
        checkpoint_dir=checkpoint,
        max_batch_size=args.batch_size,
    )
    request_config = RequestConfig(
        max_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
    )

    prompt_candidates = generate_prompt_candidates(
        engine=engine,
        request_config=request_config,
        train_rows=train_rows,
        number_of_prompts=args.number_of_prompts,
        fixed_prompt=args.fixed_prompt,
    )

    per_prompt: List[Dict[str, Any]] = []
    best_prompt_idx = -1
    best_prompt = ""
    best_score = -1.0
    for i, prompt in enumerate(prompt_candidates):
        val_rows_scored = run_model_on_rows(
            engine=engine,
            request_config=request_config,
            rows=val_rows,
            prompt_text=prompt,
            batch_size=args.batch_size,
        )
        val_metrics = score_prediction_rows(val_rows_scored)
        item = {
            "prompt_index": i,
            "prompt": prompt,
            "rouge1": val_metrics["rouge1"],
            "rouge2": val_metrics["rouge2"],
            "rougeL": val_metrics["rougeL"],
            "rouge_avg": val_metrics["rouge_avg"],
        }
        per_prompt.append(item)
        if item["rouge_avg"] > best_score:
            best_score = item["rouge_avg"]
            best_prompt_idx = i
            best_prompt = prompt

    test_rows_scored = run_model_on_rows(
        engine=engine,
        request_config=request_config,
        rows=test_rows,
        prompt_text=best_prompt,
        batch_size=args.batch_size,
    )
    write_predictions_jsonl(test_rows_scored, str(pred_path))

    test_metrics = score_prediction_rows(test_rows_scored)
    write_metrics_json(test_metrics, str(metrics_json))

    summary = {
        "meta": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": safe_git_commit(),
            "task": "sum",
            "metric": "rouge1/rouge2/rougeL_f1",
            "checkpoint": checkpoint,
            "checkpoint_choice": args.checkpoint_choice,
            "resolved_from_sum_run": resolved_version_dir,
            "base_model": base_model,
            "model_type": model_type,
            "train_file": os.path.abspath(args.train_file),
            "val_file": os.path.abspath(args.val_file),
            "test_file": os.path.abspath(args.test_file),
            "number_of_prompts": args.number_of_prompts,
            "fixed_prompt": args.fixed_prompt,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_new_tokens": args.max_new_tokens,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "limit": args.limit,
            "n_val": len(val_rows),
            "n_test": len(test_rows),
        },
        "selection": {
            "best_prompt_index": best_prompt_idx,
            "best_prompt": best_prompt,
            "best_val_rouge_avg": best_score,
            "per_prompt": per_prompt,
        },
        "artifacts": {
            "predictions_jsonl": str(pred_path.resolve()),
            "metrics_json": str(metrics_json.resolve()),
        },
        "test_metrics": test_metrics,
    }
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("=== SUM Test Eval Summary ===")
    print(f"checkpoint: {checkpoint}")
    if resolved_version_dir:
        print(f"resolved_sum_run_version_dir: {resolved_version_dir}")
    print(f"best_prompt_index: {best_prompt_idx}")
    print(f"best_val_rouge_avg: {best_score:.4f}")
    print(
        "test_rouge1/2/L/avg: "
        f"{test_metrics['rouge1']:.4f}/{test_metrics['rouge2']:.4f}/"
        f"{test_metrics['rougeL']:.4f}/{test_metrics['rouge_avg']:.4f}"
    )
    print(f"predictions_jsonl: {pred_path}")
    print(f"metrics_json: {metrics_json}")
    print(f"run_summary_json: {output_json}")


if __name__ == "__main__":
    main()
