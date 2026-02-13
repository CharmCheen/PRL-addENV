#!/usr/bin/env python3
import argparse
import json
import os
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

from swift.llm import InferArguments, PtEngine, RequestConfig
from swift.llm.infer.utils import prepare_model_template
from swift.plugin.orm import extract_xml_answer


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def batched(seq: List[Any], n: int):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def safe_git_commit() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        return out
    except Exception:
        return "unknown"


def load_checkpoint_base_info(checkpoint_dir: str) -> Tuple[str, str]:
    args_path = Path(checkpoint_dir) / "args.json"
    if not args_path.exists():
        raise FileNotFoundError(f"Missing args.json in checkpoint: {args_path}")
    with open(args_path, "r", encoding="utf-8") as f:
        ckpt_args = json.load(f)
    model = ckpt_args.get("model")
    model_type = ckpt_args.get("model_type")
    if not model or not model_type:
        raise ValueError(f"args.json missing model/model_type: {args_path}")
    return model, model_type


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


def generate_prompt_candidates(
    engine: PtEngine,
    request_config: RequestConfig,
    train_rows: List[Dict[str, Any]],
    selection_mode: str,
    number_of_prompts: int,
    fixed_prompt: str,
) -> List[str]:
    if selection_mode == "single":
        return [fixed_prompt]

    reasoning_system = train_rows[0]["messages"][0]["content"]
    reasoning_prompt = train_rows[0]["messages"][1]["content"]

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
        if extracted:
            prompts.append(extracted)
        else:
            prompts.append(resp)

    if not prompts:
        prompts = [fixed_prompt]
    return prompts


def evaluate_prompt(
    engine: PtEngine,
    request_config: RequestConfig,
    test_rows: List[Dict[str, Any]],
    prompt_text: str,
    batch_size: int,
) -> Tuple[float, int, int]:
    correct = 0
    total = 0

    for batch in batched(test_rows, batch_size):
        requests = []
        refs = []
        for row in batch:
            sentence = row["messages"][1]["content"]
            refs.append(str(row["solution"]).strip())
            requests.append(
                {
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": f"{prompt_text}\n{sentence}" if prompt_text else sentence},
                    ]
                }
            )

        responses = engine.infer(requests, request_config, use_tqdm=False)
        for resp, ref in zip(responses, refs):
            pred = resp.choices[0].message.content.strip()
            if pred == ref:
                correct += 1
            total += 1

    accuracy = (correct / total) if total else 0.0
    return accuracy, correct, total


def main() -> None:
    parser = argparse.ArgumentParser(description="MR test-only paper-aligned evaluator (exact-match accuracy).")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint dir (adapter checkpoint).")
    parser.add_argument("--base-model", default=None, help="Base model id/path. If omitted, read from checkpoint args.json")
    parser.add_argument("--model-type", default=None, help="Model type. If omitted, read from checkpoint args.json")
    parser.add_argument("--test-file", default="datasets/original/mr_test.jsonl")
    parser.add_argument("--train-file", default="datasets/original/mr_train.jsonl")
    parser.add_argument("--selection-mode", choices=["single", "best"], default="single")
    parser.add_argument("--number-of-prompts", type=int, default=int(os.environ.get("NUMBER_OF_PROMPTS", "1")))
    parser.add_argument("--fixed-prompt", default="")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    random.seed(args.seed)

    ckpt = os.path.abspath(args.checkpoint)
    base_model, model_type = load_checkpoint_base_info(ckpt)
    if args.base_model:
        base_model = args.base_model
    if args.model_type:
        model_type = args.model_type

    test_rows = read_jsonl(args.test_file)
    train_rows = read_jsonl(args.train_file)
    if args.limit is not None:
        test_rows = test_rows[: args.limit]

    engine = build_engine(base_model, model_type, ckpt, max_batch_size=args.batch_size)
    request_config = RequestConfig(max_tokens=args.max_new_tokens, temperature=args.temperature, top_p=args.top_p)

    prompt_candidates = generate_prompt_candidates(
        engine=engine,
        request_config=request_config,
        train_rows=train_rows,
        selection_mode=args.selection_mode,
        number_of_prompts=args.number_of_prompts,
        fixed_prompt=args.fixed_prompt,
    )

    per_prompt: List[Dict[str, Any]] = []
    best = {
        "accuracy": -1.0,
        "correct": 0,
        "total": 0,
        "prompt": "",
        "index": -1,
    }

    for idx, prompt in enumerate(prompt_candidates):
        acc, correct, total = evaluate_prompt(
            engine=engine,
            request_config=request_config,
            test_rows=test_rows,
            prompt_text=prompt,
            batch_size=args.batch_size,
        )
        item = {
            "prompt_index": idx,
            "prompt": prompt,
            "accuracy": acc,
            "correct": correct,
            "total": total,
        }
        per_prompt.append(item)
        if acc > best["accuracy"]:
            best = {
                "accuracy": acc,
                "correct": correct,
                "total": total,
                "prompt": prompt,
                "index": idx,
            }

    timestamp = datetime.now(timezone.utc).isoformat()
    result = {
        "meta": {
            "timestamp_utc": timestamp,
            "git_commit": safe_git_commit(),
            "checkpoint": ckpt,
            "base_model": base_model,
            "model_type": model_type,
            "test_file": os.path.abspath(args.test_file),
            "train_file": os.path.abspath(args.train_file),
            "selection_mode": args.selection_mode,
            "number_of_prompts": args.number_of_prompts,
            "fixed_prompt": args.fixed_prompt,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "batch_size": args.batch_size,
            "limit": args.limit,
            "seed": args.seed,
            "metric": "exact_match_accuracy",
            "definition": "pred.strip() == solution.strip()",
        },
        "best": best,
        "per_prompt": per_prompt,
    }

    if args.output_json is None:
        out_dir = Path("repro/mr_paper/results")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_name = datetime.now().strftime("mr_test_eval_%Y%m%d-%H%M%S.json")
        output_json = out_dir / out_name
    else:
        output_json = Path(args.output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("=== MR Test Eval Summary ===")
    print(f"checkpoint: {ckpt}")
    print(f"base_model: {base_model}")
    print(f"model_type: {model_type}")
    print(f"test_file: {args.test_file}")
    print(f"selection_mode: {args.selection_mode}")
    print(f"number_of_prompts: {args.number_of_prompts}")
    print(f"temperature/top_p/max_new_tokens: {args.temperature}/{args.top_p}/{args.max_new_tokens}")
    print(f"best_prompt_index: {best['index']}")
    print(f"best_accuracy: {best['accuracy']:.6f} ({best['correct']}/{best['total']})")
    print(f"result_json: {output_json}")


if __name__ == "__main__":
    main()
