#!/usr/bin/env python3
import argparse
import json
import os
import random
import statistics
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from mosestokenizer import MosesTokenizer
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


def pick_default_base_model() -> str:
    local_global = Path("/qiuyeqing/.cache/modelscope/models/Qwen/Qwen2___5-7B-Instruct")
    if local_global.exists():
        return str(local_global.resolve())
    local = Path(".cache/modelscope/models/Qwen/Qwen2___5-7B-Instruct")
    if local.exists():
        return str(local.resolve())
    return "Qwen/Qwen2.5-7B-Instruct"


def build_engine(base_model: str, model_type: str, checkpoint_dir: Optional[str], max_batch_size: int) -> PtEngine:
    adapters = [checkpoint_dir] if checkpoint_dir else []
    infer_args = InferArguments(
        model=base_model,
        model_type=model_type,
        adapters=adapters,
        infer_backend="pt",
        max_batch_size=max_batch_size,
    )
    model, template = prepare_model_template(infer_args)
    return PtEngine.from_model_template(model, template, max_batch_size=max_batch_size)


def _get_ngrams(tokens: List[str], n: int) -> Counter:
    if len(tokens) < n:
        return Counter()
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def _safe_f1(p: float, r: float) -> float:
    if p + r == 0.0:
        return 0.0
    return 2.0 * p * r / (p + r)


def _ratio(num: float, den: float, empty_value: float = 1.0) -> float:
    if den == 0:
        return empty_value
    return num / den


def sentence_sari(src: str, pred: str, refs: List[str], tokenizer: MosesTokenizer) -> float:
    src_tokens = tokenizer(src)
    pred_tokens = tokenizer(pred)
    ref_tokens_list = [tokenizer(r) for r in refs]
    if not ref_tokens_list:
        return 0.0

    sari_n_scores = []
    for n in [1, 2, 3, 4]:
        src_ngrams = set(_get_ngrams(src_tokens, n).keys())
        pred_ngrams = set(_get_ngrams(pred_tokens, n).keys())
        ref_ngrams_union = set()
        for ref_tokens in ref_tokens_list:
            ref_ngrams_union |= set(_get_ngrams(ref_tokens, n).keys())

        pred_add = pred_ngrams - src_ngrams
        ref_add = ref_ngrams_union - src_ngrams
        add_overlap = pred_add & ref_add
        p_add = _ratio(len(add_overlap), len(pred_add))
        r_add = _ratio(len(add_overlap), len(ref_add))
        f_add = _safe_f1(p_add, r_add)

        pred_keep = pred_ngrams & src_ngrams
        ref_keep = ref_ngrams_union & src_ngrams
        keep_overlap = pred_keep & ref_keep
        p_keep = _ratio(len(keep_overlap), len(pred_keep))
        r_keep = _ratio(len(keep_overlap), len(ref_keep))
        f_keep = _safe_f1(p_keep, r_keep)

        pred_del = src_ngrams - pred_ngrams
        ref_del = src_ngrams - ref_ngrams_union
        del_overlap = pred_del & ref_del
        p_del = _ratio(len(del_overlap), len(pred_del))

        sari_n_scores.append((f_add + f_keep + p_del) / 3.0)

    return float(sum(sari_n_scores) / 4.0 * 100.0)


def run_model_on_split(
    engine: PtEngine,
    request_config: RequestConfig,
    rows: List[Dict[str, Any]],
    prompt_text: str,
    batch_size: int,
) -> Tuple[List[str], List[str], List[str]]:
    preds: List[str] = []
    srcs: List[str] = []
    refs: List[str] = []
    for batch in batched(rows, batch_size):
        requests = []
        for row in batch:
            source = row["messages"][1]["content"]
            requests.append({
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": f"{prompt_text}\n{source}" if prompt_text else source},
                ]
            })
            srcs.append(source.strip())
            refs.append(str(row["solution"]).strip())
        responses = engine.infer(requests, request_config, use_tqdm=False)
        preds.extend([r.choices[0].message.content.strip() for r in responses])
    return srcs, preds, refs


def compute_sari(srcs: List[str], preds: List[str], refs: List[str]) -> float:
    tok = MosesTokenizer("en")
    values = [sentence_sari(src=s, pred=p, refs=[r], tokenizer=tok) for s, p, r in zip(srcs, preds, refs)]
    return float(statistics.mean(values)) if values else 0.0


def generate_prompt_candidates(
    engine: PtEngine,
    request_config: RequestConfig,
    train_rows: List[Dict[str, Any]],
    number_of_prompts: int,
    fixed_prompt: str,
) -> List[str]:
    if fixed_prompt:
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
        prompts.append(extracted if extracted else resp)
    return prompts


def select_prompt_on_validation(
    engine: PtEngine,
    request_config: RequestConfig,
    val_rows: List[Dict[str, Any]],
    prompt_candidates: List[str],
    batch_size: int,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    best: Dict[str, Any] = {
        "index": -1,
        "prompt": "",
        "score": -1.0,
    }
    per_prompt: List[Dict[str, Any]] = []
    for idx, prompt in enumerate(prompt_candidates):
        srcs, preds, refs = run_model_on_split(engine, request_config, val_rows, prompt, batch_size)
        sari = compute_sari(srcs, preds, refs)
        item = {
            "prompt_index": idx,
            "prompt": prompt,
            "score": sari,
        }
        per_prompt.append(item)
        if sari > best["score"]:
            best = item
    return best, per_prompt


def main() -> None:
    parser = argparse.ArgumentParser(description="SIM paper-aligned evaluator: val prompt selection + test report.")
    parser.add_argument("--checkpoint", default="", help="Adapter checkpoint dir; empty means base model only.")
    parser.add_argument("--base-model", default=pick_default_base_model())
    parser.add_argument("--model-type", default="qwen2_5")
    parser.add_argument("--train-file", default="datasets/original/sim_train.jsonl")
    parser.add_argument("--val-file", default="datasets/original/sim_val.jsonl")
    parser.add_argument("--test-file", default="datasets/original/sim_test.jsonl")
    parser.add_argument("--number-of-prompts", type=int, default=5)
    parser.add_argument("--fixed-prompt", default="")
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None, help="Debug limit applied to val and test.")
    parser.add_argument("--output-json", default="")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_rows = read_jsonl(args.train_file)
    val_rows = read_jsonl(args.val_file)
    test_rows = read_jsonl(args.test_file)
    if args.limit is not None:
        val_rows = val_rows[:args.limit]
        test_rows = test_rows[:args.limit]

    checkpoint = os.path.abspath(args.checkpoint) if args.checkpoint else ""
    engine = build_engine(args.base_model, args.model_type, checkpoint or None, args.batch_size)
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

    selected, validation_per_prompt = select_prompt_on_validation(
        engine=engine,
        request_config=request_config,
        val_rows=val_rows,
        prompt_candidates=prompt_candidates,
        batch_size=args.batch_size,
    )

    srcs, preds, refs = run_model_on_split(
        engine=engine,
        request_config=request_config,
        rows=test_rows,
        prompt_text=selected["prompt"],
        batch_size=args.batch_size,
    )
    test_sari = compute_sari(srcs, preds, refs)

    result = {
        "meta": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "git_commit": safe_git_commit(),
            "checkpoint": checkpoint if checkpoint else "none",
            "base_model": os.path.abspath(args.base_model) if os.path.exists(args.base_model) else args.base_model,
            "model_type": args.model_type,
            "task": "sim",
            "split": "test",
            "selection_split": "val",
            "metric": "sari",
            "seed": args.seed,
            "batch_size": args.batch_size,
            "decode_params": {
                "temperature": args.temperature,
                "top_p": args.top_p,
                "max_new_tokens": args.max_new_tokens,
            },
            "number_of_prompts": args.number_of_prompts,
            "limit": args.limit,
            "n_train": len(train_rows),
            "n_val": len(val_rows),
            "n_test": len(test_rows),
        },
        "validation": {
            "selected_prompt": selected["prompt"],
            "score": selected["score"],
            "per_prompt": validation_per_prompt,
        },
        "test": {
            "sari": test_sari,
        },
    }

    if args.output_json:
        output_json = Path(args.output_json)
    else:
        output_json = Path("repro/sim_paper/results") / datetime.now().strftime("sim_paper_eval_%Y%m%d-%H%M%S.json")
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("=== SIM Paper Eval Summary ===")
    print(f"seed: {args.seed}")
    print(f"n_val/n_test: {len(val_rows)}/{len(test_rows)}")
    print(f"selected_prompt_index: {selected['prompt_index']}")
    print(f"validation_sari: {selected['score']:.4f}")
    print(f"test_sari: {test_sari:.4f}")
    print(f"output_json: {output_json}")


if __name__ == "__main__":
    main()
