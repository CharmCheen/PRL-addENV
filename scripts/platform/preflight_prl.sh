#!/usr/bin/env bash
set -euo pipefail

# Preflight checks for non-interactive PRL runs.
#
# Usage:
#   bash scripts/platform/preflight_prl.sh mr

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <mr|sum|sim>"
  exit 1
fi

DATASET_NAME="$1"
case "${DATASET_NAME}" in
  mr|sum|sim) ;;
  *)
    echo "Unsupported dataset: ${DATASET_NAME}. Expected one of: mr, sum, sim"
    exit 1
    ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

require_file() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    echo "Missing required file: ${path}"
    exit 1
  fi
}

require_cmd() {
  local cmd="$1"
  if ! command -v "${cmd}" >/dev/null 2>&1; then
    echo "Missing required command: ${cmd}"
    exit 1
  fi
}

require_cmd bash

if command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN=python3
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN=python
else
  echo "Missing required command: python3 (or python)"
  exit 1
fi

require_file "scripts/${DATASET_NAME}/${DATASET_NAME}_qwen_qwen.sh"
require_file "examples/train/grpo/prompt.txt"
require_file "datasets/original/${DATASET_NAME}_train.jsonl"
require_file "datasets/original/${DATASET_NAME}_val.jsonl"
require_file "datasets/original/${DATASET_NAME}_infer.jsonl"
require_file "requirements.txt"

# Check for GPU availability (single GPU is sufficient).
if command -v nvidia-smi >/dev/null 2>&1; then
  GPU_COUNT="$(nvidia-smi --list-gpus | wc -l | tr -d ' ')"
  if [[ "${GPU_COUNT}" -lt 1 ]]; then
    echo "Preflight failed: no GPUs detected."
    exit 1
  fi
else
  echo "Preflight failed: nvidia-smi not found."
  exit 1
fi

# Ensure required Python packages are importable.
"${PYTHON_BIN}" - <<'PY'
import importlib
required = [
    "torch",
    "transformers",
    "accelerate",
    "datasets",
    "peft",
    "modelscope",
    "trl",
    "vllm",
    "wandb",
    "rouge",
    "mosestokenizer",
]
missing = [m for m in required if importlib.util.find_spec(m) is None]
if missing:
    raise SystemExit(f"Missing Python packages: {', '.join(missing)}")
print("Python package check: OK")
PY

if [[ -z "${WANDB_API_KEY:-}" && "${WANDB_MODE:-}" != "offline" && "${WANDB_MODE:-}" != "disabled" ]]; then
  echo "W&B note: WANDB_API_KEY is empty. Set WANDB_MODE=offline (or provide WANDB_API_KEY) for non-interactive runs."
fi

echo "Preflight check passed for dataset: ${DATASET_NAME}"
