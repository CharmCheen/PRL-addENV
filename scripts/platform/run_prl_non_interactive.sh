#!/usr/bin/env bash
set -euo pipefail

# Non-interactive launcher for compute platforms.
# It reuses the original training scripts without changing training logic.
#
# Usage:
#   bash scripts/platform/run_prl_non_interactive.sh mr
#   bash scripts/platform/run_prl_non_interactive.sh sum
#   bash scripts/platform/run_prl_non_interactive.sh sim

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

# Enter repo root no matter where the script is called from.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

# Auto-load platform env file if present.
ENV_FILE="${REPO_ROOT}/scripts/platform/platform.env"
if [[ -f "${ENV_FILE}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a
fi

# Required for unattended jobs.
export PIP_NO_INPUT="${PIP_NO_INPUT:-1}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

# Caches (override in platform env if needed).
export HF_HOME="${HF_HOME:-${REPO_ROOT}/.cache/hf}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-${REPO_ROOT}/.cache/modelscope}"

# Default to HuggingFace hub behavior unless caller overrides.
export USE_HF="${USE_HF:-1}"

# Keep W&B non-interactive by default when key is absent.
if [[ -z "${WANDB_API_KEY:-}" ]]; then
  export WANDB_MODE="${WANDB_MODE:-offline}"
fi

# Defaults required by current PRL custom logic.
export ADVERSARIAL="${ADVERSARIAL:-0}"
export REASONING="${REASONING:-True}"

if [[ "${DATASET_NAME}" == "mr" ]]; then
  export NUMBER_OF_SAMPLES="${NUMBER_OF_SAMPLES:-100}"
  export NUMBER_OF_PROMPTS="${NUMBER_OF_PROMPTS:-10}"
elif [[ "${DATASET_NAME}" == "sum" ]]; then
  export NUMBER_OF_SAMPLES="${NUMBER_OF_SAMPLES:-10}"
  export NUMBER_OF_PROMPTS="${NUMBER_OF_PROMPTS:-3}"
else
  export NUMBER_OF_SAMPLES="${NUMBER_OF_SAMPLES:-20}"
  export NUMBER_OF_PROMPTS="${NUMBER_OF_PROMPTS:-5}"
fi

RUN_SCRIPT="scripts/${DATASET_NAME}/${DATASET_NAME}_qwen_qwen.sh"
if [[ ! -f "${RUN_SCRIPT}" ]]; then
  echo "Missing run script: ${RUN_SCRIPT}"
  exit 1
fi

bash scripts/platform/preflight_prl.sh "${DATASET_NAME}"

echo "Launching ${RUN_SCRIPT} (non-interactive mode)"
echo "DATASET=${DATASET_NAME}, NUMBER_OF_SAMPLES=${NUMBER_OF_SAMPLES}, NUMBER_OF_PROMPTS=${NUMBER_OF_PROMPTS}"
echo "ADVERSARIAL=${ADVERSARIAL}, REASONING=${REASONING}, WANDB_MODE=${WANDB_MODE:-online}"

bash "${RUN_SCRIPT}"
