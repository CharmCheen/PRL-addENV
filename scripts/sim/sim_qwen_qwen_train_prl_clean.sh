#!/usr/bin/env bash
set -euo pipefail

CONDA_BASE="/qiuyeqing/tools/miniconda3"
CONDA_SH="${CONDA_BASE}/etc/profile.d/conda.sh"
if [[ ! -f "${CONDA_SH}" ]]; then
  echo "[ERROR] Missing conda init script: ${CONDA_SH}"
  exit 1
fi
source "${CONDA_SH}"
conda activate prl_clean

echo "which python: $(which python)"
python -V
echo "CONDA_DEFAULT_ENV=${CONDA_DEFAULT_ENV:-<empty>}"
if [[ "${CONDA_DEFAULT_ENV:-}" != "prl_clean" ]]; then
  echo "[ERROR] conda activate prl_clean failed, env=${CONDA_DEFAULT_ENV:-<empty>}"
  exit 1
fi

export DATASET="${DATASET:-sim}"
if [[ -z "${DATASET}" ]]; then
  echo "[ERROR] DATASET is empty."
  exit 1
fi

export NUMBER_OF_SAMPLES="${NUMBER_OF_SAMPLES:-20}"
export NUMBER_OF_PROMPTS="${NUMBER_OF_PROMPTS:-5}"
export ADVERSARIAL="${ADVERSARIAL:-0}"
export REASONING="${REASONING:-True}"

BASE_OUTPUT_DIR="/qiuyeqing/llama_prl/PRL-REDO/PRL-addENV/output/sim"
TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
RUN_DIR="${BASE_OUTPUT_DIR}/sim-qwen-qwen-${TIMESTAMP}"
LOG_FILE="${RUN_DIR}/sim.log"
mkdir -p "${RUN_DIR}"

count_visible_gpus() {
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    local cleaned
    cleaned="$(echo "${CUDA_VISIBLE_DEVICES}" | tr -d ' ')"
    if [[ -z "${cleaned}" ]]; then
      echo "1"
      return
    fi
    awk -F',' '{print NF}' <<<"${cleaned}"
    return
  fi
  python - <<'PY'
try:
    import torch
    n = int(torch.cuda.device_count())
    print(n if n > 0 else 1)
except Exception:
    print(1)
PY
}

SWIFT_BIN="${CONDA_PREFIX}/bin/swift"
if [[ -x "${SWIFT_BIN}" ]]; then
  SWIFT_CMD=("${SWIFT_BIN}")
  SWIFT_ENTRY="${SWIFT_BIN}"
else
  SWIFT_CMD=("python" "-m" "swift")
  SWIFT_ENTRY="python -m swift"
fi

export NPROC_PER_NODE="${NPROC_PER_NODE:-$(count_visible_gpus)}"
if ! [[ "${NPROC_PER_NODE}" =~ ^[0-9]+$ ]] || [[ "${NPROC_PER_NODE}" -lt 1 ]]; then
  echo "[ERROR] NPROC_PER_NODE must be positive integer, got '${NPROC_PER_NODE}'"
  exit 1
fi
export MASTER_ADDR="127.0.0.1"
export MASTER_PORT="${MASTER_PORT:-29501}"
export TORCHELASTIC_USE_AGENT_STORE=0
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

if [[ -z "${NCCL_SOCKET_IFNAME:-}" ]]; then
  if command -v ip >/dev/null 2>&1 && ip link show eth0 >/dev/null 2>&1; then
    export NCCL_SOCKET_IFNAME="eth0"
  else
    export NCCL_SOCKET_IFNAME="^lo,docker0"
  fi
fi

USE_LMDEPLOY="${USE_LMDEPLOY:-0}"
if [[ "${USE_LMDEPLOY}" == "1" ]]; then
  USE_LMDEPLOY_ARG="true"
else
  USE_LMDEPLOY_ARG="false"
fi

REPORT_TO="${REPORT_TO:-wandb}"
MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
MODEL_TYPE="${MODEL_TYPE:-qwen2_5}"

FINAL_CMD=(
  "${SWIFT_CMD[@]}" rlhf
  --rlhf_type grpo
  --model "${MODEL}"
  --model_type "${MODEL_TYPE}"
  --dataset "datasets/original/${DATASET}_train.jsonl"
  --val_dataset "datasets/original/${DATASET}_val.jsonl"
  --reward_funcs accuracy format
  --torch_dtype bfloat16
  --gradient_checkpointing_kwargs '{"use_reentrant": false}'
  --use_lmdeploy "${USE_LMDEPLOY_ARG}"
  --train_type lora
  --lora_rank 8
  --seed "${SEED:-5}"
  --lora_alpha 32
  --max_completion_length "${MAX_COMPLETION_LENGTH:-1024}"
  --num_train_epochs "${NUM_TRAIN_EPOCHS:-10}"
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE:-4}"
  --per_device_eval_batch_size "${PER_DEVICE_EVAL_BATCH_SIZE:-4}"
  --learning_rate "${LEARNING_RATE:-1e-6}"
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-1}"
  --eval_steps "${EVAL_STEPS:-100}"
  --save_steps "${SAVE_STEPS:-100}"
  --save_total_limit 20
  --max_length "${MAX_LENGTH:-2048}"
  --output_dir "${RUN_DIR}"
  --warmup_ratio 0
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS:-1}"
  --dataset_num_proc "${DATASET_NUM_PROC:-4}"
  --num_generations "${NUM_GENERATIONS:-4}"
  --temperature "${TEMPERATURE:-0.9}"
  --report_to "${REPORT_TO}"
  --logging_steps "${LOGGING_STEPS:-5}"
  --system examples/train/grpo/prompt.txt
  --log_completions true
  --num_iterations 1
  --num_infer_workers "${NUM_INFER_WORKERS:-1}"
)

echo "SWIFT_ENTRY=${SWIFT_ENTRY}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "NPROC_PER_NODE=${NPROC_PER_NODE}"
echo "MASTER_ADDR=${MASTER_ADDR}"
echo "MASTER_PORT=${MASTER_PORT}"
echo "RUN_DIR=${RUN_DIR}"
echo "LOG_FILE=${LOG_FILE}"
echo "NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME}"
echo "TORCHELASTIC_USE_AGENT_STORE=${TORCHELASTIC_USE_AGENT_STORE}"
echo "NCCL_ASYNC_ERROR_HANDLING=${NCCL_ASYNC_ERROR_HANDLING}"
echo "NCCL_DEBUG=${NCCL_DEBUG}"
echo -n "FINAL_CMD="
printf '%q ' "${FINAL_CMD[@]}"
echo

"${FINAL_CMD[@]}" 2>&1 | tee "${LOG_FILE}"

