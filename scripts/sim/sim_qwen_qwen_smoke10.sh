#!/usr/bin/env bash
set -euo pipefail

if [[ "${CONDA_DEFAULT_ENV:-}" != "prl_clean" ]]; then
  echo "[ERROR] CONDA_DEFAULT_ENV must be 'prl_clean', got '${CONDA_DEFAULT_ENV:-<empty>}'"
  exit 1
fi

if [[ -z "${DATASET:-}" ]]; then
  echo "[ERROR] DATASET is empty."
  echo "Usage example:"
  echo "  DATASET=sim CUDA_VISIBLE_DEVICES=0,1 NPROC_PER_NODE=2 bash scripts/sim/sim_qwen_qwen_smoke10.sh"
  exit 1
fi

if ! command -v swift >/dev/null 2>&1; then
  echo "[ERROR] 'swift' command not found in PATH."
  echo "Please ensure the swift CLI is installed in current env: ${CONDA_DEFAULT_ENV:-<empty>}"
  exit 1
fi

BASE_OUTPUT_DIR="output/sim"
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

NPROC_PER_NODE="${NPROC_PER_NODE:-$(count_visible_gpus)}"
if ! [[ "${NPROC_PER_NODE}" =~ ^[0-9]+$ ]] || [[ "${NPROC_PER_NODE}" -lt 1 ]]; then
  echo "[ERROR] NPROC_PER_NODE must be a positive integer, got '${NPROC_PER_NODE}'"
  exit 1
fi

export MASTER_ADDR="127.0.0.1"
export MASTER_PORT="${MASTER_PORT:-29501}"
export TORCHELASTIC_USE_AGENT_STORE="${TORCHELASTIC_USE_AGENT_STORE:-0}"

if [[ -z "${NCCL_SOCKET_IFNAME:-}" ]]; then
  if ip link show eth0 >/dev/null 2>&1; then
    export NCCL_SOCKET_IFNAME="eth0"
  else
    export NCCL_SOCKET_IFNAME="^lo,docker0"
  fi
fi

SMOKE_TEST="${SMOKE_TEST:-1}"
SMOKE_STEPS="${SMOKE_STEPS:-10}"
if [[ "${SMOKE_TEST}" != "0" ]]; then
  EXTRA_SMOKE_ARGS=(--max_steps "${SMOKE_STEPS}" --logging_steps 1 --eval_steps 1000000 --save_steps 1000000)
else
  EXTRA_SMOKE_ARGS=()
fi

REPORT_TO="${REPORT_TO:-none}"

FINAL_CMD=(
  swift rlhf
  --rlhf_type grpo
  --model Qwen/Qwen2.5-7B-Instruct
  --model_type qwen2_5
  --dataset "datasets/original/${DATASET}_train.jsonl"
  --val_dataset "datasets/original/${DATASET}_val.jsonl"
  --reward_funcs accuracy format
  --torch_dtype bfloat16
  --gradient_checkpointing_kwargs '{"use_reentrant": false}'
  --use_lmdeploy true
  --train_type lora
  --lora_rank 8
  --seed 5
  --lora_alpha 32
  --max_completion_length 1024
  --num_train_epochs 10
  --per_device_train_batch_size 4
  --per_device_eval_batch_size 4
  --learning_rate 1e-6
  --gradient_accumulation_steps 1
  --save_total_limit 20
  --max_length 2048
  --output_dir "${RUN_DIR}"
  --warmup_ratio 0
  --dataloader_num_workers 1
  --dataset_num_proc 4
  --num_generations 4
  --temperature 0.9
  --report_to "${REPORT_TO}"
  --system examples/train/grpo/prompt.txt
  --log_completions true
  --num_iterations 1
  --num_infer_workers 1
)
if [[ "${#EXTRA_SMOKE_ARGS[@]}" -gt 0 ]]; then
  FINAL_CMD+=("${EXTRA_SMOKE_ARGS[@]}")
fi

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "NPROC_PER_NODE=${NPROC_PER_NODE}"
echo "MASTER_ADDR=${MASTER_ADDR}"
echo "MASTER_PORT=${MASTER_PORT}"
echo "RUN_DIR=${RUN_DIR}"
echo "LOG_FILE=${LOG_FILE}"
echo "NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME}"
echo "TORCHELASTIC_USE_AGENT_STORE=${TORCHELASTIC_USE_AGENT_STORE}"
echo -n "FINAL_CMD="
printf '%q ' "${FINAL_CMD[@]}"
echo

NPROC_PER_NODE="${NPROC_PER_NODE}" \
"${FINAL_CMD[@]}" 2>&1 | tee "${LOG_FILE}"

