#!/usr/bin/env bash
set -euo pipefail

CONDA_BASE="/qiuyeqing/tools/miniconda3"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate prl_clean

echo "which python: $(which python)"
python -V
echo "CONDA_DEFAULT_ENV=${CONDA_DEFAULT_ENV:-<empty>}"
if [[ "${CONDA_DEFAULT_ENV:-}" != "prl_clean" ]]; then
  echo "[ERROR] conda activate prl_clean failed."
  exit 1
fi

export DATASET="${DATASET:-sum}"
export NUMBER_OF_SAMPLES="${NUMBER_OF_SAMPLES:-10}"
export NUMBER_OF_PROMPTS="${NUMBER_OF_PROMPTS:-3}"
export ADVERSARIAL="${ADVERSARIAL:-0}"
export REASONING="${REASONING:-True}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

count_visible_gpus() {
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    local cleaned
    cleaned="$(echo "${CUDA_VISIBLE_DEVICES}" | tr -d ' ')"
    if [[ -n "${cleaned}" ]]; then
      awk -F',' '{print NF}' <<<"${cleaned}"
      return
    fi
  fi
  if command -v nvidia-smi >/dev/null 2>&1; then
    local n
    n="$(nvidia-smi -L 2>/dev/null | grep -c '^GPU' || true)"
    if [[ "${n}" -gt 0 ]]; then
      echo "${n}"
      return
    fi
  fi
  echo "1"
}

choose_master_port() {
  python - <<'PY'
import socket
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind(('', 0))
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    print(s.getsockname()[1])
PY
}

VISIBLE_GPUS="$(count_visible_gpus)"
if [[ -z "${NPROC_PER_NODE:-}" ]]; then
  NPROC_PER_NODE="${VISIBLE_GPUS}"
elif ! [[ "${NPROC_PER_NODE}" =~ ^[0-9]+$ ]] || [[ "${NPROC_PER_NODE}" -lt 1 ]]; then
  echo "[ERROR] NPROC_PER_NODE must be positive integer, got '${NPROC_PER_NODE}'"
  exit 1
elif [[ "${NPROC_PER_NODE}" -gt "${VISIBLE_GPUS}" ]]; then
  echo "[WARN] NPROC_PER_NODE=${NPROC_PER_NODE} > visible_gpus=${VISIBLE_GPUS}; auto-downgrade."
  NPROC_PER_NODE="${VISIBLE_GPUS}"
fi
export NPROC_PER_NODE
export MASTER_ADDR="127.0.0.1"
export MASTER_PORT="${MASTER_PORT:-$(choose_master_port)}"
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

BASE_OUTPUT_DIR="/qiuyeqing/llama_prl/PRL-REDO/PRL-addENV/output/sum"
TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
RUN_DIR="${BASE_OUTPUT_DIR}/sum-qwen-qwen-${TIMESTAMP}"
LOG_FILE="${RUN_DIR}/sum.log"
mkdir -p "${RUN_DIR}"

SWIFT_BIN="${CONDA_PREFIX}/bin/swift"
if [[ -x "${SWIFT_BIN}" ]]; then
  SWIFT_CMD=("${SWIFT_BIN}")
else
  SWIFT_CMD=("python" "-m" "swift")
fi

USE_LMDEPLOY="${USE_LMDEPLOY:-1}"
if [[ "${USE_LMDEPLOY}" == "1" ]]; then
  USE_LMDEPLOY_ARG="true"
else
  USE_LMDEPLOY_ARG="false"
fi

DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-1}"
if ! [[ "${DATALOADER_NUM_WORKERS}" =~ ^[0-9]+$ ]]; then
  echo "[ERROR] DATALOADER_NUM_WORKERS must be non-negative integer, got '${DATALOADER_NUM_WORKERS}'"
  exit 1
fi
if df -k /dev/shm >/dev/null 2>&1; then
  SHM_KB="$(df -k /dev/shm | awk 'NR==2 {print $2}')"
  if [[ -n "${SHM_KB}" ]] && [[ "${SHM_KB}" -lt 8388608 ]] && [[ "${DATALOADER_NUM_WORKERS}" -gt 2 ]]; then
    echo "[WARN] /dev/shm < 8GB; cap DATALOADER_NUM_WORKERS to 2 (was ${DATALOADER_NUM_WORKERS})."
    DATALOADER_NUM_WORKERS=2
  fi
fi

FINAL_CMD=(
  "${SWIFT_CMD[@]}" rlhf
  --rlhf_type grpo
  --model Qwen/Qwen2.5-7B-Instruct
  --model_type qwen2_5
  --dataset "datasets/original/${DATASET}_train.jsonl"
  --val_dataset "datasets/original/${DATASET}_val.jsonl"
  --reward_funcs accuracy format
  --torch_dtype bfloat16
  --gradient_checkpointing_kwargs '{"use_reentrant": false}'
  --use_lmdeploy "${USE_LMDEPLOY_ARG}"
  --train_type lora
  --lora_rank 8
  --lora_alpha 32
  --seed "${SEED:-5}"
  --max_completion_length 1024
  --num_train_epochs 10
  --per_device_train_batch_size 4
  --per_device_eval_batch_size 4
  --learning_rate 1e-6
  --gradient_accumulation_steps 1
  --eval_steps 100
  --save_steps 100
  --save_total_limit 20
  --max_length 2048
  --output_dir "${RUN_DIR}"
  --warmup_ratio 0
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS}"
  --dataset_num_proc 4
  --num_generations 4
  --temperature 0.9
  --report_to "${REPORT_TO:-wandb}"
  --logging_steps 5
  --system examples/train/grpo/prompt.txt
  --log_completions true
  --num_iterations 1
  --num_infer_workers 1
)

echo "RUN_DIR=${RUN_DIR}"
echo "LOG_FILE=${LOG_FILE}"
echo -n "FINAL_CMD="
printf '%q ' "${FINAL_CMD[@]}"
echo

"${FINAL_CMD[@]}" 2>&1 | tee "${LOG_FILE}"
