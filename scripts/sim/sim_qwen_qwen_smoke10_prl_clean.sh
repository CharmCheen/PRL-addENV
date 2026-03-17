#!/usr/bin/env bash
set -euo pipefail

CONDA_BASE="/qiuyeqing/tools/miniconda3"
if [[ ! -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
  echo "[ERROR] Missing conda init script: ${CONDA_BASE}/etc/profile.d/conda.sh"
  exit 1
fi
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate prl_clean

echo "which python: $(which python)"
python -V
echo "CONDA_DEFAULT_ENV=${CONDA_DEFAULT_ENV:-<empty>}"
if [[ "${CONDA_DEFAULT_ENV:-}" != "prl_clean" ]]; then
  echo "[ERROR] conda activate prl_clean failed, env=${CONDA_DEFAULT_ENV:-<empty>}"
  exit 1
fi

if [[ -z "${DATASET:-}" ]]; then
  echo "[ERROR] DATASET is empty."
  echo "Usage: DATASET=sim [CUDA_VISIBLE_DEVICES=0,1] [NPROC_PER_NODE=2] bash scripts/sim/sim_qwen_qwen_smoke10_prl_clean.sh"
  exit 1
fi

export NUMBER_OF_SAMPLES="${NUMBER_OF_SAMPLES:-20}"
export NUMBER_OF_PROMPTS="${NUMBER_OF_PROMPTS:-5}"
export ADVERSARIAL="${ADVERSARIAL:-0}"
export REASONING="${REASONING:-True}"
# USE_HUMAN_LABELS=1 means the reward signal aligns with human gold labels (ADVERSARIAL=0).
# USE_HUMAN_LABELS=0 means adversarial mode where rewards oppose the gold labels (ADVERSARIAL=1).
export USE_HUMAN_LABELS=$([ "${ADVERSARIAL}" = "0" ] && echo 1 || echo 0)
TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
RUN_DIR="${BASE_OUTPUT_DIR}/${TIMESTAMP}"
LOG_FILE="${RUN_DIR}/sim.log"
mkdir -p "${RUN_DIR}"

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

SWIFT_BIN="${CONDA_PREFIX}/bin/swift"
if [[ -x "${SWIFT_BIN}" ]]; then
  SWIFT_CMD=("${SWIFT_BIN}")
  SWIFT_ENTRY="${SWIFT_BIN}"
else
  SWIFT_CMD=("python" "-m" "swift")
  SWIFT_ENTRY="python -m swift"
fi

VISIBLE_GPUS="$(count_visible_gpus)"
export NPROC_PER_NODE="${NPROC_PER_NODE:-${VISIBLE_GPUS}}"
if ! [[ "${NPROC_PER_NODE}" =~ ^[0-9]+$ ]] || [[ "${NPROC_PER_NODE}" -lt 1 ]]; then
  echo "[ERROR] NPROC_PER_NODE must be positive integer, got '${NPROC_PER_NODE}'"
  exit 1
fi
if [[ "${NPROC_PER_NODE}" -gt "${VISIBLE_GPUS}" ]]; then
  echo "[WARN] NPROC_PER_NODE=${NPROC_PER_NODE} > visible_gpus=${VISIBLE_GPUS}; auto-downgrade."
  NPROC_PER_NODE="${VISIBLE_GPUS}"
fi
export MASTER_ADDR="127.0.0.1"
if [[ -z "${MASTER_PORT:-}" ]]; then
  MASTER_PORT="$(python - <<'PY'
import socket
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind(('', 0))
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    print(s.getsockname()[1])
PY
)"
fi
export MASTER_PORT
export TORCHELASTIC_USE_AGENT_STORE=0
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-DETAIL}"

choose_primary_iface() {
  if ! command -v ip >/dev/null 2>&1; then
    return 1
  fi
  ip -o link show \
    | awk -F': ' '{print $2}' \
    | sed -E 's/@.*$//' \
    | grep -Ev '^(lo|docker0|veth.*|br-.*|cni.*|flannel.*)$' \
    | head -n1
}

if [[ -z "${NCCL_SOCKET_IFNAME:-}" ]]; then
  if [[ "${MASTER_ADDR}" == "127.0.0.1" || "${MASTER_ADDR}" == "localhost" ]]; then
    export NCCL_SOCKET_IFNAME="lo"
  elif iface="$(choose_primary_iface)"; [[ -n "${iface:-}" ]]; then
    export NCCL_SOCKET_IFNAME="${iface}"
  else
    export NCCL_SOCKET_IFNAME="^lo,docker0"
  fi
fi
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-${NCCL_SOCKET_IFNAME}}"

SMOKE_TEST="${SMOKE_TEST:-1}"
SMOKE_STEPS="${SMOKE_STEPS:-10}"
USE_LMDEPLOY="${USE_LMDEPLOY:-0}"
if [[ "${USE_LMDEPLOY}" == "1" ]]; then
  USE_LMDEPLOY_ARG="true"
else
  USE_LMDEPLOY_ARG="false"
fi

REPORT_TO="${REPORT_TO:-none}"
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
  --seed 5
  --lora_alpha 32
  --max_completion_length "${MAX_COMPLETION_LENGTH:-512}"
  --num_train_epochs "${NUM_TRAIN_EPOCHS:-10}"
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE:-2}"
  --per_device_eval_batch_size "${PER_DEVICE_EVAL_BATCH_SIZE:-2}"
  --learning_rate "${LEARNING_RATE:-1e-6}"
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-1}"
  --save_total_limit 20
  --max_length "${MAX_LENGTH:-1024}"
  --output_dir "${RUN_DIR}"
  --warmup_ratio 0
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS:-1}"
  --dataset_num_proc "${DATASET_NUM_PROC:-1}"
  --num_generations "${NUM_GENERATIONS:-2}"
  --temperature "${TEMPERATURE:-0.9}"
  --report_to "${REPORT_TO}"
  --logging_steps 1
  --system examples/train/grpo/prompt.txt
  --log_completions true
  --num_iterations 1
  --num_infer_workers "${NUM_INFER_WORKERS:-1}"
)

if [[ "${SMOKE_TEST}" != "0" ]]; then
  FINAL_CMD+=(--max_steps "${SMOKE_STEPS}" --eval_steps 1000000 --save_steps 1000000)
fi

echo "SWIFT_ENTRY=${SWIFT_ENTRY}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "NPROC_PER_NODE=${NPROC_PER_NODE}"
echo "MASTER_ADDR=${MASTER_ADDR}"
echo "MASTER_PORT=${MASTER_PORT}"
echo "RUN_DIR=${RUN_DIR}"
echo "LOG_FILE=${LOG_FILE}"
echo "NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME}"
echo "GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME}"
echo "TORCHELASTIC_USE_AGENT_STORE=${TORCHELASTIC_USE_AGENT_STORE}"
echo "TORCH_NCCL_ASYNC_ERROR_HANDLING=${TORCH_NCCL_ASYNC_ERROR_HANDLING}"
echo "NCCL_ASYNC_ERROR_HANDLING=${NCCL_ASYNC_ERROR_HANDLING}"
echo "NCCL_IB_DISABLE=${NCCL_IB_DISABLE}"
echo "NCCL_DEBUG=${NCCL_DEBUG}"
echo "TORCH_DISTRIBUTED_DEBUG=${TORCH_DISTRIBUTED_DEBUG}"
echo "ADVERSARIAL=${ADVERSARIAL} USE_HUMAN_LABELS=${USE_HUMAN_LABELS}"
echo -n "FINAL_CMD="
printf '%q ' "${FINAL_CMD[@]}"
echo

"${FINAL_CMD[@]}" 2>&1 | tee "${LOG_FILE}"

# smoke success check: require observable runtime step progression with step>=10
MAX_STEP="$(grep -Eiv 'FINAL_CMD|max_steps|eval_steps|save_steps|logging_steps' "${LOG_FILE}" \
  | grep -Eio '(global[_ ]?step|step)[^0-9]{0,12}[0-9]+' \
  | grep -Eo '[0-9]+' \
  | sort -n \
  | tail -1 || true)"

if [[ -z "${MAX_STEP}" ]]; then
  MAX_STEP="0"
fi

if [[ "${SMOKE_TEST}" != "0" ]]; then
  if [[ "${MAX_STEP}" -ge "${SMOKE_STEPS}" ]]; then
    echo "[OK] Smoke test reached step>=${SMOKE_STEPS}. max_observed_step=${MAX_STEP}"
  else
    echo "[ERROR] Smoke test did NOT reach step>=${SMOKE_STEPS}. max_observed_step=${MAX_STEP}"
    echo "[ERROR] Check log: ${LOG_FILE}"
    exit 1
  fi
fi
