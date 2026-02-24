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
touch "${LOG_FILE}"

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
export NCCL_BLOCKING_WAIT="${NCCL_BLOCKING_WAIT:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-DETAIL}"
CPU_CORES="$(nproc --all 2>/dev/null || echo 16)"
THREADS_PER_RANK=$(( CPU_CORES / NPROC_PER_NODE ))
if [[ "${THREADS_PER_RANK}" -lt 1 ]]; then
  THREADS_PER_RANK=1
fi
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${THREADS_PER_RANK}}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${THREADS_PER_RANK}}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONFAULTHANDLER="${PYTHONFAULTHANDLER:-1}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export TORCH_SHOW_CPP_STACKTRACES="${TORCH_SHOW_CPP_STACKTRACES:-1}"
export HEARTBEAT_INTERVAL_SEC="${HEARTBEAT_INTERVAL_SEC:-300}"
export STALL_TIMEOUT_SEC="${STALL_TIMEOUT_SEC:-1200}"
export ENABLE_HANG_DUMP="${ENABLE_HANG_DUMP:-1}"

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
  if iface="$(choose_primary_iface)"; [[ -n "${iface:-}" ]]; then
    export NCCL_SOCKET_IFNAME="${iface}"
  fi
fi
if [[ -z "${GLOO_SOCKET_IFNAME:-}" && -n "${NCCL_SOCKET_IFNAME:-}" ]]; then
  export GLOO_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME}"
fi

USE_LMDEPLOY="${USE_LMDEPLOY:-0}"
if [[ "${USE_LMDEPLOY}" == "1" ]]; then
  USE_LMDEPLOY_ARG="true"
else
  USE_LMDEPLOY_ARG="false"
fi

REPORT_TO="${REPORT_TO:-tensorboard}"
MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
MODEL_TYPE="${MODEL_TYPE:-qwen2_5}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-2}"
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
  --max_completion_length "${MAX_COMPLETION_LENGTH:-768}"
  --num_train_epochs "${NUM_TRAIN_EPOCHS:-10}"
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE:-3}"
  --per_device_eval_batch_size "${PER_DEVICE_EVAL_BATCH_SIZE:-4}"
  --learning_rate "${LEARNING_RATE:-1e-6}"
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-2}"
  --eval_steps "${EVAL_STEPS:-300}"
  --save_steps "${SAVE_STEPS:-300}"
  --save_total_limit 20
  --max_length "${MAX_LENGTH:-1536}"
  --output_dir "${RUN_DIR}"
  --warmup_ratio 0
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS}"
  --dataset_num_proc "${DATASET_NUM_PROC:-4}"
  --num_generations "${NUM_GENERATIONS:-2}"
  --temperature "${TEMPERATURE:-0.9}"
  --report_to "${REPORT_TO}"
  --logging_steps "${LOGGING_STEPS:-10}"
  --system examples/train/grpo/prompt.txt
  --log_completions true
  --num_iterations 1
  --num_infer_workers "${NUM_INFER_WORKERS:-1}"
)

if [[ -n "${MAX_STEPS:-}" ]]; then
  FINAL_CMD+=(--max_steps "${MAX_STEPS}")
fi
if [[ -n "${DDP_TIMEOUT:-}" ]]; then
  FINAL_CMD+=(--ddp_timeout "${DDP_TIMEOUT}")
fi

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "${LOG_FILE}"
}

collect_descendant_pids() {
  local root="$1"
  local queue=("${root}")
  local out=()
  local idx=0
  while [[ "${idx}" -lt "${#queue[@]}" ]]; do
    local current="${queue[${idx}]}"
    out+=("${current}")
    while read -r child; do
      [[ -n "${child}" ]] && queue+=("${child}")
    done < <(pgrep -P "${current}" 2>/dev/null || true)
    idx=$((idx + 1))
  done
  printf '%s\n' "${out[@]}"
}

dump_runtime_state() {
  local reason="$1"
  log "debug-dump start: reason=${reason}"
  {
    echo "----- ps (python/swift/torchrun) -----"
    ps -eo pid,ppid,stat,etime,pcpu,pmem,cmd | grep -E 'python|swift|torchrun' | grep -v grep || true
    echo "----- nvidia-smi -----"
    nvidia-smi || true
    echo "----- /dev/shm -----"
    df -h /dev/shm || true
  } | tee -a "${LOG_FILE}"
  if [[ -n "${TRAIN_PID:-}" ]] && kill -0 "${TRAIN_PID}" 2>/dev/null; then
    while read -r pid; do
      kill -USR1 "${pid}" 2>/dev/null || true
    done < <(collect_descendant_pids "${TRAIN_PID}")
    log "sent SIGUSR1 to train process tree rooted at pid=${TRAIN_PID}"
  fi
}

heartbeat_loop() {
  local started_ts="$1"
  local last_change_ts
  local last_size
  last_change_ts="$(date +%s)"
  last_size="$(stat -c %s "${LOG_FILE}" 2>/dev/null || echo 0)"
  while kill -0 "${TRAIN_PID}" 2>/dev/null; do
    sleep "${HEARTBEAT_INTERVAL_SEC}"
    local now elapsed size
    now="$(date +%s)"
    elapsed=$((now - started_ts))
    size="$(stat -c %s "${LOG_FILE}" 2>/dev/null || echo 0)"
    if [[ "${size}" != "${last_size}" ]]; then
      last_size="${size}"
      last_change_ts="${now}"
    fi
    log "heartbeat: train_pid=${TRAIN_PID} elapsed_sec=${elapsed} log_size_bytes=${size}"
    if [[ "${ENABLE_HANG_DUMP}" == "1" ]] && (( now - last_change_ts >= STALL_TIMEOUT_SEC )); then
      dump_runtime_state "log-stall-${STALL_TIMEOUT_SEC}s"
      last_change_ts="${now}"
    fi
  done
}

if df -B1 /dev/shm >/dev/null 2>&1; then
  SHM_BYTES="$(df -B1 /dev/shm | awk 'NR==2 {print $2}')"
  SHM_GB="$(awk -v v="${SHM_BYTES:-0}" 'BEGIN{printf "%.2f", v/1024/1024/1024}')"
  if [[ -n "${SHM_BYTES:-}" ]] && [[ "${SHM_BYTES}" -lt $((8 * 1024 * 1024 * 1024)) ]]; then
    log "WARN: /dev/shm is ${SHM_GB} GiB (< 8 GiB). Consider lower dataloader workers to avoid stalls."
  else
    log "INFO: /dev/shm is ${SHM_GB} GiB."
  fi
fi

log "SWIFT_ENTRY=${SWIFT_ENTRY}"
log "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
log "NPROC_PER_NODE=${NPROC_PER_NODE}"
log "MASTER_ADDR=${MASTER_ADDR}"
log "MASTER_PORT=${MASTER_PORT}"
log "RUN_DIR=${RUN_DIR}"
log "LOG_FILE=${LOG_FILE}"
log "NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-<unset>}"
log "GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-<unset>}"
log "TORCHELASTIC_USE_AGENT_STORE=${TORCHELASTIC_USE_AGENT_STORE}"
log "TORCH_NCCL_ASYNC_ERROR_HANDLING=${TORCH_NCCL_ASYNC_ERROR_HANDLING}"
log "NCCL_ASYNC_ERROR_HANDLING=${NCCL_ASYNC_ERROR_HANDLING}"
log "NCCL_BLOCKING_WAIT=${NCCL_BLOCKING_WAIT}"
log "NCCL_IB_DISABLE=${NCCL_IB_DISABLE}"
log "NCCL_DEBUG=${NCCL_DEBUG}"
log "TORCH_DISTRIBUTED_DEBUG=${TORCH_DISTRIBUTED_DEBUG}"
log "OMP_NUM_THREADS=${OMP_NUM_THREADS}"
log "MKL_NUM_THREADS=${MKL_NUM_THREADS}"
log "TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM}"
log "PYTHONFAULTHANDLER=${PYTHONFAULTHANDLER}"
log "HEARTBEAT_INTERVAL_SEC=${HEARTBEAT_INTERVAL_SEC}"
log "STALL_TIMEOUT_SEC=${STALL_TIMEOUT_SEC}"
{
  printf '[%s] FINAL_CMD=' "$(date '+%F %T')"
  printf '%q ' "${FINAL_CMD[@]}"
  printf '\n'
} | tee -a "${LOG_FILE}"

trap 'dump_runtime_state "signal-USR1"' USR1
trap 'if [[ -n "${TRAIN_PID:-}" ]] && kill -0 "${TRAIN_PID}" 2>/dev/null; then kill -TERM "${TRAIN_PID}" 2>/dev/null || true; fi' INT TERM

set +e
STARTED_TS="$(date +%s)"
"${FINAL_CMD[@]}" > >(tee -a "${LOG_FILE}") 2> >(tee -a "${LOG_FILE}" >&2) &
TRAIN_PID=$!
heartbeat_loop "${STARTED_TS}" &
HEARTBEAT_PID=$!
wait "${TRAIN_PID}"
TRAIN_RC=$?
kill "${HEARTBEAT_PID}" 2>/dev/null || true
wait "${HEARTBEAT_PID}" 2>/dev/null || true
set -e

if [[ "${TRAIN_RC}" -ne 0 ]]; then
  dump_runtime_state "train-exit-code-${TRAIN_RC}"
fi
exit "${TRAIN_RC}"
