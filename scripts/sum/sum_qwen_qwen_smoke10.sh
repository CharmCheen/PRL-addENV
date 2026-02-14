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

if [[ -z "${DATASET:-}" ]]; then
  echo "[ERROR] DATASET is empty."
  echo "Usage: DATASET=sum [CUDA_VISIBLE_DEVICES=0,1] [NPROC_PER_NODE=2] bash scripts/sum/sum_qwen_qwen_smoke10.sh"
  exit 1
fi

export NUMBER_OF_SAMPLES="${NUMBER_OF_SAMPLES:-10}"
export NUMBER_OF_PROMPTS="${NUMBER_OF_PROMPTS:-3}"
export ADVERSARIAL="${ADVERSARIAL:-0}"
export REASONING="${REASONING:-True}"

BASE_OUTPUT_DIR="/qiuyeqing/llama_prl/PRL-REDO/PRL-addENV/output/sum"
TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
RUN_DIR="${BASE_OUTPUT_DIR}/sum-qwen-qwen-${TIMESTAMP}"
LOG_FILE="${RUN_DIR}/sum.log"
mkdir -p "${BASE_OUTPUT_DIR}"
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

SMOKE_STEPS="${SMOKE_STEPS:-10}"
USE_LMDEPLOY="${USE_LMDEPLOY:-0}"
if [[ "${USE_LMDEPLOY}" == "1" ]]; then
  USE_LMDEPLOY_ARG="true"
else
  USE_LMDEPLOY_ARG="false"
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
  --seed 5
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
  --dataloader_num_workers 1
  --dataset_num_proc 4
  --num_generations 4
  --temperature 0.9
  --report_to wandb
  --logging_steps 5
  --system examples/train/grpo/prompt.txt
  --log_completions true
  --num_iterations 1
  --num_infer_workers 1
)

FINAL_CMD+=(--max_steps "${SMOKE_STEPS}" --logging_steps 1 --eval_steps 1000000 --save_steps 1000000)

echo "SWIFT_ENTRY=${SWIFT_ENTRY}"
echo "BASE_OUTPUT_DIR=${BASE_OUTPUT_DIR}"
echo "RUN_DIR=${RUN_DIR}"
echo "LOG_FILE=${LOG_FILE}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "NPROC_PER_NODE=${NPROC_PER_NODE}"
echo "MASTER_ADDR=${MASTER_ADDR}"
echo "MASTER_PORT=${MASTER_PORT}"
echo "NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME}"
echo "TORCHELASTIC_USE_AGENT_STORE=${TORCHELASTIC_USE_AGENT_STORE}"
echo "USE_LMDEPLOY=${USE_LMDEPLOY_ARG}"
echo -n "FINAL_CMD="
printf '%q ' "${FINAL_CMD[@]}"
echo

"${FINAL_CMD[@]}" 2>&1 | tee "${LOG_FILE}"

MAX_STEP="$(grep -Eiv 'FINAL_CMD|max_steps|eval_steps|save_steps|logging_steps' "${LOG_FILE}" \
  | grep -Eio '(global[_ ]?step|step)[^0-9]{0,12}[0-9]+' \
  | grep -Eo '[0-9]+' \
  | sort -n \
  | tail -1 || true)"
if [[ -z "${MAX_STEP}" ]]; then
  MAX_STEP="0"
fi

if [[ "${MAX_STEP}" -ge "${SMOKE_STEPS}" ]]; then
  echo "[OK] reached >=10 steps (max_observed_step=${MAX_STEP})"
  exit 0
fi

echo "[ERROR] did not reach >=10 steps (max_observed_step=${MAX_STEP})"
echo "[ERROR] check log: ${LOG_FILE}"
exit 1
