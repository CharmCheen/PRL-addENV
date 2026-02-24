#!/bin/bash

# ============================================================================
# Multi-GPU Training Script for GRPO (MR Dataset)
# ============================================================================
# This script is a NEW ADDITION for multi-GPU training (2/4/8 GPUs).
# The original single-GPU script (mr_qwen_qwen.sh) remains UNCHANGED.
#
# Design for platform non-interactive submission:
# - All parameters configurable via PRL_* environment variables
# - Fail-fast validation before training starts
# - No device_map pinning (allows distributed training)
# - Conservative defaults for stability
# ============================================================================

set -e  # Exit on error
set -u  # Exit on undefined variable

# ============================================================================
# Dataset Configuration (same as single-GPU version)
# ============================================================================
export DATASET=${DATASET:-mr}
export NUMBER_OF_SAMPLES=${NUMBER_OF_SAMPLES:-100}
export NUMBER_OF_PROMPTS=${NUMBER_OF_PROMPTS:-10}
export ADVERSARIAL=${ADVERSARIAL:-0}
export REASONING=${REASONING:-True}

# ============================================================================
# Multi-GPU Configuration with PRL_* Environment Variable Overrides
# ============================================================================

# GPU configuration
PRL_CUDA_VISIBLE_DEVICES=${PRL_CUDA_VISIBLE_DEVICES:-0,1,2,3}

count_visible_gpus() {
  local src="${PRL_CUDA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-}}"
  if [[ -n "${src}" ]]; then
    local cleaned
    cleaned="$(echo "${src}" | tr -d ' ')"
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
PRL_NPROC_PER_NODE=${PRL_NPROC_PER_NODE:-${VISIBLE_GPUS}}
if ! [[ "${PRL_NPROC_PER_NODE}" =~ ^[0-9]+$ ]] || [[ "${PRL_NPROC_PER_NODE}" -lt 1 ]]; then
  echo "[ERROR] PRL_NPROC_PER_NODE must be positive integer, got '${PRL_NPROC_PER_NODE}'"
  exit 1
fi
if [[ "${PRL_NPROC_PER_NODE}" -gt "${VISIBLE_GPUS}" ]]; then
  echo "[WARN] PRL_NPROC_PER_NODE=${PRL_NPROC_PER_NODE} > visible_gpus=${VISIBLE_GPUS}; auto-downgrade."
  PRL_NPROC_PER_NODE="${VISIBLE_GPUS}"
fi
PRL_MASTER_PORT=${PRL_MASTER_PORT:-$(choose_master_port)}

# Training batch configuration
PRL_PER_DEVICE_TRAIN_BATCH_SIZE=${PRL_PER_DEVICE_TRAIN_BATCH_SIZE:-4}
PRL_PER_DEVICE_EVAL_BATCH_SIZE=${PRL_PER_DEVICE_EVAL_BATCH_SIZE:-4}
PRL_GRADIENT_ACCUMULATION_STEPS=${PRL_GRADIENT_ACCUMULATION_STEPS:-1}

# Generation configuration
PRL_NUM_GENERATIONS=${PRL_NUM_GENERATIONS:-2}
PRL_NUM_INFER_WORKERS=${PRL_NUM_INFER_WORKERS:-4}
PRL_VLLM_GPU_MEMORY_UTILIZATION=${PRL_VLLM_GPU_MEMORY_UTILIZATION:-0.40}

# Data loading configuration
# NOTE: dataloader_num_workers=0 is safer for platform non-interactive mode
# to avoid multiprocessing issues. Can override with PRL_DATALOADER_NUM_WORKERS=4
# if your platform supports it.
PRL_DATALOADER_NUM_WORKERS=${PRL_DATALOADER_NUM_WORKERS:-0}
PRL_DATASET_NUM_PROC=${PRL_DATASET_NUM_PROC:-8}
if ! [[ "${PRL_DATALOADER_NUM_WORKERS}" =~ ^[0-9]+$ ]]; then
  echo "[ERROR] PRL_DATALOADER_NUM_WORKERS must be non-negative integer, got '${PRL_DATALOADER_NUM_WORKERS}'"
  exit 1
fi
if df -k /dev/shm >/dev/null 2>&1; then
  SHM_KB="$(df -k /dev/shm | awk 'NR==2 {print $2}')"
  if [[ -n "${SHM_KB}" ]] && [[ "${SHM_KB}" -lt 8388608 ]] && [[ "${PRL_DATALOADER_NUM_WORKERS}" -gt 2 ]]; then
    echo "[WARN] /dev/shm < 8GB; cap PRL_DATALOADER_NUM_WORKERS to 2 (was ${PRL_DATALOADER_NUM_WORKERS})."
    PRL_DATALOADER_NUM_WORKERS=2
  fi
fi

# Model configuration
PRL_MAX_COMPLETION_LENGTH=${PRL_MAX_COMPLETION_LENGTH:-768}
PRL_EVAL_STEPS=${PRL_EVAL_STEPS:-200}
PRL_SAVE_STEPS=${PRL_SAVE_STEPS:-200}
PRL_LOGGING_STEPS=${PRL_LOGGING_STEPS:-10}

# Advanced configuration (usually no need to change)
PRL_MAX_LENGTH=${PRL_MAX_LENGTH:-2048}
PRL_NUM_TRAIN_EPOCHS=${PRL_NUM_TRAIN_EPOCHS:-10}
PRL_LEARNING_RATE=${PRL_LEARNING_RATE:-1e-6}
PRL_TEMPERATURE=${PRL_TEMPERATURE:-0.9}
PRL_SEED=${PRL_SEED:-5}

# ============================================================================
# Validation: Check global_batch_size divisibility
# ============================================================================

GLOBAL_BATCH_SIZE=$((PRL_PER_DEVICE_TRAIN_BATCH_SIZE * PRL_NPROC_PER_NODE))

if [ $((GLOBAL_BATCH_SIZE % PRL_NUM_GENERATIONS)) -ne 0 ]; then
    echo "============================================================================"
    echo "❌ VALIDATION ERROR: global_batch_size must be divisible by num_generations"
    echo "============================================================================"
    echo "  global_batch_size = per_device_train_batch_size * nproc_per_node"
    echo "                    = ${PRL_PER_DEVICE_TRAIN_BATCH_SIZE} * ${PRL_NPROC_PER_NODE}"
    echo "                    = ${GLOBAL_BATCH_SIZE}"
    echo ""
    echo "  num_generations   = ${PRL_NUM_GENERATIONS}"
    echo ""
    echo "  ${GLOBAL_BATCH_SIZE} % ${PRL_NUM_GENERATIONS} = $((GLOBAL_BATCH_SIZE % PRL_NUM_GENERATIONS)) (must be 0)"
    echo ""
    echo "💡 Fix: Adjust PRL_PER_DEVICE_TRAIN_BATCH_SIZE or PRL_NUM_GENERATIONS"
    echo "   Example valid combinations:"
    echo "   - 4 GPUs: batch_size=4, num_generations=2 → global=16 ✓"
    echo "   - 4 GPUs: batch_size=4, num_generations=4 → global=16 ✓"
    echo "   - 4 GPUs: batch_size=2, num_generations=2 → global=8 ✓"
    echo "   - 2 GPUs: batch_size=4, num_generations=2 → global=8 ✓"
    echo "============================================================================"
    exit 1
fi

# ============================================================================
# Print Final Configuration (for platform logs)
# ============================================================================

echo "============================================================================"
echo "Multi-GPU Training Configuration (Platform Non-Interactive Mode)"
echo "============================================================================"
echo "GPU Configuration:"
echo "  PRL_CUDA_VISIBLE_DEVICES:         ${PRL_CUDA_VISIBLE_DEVICES}"
echo "  PRL_NPROC_PER_NODE:               ${PRL_NPROC_PER_NODE}"
echo "  PRL_MASTER_PORT:                  ${PRL_MASTER_PORT}"
echo ""
echo "Batch Configuration:"
echo "  PRL_PER_DEVICE_TRAIN_BATCH_SIZE:  ${PRL_PER_DEVICE_TRAIN_BATCH_SIZE}"
echo "  PRL_PER_DEVICE_EVAL_BATCH_SIZE:   ${PRL_PER_DEVICE_EVAL_BATCH_SIZE}"
echo "  PRL_GRADIENT_ACCUMULATION_STEPS:  ${PRL_GRADIENT_ACCUMULATION_STEPS}"
echo "  GLOBAL_BATCH_SIZE (computed):     ${GLOBAL_BATCH_SIZE}"
echo ""
echo "Generation Configuration:"
echo "  PRL_NUM_GENERATIONS:              ${PRL_NUM_GENERATIONS}"
echo "  PRL_NUM_INFER_WORKERS:            ${PRL_NUM_INFER_WORKERS}"
echo "  PRL_VLLM_GPU_MEMORY_UTILIZATION:  ${PRL_VLLM_GPU_MEMORY_UTILIZATION}"
echo ""
echo "Data Loading Configuration:"
echo "  PRL_DATALOADER_NUM_WORKERS:       ${PRL_DATALOADER_NUM_WORKERS}"
echo "  PRL_DATASET_NUM_PROC:             ${PRL_DATASET_NUM_PROC}"
echo ""
echo "Model Configuration:"
echo "  PRL_MAX_COMPLETION_LENGTH:        ${PRL_MAX_COMPLETION_LENGTH}"
echo "  PRL_MAX_LENGTH:                   ${PRL_MAX_LENGTH}"
echo "  PRL_NUM_TRAIN_EPOCHS:             ${PRL_NUM_TRAIN_EPOCHS}"
echo "  PRL_LEARNING_RATE:                ${PRL_LEARNING_RATE}"
echo "  PRL_TEMPERATURE:                  ${PRL_TEMPERATURE}"
echo "  PRL_SEED:                         ${PRL_SEED}"
echo ""
echo "Checkpoint Configuration:"
echo "  PRL_EVAL_STEPS:                   ${PRL_EVAL_STEPS}"
echo "  PRL_SAVE_STEPS:                   ${PRL_SAVE_STEPS}"
echo "  PRL_LOGGING_STEPS:                ${PRL_LOGGING_STEPS}"
echo ""
echo "Dataset Configuration:"
echo "  DATASET:                          ${DATASET}"
echo "  NUMBER_OF_SAMPLES:                ${NUMBER_OF_SAMPLES}"
echo "  NUMBER_OF_PROMPTS:                ${NUMBER_OF_PROMPTS}"
echo "============================================================================"
echo ""
echo "✓ Validation passed: global_batch_size (${GLOBAL_BATCH_SIZE}) % num_generations (${PRL_NUM_GENERATIONS}) = 0"
echo ""
echo "Starting training in 3 seconds..."
sleep 3

# ============================================================================
# Launch Multi-GPU Training with torchrun
# ============================================================================

CUDA_VISIBLE_DEVICES=${PRL_CUDA_VISIBLE_DEVICES} \
torchrun \
    --nproc_per_node=${PRL_NPROC_PER_NODE} \
    --master_port=${PRL_MASTER_PORT} \
    -m swift.cli.rlhf \
    --rlhf_type grpo \
    --model Qwen/Qwen2.5-7B-Instruct \
    --model_type qwen2_5 \
    --dataset datasets/original/${DATASET}_train.jsonl  \
    --val_dataset datasets/original/${DATASET}_val.jsonl \
    --reward_funcs accuracy format \
    --torch_dtype bfloat16 \
    --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
    --use_lmdeploy false \
    --use_vllm true \
    --vllm_gpu_memory_utilization ${PRL_VLLM_GPU_MEMORY_UTILIZATION} \
    --train_type lora \
    --lora_rank 8 \
    --lora_alpha 32 \
    --seed ${PRL_SEED} \
    --max_completion_length ${PRL_MAX_COMPLETION_LENGTH} \
    --num_train_epochs ${PRL_NUM_TRAIN_EPOCHS} \
    --per_device_train_batch_size ${PRL_PER_DEVICE_TRAIN_BATCH_SIZE} \
    --per_device_eval_batch_size ${PRL_PER_DEVICE_EVAL_BATCH_SIZE} \
    --learning_rate ${PRL_LEARNING_RATE} \
    --gradient_accumulation_steps ${PRL_GRADIENT_ACCUMULATION_STEPS} \
    --eval_steps ${PRL_EVAL_STEPS} \
    --save_steps ${PRL_SAVE_STEPS} \
    --save_total_limit 20 \
    --max_length ${PRL_MAX_LENGTH} \
    --output_dir output \
    --warmup_ratio 0 \
    --dataloader_num_workers ${PRL_DATALOADER_NUM_WORKERS} \
    --dataset_num_proc ${PRL_DATASET_NUM_PROC} \
    --num_generations ${PRL_NUM_GENERATIONS} \
    --temperature ${PRL_TEMPERATURE} \
    --report_to wandb \
    --logging_steps ${PRL_LOGGING_STEPS} \
    --system 'examples/train/grpo/prompt.txt' \
    --log_completions true \
    --num_iterations 1 \
    --num_infer_workers ${PRL_NUM_INFER_WORKERS}
