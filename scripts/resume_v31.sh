#!/bin/bash

# ============================================================================
# Resume Training Script for v31 (from checkpoint-2000)
# ============================================================================
#
# Max Steps 计算说明:
# -----------------
# 原始配置:
#   - NUMBER_OF_SAMPLES=20 (训练样本数)
#   - NUMBER_OF_PROMPTS=2 (每样本prompt数)
#   - PRL_NUM_TRAIN_EPOCHS=2
#   - PRL_PER_DEVICE_TRAIN_BATCH_SIZE=4
#   - PRL_NPROC_PER_NODE=3
#   - PRL_GRADIENT_ACCUMULATION_STEPS=1
#
# 计算:
#   total_samples = NUMBER_OF_SAMPLES * NUMBER_OF_PROMPTS = 20 * 2 = 40
#   global_batch_size = per_device_batch * nproc = 4 * 3 = 12
#   steps_per_epoch = ceil(total_samples / global_batch_size)
#                   = ceil(40 / 12) ≈ 4 (但实际数据集可能更大)
#
# 注意: 用户提供的 max_steps=2822 是基于实际运行日志推算的，
#       说明实际数据集规模远大于 NUMBER_OF_SAMPLES 参数暗示的规模。
#       这里直接采用用户提供的 2822 作为硬上限。
#
# Resume 策略:
#   - 从 checkpoint-2000 恢复
#   - max_steps=2822 锁死上限，防止过度训练
#   - save_steps=200 高频保存，防止再次丢失进度
# ============================================================================

set -e  # Exit on error
set -u  # Exit on undefined variable

# ============================================================================
# Dataset Configuration (与原运行完全一致)
# ============================================================================
export WANDB_MODE=offline
export DATASET=mr
export NUMBER_OF_SAMPLES=20
export NUMBER_OF_PROMPTS=2
export ADVERSARIAL=0
export REASONING=True

# ============================================================================
# Multi-GPU Configuration (与原运行完全一致: 6卡, 3进程)
# ============================================================================
PRL_CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
PRL_NPROC_PER_NODE=3
PRL_MASTER_PORT=29500

# Training batch configuration
PRL_PER_DEVICE_TRAIN_BATCH_SIZE=4
PRL_PER_DEVICE_EVAL_BATCH_SIZE=4
PRL_GRADIENT_ACCUMULATION_STEPS=1

# Generation configuration
PRL_NUM_GENERATIONS=2
PRL_NUM_INFER_WORKERS=3
PRL_VLLM_GPU_MEMORY_UTILIZATION=0.40

# Data loading configuration
PRL_DATALOADER_NUM_WORKERS=0
PRL_DATASET_NUM_PROC=8

# Model configuration
PRL_MAX_COMPLETION_LENGTH=384
PRL_MAX_LENGTH=2048
PRL_NUM_TRAIN_EPOCHS=2
PRL_LEARNING_RATE=1e-6
PRL_TEMPERATURE=0.9
PRL_SEED=5

# ============================================================================
# 修改项: 高频保存 + 同步 eval
# ============================================================================
PRL_EVAL_STEPS=200
PRL_SAVE_STEPS=200
PRL_LOGGING_STEPS=10

# ============================================================================
# Resume 专用配置
# ============================================================================
RESUME_CHECKPOINT="output/v31-20260210-185134/checkpoint-2000"
OUTPUT_DIR="output/v31-resume-final"
MAX_STEPS=2822  # 硬上限，防止从 step 2000 开始再跑完整 epoch

# ============================================================================
# Validation: Check checkpoint exists
# ============================================================================
if [ ! -d "${RESUME_CHECKPOINT}" ]; then
    echo "============================================================================"
    echo "❌ ERROR: Resume checkpoint not found!"
    echo "   Expected: ${RESUME_CHECKPOINT}"
    echo "============================================================================"
    exit 1
fi

# ============================================================================
# Validation: Check global_batch_size divisibility
# ============================================================================
GLOBAL_BATCH_SIZE=$((PRL_PER_DEVICE_TRAIN_BATCH_SIZE * PRL_NPROC_PER_NODE))

if [ $((GLOBAL_BATCH_SIZE % PRL_NUM_GENERATIONS)) -ne 0 ]; then
    echo "============================================================================"
    echo "❌ VALIDATION ERROR: global_batch_size must be divisible by num_generations"
    echo "============================================================================"
    echo "  global_batch_size = ${GLOBAL_BATCH_SIZE}"
    echo "  num_generations   = ${PRL_NUM_GENERATIONS}"
    echo "  ${GLOBAL_BATCH_SIZE} % ${PRL_NUM_GENERATIONS} = $((GLOBAL_BATCH_SIZE % PRL_NUM_GENERATIONS)) (must be 0)"
    echo "============================================================================"
    exit 1
fi

# ============================================================================
# Print Configuration
# ============================================================================
echo "============================================================================"
echo "Resume Training Configuration (v31 from checkpoint-2000)"
echo "============================================================================"
echo ""
echo ">>> RESUME SETTINGS <<<"
echo "  RESUME_CHECKPOINT:                ${RESUME_CHECKPOINT}"
echo "  OUTPUT_DIR:                       ${OUTPUT_DIR}"
echo "  MAX_STEPS (hard limit):           ${MAX_STEPS}"
echo ""
echo "GPU Configuration:"
echo "  CUDA_VISIBLE_DEVICES:             ${PRL_CUDA_VISIBLE_DEVICES}"
echo "  NPROC_PER_NODE:                   ${PRL_NPROC_PER_NODE}"
echo "  MASTER_PORT:                      ${PRL_MASTER_PORT}"
echo ""
echo "Batch Configuration:"
echo "  PER_DEVICE_TRAIN_BATCH_SIZE:      ${PRL_PER_DEVICE_TRAIN_BATCH_SIZE}"
echo "  PER_DEVICE_EVAL_BATCH_SIZE:       ${PRL_PER_DEVICE_EVAL_BATCH_SIZE}"
echo "  GRADIENT_ACCUMULATION_STEPS:      ${PRL_GRADIENT_ACCUMULATION_STEPS}"
echo "  GLOBAL_BATCH_SIZE (computed):     ${GLOBAL_BATCH_SIZE}"
echo ""
echo "Generation Configuration:"
echo "  NUM_GENERATIONS:                  ${PRL_NUM_GENERATIONS}"
echo "  NUM_INFER_WORKERS:                ${PRL_NUM_INFER_WORKERS}"
echo "  VLLM_GPU_MEMORY_UTILIZATION:      ${PRL_VLLM_GPU_MEMORY_UTILIZATION}"
echo ""
echo "Checkpoint Configuration (FIXED):"
echo "  EVAL_STEPS:                       ${PRL_EVAL_STEPS}"
echo "  SAVE_STEPS:                       ${PRL_SAVE_STEPS}"
echo "  LOGGING_STEPS:                    ${PRL_LOGGING_STEPS}"
echo ""
echo "Dataset Configuration:"
echo "  DATASET:                          ${DATASET}"
echo "  NUMBER_OF_SAMPLES:                ${NUMBER_OF_SAMPLES}"
echo "  NUMBER_OF_PROMPTS:                ${NUMBER_OF_PROMPTS}"
echo "============================================================================"
echo ""
echo "✓ Validation passed: global_batch_size (${GLOBAL_BATCH_SIZE}) % num_generations (${PRL_NUM_GENERATIONS}) = 0"
echo "✓ Checkpoint exists: ${RESUME_CHECKPOINT}"
echo ""
echo "Starting resume training in 3 seconds..."
sleep 3

# ============================================================================
# Launch Resume Training with torchrun
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
    --max_steps ${MAX_STEPS} \
    --per_device_train_batch_size ${PRL_PER_DEVICE_TRAIN_BATCH_SIZE} \
    --per_device_eval_batch_size ${PRL_PER_DEVICE_EVAL_BATCH_SIZE} \
    --learning_rate ${PRL_LEARNING_RATE} \
    --gradient_accumulation_steps ${PRL_GRADIENT_ACCUMULATION_STEPS} \
    --eval_steps ${PRL_EVAL_STEPS} \
    --save_steps ${PRL_SAVE_STEPS} \
    --save_total_limit 20 \
    --max_length ${PRL_MAX_LENGTH} \
    --output_dir ${OUTPUT_DIR} \
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
    --num_infer_workers ${PRL_NUM_INFER_WORKERS} \
    --resume_from_checkpoint ${RESUME_CHECKPOINT}

echo ""
echo "============================================================================"
echo "Resume training started from step 2000..."
echo "Target: max_steps=${MAX_STEPS}, saving every ${PRL_SAVE_STEPS} steps"
echo "Output: ${OUTPUT_DIR}"
echo "============================================================================"
