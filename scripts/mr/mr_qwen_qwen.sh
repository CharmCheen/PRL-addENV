export DATASET=mr
export NUMBER_OF_SAMPLES=100
export NUMBER_OF_PROMPTS=10
export ADVERSARIAL=0
export REASONING=True

# Single GPU setup - use direct python call instead of torch.distributed.run
# Lower vLLM GPU memory utilization to avoid OOM (training model + vLLM on same GPU)
CUDA_VISIBLE_DEVICES=0 \
swift rlhf \
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
    --vllm_gpu_memory_utilization 0.5 \
    --train_type lora \
    --lora_rank 8 \
    --lora_alpha 32 \
    --seed 5 \
    --max_completion_length 1024 \
    --num_train_epochs 10 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --learning_rate 1e-6 \
    --gradient_accumulation_steps 1 \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 20 \
    --max_length 2048 \
    --output_dir output \
    --warmup_ratio 0 \
    --dataloader_num_workers 0 \
    --dataset_num_proc 4 \
    --num_generations 4 \
    --temperature 0.9 \
    --report_to wandb \
    --logging_steps 5 \
    --system 'examples/train/grpo/prompt.txt' \
    --log_completions true \
    --num_iterations 1 \
    --num_infer_workers 1
