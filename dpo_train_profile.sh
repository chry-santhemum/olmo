#!/bin/bash
set -euo pipefail

DATASET="${1:?Usage: $0 <dataset_path> [output_dir]}"
# Convert to absolute path before changing directory
[[ -f "$DATASET" ]] && DATASET="$(realpath "$DATASET")"

cd /workspace/olmo/open-instruct
source ~/.venv/bin/activate
uv sync --active
OUTPUT_DIR="${2:-/workspace/olmo/dpo_train_profile}"

accelerate launch \
    --mixed_precision bf16 \
    --num_machines 1 \
    --num_processes 1 \
    --use_deepspeed \
    --deepspeed_config_file configs/ds_configs/stage3_no_offloading_accelerate.conf \
    open_instruct/dpo_tune_cache.py \
    --dataset_mixer_list "${DATASET}" "1.0" \
    --output_dir="$OUTPUT_DIR" \
    --model_name_or_path="allenai/Olmo-3-7B-Instruct-SFT" \
    --tokenizer_name="allenai/Olmo-3-7B-Instruct-SFT" \
    --max_seq_length=2048 \
    --per_device_train_batch_size=1 \
    --gradient_accumulation_steps=8 \
    --learning_rate=1e-6 \
    --lr_scheduler_type=linear \
    --warmup_ratio=0.1 \
    --weight_decay=0.0 \
    --max_train_steps=6 \
    --num_train_epochs=1 \
    --logging_steps=1 \
    --dpo_loss_type=dpo_norm \
    --dpo_beta=5 \
    --use_flash_attn \
    --gradient_checkpointing \
    --do_not_randomize_output_dir=true \
    --profile=true \
    --profile_steps=2 \
    --with_tracking=false \
    --push_to_hub=false \
    --try_launch_beaker_eval_jobs=false \
    --skip_model_save=true
