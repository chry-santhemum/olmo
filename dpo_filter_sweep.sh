#!/bin/bash
set -euo pipefail

cd /workspace/olmo
source setup.sh

# # Filter data with different prune percentages
# python3 dpo_filter_data.py

# train
cd /workspace/olmo/open-instruct
source ~/.venv/bin/activate
uv sync --active

DATASETS=(
    # "/workspace/olmo/dpo_filter_data/33K-baseline/dataset.jsonl"
    # "/workspace/olmo/dpo_filter_data/33K-persona-5.0pct-prune-cosine/dataset.jsonl"
    # "/workspace/olmo/dpo_filter_data/33K-persona-1.0pct-prune-cosine/dataset.jsonl"
    # "/workspace/olmo/dpo_filter_data/33K-persona-0.25pct-prune-cosine/dataset.jsonl"
    # "/workspace/olmo/dpo_filter_data/33K-persona-1.0pct-flip-cosine/dataset.jsonl"
    "/workspace/olmo/dpo_filter_data/33K-persona-1.0pct-prune-dot/dataset.jsonl"
    "/workspace/olmo/dpo_filter_data/33K-feedback-1.0pct-prune-cosine/dataset.jsonl"
)

for DATASET in "${DATASETS[@]}"; do
    # Extract name for output dir (e.g., "33K-5pct" from path)
    NAME=$(basename "$(dirname "$DATASET")")
    OUTPUT_DIR="/workspace/olmo/dpo_checkpoints/olmo3_7b_instruct_dpo_${NAME}"

    LOG_FILE="/workspace/olmo/dpo_filter_sweep_logs/${NAME}.log"
    mkdir -p /workspace/olmo/dpo_filter_sweep_logs

    echo "Training with dataset: $DATASET"
    echo "Output dir: $OUTPUT_DIR"
    echo "Log file: $LOG_FILE"

    accelerate launch \
        --mixed_precision bf16 \
        --num_machines 1 \
        --num_processes 8 \
        --use_deepspeed \
        --deepspeed_config_file configs/ds_configs/stage3_no_offloading_accelerate.conf \
        open_instruct/dpo_tune_cache.py \
        --dataset_mixer_list "${DATASET}" "1.0" \
        --output_dir="$OUTPUT_DIR" \
        --exp_name="dpo_filter_${NAME}" \
        --model_name_or_path="allenai/Olmo-3-7B-Instruct-SFT" \
        --tokenizer_name="allenai/Olmo-3-7B-Instruct-SFT" \
        --max_seq_length=16384 \
        --per_device_train_batch_size=1 \
        --gradient_accumulation_steps=16 \
        --learning_rate=1e-6 \
        --lr_scheduler_type=linear \
        --warmup_ratio=0.1 \
        --weight_decay=0.0 \
        --num_train_epochs=1 \
        --logging_steps=1 \
        --dpo_loss_type=dpo_norm \
        --dpo_beta=5 \
        --use_flash_attn \
        --gradient_checkpointing \
        --push_to_hub=false \
        --do_not_randomize_output_dir=true \
        --reference_logprobs_cache_path="/workspace/olmo/dpo_filter_data/33K-baseline/reference_logprobs.pt" \
        --with_tracking=true \
        --wandb_project_name="olmo3" \
        --wandb_entity="atticusw" \
        --try_launch_beaker_eval_jobs=false \
        2>&1 | tee "$LOG_FILE"
done
