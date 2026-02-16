#!/bin/bash
set -euo pipefail

cd /workspace/olmo/open-instruct
source ~/.venv/bin/activate
uv sync --active

LOG_DIR="/workspace/olmo/dpo_logs"
mkdir -p "$LOG_DIR"

train_dpo() {
    local DATASET_PATH="$1"
    local REFERENCE_CACHE="${2:-}"  # Optional: path to reference logprobs cache
    local CACHE_ONLY="${3:-}"       # Optional: if set, only cache reference logprobs

    local DATASET=$DATASET_PATH
    local NAME=$(basename "$DATASET_PATH")
    local OUTPUT_DIR="/workspace/olmo/dpo_checkpoints/olmo3_7b_instruct_dpo_${NAME}"
    local LOG_FILE="${LOG_DIR}/${NAME}.log"

    echo "Training with dataset: $DATASET"
    echo "Output dir: $OUTPUT_DIR"
    echo "Log file: $LOG_FILE"

    local EXTRA_ARGS=()
    if [[ -n "$REFERENCE_CACHE" ]]; then
        EXTRA_ARGS+=("--reference_logprobs_cache_path=$REFERENCE_CACHE")
        echo "Using reference cache: $REFERENCE_CACHE"
    fi
    if [[ -n "$CACHE_ONLY" ]]; then
        EXTRA_ARGS+=("--cache_reference_logprobs_only")
        echo "Cache-only mode: will exit after generating reference logprobs"
    fi

    accelerate launch \
        --mixed_precision bf16 \
        --num_machines 1 \
        --num_processes 8 \
        --use_deepspeed \
        --deepspeed_config_file configs/ds_configs/stage3_no_offloading_accelerate.conf \
        open_instruct/dpo_tune_cache.py \
        --mixer_list "${DATASET}" "1.0" \
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
        --num_epochs=1 \
        --logging_steps=1 \
        --loss_type=dpo_norm \
        --beta=5 \
        --use_flash_attn \
        --gradient_checkpointing \
        --push_to_hub=false \
        --do_not_randomize_output_dir=true \
        --with_tracking=true \
        --wandb_project="olmo3" \
        --wandb_entity="atticusw" \
        --try_launch_beaker_eval_jobs=false \
        "${EXTRA_ARGS[@]}" \
        2>&1 | tee "$LOG_FILE"
}


train_dpo "/workspace/olmo/dpo_filter_data/16K-baseline/dataset_autorated_filtered_1.jsonl" "/workspace/olmo/dpo_filter_data/16K-baseline/4cbafd709b2165c4.pt"
train_dpo "/workspace/olmo/dpo_filter_data/16K-baseline/dataset_autorated_filtered_2.jsonl" "/workspace/olmo/dpo_filter_data/16K-baseline/4cbafd709b2165c4.pt"


# # 16K filtered datasets
# train_dpo "/workspace/olmo/dpo_filter_data/16K-feedback-20.0pct-flip/dataset.jsonl" "/workspace/olmo/dpo_filter_data/16K-baseline/4cbafd709b2165c4.pt"
# # 33K filtered datasets
# train_dpo "/workspace/olmo/dpo_filter_data/33K-feedback-10.0pct-flip/dataset.jsonl" "/workspace/olmo/dpo_filter_data/33K-baseline/08e80dd1a8213080.pt"
