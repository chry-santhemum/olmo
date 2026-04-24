#!/bin/bash
set -euo pipefail

cd /workspace/olmo/open-instruct
source ~/.venv/bin/activate
uv sync --active

LOG_DIR="/workspace/olmo/dpo_logs"
mkdir -p "$LOG_DIR"

# Edit these and rerun. Each dataset is trained sequentially.
REFERENCE_CACHE="/workspace/olmo/dpo_checkpoints/33K_reference_logprobs/5a075abbabd49981.pt"
# REFERENCE_CACHE=""
DATASETS=(
    # "/workspace/olmo/filtered/16K-add-256"
    # "/workspace/olmo/filtered/16K-add-1024"
    # "/workspace/olmo/filtered/16K-autorater-discard-2"
    # "/workspace/olmo/filtered/16K-autorater-flip-1"
    # "/workspace/olmo/filtered/16K-autorater-flip-2"
    "/workspace/olmo/filtered/16K-add-2048"
    "/workspace/olmo/filtered/16K-add-512"
)
# DEEPSPEED_CONFIG="configs/ds_configs/stage3_no_offloading_accelerate.conf"
DEEPSPEED_CONFIG="configs/ds_configs/stage3_offloading_accelerate.conf"

train_dpo() {
    local DATASET_PATH="$1"

    if [[ -d "$DATASET_PATH" ]]; then
        DATASET_PATH="${DATASET_PATH}/dataset.jsonl"
    fi
    if [[ ! -f "$DATASET_PATH" ]]; then
        echo "Dataset file not found: $DATASET_PATH" >&2
        exit 1
    fi

    local DATASET_DIR
    DATASET_DIR=$(dirname "$DATASET_PATH")
    local NAME
    NAME=$(basename "$DATASET_DIR")
    local OUTPUT_DIR="/workspace/olmo/dpo_checkpoints/olmo3_7b_instruct_dpo_${NAME}"
    local LOG_FILE="${LOG_DIR}/${NAME}.log"

    echo "Training with dataset: $DATASET_PATH"
    echo "Output dir: $OUTPUT_DIR"
    echo "Log file: $LOG_FILE"

    local EXTRA_ARGS=()
    if [[ -n "$REFERENCE_CACHE" ]]; then
        EXTRA_ARGS+=("--reference_logprobs_cache_path=$REFERENCE_CACHE")
        echo "Using reference cache: $REFERENCE_CACHE"
    fi

    accelerate launch \
        --mixed_precision bf16 \
        --num_machines 1 \
        --num_processes 4 \
        --use_deepspeed \
        --deepspeed_config_file "$DEEPSPEED_CONFIG" \
        open_instruct/dpo_tune_cache.py \
        --mixer_list "$DATASET_PATH" "1.0" \
        --output_dir="$OUTPUT_DIR" \
        --exp_name="dpo_filter_${NAME}" \
        --model_name_or_path="allenai/Olmo-3-7B-Instruct-SFT" \
        --tokenizer_name="allenai/Olmo-3-7B-Instruct-SFT" \
        --max_seq_length=16384 \
        --per_device_train_batch_size=1 \
        --gradient_accumulation_steps=32 \
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

for DATASET_PATH in "${DATASETS[@]}"; do
    train_dpo "$DATASET_PATH"
done
