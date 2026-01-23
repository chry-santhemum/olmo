#!/bin/bash
set -euo pipefail

# Parse arguments
GROUP=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --group)
            GROUP="$2"
            shift 2
            ;;
        *)
            echo "Usage: $0 --group <1|2>"
            exit 1
            ;;
    esac
done

if [[ "$GROUP" != "1" && "$GROUP" != "2" ]]; then
    echo "Usage: $0 --group <1|2>"
    exit 1
fi


cd /workspace/olmo/open-instruct
source ~/.venv/bin/activate
uv sync --active

# Baseline dataset (for generating reference logprobs cache)
BASELINE_DIR="/workspace/olmo/dpo_filter_data/33K-baseline"

# Split datasets into two groups
if [[ "$GROUP" == "1" ]]; then
    FILTERED_DATASETS=(
        "/workspace/olmo/dpo_filter_data/16K-persona-50.0pct-prune"
        "/workspace/olmo/dpo_filter_data/16K-persona-20.0pct-prune"
        "/workspace/olmo/dpo_filter_data/16K-persona-10.0pct-prune"
        "/workspace/olmo/dpo_filter_data/16K-persona-5.0pct-prune"
    )
else
    FILTERED_DATASETS=(
        "/workspace/olmo/dpo_filter_data/16K-persona-50.0pct-flip"
        "/workspace/olmo/dpo_filter_data/16K-persona-20.0pct-flip"
        "/workspace/olmo/dpo_filter_data/16K-persona-10.0pct-flip"
        "/workspace/olmo/dpo_filter_data/16K-persona-5.0pct-flip"
    )
fi

LOG_DIR="/workspace/olmo/dpo_filter_sweep_logs"
mkdir -p "$LOG_DIR"

train_dpo() {
    local DATASET_DIR="$1"
    local REFERENCE_CACHE="${2:-}"  # Optional: path to reference logprobs cache
    local CACHE_ONLY="${3:-}"       # Optional: if set, only cache reference logprobs

    local DATASET="${DATASET_DIR}/dataset.jsonl"
    local NAME=$(basename "$DATASET_DIR")
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
        --num_processes 4 \
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

CACHE_PATTERN="$BASELINE_DIR/*.pt"

if [[ "$GROUP" == "1" ]]; then
    # Group 1: First generate reference logprobs cache only (no training)
    echo "=== Group 1: Generating reference logprobs cache ==="
    REFERENCE_LOGPROBS_CACHE_PATH="$BASELINE_DIR" train_dpo "$BASELINE_DIR" "" "cache_only"
else
    # Group 2: Wait for cache file to exist
    echo "=== Group 2: Waiting for reference logprobs cache ==="
    while ! ls $CACHE_PATTERN 1>/dev/null 2>&1; do
        echo "Waiting for reference cache at $CACHE_PATTERN ..."
        sleep 30
    done
fi

# Find the cache file
REFERENCE_CACHE=$(ls $CACHE_PATTERN 2>/dev/null | head -1)
if [[ -z "$REFERENCE_CACHE" ]]; then
    echo "ERROR: No reference logprobs cache found in $BASELINE_DIR"
    exit 1
fi
echo "Found reference cache: $REFERENCE_CACHE"

# Train filtered datasets using the cache
echo "=== Group $GROUP: Training filtered datasets ==="
for DATASET_DIR in "${FILTERED_DATASETS[@]}"; do
    train_dpo "$DATASET_DIR" "$REFERENCE_CACHE"
done

echo "=== Group $GROUP training complete ==="
