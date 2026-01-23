#!/bin/bash
set -euo pipefail

cd /workspace/olmo
source setup.sh

# Generate filtered datasets
# python3 dpo_filter_data.py

cd /workspace/olmo/open-instruct
source ~/.venv/bin/activate
uv sync --active

# Baseline dataset (for generating reference logprobs cache)
BASELINE_DIR="/workspace/olmo/dpo_filter_data/16K-baseline"

# Filtered datasets to train
FILTERED_DATASETS=(
    "/workspace/olmo/dpo_filter_data/16K-persona-50.0pct-prune"
    "/workspace/olmo/dpo_filter_data/16K-persona-20.0pct-prune"
    "/workspace/olmo/dpo_filter_data/16K-persona-10.0pct-prune"
    "/workspace/olmo/dpo_filter_data/16K-persona-5.0pct-prune"
    "/workspace/olmo/dpo_filter_data/16K-persona-50.0pct-flip"
    "/workspace/olmo/dpo_filter_data/16K-persona-20.0pct-flip"
    "/workspace/olmo/dpo_filter_data/16K-persona-10.0pct-flip"
    "/workspace/olmo/dpo_filter_data/16K-persona-5.0pct-flip"
)

LOG_DIR="/workspace/olmo/dpo_filter_sweep_logs"
mkdir -p "$LOG_DIR"

train_dpo() {
    local DATASET_DIR="$1"
    local REFERENCE_CACHE="${2:-}"  # Optional: path to reference logprobs cache

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
        --with_tracking=true \
        --wandb_project_name="olmo3" \
        --wandb_entity="atticusw" \
        --try_launch_beaker_eval_jobs=false \
        "${EXTRA_ARGS[@]}" \
        2>&1 | tee "$LOG_FILE"
}

# Step 1: Train baseline (this computes and saves reference logprobs)
echo "=== Step 1: Training baseline to generate reference logprobs cache ==="
REFERENCE_LOGPROBS_CACHE_PATH="$BASELINE_DIR" train_dpo "$BASELINE_DIR"

# Find the generated cache file (*.pt in baseline dir, excluding dataset files)
REFERENCE_CACHE=$(find "$BASELINE_DIR" -maxdepth 1 -name "*.pt" -type f | head -1)
if [[ -z "$REFERENCE_CACHE" ]]; then
    echo "ERROR: No reference logprobs cache found in $BASELINE_DIR"
    exit 1
fi
echo "Found reference cache: $REFERENCE_CACHE"

# Step 2: Train filtered datasets using the baseline cache
echo "=== Step 2: Training filtered datasets ==="
for DATASET_DIR in "${FILTERED_DATASETS[@]}"; do
    train_dpo "$DATASET_DIR" "$REFERENCE_CACHE"
done

echo "=== All training complete ==="
