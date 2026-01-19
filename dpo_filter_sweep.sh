#!/bin/bash
set -euo pipefail

cd /workspace/olmo
source setup.sh

# Filter data with different prune percentages
python3 dpo_filter_data.py \
    allenai/Olmo-3-7B-Instruct-SFT \
    sycophantic \
    --num_samples 32768 \
    --chunk_size 256 \
    --prune_top_pct 5.0 \
    --output_dataset_path dpo_filter_data/33K-5pct.jsonl

python3 dpo_filter_data.py \
    allenai/Olmo-3-7B-Instruct-SFT \
    sycophantic \
    --num_samples 32768 \
    --chunk_size 256 \
    --prune_top_pct 1.0 \
    --output_dataset_path dpo_filter_data/33K-1pct.jsonl

python3 dpo_filter_data.py \
    allenai/Olmo-3-7B-Instruct-SFT \
    sycophantic \
    --num_samples 32768 \
    --chunk_size 256 \
    --prune_top_pct 0.25 \
    --output_dataset_path dpo_filter_data/33K-0.25pct.jsonl



# train
# cd /workspace/olmo/open-instruct
# source .venv/bin/activate
# wandb login
# hf auth login

DATASETS=(
    "/workspace/olmo/dpo_filter_data/33K-5pct.jsonl"
    "/workspace/olmo/dpo_filter_data/33K-1pct.jsonl"
    "/workspace/olmo/dpo_filter_data/33K-0.25pct.jsonl"
)

for DATASET in "${DATASETS[@]}"; do
    # Extract name for output dir (e.g., "33K-5pct" from path)
    NAME=$(basename "$DATASET" .jsonl)
    OUTPUT_DIR="/workspace/olmo/dpo_checkpoints/olmo3_7b_instruct_dpo_${NAME}"

    LOG_FILE="/workspace/olmo/dpo_filter_sweep_logs/${NAME}.log"
    mkdir -p /workspace/olmo/dpo_filter_sweep_logs

    echo "Training with dataset: $DATASET"
    echo "Output dir: $OUTPUT_DIR"
    echo "Log file: $LOG_FILE"

    NUM_GPUS=8
    BATCH_SIZE_PER_GPU=1
    TOTAL_BATCH_SIZE=128
    GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))

    accelerate launch \
        --mixed_precision bf16 \
        --num_machines 1 \
        --num_processes $NUM_GPUS \
        --use_deepspeed \
        --deepspeed_config_file configs/ds_configs/stage3_no_offloading_accelerate.conf \
        open_instruct/dpo_tune_cache.py \
        configs/train_configs/olmo3/olmo3_7b_instruct_dpo_filtered.yaml \
        --dataset_mixer_list="${DATASET},1.0" \
        --output_dir="$OUTPUT_DIR" \
        --exp_name="dpo_filter_${NAME}" \
        --do_not_randomize_output_dir=true \
        2>&1 | tee "$LOG_FILE"
done
