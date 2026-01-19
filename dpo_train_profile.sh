#!/bin/bash
set -euo pipefail

DATASET="${1:?Usage: $0 <dataset_path> [output_dir]}"
# Convert to absolute path before changing directory
[[ -f "$DATASET" ]] && DATASET="$(realpath "$DATASET")"

cd /workspace/olmo/open-instruct
source .venv/bin/activate
OUTPUT_DIR="${2:-/workspace/olmo/dpo_train_profile}"

accelerate launch \
    --mixed_precision bf16 \
    --num_processes 1 \
    open_instruct/dpo_tune_cache.py \
    configs/train_configs/olmo3/olmo3_7b_instruct_dpo_filtered.yaml \
    --dataset_mixer_list="${DATASET},1.0" \
    --output_dir="$OUTPUT_DIR" \
    --num_train_epochs=1 \
    --logging_steps=1 \
    --profile=true \
    --profile_steps=10 \
    --with_tracking=false \
    --try_launch_beaker_eval_jobs=false \
    --do_not_randomize_output_dir=true
