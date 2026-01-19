#!/bin/bash

echo "Syncing uv environment..."
source /root/.venv/bin/activate
cd /workspace/olmo
uv sync --active
uv add --active "huggingface_hub" "wandb"

export HF_HOME="/root/hf"
export HF_HUB_ENABLE_HF_TRANSFER=0
hf auth login --token $RUNPOD_HF_TOKEN --add-to-git-credential
wandb login $RUNPOD_WANDB_TOKEN
