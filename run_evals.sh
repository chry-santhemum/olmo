#!/bin/bash
set -euo pipefail

cd /workspace/olmo
source ~/.venv/bin/activate

# Edit these and rerun.
MODELS=(
    "instr-sft"
    # "instr-dpo"
    "/workspace/olmo/dpo_checkpoints/olmo3_7b_instruct_dpo_16K-autorater-flip-2"
    "/workspace/olmo/dpo_checkpoints/olmo3_7b_instruct_dpo_16K-autorater-discard-2"
    "/workspace/olmo/dpo_checkpoints/olmo3_7b_instruct_dpo_16K-feedback-20.0pct-flip"
    "/workspace/olmo/dpo_checkpoints/olmo3_7b_instruct_dpo_16K-feedback-20.0pct-prune"
    # "/workspace/olmo/dpo_checkpoints/olmo3_7b_instruct_dpo_16K-baseline"
    # "/workspace/olmo/dpo_checkpoints/olmo3_7b_instruct_dpo_16K-add-256"
    # "/workspace/olmo/dpo_checkpoints/olmo3_7b_instruct_dpo_16K-add-1024"
    # "/workspace/olmo/dpo_checkpoints/olmo3_7b_instruct_dpo_16K-add-2048"
    # "/workspace/olmo/dpo_checkpoints/olmo3_7b_instruct_dpo_16K-all-flip"
    # "/workspace/olmo/dpo_checkpoints/olmo3_7b_instruct_dpo_16K-autorater-flip-1"
    # "/workspace/olmo/dpo_checkpoints/olmo3_7b_instruct_dpo_16K-feedback-5.0pct-flip"
)

TASK="feedback_v2"
N_SAMPLES=1670
SEED=42
MAX_CONNECTIONS=32
MAX_SAMPLES=128
RETRY_ON_ERROR=30
FAIL_ON_ERROR="0.05"
MAX_TOKENS=2048
TEMPERATURE=0.0
JUDGE_MODEL="openrouter/openai/gpt-5-mini"
FEEDBACK_BIAS_TYPES=(
    "positive"
    "negative"
)
BASE_PORT=8020
TASK_TIMEOUT=2700
LOG_DIR=""

CMD=(
    python sycophancy_eval/run_evals.py
    --model "${MODELS[@]}"
    --task "$TASK"
    --n-samples "$N_SAMPLES"
    --seed "$SEED"
    --max-connections "$MAX_CONNECTIONS"
    --max-samples "$MAX_SAMPLES"
    --retry-on-error "$RETRY_ON_ERROR"
    --fail-on-error "$FAIL_ON_ERROR"
    --max-tokens "$MAX_TOKENS"
    --temperature "$TEMPERATURE"
    --judge-model "$JUDGE_MODEL"
    --feedback-bias-types "${FEEDBACK_BIAS_TYPES[@]}"
    --base-port "$BASE_PORT"
    --task-timeout "$TASK_TIMEOUT"
)

if [[ -n "$LOG_DIR" ]]; then
    CMD+=(--log-dir "$LOG_DIR")
fi

"${CMD[@]}"
