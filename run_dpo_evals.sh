#!/bin/bash
set -e

# Track vLLM PID for cleanup
VLLM_PID=""

cleanup() {
    echo ""
    echo "Cleaning up..."
    if [ -n "$VLLM_PID" ] && kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "Stopping vLLM server (PID: $VLLM_PID)..."
        kill "$VLLM_PID" 2>/dev/null || true
        sleep 2
        # Force kill if still running
        if kill -0 "$VLLM_PID" 2>/dev/null; then
            echo "Force killing vLLM server..."
            kill -9 "$VLLM_PID" 2>/dev/null || true
        fi
    fi
    # Also kill any orphaned vllm processes
    pkill -f "vllm serve" 2>/dev/null || true
    echo "Cleanup complete."
}

trap cleanup EXIT INT TERM

# Configuration
VLLM_VENV="/workspace/olmo/.venv-vllm"
EVAL_VENV="/root/.venv"
VLLM_PORT=8020
VLLM_HOST="127.0.0.1"
CHECKPOINT_BASE_DIR="/workspace/olmo/dpo_checkpoints"
RESULTS_BASE_DIR="/workspace/olmo/sycophancy_eval/results"
N_SAMPLES=512
MAX_CONNECTIONS=16

# Judge/grader models
JUDGE_MODEL="openrouter/openai/gpt-5-mini"
GRADER_MODEL="openrouter/openai/gpt-5-mini"

# DPO checkpoints to evaluate (local paths and served model names)
CHECKPOINT_DIRS=(
    "olmo3_7b_instruct_dpo_33K-0.25pct"
    "olmo3_7b_instruct_dpo_33K-1pct"
    "olmo3_7b_instruct_dpo_33K-5pct"
)

SERVED_NAMES=(
    "olmo3-dpo-0.25pct"
    "olmo3-dpo-1pct"
    "olmo3-dpo-5pct"
)

# Default task
TASK="all"

# Parse arguments
DRY_RUN=false
SKIP_PLOT=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --n-samples)
            N_SAMPLES="$2"
            shift 2
            ;;
        --task)
            TASK="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --skip-plot)
            SKIP_PLOT=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --n-samples N   Number of samples per task (default: 512)"
            echo "  --task TASK     Task to run: feedback, answer, are_you_sure, all (default: feedback)"
            echo "  --dry-run       Print what would be run without executing"
            echo "  --skip-plot     Skip generating comparison plot at the end"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Source environment variables (for API keys)
if [ -f /workspace/olmo/.env ]; then
    source /workspace/olmo/.env
fi

# Create experiment directory with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXPERIMENT_DIR="${RESULTS_BASE_DIR}/${TIMESTAMP}_dpo_checkpoints"
mkdir -p "$EXPERIMENT_DIR"

echo "========================================"
echo "DPO Checkpoint Evaluation"
echo "========================================"
echo "Experiment: $EXPERIMENT_DIR"
echo "Task: $TASK"
echo "N_SAMPLES: $N_SAMPLES"
echo "Checkpoints: ${#CHECKPOINT_DIRS[@]}"
for i in "${!CHECKPOINT_DIRS[@]}"; do
    echo "  - ${CHECKPOINT_DIRS[$i]} -> ${SERVED_NAMES[$i]}"
done
echo "========================================"
echo ""

if [ "$DRY_RUN" = true ]; then
    echo "Dry run mode - not executing any evaluations"
    exit 0
fi

wait_for_server() {
    echo "Waiting for vLLM server to be ready..."
    local max_attempts=120
    local attempt=0
    while [ $attempt -lt $max_attempts ]; do
        if curl -s "http://${VLLM_HOST}:${VLLM_PORT}/v1/models" > /dev/null 2>&1; then
            echo "Server is ready!"
            return 0
        fi
        attempt=$((attempt + 1))
        sleep 5
    done
    echo "Server failed to start after ${max_attempts} attempts"
    return 1
}

kill_vllm_server() {
    echo "Stopping vLLM server..."
    if [ -n "$VLLM_PID" ] && kill -0 "$VLLM_PID" 2>/dev/null; then
        kill "$VLLM_PID" 2>/dev/null || true
        sleep 2
        if kill -0 "$VLLM_PID" 2>/dev/null; then
            kill -9 "$VLLM_PID" 2>/dev/null || true
        fi
    fi
    pkill -f "vllm serve" 2>/dev/null || true
    VLLM_PID=""
    sleep 2
}

run_task() {
    local served_name="$1"
    local task="$2"
    local save_dir="$3"

    mkdir -p "$save_dir"

    echo "Running ${task} for ${served_name}..."

    case $task in
        answer)
            inspect eval sycophancy_eval/tasks/answer.py \
                --model "openai/${served_name}" \
                --max-connections "$MAX_CONNECTIONS" \
                -T "n_per_template=$((N_SAMPLES / 4))" \
                -T "grader_model=$GRADER_MODEL" \
                --log-dir "${save_dir}/answer" \
                2>&1 | tee "${save_dir}/answer.log"
            ;;
        are_you_sure)
            inspect eval sycophancy_eval/tasks/are_you_sure.py \
                --model "openai/${served_name}" \
                --max-connections "$MAX_CONNECTIONS" \
                -T "n=$N_SAMPLES" \
                -T "grader_model=$GRADER_MODEL" \
                --log-dir "${save_dir}/are_you_sure" \
                2>&1 | tee "${save_dir}/are_you_sure.log"
            ;;
        feedback)
            inspect eval sycophancy_eval/tasks/feedback.py \
                --model "openai/${served_name}" \
                --max-connections "$MAX_CONNECTIONS" \
                -T "n_pairs_per_bias=$((N_SAMPLES / 4))" \
                -T "judge_model=$JUDGE_MODEL" \
                --log-dir "${save_dir}/feedback" \
                2>&1 | tee "${save_dir}/feedback.log"
            ;;
    esac
}

run_checkpoint_eval() {
    local idx="$1"
    local checkpoint_dir="${CHECKPOINT_DIRS[$idx]}"
    local served_name="${SERVED_NAMES[$idx]}"
    local checkpoint_path="${CHECKPOINT_BASE_DIR}/${checkpoint_dir}"
    local save_dir="${EXPERIMENT_DIR}/${served_name}"

    mkdir -p "$save_dir"

    echo "========================================"
    echo "Evaluating: $served_name"
    echo "Checkpoint: $checkpoint_path"
    echo "Results: $save_dir"
    echo "========================================"

    # Start vLLM server with local checkpoint
    echo "Starting vLLM server for $checkpoint_path..."
    source "$VLLM_VENV/bin/activate"
    vllm serve "$checkpoint_path" \
        --host "$VLLM_HOST" \
        --port "$VLLM_PORT" \
        --dtype bfloat16 \
        --served-model-name "$served_name" \
        > "${save_dir}/vllm.log" 2>&1 &
    VLLM_PID=$!

    if ! wait_for_server; then
        echo "Failed to start server for $served_name"
        kill_vllm_server
        return 1
    fi

    # Switch to eval venv and run evaluations
    source "$EVAL_VENV/bin/activate"

    export OPENAI_BASE_URL="http://${VLLM_HOST}:${VLLM_PORT}/v1"
    export OPENAI_API_KEY="dummy"
    export PYTHONPATH="/workspace/olmo:${PYTHONPATH}"

    cd /workspace/olmo

    # Run tasks
    if [ "$TASK" = "all" ]; then
        for t in answer are_you_sure feedback; do
            run_task "$served_name" "$t" "$save_dir"
        done
    else
        run_task "$served_name" "$TASK" "$save_dir"
    fi

    # Stop vLLM server
    kill_vllm_server

    echo "Completed evaluation for $served_name"
    echo ""
}

# Main loop - evaluate each checkpoint
for i in "${!CHECKPOINT_DIRS[@]}"; do
    run_checkpoint_eval "$i"
done

echo "========================================"
echo "All evaluations complete!"
echo "Results saved to: $EXPERIMENT_DIR"
echo "========================================"

# Generate comparison plot
if [ "$SKIP_PLOT" = false ]; then
    echo ""
    echo "Generating comparison plot..."
    source "$EVAL_VENV/bin/activate"
    export PYTHONPATH="/workspace/olmo:${PYTHONPATH}"
    python /workspace/olmo/sycophancy_eval/analyze_dpo_checkpoints.py \
        --results-dir "$EXPERIMENT_DIR"
fi
