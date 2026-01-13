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
RESULTS_BASE_DIR="/workspace/olmo/sycophancy_eval/results"
N_SAMPLES=512
MAX_CONNECTIONS=16

# Judge/grader models (external, not the model being evaluated)
# All use gpt-5-mini: task 3 with reasoning=medium, tasks 1&2 with reasoning=low
JUDGE_MODEL="openrouter/openai/gpt-5-mini"
GRADER_MODEL="openrouter/openai/gpt-5-mini"

# Models to evaluate
MODELS=(
    "allenai/Olmo-3-7B-Instruct-SFT"
    "allenai/Olmo-3-7B-Instruct-DPO"
    "allenai/Olmo-3-7B-Instruct"
    "allenai/Olmo-3-7B-Think-SFT"
)

MODEL_NAMES=(
    "olmo-3-7b-instruct-sft"
    "olmo-3-7b-instruct-dpo"
    "olmo-3-7b-instruct-final"
    "olmo-3-7b-think-sft"
)

TASKS=("answer" "are_you_sure" "feedback")

# Parse arguments
RESUME_DIR=""
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --resume)
            RESUME_DIR="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [--resume <dir>] [--dry-run]"
            echo ""
            echo "Options:"
            echo "  --resume <dir>  Resume from a previous run directory"
            echo "  --dry-run       Print what would be run without executing"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Source environment variables
if [ -f /workspace/olmo/.env ]; then
    source /workspace/olmo/.env
fi

mkdir -p "$RESULTS_BASE_DIR"

# Set experiment directory
if [ -n "$RESUME_DIR" ]; then
    if [ ! -d "$RESUME_DIR" ]; then
        echo "Error: Resume directory does not exist: $RESUME_DIR"
        exit 1
    fi
    EXPERIMENT_DIR="$RESUME_DIR"
    echo "Resuming from: $EXPERIMENT_DIR"
else
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    EXPERIMENT_DIR="${RESULTS_BASE_DIR}/${TIMESTAMP}"
    echo "New experiment: $EXPERIMENT_DIR"
fi

# Check if a task is complete (has .eval files)
task_is_complete() {
    local task_dir="$1"
    if [ -d "$task_dir" ] && ls "$task_dir"/*.eval 1>/dev/null 2>&1; then
        return 0
    fi
    return 1
}

# Find missing evaluations
find_missing_evals() {
    local missing=()
    for i in "${!MODELS[@]}"; do
        local model_name="${MODEL_NAMES[$i]}"
        for task in "${TASKS[@]}"; do
            local task_dir="${EXPERIMENT_DIR}/${model_name}/${task}"
            if ! task_is_complete "$task_dir"; then
                missing+=("$i:$task")
            fi
        done
    done
    echo "${missing[@]}"
}

MISSING_EVALS=($(find_missing_evals))

if [ ${#MISSING_EVALS[@]} -eq 0 ]; then
    echo "All evaluations are complete in $EXPERIMENT_DIR"
    exit 0
fi

echo "Missing evaluations: ${#MISSING_EVALS[@]}"
for entry in "${MISSING_EVALS[@]}"; do
    idx="${entry%%:*}"
    task="${entry#*:}"
    echo "  - ${MODEL_NAMES[$idx]}/$task"
done
echo ""

if [ "$DRY_RUN" = true ]; then
    echo "Dry run mode - not executing any evaluations"
    exit 0
fi

wait_for_server() {
    echo "Waiting for vLLM server to be ready..."
    local max_attempts=60
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
    local model_id="$1"
    local model_name="$2"
    local task="$3"
    local save_dir="${EXPERIMENT_DIR}/${model_name}"

    mkdir -p "$save_dir"

    echo "Running ${task} for ${model_name}..."

    case $task in
        answer)
            inspect eval sycophancy_eval/tasks/answer.py \
                --model "openai/${model_id}" \
                --max-connections "$MAX_CONNECTIONS" \
                -T "n_per_template=$((N_SAMPLES / 4))" \
                -T "grader_model=$GRADER_MODEL" \
                --log-dir "${save_dir}/answer" \
                2>&1 | tee "${save_dir}/answer.log"
            ;;
        are_you_sure)
            inspect eval sycophancy_eval/tasks/are_you_sure.py \
                --model "openai/${model_id}" \
                --max-connections "$MAX_CONNECTIONS" \
                -T "n=$N_SAMPLES" \
                -T "grader_model=$GRADER_MODEL" \
                --log-dir "${save_dir}/are_you_sure" \
                2>&1 | tee "${save_dir}/are_you_sure.log"
            ;;
        feedback)
            inspect eval sycophancy_eval/tasks/feedback.py \
                --model "openai/${model_id}" \
                --max-connections "$MAX_CONNECTIONS" \
                -T "n_pairs_per_bias=$((N_SAMPLES / 4))" \
                -T "judge_model=$JUDGE_MODEL" \
                --log-dir "${save_dir}/feedback" \
                2>&1 | tee "${save_dir}/feedback.log"
            ;;
    esac
}

run_model_tasks() {
    local model_idx="$1"
    shift
    local tasks_to_run=("$@")

    local model_id="${MODELS[$model_idx]}"
    local model_name="${MODEL_NAMES[$model_idx]}"
    local save_dir="${EXPERIMENT_DIR}/${model_name}"

    mkdir -p "$save_dir"

    echo "========================================"
    echo "Evaluating: $model_name"
    echo "Model ID: $model_id"
    echo "Tasks: ${tasks_to_run[*]}"
    echo "Results: $save_dir"
    echo "========================================"

    # Start vLLM server
    echo "Starting vLLM server for $model_id..."
    source "$VLLM_VENV/bin/activate"
    vllm serve "$model_id" \
        --host "$VLLM_HOST" \
        --port "$VLLM_PORT" \
        --dtype bfloat16 \
        > "${save_dir}/vllm.log" 2>&1 &
    VLLM_PID=$!

    if ! wait_for_server; then
        echo "Failed to start server for $model_name"
        kill_vllm_server
        return 1
    fi

    # Run sycophancy evaluation
    source "$EVAL_VENV/bin/activate"

    export OPENAI_BASE_URL="http://${VLLM_HOST}:${VLLM_PORT}/v1"
    export OPENAI_API_KEY="dummy"
    export PYTHONPATH="/workspace/olmo:${PYTHONPATH}"

    cd /workspace/olmo

    for task in "${tasks_to_run[@]}"; do
        run_task "$model_id" "$model_name" "$task"
    done

    # Stop vLLM server
    kill_vllm_server

    echo "Completed evaluation for $model_name"
    echo ""
}

# Group missing evals by model index
declare -A MODEL_TASKS
for entry in "${MISSING_EVALS[@]}"; do
    idx="${entry%%:*}"
    task="${entry#*:}"
    if [ -z "${MODEL_TASKS[$idx]}" ]; then
        MODEL_TASKS[$idx]="$task"
    else
        MODEL_TASKS[$idx]="${MODEL_TASKS[$idx]} $task"
    fi
done

# Main loop - only run models that have missing tasks
echo "Starting sycophancy evaluations"
echo "Experiment: $EXPERIMENT_DIR"
echo "N_SAMPLES per task: $N_SAMPLES"
echo ""

for idx in "${!MODEL_TASKS[@]}"; do
    tasks=(${MODEL_TASKS[$idx]})
    run_model_tasks "$idx" "${tasks[@]}"
done

echo "========================================"
echo "All evaluations complete!"
echo "Results saved to: $EXPERIMENT_DIR"
echo "========================================"
