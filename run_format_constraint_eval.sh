#!/bin/bash
set -e

# Run format constraint experiment: DPO vs Final (RL) checkpoint comparison
# Tests whether format constraints reduce Q&A accuracy differently across training stages

# Track vLLM PID for cleanup
VLLM_PID=""

cleanup() {
    echo ""
    echo "Cleaning up..."
    if [ -n "$VLLM_PID" ] && kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "Stopping vLLM server (PID: $VLLM_PID)..."
        kill "$VLLM_PID" 2>/dev/null || true
        sleep 2
        if kill -0 "$VLLM_PID" 2>/dev/null; then
            echo "Force killing vLLM server..."
            kill -9 "$VLLM_PID" 2>/dev/null || true
        fi
    fi
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

# Dataset generation defaults
N_ANSWER=1024
N_ARE_YOU_SURE=1024
REGENERATE_DATASET=false

# Grader model (external, not the model being evaluated)
GRADER_MODEL="openrouter/openai/gpt-5-mini"

# Models to evaluate (only DPO and Final/RL)
MODELS=(
    "allenai/Olmo-3-7B-Instruct-DPO"
    "allenai/Olmo-3-7B-Instruct"
)
MODEL_NAMES=(
    "olmo-3-7b-instruct-dpo"
    "olmo-3-7b-instruct-final"
)

# Single task
TASK="format_constraint"

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
        --n-answer)
            N_ANSWER="$2"
            REGENERATE_DATASET=true
            shift 2
            ;;
        --n-are-you-sure)
            N_ARE_YOU_SURE="$2"
            REGENERATE_DATASET=true
            shift 2
            ;;
        --regenerate-dataset)
            REGENERATE_DATASET=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Run format constraint evaluation on DPO and Final (RL) models."
            echo "Tests whether IFEval-style format constraints reduce Q&A accuracy."
            echo ""
            echo "Options:"
            echo "  --resume <dir>        Resume from a previous run directory"
            echo "  --dry-run             Print what would be run without executing"
            echo "  --n-answer <n>        Number of samples from answer.jsonl (default: 1024)"
            echo "  --n-are-you-sure <n>  Number of samples from are_you_sure.jsonl (default: 1024)"
            echo "  --regenerate-dataset  Force regenerate the dataset before running"
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

# Regenerate dataset if requested
if [ "$REGENERATE_DATASET" = true ]; then
    echo "Regenerating format_constraint dataset..."
    echo "  --n-answer: $N_ANSWER"
    echo "  --n-are-you-sure: $N_ARE_YOU_SURE"
    source "$EVAL_VENV/bin/activate"
    python /workspace/olmo/sycophancy_eval/create_format_constraint_dataset.py \
        --n-answer "$N_ANSWER" \
        --n-are-you-sure "$N_ARE_YOU_SURE"
    echo ""
fi

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

# Check if task is complete (has .eval files)
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
        local task_dir="${EXPERIMENT_DIR}/${model_name}/${TASK}"
        if ! task_is_complete "$task_dir"; then
            missing+=("$i")
        fi
    done
    echo "${missing[@]}"
}

MISSING_EVALS=($(find_missing_evals))

if [ ${#MISSING_EVALS[@]} -eq 0 ]; then
    echo "All evaluations are complete in $EXPERIMENT_DIR"
    echo ""
    echo "To analyze results, run:"
    echo "  python sycophancy_eval/analyze.py --results-dir $EXPERIMENT_DIR"
    exit 0
fi

echo "Missing evaluations: ${#MISSING_EVALS[@]}"
for idx in "${MISSING_EVALS[@]}"; do
    echo "  - ${MODEL_NAMES[$idx]}/${TASK}"
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

run_format_constraint() {
    local model_id="$1"
    local model_name="$2"
    local save_dir="${EXPERIMENT_DIR}/${model_name}"

    mkdir -p "$save_dir"

    echo "Running format_constraint for ${model_name}..."

    # n_per_template = N_SAMPLES / 2 (two template types: baseline, with_constraint)
    inspect eval sycophancy_eval/tasks/format_constraint.py \
        --model "openai/${model_id}" \
        --max-connections "$MAX_CONNECTIONS" \
        -T "n_per_template=$((N_SAMPLES / 2))" \
        -T "grader_model=$GRADER_MODEL" \
        --log-dir "${save_dir}/format_constraint" \
        2>&1 | tee "${save_dir}/format_constraint.log"
}

run_model() {
    local model_idx="$1"
    local model_id="${MODELS[$model_idx]}"
    local model_name="${MODEL_NAMES[$model_idx]}"
    local save_dir="${EXPERIMENT_DIR}/${model_name}"

    mkdir -p "$save_dir"

    echo "========================================"
    echo "Evaluating: $model_name"
    echo "Model ID: $model_id"
    echo "Task: $TASK"
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

    # Run evaluation
    source "$EVAL_VENV/bin/activate"

    export OPENAI_BASE_URL="http://${VLLM_HOST}:${VLLM_PORT}/v1"
    export OPENAI_API_KEY="dummy"
    export PYTHONPATH="/workspace/olmo:${PYTHONPATH}"

    cd /workspace/olmo

    run_format_constraint "$model_id" "$model_name"

    # Stop vLLM server
    kill_vllm_server

    echo "Completed evaluation for $model_name"
    echo ""
}

# Main loop
echo "Starting format constraint evaluation"
echo "Experiment: $EXPERIMENT_DIR"
echo "N_SAMPLES: $N_SAMPLES (${N_SAMPLES}/2 per template type)"
echo ""

for idx in "${MISSING_EVALS[@]}"; do
    run_model "$idx"
done

echo "========================================"
echo "All evaluations complete!"
echo "Results saved to: $EXPERIMENT_DIR"
echo ""
echo "To analyze results, run:"
echo "  python sycophancy_eval/analyze.py --results-dir $EXPERIMENT_DIR"
echo "========================================"
