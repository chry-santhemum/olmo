#!/bin/bash
set -e

# Track vLLM PIDs for cleanup (associative array: index -> PID)
declare -A VLLM_PIDS

cleanup() {
    echo ""
    echo "Cleaning up..."
    for idx in "${!VLLM_PIDS[@]}"; do
        local pid="${VLLM_PIDS[$idx]}"
        if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
            echo "Stopping vLLM server (PID: $pid, GPU: $idx)..."
            kill "$pid" 2>/dev/null || true
        fi
    done
    sleep 2
    # Force kill any remaining
    for idx in "${!VLLM_PIDS[@]}"; do
        local pid="${VLLM_PIDS[$idx]}"
        if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
            echo "Force killing vLLM server (PID: $pid)..."
            kill -9 "$pid" 2>/dev/null || true
        fi
    done
    # Also kill any orphaned vllm processes
    pkill -f "vllm serve" 2>/dev/null || true
    echo "Cleanup complete."
}

trap cleanup EXIT INT TERM

# Cleanup servers without triggering EXIT trap (for batch processing)
cleanup_servers() {
    for idx in "${!VLLM_PIDS[@]}"; do
        local pid="${VLLM_PIDS[$idx]}"
        if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null || true
        fi
    done
    sleep 2
    for idx in "${!VLLM_PIDS[@]}"; do
        local pid="${VLLM_PIDS[$idx]}"
        if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
            kill -9 "$pid" 2>/dev/null || true
        fi
    done
    VLLM_PIDS=()
}

# Configuration
VLLM_VENV="/workspace/olmo/.venv-vllm"
EVAL_VENV="/root/.venv"
VLLM_BASE_PORT=8020
VLLM_HOST="127.0.0.1"
CHECKPOINT_BASE_DIR="/workspace/olmo/dpo_checkpoints"
RESULTS_BASE_DIR="/workspace/olmo/sycophancy_eval/results"
N_SAMPLES=2048
MAX_CONNECTIONS=32
MAX_SAMPLES=64

# Judge/grader models
JUDGE_MODEL="openrouter/openai/gpt-5-mini"
GRADER_MODEL="openrouter/openai/gpt-5-mini"

# Detect available GPUs (respects CUDA_VISIBLE_DEVICES if set)
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    IFS=',' read -ra GPU_IDS <<< "$CUDA_VISIBLE_DEVICES"
else
    mapfile -t GPU_IDS < <(nvidia-smi -L 2>/dev/null | sed -n 's/GPU \([0-9]*\):.*/\1/p')
fi
NUM_GPUS=${#GPU_IDS[@]}

if [ "$NUM_GPUS" -eq 0 ]; then
    echo "Error: No GPUs detected"
    exit 1
fi

# DPO checkpoints to evaluate (local paths and served model names)
CHECKPOINT_DIRS=(
    "olmo3_7b_instruct_dpo_33K-baseline"
    "olmo3_7b_instruct_dpo_33K-persona-0.25pct-prune-cosine"
    "olmo3_7b_instruct_dpo_33K-persona-1.0pct-prune-cosine"
    "olmo3_7b_instruct_dpo_33K-persona-5.0pct-prune-cosine"
    "olmo3_7b_instruct_dpo_33K-persona-1.0pct-flip-cosine"
    "olmo3_7b_instruct_dpo_33K-persona-1.0pct-prune-dot"
    "olmo3_7b_instruct_dpo_33K-feedback-1.0pct-prune-cosine"
)

SERVED_NAMES=(
    "olmo3_7b_instruct_dpo_33K-baseline"
    "olmo3_7b_instruct_dpo_33K-persona-0.25pct-prune-cosine"
    "olmo3_7b_instruct_dpo_33K-persona-1.0pct-prune-cosine"
    "olmo3_7b_instruct_dpo_33K-persona-5.0pct-prune-cosine"
    "olmo3_7b_instruct_dpo_33K-persona-1.0pct-flip-cosine"
    "olmo3_7b_instruct_dpo_33K-persona-1.0pct-prune-dot"
    "olmo3_7b_instruct_dpo_33K-feedback-1.0pct-prune-cosine"
)


# Default task
TASK="feedback"

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
cd /workspace/olmo 
source ./setup.sh

# Create experiment directory with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXPERIMENT_DIR="${RESULTS_BASE_DIR}/${TIMESTAMP}_dpo_checkpoints"
mkdir -p "$EXPERIMENT_DIR"

NUM_CHECKPOINTS=${#CHECKPOINT_DIRS[@]}
NUM_BATCHES=$(( (NUM_CHECKPOINTS + NUM_GPUS - 1) / NUM_GPUS ))

echo "========================================"
echo "DPO Checkpoint Evaluation (Batch Processing)"
echo "========================================"
echo "Experiment: $EXPERIMENT_DIR"
echo "Task: $TASK"
echo "N_SAMPLES: $N_SAMPLES"
echo "Checkpoints: $NUM_CHECKPOINTS"
echo "GPUs available: $NUM_GPUS (IDs: ${GPU_IDS[*]})"
echo "Batches: $NUM_BATCHES"
echo ""
echo "Batch assignments:"
for ((start=0; start<NUM_CHECKPOINTS; start+=NUM_GPUS)); do
    batch_num=$(( start / NUM_GPUS + 1 ))
    batch_size=$((NUM_CHECKPOINTS - start))
    [ $batch_size -gt $NUM_GPUS ] && batch_size=$NUM_GPUS
    echo "  Batch $batch_num:"
    for ((slot=0; slot<batch_size; slot++)); do
        checkpoint_idx=$((start + slot))
        echo "    Slot $slot (GPU ${GPU_IDS[$slot]}, port $((VLLM_BASE_PORT + slot))): ${CHECKPOINT_DIRS[$checkpoint_idx]}"
    done
done
echo "========================================"
echo ""

if [ "$DRY_RUN" = true ]; then
    echo "Dry run mode - not executing any evaluations"
    exit 0
fi

wait_for_server_on_port() {
    local port="$1"
    local slot="$2"
    echo "[Slot $slot] Waiting for vLLM server on port $port..."
    local max_attempts=120
    local attempt=0
    while [ $attempt -lt $max_attempts ]; do
        if curl -s "http://${VLLM_HOST}:${port}/v1/models" > /dev/null 2>&1; then
            echo "[Slot $slot] Server on port $port is ready!"
            return 0
        fi
        attempt=$((attempt + 1))
        sleep 5
    done
    echo "[Slot $slot] Server on port $port failed to start after ${max_attempts} attempts"
    return 1
}

run_task() {
    local served_name="$1"
    local task="$2"
    local save_dir="$3"
    local port="$4"

    mkdir -p "$save_dir"

    echo "[Port $port] Running ${task} for ${served_name}..."

    export OPENAI_BASE_URL="http://${VLLM_HOST}:${port}/v1"
    export OPENAI_API_KEY="dummy"

    case $task in
        answer)
            inspect eval sycophancy_eval/tasks/answer.py \
                --model "openai/${served_name}" \
                --max-connections "$MAX_CONNECTIONS" \
                --max-samples "$MAX_SAMPLES" \
                -T "n_per_template=$((N_SAMPLES / 4))" \
                -T "grader_model=$GRADER_MODEL" \
                --log-dir "${save_dir}/answer" \
                2>&1 | tee "${save_dir}/answer.log"
            ;;
        are_you_sure)
            inspect eval sycophancy_eval/tasks/are_you_sure.py \
                --model "openai/${served_name}" \
                --max-connections "$MAX_CONNECTIONS" \
                --max-samples "$MAX_SAMPLES" \
                -T "n=$N_SAMPLES" \
                -T "grader_model=$GRADER_MODEL" \
                --log-dir "${save_dir}/are_you_sure" \
                2>&1 | tee "${save_dir}/are_you_sure.log"
            ;;
        feedback)
            inspect eval sycophancy_eval/tasks/feedback.py \
                --model "openai/${served_name}" \
                --max-connections "$MAX_CONNECTIONS" \
                --max-samples "$MAX_SAMPLES" \
                -T "n_pairs_per_bias=$((N_SAMPLES / 4))" \
                -T "judge_model=$JUDGE_MODEL" \
                --log-dir "${save_dir}/feedback" \
                2>&1 | tee "${save_dir}/feedback.log"
            ;;
    esac
}

start_vllm_server() {
    local checkpoint_idx="$1"
    local gpu_slot="$2"
    local checkpoint_dir="${CHECKPOINT_DIRS[$checkpoint_idx]}"
    local served_name="${SERVED_NAMES[$checkpoint_idx]}"
    local checkpoint_path="${CHECKPOINT_BASE_DIR}/${checkpoint_dir}"
    local save_dir="${EXPERIMENT_DIR}/${served_name}"
    local gpu_id="${GPU_IDS[$gpu_slot]}"
    local port=$((VLLM_BASE_PORT + gpu_slot))

    mkdir -p "$save_dir"

    echo "[Slot $gpu_slot, GPU $gpu_id] Starting vLLM server for $checkpoint_dir on port $port..."
    source "$VLLM_VENV/bin/activate"
    CUDA_VISIBLE_DEVICES="$gpu_id" vllm serve "$checkpoint_path" \
        --host "$VLLM_HOST" \
        --port "$port" \
        --dtype bfloat16 \
        --served-model-name "$served_name" \
        > "${save_dir}/vllm.log" 2>&1 &
    VLLM_PIDS[$gpu_slot]=$!
    echo "[Slot $gpu_slot, GPU $gpu_id] vLLM server started with PID ${VLLM_PIDS[$gpu_slot]}"
}

run_eval_for_checkpoint() {
    local checkpoint_idx="$1"
    local gpu_slot="$2"
    local served_name="${SERVED_NAMES[$checkpoint_idx]}"
    local save_dir="${EXPERIMENT_DIR}/${served_name}"
    local port=$((VLLM_BASE_PORT + gpu_slot))

    echo "[Slot $gpu_slot] Running evaluation for $served_name..."

    # Run tasks
    if [ "$TASK" = "all" ]; then
        for t in answer are_you_sure feedback; do
            run_task "$served_name" "$t" "$save_dir" "$port"
        done
    else
        run_task "$served_name" "$TASK" "$save_dir" "$port"
    fi

    echo "[Slot $gpu_slot] Completed evaluation for $served_name"
}

# ========================================
# Batch Processing Loop
# ========================================

# Switch to eval venv once at the start
source "$EVAL_VENV/bin/activate"
export PYTHONPATH="/workspace/olmo:${PYTHONPATH}"
cd /workspace/olmo

BATCH_IDX=0
EVAL_FAILED=false

for ((start=0; start<NUM_CHECKPOINTS; start+=NUM_GPUS)); do
    BATCH_IDX=$((BATCH_IDX + 1))
    batch_size=$((NUM_CHECKPOINTS - start))
    [ $batch_size -gt $NUM_GPUS ] && batch_size=$NUM_GPUS

    echo ""
    echo "========================================"
    echo "Batch $BATCH_IDX/$NUM_BATCHES: Processing checkpoints $start to $((start + batch_size - 1))"
    echo "========================================"

    # Start servers for this batch
    echo ""
    echo "Starting vLLM servers for batch $BATCH_IDX..."
    for ((slot=0; slot<batch_size; slot++)); do
        checkpoint_idx=$((start + slot))
        start_vllm_server "$checkpoint_idx" "$slot"
    done

    # Wait for all servers in this batch to be ready
    echo ""
    echo "Waiting for servers to be ready..."
    ALL_READY=true
    for ((slot=0; slot<batch_size; slot++)); do
        port=$((VLLM_BASE_PORT + slot))
        if ! wait_for_server_on_port "$port" "$slot"; then
            echo "[Slot $slot] Server failed to start!"
            ALL_READY=false
        fi
    done

    if [ "$ALL_READY" = false ]; then
        echo "Some servers failed to start in batch $BATCH_IDX. Exiting."
        exit 1
    fi

    echo ""
    echo "All servers ready. Running evaluations..."

    # Run evaluations in parallel for this batch
    declare -A EVAL_PIDS
    for ((slot=0; slot<batch_size; slot++)); do
        checkpoint_idx=$((start + slot))
        run_eval_for_checkpoint "$checkpoint_idx" "$slot" &
        EVAL_PIDS[$slot]=$!
        echo "Started eval for checkpoint $checkpoint_idx (slot $slot) with PID ${EVAL_PIDS[$slot]}"
    done

    # Wait for all evaluations in this batch to complete
    echo ""
    echo "Waiting for batch $BATCH_IDX evaluations to complete..."
    for slot in "${!EVAL_PIDS[@]}"; do
        checkpoint_idx=$((start + slot))
        if ! wait "${EVAL_PIDS[$slot]}"; then
            echo "[Slot $slot] Evaluation for checkpoint $checkpoint_idx failed!"
            EVAL_FAILED=true
        else
            echo "[Slot $slot] Evaluation for checkpoint $checkpoint_idx completed successfully"
        fi
    done
    unset EVAL_PIDS

    # Cleanup servers if not the last batch
    if [ $((start + NUM_GPUS)) -lt $NUM_CHECKPOINTS ]; then
        echo ""
        echo "Cleaning up servers before next batch..."
        cleanup_servers
    fi
done

# ========================================
# Final cleanup is handled by trap
# ========================================

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
