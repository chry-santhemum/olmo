#!/bin/bash
set -e

# Configuration
VLLM_VENV="/workspace/olmo/.venv-vllm"
EVAL_VENV="/root/.venv"
VLLM_PORT=8020
VLLM_HOST="127.0.0.1"
MAX_TURNS=20
RESULTS_BASE_DIR="/workspace/olmo/petri-results"

# Models to evaluate (HuggingFace model IDs)
MODELS=(
    "allenai/Olmo-3-7B-Think-SFT"
    "allenai/Olmo-3-7B-Instruct-SFT"
    "allenai/Olmo-3-7B-Instruct-DPO"
    "allenai/Olmo-3-7B-Instruct"
)

# Short names for directory naming
MODEL_NAMES=(
    "olmo-3-7b-think-sft"
    "olmo-3-7b-instruct-sft"
    "olmo-3-7b-instruct-dpo"
    "olmo-3-7b-instruct-final"
)

# Source environment variables for API keys
source /workspace/olmo/.env

# Create results directory
mkdir -p "$RESULTS_BASE_DIR"

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
    pkill -f "vllm serve" || true
    sleep 5
}

run_evaluation() {
    local model_id="$1"
    local model_name="$2"
    local save_dir="${RESULTS_BASE_DIR}/${model_name}"

    mkdir -p "$save_dir"

    echo "========================================"
    echo "Evaluating: $model_name"
    echo "Model ID: $model_id"
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

    # Wait for server to be ready
    if ! wait_for_server; then
        echo "Failed to start server for $model_name"
        kill $VLLM_PID 2>/dev/null || true
        return 1
    fi

    # Run Petri evaluation
    echo "Running Petri evaluation..."
    source "$EVAL_VENV/bin/activate"

    # Set OpenAI base URL for this evaluation
    export OPENAI_BASE_URL="http://${VLLM_HOST}:${VLLM_PORT}/v1"
    export OPENAI_API_KEY="dummy"

    cd /workspace/petri
    inspect eval petri/audit \
        --model-role "target={model: openai/${model_id}, max_connections: 16}" \
        --model-role "auditor={model: anthropic/claude-sonnet-4-5, max_connections: 50}" \
        --model-role "judge={model: anthropic/claude-opus-4-5, max_connections: 50}" \
        --max-retries 8 \
        --fail-on-error 5 \
        -T "max_turns=${MAX_TURNS}" \
        -T "transcript_save_dir=${save_dir}" \
        -T "prefill=False" \
        --log-dir "${save_dir}/logs" \
        2>&1 | tee "${save_dir}/petri.log"

    # Stop vLLM server
    kill_vllm_server

    echo "Completed evaluation for $model_name"
    echo ""
}

# Main loop
echo "Starting Petri evaluations for ${#MODELS[@]} models"
echo ""

for i in "${!MODELS[@]}"; do
    run_evaluation "${MODELS[$i]}" "${MODEL_NAMES[$i]}"
done

echo "========================================"
echo "All evaluations complete!"
echo "Results saved to: $RESULTS_BASE_DIR"
echo "========================================"
