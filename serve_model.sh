#!/bin/bash
set -e

# Configuration - can be overridden by environment variables or command line args
VLLM_VENV="${VLLM_VENV:-/root/.venv-vllm}"
CONFIG_DIR="${CONFIG_DIR:-/workspace/olmo/vllm_configs}"
CONFIG_FILE="${CONFIG_FILE:-config_instr_sft.yaml}"

# Handle --config as first argument for convenience
if [[ "$1" == "--config" ]]; then
    CONFIG_FILE="$2"
    shift 2
fi

# Resolve config file path: if not absolute, prepend CONFIG_DIR
if [[ "$CONFIG_FILE" != /* ]]; then
    CONFIG_FILE="$CONFIG_DIR/$CONFIG_FILE"
fi

# Parse config file for defaults
MODEL=$(grep '^model:' "$CONFIG_FILE" | awk '{print $2}')
HOST=$(grep '^host:' "$CONFIG_FILE" | awk '{print $2}' | tr -d '"')
PORT=$(grep '^port:' "$CONFIG_FILE" | awk '{print $2}')
DTYPE=$(grep '^dtype:' "$CONFIG_FILE" | awk '{print $2}' | tr -d '"')

# Allow command line overrides
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --host)
            HOST="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --dtype)
            DTYPE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--config CONFIG_FILE] [--model MODEL] [--host HOST] [--port PORT] [--dtype DTYPE]"
            exit 1
            ;;
    esac
done

echo "========================================"
echo "Starting vLLM server"
echo "Config: $CONFIG_FILE"
echo "Model: $MODEL"
echo "Host: $HOST"
echo "Port: $PORT"
echo "Dtype: $DTYPE"
echo "========================================"

# Activate vLLM virtual environment and serve
source "$VLLM_VENV/bin/activate"
uv add --active "vllm"

vllm serve "$MODEL" \
    --host "$HOST" \
    --port "$PORT" \
    --dtype "$DTYPE"
