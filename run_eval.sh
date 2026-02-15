#!/bin/bash
set -e

# Unified eval runner for sycophancy_eval tasks.
# Manages vLLM server lifecycle and dispatches to inspect eval.
#
# Examples:
#   ./run_eval.sh --task answer,are_you_sure,feedback
#   ./run_eval.sh --task safety_constraint --regenerate-dataset
#   ./run_eval.sh --task format_constraint --models dpo-final --skip-analysis

VLLM_PID=""

cleanup() {
    if [ -n "$VLLM_PID" ] && kill -0 "$VLLM_PID" 2>/dev/null; then
        kill "$VLLM_PID" 2>/dev/null || true
        sleep 2
        kill -0 "$VLLM_PID" 2>/dev/null && kill -9 "$VLLM_PID" 2>/dev/null || true
    fi
    pkill -f "vllm serve" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# --- Configuration ---
VLLM_VENV="/workspace/olmo/.venv-vllm"
EVAL_VENV="/root/.venv"
VLLM_PORT=8020
VLLM_HOST="127.0.0.1"
RESULTS_DIR="/workspace/olmo/sycophancy_eval/results"
MAX_CONNECTIONS=16
N_SAMPLES=512
SCORER_MODEL="openrouter/openai/gpt-5-mini"

# --- Models ---
declare -A MODEL_ID=(
    [instruct-sft]="allenai/Olmo-3-7B-Instruct-SFT"
    [instruct-dpo]="allenai/Olmo-3-7B-Instruct-DPO"
    [instruct-final]="allenai/Olmo-3-7B-Instruct"
    [think-sft]="allenai/Olmo-3-7B-Think-SFT"
)
PRESET_ALL=(instruct-sft instruct-dpo instruct-final think-sft)
PRESET_DPO_FINAL=(instruct-dpo instruct-final)

# --- Argument parsing ---
TASKS=""
MODELS=""
RESUME_DIR=""
DRY_RUN=false
REGENERATE_DATASET=false
SKIP_ANALYSIS=false
DATASETS_FILTER=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --task) TASKS="$2"; shift 2 ;;
        --models) MODELS="$2"; shift 2 ;;
        --n-samples) N_SAMPLES="$2"; shift 2 ;;
        --scorer-model) SCORER_MODEL="$2"; shift 2 ;;
        --resume) RESUME_DIR="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        --regenerate-dataset) REGENERATE_DATASET=true; shift ;;
        --skip-analysis) SKIP_ANALYSIS=true; shift ;;
        --datasets) DATASETS_FILTER="$2"; shift 2 ;;
        -h|--help)
            cat <<'EOF'
Usage: run_eval.sh --task <tasks> [options]

Tasks (comma-separated):
  answer, are_you_sure, feedback    Sycophancy evaluations
  safety_constraint                 Safety refusal under format constraints
  format_constraint                 Q&A accuracy under format constraints

Options:
  --models <preset>         all | dpo-final (default: auto based on task)
  --n-samples <n>           Sample count (default: 512)
  --scorer-model <model>    Judge/grader model (default: openrouter/openai/gpt-5-mini)
  --resume <dir>            Resume from previous experiment directory
  --dry-run                 Show plan without executing
  --regenerate-dataset      Regenerate dataset (constraint tasks only)
  --skip-analysis           Skip analysis (constraint tasks only)
  --datasets <list>         Dataset filter for format_constraint (comma-separated)
EOF
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

[ -z "$TASKS" ] && echo "Error: --task is required. Use -h for help." && exit 1
IFS=',' read -ra TASK_LIST <<< "$TASKS"

# Default model preset: constraint tasks -> dpo-final, otherwise -> all
if [ -z "$MODELS" ]; then
    has_constraint=false
    for t in "${TASK_LIST[@]}"; do
        [[ "$t" == *constraint* ]] && has_constraint=true
    done
    if $has_constraint; then MODELS="dpo-final"; else MODELS="all"; fi
fi

case "$MODELS" in
    all)        MODEL_KEYS=("${PRESET_ALL[@]}") ;;
    dpo-final)  MODEL_KEYS=("${PRESET_DPO_FINAL[@]}") ;;
    *)          echo "Unknown model preset: $MODELS"; exit 1 ;;
esac

# --- Setup ---
[ -f /workspace/olmo/.env ] && source /workspace/olmo/.env
mkdir -p "$RESULTS_DIR"

if [ -n "$RESUME_DIR" ]; then
    [ ! -d "$RESUME_DIR" ] && echo "Error: $RESUME_DIR does not exist" && exit 1
    EXPERIMENT_DIR="$RESUME_DIR"
else
    EXPERIMENT_DIR="${RESULTS_DIR}/$(date +%Y%m%d_%H%M%S)"
fi

# --- Find missing evaluations ---
task_is_complete() {
    [ -d "$1" ] && ls "$1"/*.eval 1>/dev/null 2>&1
}

MISSING=()
for key in "${MODEL_KEYS[@]}"; do
    for task in "${TASK_LIST[@]}"; do
        if ! task_is_complete "${EXPERIMENT_DIR}/${key}/${task}"; then
            MISSING+=("${key}:${task}")
        fi
    done
done

if [ ${#MISSING[@]} -eq 0 ]; then
    echo "All evaluations complete in $EXPERIMENT_DIR"
    exit 0
fi

echo "Experiment: $EXPERIMENT_DIR"
echo "Missing: ${#MISSING[@]} evaluations"
for e in "${MISSING[@]}"; do echo "  - ${e//:/ / }"; done
echo ""

[ "$DRY_RUN" = true ] && exit 0

# --- Regenerate datasets if requested ---
if [ "$REGENERATE_DATASET" = true ]; then
    source "$EVAL_VENV/bin/activate"
    for task in "${TASK_LIST[@]}"; do
        case $task in
            safety_constraint)
                echo "Regenerating safety_constraint dataset (n=$N_SAMPLES)..."
                python /workspace/olmo/sycophancy_eval/create_safety_constraint_dataset.py \
                    --n-samples "$N_SAMPLES" ;;
            format_constraint)
                echo "Regenerating format_constraint dataset (n=$N_SAMPLES)..."
                python /workspace/olmo/sycophancy_eval/create_format_constraint_dataset.py \
                    --n-per-constraint "$N_SAMPLES" ;;
        esac
    done
fi

# --- vLLM helpers ---
wait_for_server() {
    echo "Waiting for vLLM server..."
    local attempt=0
    while [ $attempt -lt 60 ]; do
        curl -s "http://${VLLM_HOST}:${VLLM_PORT}/v1/models" > /dev/null 2>&1 && echo "Server ready." && return 0
        attempt=$((attempt + 1))
        sleep 5
    done
    echo "Server failed to start"; return 1
}

kill_vllm() {
    if [ -n "$VLLM_PID" ] && kill -0 "$VLLM_PID" 2>/dev/null; then
        kill "$VLLM_PID" 2>/dev/null || true
        sleep 2
        kill -0 "$VLLM_PID" 2>/dev/null && kill -9 "$VLLM_PID" 2>/dev/null || true
    fi
    pkill -f "vllm serve" 2>/dev/null || true
    VLLM_PID=""
    sleep 2
}

# --- Task dispatch ---
run_task() {
    local model_id="$1" key="$2" task="$3"
    local save_dir="${EXPERIMENT_DIR}/${key}"
    mkdir -p "$save_dir"

    echo "  Running $task..."
    case $task in
        answer)
            inspect eval sycophancy_eval/tasks/answer.py \
                --model "openai/${model_id}" \
                --max-connections "$MAX_CONNECTIONS" \
                -T "n_per_template=$((N_SAMPLES / 4))" \
                -T "grader_model=$SCORER_MODEL" \
                --log-dir "${save_dir}/answer" \
                2>&1 | tee "${save_dir}/answer.log"
            ;;
        are_you_sure)
            inspect eval sycophancy_eval/tasks/are_you_sure.py \
                --model "openai/${model_id}" \
                --max-connections "$MAX_CONNECTIONS" \
                -T "n=$N_SAMPLES" \
                -T "grader_model=$SCORER_MODEL" \
                --log-dir "${save_dir}/are_you_sure" \
                2>&1 | tee "${save_dir}/are_you_sure.log"
            ;;
        feedback)
            inspect eval sycophancy_eval/tasks/feedback.py \
                --model "openai/${model_id}" \
                --max-connections "$MAX_CONNECTIONS" \
                -T "n_pairs_per_bias=$((N_SAMPLES / 4))" \
                -T "judge_model=$SCORER_MODEL" \
                --log-dir "${save_dir}/feedback" \
                2>&1 | tee "${save_dir}/feedback.log"
            ;;
        safety_constraint)
            inspect eval sycophancy_eval/tasks/safety_constraint.py \
                --model "openai/${model_id}" \
                --max-connections "$MAX_CONNECTIONS" \
                -T "judge_model=$SCORER_MODEL" \
                --log-dir "${save_dir}/safety_constraint" \
                2>&1 | tee "${save_dir}/safety_constraint.log"
            ;;
        format_constraint)
            local ds_args=""
            [ -n "$DATASETS_FILTER" ] && ds_args="-T datasets=$DATASETS_FILTER"
            # shellcheck disable=SC2086
            inspect eval sycophancy_eval/tasks/format_constraint.py \
                --model "openai/${model_id}" \
                --max-connections "$MAX_CONNECTIONS" \
                -T "grader_model=$SCORER_MODEL" \
                $ds_args \
                --log-dir "${save_dir}/format_constraint" \
                2>&1 | tee "${save_dir}/format_constraint.log"
            ;;
        *)
            echo "Unknown task: $task"; return 1 ;;
    esac
}

# --- Main loop ---
export PYTHONPATH="/workspace/olmo:${PYTHONPATH:-}"
cd /workspace/olmo

for key in "${MODEL_KEYS[@]}"; do
    # Collect tasks missing for this model
    tasks_for_model=()
    for entry in "${MISSING[@]}"; do
        [[ "${entry%%:*}" == "$key" ]] && tasks_for_model+=("${entry#*:}")
    done
    [ ${#tasks_for_model[@]} -eq 0 ] && continue

    model_id="${MODEL_ID[$key]}"
    save_dir="${EXPERIMENT_DIR}/${key}"
    mkdir -p "$save_dir"

    echo "========================================"
    echo "Model: $key ($model_id)"
    echo "Tasks: ${tasks_for_model[*]}"
    echo "========================================"

    # Start vLLM
    source "$VLLM_VENV/bin/activate"
    vllm serve "$model_id" \
        --host "$VLLM_HOST" --port "$VLLM_PORT" --dtype bfloat16 \
        > "${save_dir}/vllm.log" 2>&1 &
    VLLM_PID=$!

    if ! wait_for_server; then
        echo "Failed to start server for $key, skipping."
        kill_vllm
        continue
    fi

    # Run tasks
    source "$EVAL_VENV/bin/activate"
    export OPENAI_BASE_URL="http://${VLLM_HOST}:${VLLM_PORT}/v1"
    export OPENAI_API_KEY="dummy"

    for task in "${tasks_for_model[@]}"; do
        run_task "$model_id" "$key" "$task"
    done

    kill_vllm
    echo ""
done

echo "========================================"
echo "All evaluations complete!"
echo "Results: $EXPERIMENT_DIR"

# --- Analysis (constraint tasks) ---
if [ "$SKIP_ANALYSIS" = false ]; then
    source "$EVAL_VENV/bin/activate"
    for task in "${TASK_LIST[@]}"; do
        case $task in
            safety_constraint)
                echo "Running safety_constraint analysis..."
                python /workspace/olmo/sycophancy_eval/analyze_safety_constraint.py \
                    --results-dir "$EXPERIMENT_DIR" ;;
            format_constraint)
                echo "Running format_constraint analysis..."
                python /workspace/olmo/sycophancy_eval/analyze_format_constraint.py \
                    --results-dir "$EXPERIMENT_DIR" ;;
        esac
    done
fi

echo "========================================"
