#!/bin/bash
################################################################################
# DIA JSON Generation Evaluation (Scenario 1)
#
# Runs json/eval_json_generation.py over the biography dataset using the
# DIA-based JSON template method. Works on a single GPU (plain `python`) or
# multiple GPUs (data-parallel via `accelerate launch`).
#
# GPU selection (--num-gpus):
#   auto (default)  -> detect with nvidia-smi; >1 GPU => multi-GPU mode
#   1               -> force single-GPU (plain python)
#   N (>=2)         -> force N-GPU data-parallel (accelerate launch)
#
# Examples:
#   bash json/run_json_eval.sh                              # auto-detect GPUs, full dataset
#   bash json/run_json_eval.sh --max-samples 100            # quick run on 100 samples
#   bash json/run_json_eval.sh --num-gpus 4                 # force 4-GPU data-parallel
#   bash json/run_json_eval.sh --model ./models/my_dream    # use a local model dir
#   bash json/run_json_eval.sh --initial 4,6,8 --max 20,15,30
################################################################################

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

EVAL_SCRIPT="$PROJECT_DIR/json/eval_json_generation.py"

# ---- defaults (all overridable via flags / env) ----------------------------
NUM_GPUS="${NUM_GPUS:-auto}"
MODEL_PATH="${MODEL_PATH:-$PROJECT_DIR/resource/dream_dia}"
INPUT_FILE="${INPUT_FILE:-$PROJECT_DIR/data/biography_dataset_10000.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-eval_results}"
INITIAL_TOKENS="6,6,6"
MAX_TOKENS="32,32,32"
STEPS=16
TEMPERATURE=0.7
TOP_P=0.95
TOP_K=50
EXPAND_DELTA=4
CONFIDENCE_THRESHOLD=0.05
MAX_SAMPLES=""
VERBOSE=""

show_help() {
    sed -n '2,21p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    cat <<EOF

OPTIONS:
    -g, --num-gpus N|auto   GPUs to use (default: $NUM_GPUS)
    -m, --model PATH        Model dir or HF hub id (default: $MODEL_PATH)
    -i, --input FILE        Input JSONL dataset (default: data/biography_dataset_10000.jsonl)
    -o, --output DIR        Output directory (default: $OUTPUT_DIR)
        --initial TOKENS    Initial tokens per field (default: $INITIAL_TOKENS)
        --max TOKENS        Max tokens per field (default: $MAX_TOKENS)
        --steps N           Diffusion steps per block (default: $STEPS)
        --temp FLOAT        Sampling temperature (default: $TEMPERATURE)
        --top-p FLOAT       Nucleus sampling p (default: $TOP_P)
        --top-k INT         Top-k sampling (default: $TOP_K)
        --expand INT        Expand delta (default: $EXPAND_DELTA)
        --threshold FLOAT   Separator confidence threshold (default: $CONFIDENCE_THRESHOLD)
    -n, --max-samples N     Limit number of samples (default: all)
    -v, --verbose           Print detailed generation progress
    -h, --help              Show this help
EOF
}

# ---- arg parsing -----------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case $1 in
        -g|--num-gpus)    NUM_GPUS="$2"; shift 2 ;;
        -m|--model)       MODEL_PATH="$2"; shift 2 ;;
        -i|--input)       INPUT_FILE="$2"; shift 2 ;;
        -o|--output)      OUTPUT_DIR="$2"; shift 2 ;;
        --initial)        INITIAL_TOKENS="$2"; shift 2 ;;
        --max)            MAX_TOKENS="$2"; shift 2 ;;
        --steps)          STEPS="$2"; shift 2 ;;
        --temp)           TEMPERATURE="$2"; shift 2 ;;
        --top-p)          TOP_P="$2"; shift 2 ;;
        --top-k)          TOP_K="$2"; shift 2 ;;
        --expand)         EXPAND_DELTA="$2"; shift 2 ;;
        --threshold)      CONFIDENCE_THRESHOLD="$2"; shift 2 ;;
        -n|--max-samples) MAX_SAMPLES="$2"; shift 2 ;;
        -v|--verbose)     VERBOSE="--verbose"; shift ;;
        -h|--help)        show_help; exit 0 ;;
        *) echo "Unknown option: $1 (use -h for help)"; exit 1 ;;
    esac
done

# ---- resolve GPU mode ------------------------------------------------------
DETECTED_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l || echo 0)
if [ "$NUM_GPUS" = "auto" ]; then
    if [ "$DETECTED_GPUS" -le 1 ]; then NUM_GPUS=1; else NUM_GPUS=$DETECTED_GPUS; fi
fi

# ---- sanity checks ---------------------------------------------------------
[ -f "$EVAL_SCRIPT" ] || { echo "ERROR: not found: $EVAL_SCRIPT"; exit 1; }
[ -f "$INPUT_FILE" ]  || echo "WARNING: dataset not found: $INPUT_FILE (run json/process_biography_dataset.py first)"

echo "========================================"
echo "DIA JSON Generation Evaluation"
echo "========================================"
echo "GPUs:            $NUM_GPUS (detected: $DETECTED_GPUS)"
echo "Model:           $MODEL_PATH"
echo "Input:           $INPUT_FILE"
echo "Output:          $OUTPUT_DIR"
echo "Initial / Max:   $INITIAL_TOKENS / $MAX_TOKENS"
echo "Steps:           $STEPS"
echo "Temp/top-p/top-k:$TEMPERATURE / $TOP_P / $TOP_K"
echo "Expand/thresh:   $EXPAND_DELTA / $CONFIDENCE_THRESHOLD"
[ -n "$MAX_SAMPLES" ] && echo "Max samples:     $MAX_SAMPLES" || echo "Max samples:     all"
echo "========================================"

# ---- build common args -----------------------------------------------------
COMMON_ARGS=(
    --model_path "$MODEL_PATH"
    --input "$INPUT_FILE"
    --output_dir "$OUTPUT_DIR"
    --initial_tokens "$INITIAL_TOKENS"
    --max_tokens "$MAX_TOKENS"
    --steps "$STEPS"
    --temperature "$TEMPERATURE"
    --top_p "$TOP_P"
    --top_k "$TOP_K"
    --expand_delta "$EXPAND_DELTA"
    --confidence_threshold "$CONFIDENCE_THRESHOLD"
)
[ -n "$MAX_SAMPLES" ] && COMMON_ARGS+=(--max_samples "$MAX_SAMPLES")
[ -n "$VERBOSE" ]     && COMMON_ARGS+=("$VERBOSE")

# ---- launch ----------------------------------------------------------------
if [ "$NUM_GPUS" -gt 1 ]; then
    echo "Mode: multi-GPU data-parallel ($NUM_GPUS processes)"
    accelerate launch --num_processes="$NUM_GPUS" "$EVAL_SCRIPT" "${COMMON_ARGS[@]}"
else
    echo "Mode: single-GPU"
    python "$EVAL_SCRIPT" "${COMMON_ARGS[@]}"
fi
