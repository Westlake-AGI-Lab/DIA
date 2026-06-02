
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
# Run from this scripts dir so `eval_dia.py` and the evals_results/ output path resolve
# regardless of where this script is invoked from (e.g. `bash reasoning/run_dia.sh`).
cd "$SCRIPT_DIR"

tasks="gsm8k_cot"
nshots="8"
lengths="256"
temperatures="0"

# Local model dir containing the DIA code (modeling_dream.py / generation_utils_dia.py)
# + downloaded Dream-v0-Base-7B weights. Override with an absolute path or HF hub id if needed.
model="$PROJECT_DIR/resource/dream_dia"
# Create arrays from space-separated strings
read -ra TASKS_ARRAY <<< "$tasks"
read -ra NSHOTS_ARRAY <<< "$nshots"
read -ra LENGTH_ARRAY <<< "$lengths"
read -ra TEMP_ARRAY <<< "$temperatures"

export HF_ALLOW_CODE_EVAL=1
### NOTICE: use postprocess for humaneval
# python postprocess_code.py {the samples_xxx.jsonl file under output_path}

# Iterate through the arrays
for i in "${!TASKS_ARRAY[@]}"; do
    output_path=evals_results/${TASKS_ARRAY[$i]}-ns${NSHOTS_ARRAY[$i]}
    echo "Task: ${TASKS_ARRAY[$i]}, Shots: ${NSHOTS_ARRAY[$i]}; Output: $output_path"
    accelerate launch eval_dia.py --model dream \
        --model_args pretrained=${model},max_new_tokens=${LENGTH_ARRAY[$i]},diffusion_steps=4,add_bos_token=true,temperature=${TEMP_ARRAY[$i]},top_p=0.95 \
        --tasks ${TASKS_ARRAY[$i]} \
        --num_fewshot ${NSHOTS_ARRAY[$i]} \
        --batch_size 1 \
        --output_path $output_path \
        --log_samples \
        --limit 5 \
        --confirm_run_unsafe_code
done
