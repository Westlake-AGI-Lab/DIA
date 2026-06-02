#!/usr/bin/env python
# coding=utf-8
"""
Biography Dataset JSON Generation Evaluation Script

Input: biography_dataset_10000.jsonl
Output: samples_biography_json_<timestamp>.jsonl

Output format:
{
    "doc_id": int,
    "input": str,
    "ground_truth": dict,
    "resps": [str],
    "latency": float
}

Supports multi-GPU evaluation with accelerate:
    accelerate launch --num_processes=N eval_json_generation.py [OPTIONS]
"""

import torch
import json
import time
import sys
import os
from datetime import datetime
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from json_template_generation import generate_with_json_template, build_separator_token_sets
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
from accelerate import Accelerator


def load_biography_dataset(filepath):
    """Load biography dataset from JSONL file"""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def evaluate_json_generation(
    input_file="biography_dataset_10000.jsonl",
    output_dir="eval_results",
    model_path="resource/dream_dia",
    initial_tokens_per_field=[6, 6, 6],
    max_tokens_per_field=[32, 32, 32],
    steps_per_block=16,
    temperature=0.7,
    top_p=0.95,
    top_k=50,
    expand_delta=4,
    confidence_threshold=0.05,
    max_samples=None,
    verbose=False
):
    """
    Evaluate JSON generation using DIA-based JSON template method.

    Args:
        input_file: Input dataset file path
        output_dir: Output directory
        model_path: Path to the Dream model or HF hub model id
        initial_tokens_per_field: Initial mask tokens per field
            - single int: same for all blocks
            - list of 3 ints: per-block values [full_name, birth_date, profession]
        max_tokens_per_field: Max tokens per field (mask portion only)
            - single int: same for all blocks
            - list of 3 ints: per-block values
        steps_per_block: Diffusion steps per block
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        top_k: Top-k sampling parameter
        expand_delta: Tokens to add per expansion step
        confidence_threshold: Separator confidence threshold
        max_samples: Max samples to evaluate (None = all)
        verbose: Show detailed generation progress
    """
    print("=" * 60)
    print("Biography JSON Generation Evaluation (DIA Method)")
    print("=" * 60)

    # Initialize accelerator
    accelerator = Accelerator()

    if accelerator.is_main_process:
        print(f"\nUsing {accelerator.num_processes} GPU(s)")
        print(f"Current process: {accelerator.process_index}")

    # Create output directory
    output_dir = Path(output_dir)
    if accelerator.is_main_process:
        output_dir.mkdir(exist_ok=True, parents=True)

    # Generate output filename with timestamp
    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S.%f")
    output_file = output_dir / f"samples_biography_json_{timestamp}.jsonl"

    if accelerator.is_main_process:
        print(f"\nInput: {input_file}")
        print(f"Output: {output_file}")
        print(f"Model: {model_path}")
        print(f"Method: JSON Template Generation (DIA-based)")

    # Load data
    if accelerator.is_main_process:
        print("\n" + "-" * 60)
        print("Loading dataset...")
    dataset = load_biography_dataset(input_file)
    if max_samples:
        dataset = dataset[:max_samples]
    if accelerator.is_main_process:
        print(f"Loaded {len(dataset)} samples")

    # Data sharding across processes
    samples_per_process = len(dataset) // accelerator.num_processes
    start_idx = accelerator.process_index * samples_per_process
    if accelerator.process_index == accelerator.num_processes - 1:
        end_idx = len(dataset)
    else:
        end_idx = start_idx + samples_per_process

    local_dataset = dataset[start_idx:end_idx]

    if accelerator.is_main_process:
        print(f"\nData distribution:")
        for i in range(accelerator.num_processes):
            proc_start = i * samples_per_process
            if i == accelerator.num_processes - 1:
                proc_end = len(dataset)
            else:
                proc_end = proc_start + samples_per_process
            print(f"  GPU {i}: samples {proc_start} to {proc_end-1} ({proc_end - proc_start} samples)")

    # Load model
    if accelerator.is_main_process:
        print("\n" + "-" * 60)
        print("Loading model...")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )

    model = accelerator.prepare(model)
    model.eval()

    # Build the separator-token sets once (loads/caches separator_tokens.json), then
    # reuse across all samples instead of rebuilding per generation call.
    separator_token_sets = build_separator_token_sets(tokenizer)

    if accelerator.is_main_process:
        print("Model loaded successfully")

    # Evaluation
    if accelerator.is_main_process:
        print("\n" + "-" * 60)
        print("Starting evaluation...")
        print("-" * 60)

    results = []
    total_latency = 0.0

    temp_output_file = output_dir / f"temp_gpu{accelerator.process_index}_{timestamp}.jsonl"

    if accelerator.is_main_process:
        pbar = tqdm(local_dataset, desc=f"Evaluating (GPU {accelerator.process_index})", unit="sample")
    else:
        pbar = local_dataset

    with open(temp_output_file, 'w', encoding='utf-8') as f_out:
        for idx, sample in enumerate(pbar):
            doc_id = start_idx + idx
            input_text = sample["input_text"]
            ground_truth = sample["answer"]

            prompt = f"""Please extract information from the following text and generate a JSON object.

Text: {input_text}

Now give your answer:
"""

            input_ids = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=True).to(accelerator.device)

            start_time = time.time()

            try:
                with torch.no_grad():
                    output = generate_with_json_template(
                        model=model,
                        tokenizer=tokenizer,
                        input_ids=input_ids,
                        initial_tokens_per_field=initial_tokens_per_field,
                        max_tokens_per_field=max_tokens_per_field,
                        steps_per_block=steps_per_block,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        expand_delta=expand_delta,
                        confidence_threshold=confidence_threshold,
                        separator_token_sets=separator_token_sets,
                        verbose=verbose and accelerator.is_main_process
                    )
                    generated_text = tokenizer.decode(
                        output[0, input_ids.shape[1]:].tolist(),
                        skip_special_tokens=True
                    )

                latency = time.time() - start_time
                total_latency += latency

                result = {
                    "doc_id": doc_id,
                    "input": input_text,
                    "ground_truth": ground_truth,
                    "resps": [generated_text],
                    "latency": latency
                }

            except Exception as e:
                if not verbose and accelerator.is_main_process:
                    tqdm.write(f"\nError processing sample {doc_id}: {e}")
                latency = time.time() - start_time
                result = {
                    "doc_id": doc_id,
                    "input": input_text,
                    "ground_truth": ground_truth,
                    "resps": ["[ERROR]"],
                    "latency": latency,
                    "error": str(e)
                }

            f_out.write(json.dumps(result, ensure_ascii=False) + '\n')
            f_out.flush()

            results.append(result)

            if accelerator.is_main_process and hasattr(pbar, 'set_postfix'):
                avg_latency = total_latency / (idx + 1)
                pbar.set_postfix({
                    'avg_latency': f'{avg_latency:.2f}s',
                    'current_latency': f'{latency:.2f}s'
                })

    accelerator.wait_for_everyone()

    # Main process merges results
    if accelerator.is_main_process:
        print("\n" + "-" * 60)
        print("Merging results from all GPUs...")

        all_results = []
        for i in range(accelerator.num_processes):
            temp_file = output_dir / f"temp_gpu{i}_{timestamp}.jsonl"
            if temp_file.exists():
                with open(temp_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        all_results.append(json.loads(line.strip()))
                temp_file.unlink()

        all_results.sort(key=lambda x: x['doc_id'])

        with open(output_file, 'w', encoding='utf-8') as f_out:
            for result in all_results:
                f_out.write(json.dumps(result, ensure_ascii=False) + '\n')

        total_latency_all = sum(r['latency'] for r in all_results)
        print("=" * 60)
        print("Evaluation Complete")
        print("=" * 60)
        print(f"Total samples: {len(all_results)}")
        print(f"Total latency: {total_latency_all:.2f}s")
        print(f"Average latency: {total_latency_all / len(all_results):.2f}s per sample")
        print(f"Results saved to: {output_file}")
        print("=" * 60)

    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        return all_results, output_file
    else:
        return None, None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate JSON generation on biography dataset using DIA-based JSON template method",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default parameters
  python json/eval_json_generation.py --model_path resource/dream_dia

  # Same initial/max tokens for all blocks
  python json/eval_json_generation.py --model_path resource/dream_dia --initial_tokens 6 --max_tokens 32

  # Per-block token limits (comma-separated: full_name, birth_date, profession)
  python json/eval_json_generation.py --model_path resource/dream_dia --initial_tokens 4,6,8 --max_tokens 20,15,30

  # Test with 10 samples
  python json/eval_json_generation.py --model_path resource/dream_dia --max_samples 10 --verbose

  # Multi-GPU evaluation
  accelerate launch --num_processes=4 json/eval_json_generation.py --model_path resource/dream_dia
        """
    )
    parser.add_argument("--input", type=str, default="biography_dataset_10000.jsonl", help="Input dataset file")
    parser.add_argument("--output_dir", type=str, default="eval_results", help="Output directory")
    parser.add_argument("--model_path", type=str, default="resource/dream_dia",
                        help="Path to the Dream model directory or HF hub model id")
    parser.add_argument("--initial_tokens", type=str, default="6,6,6",
                        help="Initial tokens per field (int or comma-separated list for each block)")
    parser.add_argument("--max_tokens", type=str, default="32,32,32",
                        help="Max tokens per field (int or comma-separated list for each block)")
    parser.add_argument("--steps", type=int, default=16, help="Steps per block")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.95, help="Nucleus sampling parameter")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling parameter")
    parser.add_argument("--expand_delta", type=int, default=4, help="Expand delta")
    parser.add_argument("--confidence_threshold", type=float, default=0.05, help="Confidence threshold")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of samples to evaluate")
    parser.add_argument("--verbose", action="store_true", help="Show detailed generation process")

    args = parser.parse_args()

    def parse_tokens_arg(arg_str):
        if ',' in arg_str:
            return [int(x.strip()) for x in arg_str.split(',')]
        else:
            val = int(arg_str)
            return [val, val, val]

    initial_tokens = parse_tokens_arg(args.initial_tokens)
    max_tokens = parse_tokens_arg(args.max_tokens)

    results, output_file = evaluate_json_generation(
        input_file=args.input,
        output_dir=args.output_dir,
        model_path=args.model_path,
        initial_tokens_per_field=initial_tokens,
        max_tokens_per_field=max_tokens,
        steps_per_block=args.steps,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        expand_delta=args.expand_delta,
        confidence_threshold=args.confidence_threshold,
        max_samples=args.max_samples,
        verbose=args.verbose
    )
