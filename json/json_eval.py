import json
import re
import os
import argparse
from pathlib import Path
from typing import List, Dict, Tuple


def read_json_results(result_dir: str) -> List[Dict]:
    """
    Read all sample files from the specified directory.

    Args:
        result_dir: Path to result directory containing JSONL files

    Returns:
        List of sample dicts (deduplicated by doc_id)
    """
    samples = []
    result_path = Path(result_dir)

    if not result_path.exists():
        print(f"Warning: directory not found: {result_dir}")
        return samples

    jsonl_files = list(result_path.glob("*.jsonl"))
    print(f"Found {len(jsonl_files)} files in {result_dir}")

    doc_ids = set()

    for file_path in jsonl_files:
        print(f"  Reading: {file_path.name}")
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    if data['doc_id'] not in doc_ids:
                        doc_ids.add(data['doc_id'])
                        samples.append(data)
                except json.JSONDecodeError:
                    continue

    print(f"Total: {len(samples)} samples (deduplicated)\n")
    return samples


def extract_json_from_response_regex(response: str) -> str:
    """
    Extract JSON string from response using regex.
    Finds the first '{' and last '}', returns the substring between them.
    """
    start_idx = response.find('{')
    if start_idx == -1:
        return ""

    end_idx = response.rfind('}')
    if end_idx == -1 or end_idx < start_idx:
        return ""

    return response[start_idx:end_idx + 1]


def extract_json_from_response_prefix(response: str) -> str:
    """
    Extract JSON string by removing 'Answer: ' prefix if present.
    """
    prefix = "Answer: "
    if response.startswith(prefix):
        return response[len(prefix):].strip()
    else:
        return response.strip()


def is_valid_json(json_str: str) -> Tuple[bool, Dict]:
    """Check if a string is valid JSON. Returns (is_valid, parsed_dict)."""
    if not json_str:
        return False, {}

    try:
        parsed = json.loads(json_str)
        if isinstance(parsed, dict):
            return True, parsed
        else:
            return False, {}
    except (json.JSONDecodeError, ValueError):
        return False, {}


def check_hallucination(parsed_json: Dict, expected_fields: set) -> bool:
    """Check if JSON contains extra fields beyond expected_fields."""
    actual_fields = {key.lower().replace(' ', '_') for key in parsed_json.keys()}
    normalized_expected = {field.lower().replace(' ', '_') for field in expected_fields}
    extra_fields = actual_fields - normalized_expected
    return len(extra_fields) > 0


def evaluate_json_quality(samples: List[Dict], expected_fields: set = None) -> Dict:
    """
    Evaluate JSON generation quality using two extraction methods.

    Args:
        samples: List of sample dicts
        expected_fields: Expected JSON fields (default: full_name, birth_date, profession)

    Returns:
        Dict with metrics for both extraction methods
    """
    if expected_fields is None:
        expected_fields = {"birth_date", "full_name", "profession"}

    total_samples = len(samples)

    results_regex = {
        'method': 'regex',
        'valid_json_count': 0,
        'hallucination_count': 0,
        'valid_json_samples': [],
        'invalid_samples': [],
        'hallucination_samples': []
    }

    results_prefix = {
        'method': 'prefix_removal',
        'valid_json_count': 0,
        'hallucination_count': 0,
        'valid_json_samples': [],
        'invalid_samples': [],
        'hallucination_samples': []
    }

    for sample in samples:
        doc_id = sample.get('doc_id', 'unknown')

        if 'resps' in sample and len(sample['resps']) > 0:
            response = sample['resps'][0]
            if isinstance(response, list) and len(response) > 0:
                response = response[0]
        else:
            results_regex['invalid_samples'].append(doc_id)
            results_prefix['invalid_samples'].append(doc_id)
            continue

        # Method 1: regex extraction
        json_str_regex = extract_json_from_response_regex(response)
        is_valid_regex, parsed_json_regex = is_valid_json(json_str_regex)

        if is_valid_regex:
            results_regex['valid_json_count'] += 1
            results_regex['valid_json_samples'].append(doc_id)

            has_hallucination = check_hallucination(parsed_json_regex, expected_fields)
            if has_hallucination:
                results_regex['hallucination_count'] += 1
                results_regex['hallucination_samples'].append({
                    'doc_id': doc_id,
                    'extra_fields': list(set(parsed_json_regex.keys()) - expected_fields)
                })
        else:
            results_regex['invalid_samples'].append(doc_id)

        # Method 2: prefix removal
        json_str_prefix = extract_json_from_response_prefix(response)
        is_valid_prefix, parsed_json_prefix = is_valid_json(json_str_prefix)

        if is_valid_prefix:
            results_prefix['valid_json_count'] += 1
            results_prefix['valid_json_samples'].append(doc_id)

            has_hallucination = check_hallucination(parsed_json_prefix, expected_fields)
            if has_hallucination:
                results_prefix['hallucination_count'] += 1
                results_prefix['hallucination_samples'].append({
                    'doc_id': doc_id,
                    'extra_fields': list(set(parsed_json_prefix.keys()) - expected_fields)
                })
        else:
            results_prefix['invalid_samples'].append(doc_id)

    # Compute metrics - Method 1
    s_fmt_regex = results_regex['valid_json_count'] / total_samples if total_samples > 0 else 0
    s_hal_regex = results_regex['hallucination_count'] / results_regex['valid_json_count'] if results_regex['valid_json_count'] > 0 else 0

    results_regex['S_fmt'] = s_fmt_regex
    results_regex['S_Hal'] = s_hal_regex

    # Compute metrics - Method 2
    s_fmt_prefix = results_prefix['valid_json_count'] / total_samples if total_samples > 0 else 0
    s_hal_prefix = results_prefix['hallucination_count'] / results_prefix['valid_json_count'] if results_prefix['valid_json_count'] > 0 else 0

    results_prefix['S_fmt'] = s_fmt_prefix
    results_prefix['S_Hal'] = s_hal_prefix

    return {
        'total_samples': total_samples,
        'regex_extraction': results_regex,
        'prefix_removal_extraction': results_prefix
    }


def batch_evaluate(base_dir: str, expected_fields: set = None) -> Dict[str, Dict]:
    """
    Batch evaluate multiple result subdirectories.

    Args:
        base_dir: Base directory containing result subdirectories
        expected_fields: Expected JSON fields

    Returns:
        Evaluation results per subdirectory
    """
    if expected_fields is None:
        expected_fields = {"birth_date", "full_name", "profession"}

    base_path = Path(base_dir)

    if not base_path.exists():
        print(f"Error: directory not found: {base_dir}")
        return {}

    subdirs = [d for d in base_path.iterdir() if d.is_dir()]

    results = {}

    print("=" * 80)
    print(f"Batch evaluation: {len(subdirs)} directories")
    print("=" * 80)
    print()

    for subdir in subdirs:
        dir_name = subdir.name
        print(f"{'=' * 80}")
        print(f"Evaluating: {dir_name}")
        print(f"{'=' * 80}")

        samples = read_json_results(str(subdir))

        if not samples:
            print(f"Warning: no samples in {dir_name}\n")
            continue

        result = evaluate_json_quality(samples, expected_fields)
        results[dir_name] = result

        print(f"Total samples: {result['total_samples']}")
        print()
        print("[Method 1: Regex extraction]")
        regex_result = result['regex_extraction']
        print(f"  Valid JSON: {regex_result['valid_json_count']}")
        print(f"  Hallucinations: {regex_result['hallucination_count']}")
        print(f"  S_fmt (valid JSON rate): {regex_result['S_fmt']:.4f} ({regex_result['S_fmt']*100:.2f}%)")
        print(f"  S_Hal (hallucination rate): {regex_result['S_Hal']:.4f} ({regex_result['S_Hal']*100:.2f}%)")
        print()
        print("[Method 2: Prefix removal]")
        prefix_result = result['prefix_removal_extraction']
        print(f"  Valid JSON: {prefix_result['valid_json_count']}")
        print(f"  Hallucinations: {prefix_result['hallucination_count']}")
        print(f"  S_fmt (valid JSON rate): {prefix_result['S_fmt']:.4f} ({prefix_result['S_fmt']*100:.2f}%)")
        print(f"  S_Hal (hallucination rate): {prefix_result['S_Hal']:.4f} ({prefix_result['S_Hal']*100:.2f}%)")
        print()

    return results


def print_summary(results: Dict[str, Dict]):
    """Print summary table for both extraction methods."""
    print("\n" + "=" * 100)
    print("Summary - Method 1: Regex extraction")
    print("=" * 100)
    print()
    print(f"{'Directory':<30} {'Total':<10} {'Valid JSON':<12} {'S_fmt':<12} {'Halluc.':<10} {'S_Hal':<12}")
    print("-" * 100)

    for dir_name, result in results.items():
        regex_result = result['regex_extraction']
        print(f"{dir_name:<30} "
              f"{result['total_samples']:<10} "
              f"{regex_result['valid_json_count']:<12} "
              f"{regex_result['S_fmt']*100:>6.2f}%     "
              f"{regex_result['hallucination_count']:<10} "
              f"{regex_result['S_Hal']*100:>6.2f}%")

    print("=" * 100)
    print()

    print("\n" + "=" * 100)
    print("Summary - Method 2: Prefix removal")
    print("=" * 100)
    print()
    print(f"{'Directory':<30} {'Total':<10} {'Valid JSON':<12} {'S_fmt':<12} {'Halluc.':<10} {'S_Hal':<12}")
    print("-" * 100)

    for dir_name, result in results.items():
        prefix_result = result['prefix_removal_extraction']
        print(f"{dir_name:<30} "
              f"{result['total_samples']:<10} "
              f"{prefix_result['valid_json_count']:<12} "
              f"{prefix_result['S_fmt']*100:>6.2f}%     "
              f"{prefix_result['hallucination_count']:<10} "
              f"{prefix_result['S_Hal']*100:>6.2f}%")

    print("=" * 100)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate JSON generation quality (S_fmt and S_Hal metrics)")
    parser.add_argument("--base_dir", type=str, required=True,
                        help="Base directory containing evaluation result subdirectories")
    parser.add_argument("--output_file", type=str, default="./json_eval_results.json",
                        help="Output JSON file for detailed results")
    parser.add_argument("--expected_fields", type=str, nargs='+',
                        default=["birth_date", "full_name", "profession"],
                        help="Expected JSON field names")
    args = parser.parse_args()

    expected_fields = set(args.expected_fields)
    results = batch_evaluate(args.base_dir, expected_fields)
    print_summary(results)

    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nDetailed results saved to: {args.output_file}")
