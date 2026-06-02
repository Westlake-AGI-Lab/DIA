#!/usr/bin/env python3
"""
Wikipedia Biography Dataset Processing Script

Extracts structured JSON-format data from raw Wikipedia biography files.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm


def parse_box_line(box_line: str) -> Dict[str, str]:
    """
    Parse a line from the box file, extracting all fields.

    Args:
        box_line: Format like "name_1:lenny\tname_2:randle\tbirth_date_1:12"

    Returns:
        Dict with field names (without sequence numbers) as keys and
        concatenated values as values.
    """
    fields = {}
    tokens = box_line.strip().split('\t')

    for token in tokens:
        if ':' not in token:
            continue

        key_part, value = token.split(':', 1)

        # Extract field name (strip sequence number)
        # e.g. "name_1" -> "name", "birth_date_2" -> "birth_date"
        if '_' in key_part:
            parts = key_part.rsplit('_', 1)
            if len(parts) == 2 and parts[1].isdigit():
                field_name = parts[0]
            else:
                field_name = key_part
        else:
            field_name = key_part

        if value == '<none>':
            continue

        if field_name not in fields:
            fields[field_name] = []
        fields[field_name].append(value)

    result = {}
    for field_name, tokens in fields.items():
        result[field_name] = ' '.join(tokens)

    return result


def extract_name(fields: Dict[str, str]) -> str:
    """Extract person's name."""
    for key in ['name', 'fullname', 'birth_name']:
        if key in fields:
            return fields[key]

    if 'article_title' in fields:
        return fields['article_title']

    return "Unknown"


def extract_birth_date(fields: Dict[str, str]) -> str:
    """Extract birth date."""
    if 'birth_date' in fields:
        return fields['birth_date']
    return "Unknown"


def extract_profession(fields: Dict[str, str]) -> str:
    """Extract profession/occupation."""
    for key in ['occupation', 'profession', 'position', 'office', 'known_for']:
        if key in fields:
            return fields[key]

    if 'position' in fields:
        return fields['position']

    return "Unknown"


def process_dataset(
    data_dir: Path,
    output_file: Path,
    max_samples: Optional[int] = None,
    output_format: str = 'jsonl'
):
    """
    Process the entire Wikipedia Biography Dataset.

    Args:
        data_dir: Path to data directory
        output_file: Output file path
        max_samples: Maximum samples to process (None = all)
        output_format: Output format ('jsonl' or 'json')
    """
    title_file = data_dir / 'test.title'
    box_file = data_dir / 'test.box'
    sent_file = data_dir / 'test.sent'
    nb_file = data_dir / 'test.nb'

    print("Reading data files...")
    with open(title_file, 'r', encoding='utf-8') as f:
        titles = [line.strip() for line in f]

    with open(box_file, 'r', encoding='utf-8') as f:
        boxes = [line.strip() for line in f]

    with open(sent_file, 'r', encoding='utf-8') as f:
        all_sentences = [line.strip() for line in f]

    with open(nb_file, 'r', encoding='utf-8') as f:
        sentence_counts = [int(line.strip()) for line in f]

    n_samples = len(titles)
    assert len(boxes) == n_samples, "box file line count mismatch"
    assert len(sentence_counts) == n_samples, "nb file line count mismatch"

    print(f"Data loaded. Total samples: {n_samples}")

    if max_samples:
        n_samples = min(n_samples, max_samples)
        print(f"Limited to: {n_samples}")

    processed_data = []
    sentence_idx = 0

    print("Processing data...")
    for i in tqdm(range(n_samples), desc="Progress"):
        title = titles[i]
        box_line = boxes[i]
        n_sents = sentence_counts[i]

        biography_sentences = all_sentences[sentence_idx:sentence_idx + n_sents]
        sentence_idx += n_sents

        biography_text = ' '.join(biography_sentences)

        fields = parse_box_line(box_line)

        name = extract_name(fields)
        birth_date = extract_birth_date(fields)
        profession = extract_profession(fields)

        sample = {
            "input_text": f"{title}: {biography_text}",
            "answer": {
                "full_name": name,
                "birth_date": birth_date,
                "profession": profession
            }
        }

        processed_data.append(sample)

    print(f"Saving results to: {output_file}")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if output_format == 'jsonl':
        with open(output_file, 'w', encoding='utf-8') as f:
            for sample in processed_data:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        print(f"Saved {len(processed_data)} records to JSONL file")
    else:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(processed_data)} records to JSON file")

    # Field coverage statistics
    print("\nData statistics:")
    print(f"  Total samples: {len(processed_data)}")

    name_coverage = sum(1 for s in processed_data if s['answer']['full_name'] != 'Unknown')
    birth_coverage = sum(1 for s in processed_data if s['answer']['birth_date'] != 'Unknown')
    prof_coverage = sum(1 for s in processed_data if s['answer']['profession'] != 'Unknown')

    print(f"  full_name coverage: {name_coverage}/{len(processed_data)} ({name_coverage/len(processed_data)*100:.1f}%)")
    print(f"  birth_date coverage: {birth_coverage}/{len(processed_data)} ({birth_coverage/len(processed_data)*100:.1f}%)")
    print(f"  profession coverage: {prof_coverage}/{len(processed_data)} ({prof_coverage/len(processed_data)*100:.1f}%)")

    print("\nData examples (first 3):")
    for i, sample in enumerate(processed_data[:3]):
        print(f"\n--- Example {i+1} ---")
        print(f"Input: {sample['input_text'][:200]}...")
        print(f"Answer: {json.dumps(sample['answer'], ensure_ascii=False, indent=2)}")


def main():
    parser = argparse.ArgumentParser(
        description='Process Wikipedia Biography Dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all data, output as JSONL
  python process_biography_dataset.py --output processed_data.jsonl

  # Process first 1000 records, output as JSON
  python process_biography_dataset.py --max-samples 1000 --format json --output sample_1000.json

  # Specify data directory
  python process_biography_dataset.py --data-dir /path/to/data --output output.jsonl
        """
    )

    parser.add_argument(
        '--data-dir',
        type=str,
        default='./wikipedia-biography-dataset/wikipedia-biography-dataset/test',
        help='Data directory (default: ./wikipedia-biography-dataset/wikipedia-biography-dataset/test)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='processed_biography_data.jsonl',
        help='Output file path (default: processed_biography_data.jsonl)'
    )

    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='Maximum samples to process (default: all)'
    )

    parser.add_argument(
        '--format',
        type=str,
        choices=['json', 'jsonl'],
        default='jsonl',
        help='Output format (default: jsonl)'
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_file = Path(args.output)

    if not data_dir.exists():
        print(f"Error: data directory not found: {data_dir}")
        return

    required_files = ['test.title', 'test.box', 'test.sent', 'test.nb']
    for fname in required_files:
        if not (data_dir / fname).exists():
            print(f"Error: missing required file: {fname}")
            return

    process_dataset(
        data_dir=data_dir,
        output_file=output_file,
        max_samples=args.max_samples,
        output_format=args.format
    )

    print("\nProcessing complete!")


if __name__ == '__main__':
    main()
