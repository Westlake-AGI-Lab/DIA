#!/usr/bin/env python
# coding=utf-8
"""
Pre-build separator token sets.

Run this script to pre-generate the separator_tokens.json file,
so subsequent runs don't need to re-scan the vocabulary.
"""

import argparse
from transformers import AutoTokenizer
from json_template_generation import build_separator_token_sets

def main():
    parser = argparse.ArgumentParser(description="Pre-build separator token sets")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the Dream model directory or HF hub model id")
    parser.add_argument("--output", type=str, default="separator_tokens.json",
                        help="Output JSON file path")
    args = parser.parse_args()

    print("=" * 60)
    print("Building Separator Token Sets")
    print("=" * 60)

    print(f"\nLoading tokenizer from: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    print(f"Tokenizer loaded, vocab size: {len(tokenizer)}")

    print("\n" + "-" * 60)
    comma_tokens, quote_tokens = build_separator_token_sets(tokenizer, json_file=args.output)

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)
    print(f"Token sets saved to: {args.output}")
    print(f"  - Comma-ending tokens: {len(comma_tokens)}")
    print(f"  - Quote-ending tokens: {len(quote_tokens)}")
    print("\nYou can now run evaluation without rescanning the vocabulary.")

if __name__ == "__main__":
    main()
