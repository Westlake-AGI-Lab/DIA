#!/usr/bin/env python3
"""
JSON Template Injection for dLLM Biography Information Extraction

This module implements block-based generation with JSON structure injection
based on the DIA (Dynamic Infilling with Anchoring) method.

Three blocks structure:
- Block 1: Answer: {"name": "<mask>...<mask>",
- Block 2: "date_of_birth": "<mask>...<mask>",
- Block 3: "profession": "<mask>...<mask>"}

Each block uses separator confidence (comma ',' or quote '"') for
variable-length generation.
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple, List, Union
from transformers import PreTrainedTokenizer
import os
import json


def create_json_template_blocks(
    tokenizer: PreTrainedTokenizer,
    initial_tokens_per_field: Union[int, List[int]] = 8,
    device: str = "cuda"
) -> Tuple[torch.LongTensor, List[Tuple[int, int, int, int]], torch.LongTensor]:
    """
    Create the block structure for JSON template.

    Design notes:
    - Each block consists of: [prefix] [mask tokens] [suffix]
    - block_info records mask token ranges; suffix length is tracked separately
    - Suffix is fixed in the template, not expanded, but avoided during generation

    Args:
        tokenizer: Tokenizer
        initial_tokens_per_field: Initial number of mask tokens per field.
                                  Can be a single int (same for all blocks)
                                  or a length-3 list (one per block).
        device: Device

    Returns:
        template_ids: Complete template token sequence
        block_info: Per-block (block_start, block_end, mask_offset, suffix_len)
                    - block_start: Start position in sequence (includes prefix)
                    - block_end: End position in sequence (includes suffix)
                    - mask_offset: Offset of mask region within the block (prefix length)
                    - suffix_len: Length of the suffix
        separator_ids: Separator token IDs (for confidence calculation)
    """
    mask_token = tokenizer.mask_token
    mask_token_id = tokenizer.convert_tokens_to_ids(mask_token)

    if isinstance(initial_tokens_per_field, int):
        initial_tokens_list = [initial_tokens_per_field] * 3
    elif isinstance(initial_tokens_per_field, (list, tuple)):
        if len(initial_tokens_per_field) != 3:
            raise ValueError(f"initial_tokens_per_field list must have length 3, got {len(initial_tokens_per_field)}")
        initial_tokens_list = list(initial_tokens_per_field)
    else:
        raise TypeError(f"initial_tokens_per_field must be int or list, got {type(initial_tokens_per_field)}")

    # Encode the fixed JSON structure parts
    # Block 1: Answer: {"full name": "  (value starts with a quote)
    block1_prefix = tokenizer.encode('Answer: {"full name": "', add_special_tokens=False)
    block1_suffix = tokenizer.encode('",', add_special_tokens=False)

    # Block 2: "birth_date": "  (value starts with a quote)
    block2_prefix = tokenizer.encode(' "birth_date": "', add_special_tokens=False)
    block2_suffix = tokenizer.encode('",', add_special_tokens=False)

    # Block 3: "profession": "  (value starts with a quote)
    block3_prefix = tokenizer.encode(' "profession": "', add_special_tokens=False)
    block3_suffix = tokenizer.encode('"}', add_special_tokens=False)

    blocks = []
    block_info = []
    current_pos = 0

    # Block 1
    block1 = (
        block1_prefix +
        [mask_token_id] * initial_tokens_list[0] +
        block1_suffix
    )
    block1_start = current_pos
    block1_end = current_pos + len(block1)
    block1_mask_offset = len(block1_prefix)
    block1_suffix_len = len(block1_suffix)
    blocks.append(block1)
    block_info.append((block1_start, block1_end, block1_mask_offset, block1_suffix_len))
    current_pos += len(block1)

    # Block 2
    block2 = (
        block2_prefix +
        [mask_token_id] * initial_tokens_list[1] +
        block2_suffix
    )
    block2_start = current_pos
    block2_end = current_pos + len(block2)
    block2_mask_offset = len(block2_prefix)
    block2_suffix_len = len(block2_suffix)
    blocks.append(block2)
    block_info.append((block2_start, block2_end, block2_mask_offset, block2_suffix_len))
    current_pos += len(block2)

    # Block 3
    block3 = (
        block3_prefix +
        [mask_token_id] * initial_tokens_list[2] +
        block3_suffix
    )
    block3_start = current_pos
    block3_end = current_pos + len(block3)
    block3_mask_offset = len(block3_prefix)
    block3_suffix_len = len(block3_suffix)
    blocks.append(block3)
    block_info.append((block3_start, block3_end, block3_mask_offset, block3_suffix_len))

    # Merge all blocks
    template_ids = torch.tensor(
        [token for block in blocks for token in block],
        dtype=torch.long,
        device=device
    )

    # Separator IDs: quote " and comma , (for end detection)
    quote_id = tokenizer.convert_tokens_to_ids('"')
    comma_id = tokenizer.convert_tokens_to_ids(',')
    separator_ids = torch.tensor([quote_id, comma_id], dtype=torch.long, device=device)

    return template_ids, block_info, separator_ids


def build_separator_token_sets(tokenizer, json_file="separator_tokens.json"):
    """
    Pre-build the set of tokens ending with separator characters and cache to a JSON file.

    Scans the entire vocabulary to find all token IDs whose decoded string
    ends with a comma or quote. These sets are used for fast lookup during
    confidence checking without repeated decoding.

    If the JSON file exists, loads from it directly.

    Args:
        tokenizer: PreTrainedTokenizer
        json_file: Path to the JSON file for caching token sets

    Returns:
        comma_ending_tokens: Set of token IDs ending with ','
        quote_ending_tokens: Set of token IDs ending with '"'
    """
    if os.path.exists(json_file):
        print(f"Loading separator token sets from: {json_file}")
        with open(json_file, 'r') as f:
            data = json.load(f)
            comma_ending_tokens = set(data['comma_ending_tokens'])
            quote_ending_tokens = set(data['quote_ending_tokens'])
            print(f"Loaded {len(comma_ending_tokens)} tokens ending with ','")
            print(f"Loaded {len(quote_ending_tokens)} tokens ending with '\"'")
            return comma_ending_tokens, quote_ending_tokens

    vocab_size = len(tokenizer)
    comma_ending_tokens = set()
    quote_ending_tokens = set()

    print(f"Building separator token sets from vocab (size={vocab_size})...")

    for token_id in range(vocab_size):
        try:
            token_str = tokenizer.decode([token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False)

            if token_str.endswith(','):
                comma_ending_tokens.add(token_id)

            if token_str.endswith('"'):
                quote_ending_tokens.add(token_id)

        except Exception:
            continue

    print(f"Found {len(comma_ending_tokens)} tokens ending with ','")
    print(f"Found {len(quote_ending_tokens)} tokens ending with '\"'")

    data = {
        'comma_ending_tokens': list(comma_ending_tokens),
        'quote_ending_tokens': list(quote_ending_tokens)
    }
    with open(json_file, 'w') as f:
        json.dump(data, f)
    print(f"Saved to: {json_file}")

    return comma_ending_tokens, quote_ending_tokens


def calculate_separator_confidence(
    block_logits: torch.Tensor,
    separator_token_sets: Tuple[set, set],
    mask_offset: int,
    confidence_threshold: float,
    is_last_block: bool = False,
    tokenizer = None,
    verbose: bool = False
) -> Tuple[float, int]:
    """
    Calculate separator confidence to determine the actual end position of a block.

    Based on the DIA algorithm. Uses a token set instead of a single token ID.

    Strategy: find the **first** separator position meeting the confidence threshold.
    - Non-last blocks: check all tokens ending with comma
    - Last block: check all tokens ending with quote

    Args:
        block_logits: [batch_size, block_len, vocab_size] logits for the entire block
        separator_token_sets: (comma_ending_tokens, quote_ending_tokens) token ID sets
        mask_offset: Start offset of the mask region within the block
        confidence_threshold: Confidence threshold
        is_last_block: Whether this is the last block
        tokenizer: Tokenizer for decoding (optional, debug only)
        verbose: Print detailed debug info

    Returns:
        confidence: Separator confidence (0.0 if not found)
        end_position: Separator position relative to block start (-1 if not found)
    """
    confidence = F.softmax(block_logits, dim=-1)
    predicted_tokens = torch.argmax(block_logits, dim=-1)

    comma_ending_tokens, quote_ending_tokens = separator_token_sets

    target_tokens = quote_ending_tokens if is_last_block else comma_ending_tokens
    target_name = '"' if is_last_block else ','

    if verbose and tokenizer:
        print(f"\n  Debug: Scanning for tokens ending with '{target_name}' ({len(target_tokens)} candidates)")
        predicted_sequence = predicted_tokens[0, mask_offset:].tolist()
        decoded_text = tokenizer.decode(predicted_sequence, skip_special_tokens=False)
        print(f"  Predicted text from mask_offset: {decoded_text[:100]}...")

        print(f"  First tokens from mask_offset:")
        for idx, i in enumerate(range(mask_offset, min(mask_offset + 10, block_logits.shape[1]))):
            token_id = predicted_tokens[0, i].item()
            token_conf = confidence[0, i, token_id].item()
            token_str = tokenizer.decode([token_id], skip_special_tokens=False)
            is_target = "ENDS_WITH_TARGET" if token_id in target_tokens else ""
            print(f"    [{i-mask_offset}] token_id={token_id:5d} conf={token_conf:.4f} '{token_str}' {is_target}")

    # Scan from mask region to block end, find the first matching position
    for i in range(mask_offset, block_logits.shape[1]):
        token_id = predicted_tokens[:, i].item()

        if token_id in target_tokens:
            conf = confidence[:, i, token_id].item()

            if verbose and tokenizer:
                token_str = tokenizer.decode([token_id], skip_special_tokens=False)
                print(f"  Found token ending with '{target_name}': '{token_str}' (id={token_id}) at position {i} (offset={i-mask_offset}) with confidence={conf:.4f}")

            if conf >= confidence_threshold:
                if verbose:
                    print(f"  Confidence >= threshold ({confidence_threshold:.4f}), accepting this position")
                return (float(conf), int(i))
            elif verbose:
                print(f"  Confidence < threshold ({confidence_threshold:.4f}), continue searching...")

    if verbose:
        print(f"  No token ending with '{target_name}' found with sufficient confidence")

    return (0.0, -1)


def expand_block(
    x: torch.Tensor,
    expand_size: int,
    block_info: Tuple[int, int, int, int],
    mask_token_id: int,
    is_last_block: bool = False
) -> torch.Tensor:
    """
    Expand a block by inserting mask tokens before its suffix.

    Key design: mask tokens are always inserted before the suffix,
    keeping the fixed JSON structure intact.

    Args:
        x: Current sequence [batch_size, seq_len]
        expand_size: Number of mask tokens to insert
        block_info: (block_start, block_end, mask_offset, suffix_len)
        mask_token_id: Mask token ID
        is_last_block: Whether this is the last block (kept for API compatibility)

    Returns:
        Expanded sequence
    """
    expand_seq = torch.full(
        (x.shape[0], expand_size),
        mask_token_id,
        dtype=torch.long,
        device=x.device
    )

    block_start, block_end, mask_offset, suffix_len = block_info

    # Insert before suffix (keep suffix fixed)
    insert_pos = block_end - suffix_len

    if insert_pos >= x.shape[1]:
        return torch.cat((x, expand_seq), dim=1)
    else:
        former_x = x[:, :insert_pos]
        latter_x = x[:, insert_pos:]
        return torch.cat((former_x, expand_seq, latter_x), dim=1)


def truncate_block(
    x: torch.Tensor,
    end_position: int,
    block_info: Tuple[int, int, int, int],
    is_last_block: bool = False
) -> torch.Tensor:
    """
    Truncate a block to its actual needed length.

    Args:
        x: Current sequence [batch_size, seq_len]
        end_position: Actual end position (absolute position)
        block_info: (block_start, block_end, mask_offset, suffix_len)
        is_last_block: Whether this is the last block

    Returns:
        Truncated sequence
    """
    block_start, block_end, mask_offset, suffix_len = block_info

    if is_last_block:
        # Last block: truncate then re-append the fixed suffix
        truncated = x[:, :end_position]
        suffix_tokens = x[:, block_end - suffix_len:block_end]
        return torch.cat((truncated, suffix_tokens), dim=1)
    else:
        # Non-last block: truncate and keep subsequent content
        return torch.cat((x[:, :end_position], x[:, block_end:]), dim=1)


def update_block_info(
    block_info_list: List[Tuple[int, int, int, int]],
    current_block_idx: int,
    delta: int,
    operation: str = "expand"
) -> List[Tuple[int, int, int, int]]:
    """
    Update all block info entries after an expand or truncate operation.

    - block_end changes
    - mask_offset and suffix_len stay the same
    - Subsequent blocks shift position

    Args:
        block_info_list: Current block info list [(block_start, block_end, mask_offset, suffix_len), ...]
        current_block_idx: Index of the block being operated on
        delta: Change amount (positive = expand, negative = truncate)
        operation: "expand" or "truncate"

    Returns:
        Updated block info list
    """
    new_info_list = []

    for i, (start, end, mask_offset, suffix_len) in enumerate(block_info_list):
        if i < current_block_idx:
            new_info_list.append((start, end, mask_offset, suffix_len))
        elif i == current_block_idx:
            if operation == "expand":
                new_info_list.append((start, end + delta, mask_offset, suffix_len))
            elif operation == "truncate":
                new_info_list.append((start, end - delta, mask_offset, suffix_len))
            else:
                raise ValueError(f"Unknown operation: {operation}")
        else:
            if operation == "expand":
                new_info_list.append((start + delta, end + delta, mask_offset, suffix_len))
            elif operation == "truncate":
                new_info_list.append((start - delta, end - delta, mask_offset, suffix_len))
            else:
                raise ValueError(f"Unknown operation: {operation}")

    return new_info_list


def sample_tokens(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    return_confidence: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sample tokens from logits with temperature, top-k, and top-p filtering.

    Args:
        logits: [num_masks, vocab_size]
        temperature: Temperature for scaling
        top_p: Nucleus sampling parameter
        top_k: Top-k sampling parameter
        return_confidence: Whether to return confidence scores

    Returns:
        confidence: Confidence of sampled tokens
        sampled_tokens: Sampled token IDs
    """
    if temperature != 1.0:
        logits = logits / temperature

    if top_k is not None and top_k > 0:
        top_k = min(top_k, logits.size(-1))
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits = logits.masked_fill(indices_to_remove, float('-inf'))

    if top_p is not None and top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices_to_remove.scatter(
            -1, sorted_indices, sorted_indices_to_remove
        )
        logits = logits.masked_fill(indices_to_remove, float('-inf'))

    probs = F.softmax(logits, dim=-1)
    sampled_tokens = torch.argmax(probs, dim=-1)

    if return_confidence:
        confidence = probs.gather(-1, sampled_tokens.unsqueeze(-1)).squeeze(-1)
        return confidence, sampled_tokens
    else:
        return sampled_tokens


def print_debug_info(
    tokenizer: PreTrainedTokenizer,
    x: torch.Tensor,
    block_info: List[Tuple[int, int, int, int]],
    current_block: int,
    step: int
):
    """
    Print debug information for the current generation state.

    Args:
        tokenizer: Tokenizer
        x: Current sequence
        block_info: Block info list [(block_start, block_end, mask_offset, suffix_len), ...]
        current_block: Current block index
        step: Current step number
    """
    print(f"\n{'='*60}")
    print(f"Step {step} - Block {current_block}")
    print(f"{'='*60}")

    for i, (start, end, mask_offset, suffix_len) in enumerate(block_info):
        block_tokens = x[0, start:end].tolist()
        block_text = tokenizer.decode(block_tokens, skip_special_tokens=False)
        marker = " <-- CURRENT" if i == current_block else ""
        print(f"Block {i}: [{start}:{end}] (mask offset: {mask_offset}, suffix len: {suffix_len}) {block_text}{marker}")

    print(f"\nFull generation:")
    full_text = tokenizer.decode(x[0].tolist(), skip_special_tokens=False)
    print(full_text[:300] + "..." if len(full_text) > 300 else full_text)


def generate_with_json_template(
    model,
    tokenizer: PreTrainedTokenizer,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    initial_tokens_per_field: Union[int, List[int]] = 8,
    max_tokens_per_field: Union[int, List[int]] = 32,
    steps_per_block: int = 16,
    temperature: float = 0.7,
    top_p: float = 0.95,
    top_k: int = 50,
    expand_delta: int = 1,
    confidence_threshold: float = 0.05,
    separator_token_sets: Optional[Tuple[set, set]] = None,
    verbose: bool = True
) -> torch.Tensor:
    """
    Generate using JSON template with dynamic block expansion (DIA method).

    Two-phase process per block:
    Phase 1 — Length adjustment: check separator confidence and expand/truncate
    Phase 2 — Diffusion generation: fill mask tokens via MaskGIT-style sampling

    Args:
        model: Dream model
        tokenizer: Tokenizer
        input_ids: Input token IDs [batch_size, input_length]
        attention_mask: Attention mask
        initial_tokens_per_field: Initial mask tokens per field.
                                  Can be a single int or a length-3 list.
        max_tokens_per_field: Maximum mask tokens per field (excluding prefix/suffix).
                              Can be a single int or a length-3 list.
        steps_per_block: Diffusion steps per block
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        top_k: Top-k sampling parameter
        expand_delta: Number of tokens to add per expansion
        confidence_threshold: Separator confidence threshold (below -> expand block)
        separator_token_sets: Optional pre-built (comma_ending_tokens, quote_ending_tokens).
                              If None, they are built (or loaded from cache) on first use.
                              For batch evaluation, build once and pass in to avoid
                              re-reading/re-scanning the vocabulary on every sample.
        verbose: Print detailed progress

    Returns:
        Generated full sequence [batch_size, total_length]
    """
    device = input_ids.device
    batch_size = input_ids.shape[0]

    if batch_size != 1:
        raise ValueError(f"Currently only supports batch_size=1, got {batch_size}")

    if isinstance(max_tokens_per_field, int):
        max_tokens_list = [max_tokens_per_field] * 3
    elif isinstance(max_tokens_per_field, (list, tuple)):
        if len(max_tokens_per_field) != 3:
            raise ValueError(f"max_tokens_per_field list must have length 3, got {len(max_tokens_per_field)}")
        max_tokens_list = list(max_tokens_per_field)
    else:
        raise TypeError(f"max_tokens_per_field must be int or list, got {type(max_tokens_per_field)}")

    template_ids, block_info, separator_ids = create_json_template_blocks(
        tokenizer=tokenizer,
        initial_tokens_per_field=initial_tokens_per_field,
        device=device
    )

    if separator_token_sets is None:
        separator_token_sets = build_separator_token_sets(tokenizer)

    x = torch.cat([input_ids, template_ids.unsqueeze(0)], dim=1)

    input_length = input_ids.shape[1]
    block_info = [
        (start + input_length, end + input_length, mask_offset, suffix_len)
        for start, end, mask_offset, suffix_len in block_info
    ]

    mask_token_id = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    if verbose:
        print(f"\n{'#'*60}")
        print(f"# JSON Template Generation Started")
        print(f"{'#'*60}")
        print(f"Input length: {input_length}")
        print(f"Template length: {template_ids.shape[0]}")
        print(f"Total length: {x.shape[1]}")
        print(f"Number of blocks: {len(block_info)}")
        print(f"Steps per block: {steps_per_block}")

    # Generate block by block
    for block_idx in range(len(block_info)):
        if verbose:
            print(f"\n{'='*60}")
            print(f"Processing Block {block_idx + 1}/{len(block_info)}")
            print(f"{'='*60}")

        # Phase 1: Dynamic block length adjustment
        while True:
            block_start, block_end, mask_offset, suffix_len = block_info[block_idx]

            current_mask_length = block_end - block_start - mask_offset - suffix_len

            if current_mask_length >= max_tokens_list[block_idx]:
                if verbose:
                    print(f"Mask reached max length ({max_tokens_list[block_idx]} tokens), skipping expansion")
                break

            with torch.no_grad():
                logits = model(x).logits
                logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
                block_logits = logits[:, block_start:block_end, :]

            is_last = (block_idx == len(block_info) - 1)
            conf, pos = calculate_separator_confidence(
                block_logits, separator_token_sets, mask_offset, confidence_threshold,
                is_last_block=is_last, tokenizer=tokenizer, verbose=verbose
            )

            if verbose:
                print(f"  Separator confidence: {conf:.4f} at position {pos}")

            if pos == -1:
                if verbose:
                    print(f"  Expanding block by {expand_delta} tokens")

                is_last = (block_idx == len(block_info) - 1)
                x = expand_block(x, expand_delta, block_info[block_idx], mask_token_id, is_last_block=is_last)
                block_info = update_block_info(
                    block_info, block_idx, expand_delta, operation="expand"
                )
            else:
                is_last = (block_idx == len(block_info) - 1)

                if is_last:
                    absolute_end = block_start + pos + 1
                else:
                    absolute_end = block_start + pos + 1

                if block_idx < len(block_info) - 1:
                    next_block_start = block_info[block_idx + 1][0]
                    if absolute_end < next_block_start:
                        absolute_end = next_block_start

                if verbose:
                    if is_last:
                        print(f"  Truncating block to position {absolute_end} (quote at position {pos} in block, absolute {block_start + pos})")
                    else:
                        print(f"  Truncating block to position {absolute_end} (quote at position {pos} in block, absolute {block_start + pos})")

                x = truncate_block(x, absolute_end, block_info[block_idx], is_last)

                old_block_end = block_info[block_idx][1]

                if is_last:
                    suffix_len = block_info[block_idx][3]
                    new_block_end = absolute_end + suffix_len
                else:
                    new_block_end = absolute_end

                reduced_length = old_block_end - new_block_end
                block_info = update_block_info(
                    block_info, block_idx, reduced_length, operation="truncate"
                )
                block_info[block_idx] = (block_info[block_idx][0], new_block_end, block_info[block_idx][2], block_info[block_idx][3])

                break

        # Phase 2: Diffusion generation to fill the block
        if verbose:
            print(f"\n  Starting diffusion generation for Block {block_idx + 1}")

        block_start, block_end, mask_offset, suffix_len = block_info[block_idx]
        is_last = (block_idx == len(block_info) - 1)

        mask_start = block_start + mask_offset
        mask_end = block_end - suffix_len

        timesteps = torch.linspace(1, 0.001, steps_per_block + 1, device=device)

        for step in range(steps_per_block):
            block_mask_index = (x[:, mask_start:mask_end] == mask_token_id)
            num_masks = block_mask_index.sum().item()

            if num_masks == 0:
                if verbose:
                    print(f"  All masks filled at step {step}")
                break

            with torch.no_grad():
                logits = model(x).logits
                logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
                block_logits = logits[:, mask_start:mask_end, :]
                mask_logits = block_logits[block_mask_index]

            t = timesteps[step]
            s = timesteps[step + 1]

            confidence, sampled_tokens = sample_tokens(
                mask_logits, temperature=temperature, top_p=top_p, top_k=top_k
            )

            num_to_fill = int(num_masks * (1 - s / t)) if step < steps_per_block - 1 else num_masks

            if num_to_fill > 0:
                _, indices = torch.topk(confidence, min(num_to_fill, len(confidence)))

                mask_positions = torch.where(block_mask_index)
                for i_idx, idx in enumerate(indices):
                    if i_idx < len(mask_positions[0]) and idx < len(sampled_tokens):
                        batch_idx = mask_positions[0][idx]
                        pos_idx = mask_positions[1][idx]
                        x[batch_idx, mask_start + pos_idx] = sampled_tokens[idx]

            if verbose and step % 4 == 0:
                current_text = tokenizer.decode(
                    x[0, mask_start:mask_end].tolist(),
                    skip_special_tokens=False
                )
                print(f"  Step {step:2d}/{steps_per_block}: {current_text[:80]}...")

        if verbose:
            final_text = tokenizer.decode(
                x[0, mask_start:mask_end].tolist(),
                skip_special_tokens=True
            )
            print(f"  Block {block_idx + 1} completed: {final_text}")

    if verbose:
        print(f"\n{'#'*60}")
        print(f"# Generation Complete")
        print(f"{'#'*60}")

    return x


if __name__ == "__main__":
    print("JSON Template Generation Module")
    print("This module provides block-based JSON generation for dLLM models")
