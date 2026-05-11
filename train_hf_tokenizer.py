#!/usr/bin/env python3
"""
ALTA Tokenizer Training Script using Hugging Face Tokenizers
============================================================

This script trains a Byte-Level BPE tokenizer optimized for Kinyarwanda language
using the Hugging Face tokenizers library.

Key Features:
- Byte-Level BPE with byte_fallback=True (no OOV errors)
- Vocabulary size: 65,536 tokens (2^16 for hardware optimization)
- Custom regex pre-tokenizer that preserves Kinyarwanda apostrophes
- 20 special tokens for various NLP tasks
- Memory-efficient streaming data loader
- Multi-core CPU optimization

Usage:
    python train_hf_tokenizer.py --data_path <path_to_data> --output_dir <output_directory>

Author: ALTA Team
"""

import argparse
import json
import os
from pathlib import Path
from typing import Iterator, Union, List
import multiprocessing

from tokenizers import Tokenizer, normalizers, pre_tokenizers, decoders, Regex
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.normalizers import NFD, StripAccents, Lowercase, Replace, Sequence
from tokenizers.pre_tokenizers import Split, Sequence as PreSeq


def create_data_iterator(
    data_path: Union[str, Path],
    batch_size: int = 1000
) -> Iterator[List[str]]:
    """
    Memory-efficient generator that streams text from datasets.

    Supports multiple formats:
    - Plain text files (.txt)
    - JSONL files (.jsonl) with 'text' field
    - Directory of text/jsonl files

    Args:
        data_path: Path to data file or directory
        batch_size: Number of texts to yield at once

    Yields:
        Batches of text strings
    """
    data_path = Path(data_path)

    def process_file(file_path: Path) -> Iterator[str]:
        """Process a single file and yield text lines."""
        if file_path.suffix == '.jsonl':
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        if 'text' in data:
                            yield data['text']
                    except json.JSONDecodeError:
                        continue
        elif file_path.suffix == '.txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:  # Skip empty lines
                        yield line
        else:
            print(f"Warning: Unsupported file format {file_path.suffix}, skipping...")

    # Collect all files to process
    if data_path.is_file():
        files = [data_path]
    elif data_path.is_dir():
        files = list(data_path.glob('**/*.txt')) + list(data_path.glob('**/*.jsonl'))
        if not files:
            raise ValueError(f"No .txt or .jsonl files found in {data_path}")
    else:
        raise ValueError(f"Data path {data_path} does not exist")

    print(f"Found {len(files)} file(s) to process")

    # Stream data in batches
    batch = []
    for file_path in files:
        print(f"Processing: {file_path}")
        for text in process_file(file_path):
            batch.append(text)
            if len(batch) >= batch_size:
                yield batch
                batch = []

    # Yield remaining items
    if batch:
        yield batch


def create_kinyarwanda_tokenizer(vocab_size: int = 65536) -> Tokenizer:
    """
    Create and configure a Byte-Level BPE tokenizer for Kinyarwanda.

    Args:
        vocab_size: Target vocabulary size (default: 65536)

    Returns:
        Configured Tokenizer instance
    """
    # Initialize BPE model with byte_fallback for no OOV errors
    tokenizer = Tokenizer(BPE(byte_fallback=True))

    # Define special tokens (20 tokens as specified)
    special_tokens = [
        "<|pad|>",
        "<|unk|>",
        "<|bos|>",
        "<|eos|>",
        "<|im_start|>",
        "<|im_end|>",
        "<|begin_of_text|>",
        "<|start_header_id|>",
        "<|end_header_id|>",
        "<|eot_id|>",
        "<|thought_start|>",
        "<|thought_end|>",
        "<|tool_start|>",
        "<|tool_end|>",
        "<|tool_call|>",
        "<|tool_result|>",
        "<|fim_prefix|>",
        "<|fim_middle|>",
        "<|fim_suffix|>",
        "<|fim_pad|>",
    ]

    # Configure trainer
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        show_progress=True,
        initial_alphabet=[]  # Let byte-level BPE handle the alphabet
    )

    # Normalization: Basic Unicode normalization
    # We keep it minimal to preserve Kinyarwanda-specific characters
    tokenizer.normalizer = normalizers.Sequence([
        normalizers.Replace("'", "'"),  # Normalize fancy apostrophes to standard
        normalizers.Replace("'", "'"),
        normalizers.Replace(""", '"'),  # Normalize fancy quotes
        normalizers.Replace(""", '"'),
        normalizers.NFD(),  # Unicode normalization
    ])

    # Custom Pre-tokenizer for Kinyarwanda
    # Key insight: Kinyarwanda uses apostrophes in words (e.g., "w'abatwa", "n'inka")
    # We create a regex that treats apostrophes as part of words, not delimiters

    # This regex pattern:
    # - \p{L}+(?:'\p{L}+)*  : Matches letter sequences with optional apostrophes inside
    # - \p{N}+              : Matches numbers
    # - \p{P}               : Matches punctuation (split from words)
    # - \s+                 : Matches whitespace
    # - [^\s\p{L}\p{N}\p{P}]+ : Matches any other characters

    kinyarwanda_pattern = r"""(?x)
        \p{L}+(?:'\p{L}+)*   # Words with optional apostrophes (e.g., w'abatwa)
        | \p{N}+             # Numbers
        | \p{P}              # Punctuation (separated)
        | \s+                # Whitespace
        | [^\s\p{L}\p{N}\p{P}]+  # Other characters
    """

    tokenizer.pre_tokenizer = pre_tokenizers.Split(
        pattern=Regex(kinyarwanda_pattern.strip()),
        behavior="isolated"
    )

    # Decoder: ByteLevel decoder to convert back from byte-level tokens
    tokenizer.decoder = decoders.ByteLevel()

    # Store configuration for later reference
    tokenizer.trainer = trainer

    return tokenizer


def train_tokenizer(
    data_path: Union[str, Path],
    output_dir: Union[str, Path],
    vocab_size: int = 65536,
    batch_size: int = 1000
) -> Tokenizer:
    """
    Train the Kinyarwanda tokenizer on the provided dataset.

    Args:
        data_path: Path to training data (file or directory)
        output_dir: Directory to save the trained tokenizer
        vocab_size: Target vocabulary size
        batch_size: Batch size for data streaming

    Returns:
        Trained tokenizer
    """
    print("="*70)
    print("ALTA Tokenizer Training - Hugging Face Backend")
    print("="*70)
    print(f"Configuration:")
    print(f"  - Algorithm: Byte-Level BPE with byte_fallback=True")
    print(f"  - Vocabulary Size: {vocab_size:,}")
    print(f"  - Data Path: {data_path}")
    print(f"  - Output Directory: {output_dir}")
    print(f"  - CPU Cores: {multiprocessing.cpu_count()}")
    print(f"  - Batch Size: {batch_size}")
    print("="*70)

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create tokenizer
    print("\n[1/3] Creating tokenizer with Kinyarwanda-specific configuration...")
    tokenizer = create_kinyarwanda_tokenizer(vocab_size=vocab_size)

    # Create data iterator
    print("\n[2/3] Streaming training data...")
    data_iterator = create_data_iterator(data_path, batch_size=batch_size)

    # Train tokenizer
    print("\n[3/3] Training tokenizer (this may take a while)...")
    tokenizer.train_from_iterator(
        data_iterator,
        trainer=tokenizer.trainer,
        length=None  # Unknown length for streaming
    )

    # Save tokenizer
    print("\n[4/4] Saving tokenizer...")
    tokenizer_path = output_dir / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))
    print(f"✓ Tokenizer saved to: {tokenizer_path}")

    # Save configuration
    config = {
        "vocab_size": vocab_size,
        "model_type": "BPE",
        "byte_fallback": True,
        "special_tokens": [
            "<|pad|>", "<|unk|>", "<|bos|>", "<|eos|>",
            "<|im_start|>", "<|im_end|>", "<|begin_of_text|>",
            "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>",
            "<|thought_start|>", "<|thought_end|>",
            "<|tool_start|>", "<|tool_end|>", "<|tool_call|>", "<|tool_result|>",
            "<|fim_prefix|>", "<|fim_middle|>", "<|fim_suffix|>", "<|fim_pad|>"
        ],
        "language": "Kinyarwanda",
        "apostrophe_handling": "preserved_in_words"
    }

    config_path = output_dir / "tokenizer_config.json"
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"✓ Config saved to: {config_path}")

    print("\n" + "="*70)
    print("Training completed successfully!")
    print("="*70)

    return tokenizer


def test_tokenizer(tokenizer: Tokenizer):
    """
    Test the trained tokenizer with sample Kinyarwanda text.

    Args:
        tokenizer: Trained tokenizer to test
    """
    print("\n" + "="*70)
    print("Testing Tokenizer")
    print("="*70)

    # Test samples with apostrophes (Kinyarwanda elision)
    test_samples = [
        "Nuko Semugeshi akunda uwo mutwe w'abatwa.",
        "Umwana w'umwami n'inka y'umutware.",
        "Yagiye ku isoko n'inshuti ze.",
        "Hari abantu b'inyangamugayo.",
    ]

    print("\nSample encodings:\n")
    for text in test_samples:
        # Encode
        encoded = tokenizer.encode(text)
        tokens = encoded.tokens
        ids = encoded.ids

        # Decode
        decoded = tokenizer.decode(ids)

        print(f"Text:    {text}")
        print(f"Tokens:  {tokens[:10]}..." if len(tokens) > 10 else f"Tokens:  {tokens}")
        print(f"IDs:     {ids[:10]}..." if len(ids) > 10 else f"IDs:     {ids}")
        print(f"Decoded: {decoded}")
        print(f"Match:   {'✓' if decoded.strip() == text else '✗'}")
        print()

    # Test special tokens
    print("Testing special tokens:")
    special_text = "<|bos|>Muraho!<|eos|>"
    encoded = tokenizer.encode(special_text)
    decoded = tokenizer.decode(encoded.ids)
    print(f"Input:   {special_text}")
    print(f"Decoded: {decoded}")
    print(f"Match:   {'✓' if decoded == special_text else '✗'}")

    print("\n" + "="*70)


def main():
    """Main entry point for the training script."""
    parser = argparse.ArgumentParser(
        description="Train ALTA Kinyarwanda Tokenizer using Hugging Face Tokenizers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train on a single text file
  python train_hf_tokenizer.py --data_path corpus.txt --output_dir ./tokenizer_output

  # Train on a directory of files
  python train_hf_tokenizer.py --data_path ./data/ --output_dir ./tokenizer_output

  # Train with custom vocabulary size
  python train_hf_tokenizer.py --data_path corpus.txt --output_dir ./tokenizer_output --vocab_size 32768
        """
    )

    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to training data (text file, JSONL file, or directory)"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the trained tokenizer"
    )

    parser.add_argument(
        "--vocab_size",
        type=int,
        default=65536,
        help="Vocabulary size (default: 65536 for 2^16 hardware optimization)"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=1000,
        help="Batch size for streaming data (default: 1000)"
    )

    parser.add_argument(
        "--test",
        action="store_true",
        help="Run tests after training"
    )

    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Data path not found: {args.data_path}")

    if args.vocab_size < 1000:
        raise ValueError("Vocabulary size must be at least 1000")

    # Train tokenizer
    tokenizer = train_tokenizer(
        data_path=args.data_path,
        output_dir=args.output_dir,
        vocab_size=args.vocab_size,
        batch_size=args.batch_size
    )

    # Test if requested
    if args.test:
        test_tokenizer(tokenizer)

    print("\n✓ All done! You can now use your tokenizer with:")
    print(f"  from tokenizers import Tokenizer")
    print(f"  tokenizer = Tokenizer.from_file('{args.output_dir}/tokenizer.json')")


if __name__ == "__main__":
    main()
