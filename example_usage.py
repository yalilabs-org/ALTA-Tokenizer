#!/usr/bin/env python3
"""
Example usage of ALTA Tokenizer (Hugging Face Edition)

This script demonstrates how to use the trained tokenizer for various tasks.
"""

from alta_tokenizer import ALTATokenizer
from pathlib import Path


def example_basic_usage():
    """Basic encoding and decoding example."""
    print("="*70)
    print("Example 1: Basic Usage")
    print("="*70)

    # Load tokenizer (update path to your trained tokenizer)
    tokenizer = ALTATokenizer.from_pretrained("./trained_tokenizer/tokenizer.json")

    # Sample Kinyarwanda text
    text = "Nuko Semugeshi akunda uwo mutwe w'abatwa."

    # Encode
    token_ids = tokenizer.encode(text)
    print(f"\nOriginal text: {text}")
    print(f"Token IDs: {token_ids}")
    print(f"Number of tokens: {len(token_ids)}")

    # Decode
    decoded_text = tokenizer.decode(token_ids)
    print(f"Decoded text: {decoded_text}")
    print(f"Match: {'✓' if decoded_text == text else '✗'}")


def example_apostrophe_handling():
    """Demonstrate apostrophe handling in Kinyarwanda."""
    print("\n" + "="*70)
    print("Example 2: Apostrophe Handling (Kinyarwanda Elision)")
    print("="*70)

    tokenizer = ALTATokenizer.from_pretrained("./trained_tokenizer/tokenizer.json")

    # Words with apostrophes (common in Kinyarwanda)
    test_cases = [
        "w'abatwa",           # with the Batwa
        "n'inka",             # and cows
        "y'umutware",         # of the chief
        "b'inyangamugayo",    # of honesty
        "Umwana w'umwami"     # The child of the king
    ]

    print("\nApostrophe preservation in action:\n")
    for text in test_cases:
        token_ids = tokenizer.encode(text)
        print(f"{text:25} → {len(token_ids):2} token(s)")


def example_special_tokens():
    """Using special tokens for structured text."""
    print("\n" + "="*70)
    print("Example 3: Special Tokens")
    print("="*70)

    tokenizer = ALTATokenizer.from_pretrained("./trained_tokenizer/tokenizer.json")

    # Chat format example
    chat_text = "<|bos|><|im_start|>user\nMuraho! Amakuru?<|im_end|><|im_start|>assistant\nNi meza, murakoze!<|im_end|><|eos|>"

    token_ids = tokenizer.encode(chat_text)
    decoded = tokenizer.decode(token_ids, skip_special_tokens=False)

    print("\nChat-formatted text:")
    print(f"Original: {chat_text}")
    print(f"Tokens: {len(token_ids)}")
    print(f"Decoded: {decoded}")


def example_batch_processing():
    """Batch encoding and decoding."""
    print("\n" + "="*70)
    print("Example 4: Batch Processing")
    print("="*70)

    tokenizer = ALTATokenizer.from_pretrained("./trained_tokenizer/tokenizer.json")

    # Enable padding for batch processing
    tokenizer.enable_padding(
        direction="right",
        pad_id=0,
        pad_token="<|pad|>",
        length=None  # Dynamic padding
    )

    # Batch of texts with different lengths
    texts = [
        "Muraho!",
        "Amakuru yawe?",
        "Nuko Semugeshi akunda uwo mutwe w'abatwa.",
        "Ejo hazaza abashyitsi."
    ]

    print("\nBatch encoding:")
    encoded_batch = tokenizer.encode_batch(texts)

    for i, (text, ids) in enumerate(zip(texts, encoded_batch)):
        print(f"{i+1}. {text:45} → {len(ids):2} tokens")

    # Decode batch
    decoded_batch = tokenizer.decode_batch(encoded_batch)
    print("\nDecoded batch matches:")
    for orig, decoded in zip(texts, decoded_batch):
        match = "✓" if orig.strip() == decoded.strip() else "✗"
        print(f"{match} {orig}")


def example_compression_analysis():
    """Analyze compression rate."""
    print("\n" + "="*70)
    print("Example 5: Compression Analysis")
    print("="*70)

    tokenizer = ALTATokenizer.from_pretrained("./trained_tokenizer/tokenizer.json")

    # Sample text
    text = """
    Nuko Semugeshi akunda uwo mutwe w'abatwa, bitwaga Ishabi, awugira intore ze.
    Bukeye bataha biyereka ingabo; dore ko hambere nta mihamirizo yindi yabaga mu Rwanda;
    guhamiriza byaje vuba biturutse i Burundi. Ubwo bataha Semugeshi n'abatware be
    barabitegereza basanga ari abahanga bose.
    """.strip()

    # Encode
    token_ids = tokenizer.encode(text)

    # Calculate metrics
    char_count = len(text)
    token_count = len(token_ids)
    compression_ratio = char_count / token_count

    print(f"\nText analysis:")
    print(f"Characters: {char_count:,}")
    print(f"Tokens: {token_count:,}")
    print(f"Compression ratio: {compression_ratio:.2f}x")
    print(f"Efficiency: {(1 - token_count/char_count) * 100:.1f}% size reduction")


def example_vocabulary_inspection():
    """Inspect vocabulary."""
    print("\n" + "="*70)
    print("Example 6: Vocabulary Inspection")
    print("="*70)

    tokenizer = ALTATokenizer.from_pretrained("./trained_tokenizer/tokenizer.json")

    print(f"\nVocabulary size: {tokenizer.vocab_size:,}")

    # Check special tokens
    special_tokens = [
        "<|pad|>", "<|unk|>", "<|bos|>", "<|eos|>",
        "<|im_start|>", "<|im_end|>", "<|tool_call|>"
    ]

    print("\nSpecial token IDs:")
    for token in special_tokens:
        token_id = tokenizer.token_to_id(token)
        if token_id is not None:
            print(f"  {token:20} → ID: {token_id}")
        else:
            print(f"  {token:20} → Not found")

    # Show some vocabulary samples
    print("\nSample tokens from vocabulary:")
    vocab = tokenizer.vocab
    sample_tokens = list(vocab.items())[:20]
    for token, idx in sample_tokens:
        # Display printable tokens only
        if len(token) < 30 and token.isprintable():
            print(f"  {token:30} → {idx}")


def main():
    """Run all examples."""
    print("\n")
    print("╔" + "="*68 + "╗")
    print("║" + " "*10 + "ALTA Tokenizer - Usage Examples" + " "*27 + "║")
    print("║" + " "*15 + "(Hugging Face Edition)" + " "*32 + "║")
    print("╚" + "="*68 + "╝")

    try:
        example_basic_usage()
        example_apostrophe_handling()
        example_special_tokens()
        example_batch_processing()
        example_compression_analysis()
        example_vocabulary_inspection()

        print("\n" + "="*70)
        print("✓ All examples completed successfully!")
        print("="*70 + "\n")

    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\nPlease train a tokenizer first using:")
        print("  python train_hf_tokenizer.py --data_path <your_data> --output_dir ./trained_tokenizer")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
