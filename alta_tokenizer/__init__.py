"""
ALTA Tokenizer - Hugging Face Edition

A high-performance Kinyarwanda tokenizer built on Hugging Face tokenizers library.
"""

from pathlib import Path
from typing import Optional, Union, List
from tokenizers import Tokenizer

__version__ = "3.0.0"
__author__ = "ALTA Team"


class ALTATokenizer:
    """
    ALTA Tokenizer wrapper for easy integration with Hugging Face tokenizers.

    This class provides a convenient interface for the trained Byte-Level BPE
    tokenizer optimized for Kinyarwanda language.

    Example:
        >>> from alta_tokenizer import ALTATokenizer
        >>> tokenizer = ALTATokenizer.from_pretrained("path/to/tokenizer.json")
        >>> tokens = tokenizer.encode("Muraho! Amakuru?")
        >>> text = tokenizer.decode(tokens)
    """

    def __init__(self, tokenizer: Tokenizer):
        """
        Initialize ALTA tokenizer.

        Args:
            tokenizer: A Hugging Face Tokenizer instance
        """
        self.tokenizer = tokenizer
        self._vocab_size = None

    @classmethod
    def from_pretrained(cls, tokenizer_path: Union[str, Path]) -> "ALTATokenizer":
        """
        Load a pretrained tokenizer from file.

        Args:
            tokenizer_path: Path to tokenizer.json file

        Returns:
            ALTATokenizer instance

        Raises:
            FileNotFoundError: If tokenizer file doesn't exist
        """
        tokenizer_path = Path(tokenizer_path)

        if not tokenizer_path.exists():
            raise FileNotFoundError(f"Tokenizer file not found: {tokenizer_path}")

        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        return cls(tokenizer)

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        """
        Encode text to token IDs.

        Args:
            text: Text to encode
            add_special_tokens: Whether to add special tokens (BOS/EOS)

        Returns:
            List of token IDs
        """
        encoding = self.tokenizer.encode(text, add_special_tokens=add_special_tokens)
        return encoding.ids

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs back to text.

        Args:
            ids: List of token IDs
            skip_special_tokens: Whether to skip special tokens in output

        Returns:
            Decoded text string
        """
        return self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

    def encode_batch(self, texts: List[str], add_special_tokens: bool = False) -> List[List[int]]:
        """
        Encode a batch of texts.

        Args:
            texts: List of texts to encode
            add_special_tokens: Whether to add special tokens

        Returns:
            List of token ID lists
        """
        encodings = self.tokenizer.encode_batch(texts, add_special_tokens=add_special_tokens)
        return [enc.ids for enc in encodings]

    def decode_batch(self, ids_list: List[List[int]], skip_special_tokens: bool = True) -> List[str]:
        """
        Decode a batch of token ID sequences.

        Args:
            ids_list: List of token ID lists
            skip_special_tokens: Whether to skip special tokens

        Returns:
            List of decoded text strings
        """
        return self.tokenizer.decode_batch(ids_list, skip_special_tokens=skip_special_tokens)

    @property
    def vocab_size(self) -> int:
        """Get vocabulary size."""
        if self._vocab_size is None:
            self._vocab_size = self.tokenizer.get_vocab_size()
        return self._vocab_size

    @property
    def vocab(self) -> dict:
        """Get vocabulary mapping."""
        return self.tokenizer.get_vocab()

    def token_to_id(self, token: str) -> Optional[int]:
        """
        Convert token string to ID.

        Args:
            token: Token string

        Returns:
            Token ID or None if not found
        """
        return self.tokenizer.token_to_id(token)

    def id_to_token(self, id: int) -> Optional[str]:
        """
        Convert token ID to string.

        Args:
            id: Token ID

        Returns:
            Token string or None if not found
        """
        return self.tokenizer.id_to_token(id)

    def save(self, path: Union[str, Path], pretty: bool = True):
        """
        Save tokenizer to file.

        Args:
            path: Path to save tokenizer
            pretty: Whether to save in pretty format
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.tokenizer.save(str(path), pretty=pretty)

    def enable_padding(
        self,
        direction: str = "right",
        pad_id: int = 0,
        pad_token: str = "<|pad|>",
        length: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None
    ):
        """
        Enable padding for batch processing.

        Args:
            direction: Padding direction ("left" or "right")
            pad_id: ID of padding token
            pad_token: String representation of padding token
            length: Fixed length to pad to (None for dynamic)
            pad_to_multiple_of: Pad to multiple of this value
        """
        self.tokenizer.enable_padding(
            direction=direction,
            pad_id=pad_id,
            pad_token=pad_token,
            length=length,
            pad_to_multiple_of=pad_to_multiple_of
        )

    def enable_truncation(
        self,
        max_length: int,
        stride: int = 0,
        strategy: str = "longest_first"
    ):
        """
        Enable truncation for sequences.

        Args:
            max_length: Maximum sequence length
            stride: Stride for truncation
            strategy: Truncation strategy
        """
        self.tokenizer.enable_truncation(
            max_length=max_length,
            stride=stride,
            strategy=strategy
        )

    def __repr__(self) -> str:
        """String representation."""
        return f"ALTATokenizer(vocab_size={self.vocab_size})"


# Convenience function for quick loading
def load_tokenizer(path: Union[str, Path]) -> ALTATokenizer:
    """
    Load ALTA tokenizer from file.

    Args:
        path: Path to tokenizer.json

    Returns:
        ALTATokenizer instance
    """
    return ALTATokenizer.from_pretrained(path)
