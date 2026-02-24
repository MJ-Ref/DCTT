"""Embedding extraction from language models.

This module provides utilities for extracting token embedding matrices
from HuggingFace transformer models, with support for caching and
various model architectures.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from dctt.core.exceptions import EmbeddingExtractionError

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizer

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingInfo:
    """Information about extracted embeddings.

    Attributes:
        model_name: HuggingFace model identifier.
        model_revision: Model revision/commit hash.
        vocab_size: Number of tokens in vocabulary.
        embedding_dim: Dimensionality of embeddings.
        embedding_source: Which weight matrix was extracted.
        dtype: Data type of embeddings.
        hash: SHA256 hash of embedding matrix (first 16 chars).
    """

    model_name: str
    model_revision: str
    vocab_size: int
    embedding_dim: int
    embedding_source: str
    dtype: str
    hash: str

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "model_name": self.model_name,
            "model_revision": self.model_revision,
            "vocab_size": self.vocab_size,
            "embedding_dim": self.embedding_dim,
            "embedding_source": self.embedding_source,
            "dtype": self.dtype,
            "hash": self.hash,
        }


class EmbeddingExtractor:
    """Extracts token embeddings from HuggingFace models.

    Supports various model architectures and handles the differences in
    how embeddings are stored (tied vs untied, different attribute names).

    Example:
        >>> extractor = EmbeddingExtractor()
        >>> embeddings, tokenizer = extractor.extract("Qwen/Qwen2.5-7B")
        >>> print(embeddings.shape)
        (151936, 3584)
    """

    # Common embedding attribute paths for different model families
    EMBEDDING_PATHS = {
        "default": [
            "model.embed_tokens.weight",
            "transformer.wte.weight",
            "embeddings.word_embeddings.weight",
            "word_embedding.weight",
        ],
        "qwen2": ["model.embed_tokens.weight"],
        "llama": ["model.embed_tokens.weight"],
        "gpt2": ["transformer.wte.weight"],
        "bert": ["embeddings.word_embeddings.weight"],
    }

    def __init__(
        self,
        device: str = "cpu",
        torch_dtype: str | None = None,
        trust_remote_code: bool = True,
        low_cpu_mem_usage: bool = True,
    ) -> None:
        """Initialize extractor.

        Args:
            device: Device to load model on ("cpu", "mps", "cuda").
            torch_dtype: Data type for model loading (e.g., "bfloat16").
            trust_remote_code: Whether to trust remote code for custom models.
            low_cpu_mem_usage: Use memory-efficient loading.
        """
        self.device = device
        self.torch_dtype = torch_dtype
        self.trust_remote_code = trust_remote_code
        self.low_cpu_mem_usage = low_cpu_mem_usage

    def extract(
        self,
        model_name: str,
        revision: str = "main",
        embedding_source: str | None = None,
    ) -> tuple[NDArray[np.float64], "PreTrainedTokenizer"]:
        """Extract embeddings from a HuggingFace model.

        Args:
            model_name: HuggingFace model identifier (e.g., "Qwen/Qwen2.5-7B").
            revision: Model revision/commit hash.
            embedding_source: Specific attribute path to extract (auto-detected if None).

        Returns:
            Tuple of (embeddings, tokenizer):
                - embeddings: Float64 array of shape (vocab_size, embedding_dim).
                - tokenizer: The model's tokenizer.

        Raises:
            EmbeddingExtractionError: If extraction fails.
        """
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:
            raise EmbeddingExtractionError(
                model_name, "transformers not installed"
            ) from e

        logger.info(f"Loading model: {model_name} (revision: {revision})")

        try:
            # Determine torch dtype
            dtype = None
            if self.torch_dtype:
                dtype = getattr(torch, self.torch_dtype, None)

            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                revision=revision,
                trust_remote_code=self.trust_remote_code,
            )

            # Load model
            device_map = self.device if self.device not in {"cpu", "mps"} else None
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                revision=revision,
                torch_dtype=dtype,
                trust_remote_code=self.trust_remote_code,
                low_cpu_mem_usage=self.low_cpu_mem_usage,
                # MPS device_map can overflow allocator warmup for full-sized models.
                device_map=device_map,
            )

            # Extract embeddings
            embeddings = self._extract_embeddings(
                model, model_name, embedding_source
            )

            logger.info(
                f"Extracted embeddings: shape={embeddings.shape}, "
                f"dtype={embeddings.dtype}"
            )

            return embeddings, tokenizer

        except Exception as e:
            raise EmbeddingExtractionError(model_name, str(e)) from e

    def _extract_embeddings(
        self,
        model: "PreTrainedModel",
        model_name: str,
        embedding_source: str | None,
    ) -> NDArray[np.float64]:
        """Extract embedding weights from model.

        Args:
            model: Loaded HuggingFace model.
            model_name: Model identifier for error messages.
            embedding_source: Specific attribute path, or None for auto-detect.

        Returns:
            Embedding matrix as float64 numpy array.
        """
        import torch

        # Determine which paths to try
        if embedding_source:
            paths_to_try = [embedding_source]
        else:
            # Determine model family
            family = self._detect_model_family(model_name)
            paths_to_try = self.EMBEDDING_PATHS.get(
                family, self.EMBEDDING_PATHS["default"]
            )

        # Try each path
        for path in paths_to_try:
            tensor = self._get_nested_attr(model, path)
            if tensor is not None:
                logger.info(f"Found embeddings at: {path}")
                # Convert to numpy float64
                if isinstance(tensor, torch.nn.Parameter):
                    tensor = tensor.data
                return tensor.detach().cpu().float().numpy().astype(np.float64)

        # Fallback: look for any embedding layer
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Embedding) and "embed" in name.lower():
                logger.info(f"Found embeddings via module search: {name}")
                return (
                    module.weight.detach().cpu().float().numpy().astype(np.float64)
                )

        raise EmbeddingExtractionError(
            model_name,
            f"Could not find embedding weights. Tried paths: {paths_to_try}",
        )

    def _detect_model_family(self, model_name: str) -> str:
        """Detect model family from name."""
        model_lower = model_name.lower()
        if "qwen" in model_lower:
            return "qwen2"
        if "llama" in model_lower:
            return "llama"
        if "gpt2" in model_lower or "gpt-2" in model_lower:
            return "gpt2"
        if "bert" in model_lower:
            return "bert"
        return "default"

    def _get_nested_attr(self, obj: object, path: str) -> object | None:
        """Get nested attribute from object using dot notation."""
        parts = path.split(".")
        current = obj
        for part in parts:
            if hasattr(current, part):
                current = getattr(current, part)
            else:
                return None
        return current


def extract_embeddings(
    model_name: str,
    revision: str = "main",
    device: str = "cpu",
    torch_dtype: str | None = "bfloat16",
    trust_remote_code: bool = True,
    embedding_source: str | None = None,
) -> tuple[NDArray[np.float64], "PreTrainedTokenizer"]:
    """Convenience function to extract embeddings.

    Args:
        model_name: HuggingFace model identifier.
        revision: Model revision.
        device: Device to load model on.
        torch_dtype: Data type for model loading.
        trust_remote_code: Whether to trust remote code.
        embedding_source: Specific attribute path to extract.

    Returns:
        Tuple of (embeddings, tokenizer).
    """
    extractor = EmbeddingExtractor(
        device=device,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
    )
    return extractor.extract(model_name, revision, embedding_source)


def get_embedding_info(
    embeddings: NDArray[np.float64],
    model_name: str,
    model_revision: str = "main",
    embedding_source: str = "auto",
) -> EmbeddingInfo:
    """Get information about extracted embeddings.

    Args:
        embeddings: Extracted embedding matrix.
        model_name: Model identifier.
        model_revision: Model revision.
        embedding_source: How embeddings were extracted.

    Returns:
        EmbeddingInfo with metadata about the embeddings.
    """
    vocab_size, embedding_dim = embeddings.shape
    emb_hash = hashlib.sha256(embeddings.tobytes()).hexdigest()[:16]

    return EmbeddingInfo(
        model_name=model_name,
        model_revision=model_revision,
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        embedding_source=embedding_source,
        dtype=str(embeddings.dtype),
        hash=emb_hash,
    )
