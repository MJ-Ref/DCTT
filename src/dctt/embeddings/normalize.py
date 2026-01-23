"""Embedding normalization utilities.

This module provides functions for normalizing embeddings, which is
required for meaningful cosine distance computation.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def compute_norms(embeddings: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute L2 norms of embedding vectors.

    Args:
        embeddings: Embedding matrix of shape (vocab_size, dim).

    Returns:
        Array of norms with shape (vocab_size,).
    """
    return np.linalg.norm(embeddings, axis=1)


def normalize_embeddings(
    embeddings: NDArray[np.float64],
    eps: float = 1e-10,
    return_norms: bool = False,
) -> NDArray[np.float64] | tuple[NDArray[np.float64], NDArray[np.float64]]:
    """L2-normalize embedding vectors.

    Normalizes each row (embedding vector) to unit length. This is
    required for meaningful cosine distance computation and is the
    standard preprocessing for DCTT analysis.

    Args:
        embeddings: Embedding matrix of shape (vocab_size, dim).
        eps: Small constant to prevent division by zero.
        return_norms: If True, also return the original norms.

    Returns:
        If return_norms=False: Normalized embeddings of shape (vocab_size, dim).
        If return_norms=True: Tuple of (normalized_embeddings, norms).
    """
    norms = compute_norms(embeddings)
    # Avoid division by zero
    norms_safe = np.maximum(norms, eps)
    normalized = embeddings / norms_safe[:, np.newaxis]

    if return_norms:
        return normalized, norms
    return normalized


def check_normalization(
    embeddings: NDArray[np.float64],
    tolerance: float = 1e-6,
) -> bool:
    """Check if embeddings are already L2-normalized.

    Args:
        embeddings: Embedding matrix of shape (vocab_size, dim).
        tolerance: Maximum deviation from unit norm.

    Returns:
        True if all embeddings have unit norm (within tolerance).
    """
    norms = compute_norms(embeddings)
    return np.allclose(norms, 1.0, atol=tolerance)


def normalize_single(
    embedding: NDArray[np.float64],
    eps: float = 1e-10,
) -> NDArray[np.float64]:
    """L2-normalize a single embedding vector.

    Args:
        embedding: Single embedding vector of shape (dim,).
        eps: Small constant to prevent division by zero.

    Returns:
        Normalized embedding of shape (dim,).
    """
    norm = np.linalg.norm(embedding)
    return embedding / max(norm, eps)
