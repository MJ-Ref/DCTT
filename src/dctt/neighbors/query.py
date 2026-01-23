"""Utility functions for kNN queries.

This module provides convenience functions for common query patterns
used throughout the DCTT pipeline.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from dctt.neighbors.index import VectorIndex


def query_neighbors(
    index: VectorIndex,
    embeddings: NDArray[np.float64],
    token_id: int,
    k: int,
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """Query k nearest neighbors for a single token.

    Args:
        index: Built vector index.
        embeddings: Embedding matrix.
        token_id: Index of the token to query.
        k: Number of neighbors to return.

    Returns:
        Tuple of (neighbor_indices, distances).
    """
    query_vector = embeddings[token_id]
    return index.query_single(
        query_vector,
        k=k,
        exclude_self=True,
        self_index=token_id,
    )


def batch_query_neighbors(
    index: VectorIndex,
    embeddings: NDArray[np.float64],
    token_ids: list[int] | NDArray[np.int64],
    k: int,
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """Query k nearest neighbors for multiple tokens.

    Args:
        index: Built vector index.
        embeddings: Embedding matrix.
        token_ids: Indices of tokens to query.
        k: Number of neighbors per token.

    Returns:
        Tuple of (neighbor_indices, distances), both of shape (n_tokens, k).
    """
    token_ids = np.asarray(token_ids)
    query_vectors = embeddings[token_ids]
    return index.query(query_vectors, k=k, exclude_self=True)


def get_neighbors_for_all_tokens(
    index: VectorIndex,
    embeddings: NDArray[np.float64],
    k: int,
    batch_size: int = 10000,
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """Query k nearest neighbors for all tokens in vocabulary.

    Processes in batches to manage memory usage.

    Args:
        index: Built vector index.
        embeddings: Embedding matrix of shape (vocab_size, dim).
        k: Number of neighbors per token.
        batch_size: Number of tokens to process per batch.

    Returns:
        Tuple of (neighbor_indices, distances), both of shape (vocab_size, k).
    """
    vocab_size = embeddings.shape[0]
    all_indices = np.zeros((vocab_size, k), dtype=np.int64)
    all_distances = np.zeros((vocab_size, k), dtype=np.float64)

    for start in range(0, vocab_size, batch_size):
        end = min(start + batch_size, vocab_size)
        token_ids = np.arange(start, end)
        indices, distances = batch_query_neighbors(index, embeddings, token_ids, k)
        all_indices[start:end] = indices
        all_distances[start:end] = distances

    return all_indices, all_distances


def compute_jaccard_overlap(
    neighbors_a: NDArray[np.int64],
    neighbors_b: NDArray[np.int64],
) -> float:
    """Compute Jaccard overlap between two neighbor sets.

    Used for semantic preservation validation during repair.

    Args:
        neighbors_a: First neighbor set.
        neighbors_b: Second neighbor set.

    Returns:
        Jaccard similarity in [0, 1].
    """
    set_a = set(neighbors_a.tolist())
    set_b = set(neighbors_b.tolist())
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def compute_neighbor_stability(
    index: VectorIndex,
    embeddings: NDArray[np.float64],
    token_id: int,
    k_values: list[int],
) -> dict[int, NDArray[np.int64]]:
    """Compute neighbors at multiple k values for stability analysis.

    Args:
        index: Built vector index.
        embeddings: Embedding matrix.
        token_id: Token to analyze.
        k_values: List of k values to query.

    Returns:
        Dictionary mapping k -> neighbor indices.
    """
    max_k = max(k_values)
    query_vector = embeddings[token_id]
    all_neighbors, _ = index.query_single(
        query_vector,
        k=max_k,
        exclude_self=True,
        self_index=token_id,
    )

    return {k: all_neighbors[:k] for k in k_values}
