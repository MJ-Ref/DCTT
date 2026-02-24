"""Unit tests for USearch index behavior."""

from __future__ import annotations

import numpy as np

from dctt.neighbors.usearch_index import USearchIndex


def test_query_single_excludes_self_for_exact_vector(
    sample_embeddings_small: np.ndarray,
) -> None:
    """Self index is excluded for exact in-index queries."""
    index = USearchIndex()
    index.build(sample_embeddings_small, metric="ip", seed=42)

    token_id = 7
    neighbors, _ = index.query_single(
        sample_embeddings_small[token_id],
        k=10,
        exclude_self=True,
        self_index=token_id,
    )

    assert token_id not in neighbors


def test_query_single_excludes_self_for_perturbed_vector(
    sample_embeddings_small: np.ndarray,
) -> None:
    """Explicit self_index exclusion holds for perturbed query vectors."""
    index = USearchIndex()
    index.build(sample_embeddings_small, metric="ip", seed=42)

    token_id = 11
    perturbed = sample_embeddings_small[token_id].copy()
    perturbed = perturbed + 0.01
    perturbed = perturbed / np.linalg.norm(perturbed)

    neighbors, _ = index.query_single(
        perturbed,
        k=10,
        exclude_self=True,
        self_index=token_id,
    )

    assert token_id not in neighbors
