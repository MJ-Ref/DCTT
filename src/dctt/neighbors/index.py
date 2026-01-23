"""Abstract base class for vector index implementations.

This module defines the interface that all vector index implementations
must follow, enabling interchangeable backends (USearch, FAISS, etc.).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    pass


class VectorIndex(ABC):
    """Abstract base class for kNN vector index implementations.

    All index implementations must support:
    - Building from embeddings
    - Querying k nearest neighbors
    - Saving/loading from disk
    - Configuration hashing for reproducibility
    """

    @abstractmethod
    def build(
        self,
        embeddings: NDArray[np.float64],
        metric: str = "cos",
        seed: int | None = None,
    ) -> None:
        """Build the index from embeddings.

        Args:
            embeddings: Embedding matrix of shape (n_vectors, dim).
            metric: Distance metric ("cos" for cosine, "l2" for Euclidean).
            seed: Random seed for reproducible index construction.
        """
        pass

    @abstractmethod
    def query(
        self,
        query_vectors: NDArray[np.float64],
        k: int,
        exclude_self: bool = True,
    ) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
        """Query k nearest neighbors for each query vector.

        Args:
            query_vectors: Query vectors of shape (n_queries, dim).
            k: Number of neighbors to return.
            exclude_self: If True, exclude the query vector itself from results
                         (assumes query vectors are in the index).

        Returns:
            Tuple of (indices, distances):
                - indices: Shape (n_queries, k), neighbor indices.
                - distances: Shape (n_queries, k), distances to neighbors.
        """
        pass

    @abstractmethod
    def query_single(
        self,
        query_vector: NDArray[np.float64],
        k: int,
        exclude_self: bool = True,
        self_index: int | None = None,
    ) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
        """Query k nearest neighbors for a single vector.

        Args:
            query_vector: Single query vector of shape (dim,).
            k: Number of neighbors to return.
            exclude_self: If True, exclude self_index from results.
            self_index: Index of the query vector in the index (for self-exclusion).

        Returns:
            Tuple of (indices, distances):
                - indices: Shape (k,), neighbor indices.
                - distances: Shape (k,), distances to neighbors.
        """
        pass

    @abstractmethod
    def save(self, path: str | Path) -> None:
        """Save index to disk.

        Args:
            path: File path to save the index.
        """
        pass

    @abstractmethod
    def load(self, path: str | Path) -> None:
        """Load index from disk.

        Args:
            path: File path to load the index from.
        """
        pass

    @property
    @abstractmethod
    def config_hash(self) -> str:
        """Hash of index configuration for reproducibility tracking.

        Returns:
            Hexadecimal hash string of the index configuration.
        """
        pass

    @property
    @abstractmethod
    def n_vectors(self) -> int:
        """Number of vectors in the index."""
        pass

    @property
    @abstractmethod
    def dim(self) -> int:
        """Dimensionality of vectors in the index."""
        pass

    @property
    @abstractmethod
    def is_built(self) -> bool:
        """Whether the index has been built."""
        pass
