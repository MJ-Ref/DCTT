"""USearch-based vector index implementation.

USearch provides HNSW (Hierarchical Navigable Small World) indexing with
excellent performance on Apple Silicon via ARM NEON SIMD optimizations.
This is the recommended backend for M3 Max systems.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from dctt.neighbors.index import VectorIndex
from dctt.core.exceptions import IndexBuildError, IndexQueryError

if TYPE_CHECKING:
    pass


class USearchIndex(VectorIndex):
    """HNSW index using USearch library.

    USearch provides:
    - Fast approximate nearest neighbor search
    - ARM NEON optimizations for Apple Silicon
    - Deterministic index construction with seeds
    - Multiple distance metrics

    Example:
        >>> index = USearchIndex(connectivity=32, expansion_add=128, expansion_search=64)
        >>> index.build(embeddings, metric="cos", seed=42)
        >>> indices, distances = index.query(query_vectors, k=50)
    """

    def __init__(
        self,
        connectivity: int = 32,
        expansion_add: int = 128,
        expansion_search: int = 64,
        threads: int = 0,
        dtype: str = "f32",
    ) -> None:
        """Initialize USearch index with HNSW parameters.

        Args:
            connectivity: Number of bi-directional links per node (M parameter).
                         Higher values improve recall but increase memory/time.
            expansion_add: Size of dynamic candidate list during construction
                          (efConstruction). Higher values improve index quality.
            expansion_search: Size of dynamic candidate list during search (ef).
                             Higher values improve recall at query time.
            threads: Number of threads for parallel operations (0 = auto).
            dtype: Data type for stored vectors ("f32", "f16", "i8").
        """
        self.connectivity = connectivity
        self.expansion_add = expansion_add
        self.expansion_search = expansion_search
        self.threads = threads
        self.dtype = dtype

        self._index = None
        self._metric = None
        self._seed = None
        self._n_vectors = 0
        self._dim = 0
        self._embeddings: NDArray[np.float64] | None = None

    def build(
        self,
        embeddings: NDArray[np.float64],
        metric: str = "cos",
        seed: int | None = None,
    ) -> None:
        """Build the HNSW index from embeddings.

        Args:
            embeddings: Embedding matrix of shape (n_vectors, dim).
                       Should be L2-normalized for cosine distance.
            metric: Distance metric ("cos" for cosine, "l2" for Euclidean).
            seed: Random seed for reproducible construction.

        Raises:
            IndexBuildError: If index construction fails.
        """
        try:
            from usearch.index import Index, MetricKind
        except ImportError as e:
            raise IndexBuildError(
                "usearch", "USearch not installed. Install with: pip install usearch"
            ) from e

        self._n_vectors, self._dim = embeddings.shape
        self._metric = metric
        self._seed = seed

        # Store embeddings for self-exclusion during queries
        self._embeddings = embeddings.astype(np.float32)

        # Map metric string to USearch MetricKind
        metric_map = {
            "cos": MetricKind.Cos,
            "cosine": MetricKind.Cos,
            "ip": MetricKind.IP,
            "inner_product": MetricKind.IP,
            "l2": MetricKind.L2sq,
            "l2sq": MetricKind.L2sq,
            "euclidean": MetricKind.L2sq,
        }

        if metric.lower() not in metric_map:
            raise IndexBuildError(
                "usearch",
                f"Unknown metric '{metric}'. Supported: {list(metric_map.keys())}",
            )

        usearch_metric = metric_map[metric.lower()]

        # Map dtype string to numpy dtype
        dtype_map = {
            "f32": np.float32,
            "f16": np.float16,
            "i8": np.int8,
        }

        if self.dtype not in dtype_map:
            raise IndexBuildError(
                "usearch", f"Unknown dtype '{self.dtype}'. Supported: {list(dtype_map.keys())}"
            )

        try:
            # Create index
            self._index = Index(
                ndim=self._dim,
                metric=usearch_metric,
                dtype=dtype_map[self.dtype],
                connectivity=self.connectivity,
                expansion_add=self.expansion_add,
                expansion_search=self.expansion_search,
            )

            # Add vectors with sequential IDs
            keys = np.arange(self._n_vectors, dtype=np.uint64)
            vectors = self._embeddings

            # Build index
            self._index.add(keys, vectors, threads=self.threads)

        except Exception as e:
            raise IndexBuildError("usearch", str(e)) from e

    def query(
        self,
        query_vectors: NDArray[np.float64],
        k: int,
        exclude_self: bool = True,
    ) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
        """Query k nearest neighbors for multiple vectors.

        Args:
            query_vectors: Query vectors of shape (n_queries, dim).
            k: Number of neighbors to return.
            exclude_self: If True and query vectors are in the index,
                         exclude exact matches (distance ~0).

        Returns:
            Tuple of (indices, distances), each of shape (n_queries, k).

        Raises:
            IndexQueryError: If query fails.
        """
        if self._index is None:
            raise IndexQueryError("Index not built. Call build() first.")

        try:
            query_vectors = query_vectors.astype(np.float32)

            # Query k+1 if excluding self to have k results after filtering
            search_k = k + 1 if exclude_self else k

            results = self._index.search(query_vectors, search_k, threads=self.threads)

            # Extract indices and distances
            indices = results.keys.astype(np.int64)
            distances = results.distances.astype(np.float64)

            # Ensure 2D arrays (usearch returns 1D for single query)
            if indices.ndim == 1:
                indices = indices.reshape(1, -1)
                distances = distances.reshape(1, -1)

            if exclude_self:
                # Filter out self-matches (distance very close to 0)
                n_queries = query_vectors.shape[0]
                filtered_indices = np.zeros((n_queries, k), dtype=np.int64)
                filtered_distances = np.zeros((n_queries, k), dtype=np.float64)

                for i in range(n_queries):
                    # Keep first k neighbors that aren't exact self-matches
                    mask = distances[i] > 1e-8
                    valid_indices = indices[i][mask]
                    valid_distances = distances[i][mask]

                    # Take first k
                    n_valid = min(len(valid_indices), k)
                    filtered_indices[i, :n_valid] = valid_indices[:n_valid]
                    filtered_distances[i, :n_valid] = valid_distances[:n_valid]

                    # If not enough, fill with last valid
                    if n_valid < k and n_valid > 0:
                        filtered_indices[i, n_valid:] = valid_indices[n_valid - 1]
                        filtered_distances[i, n_valid:] = valid_distances[n_valid - 1]

                return filtered_indices, filtered_distances

            return indices[:, :k], distances[:, :k]

        except Exception as e:
            raise IndexQueryError(str(e)) from e

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
            self_index: Index of the query vector (for self-exclusion).

        Returns:
            Tuple of (indices, distances), each of shape (k,).
        """
        query_2d = query_vector.reshape(1, -1)
        indices, distances = self.query(query_2d, k, exclude_self=exclude_self)
        return indices[0], distances[0]

    def save(self, path: str | Path) -> None:
        """Save index to disk.

        Args:
            path: File path to save the index (.usearch extension recommended).
        """
        if self._index is None:
            raise IndexQueryError("Index not built. Nothing to save.")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save index
        self._index.save(str(path))

        # Save metadata
        metadata = {
            "connectivity": self.connectivity,
            "expansion_add": self.expansion_add,
            "expansion_search": self.expansion_search,
            "threads": self.threads,
            "dtype": self.dtype,
            "metric": self._metric,
            "seed": self._seed,
            "n_vectors": self._n_vectors,
            "dim": self._dim,
        }
        metadata_path = path.with_suffix(".meta.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

    def load(self, path: str | Path) -> None:
        """Load index from disk.

        Args:
            path: File path to load the index from.
        """
        try:
            from usearch.index import Index
        except ImportError as e:
            raise IndexBuildError(
                "usearch", "USearch not installed. Install with: pip install usearch"
            ) from e

        path = Path(path)

        # Load metadata
        metadata_path = path.with_suffix(".meta.json")
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)
            self.connectivity = metadata.get("connectivity", self.connectivity)
            self.expansion_add = metadata.get("expansion_add", self.expansion_add)
            self.expansion_search = metadata.get("expansion_search", self.expansion_search)
            self.threads = metadata.get("threads", self.threads)
            self.dtype = metadata.get("dtype", self.dtype)
            self._metric = metadata.get("metric")
            self._seed = metadata.get("seed")
            self._n_vectors = metadata.get("n_vectors", 0)
            self._dim = metadata.get("dim", 0)

        # Load index
        self._index = Index.restore(str(path))

    @property
    def config_hash(self) -> str:
        """Hash of index configuration for reproducibility."""
        config = {
            "type": "usearch",
            "connectivity": self.connectivity,
            "expansion_add": self.expansion_add,
            "expansion_search": self.expansion_search,
            "dtype": self.dtype,
            "metric": self._metric,
            "seed": self._seed,
        }
        config_str = json.dumps(config, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]

    @property
    def n_vectors(self) -> int:
        """Number of vectors in the index."""
        return self._n_vectors

    @property
    def dim(self) -> int:
        """Dimensionality of vectors in the index."""
        return self._dim

    @property
    def is_built(self) -> bool:
        """Whether the index has been built."""
        return self._index is not None


def build_usearch_index(
    embeddings: NDArray[np.float64],
    metric: str = "cos",
    seed: int | None = None,
    connectivity: int = 32,
    expansion_add: int = 128,
    expansion_search: int = 64,
    threads: int = 0,
) -> USearchIndex:
    """Convenience function to build a USearch index.

    Args:
        embeddings: Embedding matrix of shape (n_vectors, dim).
        metric: Distance metric ("cos" or "l2").
        seed: Random seed for reproducibility.
        connectivity: HNSW M parameter.
        expansion_add: HNSW efConstruction parameter.
        expansion_search: HNSW ef parameter.
        threads: Number of threads (0 = auto).

    Returns:
        Built USearchIndex ready for queries.
    """
    index = USearchIndex(
        connectivity=connectivity,
        expansion_add=expansion_add,
        expansion_search=expansion_search,
        threads=threads,
    )
    index.build(embeddings, metric=metric, seed=seed)
    return index
