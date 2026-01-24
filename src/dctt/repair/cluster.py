"""Cluster detection for joint embedding repair.

This module implements cluster detection for pathological tokens based on
mutual k-NN relationships. The key insight is that high-severity tokens
often cluster in regions where ALL neighbors have poor geometry, making
single-token repair ineffective.

By identifying connected components of pathological tokens and repairing
them jointly, the centered displacement covariance CAN change (because
multiple tokens are moving together).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from dctt.neighbors.index import VectorIndex


@dataclass
class TokenCluster:
    """A cluster of pathological tokens for joint repair."""

    cluster_id: int
    token_ids: list[int]
    mean_severity: float
    max_severity: float
    n_tokens: int

    # Cluster geometry statistics
    internal_connectivity: float  # Fraction of edges within cluster
    boundary_tokens: list[int] = field(default_factory=list)  # Tokens with healthy neighbors


@dataclass
class ClusteringResult:
    """Result of cluster detection."""

    clusters: list[TokenCluster]
    n_clusters: int
    n_isolated: int  # Tokens not in any cluster
    total_pathological: int
    adjacency_matrix: NDArray[np.bool_] | None = None


class PathologicalClusterDetector:
    """Detects clusters of pathological tokens using mutual k-NN graph.

    The algorithm:
    1. Build mutual k-NN graph: edge (i,j) exists if i is in j's k-NN AND j is in i's k-NN
    2. Restrict to high-severity tokens
    3. Find connected components using BFS/union-find
    4. Return clusters sorted by mean severity

    This approach groups tokens that are "stuck together" in embedding space,
    where moving one token alone cannot escape the pathological region.
    """

    def __init__(
        self,
        mutual_k: int = 20,
        min_cluster_size: int = 3,
        severity_threshold: float = 0.0,
    ) -> None:
        """Initialize detector.

        Args:
            mutual_k: k for mutual k-NN graph construction.
            min_cluster_size: Minimum tokens to form a cluster.
            severity_threshold: Only include tokens with severity > threshold.
        """
        self.mutual_k = mutual_k
        self.min_cluster_size = min_cluster_size
        self.severity_threshold = severity_threshold

    def detect(
        self,
        token_ids: list[int],
        severities: dict[int, float],
        embeddings: NDArray[np.float64],
        index: "VectorIndex",
    ) -> ClusteringResult:
        """Detect clusters of pathological tokens.

        Args:
            token_ids: Token IDs to consider (high-severity candidates).
            severities: Mapping of token_id to severity score.
            embeddings: Full embedding matrix.
            index: kNN index for neighbor queries.

        Returns:
            ClusteringResult with detected clusters.
        """
        # Filter to high-severity tokens
        high_severity_ids = [
            tid for tid in token_ids
            if severities.get(tid, 0) > self.severity_threshold
        ]

        if len(high_severity_ids) < self.min_cluster_size:
            return ClusteringResult(
                clusters=[],
                n_clusters=0,
                n_isolated=len(high_severity_ids),
                total_pathological=len(high_severity_ids),
            )

        # Build mutual k-NN graph
        token_set = set(high_severity_ids)
        id_to_idx = {tid: i for i, tid in enumerate(high_severity_ids)}
        n = len(high_severity_ids)

        # Get k-NN for each token
        neighbors_map: dict[int, set[int]] = {}
        for tid in high_severity_ids:
            query_vec = embeddings[tid].reshape(1, -1)
            neighbor_ids, _ = index.query(query_vec, k=self.mutual_k, exclude_self=True)
            # Only keep neighbors that are also high-severity
            neighbors_map[tid] = set(neighbor_ids[0].tolist()) & token_set

        # Build adjacency: mutual k-NN (edge exists if both are in each other's k-NN)
        adjacency = np.zeros((n, n), dtype=bool)
        for tid in high_severity_ids:
            i = id_to_idx[tid]
            for neighbor_id in neighbors_map[tid]:
                if neighbor_id in id_to_idx:
                    j = id_to_idx[neighbor_id]
                    # Check mutual: tid in neighbor's k-NN AND neighbor in tid's k-NN
                    if tid in neighbors_map.get(neighbor_id, set()):
                        adjacency[i, j] = True
                        adjacency[j, i] = True

        # Find connected components using union-find
        parent = list(range(n))

        def find(x: int) -> int:
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x: int, y: int) -> None:
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        for i in range(n):
            for j in range(i + 1, n):
                if adjacency[i, j]:
                    union(i, j)

        # Group by component
        components: dict[int, list[int]] = {}
        for i in range(n):
            root = find(i)
            if root not in components:
                components[root] = []
            components[root].append(high_severity_ids[i])

        # Build clusters
        clusters = []
        isolated_count = 0

        for cluster_id, (_, token_list) in enumerate(
            sorted(components.items(), key=lambda x: -len(x[1]))
        ):
            if len(token_list) < self.min_cluster_size:
                isolated_count += len(token_list)
                continue

            cluster_severities = [severities.get(tid, 0) for tid in token_list]
            mean_sev = float(np.mean(cluster_severities))
            max_sev = float(np.max(cluster_severities))

            # Compute internal connectivity
            cluster_set = set(token_list)
            internal_edges = 0
            total_edges = 0
            boundary = []

            for tid in token_list:
                tid_neighbors = neighbors_map.get(tid, set())
                internal = len(tid_neighbors & cluster_set)
                external = len(tid_neighbors - cluster_set)
                internal_edges += internal
                total_edges += internal + external

                if external > 0:
                    boundary.append(tid)

            connectivity = internal_edges / total_edges if total_edges > 0 else 0.0

            clusters.append(TokenCluster(
                cluster_id=cluster_id,
                token_ids=token_list,
                mean_severity=mean_sev,
                max_severity=max_sev,
                n_tokens=len(token_list),
                internal_connectivity=connectivity,
                boundary_tokens=boundary,
            ))

        # Sort by mean severity (worst first)
        clusters.sort(key=lambda c: -c.mean_severity)

        return ClusteringResult(
            clusters=clusters,
            n_clusters=len(clusters),
            n_isolated=isolated_count,
            total_pathological=len(high_severity_ids),
            adjacency_matrix=adjacency,
        )


def find_pathological_clusters(
    token_ids: list[int],
    severities: dict[int, float],
    embeddings: NDArray[np.float64],
    index: "VectorIndex",
    mutual_k: int = 20,
    min_cluster_size: int = 3,
) -> ClusteringResult:
    """Convenience function to find pathological clusters.

    Args:
        token_ids: Candidate token IDs.
        severities: Token severity scores.
        embeddings: Embedding matrix.
        index: kNN index.
        mutual_k: k for mutual k-NN.
        min_cluster_size: Minimum cluster size.

    Returns:
        ClusteringResult with detected clusters.
    """
    detector = PathologicalClusterDetector(
        mutual_k=mutual_k,
        min_cluster_size=min_cluster_size,
    )
    return detector.detect(token_ids, severities, embeddings, index)
