"""Core type definitions for DCTT.

This module defines all the fundamental data structures used throughout the
DCTT framework, including token information, diagnostic results, and repair
configurations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Protocol, TypeVar, runtime_checkable

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from collections.abc import Sequence


# Type aliases
EmbeddingVector = NDArray[np.float64]
EmbeddingMatrix = NDArray[np.float64]
NeighborIndices = NDArray[np.int64]
NeighborDistances = NDArray[np.float64]


class TokenType(Enum):
    """Classification of token types for bucketing."""

    FULL_WORD = auto()  # Complete words (alphabetic)
    SUBWORD = auto()  # BPE subword pieces
    SPECIAL = auto()  # BOS, EOS, PAD, etc.
    PUNCTUATION = auto()  # Punctuation marks
    NUMERIC = auto()  # Numbers and numeric tokens
    CODE_SYMBOL = auto()  # Programming symbols: {, }, =, etc.
    WHITESPACE = auto()  # Space, newline, tab tokens
    UNKNOWN = auto()  # Unclassified tokens


class FrequencyTier(Enum):
    """Token frequency tiers for adaptive thresholding."""

    HIGH = auto()  # Top 20% by frequency
    MID = auto()  # Middle 60%
    LOW = auto()  # Bottom 20%


@dataclass(frozen=True, slots=True)
class TokenInfo:
    """Immutable token information.

    Attributes:
        token_id: Unique token identifier in vocabulary.
        token_str: String representation of the token.
        token_type: Classification of the token type.
        frequency: Token frequency (from corpus or estimate).
        frequency_tier: Frequency tier for bucketing.
        norm: Original L2 norm before normalization.
    """

    token_id: int
    token_str: str
    token_type: TokenType
    frequency: float
    frequency_tier: FrequencyTier
    norm: float

    def __repr__(self) -> str:
        return (
            f"TokenInfo(id={self.token_id}, str={self.token_str!r}, "
            f"type={self.token_type.name}, tier={self.frequency_tier.name})"
        )


@dataclass(slots=True)
class Stage1Result:
    """Stage 1 metric results: basic local outlier checks.

    Attributes:
        token_id: Token being analyzed.
        mu_k: Mean k-NN distance.
        med_k: Median k-NN distance.
        spread_q: Quantile spread ratio (q90/q10).
        lof: Local Outlier Factor score (optional).
        distances: Raw k-NN distances (optional, for analysis).
        fail: Whether token fails Stage 1 thresholds.
    """

    token_id: int
    mu_k: float
    med_k: float
    spread_q: float
    lof: float | None = None
    distances: NDArray[np.float64] | None = None
    fail: bool = False

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "token_id": self.token_id,
            "mu_k": self.mu_k,
            "med_k": self.med_k,
            "spread_q": self.spread_q,
            "lof": self.lof,
            "fail": self.fail,
        }


@dataclass(slots=True)
class Stage2Result:
    """Stage 2 metric results: core spectral geometry.

    Attributes:
        token_id: Token being analyzed.
        dim95: Effective dimension at 95% explained variance.
        pr: Participation ratio.
        cond: Local condition number.
        logdet: Log-determinant of local covariance.
        anisotropy: Ratio of largest eigenvalue to mean.
        eigenvalues: Full eigenvalue spectrum (optional).
        fail: Whether token fails Stage 2 thresholds.
    """

    token_id: int
    dim95: int
    pr: float
    cond: float
    logdet: float
    anisotropy: float
    eigenvalues: NDArray[np.float64] | None = None
    fail: bool = False

    def to_dict(self, include_eigenvalues: bool = False) -> dict:
        """Convert to dictionary for serialization."""
        result = {
            "token_id": self.token_id,
            "dim95": self.dim95,
            "pr": self.pr,
            "cond": self.cond,
            "logdet": self.logdet,
            "anisotropy": self.anisotropy,
            "fail": self.fail,
        }
        if include_eigenvalues and self.eigenvalues is not None:
            result["eigenvalues"] = self.eigenvalues.tolist()
        return result


@dataclass(slots=True)
class Stage3Result:
    """Stage 3 metric results: advanced/optional metrics.

    Attributes:
        token_id: Token being analyzed.
        mle_intrinsic_dim: MLE intrinsic dimension estimate.
        tda_h1_lifetime: Maximum H1 persistence lifetime (TDA).
        tda_h1_count: Count of significant H1 features.
        fail: Whether token fails Stage 3 thresholds.
    """

    token_id: int
    mle_intrinsic_dim: float | None = None
    tda_h1_lifetime: float | None = None
    tda_h1_count: int | None = None
    fail: bool = False

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "token_id": self.token_id,
            "mle_intrinsic_dim": self.mle_intrinsic_dim,
            "tda_h1_lifetime": self.tda_h1_lifetime,
            "tda_h1_count": self.tda_h1_count,
            "fail": self.fail,
        }


@dataclass(slots=True)
class DiagnosticResult:
    """Complete diagnostic result for a token.

    Aggregates results from all stages along with severity and consistency
    scores used for repair candidate selection.

    Attributes:
        token_info: Token metadata.
        stage1: Stage 1 results.
        stage2: Stage 2 results.
        stage3: Stage 3 results (optional).
        severity: Composite severity score.
        consistency: Consistency score (cons@k).
        priority: Priority score for repair (severity * consistency * importance).
        bucket: Tuple of (frequency_tier, token_type) for thresholding.
    """

    token_info: TokenInfo
    stage1: Stage1Result
    stage2: Stage2Result
    stage3: Stage3Result | None = None
    severity: float = 0.0
    consistency: float = 0.0
    priority: float = 0.0
    bucket: tuple[FrequencyTier, TokenType] | None = None

    @property
    def token_id(self) -> int:
        """Convenience accessor for token ID."""
        return self.token_info.token_id

    @property
    def fails_any_stage(self) -> bool:
        """Whether token fails any diagnostic stage."""
        if self.stage1.fail or self.stage2.fail:
            return True
        if self.stage3 is not None and self.stage3.fail:
            return True
        return False

    def to_dict(self, include_eigenvalues: bool = False) -> dict:
        """Convert to dictionary for serialization."""
        result = {
            "token_info": {
                "token_id": self.token_info.token_id,
                "token_str": self.token_info.token_str,
                "token_type": self.token_info.token_type.name,
                "frequency": self.token_info.frequency,
                "frequency_tier": self.token_info.frequency_tier.name,
                "norm": self.token_info.norm,
            },
            "stage1": self.stage1.to_dict(),
            "stage2": self.stage2.to_dict(include_eigenvalues),
            "severity": self.severity,
            "consistency": self.consistency,
            "priority": self.priority,
        }
        if self.stage3 is not None:
            result["stage3"] = self.stage3.to_dict()
        if self.bucket is not None:
            result["bucket"] = (self.bucket[0].name, self.bucket[1].name)
        return result


@dataclass(slots=True)
class RepairConfig:
    """Configuration for embedding repair optimization.

    Attributes:
        max_outer_iters: Number of neighbor recomputation iterations.
        max_inner_steps: Gradient steps per outer iteration.
        learning_rate: Step size for gradient descent.
        lambda_anchor: Weight for anchor loss (stay close to original).
        lambda_nn_preserve: Weight for nearest neighbor preservation.
        lambda_logit_preserve: Weight for logit preservation (expensive).
        delta_max: Maximum allowed embedding change norm.
        geometry_loss: Which geometry loss to use.
        eps: Numerical stability constant.
    """

    max_outer_iters: int = 5
    max_inner_steps: int = 100
    learning_rate: float = 0.01
    lambda_anchor: float = 1.0
    lambda_nn_preserve: float = 0.5
    lambda_logit_preserve: float = 0.0
    delta_max: float = 0.3
    geometry_loss: str = "cond"  # "cond", "logdet", "pr", "combined"
    eps: float = 1e-10


@dataclass(slots=True)
class RepairResult:
    """Result of repairing a token embedding.

    Attributes:
        token_id: Token that was repaired.
        original_embedding: Original normalized embedding vector.
        repaired_embedding: New embedding after repair.
        delta_norm: L2 norm of the change.
        geometry_before: Geometry metrics before repair.
        geometry_after: Geometry metrics after repair.
        semantic_validation: Semantic preservation metrics.
        converged: Whether optimization converged.
        iterations: Total iterations used.
        final_loss: Final optimization loss value.
    """

    token_id: int
    original_embedding: EmbeddingVector
    repaired_embedding: EmbeddingVector
    delta_norm: float
    geometry_before: dict[str, float] = field(default_factory=dict)
    geometry_after: dict[str, float] = field(default_factory=dict)
    semantic_validation: dict[str, float] = field(default_factory=dict)
    converged: bool = False
    iterations: int = 0
    final_loss: float = float("inf")

    @property
    def geometry_improvement(self) -> dict[str, float]:
        """Compute improvement in each geometry metric."""
        improvements = {}
        for key in self.geometry_before:
            if key in self.geometry_after:
                before = self.geometry_before[key]
                after = self.geometry_after[key]
                # For cond and anisotropy, lower is better
                # For pr and logdet, higher is better
                if key in ("cond", "anisotropy"):
                    improvements[key] = before - after
                else:
                    improvements[key] = after - before
        return improvements

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "token_id": self.token_id,
            "delta_norm": self.delta_norm,
            "geometry_before": self.geometry_before,
            "geometry_after": self.geometry_after,
            "geometry_improvement": self.geometry_improvement,
            "semantic_validation": self.semantic_validation,
            "converged": self.converged,
            "iterations": self.iterations,
            "final_loss": self.final_loss,
        }


# Protocol definitions for extensibility


@runtime_checkable
class Metric(Protocol):
    """Protocol for diagnostic metrics."""

    @property
    def name(self) -> str:
        """Unique metric identifier."""
        ...

    @property
    def higher_is_worse(self) -> bool:
        """Whether higher values indicate pathology."""
        ...

    def compute(
        self,
        embeddings: EmbeddingMatrix,
        token_id: int,
        neighbors: NeighborIndices,
    ) -> float:
        """Compute metric for a single token."""
        ...


@runtime_checkable
class VectorIndexProtocol(Protocol):
    """Protocol for kNN index implementations."""

    def build(
        self,
        embeddings: EmbeddingMatrix,
        metric: str = "cos",
        seed: int | None = None,
    ) -> None:
        """Build the index from embeddings."""
        ...

    def query(
        self,
        query_vectors: EmbeddingMatrix,
        k: int,
        exclude_self: bool = True,
    ) -> tuple[NeighborIndices, NeighborDistances]:
        """Query k nearest neighbors."""
        ...

    def save(self, path: str) -> None:
        """Save index to disk."""
        ...

    def load(self, path: str) -> None:
        """Load index from disk."""
        ...

    @property
    def config_hash(self) -> str:
        """Hash of index configuration for reproducibility."""
        ...


@runtime_checkable
class StressTestProtocol(Protocol):
    """Protocol for stress test implementations."""

    @property
    def name(self) -> str:
        """Test suite name."""
        ...

    @property
    def target_tokens(self) -> list[int]:
        """Token IDs this test targets."""
        ...

    def run(
        self,
        model: object,
        tokenizer: object,
        token_ids: Sequence[int],
    ) -> dict[int, float]:
        """Run test and return failure rates per token."""
        ...


# Bucket type for thresholding
Bucket = tuple[FrequencyTier, TokenType]


@dataclass(slots=True)
class BucketStats:
    """Statistics for a frequency-type bucket.

    Used for adaptive thresholding of metrics.
    """

    bucket: Bucket
    count: int
    metrics: dict[str, dict[str, float]] = field(default_factory=dict)

    def add_metric_stats(
        self,
        metric_name: str,
        median: float,
        mad: float,
        q01: float,
        q99: float,
    ) -> None:
        """Add statistics for a metric."""
        self.metrics[metric_name] = {
            "median": median,
            "mad": mad,
            "q01": q01,
            "q99": q99,
        }
