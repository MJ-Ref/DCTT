"""Core abstractions and types for DCTT."""

from dctt.core.types import (
    DiagnosticResult,
    FrequencyTier,
    RepairConfig,
    RepairResult,
    Stage1Result,
    Stage2Result,
    Stage3Result,
    TokenInfo,
    TokenType,
)
from dctt.core.exceptions import (
    DCTTError,
    EmbeddingExtractionError,
    IndexBuildError,
    MetricComputationError,
    RepairError,
)
from dctt.core.registry import MetricRegistry

__all__ = [
    # Types
    "TokenType",
    "FrequencyTier",
    "TokenInfo",
    "Stage1Result",
    "Stage2Result",
    "Stage3Result",
    "DiagnosticResult",
    "RepairConfig",
    "RepairResult",
    # Exceptions
    "DCTTError",
    "EmbeddingExtractionError",
    "IndexBuildError",
    "MetricComputationError",
    "RepairError",
    # Registry
    "MetricRegistry",
]
