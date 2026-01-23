"""Custom exceptions for DCTT.

All DCTT-specific exceptions inherit from DCTTError for easy catching.
"""

from __future__ import annotations


class DCTTError(Exception):
    """Base exception for all DCTT errors."""

    pass


class EmbeddingExtractionError(DCTTError):
    """Error during embedding extraction from model."""

    def __init__(self, model_name: str, message: str) -> None:
        self.model_name = model_name
        super().__init__(f"Failed to extract embeddings from {model_name}: {message}")


class IndexBuildError(DCTTError):
    """Error during kNN index construction."""

    def __init__(self, index_type: str, message: str) -> None:
        self.index_type = index_type
        super().__init__(f"Failed to build {index_type} index: {message}")


class IndexQueryError(DCTTError):
    """Error during kNN index query."""

    def __init__(self, message: str) -> None:
        super().__init__(f"Index query failed: {message}")


class MetricComputationError(DCTTError):
    """Error during metric computation."""

    def __init__(self, metric_name: str, token_id: int, message: str) -> None:
        self.metric_name = metric_name
        self.token_id = token_id
        super().__init__(
            f"Failed to compute {metric_name} for token {token_id}: {message}"
        )


class RepairError(DCTTError):
    """Error during embedding repair."""

    def __init__(self, token_id: int, message: str) -> None:
        self.token_id = token_id
        super().__init__(f"Failed to repair embedding for token {token_id}: {message}")


class RepairConvergenceError(RepairError):
    """Repair optimization failed to converge."""

    def __init__(self, token_id: int, iterations: int, final_loss: float) -> None:
        self.iterations = iterations
        self.final_loss = final_loss
        super().__init__(
            token_id,
            f"Optimization did not converge after {iterations} iterations "
            f"(final loss: {final_loss:.6f})",
        )


class SemanticDriftError(RepairError):
    """Repair caused unacceptable semantic drift."""

    def __init__(
        self, token_id: int, metric_name: str, value: float, threshold: float
    ) -> None:
        self.metric_name = metric_name
        self.value = value
        self.threshold = threshold
        super().__init__(
            token_id,
            f"Semantic drift exceeded threshold: {metric_name}={value:.4f} "
            f"(threshold: {threshold:.4f})",
        )


class ConfigurationError(DCTTError):
    """Invalid configuration."""

    def __init__(self, message: str) -> None:
        super().__init__(f"Configuration error: {message}")


class StressTestError(DCTTError):
    """Error during stress test execution."""

    def __init__(self, test_name: str, message: str) -> None:
        self.test_name = test_name
        super().__init__(f"Stress test '{test_name}' failed: {message}")


class BenchmarkError(DCTTError):
    """Error during benchmark evaluation."""

    def __init__(self, benchmark_name: str, message: str) -> None:
        self.benchmark_name = benchmark_name
        super().__init__(f"Benchmark '{benchmark_name}' failed: {message}")


class DataError(DCTTError):
    """Error with data loading or validation."""

    pass


class CacheError(DCTTError):
    """Error with caching operations."""

    pass
