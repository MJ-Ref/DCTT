"""Diagnostic metrics for embedding geometry analysis."""

from dctt.metrics.stage1 import (
    compute_stage1_metrics,
    compute_mean_knn_distance,
    compute_spread_ratio,
    compute_lof,
)
from dctt.metrics.stage2 import (
    compute_stage2_metrics,
    compute_displacement_matrix,
    compute_local_covariance,
    compute_participation_ratio,
    compute_condition_number,
    compute_effective_dimension,
    compute_log_determinant,
    compute_anisotropy,
)
from dctt.metrics.severity import compute_severity_score, SeverityScorer
from dctt.metrics.consistency import compute_consistency, ConsistencyEstimator
from dctt.metrics.thresholding import (
    compute_adaptive_thresholds,
    AdaptiveThresholder,
)
from dctt.metrics.stage3 import (
    Stage3Result,
    MLEDimensionEstimator,
    TDAAnalyzer,
    compute_stage3_metrics,
    should_run_stage3,
)

__all__ = [
    # Stage 1
    "compute_stage1_metrics",
    "compute_mean_knn_distance",
    "compute_spread_ratio",
    "compute_lof",
    # Stage 2
    "compute_stage2_metrics",
    "compute_displacement_matrix",
    "compute_local_covariance",
    "compute_participation_ratio",
    "compute_condition_number",
    "compute_effective_dimension",
    "compute_log_determinant",
    "compute_anisotropy",
    # Severity
    "compute_severity_score",
    "SeverityScorer",
    # Consistency
    "compute_consistency",
    "ConsistencyEstimator",
    # Thresholding
    "compute_adaptive_thresholds",
    "AdaptiveThresholder",
    # Stage 3
    "Stage3Result",
    "MLEDimensionEstimator",
    "TDAAnalyzer",
    "compute_stage3_metrics",
    "should_run_stage3",
]
