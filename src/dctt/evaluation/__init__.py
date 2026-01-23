"""Evaluation utilities for DCTT experiments."""

from dctt.evaluation.predictive import (
    compute_predictive_validity,
    PredictiveValidityAnalyzer,
)
from dctt.evaluation.causal import (
    compute_causal_effect,
    CausalRepairAnalyzer,
)
from dctt.evaluation.matched_controls import (
    create_matched_controls,
    MatchedControlSelector,
)
from dctt.evaluation.statistics import (
    bootstrap_ci,
    paired_permutation_test,
    compute_effect_size,
    StatisticalAnalyzer,
)

__all__ = [
    # Predictive
    "compute_predictive_validity",
    "PredictiveValidityAnalyzer",
    # Causal
    "compute_causal_effect",
    "CausalRepairAnalyzer",
    # Matched controls
    "create_matched_controls",
    "MatchedControlSelector",
    # Statistics
    "bootstrap_ci",
    "paired_permutation_test",
    "compute_effect_size",
    "StatisticalAnalyzer",
]
