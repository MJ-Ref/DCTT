"""Predictive validity analysis for DCTT diagnostics.

This module implements analysis to verify that geometry metrics
predict stress test failures beyond confounds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from dctt.core.types import DiagnosticResult
    from dctt.stress_tests.base import StressTestResult


@dataclass
class PredictiveValidityResult:
    """Results of predictive validity analysis."""

    auc: float
    pr_auc: float
    feature_importance: dict[str, float]
    baseline_auc: float  # Frequency-only baseline
    improvement_over_baseline: float


class PredictiveValidityAnalyzer:
    """Analyzes whether geometry metrics predict failures."""

    def __init__(self) -> None:
        """Initialize analyzer."""
        pass

    def analyze(
        self,
        diagnostic_results: list["DiagnosticResult"],
        stress_test_results: dict[int, float],  # token_id -> failure_rate
    ) -> PredictiveValidityResult:
        """Analyze predictive validity of geometry metrics.

        Args:
            diagnostic_results: Diagnostic results with metrics.
            stress_test_results: Token failure rates from stress tests.

        Returns:
            PredictiveValidityResult with AUC and feature importance.
        """
        # Build feature matrix
        X = []
        y = []
        feature_names = ["cond", "pr", "logdet", "spread_q", "frequency"]

        for result in diagnostic_results:
            if result.token_id not in stress_test_results:
                continue

            features = [
                result.stage2.cond,
                result.stage2.pr,
                result.stage2.logdet,
                result.stage1.spread_q,
                np.log(result.token_info.frequency + 1),
            ]
            X.append(features)
            y.append(stress_test_results[result.token_id])

        X = np.array(X)
        y = np.array(y)

        if len(X) == 0:
            return PredictiveValidityResult(
                auc=0.5,
                pr_auc=0.5,
                feature_importance={},
                baseline_auc=0.5,
                improvement_over_baseline=0.0,
            )

        # Binarize failure rate
        y_binary = (y > 0.5).astype(int)

        # Compute AUC using severity score as predictor
        try:
            from sklearn.metrics import roc_auc_score, average_precision_score
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import StandardScaler

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Full model
            model = LogisticRegression(max_iter=1000)
            model.fit(X_scaled, y_binary)
            y_pred = model.predict_proba(X_scaled)[:, 1]
            auc = roc_auc_score(y_binary, y_pred)
            pr_auc = average_precision_score(y_binary, y_pred)

            # Baseline (frequency only)
            X_freq = X_scaled[:, -1].reshape(-1, 1)
            baseline_model = LogisticRegression(max_iter=1000)
            baseline_model.fit(X_freq, y_binary)
            y_baseline = baseline_model.predict_proba(X_freq)[:, 1]
            baseline_auc = roc_auc_score(y_binary, y_baseline)

            # Feature importance
            importance = dict(zip(feature_names, np.abs(model.coef_[0])))

        except Exception:
            # Fallback if sklearn not available
            auc = 0.5
            pr_auc = 0.5
            baseline_auc = 0.5
            importance = {}

        return PredictiveValidityResult(
            auc=auc,
            pr_auc=pr_auc,
            feature_importance=importance,
            baseline_auc=baseline_auc,
            improvement_over_baseline=auc - baseline_auc,
        )


def compute_predictive_validity(
    diagnostic_results: list["DiagnosticResult"],
    stress_test_results: dict[int, float],
) -> PredictiveValidityResult:
    """Convenience function for predictive validity analysis."""
    analyzer = PredictiveValidityAnalyzer()
    return analyzer.analyze(diagnostic_results, stress_test_results)
