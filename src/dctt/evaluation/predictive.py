"""Predictive validity analysis for DCTT diagnostics.

This module implements comprehensive analysis to verify that geometry metrics
predict stress test failures beyond frequency and token type confounds.

Key analyses:
1. Baseline vs geometry model comparison (ROC-AUC, PR-AUC)
2. Feature ablation (leave-one-out importance)
3. Within-bucket analysis (does geometry predict within same confound strata?)
4. Bootstrap confidence intervals
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from dctt.core.types import DiagnosticResult

logger = logging.getLogger(__name__)


@dataclass
class ModelComparison:
    """Comparison of different feature sets."""

    name: str
    features: list[str]
    auc: float
    auc_ci_low: float
    auc_ci_high: float
    pr_auc: float
    pr_auc_ci_low: float
    pr_auc_ci_high: float
    n_samples: int


@dataclass
class FeatureAblation:
    """Feature ablation result."""

    feature_removed: str
    auc_without: float
    auc_drop: float  # Full model AUC - AUC without feature
    importance_rank: int


@dataclass
class BucketAnalysis:
    """Within-bucket predictive validity."""

    bucket: tuple
    n_samples: int
    auc: float
    pr_auc: float
    geometry_predicts: bool  # True if AUC > 0.55


@dataclass
class PredictiveValidityResult:
    """Comprehensive predictive validity results."""

    # Model comparisons
    baseline_model: ModelComparison  # frequency + type + norm
    geometry_model: ModelComparison  # geometry features only
    full_model: ModelComparison  # all features

    # Key metrics
    improvement_over_baseline: float
    geometry_adds_value: bool  # True if full > baseline significantly

    # Feature analysis
    feature_importance: dict[str, float]
    feature_ablations: list[FeatureAblation]

    # Within-bucket analysis
    bucket_analyses: list[BucketAnalysis]
    buckets_where_geometry_predicts: int
    total_buckets: int

    # Sample info
    n_tokens: int
    n_positive: int  # Tokens with failures
    positive_rate: float


class PredictiveValidityAnalyzer:
    """Comprehensive predictive validity analysis.

    Demonstrates that geometry metrics predict failures beyond
    frequency/type confounds through:
    1. Model comparison (baseline vs geometry vs full)
    2. Feature ablation analysis
    3. Within-bucket stratified analysis
    4. Bootstrap confidence intervals
    """

    def __init__(
        self,
        n_bootstrap: int = 100,
        random_state: int = 42,
        failure_threshold: float = 0.3,
        strict_mode: bool = True,
    ) -> None:
        """Initialize analyzer.

        Args:
            n_bootstrap: Number of bootstrap samples for CI.
            random_state: Random seed for reproducibility.
            failure_threshold: Threshold for binarizing failure rate.
            strict_mode: Raise on evaluator failures instead of silent fallback.
        """
        self.n_bootstrap = n_bootstrap
        self.random_state = random_state
        self.failure_threshold = failure_threshold
        self.strict_mode = strict_mode

    def analyze(
        self,
        diagnostic_results: list["DiagnosticResult"],
        stress_test_results: dict[int, float],
    ) -> PredictiveValidityResult:
        """Run comprehensive predictive validity analysis.

        Args:
            diagnostic_results: Diagnostic results with metrics.
            stress_test_results: Token failure rates from stress tests.

        Returns:
            Comprehensive PredictiveValidityResult.
        """
        # Build feature matrix
        data = self._build_feature_matrix(diagnostic_results, stress_test_results)

        if data["n_samples"] == 0:
            return self._empty_result()

        X = data["X"]
        y = data["y"]
        feature_names = data["feature_names"]
        buckets = data["buckets"]

        # Define feature sets
        confound_features = [name for name in feature_names if name.startswith("conf_")]
        geometry_features = ["cond", "pr", "logdet", "spread_q", "anisotropy", "severity"]
        all_features = confound_features + geometry_features

        # Get feature indices
        confound_idx = [feature_names.index(f) for f in confound_features if f in feature_names]
        geometry_idx = [feature_names.index(f) for f in geometry_features if f in feature_names]
        all_idx = list(range(len(feature_names)))

        # Model comparisons with bootstrap CI
        baseline_model = self._evaluate_model(
            X[:, confound_idx], y, "baseline (confounds)",
            [feature_names[i] for i in confound_idx]
        )

        geometry_model = self._evaluate_model(
            X[:, geometry_idx], y, "geometry_only",
            [feature_names[i] for i in geometry_idx]
        )

        full_model = self._evaluate_model(
            X[:, all_idx], y, "full",
            feature_names
        )

        # Feature importance from full model
        feature_importance = self._compute_feature_importance(X, y, feature_names)

        # Feature ablation
        feature_ablations = self._run_feature_ablation(X, y, feature_names, full_model.auc)

        # Within-bucket analysis
        bucket_analyses = self._within_bucket_analysis(X, y, buckets, geometry_idx, feature_names)

        # Compute summary statistics
        improvement = full_model.auc - baseline_model.auc
        geometry_adds_value = (
            full_model.auc_ci_low > baseline_model.auc_ci_high
        )  # Non-overlapping CIs

        buckets_predicting = sum(1 for b in bucket_analyses if b.geometry_predicts)

        n_positive = int(y.sum())
        positive_rate = n_positive / len(y) if len(y) > 0 else 0.0

        return PredictiveValidityResult(
            baseline_model=baseline_model,
            geometry_model=geometry_model,
            full_model=full_model,
            improvement_over_baseline=improvement,
            geometry_adds_value=geometry_adds_value,
            feature_importance=feature_importance,
            feature_ablations=feature_ablations,
            bucket_analyses=bucket_analyses,
            buckets_where_geometry_predicts=buckets_predicting,
            total_buckets=len(bucket_analyses),
            n_tokens=len(y),
            n_positive=n_positive,
            positive_rate=positive_rate,
        )

    def _build_feature_matrix(
        self,
        diagnostic_results: list["DiagnosticResult"],
        stress_test_results: dict[int, float],
    ) -> dict:
        """Build feature matrix from diagnostic results."""
        token_type_values = sorted({
            str(result.token_info.token_type.name).lower()
            for result in diagnostic_results
            if result.token_info.token_type is not None
        })
        tier_values = sorted({
            str(result.token_info.frequency_tier.name).lower()
            for result in diagnostic_results
            if result.token_info.frequency_tier is not None
        })

        confound_feature_names = [
            "conf_log_frequency",
            "conf_norm",
            "conf_token_len",
            "conf_is_ascii",
            "conf_has_digit",
        ]
        confound_feature_names.extend([f"conf_type_{value}" for value in token_type_values])
        confound_feature_names.extend([f"conf_tier_{value}" for value in tier_values])

        geometry_feature_names = [
            "cond",
            "pr",
            "logdet",
            "spread_q",
            "anisotropy",
            "severity",
        ]
        feature_names = confound_feature_names + geometry_feature_names

        X = []
        y = []
        buckets = []

        for result in diagnostic_results:
            if result.token_id not in stress_test_results:
                continue

            failure_rate = stress_test_results[result.token_id]

            token_str = str(result.token_info.token_str or "")
            token_type_name = str(result.token_info.token_type.name).lower()
            tier_name = str(result.token_info.frequency_tier.name).lower()

            freq = max(float(result.token_info.frequency), 0.0)
            conf_features = [
                np.log1p(freq),
                float(result.token_info.norm),
                float(len(token_str)),
                float(token_str.isascii()),
                float(any(char.isdigit() for char in token_str)),
            ]
            conf_features.extend([
                1.0 if token_type_name == value else 0.0
                for value in token_type_values
            ])
            conf_features.extend([
                1.0 if tier_name == value else 0.0
                for value in tier_values
            ])

            geometry_features = [
                float(result.stage2.cond),
                float(result.stage2.pr),
                float(result.stage2.logdet),
                float(result.stage1.spread_q),
                float(result.stage2.anisotropy),
                float(result.severity),
            ]
            features = conf_features + geometry_features

            X.append(features)
            y.append(1 if failure_rate > self.failure_threshold else 0)
            buckets.append(result.bucket)

        if len(X) == 0:
            return {"n_samples": 0}

        return {
            "X": np.array(X),
            "y": np.array(y),
            "feature_names": feature_names,
            "buckets": buckets,
            "n_samples": len(X),
        }

    def _cross_val_predict_proba(
        self,
        X_scaled: NDArray,
        y: NDArray,
        model,
    ) -> NDArray:
        """Return cross-validated probabilities when class support allows it."""
        from sklearn.model_selection import StratifiedKFold, cross_val_predict

        class_counts = np.bincount(y.astype(int))
        if class_counts.size < 2:
            model.fit(X_scaled, y)
            return model.predict_proba(X_scaled)[:, 1]

        min_class_count = int(class_counts.min())
        n_splits = min(5, min_class_count)
        if n_splits < 2:
            model.fit(X_scaled, y)
            return model.predict_proba(X_scaled)[:, 1]

        cv = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=self.random_state,
        )
        return cross_val_predict(
            model,
            X_scaled,
            y,
            cv=cv,
            method="predict_proba",
        )[:, 1]

    def _evaluate_model(
        self,
        X: NDArray,
        y: NDArray,
        name: str,
        features: list[str],
    ) -> ModelComparison:
        """Evaluate a model with bootstrap confidence intervals."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import roc_auc_score, average_precision_score

        # Handle edge cases
        if len(np.unique(y)) < 2:
            return ModelComparison(
                name=name, features=features,
                auc=0.5, auc_ci_low=0.5, auc_ci_high=0.5,
                pr_auc=y.mean(), pr_auc_ci_low=y.mean(), pr_auc_ci_high=y.mean(),
                n_samples=len(y)
            )

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Use cross-validation predictions to avoid overfitting
        model = LogisticRegression(max_iter=1000, random_state=self.random_state)

        try:
            y_pred = self._cross_val_predict_proba(X_scaled, y, model)
        except Exception as exc:
            if self.strict_mode:
                raise RuntimeError(f"Cross-validated prediction failed for model '{name}'") from exc
            logger.warning(
                "Falling back to direct fit for model '%s' due to CV failure: %s",
                name,
                exc,
            )
            model.fit(X_scaled, y)
            y_pred = model.predict_proba(X_scaled)[:, 1]

        auc = roc_auc_score(y, y_pred)
        pr_auc = average_precision_score(y, y_pred)

        # Bootstrap for confidence intervals
        rng = np.random.default_rng(self.random_state)
        auc_samples = []
        pr_auc_samples = []

        for _ in range(self.n_bootstrap):
            idx = rng.choice(len(y), size=len(y), replace=True)
            if len(np.unique(y[idx])) < 2:
                continue
            try:
                auc_samples.append(roc_auc_score(y[idx], y_pred[idx]))
                pr_auc_samples.append(average_precision_score(y[idx], y_pred[idx]))
            except ValueError:
                continue

        if auc_samples:
            auc_ci_low, auc_ci_high = np.percentile(auc_samples, [2.5, 97.5])
            pr_auc_ci_low, pr_auc_ci_high = np.percentile(pr_auc_samples, [2.5, 97.5])
        else:
            auc_ci_low, auc_ci_high = auc, auc
            pr_auc_ci_low, pr_auc_ci_high = pr_auc, pr_auc

        return ModelComparison(
            name=name,
            features=features,
            auc=auc,
            auc_ci_low=auc_ci_low,
            auc_ci_high=auc_ci_high,
            pr_auc=pr_auc,
            pr_auc_ci_low=pr_auc_ci_low,
            pr_auc_ci_high=pr_auc_ci_high,
            n_samples=len(y),
        )

    def _compute_feature_importance(
        self,
        X: NDArray,
        y: NDArray,
        feature_names: list[str],
    ) -> dict[str, float]:
        """Compute feature importance using coefficient magnitudes."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        if len(np.unique(y)) < 2:
            return {f: 0.0 for f in feature_names}

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = LogisticRegression(max_iter=1000, random_state=self.random_state)
        try:
            model.fit(X_scaled, y)
        except Exception as exc:
            if self.strict_mode:
                raise RuntimeError("Feature importance fit failed") from exc
            logger.warning("Feature importance fit failed; returning zeros: %s", exc)
            return {f: 0.0 for f in feature_names}

        importance = dict(zip(feature_names, np.abs(model.coef_[0])))
        return importance

    def _run_feature_ablation(
        self,
        X: NDArray,
        y: NDArray,
        feature_names: list[str],
        full_auc: float,
    ) -> list[FeatureAblation]:
        """Run leave-one-out feature ablation."""
        ablations = []

        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import roc_auc_score

        if len(np.unique(y)) < 2:
            return []

        for i, feature in enumerate(feature_names):
            # Create X without this feature
            mask = [j for j in range(len(feature_names)) if j != i]
            X_ablated = X[:, mask]

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_ablated)

            model = LogisticRegression(max_iter=1000, random_state=self.random_state)

            try:
                y_pred = self._cross_val_predict_proba(X_scaled, y, model)
                auc_without = roc_auc_score(y, y_pred)
            except Exception as exc:
                if self.strict_mode:
                    raise RuntimeError(
                        f"Feature ablation failed for feature '{feature}'"
                    ) from exc
                logger.warning(
                    "Feature ablation fallback for '%s' after CV failure: %s",
                    feature,
                    exc,
                )
                model.fit(X_scaled, y)
                y_pred = model.predict_proba(X_scaled)[:, 1]
                auc_without = roc_auc_score(y, y_pred)

            ablations.append(FeatureAblation(
                feature_removed=feature,
                auc_without=auc_without,
                auc_drop=full_auc - auc_without,
                importance_rank=0,  # Set later
            ))

        # Rank by importance (largest AUC drop = most important)
        ablations.sort(key=lambda x: x.auc_drop, reverse=True)
        for i, ablation in enumerate(ablations):
            ablation.importance_rank = i + 1

        return ablations

    def _within_bucket_analysis(
        self,
        X: NDArray,
        y: NDArray,
        buckets: list,
        geometry_idx: list[int],
        feature_names: list[str],
    ) -> list[BucketAnalysis]:
        """Analyze predictive power within each bucket."""
        analyses = []

        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import roc_auc_score, average_precision_score

        # Group by bucket
        bucket_data: dict[tuple, list[int]] = {}
        for i, bucket in enumerate(buckets):
            if bucket is None:
                continue
            if bucket not in bucket_data:
                bucket_data[bucket] = []
            bucket_data[bucket].append(i)

        for bucket, indices in bucket_data.items():
            if len(indices) < 20:  # Skip small buckets
                continue

            X_bucket = X[indices][:, geometry_idx]
            y_bucket = y[indices]

            if len(np.unique(y_bucket)) < 2:
                continue

            try:
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X_bucket)

                model = LogisticRegression(max_iter=1000, random_state=self.random_state)
                y_pred = self._cross_val_predict_proba(X_scaled, y_bucket, model)

                auc = roc_auc_score(y_bucket, y_pred)
                pr_auc = average_precision_score(y_bucket, y_pred)

                analyses.append(BucketAnalysis(
                    bucket=bucket,
                    n_samples=len(indices),
                    auc=auc,
                    pr_auc=pr_auc,
                    geometry_predicts=auc > 0.55,
                ))
            except Exception as exc:
                if self.strict_mode:
                    raise RuntimeError(f"Within-bucket analysis failed for bucket={bucket}") from exc
                logger.warning(
                    "Skipping bucket %s after analysis failure: %s",
                    bucket,
                    exc,
                )
                continue

        return analyses

    def _empty_result(self) -> PredictiveValidityResult:
        """Return empty result when no data available."""
        empty_comparison = ModelComparison(
            name="empty", features=[],
            auc=0.5, auc_ci_low=0.5, auc_ci_high=0.5,
            pr_auc=0.5, pr_auc_ci_low=0.5, pr_auc_ci_high=0.5,
            n_samples=0
        )

        return PredictiveValidityResult(
            baseline_model=empty_comparison,
            geometry_model=empty_comparison,
            full_model=empty_comparison,
            improvement_over_baseline=0.0,
            geometry_adds_value=False,
            feature_importance={},
            feature_ablations=[],
            bucket_analyses=[],
            buckets_where_geometry_predicts=0,
            total_buckets=0,
            n_tokens=0,
            n_positive=0,
            positive_rate=0.0,
        )


def compute_predictive_validity(
    diagnostic_results: list["DiagnosticResult"],
    stress_test_results: dict[int, float],
    n_bootstrap: int = 100,
) -> PredictiveValidityResult:
    """Convenience function for predictive validity analysis.

    Args:
        diagnostic_results: Diagnostic results with metrics.
        stress_test_results: Token failure rates.
        n_bootstrap: Number of bootstrap samples for CI.

    Returns:
        Comprehensive PredictiveValidityResult.
    """
    analyzer = PredictiveValidityAnalyzer(n_bootstrap=n_bootstrap)
    return analyzer.analyze(diagnostic_results, stress_test_results)


def format_validity_report(result: PredictiveValidityResult) -> str:
    """Format predictive validity results as a readable report."""
    lines = [
        "=" * 70,
        "PREDICTIVE VALIDITY ANALYSIS",
        "=" * 70,
        "",
        f"Samples: {result.n_tokens} tokens, {result.n_positive} with failures ({result.positive_rate:.1%})",
        "",
        "MODEL COMPARISON",
        "-" * 40,
        f"  Baseline (confounds): AUC = {result.baseline_model.auc:.3f} "
        f"[{result.baseline_model.auc_ci_low:.3f}, {result.baseline_model.auc_ci_high:.3f}]",
        f"  Geometry only:         AUC = {result.geometry_model.auc:.3f} "
        f"[{result.geometry_model.auc_ci_low:.3f}, {result.geometry_model.auc_ci_high:.3f}]",
        f"  Full model:            AUC = {result.full_model.auc:.3f} "
        f"[{result.full_model.auc_ci_low:.3f}, {result.full_model.auc_ci_high:.3f}]",
        "",
        f"  Improvement over baseline: {result.improvement_over_baseline:+.3f}",
        f"  Geometry adds value: {'YES' if result.geometry_adds_value else 'NO'}",
        "",
        "PR-AUC (for imbalanced data)",
        "-" * 40,
        f"  Baseline:    {result.baseline_model.pr_auc:.3f}",
        f"  Geometry:    {result.geometry_model.pr_auc:.3f}",
        f"  Full model:  {result.full_model.pr_auc:.3f}",
        "",
    ]

    if result.feature_importance:
        lines.extend([
            "FEATURE IMPORTANCE (coefficient magnitude)",
            "-" * 40,
        ])
        sorted_importance = sorted(
            result.feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )
        for feature, importance in sorted_importance:
            lines.append(f"  {feature:20s}: {importance:.3f}")
        lines.append("")

    if result.feature_ablations:
        lines.extend([
            "FEATURE ABLATION (AUC drop when removed)",
            "-" * 40,
        ])
        for ablation in result.feature_ablations[:8]:  # Top 8
            lines.append(
                f"  {ablation.importance_rank}. {ablation.feature_removed:15s}: "
                f"AUC drop = {ablation.auc_drop:+.3f}"
            )
        lines.append("")

    if result.bucket_analyses:
        lines.extend([
            "WITHIN-BUCKET ANALYSIS",
            "-" * 40,
            f"  Buckets where geometry predicts: {result.buckets_where_geometry_predicts}/{result.total_buckets}",
            "",
        ])
        for ba in sorted(result.bucket_analyses, key=lambda x: x.auc, reverse=True)[:5]:
            bucket_str = f"{ba.bucket[0].name}+{ba.bucket[1].name}"
            lines.append(
                f"  {bucket_str:25s}: AUC = {ba.auc:.3f}, n = {ba.n_samples}"
            )
        lines.append("")

    lines.append("=" * 70)

    return "\n".join(lines)
