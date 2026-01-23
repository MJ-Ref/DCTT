"""Causal effect analysis for repair experiments.

This module implements analysis of causal effects from embedding repairs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from dctt.core.types import RepairResult
    from dctt.stress_tests.base import StressTestResult


@dataclass
class CausalEffectResult:
    """Results of causal effect analysis."""

    ate: float  # Average Treatment Effect
    ate_ci_lower: float
    ate_ci_upper: float
    p_value: float
    n_treatment: int
    n_control: int


class CausalRepairAnalyzer:
    """Analyzes causal effects of embedding repairs."""

    def __init__(self, seed: int = 42) -> None:
        """Initialize analyzer."""
        self.seed = seed

    def analyze(
        self,
        treatment_results: dict[int, float],  # token_id -> post-repair failure rate
        control_results: dict[int, float],  # token_id -> post-placebo failure rate
        baseline_results: dict[int, float],  # token_id -> pre-repair failure rate
    ) -> CausalEffectResult:
        """Analyze causal effect of repairs.

        Args:
            treatment_results: Failure rates after repair.
            control_results: Failure rates after placebo.
            baseline_results: Failure rates before any intervention.

        Returns:
            CausalEffectResult with ATE and statistical tests.
        """
        # Compute changes for treatment group
        treatment_changes = []
        for token_id, post_rate in treatment_results.items():
            if token_id in baseline_results:
                pre_rate = baseline_results[token_id]
                treatment_changes.append(post_rate - pre_rate)

        # Compute changes for control group
        control_changes = []
        for token_id, post_rate in control_results.items():
            if token_id in baseline_results:
                pre_rate = baseline_results[token_id]
                control_changes.append(post_rate - pre_rate)

        treatment_changes = np.array(treatment_changes)
        control_changes = np.array(control_changes)

        if len(treatment_changes) == 0 or len(control_changes) == 0:
            return CausalEffectResult(
                ate=0.0,
                ate_ci_lower=0.0,
                ate_ci_upper=0.0,
                p_value=1.0,
                n_treatment=len(treatment_changes),
                n_control=len(control_changes),
            )

        # Compute ATE (difference-in-differences)
        ate = np.mean(treatment_changes) - np.mean(control_changes)

        # Bootstrap CI
        from dctt.evaluation.statistics import bootstrap_ci
        combined = np.concatenate([treatment_changes, control_changes])
        n_treatment = len(treatment_changes)

        def did_statistic(data: np.ndarray) -> float:
            return np.mean(data[:n_treatment]) - np.mean(data[n_treatment:])

        ci = bootstrap_ci(combined, did_statistic, seed=self.seed)

        # Permutation test
        from dctt.evaluation.statistics import paired_permutation_test
        # Simple two-sample test
        rng = np.random.default_rng(self.seed)
        observed = ate
        count = 0
        n_perms = 10000

        for _ in range(n_perms):
            perm = rng.permutation(combined)
            perm_ate = np.mean(perm[:n_treatment]) - np.mean(perm[n_treatment:])
            if abs(perm_ate) >= abs(observed):
                count += 1

        p_value = (count + 1) / (n_perms + 1)

        return CausalEffectResult(
            ate=ate,
            ate_ci_lower=ci.lower,
            ate_ci_upper=ci.upper,
            p_value=p_value,
            n_treatment=n_treatment,
            n_control=len(control_changes),
        )


def compute_causal_effect(
    treatment_results: dict[int, float],
    control_results: dict[int, float],
    baseline_results: dict[int, float],
) -> CausalEffectResult:
    """Convenience function for causal effect analysis."""
    analyzer = CausalRepairAnalyzer()
    return analyzer.analyze(treatment_results, control_results, baseline_results)
