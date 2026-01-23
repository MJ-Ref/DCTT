"""Matched control selection for causal analysis.

This module implements propensity score matching and stratified sampling
for creating valid control groups.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from dctt.core.types import DiagnosticResult


@dataclass
class MatchedPair:
    """A matched treatment-control pair."""

    treatment_id: int
    control_id: int
    match_quality: float  # Lower is better (distance metric)


class MatchedControlSelector:
    """Selects matched controls for treatment tokens.

    Uses stratified sampling within (frequency_tier, token_type) buckets
    to ensure fair comparison.
    """

    def __init__(
        self,
        match_on: list[str] | None = None,
        n_controls_per_treatment: int = 1,
    ) -> None:
        """Initialize selector.

        Args:
            match_on: Variables to match on.
            n_controls_per_treatment: Number of controls per treatment.
        """
        self.match_on = match_on or ["frequency_tier", "token_type"]
        self.n_controls_per_treatment = n_controls_per_treatment

    def select(
        self,
        treatment_results: list["DiagnosticResult"],
        all_results: list["DiagnosticResult"],
    ) -> list["DiagnosticResult"]:
        """Select matched controls for treatment tokens.

        Args:
            treatment_results: Treatment group tokens.
            all_results: All diagnostic results to select from.

        Returns:
            List of matched control tokens.
        """
        treatment_ids = {r.token_id for r in treatment_results}

        # Group potential controls by bucket
        potential_controls: dict[tuple, list["DiagnosticResult"]] = {}
        for result in all_results:
            if result.token_id in treatment_ids:
                continue
            if result.bucket is None:
                continue
            if result.bucket not in potential_controls:
                potential_controls[result.bucket] = []
            potential_controls[result.bucket].append(result)

        # Match each treatment to control in same bucket
        controls = []
        used_ids = set()

        for treatment in treatment_results:
            bucket = treatment.bucket
            if bucket is None or bucket not in potential_controls:
                continue

            # Find best match (lowest severity in same bucket)
            candidates = [
                r for r in potential_controls[bucket]
                if r.token_id not in used_ids
            ]

            if not candidates:
                continue

            # Sort by severity (ascending) - want "normal" tokens as controls
            candidates.sort(key=lambda r: r.severity)

            # Take top n_controls
            for i in range(min(self.n_controls_per_treatment, len(candidates))):
                control = candidates[i]
                controls.append(control)
                used_ids.add(control.token_id)

        return controls


def create_matched_controls(
    treatment_results: list["DiagnosticResult"],
    all_results: list["DiagnosticResult"],
    n_controls: int = 1,
) -> list["DiagnosticResult"]:
    """Convenience function to create matched controls.

    Args:
        treatment_results: Treatment group.
        all_results: All results to select from.
        n_controls: Controls per treatment.

    Returns:
        List of matched control tokens.
    """
    selector = MatchedControlSelector(n_controls_per_treatment=n_controls)
    return selector.select(treatment_results, all_results)
