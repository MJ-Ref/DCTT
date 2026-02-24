"""Forced-token minimal-pair stress tests.

This module implements stress tests that force a target token to be copied
or inserted, and compares performance against a matched control token from
the same confound bucket.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from dctt.core.registry import register_stress_test
from dctt.stress_tests.base import FailureType, StressTest, StressTestResult

if TYPE_CHECKING:
    from dctt.core.types import DiagnosticResult


def build_minimal_pair_control_map(
    diagnostic_results: list["DiagnosticResult"],
    seed: int = 42,
) -> dict[int, tuple[int, str]]:
    """Build token -> matched control token map by diagnostic bucket.

    Controls are selected from the lower-severity subset within each
    (frequency tier, token type) bucket to preserve confounds while providing
    contrast for minimal-pair comparisons.
    """
    rng = np.random.default_rng(seed)
    bucket_to_results: dict[tuple, list["DiagnosticResult"]] = {}

    for result in diagnostic_results:
        bucket = result.bucket
        if bucket is None:
            continue
        bucket_to_results.setdefault(bucket, []).append(result)

    control_map: dict[int, tuple[int, str]] = {}

    for bucket_results in bucket_to_results.values():
        ordered = sorted(bucket_results, key=lambda item: item.severity)
        pool_size = max(3, len(ordered) // 5)
        control_pool = ordered[:pool_size]

        for result in ordered:
            candidates = [item for item in control_pool if item.token_id != result.token_id]
            if not candidates:
                candidates = [item for item in ordered if item.token_id != result.token_id]

            if not candidates:
                control = result
            else:
                control = candidates[int(rng.integers(0, len(candidates)))]

            control_map[result.token_id] = (
                control.token_id,
                control.token_info.token_str,
            )

    return control_map


@register_stress_test("forced_token_minimal_pair")
class ForcedTokenMinimalPairTest(StressTest):
    """Stress test with forced-token prompts and matched control comparisons."""

    def __init__(
        self,
        control_map: dict[int, tuple[int, str]],
        seed: int = 42,
        scoring_mode: str = "generation",
        min_logprob_margin: float = 0.0,
    ) -> None:
        self.control_map = control_map
        self.seed = seed
        self.scoring_mode = scoring_mode
        self.min_logprob_margin = float(min_logprob_margin)
        self._templates = [
            (
                "Copy the value between <token> and </token> exactly once.\n"
                "<token>{payload}</token>\n"
                "Output only the copied value."
            ),
            (
                "Replace [TOK] with the exact token shown in <token>.\n"
                "<token>{payload}</token>\n"
                "Text: A[TOK]B\n"
                "Output only the final text."
            ),
            (
                "Return exactly: START{payload}END\n"
                "Do not add extra text."
            ),
        ]

    @property
    def name(self) -> str:
        return "forced_token_minimal_pair"

    @property
    def description(self) -> str:
        return "Forced-token copy and insertion tasks with matched controls"

    @property
    def target_token_types(self) -> list[str]:
        return [
            "full_word",
            "subword",
            "numeric",
            "punctuation",
            "code_symbol",
            "whitespace",
            "special",
            "unknown",
        ]

    def _token_payload(self, token_str: str) -> tuple[str, bool]:
        """Return prompt payload and whether token should be escaped."""
        if token_str.strip() == "":
            escaped = token_str.encode("unicode_escape").decode("ascii")
            return escaped if escaped else "\\u0020", True
        return token_str, False

    def _passes_case(
        self,
        response: str,
        expected: str,
        escaped_mode: bool,
    ) -> bool:
        if not response.strip():
            return False
        if escaped_mode:
            response_norm = response.strip().replace("`", "")
            return expected in response_norm
        return expected in response

    def generate_test_cases(
        self,
        token_id: int,
        token_str: str,
        n_cases: int = 6,
    ) -> list[dict]:
        """Generate minimal-pair cases for token and matched control."""
        control_id, control_token = self.control_map.get(token_id, (token_id, token_str))

        target_payload, target_escaped = self._token_payload(token_str)
        control_payload, control_escaped = self._token_payload(control_token)

        cases = []
        for idx in range(max(1, n_cases)):
            template = self._templates[idx % len(self._templates)]
            cases.append({
                "target_prompt": template.format(payload=target_payload),
                "control_prompt": template.format(payload=control_payload),
                "target_expected": target_payload,
                "control_expected": control_payload,
                "target_escaped_mode": target_escaped,
                "control_escaped_mode": control_escaped,
                "control_token_id": control_id,
                "control_token_str": control_token,
            })

        return cases

    def evaluate_response(
        self,
        test_case: dict,
        response: str,
    ) -> tuple[bool, FailureType | None]:
        """Evaluate target-token response for compatibility with StressTest API."""
        passed = self._passes_case(
            response,
            test_case["target_expected"],
            bool(test_case["target_escaped_mode"]),
        )
        if passed:
            return True, None
        return False, FailureType.WRONG_OUTPUT

    def run_single(
        self,
        token_id: int,
        token_str: str,
        model_fn,
        n_cases: int = 6,
    ) -> StressTestResult:
        """Run forced-token minimal-pair test for a single token."""
        cases = self.generate_test_cases(token_id, token_str, n_cases)

        target_failures = 0
        control_failures = 0
        failure_types: dict[FailureType, int] = {}
        details: list[dict] = []

        for case in cases:
            target_response = ""
            control_response = ""
            target_passed = False
            control_passed = False
            target_margin: float | None = None
            control_margin: float | None = None
            target_scores: dict[str, float] | None = None
            control_scores: dict[str, float] | None = None

            use_logprob_choice = (
                self.scoring_mode == "logprob_choice"
                and hasattr(model_fn, "score_options")
            )

            if use_logprob_choice:
                try:
                    target_eval = model_fn.score_options(
                        case["target_prompt"],
                        [case["target_expected"], case["control_expected"]],
                    )
                    target_margin = float(target_eval.get("margin", 0.0))
                    target_scores = {
                        str(k): float(v)
                        for k, v in target_eval.get("scores", {}).items()
                    }
                    target_best = str(target_eval.get("best_option", ""))
                    target_passed = (
                        target_best == case["target_expected"]
                        and target_margin >= self.min_logprob_margin
                    )
                    target_response = "<logprob-choice>"
                except Exception as exc:
                    target_response = f"<error:{exc}>"
                    target_passed = False

                try:
                    control_eval = model_fn.score_options(
                        case["control_prompt"],
                        [case["control_expected"], case["target_expected"]],
                    )
                    control_margin = float(control_eval.get("margin", 0.0))
                    control_scores = {
                        str(k): float(v)
                        for k, v in control_eval.get("scores", {}).items()
                    }
                    control_best = str(control_eval.get("best_option", ""))
                    control_passed = (
                        control_best == case["control_expected"]
                        and control_margin >= self.min_logprob_margin
                    )
                    control_response = "<logprob-choice>"
                except Exception as exc:
                    control_response = f"<error:{exc}>"
                    control_passed = False
            else:
                try:
                    target_response = model_fn(case["target_prompt"])
                    target_passed = self._passes_case(
                        target_response,
                        case["target_expected"],
                        bool(case["target_escaped_mode"]),
                    )
                except Exception as exc:
                    target_response = f"<error:{exc}>"
                    target_passed = False

                try:
                    control_response = model_fn(case["control_prompt"])
                    control_passed = self._passes_case(
                        control_response,
                        case["control_expected"],
                        bool(case["control_escaped_mode"]),
                    )
                except Exception as exc:
                    control_response = f"<error:{exc}>"
                    control_passed = False

            if not target_passed:
                target_failures += 1
                failure_types[FailureType.WRONG_OUTPUT] = (
                    failure_types.get(FailureType.WRONG_OUTPUT, 0) + 1
                )

            if not control_passed:
                control_failures += 1

            details.append({
                "target_prompt": case["target_prompt"],
                "control_prompt": case["control_prompt"],
                "target_passed": target_passed,
                "control_passed": control_passed,
                "target_response": target_response[:400],
                "control_response": control_response[:400],
                "scoring_mode": "logprob_choice" if use_logprob_choice else "generation",
                "target_margin": target_margin,
                "control_margin": control_margin,
                "target_scores": target_scores or {},
                "control_scores": control_scores or {},
            })

        total_cases = len(cases)
        target_failure_rate = target_failures / total_cases if total_cases else 0.0
        control_failure_rate = control_failures / total_cases if total_cases else 0.0

        return StressTestResult(
            token_id=token_id,
            token_str=token_str,
            total_tests=total_cases,
            failures=target_failures,
            failure_rate=target_failure_rate,
            failure_types=failure_types,
            details={
                "control_token_id": cases[0]["control_token_id"] if cases else token_id,
                "control_token_str": cases[0]["control_token_str"] if cases else token_str,
                "scoring_mode": self.scoring_mode,
                "min_logprob_margin": self.min_logprob_margin,
                "control_failure_rate": control_failure_rate,
                "failure_gap": target_failure_rate - control_failure_rate,
                "cases": details,
            },
        )
