"""Stress test runner.

This module provides utilities for running stress tests across
multiple tokens and aggregating results.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

from dctt.stress_tests.base import StressTest, StressTestResult

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)


@dataclass
class RunnerConfig:
    """Configuration for stress test runner."""

    n_cases_per_token: int = 10
    timeout_seconds: float = 30.0
    max_retries: int = 3
    batch_size: int = 10


class StressTestRunner:
    """Runs stress tests across multiple tokens.

    Example:
        >>> runner = StressTestRunner([CodeSyntaxTest(), MathFormatTest()])
        >>> results = runner.run(tokens, model_fn)
    """

    def __init__(
        self,
        tests: Sequence[StressTest],
        config: RunnerConfig | None = None,
    ) -> None:
        """Initialize runner.

        Args:
            tests: Stress tests to run.
            config: Runner configuration.
        """
        self.tests = list(tests)
        self.config = config or RunnerConfig()

    def run(
        self,
        tokens: Sequence[tuple[int, str]],
        model_fn: Callable[[str], str],
    ) -> dict[str, list[StressTestResult]]:
        """Run all stress tests on all tokens.

        Args:
            tokens: Sequence of (token_id, token_str) tuples.
            model_fn: Function that takes prompt and returns model response.

        Returns:
            Dictionary mapping test name to list of results.
        """
        all_results = {}

        for test in self.tests:
            logger.info(f"Running stress test: {test.name}")
            results = []

            for token_id, token_str in tokens:
                try:
                    result = test.run_single(
                        token_id=token_id,
                        token_str=token_str,
                        model_fn=model_fn,
                        n_cases=self.config.n_cases_per_token,
                    )
                    results.append(result)
                except Exception as e:
                    logger.warning(
                        f"Error running {test.name} on token {token_id}: {e}"
                    )
                    # Create failed result
                    results.append(StressTestResult(
                        token_id=token_id,
                        token_str=token_str,
                        total_tests=0,
                        failures=0,
                        failure_rate=1.0,
                        details={"error": str(e)},
                    ))

            all_results[test.name] = results
            logger.info(
                f"Completed {test.name}: {len(results)} tokens tested"
            )

        return all_results

    def run_for_token_type(
        self,
        tokens: Sequence[tuple[int, str]],
        model_fn: Callable[[str], str],
        token_type: str,
    ) -> dict[str, list[StressTestResult]]:
        """Run tests relevant to a specific token type.

        Args:
            tokens: Sequence of (token_id, token_str) tuples.
            model_fn: Model function.
            token_type: Token type to filter tests by.

        Returns:
            Dictionary mapping test name to results.
        """
        relevant_tests = [
            t for t in self.tests
            if token_type in t.target_token_types
        ]

        runner = StressTestRunner(relevant_tests, self.config)
        return runner.run(tokens, model_fn)

    def aggregate_results(
        self,
        results: dict[str, list[StressTestResult]],
    ) -> dict[int, dict[str, float]]:
        """Aggregate results by token.

        Args:
            results: Results from run().

        Returns:
            Dictionary mapping token_id to test_name -> failure_rate.
        """
        aggregated: dict[int, dict[str, float]] = {}

        for test_name, test_results in results.items():
            for result in test_results:
                if result.token_id not in aggregated:
                    aggregated[result.token_id] = {}
                aggregated[result.token_id][test_name] = result.failure_rate

        return aggregated

    def compute_overall_failure_rate(
        self,
        results: dict[str, list[StressTestResult]],
    ) -> dict[int, float]:
        """Compute overall failure rate per token.

        Args:
            results: Results from run().

        Returns:
            Dictionary mapping token_id to overall failure rate.
        """
        aggregated = self.aggregate_results(results)

        overall = {}
        for token_id, test_rates in aggregated.items():
            if test_rates:
                overall[token_id] = sum(test_rates.values()) / len(test_rates)
            else:
                overall[token_id] = 0.0

        return overall


def run_stress_tests(
    tokens: Sequence[tuple[int, str]],
    model_fn: Callable[[str], str],
    n_cases: int = 10,
) -> dict[str, list[StressTestResult]]:
    """Convenience function to run default stress tests.

    Args:
        tokens: Tokens to test.
        model_fn: Model function.
        n_cases: Cases per token.

    Returns:
        Results dictionary.
    """
    from dctt.stress_tests.code_syntax import CodeSyntaxTest
    from dctt.stress_tests.math_format import MathFormatTest

    tests = [CodeSyntaxTest(), MathFormatTest()]
    config = RunnerConfig(n_cases_per_token=n_cases)
    runner = StressTestRunner(tests, config)

    return runner.run(tokens, model_fn)
