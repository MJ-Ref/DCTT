"""Base classes for stress tests.

This module defines the abstract interface for stress tests that
evaluate token-level failure rates for causal validation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


class FailureType(Enum):
    """Types of failures in stress tests."""

    SYNTAX_ERROR = auto()  # Code doesn't parse
    COMPILE_ERROR = auto()  # Code doesn't compile
    RUNTIME_ERROR = auto()  # Code crashes at runtime
    WRONG_OUTPUT = auto()  # Incorrect result
    FORMAT_ERROR = auto()  # Wrong formatting
    TIMEOUT = auto()  # Execution timeout


@dataclass
class StressTestResult:
    """Result of a stress test for a single token.

    Attributes:
        token_id: Token being tested.
        token_str: String representation of token.
        total_tests: Number of test cases run.
        failures: Number of failures.
        failure_rate: Proportion of failures.
        failure_types: Count of each failure type.
        details: Additional result details.
    """

    token_id: int
    token_str: str
    total_tests: int
    failures: int
    failure_rate: float
    failure_types: dict[FailureType, int] = field(default_factory=dict)
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "token_id": self.token_id,
            "token_str": self.token_str,
            "total_tests": self.total_tests,
            "failures": self.failures,
            "failure_rate": self.failure_rate,
            "failure_types": {k.name: v for k, v in self.failure_types.items()},
            "details": self.details,
        }


class StressTest(ABC):
    """Abstract base class for stress tests.

    Stress tests evaluate model behavior on tasks that specifically
    exercise certain tokens, enabling causal validation of geometry
    metrics.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Test suite name."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description of the test."""
        pass

    @property
    @abstractmethod
    def target_token_types(self) -> list[str]:
        """Token types this test targets (e.g., "punctuation", "operator")."""
        pass

    @abstractmethod
    def generate_test_cases(
        self,
        token_id: int,
        token_str: str,
        n_cases: int = 10,
    ) -> list[dict]:
        """Generate test cases for a specific token.

        Args:
            token_id: Token to test.
            token_str: String representation.
            n_cases: Number of test cases to generate.

        Returns:
            List of test case dictionaries.
        """
        pass

    @abstractmethod
    def evaluate_response(
        self,
        test_case: dict,
        response: str,
    ) -> tuple[bool, FailureType | None]:
        """Evaluate model response against expected outcome.

        Args:
            test_case: The test case dictionary.
            response: Model's response.

        Returns:
            Tuple of (passed, failure_type or None).
        """
        pass

    def run_single(
        self,
        token_id: int,
        token_str: str,
        model_fn,
        n_cases: int = 10,
    ) -> StressTestResult:
        """Run stress test for a single token.

        Args:
            token_id: Token to test.
            token_str: String representation.
            model_fn: Function that takes prompt and returns response.
            n_cases: Number of test cases.

        Returns:
            StressTestResult for the token.
        """
        test_cases = self.generate_test_cases(token_id, token_str, n_cases)

        failures = 0
        failure_types: dict[FailureType, int] = {}

        for case in test_cases:
            prompt = case.get("prompt", "")
            try:
                response = model_fn(prompt)
                passed, failure_type = self.evaluate_response(case, response)
            except Exception:
                passed = False
                failure_type = FailureType.RUNTIME_ERROR

            if not passed:
                failures += 1
                if failure_type:
                    failure_types[failure_type] = failure_types.get(failure_type, 0) + 1

        failure_rate = failures / len(test_cases) if test_cases else 0.0

        return StressTestResult(
            token_id=token_id,
            token_str=token_str,
            total_tests=len(test_cases),
            failures=failures,
            failure_rate=failure_rate,
            failure_types=failure_types,
        )

    def run_batch(
        self,
        tokens: Sequence[tuple[int, str]],
        model_fn,
        n_cases: int = 10,
    ) -> list[StressTestResult]:
        """Run stress test for multiple tokens.

        Args:
            tokens: Sequence of (token_id, token_str) tuples.
            model_fn: Function that takes prompt and returns response.
            n_cases: Number of test cases per token.

        Returns:
            List of StressTestResult.
        """
        results = []
        for token_id, token_str in tokens:
            result = self.run_single(token_id, token_str, model_fn, n_cases)
            results.append(result)
        return results
