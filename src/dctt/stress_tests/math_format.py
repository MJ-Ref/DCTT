"""Math formatting stress tests.

This module implements stress tests for mathematical expression
formatting and arithmetic correctness.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from dctt.stress_tests.base import StressTest, StressTestResult, FailureType
from dctt.core.registry import register_stress_test

if TYPE_CHECKING:
    pass


@register_stress_test("math_format")
class MathFormatTest(StressTest):
    """Stress test for mathematical expression formatting.

    Tests model's ability to produce correctly formatted mathematical
    expressions with proper operator usage and parentheses.
    """

    @property
    def name(self) -> str:
        return "math_format"

    @property
    def description(self) -> str:
        return "Tests mathematical expression formatting and arithmetic"

    @property
    def target_token_types(self) -> list[str]:
        return ["punctuation", "numeric", "code_symbol"]

    def generate_test_cases(
        self,
        token_id: int,
        token_str: str,
        n_cases: int = 10,
    ) -> list[dict]:
        """Generate math-related test cases."""
        cases = []

        # Arithmetic problems
        arithmetic = [
            ("What is 15 + 27?", 42),
            ("What is 100 - 37?", 63),
            ("What is 12 * 8?", 96),
            ("What is 144 / 12?", 12),
            ("What is 2^10?", 1024),
            ("What is (5 + 3) * 2?", 16),
            ("What is 100 - (20 + 30)?", 50),
        ]

        for prompt, expected in arithmetic[:n_cases]:
            cases.append({
                "prompt": f"{prompt} Answer with just the number.",
                "type": "arithmetic",
                "expected": expected,
                "token_id": token_id,
                "token_str": token_str,
            })

        # Expression formatting
        formatting = [
            ("Write the expression: a plus b times c", "a + b * c"),
            ("Write the expression: x squared plus y squared", "x^2 + y^2"),
            ("Write the fraction: a over b", "a/b"),
        ]

        for prompt, pattern in formatting[:n_cases - len(cases)]:
            cases.append({
                "prompt": prompt,
                "type": "formatting",
                "pattern": pattern,
                "token_id": token_id,
                "token_str": token_str,
            })

        return cases[:n_cases]

    def evaluate_response(
        self,
        test_case: dict,
        response: str,
    ) -> tuple[bool, FailureType | None]:
        """Evaluate mathematical response.

        Args:
            test_case: The test case dictionary.
            response: Model's response.

        Returns:
            Tuple of (passed, failure_type or None).
        """
        if test_case.get("type") == "arithmetic":
            expected = test_case.get("expected")

            # Extract number from response
            numbers = re.findall(r"-?\d+\.?\d*", response)
            if not numbers:
                return False, FailureType.FORMAT_ERROR

            try:
                # Check if any extracted number matches expected
                for num_str in numbers:
                    num = float(num_str)
                    if abs(num - expected) < 0.01:
                        return True, None
            except ValueError:
                pass

            return False, FailureType.WRONG_OUTPUT

        elif test_case.get("type") == "formatting":
            pattern = test_case.get("pattern", "")

            # Normalize response
            response_normalized = response.lower().replace(" ", "")
            pattern_normalized = pattern.lower().replace(" ", "")

            # Check for key elements
            if pattern_normalized in response_normalized:
                return True, None

            # Check for balanced parentheses
            if not self._check_balanced(response):
                return False, FailureType.SYNTAX_ERROR

            return False, FailureType.FORMAT_ERROR

        return True, None

    def _check_balanced(self, expr: str) -> bool:
        """Check if parentheses are balanced."""
        count = 0
        for char in expr:
            if char == "(":
                count += 1
            elif char == ")":
                count -= 1
            if count < 0:
                return False
        return count == 0


@register_stress_test("arithmetic_chain")
class ArithmeticChainTest(StressTest):
    """Test for multi-step arithmetic problems."""

    @property
    def name(self) -> str:
        return "arithmetic_chain"

    @property
    def description(self) -> str:
        return "Tests multi-step arithmetic reasoning"

    @property
    def target_token_types(self) -> list[str]:
        return ["numeric", "punctuation"]

    def generate_test_cases(
        self,
        token_id: int,
        token_str: str,
        n_cases: int = 10,
    ) -> list[dict]:
        """Generate multi-step arithmetic problems."""
        cases = []

        problems = [
            {
                "prompt": "Calculate step by step: 5 + 3 = ?, then multiply by 2 = ?",
                "steps": [8, 16],
                "final": 16,
            },
            {
                "prompt": "Calculate: (10 - 3) * 4 + 2",
                "steps": [7, 28, 30],
                "final": 30,
            },
            {
                "prompt": "If x = 5 and y = 3, what is x * y + x?",
                "steps": [15, 20],
                "final": 20,
            },
        ]

        for problem in problems[:n_cases]:
            cases.append({
                "prompt": problem["prompt"],
                "type": "chain",
                "expected_final": problem["final"],
                "token_id": token_id,
                "token_str": token_str,
            })

        return cases

    def evaluate_response(
        self,
        test_case: dict,
        response: str,
    ) -> tuple[bool, FailureType | None]:
        """Check if final answer is correct."""
        expected = test_case.get("expected_final")

        numbers = re.findall(r"-?\d+\.?\d*", response)
        if not numbers:
            return False, FailureType.FORMAT_ERROR

        try:
            # Check last number (likely the final answer)
            final = float(numbers[-1])
            if abs(final - expected) < 0.01:
                return True, None
        except ValueError:
            pass

        return False, FailureType.WRONG_OUTPUT
