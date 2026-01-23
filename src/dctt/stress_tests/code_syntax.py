"""Code syntax stress tests.

This module implements stress tests that evaluate model performance
on code generation tasks, specifically targeting syntax-critical tokens
like brackets, parentheses, and operators.
"""

from __future__ import annotations

import ast
import re
from typing import TYPE_CHECKING

from dctt.stress_tests.base import StressTest, StressTestResult, FailureType
from dctt.core.registry import register_stress_test

if TYPE_CHECKING:
    pass


@register_stress_test("code_syntax")
class CodeSyntaxTest(StressTest):
    """Stress test for code syntax correctness.

    Tests model's ability to generate syntactically correct code,
    with particular focus on bracket balancing and proper use of
    punctuation.
    """

    @property
    def name(self) -> str:
        return "code_syntax"

    @property
    def description(self) -> str:
        return "Tests syntactic correctness of generated Python code"

    @property
    def target_token_types(self) -> list[str]:
        return ["punctuation", "code_symbol", "whitespace"]

    def __init__(self, language: str = "python") -> None:
        """Initialize test.

        Args:
            language: Programming language to test.
        """
        self.language = language
        self._prompts = self._get_prompts()

    def _get_prompts(self) -> list[dict]:
        """Get prompt templates for code generation."""
        return [
            {
                "type": "function_def",
                "prompt": "Write a Python function called `{name}` that {task}. Only output the function code, nothing else.",
                "tasks": [
                    ("add_numbers", "takes two numbers and returns their sum"),
                    ("multiply", "takes two numbers and returns their product"),
                    ("factorial", "computes the factorial of a number"),
                    ("is_even", "returns True if the number is even"),
                    ("reverse_string", "reverses a string"),
                ],
            },
            {
                "type": "bracket_completion",
                "prompt": "Complete this Python code by adding the missing closing brackets:\n```python\n{code}\n```\nOutput only the complete code.",
                "codes": [
                    "def foo(x):\n    return (x + 1",
                    "data = {'key': [1, 2, 3",
                    "result = sum([x for x in range(10",
                    "if (a > b) and (c < d",
                ],
            },
            {
                "type": "indentation",
                "prompt": "Fix the indentation in this Python code:\n```python\n{code}\n```\nOutput only the corrected code.",
                "codes": [
                    "def foo():\nreturn 1",
                    "for i in range(10):\nprint(i)",
                    "if True:\nx = 1\nelse:\ny = 2",
                ],
            },
        ]

    def generate_test_cases(
        self,
        token_id: int,
        token_str: str,
        n_cases: int = 10,
    ) -> list[dict]:
        """Generate test cases that exercise the target token.

        Args:
            token_id: Token to test.
            token_str: String representation.
            n_cases: Number of test cases.

        Returns:
            List of test case dictionaries.
        """
        cases = []

        # Determine which prompts are relevant for this token
        relevant_prompts = []

        # Bracket tokens
        if token_str in "()[]{}":
            relevant_prompts.extend(
                [p for p in self._prompts if p["type"] == "bracket_completion"]
            )

        # Colon, def, etc
        if token_str in (":", "def", "class", "if", "for", "while"):
            relevant_prompts.extend(
                [p for p in self._prompts if p["type"] == "function_def"]
            )

        # Whitespace/indentation
        if token_str.strip() == "" or token_str in ("\n", "\t", "    "):
            relevant_prompts.extend(
                [p for p in self._prompts if p["type"] == "indentation"]
            )

        # Default to function_def for other tokens
        if not relevant_prompts:
            relevant_prompts = [p for p in self._prompts if p["type"] == "function_def"]

        # Generate cases from relevant prompts
        for prompt_template in relevant_prompts:
            if prompt_template["type"] == "function_def":
                for name, task in prompt_template["tasks"][:n_cases]:
                    prompt = prompt_template["prompt"].format(name=name, task=task)
                    cases.append({
                        "prompt": prompt,
                        "type": "function_def",
                        "expected_contains": f"def {name}",
                        "token_id": token_id,
                        "token_str": token_str,
                    })
            elif prompt_template["type"] in ("bracket_completion", "indentation"):
                for code in prompt_template.get("codes", [])[:n_cases]:
                    prompt = prompt_template["prompt"].format(code=code)
                    cases.append({
                        "prompt": prompt,
                        "type": prompt_template["type"],
                        "original_code": code,
                        "token_id": token_id,
                        "token_str": token_str,
                    })

        return cases[:n_cases]

    def evaluate_response(
        self,
        test_case: dict,
        response: str,
    ) -> tuple[bool, FailureType | None]:
        """Evaluate if the response is syntactically correct Python.

        Args:
            test_case: The test case dictionary.
            response: Model's response.

        Returns:
            Tuple of (passed, failure_type or None).
        """
        # Extract code from response (handle markdown blocks)
        code = self._extract_code(response)

        if not code.strip():
            return False, FailureType.FORMAT_ERROR

        # Try to parse the code
        try:
            ast.parse(code)
        except SyntaxError:
            return False, FailureType.SYNTAX_ERROR

        # Additional checks based on test type
        if test_case.get("type") == "function_def":
            expected = test_case.get("expected_contains", "")
            if expected and expected not in code:
                return False, FailureType.WRONG_OUTPUT

        return True, None

    def _extract_code(self, response: str) -> str:
        """Extract code from response, handling markdown blocks."""
        # Try to find code block
        match = re.search(r"```(?:python)?\n?(.*?)```", response, re.DOTALL)
        if match:
            return match.group(1).strip()

        # Otherwise return cleaned response
        lines = []
        for line in response.split("\n"):
            # Skip obvious non-code lines
            if line.startswith("#") and not line.startswith("# "):
                continue
            lines.append(line)

        return "\n".join(lines).strip()


@register_stress_test("bracket_balance")
class BracketBalanceTest(StressTest):
    """Focused test for bracket balancing."""

    @property
    def name(self) -> str:
        return "bracket_balance"

    @property
    def description(self) -> str:
        return "Tests ability to maintain balanced brackets"

    @property
    def target_token_types(self) -> list[str]:
        return ["punctuation", "code_symbol"]

    def generate_test_cases(
        self,
        token_id: int,
        token_str: str,
        n_cases: int = 10,
    ) -> list[dict]:
        """Generate bracket balancing test cases."""
        cases = []

        templates = [
            "Complete: (a + b",
            "Complete: [[1, 2], [3, 4",
            "Complete: {'a': {'b': 1",
            "Complete: func(arg1, func2(arg2",
            "Complete: [x for x in range(10) if x > 5",
        ]

        for template in templates[:n_cases]:
            cases.append({
                "prompt": template,
                "type": "bracket_completion",
                "token_id": token_id,
                "token_str": token_str,
            })

        return cases

    def evaluate_response(
        self,
        test_case: dict,
        response: str,
    ) -> tuple[bool, FailureType | None]:
        """Check if brackets are balanced."""
        # Count brackets
        brackets = {"(": ")", "[": "]", "{": "}"}
        stack = []

        for char in response:
            if char in brackets:
                stack.append(brackets[char])
            elif char in brackets.values():
                if not stack or stack[-1] != char:
                    return False, FailureType.SYNTAX_ERROR
                stack.pop()

        if stack:
            return False, FailureType.SYNTAX_ERROR

        return True, None
