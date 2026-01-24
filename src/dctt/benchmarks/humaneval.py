"""HumanEval benchmark wrapper for code generation evaluation.

This module provides integration with the HumanEval benchmark for measuring
the impact of embedding repairs on code generation quality.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class HumanEvalResult:
    """Results from HumanEval evaluation."""

    pass_at_1: float
    pass_at_10: float | None = None
    pass_at_100: float | None = None
    n_problems: int = 164
    n_samples_per_problem: int = 1
    results_by_problem: dict[str, dict[str, Any]] = field(default_factory=dict)
    total_time_seconds: float = 0.0


@dataclass
class HumanEvalConfig:
    """Configuration for HumanEval evaluation."""

    n_samples: int = 1
    temperature: float = 0.2
    top_p: float = 0.95
    max_tokens: int = 512
    timeout_per_problem: float = 10.0
    num_workers: int = 4


class HumanEvalRunner:
    """Runs HumanEval benchmark on a model."""

    def __init__(
        self,
        model_name: str,
        config: HumanEvalConfig | None = None,
    ) -> None:
        """Initialize HumanEval runner.

        Args:
            model_name: HuggingFace model name or path.
            config: Evaluation configuration.
        """
        self.model_name = model_name
        self.config = config or HumanEvalConfig()
        self._problems: list[dict[str, Any]] | None = None

    def load_problems(self) -> list[dict[str, Any]]:
        """Load HumanEval problems.

        Returns:
            List of problem dictionaries.
        """
        if self._problems is not None:
            return self._problems

        try:
            from human_eval.data import read_problems

            self._problems = list(read_problems().values())
        except ImportError:
            # Fallback: try to load from datasets
            try:
                from datasets import load_dataset

                dataset = load_dataset("openai_humaneval", split="test")
                self._problems = [dict(row) for row in dataset]
            except Exception as e:
                raise ImportError(
                    "Could not load HumanEval. Install with: "
                    "pip install human-eval or pip install datasets"
                ) from e

        return self._problems

    def generate_completions(
        self,
        model: Any,
        tokenizer: Any,
        problems: list[dict[str, Any]] | None = None,
    ) -> dict[str, list[str]]:
        """Generate completions for HumanEval problems.

        Args:
            model: The language model.
            tokenizer: The tokenizer.
            problems: Optional subset of problems.

        Returns:
            Dictionary mapping task_id to list of completions.
        """
        import torch

        if problems is None:
            problems = self.load_problems()

        completions: dict[str, list[str]] = {}

        for problem in problems:
            task_id = problem["task_id"]
            prompt = problem["prompt"]

            samples = []
            for _ in range(self.config.n_samples):
                inputs = tokenizer(prompt, return_tensors="pt")
                if torch.cuda.is_available():
                    inputs = {k: v.cuda() for k, v in inputs.items()}
                elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    inputs = {k: v.to("mps") for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=self.config.max_tokens,
                        temperature=self.config.temperature,
                        top_p=self.config.top_p,
                        do_sample=self.config.temperature > 0,
                        pad_token_id=tokenizer.eos_token_id,
                    )

                completion = tokenizer.decode(
                    outputs[0][inputs["input_ids"].shape[1] :],
                    skip_special_tokens=True,
                )

                # Stop at function end or common stop sequences
                stop_sequences = ["\ndef ", "\nclass ", "\nif __name__"]
                for stop in stop_sequences:
                    if stop in completion:
                        completion = completion[: completion.index(stop)]

                samples.append(completion)

            completions[task_id] = samples

        return completions

    def evaluate_completions(
        self,
        completions: dict[str, list[str]],
        problems: list[dict[str, Any]] | None = None,
    ) -> HumanEvalResult:
        """Evaluate completions against test cases.

        Args:
            completions: Dictionary mapping task_id to completions.
            problems: Optional problem list (for test cases).

        Returns:
            HumanEvalResult with pass rates.
        """
        import time

        if problems is None:
            problems = self.load_problems()

        problem_map = {p["task_id"]: p for p in problems}
        results_by_problem: dict[str, dict[str, Any]] = {}

        start_time = time.time()
        n_passed = 0
        n_total = 0

        for task_id, samples in completions.items():
            problem = problem_map.get(task_id)
            if problem is None:
                continue

            prompt = problem["prompt"]
            test = problem.get("test", "")
            entry_point = problem.get("entry_point", "")

            sample_results = []
            for sample in samples:
                full_code = prompt + sample + "\n" + test
                passed = self._run_test(full_code, entry_point)
                sample_results.append(passed)
                n_total += 1
                if passed:
                    n_passed += 1

            results_by_problem[task_id] = {
                "samples": samples,
                "passed": sample_results,
                "any_passed": any(sample_results),
            }

        total_time = time.time() - start_time

        # Compute pass@k
        pass_at_1 = self._compute_pass_at_k(results_by_problem, k=1)
        pass_at_10 = (
            self._compute_pass_at_k(results_by_problem, k=10)
            if self.config.n_samples >= 10
            else None
        )
        pass_at_100 = (
            self._compute_pass_at_k(results_by_problem, k=100)
            if self.config.n_samples >= 100
            else None
        )

        return HumanEvalResult(
            pass_at_1=pass_at_1,
            pass_at_10=pass_at_10,
            pass_at_100=pass_at_100,
            n_problems=len(results_by_problem),
            n_samples_per_problem=self.config.n_samples,
            results_by_problem=results_by_problem,
            total_time_seconds=total_time,
        )

    def _run_test(self, code: str, entry_point: str) -> bool:
        """Run test code in isolated environment.

        Args:
            code: Full code including function and tests.
            entry_point: Function name to test.

        Returns:
            True if all tests pass.
        """
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            f.flush()

            try:
                result = subprocess.run(
                    ["python", f.name],
                    capture_output=True,
                    timeout=self.config.timeout_per_problem,
                )
                return result.returncode == 0
            except subprocess.TimeoutExpired:
                return False
            except Exception:
                return False
            finally:
                Path(f.name).unlink(missing_ok=True)

    def _compute_pass_at_k(
        self,
        results: dict[str, dict[str, Any]],
        k: int,
    ) -> float:
        """Compute pass@k metric.

        Args:
            results: Results by problem.
            k: Number of samples to consider.

        Returns:
            pass@k score.
        """
        if not results:
            return 0.0

        scores = []
        for task_id, result in results.items():
            passed = result["passed"]
            n = len(passed)
            c = sum(passed)

            if n < k:
                # Not enough samples, use what we have
                score = 1.0 if c > 0 else 0.0
            else:
                # Exact pass@k computation
                # pass@k = 1 - C(n-c, k) / C(n, k)
                score = 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

            scores.append(score)

        return float(np.mean(scores))

    def run(
        self,
        model: Any | None = None,
        tokenizer: Any | None = None,
    ) -> HumanEvalResult:
        """Run full HumanEval evaluation.

        Args:
            model: Optional model (loads from model_name if not provided).
            tokenizer: Optional tokenizer.

        Returns:
            HumanEvalResult with pass rates.
        """
        if model is None or tokenizer is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype="auto",
                device_map="auto",
            )

        problems = self.load_problems()
        completions = self.generate_completions(model, tokenizer, problems)
        return self.evaluate_completions(completions, problems)


def run_humaneval(
    model_name: str,
    n_samples: int = 1,
    temperature: float = 0.2,
) -> HumanEvalResult:
    """Convenience function to run HumanEval.

    Args:
        model_name: HuggingFace model name or path.
        n_samples: Number of samples per problem.
        temperature: Sampling temperature.

    Returns:
        HumanEvalResult with pass rates.
    """
    config = HumanEvalConfig(n_samples=n_samples, temperature=temperature)
    runner = HumanEvalRunner(model_name, config)
    return runner.run()


def compare_humaneval_results(
    baseline: HumanEvalResult,
    repaired: HumanEvalResult,
) -> dict[str, Any]:
    """Compare HumanEval results before and after repair.

    Args:
        baseline: Results before repair.
        repaired: Results after repair.

    Returns:
        Comparison statistics.
    """
    delta_pass_1 = repaired.pass_at_1 - baseline.pass_at_1

    # Per-problem comparison
    improved = []
    regressed = []
    unchanged = []

    for task_id in baseline.results_by_problem:
        if task_id not in repaired.results_by_problem:
            continue

        baseline_passed = baseline.results_by_problem[task_id]["any_passed"]
        repaired_passed = repaired.results_by_problem[task_id]["any_passed"]

        if repaired_passed and not baseline_passed:
            improved.append(task_id)
        elif baseline_passed and not repaired_passed:
            regressed.append(task_id)
        else:
            unchanged.append(task_id)

    return {
        "delta_pass_at_1": delta_pass_1,
        "delta_pass_at_10": (
            (repaired.pass_at_10 - baseline.pass_at_10)
            if baseline.pass_at_10 is not None and repaired.pass_at_10 is not None
            else None
        ),
        "n_improved": len(improved),
        "n_regressed": len(regressed),
        "n_unchanged": len(unchanged),
        "improved_tasks": improved,
        "regressed_tasks": regressed,
    }
