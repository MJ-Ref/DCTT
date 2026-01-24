"""GSM8K benchmark wrapper for math reasoning evaluation.

This module provides integration with the GSM8K benchmark for measuring
the impact of embedding repairs on mathematical reasoning.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class GSM8KResult:
    """Results from GSM8K evaluation."""

    accuracy: float
    n_correct: int
    n_total: int
    results_by_problem: list[dict[str, Any]] = field(default_factory=list)
    total_time_seconds: float = 0.0


@dataclass
class GSM8KConfig:
    """Configuration for GSM8K evaluation."""

    n_shots: int = 8
    max_tokens: int = 512
    temperature: float = 0.0
    use_chain_of_thought: bool = True
    timeout_per_problem: float = 30.0


# Few-shot examples for chain-of-thought prompting
FEW_SHOT_EXAMPLES = [
    {
        "question": "There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
        "answer": "There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6 trees planted. The answer is 6.",
    },
    {
        "question": "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
        "answer": "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5.",
    },
    {
        "question": "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
        "answer": "Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The answer is 39.",
    },
    {
        "question": "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?",
        "answer": "Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8 lollipops. The answer is 8.",
    },
    {
        "question": "Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?",
        "answer": "Shawn started with 5 toys. He got 2 toys from his mom and 2 from his dad. So he got 2 + 2 = 4 more toys. 5 + 4 = 9. The answer is 9.",
    },
    {
        "question": "There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?",
        "answer": "There were originally 9 computers. For each of 4 days (monday to thursday), 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 = 29. The answer is 29.",
    },
    {
        "question": "Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?",
        "answer": "Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more on wednesday, he had 35 - 2 = 33 golf balls. The answer is 33.",
    },
    {
        "question": "Olivia has $23. She bought five bagels for $3 each. How much money does she have left?",
        "answer": "Olivia had $23. She bought 5 bagels for $3 each, spending 5 * $3 = $15. She has $23 - $15 = $8 left. The answer is 8.",
    },
]


class GSM8KRunner:
    """Runs GSM8K benchmark on a model."""

    def __init__(
        self,
        model_name: str,
        config: GSM8KConfig | None = None,
    ) -> None:
        """Initialize GSM8K runner.

        Args:
            model_name: HuggingFace model name or path.
            config: Evaluation configuration.
        """
        self.model_name = model_name
        self.config = config or GSM8KConfig()
        self._dataset: list[dict[str, Any]] | None = None

    def load_dataset(self) -> list[dict[str, Any]]:
        """Load GSM8K dataset.

        Returns:
            List of problem dictionaries.
        """
        if self._dataset is not None:
            return self._dataset

        try:
            from datasets import load_dataset

            dataset = load_dataset("gsm8k", "main", split="test")
            self._dataset = [dict(row) for row in dataset]
        except Exception as e:
            raise ImportError(
                "Could not load GSM8K. Install with: pip install datasets"
            ) from e

        return self._dataset

    def build_prompt(self, question: str) -> str:
        """Build prompt with few-shot examples.

        Args:
            question: The question to answer.

        Returns:
            Full prompt with examples.
        """
        if not self.config.use_chain_of_thought:
            return f"Question: {question}\nAnswer: "

        prompt_parts = []
        for example in FEW_SHOT_EXAMPLES[: self.config.n_shots]:
            prompt_parts.append(f"Question: {example['question']}")
            prompt_parts.append(f"Answer: {example['answer']}")
            prompt_parts.append("")

        prompt_parts.append(f"Question: {question}")
        prompt_parts.append("Answer:")

        return "\n".join(prompt_parts)

    def extract_answer(self, response: str) -> int | float | None:
        """Extract numerical answer from response.

        Args:
            response: Model response.

        Returns:
            Extracted number or None.
        """
        # Look for "The answer is X" pattern
        answer_pattern = r"[Tt]he answer is[:\s]*(-?[\d,]+\.?\d*)"
        match = re.search(answer_pattern, response)
        if match:
            try:
                return float(match.group(1).replace(",", ""))
            except ValueError:
                pass

        # Fallback: find last number in response
        numbers = re.findall(r"-?[\d,]+\.?\d*", response)
        if numbers:
            try:
                return float(numbers[-1].replace(",", ""))
            except ValueError:
                pass

        return None

    def extract_gold_answer(self, answer_str: str) -> int | float | None:
        """Extract gold answer from GSM8K answer string.

        Args:
            answer_str: Answer string from dataset.

        Returns:
            Extracted number.
        """
        # GSM8K format: "#### NUMBER"
        match = re.search(r"####\s*(-?[\d,]+\.?\d*)", answer_str)
        if match:
            try:
                return float(match.group(1).replace(",", ""))
            except ValueError:
                pass
        return None

    def generate_answer(
        self,
        model: Any,
        tokenizer: Any,
        question: str,
    ) -> str:
        """Generate answer for a question.

        Args:
            model: The language model.
            tokenizer: The tokenizer.
            question: The question.

        Returns:
            Generated response.
        """
        import torch

        prompt = self.build_prompt(question)
        inputs = tokenizer(prompt, return_tensors="pt")

        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            inputs = {k: v.to("mps") for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=self.config.max_tokens,
                temperature=self.config.temperature if self.config.temperature > 0 else None,
                do_sample=self.config.temperature > 0,
                pad_token_id=tokenizer.eos_token_id,
            )

        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
        )

        return response

    def evaluate(
        self,
        model: Any,
        tokenizer: Any,
        problems: list[dict[str, Any]] | None = None,
        max_problems: int | None = None,
    ) -> GSM8KResult:
        """Run evaluation on GSM8K.

        Args:
            model: The language model.
            tokenizer: The tokenizer.
            problems: Optional subset of problems.
            max_problems: Optional limit on number of problems.

        Returns:
            GSM8KResult with accuracy.
        """
        import time

        if problems is None:
            problems = self.load_dataset()

        if max_problems is not None:
            problems = problems[:max_problems]

        start_time = time.time()
        results = []
        n_correct = 0

        for problem in problems:
            question = problem["question"]
            gold_answer_str = problem["answer"]

            gold = self.extract_gold_answer(gold_answer_str)
            response = self.generate_answer(model, tokenizer, question)
            predicted = self.extract_answer(response)

            is_correct = (
                gold is not None
                and predicted is not None
                and abs(gold - predicted) < 1e-6
            )

            if is_correct:
                n_correct += 1

            results.append({
                "question": question,
                "gold_answer": gold,
                "predicted_answer": predicted,
                "response": response,
                "correct": is_correct,
            })

        total_time = time.time() - start_time

        return GSM8KResult(
            accuracy=n_correct / len(results) if results else 0.0,
            n_correct=n_correct,
            n_total=len(results),
            results_by_problem=results,
            total_time_seconds=total_time,
        )

    def run(
        self,
        model: Any | None = None,
        tokenizer: Any | None = None,
        max_problems: int | None = None,
    ) -> GSM8KResult:
        """Run full GSM8K evaluation.

        Args:
            model: Optional model (loads from model_name if not provided).
            tokenizer: Optional tokenizer.
            max_problems: Optional limit on problems.

        Returns:
            GSM8KResult with accuracy.
        """
        if model is None or tokenizer is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype="auto",
                device_map="auto",
            )

        return self.evaluate(model, tokenizer, max_problems=max_problems)


def run_gsm8k(
    model_name: str,
    n_shots: int = 8,
    max_problems: int | None = None,
) -> GSM8KResult:
    """Convenience function to run GSM8K.

    Args:
        model_name: HuggingFace model name or path.
        n_shots: Number of few-shot examples.
        max_problems: Optional limit on problems.

    Returns:
        GSM8KResult with accuracy.
    """
    config = GSM8KConfig(n_shots=n_shots)
    runner = GSM8KRunner(model_name, config)
    return runner.run(max_problems=max_problems)


def compare_gsm8k_results(
    baseline: GSM8KResult,
    repaired: GSM8KResult,
) -> dict[str, Any]:
    """Compare GSM8K results before and after repair.

    Args:
        baseline: Results before repair.
        repaired: Results after repair.

    Returns:
        Comparison statistics.
    """
    delta_accuracy = repaired.accuracy - baseline.accuracy

    # Per-problem comparison
    improved = []
    regressed = []

    for i, (base, rep) in enumerate(
        zip(baseline.results_by_problem, repaired.results_by_problem)
    ):
        if rep["correct"] and not base["correct"]:
            improved.append(i)
        elif base["correct"] and not rep["correct"]:
            regressed.append(i)

    return {
        "delta_accuracy": delta_accuracy,
        "baseline_accuracy": baseline.accuracy,
        "repaired_accuracy": repaired.accuracy,
        "n_improved": len(improved),
        "n_regressed": len(regressed),
        "net_improvement": len(improved) - len(regressed),
        "improved_indices": improved,
        "regressed_indices": regressed,
    }
