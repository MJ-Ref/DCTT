"""External benchmark wrappers for evaluation."""

from dctt.benchmarks.humaneval import (
    HumanEvalConfig,
    HumanEvalResult,
    HumanEvalRunner,
    compare_humaneval_results,
    run_humaneval,
)
from dctt.benchmarks.gsm8k import (
    GSM8KConfig,
    GSM8KResult,
    GSM8KRunner,
    compare_gsm8k_results,
    run_gsm8k,
)

__all__ = [
    # HumanEval
    "HumanEvalConfig",
    "HumanEvalResult",
    "HumanEvalRunner",
    "compare_humaneval_results",
    "run_humaneval",
    # GSM8K
    "GSM8KConfig",
    "GSM8KResult",
    "GSM8KRunner",
    "compare_gsm8k_results",
    "run_gsm8k",
]
