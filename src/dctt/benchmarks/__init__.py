"""External benchmark wrappers for evaluation."""

from dctt.benchmarks.humaneval import HumanEvalBenchmark, run_humaneval
from dctt.benchmarks.gsm8k import GSM8kBenchmark, run_gsm8k
from dctt.benchmarks.regression import RegressionSuite, run_regression_suite

__all__ = [
    "HumanEvalBenchmark",
    "run_humaneval",
    "GSM8kBenchmark",
    "run_gsm8k",
    "RegressionSuite",
    "run_regression_suite",
]
