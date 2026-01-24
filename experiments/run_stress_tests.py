#!/usr/bin/env python3
"""Stress Test Experiment for DCTT.

This script runs stress tests on high-severity vs low-severity tokens to validate
that geometry metrics predict model failures.

Usage:
    python experiments/run_stress_tests.py model=qwen2_5_coder_7b
    python experiments/run_stress_tests.py model=qwen2_5_coder_7b stress_test.n_tokens=50
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_diagnostic_results(results_path: Path) -> list[dict]:
    """Load diagnostic results from JSON file."""
    with open(results_path) as f:
        return json.load(f)


def select_tokens_for_stress_test(
    results: list[dict],
    n_high_severity: int = 50,
    n_low_severity: int = 50,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """Select high and low severity tokens for stress testing.

    Args:
        results: Diagnostic results.
        n_high_severity: Number of high-severity tokens to select.
        n_low_severity: Number of low-severity tokens to select.
        seed: Random seed.

    Returns:
        Tuple of (high_severity_tokens, low_severity_tokens).
    """
    # Sort by severity
    sorted_results = sorted(results, key=lambda r: r['severity'], reverse=True)

    # Get high severity (top percentile)
    high_severity = sorted_results[:n_high_severity]

    # Get low severity (bottom percentile)
    low_severity = sorted_results[-n_low_severity:]

    return high_severity, low_severity


def create_model_fn(model, tokenizer, device: str = "cpu") -> Callable[[str], str]:
    """Create a model inference function.

    Args:
        model: HuggingFace model.
        tokenizer: Tokenizer.
        device: Device to run on.

    Returns:
        Function that takes prompt and returns model response.
    """
    import torch

    def model_fn(prompt: str, max_new_tokens: int = 256) -> str:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )

        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        return response

    return model_fn


def run_syntax_stress_test(
    token_id: int,
    token_str: str,
    model_fn: Callable[[str], str],
    n_prompts: int = 5,
) -> dict:
    """Run syntax stress test for a single token.

    Args:
        token_id: Token ID.
        token_str: Token string.
        model_fn: Model inference function.
        n_prompts: Number of prompts to test.

    Returns:
        Dictionary with test results.
    """
    import ast
    import re

    # Generate prompts that are likely to exercise various tokens
    prompts = [
        f"Write a Python function that adds two numbers. Only output the code:\n```python\n",
        f"Complete this Python code:\ndef calculate(x, y):\n    result = (x + y",
        f"Write a Python function to reverse a string. Only output the code:\n```python\n",
        f"Fix this Python code to make it valid:\nif (x > 0:\n    print(x)",
        f"Write a Python function that returns the factorial of n. Only output code:\n```python\n",
    ][:n_prompts]

    results = []
    for prompt in prompts:
        try:
            response = model_fn(prompt)

            # Extract code from response
            code_match = re.search(r"```(?:python)?\n?(.*?)```", response, re.DOTALL)
            if code_match:
                code = code_match.group(1).strip()
            else:
                code = response.strip()

            # Try to parse
            try:
                ast.parse(code)
                passed = True
                error = None
            except SyntaxError as e:
                passed = False
                error = str(e)

        except Exception as e:
            passed = False
            error = str(e)

        results.append({
            "prompt": prompt[:50] + "...",
            "passed": passed,
            "error": error,
        })

    n_passed = sum(1 for r in results if r["passed"])
    failure_rate = 1 - (n_passed / len(results)) if results else 1.0

    return {
        "token_id": token_id,
        "token_str": token_str,
        "n_tests": len(results),
        "n_passed": n_passed,
        "failure_rate": failure_rate,
        "details": results,
    }


def analyze_stress_test_results(
    high_severity_results: list[dict],
    low_severity_results: list[dict],
) -> dict:
    """Analyze stress test results and compute statistics.

    Args:
        high_severity_results: Results for high-severity tokens.
        low_severity_results: Results for low-severity tokens.

    Returns:
        Analysis results.
    """
    high_failure_rates = [r["failure_rate"] for r in high_severity_results]
    low_failure_rates = [r["failure_rate"] for r in low_severity_results]

    # Compute statistics
    high_mean = np.mean(high_failure_rates)
    low_mean = np.mean(low_failure_rates)

    # Effect size (Cohen's d)
    pooled_std = np.sqrt((np.var(high_failure_rates) + np.var(low_failure_rates)) / 2)
    cohens_d = (high_mean - low_mean) / pooled_std if pooled_std > 0 else 0

    # Simple t-test
    from scipy import stats
    t_stat, p_value = stats.ttest_ind(high_failure_rates, low_failure_rates)

    return {
        "high_severity": {
            "n_tokens": len(high_severity_results),
            "mean_failure_rate": float(high_mean),
            "std_failure_rate": float(np.std(high_failure_rates)),
        },
        "low_severity": {
            "n_tokens": len(low_severity_results),
            "mean_failure_rate": float(low_mean),
            "std_failure_rate": float(np.std(low_failure_rates)),
        },
        "effect_size_cohens_d": float(cohens_d),
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "significant": p_value < 0.05,
    }


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Run stress test experiment."""
    logger.info("Starting DCTT Stress Test Experiment")
    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    # Get stress test config with defaults
    stress_cfg = OmegaConf.to_container(cfg.get("stress_test", {}))
    n_tokens = stress_cfg.get("n_tokens", 50)
    n_prompts = stress_cfg.get("n_prompts", 5)
    use_mock = stress_cfg.get("use_mock", False)

    # Find latest diagnostic results
    results_dirs = sorted(Path("outputs/runs").glob("*/*/diagnostic_results.json"))
    if not results_dirs:
        logger.error("No diagnostic results found. Run census first.")
        return

    latest_results_path = results_dirs[-1]
    logger.info(f"Loading diagnostic results from: {latest_results_path}")

    results = load_diagnostic_results(latest_results_path)
    logger.info(f"Loaded {len(results)} diagnostic results")

    # Select tokens
    high_severity, low_severity = select_tokens_for_stress_test(
        results, n_high_severity=n_tokens, n_low_severity=n_tokens, seed=cfg.seed
    )

    logger.info(f"Selected {len(high_severity)} high-severity tokens")
    logger.info(f"Selected {len(low_severity)} low-severity tokens")

    # Show severity ranges
    high_severities = [r['severity'] for r in high_severity]
    low_severities = [r['severity'] for r in low_severity]
    logger.info(f"High severity range: [{min(high_severities):.2f}, {max(high_severities):.2f}]")
    logger.info(f"Low severity range: [{min(low_severities):.2f}, {max(low_severities):.2f}]")

    # Load tokenizer to decode token strings
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.name,
        trust_remote_code=cfg.model.tokenizer.trust_remote_code,
    )

    # Decode token strings
    for token_list in [high_severity, low_severity]:
        for token in token_list:
            token_id = token['token_info']['token_id']
            try:
                token['decoded_str'] = tokenizer.decode([token_id])
            except:
                token['decoded_str'] = f"<token_{token_id}>"

    # Show sample tokens
    logger.info("Sample high-severity tokens:")
    for t in high_severity[:5]:
        logger.info(f"  {t['token_info']['token_id']}: '{t['decoded_str'][:20]}' severity={t['severity']:.2f}")

    logger.info("Sample low-severity tokens:")
    for t in low_severity[:5]:
        logger.info(f"  {t['token_info']['token_id']}: '{t['decoded_str'][:20]}' severity={t['severity']:.2f}")

    # Create model function
    if use_mock:
        logger.info("Using mock model (for testing)")
        # Mock model that returns random valid/invalid code
        import random
        random.seed(cfg.seed)
        def model_fn(prompt: str) -> str:
            if random.random() < 0.3:
                return "def foo(x):\n    return x + 1"
            else:
                return "def foo(x:\n    return x"  # Invalid
    else:
        logger.info("Loading model for inference...")
        import torch
        from transformers import AutoModelForCausalLM

        # Try MPS first, fall back to CPU if it fails
        device = cfg.compute.device
        if device == "mps" and torch.backends.mps.is_available():
            try:
                logger.info("Attempting to load model on MPS...")
                model = AutoModelForCausalLM.from_pretrained(
                    cfg.model.name,
                    torch_dtype=torch.float16,  # Use float16 for MPS
                    trust_remote_code=cfg.model.trust_remote_code,
                    low_cpu_mem_usage=True,
                    device_map="mps",
                )
                model.eval()
                logger.info("Model loaded on MPS successfully")
            except Exception as e:
                logger.warning(f"Failed to load on MPS: {e}, falling back to CPU")
                device = "cpu"
                model = AutoModelForCausalLM.from_pretrained(
                    cfg.model.name,
                    torch_dtype=torch.float32,
                    trust_remote_code=cfg.model.trust_remote_code,
                    low_cpu_mem_usage=True,
                )
                model.eval()
        else:
            device = "cpu"
            logger.info(f"Using device: {device}")
            model = AutoModelForCausalLM.from_pretrained(
                cfg.model.name,
                torch_dtype=torch.float32,
                trust_remote_code=cfg.model.trust_remote_code,
                low_cpu_mem_usage=True,
            )
            model.to(device)
            model.eval()

        model_fn = create_model_fn(model, tokenizer, device)

    # Run stress tests
    logger.info(f"Running stress tests with {n_prompts} prompts per token...")

    high_results = []
    for token in tqdm(high_severity, desc="High severity"):
        result = run_syntax_stress_test(
            token['token_info']['token_id'],
            token.get('decoded_str', ''),
            model_fn,
            n_prompts=n_prompts,
        )
        result['severity'] = token['severity']
        high_results.append(result)

    low_results = []
    for token in tqdm(low_severity, desc="Low severity"):
        result = run_syntax_stress_test(
            token['token_info']['token_id'],
            token.get('decoded_str', ''),
            model_fn,
            n_prompts=n_prompts,
        )
        result['severity'] = token['severity']
        low_results.append(result)

    # Analyze results
    logger.info("Analyzing results...")
    analysis = analyze_stress_test_results(high_results, low_results)

    # Log summary
    logger.info("=" * 60)
    logger.info("STRESS TEST RESULTS")
    logger.info("=" * 60)
    logger.info(f"High-severity tokens:")
    logger.info(f"  Mean failure rate: {analysis['high_severity']['mean_failure_rate']:.3f}")
    logger.info(f"  Std failure rate: {analysis['high_severity']['std_failure_rate']:.3f}")
    logger.info(f"Low-severity tokens:")
    logger.info(f"  Mean failure rate: {analysis['low_severity']['mean_failure_rate']:.3f}")
    logger.info(f"  Std failure rate: {analysis['low_severity']['std_failure_rate']:.3f}")
    logger.info(f"Effect size (Cohen's d): {analysis['effect_size_cohens_d']:.3f}")
    logger.info(f"T-statistic: {analysis['t_statistic']:.3f}")
    logger.info(f"P-value: {analysis['p_value']:.6f}")
    logger.info(f"Significant (p < 0.05): {analysis['significant']}")

    # Compute correlation between severity and failure rate
    all_severities = [r['severity'] for r in high_results + low_results]
    all_failure_rates = [r['failure_rate'] for r in high_results + low_results]
    correlation = np.corrcoef(all_severities, all_failure_rates)[0, 1]
    logger.info(f"Correlation (severity vs failure rate): r = {correlation:.3f}")

    analysis['severity_failure_correlation'] = float(correlation)

    # Save results
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert numpy types to Python types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(v) for v in obj]
        return obj

    output = convert_numpy({
        "analysis": analysis,
        "high_severity_results": high_results,
        "low_severity_results": low_results,
        "config": {
            "n_tokens": n_tokens,
            "n_prompts": n_prompts,
            "model": cfg.model.name,
        },
    })

    with open(output_dir / "stress_test_results.json", "w") as f:
        json.dump(output, f, indent=2)

    logger.info(f"Results saved to {output_dir / 'stress_test_results.json'}")


if __name__ == "__main__":
    main()
