#!/usr/bin/env python3
"""Run predictive validity experiment for DCTT.

This experiment evaluates whether geometry metrics predict stress test failures
beyond frequency and token type confounds.

Usage:
    python experiments/run_predictive_validity.py model=qwen2_5_coder_7b
    python experiments/run_predictive_validity.py model=llama3_8b
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

from dctt.core.types import DiagnosticResult, TokenInfo
from dctt.embeddings.extract import EmbeddingExtractor
from dctt.evaluation.predictive import PredictiveValidityAnalyzer, PredictiveValidityResult
from dctt.evaluation.statistics import bootstrap_ci, compute_effect_size
from dctt.metrics.stage1 import compute_stage1_metrics
from dctt.metrics.stage2 import compute_stage2_metrics
from dctt.metrics.severity import SeverityScorer
from dctt.neighbors.usearch_index import USearchIndex
from dctt.stress_tests.code_syntax import CodeSyntaxTest
from dctt.stress_tests.math_format import MathFormatTest
from dctt.stress_tests.runner import StressTestRunner
from dctt.tracking.wandb_utils import WandbTracker

log = logging.getLogger(__name__)


def run_diagnostics(
    cfg: DictConfig,
    embeddings: np.ndarray,
    index: USearchIndex,
    token_infos: list[TokenInfo],
) -> list[DiagnosticResult]:
    """Run diagnostic pipeline on tokens.

    Args:
        cfg: Configuration.
        embeddings: Normalized embeddings.
        index: kNN index.
        token_infos: Token information.

    Returns:
        List of diagnostic results.
    """
    k = cfg.neighbors.k
    results = []

    log.info(f"Running diagnostics on {len(token_infos)} tokens...")

    for i, token_info in enumerate(token_infos):
        if i % 1000 == 0:
            log.info(f"  Processing token {i}/{len(token_infos)}")

        token_id = token_info.token_id
        embedding = embeddings[token_id]

        # Query neighbors
        neighbor_ids, distances = index.query(embedding, k=k + 1)
        # Remove self if present
        mask = neighbor_ids != token_id
        neighbor_ids = neighbor_ids[mask][:k]
        distances = distances[mask][:k]

        # Stage 1 metrics
        stage1 = compute_stage1_metrics(distances)

        # Stage 2 metrics
        neighbor_embeddings = embeddings[neighbor_ids]
        stage2 = compute_stage2_metrics(embedding, neighbor_embeddings)

        results.append(DiagnosticResult(
            token_id=token_id,
            token_info=token_info,
            stage1=stage1,
            stage2=stage2,
            severity=0.0,  # Will be computed later
            bucket=None,
        ))

    # Compute severity scores
    scorer = SeverityScorer()
    results = scorer.score_batch(results)

    return results


def run_stress_tests(
    cfg: DictConfig,
    diagnostic_results: list[DiagnosticResult],
    model: Any,
    tokenizer: Any,
) -> dict[int, float]:
    """Run stress tests to get failure rates.

    Args:
        cfg: Configuration.
        diagnostic_results: Diagnostic results with token info.
        model: Language model.
        tokenizer: Tokenizer.

    Returns:
        Dictionary mapping token_id to failure rate.
    """
    # Initialize stress tests
    tests = [
        CodeSyntaxTest(languages=["python"]),
        MathFormatTest(),
    ]

    runner = StressTestRunner(
        tests=tests,
        model=model,
        tokenizer=tokenizer,
    )

    # Get tokens to test
    token_ids = [r.token_id for r in diagnostic_results]

    log.info(f"Running stress tests on {len(token_ids)} tokens...")
    results = runner.run(token_ids, n_samples=cfg.stress_tests.n_samples)

    # Aggregate failure rates
    failure_rates: dict[int, float] = {}
    for token_id, token_results in results.items():
        failures = sum(1 for r in token_results if not r.passed)
        failure_rates[token_id] = failures / len(token_results) if token_results else 0.0

    return failure_rates


def analyze_predictive_validity(
    diagnostic_results: list[DiagnosticResult],
    stress_test_results: dict[int, float],
) -> PredictiveValidityResult:
    """Analyze predictive validity of geometry metrics.

    Args:
        diagnostic_results: Diagnostic results.
        stress_test_results: Token failure rates.

    Returns:
        Predictive validity analysis results.
    """
    analyzer = PredictiveValidityAnalyzer()
    return analyzer.analyze(diagnostic_results, stress_test_results)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main entry point for predictive validity experiment."""
    log.info("Starting predictive validity experiment")
    log.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    # Initialize tracking
    tracker = WandbTracker(
        project=cfg.tracking.project,
        experiment_name="predictive_validity",
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    try:
        # Extract embeddings
        log.info("Extracting embeddings...")
        extractor = EmbeddingExtractor(cfg.model.name)
        embeddings, tokenizer = extractor.extract()
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        # Build index
        log.info("Building kNN index...")
        index = USearchIndex(
            dim=embeddings.shape[1],
            metric=cfg.neighbors.metric,
            seed=cfg.seed,
        )
        index.build(embeddings)

        # Get token info
        vocab_size = len(tokenizer)
        token_infos = []
        for token_id in range(vocab_size):
            token_str = tokenizer.decode([token_id])
            token_infos.append(TokenInfo(
                token_id=token_id,
                token_string=token_str,
                frequency=1,  # Would load from corpus
                token_type="unknown",
            ))

        # Sample tokens for experiment
        if cfg.experiment.tokens.mode == "sample":
            n_sample = cfg.experiment.tokens.sample_size
            rng = np.random.default_rng(cfg.seed)
            indices = rng.choice(len(token_infos), size=min(n_sample, len(token_infos)), replace=False)
            token_infos = [token_infos[i] for i in indices]

        # Run diagnostics
        diagnostic_results = run_diagnostics(cfg, embeddings, index, token_infos)

        # Load model for stress tests
        log.info("Loading model for stress tests...")
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            cfg.model.name,
            torch_dtype="auto",
            device_map="auto",
        )

        # Run stress tests
        stress_test_results = run_stress_tests(cfg, diagnostic_results, model, tokenizer)

        # Analyze predictive validity
        log.info("Analyzing predictive validity...")
        validity_result = analyze_predictive_validity(diagnostic_results, stress_test_results)

        # Log results
        log.info(f"Results:")
        log.info(f"  Full model AUC: {validity_result.auc:.4f}")
        log.info(f"  PR-AUC: {validity_result.pr_auc:.4f}")
        log.info(f"  Baseline (freq-only) AUC: {validity_result.baseline_auc:.4f}")
        log.info(f"  Improvement: {validity_result.improvement_over_baseline:.4f}")
        log.info(f"  Feature importance: {validity_result.feature_importance}")

        tracker.log({
            "auc": validity_result.auc,
            "pr_auc": validity_result.pr_auc,
            "baseline_auc": validity_result.baseline_auc,
            "improvement_over_baseline": validity_result.improvement_over_baseline,
        })

        # Save results
        output_dir = Path(cfg.output_dir) / "predictive_validity"
        output_dir.mkdir(parents=True, exist_ok=True)

        import json
        results_dict = {
            "auc": validity_result.auc,
            "pr_auc": validity_result.pr_auc,
            "baseline_auc": validity_result.baseline_auc,
            "improvement_over_baseline": validity_result.improvement_over_baseline,
            "feature_importance": validity_result.feature_importance,
            "n_tokens": len(diagnostic_results),
        }

        with open(output_dir / "results.json", "w") as f:
            json.dump(results_dict, f, indent=2)

        log.info(f"Results saved to {output_dir}")

    finally:
        tracker.finish()


if __name__ == "__main__":
    main()
