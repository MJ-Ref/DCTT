#!/usr/bin/env python3
"""Run predictive validity experiment for DCTT.

This experiment evaluates whether geometry metrics predict stress test failures
beyond frequency and token type confounds.

Produces:
1. Model comparison: baseline vs geometry vs full
2. Feature importance and ablation analysis
3. Within-bucket analysis (geometry predicts within same confound strata)
4. Bootstrap confidence intervals

Usage:
    python experiments/run_predictive_validity.py model=qwen2_5_coder_7b
    python experiments/run_predictive_validity.py model=qwen2_5_coder_7b experiment.tokens.mode=sample experiment.tokens.sample_size=1000
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from dctt.core.types import DiagnosticResult, TokenInfo, TokenType, FrequencyTier
from dctt.evaluation.predictive import (
    PredictiveValidityAnalyzer,
    PredictiveValidityResult,
    format_validity_report,
)
from dctt.metrics.stage1 import compute_stage1_metrics
from dctt.metrics.stage2 import compute_stage2_metrics
from dctt.metrics.severity import SeverityScorer
from dctt.neighbors.usearch_index import USearchIndex

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def classify_token_type(token_str: str) -> TokenType:
    """Classify token into type category."""
    if token_str.strip() == "":
        return TokenType.WHITESPACE
    elif token_str.isalpha():
        return TokenType.FULL_WORD
    elif token_str.isdigit():
        return TokenType.NUMERIC
    elif token_str in "{}[]()<>":
        return TokenType.CODE_SYMBOL
    elif all(c in ".,;:!?'\"-" for c in token_str.strip()):
        return TokenType.PUNCTUATION
    elif token_str.startswith("▁") or token_str.startswith("Ġ"):
        # BPE subword markers
        return TokenType.SUBWORD
    else:
        return TokenType.SUBWORD


def run_diagnostics(
    cfg: DictConfig,
    embeddings: np.ndarray,
    index: USearchIndex,
    token_infos: list[TokenInfo],
) -> list[DiagnosticResult]:
    """Run diagnostic pipeline on tokens."""
    k = cfg.neighbors.k
    results = []

    logger.info(f"Running diagnostics on {len(token_infos)} tokens...")

    for token_info in tqdm(token_infos, desc="Computing diagnostics"):
        token_id = token_info.token_id

        # Query neighbors
        query_vec = embeddings[token_id].reshape(1, -1)
        neighbor_ids, distances = index.query(query_vec, k=k, exclude_self=True)
        neighbor_ids = neighbor_ids[0]
        distances = distances[0]

        # Stage 1 metrics
        stage1 = compute_stage1_metrics(
            distances=distances,
            token_id=token_id,
        )

        # Stage 2 metrics
        stage2 = compute_stage2_metrics(
            embeddings=embeddings,
            token_id=token_id,
            neighbor_ids=neighbor_ids,
        )

        # Create diagnostic result
        result = DiagnosticResult(
            token_info=token_info,
            stage1=stage1,
            stage2=stage2,
            severity=0.0,
            bucket=(token_info.frequency_tier, token_info.token_type),
        )
        results.append(result)

    # Compute severity scores
    logger.info("Computing severity scores...")
    scorer = SeverityScorer()
    scorer.fit(results)
    for result in results:
        result.severity = scorer.compute_severity(result)

    return results


def simulate_stress_test_failures(
    diagnostic_results: list[DiagnosticResult],
    seed: int = 42,
) -> dict[int, float]:
    """Simulate stress test failures based on geometry metrics.

    For validation purposes, we create synthetic failures that have
    a known relationship with geometry metrics plus noise.
    This allows us to verify the analysis pipeline works correctly.

    In production, replace with actual stress test results.
    """
    rng = np.random.default_rng(seed)
    failures = {}

    for result in diagnostic_results:
        # Create failure probability based on geometry
        # High cond, low pr, high spread_q -> higher failure rate
        geometry_score = (
            0.3 * np.log1p(result.stage2.cond) / 5 +  # Normalize cond contribution
            0.3 * (1 - result.stage2.pr / 50) +  # Low PR -> higher failure
            0.2 * min(result.stage1.spread_q / 10, 1) +  # High spread -> failure
            0.2 * result.severity / 10  # Severity contribution
        )

        # Clip to [0, 1] and add noise
        base_prob = np.clip(geometry_score, 0, 1)
        noise = rng.normal(0, 0.15)
        failure_rate = np.clip(base_prob + noise, 0, 1)

        failures[result.token_id] = failure_rate

    return failures


def run_stress_tests(
    cfg: DictConfig,
    diagnostic_results: list[DiagnosticResult],
    model: Any,
    tokenizer: Any,
) -> dict[int, float]:
    """Run actual stress tests to get failure rates."""
    from dctt.stress_tests.code_syntax import CodeSyntaxTest
    from dctt.stress_tests.math_format import MathFormatTest
    from dctt.stress_tests.runner import StressTestRunner

    tests = [
        CodeSyntaxTest(languages=["python"]),
        MathFormatTest(),
    ]

    runner = StressTestRunner(
        tests=tests,
        model=model,
        tokenizer=tokenizer,
    )

    token_ids = [r.token_id for r in diagnostic_results]
    n_samples = cfg.get("stress_tests", {}).get("n_samples", 10)

    logger.info(f"Running stress tests on {len(token_ids)} tokens...")
    results = runner.run(token_ids, n_samples=n_samples)

    failure_rates: dict[int, float] = {}
    for token_id, token_results in results.items():
        failures = sum(1 for r in token_results if not r.passed)
        failure_rates[token_id] = failures / len(token_results) if token_results else 0.0

    return failure_rates


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main entry point for predictive validity experiment."""
    logger.info("Starting predictive validity experiment")
    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    # Set seeds
    from dctt.tracking.reproducibility import set_all_seeds
    set_all_seeds(cfg.seed)

    # Initialize W&B if enabled
    run = None
    if cfg.wandb.enabled:
        from dctt.tracking.wandb_utils import init_wandb_from_hydra
        run = init_wandb_from_hydra(cfg, tags=["predictive_validity"])

    try:
        # Load cached embeddings
        from dctt.embeddings import extract_embeddings
        from dctt.embeddings.normalize import normalize_embeddings
        from dctt.embeddings.cache import EmbeddingCache

        cache_dir = Path(__file__).parent.parent / "outputs" / "embeddings"
        cache = EmbeddingCache(str(cache_dir))
        cache_key = cache.make_key(cfg.model.name, cfg.model.revision)

        if cache.has(cache_key):
            logger.info("Loading embeddings from cache")
            embeddings, metadata = cache.load(cache_key)
        else:
            logger.info("Extracting embeddings from model")
            embeddings_raw, tokenizer = extract_embeddings(
                model_name=cfg.model.name,
                revision=cfg.model.revision,
                device=cfg.compute.device,
                torch_dtype=cfg.model.torch_dtype,
            )
            embeddings, norms = normalize_embeddings(embeddings_raw, return_norms=True)
            from dctt.embeddings import get_embedding_info
            info = get_embedding_info(embeddings, cfg.model.name, cfg.model.revision)
            cache.save(cache_key, embeddings, info)

        vocab_size, embedding_dim = embeddings.shape
        logger.info(f"Embeddings: {vocab_size} x {embedding_dim}")

        # Load tokenizer
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            cfg.model.name,
            trust_remote_code=cfg.model.tokenizer.trust_remote_code,
        )

        # Build or load index
        index_cache_dir = Path(__file__).parent.parent / "outputs" / "indices"
        index_cache_path = index_cache_dir / f"{cache_key}_{cfg.neighbors.metric}.usearch"

        index = USearchIndex(
            connectivity=cfg.compute.index.connectivity,
            expansion_add=cfg.compute.index.expansion_add,
            expansion_search=cfg.compute.index.expansion_search,
        )

        if index_cache_path.exists():
            logger.info(f"Loading index from {index_cache_path}")
            index.load(str(index_cache_path))
        else:
            logger.info("Building kNN index...")
            index.build(embeddings, metric=cfg.neighbors.metric, seed=cfg.seed)
            index_cache_dir.mkdir(parents=True, exist_ok=True)
            index.save(str(index_cache_path))

        # Build token info with proper classification
        logger.info("Building token info...")
        token_infos = []
        for token_id in range(vocab_size):
            try:
                token_str = tokenizer.decode([token_id])
            except:
                token_str = f"<token_{token_id}>"

            token_type = classify_token_type(token_str)

            # Assign frequency tier based on token_id (approximation)
            # Lower IDs tend to be more frequent in BPE
            if token_id < vocab_size * 0.2:
                freq_tier = FrequencyTier.HIGH
            elif token_id < vocab_size * 0.8:
                freq_tier = FrequencyTier.MID
            else:
                freq_tier = FrequencyTier.LOW

            token_infos.append(TokenInfo(
                token_id=token_id,
                token_str=token_str,
                token_type=token_type,
                frequency=1.0 / (token_id + 1),  # Approximate frequency
                frequency_tier=freq_tier,
                norm=1.0,
            ))

        # Sample tokens for experiment
        experiment_cfg = cfg.get("experiment", {})
        tokens_cfg = experiment_cfg.get("tokens", {})
        if tokens_cfg.get("mode") == "sample":
            n_sample = tokens_cfg.get("sample_size", 1000)
            rng = np.random.default_rng(cfg.seed)
            indices = rng.choice(len(token_infos), size=min(n_sample, len(token_infos)), replace=False)
            token_infos = [token_infos[i] for i in indices]
            logger.info(f"Sampled {len(token_infos)} tokens")

        # Run diagnostics
        diagnostic_results = run_diagnostics(cfg, embeddings, index, token_infos)

        # Get stress test results
        use_simulated = cfg.get("predictive_validity", {}).get("use_simulated_failures", True)

        if use_simulated:
            logger.info("Using simulated stress test failures for validation...")
            stress_test_results = simulate_stress_test_failures(diagnostic_results, seed=cfg.seed)
        else:
            # Load model and run actual stress tests
            logger.info("Loading model for stress tests...")
            from transformers import AutoModelForCausalLM
            import torch

            device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
            model = AutoModelForCausalLM.from_pretrained(
                cfg.model.name,
                torch_dtype=torch.float16 if device != "cpu" else torch.float32,
                device_map=device,
                trust_remote_code=True,
            )
            stress_test_results = run_stress_tests(cfg, diagnostic_results, model, tokenizer)

        # Run comprehensive predictive validity analysis
        logger.info("Running predictive validity analysis...")
        analyzer = PredictiveValidityAnalyzer(
            n_bootstrap=cfg.get("predictive_validity", {}).get("n_bootstrap", 100),
            random_state=cfg.seed,
            failure_threshold=cfg.get("predictive_validity", {}).get("failure_threshold", 0.3),
        )
        validity_result = analyzer.analyze(diagnostic_results, stress_test_results)

        # Print formatted report
        report = format_validity_report(validity_result)
        logger.info("\n" + report)

        # Log to W&B
        if run is not None:
            import wandb
            wandb.log({
                "baseline_auc": validity_result.baseline_model.auc,
                "geometry_auc": validity_result.geometry_model.auc,
                "full_auc": validity_result.full_model.auc,
                "improvement_over_baseline": validity_result.improvement_over_baseline,
                "geometry_adds_value": validity_result.geometry_adds_value,
                "baseline_pr_auc": validity_result.baseline_model.pr_auc,
                "geometry_pr_auc": validity_result.geometry_model.pr_auc,
                "full_pr_auc": validity_result.full_model.pr_auc,
                "buckets_where_geometry_predicts": validity_result.buckets_where_geometry_predicts,
                "total_buckets": validity_result.total_buckets,
            })

        # Save results
        output_dir = Path(cfg.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        def convert_numpy(obj):
            """Convert numpy types to Python types for JSON serialization."""
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

        # Convert to serializable format
        results_dict = {
            "summary": {
                "n_tokens": int(validity_result.n_tokens),
                "n_positive": int(validity_result.n_positive),
                "positive_rate": float(validity_result.positive_rate),
                "improvement_over_baseline": float(validity_result.improvement_over_baseline),
                "geometry_adds_value": bool(validity_result.geometry_adds_value),
            },
            "model_comparison": {
                "baseline": {
                    "auc": validity_result.baseline_model.auc,
                    "auc_ci": [validity_result.baseline_model.auc_ci_low, validity_result.baseline_model.auc_ci_high],
                    "pr_auc": validity_result.baseline_model.pr_auc,
                    "features": validity_result.baseline_model.features,
                },
                "geometry": {
                    "auc": validity_result.geometry_model.auc,
                    "auc_ci": [validity_result.geometry_model.auc_ci_low, validity_result.geometry_model.auc_ci_high],
                    "pr_auc": validity_result.geometry_model.pr_auc,
                    "features": validity_result.geometry_model.features,
                },
                "full": {
                    "auc": validity_result.full_model.auc,
                    "auc_ci": [validity_result.full_model.auc_ci_low, validity_result.full_model.auc_ci_high],
                    "pr_auc": validity_result.full_model.pr_auc,
                    "features": validity_result.full_model.features,
                },
            },
            "feature_importance": validity_result.feature_importance,
            "feature_ablations": [
                {
                    "feature": a.feature_removed,
                    "auc_without": a.auc_without,
                    "auc_drop": a.auc_drop,
                    "rank": a.importance_rank,
                }
                for a in validity_result.feature_ablations
            ],
            "bucket_analysis": {
                "buckets_where_geometry_predicts": validity_result.buckets_where_geometry_predicts,
                "total_buckets": validity_result.total_buckets,
                "details": [
                    {
                        "bucket": f"{b.bucket[0].name}+{b.bucket[1].name}",
                        "n_samples": b.n_samples,
                        "auc": b.auc,
                        "pr_auc": b.pr_auc,
                        "geometry_predicts": b.geometry_predicts,
                    }
                    for b in validity_result.bucket_analyses
                ],
            },
            "config": {
                "model": cfg.model.name,
                "n_bootstrap": cfg.get("predictive_validity", {}).get("n_bootstrap", 100),
                "use_simulated_failures": use_simulated,
            },
        }

        with open(output_dir / "predictive_validity_results.json", "w") as f:
            json.dump(convert_numpy(results_dict), f, indent=2)

        # Save report
        with open(output_dir / "predictive_validity_report.txt", "w") as f:
            f.write(report)

        logger.info(f"Results saved to {output_dir}")

    finally:
        if run is not None:
            from dctt.tracking.wandb_utils import finish_run
            finish_run()


if __name__ == "__main__":
    main()
