"""Diagnostic Census Experiment.

This script runs a comprehensive diagnostic census on model embeddings,
computing Stage 1 and Stage 2 metrics for all tokens in the vocabulary.

Usage:
    python experiments/run_census.py model=qwen2_5_coder_7b
    python experiments/run_census.py model=llama3_8b experiment.tokens.mode=sample experiment.tokens.sample_size=1000
"""

from __future__ import annotations

import logging
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Run diagnostic census experiment."""
    logger.info("Starting DCTT Census Experiment")
    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    # Set seeds
    from dctt.tracking.reproducibility import set_all_seeds
    set_all_seeds(cfg.seed)

    # Initialize W&B if enabled
    run = None
    if cfg.wandb.enabled:
        from dctt.tracking.wandb_utils import init_wandb_from_hydra
        run = init_wandb_from_hydra(cfg, tags=["census"])

    try:
        # Extract embeddings
        logger.info(f"Loading model: {cfg.model.name}")
        from dctt.embeddings import extract_embeddings, get_embedding_info
        from dctt.embeddings.normalize import normalize_embeddings
        from dctt.embeddings.cache import EmbeddingCache

        cache = EmbeddingCache(cfg.output_dir + "/embeddings")
        cache_key = cache.make_key(cfg.model.name, cfg.model.revision)

        if cache.has(cache_key):
            logger.info("Loading embeddings from cache")
            embeddings, _ = cache.load(cache_key)
        else:
            logger.info("Extracting embeddings from model")
            embeddings_raw, tokenizer = extract_embeddings(
                model_name=cfg.model.name,
                revision=cfg.model.revision,
                device=cfg.compute.device,
                torch_dtype=cfg.model.torch_dtype,
            )
            embeddings, norms = normalize_embeddings(embeddings_raw, return_norms=True)

            # Cache embeddings
            info = get_embedding_info(embeddings, cfg.model.name, cfg.model.revision)
            cache.save(cache_key, embeddings, info)

        vocab_size, embedding_dim = embeddings.shape
        logger.info(f"Embeddings shape: {vocab_size} x {embedding_dim}")

        # Build kNN index
        logger.info("Building kNN index")
        from dctt.neighbors.usearch_index import USearchIndex

        index = USearchIndex(
            connectivity=cfg.compute.index.connectivity,
            expansion_add=cfg.compute.index.expansion_add,
            expansion_search=cfg.compute.index.expansion_search,
        )
        index.build(embeddings, metric=cfg.neighbors.metric, seed=cfg.seed)

        # Determine which tokens to analyze
        experiment_cfg = cfg.experiment
        if experiment_cfg.tokens.mode == "all":
            token_ids = list(range(vocab_size))
        elif experiment_cfg.tokens.mode == "sample":
            sample_size = experiment_cfg.tokens.sample_size or 1000
            rng = np.random.default_rng(cfg.seed)
            token_ids = rng.choice(vocab_size, size=min(sample_size, vocab_size), replace=False).tolist()
        else:
            token_ids = experiment_cfg.tokens.subset_ids or list(range(1000))

        logger.info(f"Analyzing {len(token_ids)} tokens")

        # Compute metrics
        from dctt.metrics.stage1 import compute_stage1_metrics
        from dctt.metrics.stage2 import compute_stage2_metrics
        from dctt.core.types import DiagnosticResult, TokenInfo, TokenType, FrequencyTier

        k = cfg.neighbors.k
        results = []

        for token_id in tqdm(token_ids, desc="Computing metrics"):
            # Query neighbors
            query_vec = embeddings[token_id].reshape(1, -1)
            neighbors, distances = index.query(query_vec, k=k, exclude_self=True)
            neighbors = neighbors[0]
            distances = distances[0]

            # Stage 1 metrics
            s1_result = compute_stage1_metrics(
                distances=distances,
                token_id=token_id,
            )

            # Stage 2 metrics
            s2_result = compute_stage2_metrics(
                embeddings=embeddings,
                token_id=token_id,
                neighbor_ids=neighbors,
                return_eigenvalues=experiment_cfg.outputs.save_eigenvalues,
            )

            # Create token info (simplified - would need tokenizer for full info)
            token_info = TokenInfo(
                token_id=token_id,
                token_str=f"token_{token_id}",  # Would use tokenizer.decode([token_id])
                token_type=TokenType.UNKNOWN,
                frequency=1.0,  # Would load from corpus stats
                frequency_tier=FrequencyTier.MID,
                norm=1.0,  # Already normalized
            )

            # Create diagnostic result
            result = DiagnosticResult(
                token_info=token_info,
                stage1=s1_result,
                stage2=s2_result,
                severity=0.0,  # Computed after fitting severity scorer
                consistency=0.0,  # Computed via consistency estimator
            )
            results.append(result)

        logger.info(f"Computed metrics for {len(results)} tokens")

        # Compute severity scores
        logger.info("Computing severity scores")
        from dctt.metrics.severity import SeverityScorer
        from dctt.metrics.thresholding import AdaptiveThresholder, ThresholdConfig

        # Assign buckets (simplified)
        for result in results:
            result.bucket = (result.token_info.frequency_tier, result.token_info.token_type)

        # Fit severity scorer
        scorer = SeverityScorer()
        scorer.fit(results)

        for result in results:
            result.severity = scorer.compute_severity(result)

        # Fit thresholder and apply
        thresholder = AdaptiveThresholder(ThresholdConfig())
        thresholder.fit(results)
        results = thresholder.apply_thresholds(results)

        # Count failures
        n_failing = sum(1 for r in results if r.fails_any_stage)
        logger.info(f"Tokens failing diagnostics: {n_failing} / {len(results)}")

        # Save results
        output_dir = Path(cfg.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        from dctt.tracking.artifacts import save_results_json
        save_results_json(
            results,
            output_dir / "diagnostic_results.json",
            include_eigenvalues=experiment_cfg.outputs.save_eigenvalues,
        )

        # Log to W&B
        if run is not None:
            from dctt.tracking.wandb_utils import log_diagnostic_summary
            log_diagnostic_summary(results)

            # Log artifact
            from dctt.tracking.artifacts import log_diagnostic_results_artifact
            from dctt.tracking.reproducibility import create_run_manifest

            manifest = create_run_manifest(
                cfg,
                cfg.model.name,
                cfg.model.revision,
                embeddings,
                "unknown",
                "usearch",
                index.config_hash,
                cfg.seed,
            )
            log_diagnostic_results_artifact(
                run,
                output_dir / "diagnostic_results.json",
                manifest,
                summary={"n_tokens": len(results), "n_failing": n_failing},
            )

        logger.info(f"Results saved to {output_dir}")

    finally:
        if run is not None:
            from dctt.tracking.wandb_utils import finish_run
            finish_run()


if __name__ == "__main__":
    main()
