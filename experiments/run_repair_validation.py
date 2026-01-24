#!/usr/bin/env python3
"""Repair Validation Experiment for DCTT.

This script validates the repair system by:
1. Loading cached embeddings from a real model
2. Running diagnostic census to find high-severity tokens
3. Repairing top-N severity tokens
4. Validating geometry improvement and semantic preservation

Usage:
    python experiments/run_repair_validation.py model=qwen2_5_coder_7b
    python experiments/run_repair_validation.py model=qwen2_5_coder_7b repair.n_tokens=20
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def convert_numpy(obj: Any) -> Any:
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


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Run repair validation experiment."""
    logger.info("Starting DCTT Repair Validation Experiment")
    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    # Get repair config with defaults
    repair_cfg = OmegaConf.to_container(cfg.get("repair_validation", {}))
    n_tokens_to_repair = repair_cfg.get("n_tokens", 10)
    n_diagnostic_samples = repair_cfg.get("n_diagnostic_samples", 5000)

    from dctt.tracking.reproducibility import set_all_seeds
    set_all_seeds(cfg.seed)

    # Load cached embeddings
    logger.info("Loading cached embeddings...")
    cache_dir = Path(__file__).parent.parent / "outputs" / "embeddings"

    from dctt.embeddings.cache import EmbeddingCache
    cache = EmbeddingCache(str(cache_dir))
    cache_key = cache.make_key(cfg.model.name, cfg.model.revision)

    if not cache.has(cache_key):
        logger.error(f"No cached embeddings found for {cfg.model.name}")
        logger.error("Run: python experiments/run_census.py model=qwen2_5_coder_7b first")
        return

    embeddings, metadata = cache.load(cache_key)
    vocab_size, embedding_dim = embeddings.shape
    logger.info(f"Loaded embeddings: {vocab_size} x {embedding_dim}")

    # Load tokenizer for token strings
    logger.info("Loading tokenizer...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.name,
        trust_remote_code=cfg.model.tokenizer.trust_remote_code,
    )

    # Build kNN index
    logger.info("Building kNN index...")
    index_cache_dir = Path(__file__).parent.parent / "outputs" / "indices"
    index_cache_path = index_cache_dir / f"{cache_key}_{cfg.neighbors.metric}.usearch"

    from dctt.neighbors.usearch_index import USearchIndex
    index = USearchIndex(
        connectivity=cfg.compute.index.connectivity,
        expansion_add=cfg.compute.index.expansion_add,
        expansion_search=cfg.compute.index.expansion_search,
    )

    if index_cache_path.exists():
        logger.info(f"Loading cached index from {index_cache_path}")
        index.load(str(index_cache_path))
    else:
        logger.info("Building index from scratch...")
        index.build(embeddings, metric=cfg.neighbors.metric, seed=cfg.seed)
        index_cache_dir.mkdir(parents=True, exist_ok=True)
        index.save(str(index_cache_path))
        logger.info(f"Index saved to {index_cache_path}")

    # Sample tokens for diagnostic (faster than full census)
    logger.info(f"Running diagnostic on {n_diagnostic_samples} sampled tokens...")
    rng = np.random.default_rng(cfg.seed)
    sample_indices = rng.choice(vocab_size, size=min(n_diagnostic_samples, vocab_size), replace=False)
    sample_indices = np.sort(sample_indices)

    # Import metric functions
    from dctt.metrics.stage1 import compute_stage1_metrics
    from dctt.metrics.stage2 import compute_stage2_metrics
    from dctt.core.types import TokenInfo, TokenType, FrequencyTier

    # Run diagnostics
    diagnostic_results = []
    k = cfg.neighbors.k

    for token_id in tqdm(sample_indices, desc="Computing diagnostics"):
        token_id = int(token_id)
        query_vec = embeddings[token_id].reshape(1, -1)
        neighbors, distances = index.query(query_vec, k=k, exclude_self=True)
        neighbors = neighbors[0]
        distances = distances[0]

        # Compute Stage 1
        stage1 = compute_stage1_metrics(token_id, distances)

        # Compute Stage 2
        stage2 = compute_stage2_metrics(
            embeddings=embeddings,
            token_id=token_id,
            neighbor_ids=neighbors,
            return_eigenvalues=True,
        )

        # Get token string
        try:
            token_str = tokenizer.decode([token_id])
        except:
            token_str = f"<token_{token_id}>"

        # Compute simple severity (combination of cond and PR)
        # Higher cond = worse, lower PR = worse
        severity = np.log1p(stage2.cond) + np.log1p(k / (stage2.pr + 1e-10))

        diagnostic_results.append({
            "token_id": token_id,
            "token_str": token_str,
            "stage1": {
                "mu_k": stage1.mu_k,
                "med_k": stage1.med_k,
                "spread_q": stage1.spread_q,
                "fail": stage1.fail,
            },
            "stage2": {
                "dim95": stage2.dim95,
                "pr": stage2.pr,
                "cond": stage2.cond,
                "logdet": stage2.logdet,
                "anisotropy": stage2.anisotropy,
                "fail": stage2.fail,
            },
            "severity": float(severity),
        })

    # Sort by severity and select top tokens for repair
    diagnostic_results.sort(key=lambda x: x["severity"], reverse=True)
    top_tokens = diagnostic_results[:n_tokens_to_repair]

    logger.info(f"\nTop {n_tokens_to_repair} severity tokens:")
    for i, t in enumerate(top_tokens):
        token_repr = repr(t["token_str"])[:30]
        logger.info(f"  {i+1}. Token {t['token_id']}: {token_repr}")
        logger.info(f"     severity={t['severity']:.2f}, cond={t['stage2']['cond']:.2f}, pr={t['stage2']['pr']:.2f}")

    # Run repairs
    logger.info(f"\nRepairing {n_tokens_to_repair} tokens...")
    from dctt.repair.optimizer import EmbeddingRepairOptimizer
    from dctt.core.types import RepairConfig

    repair_config = RepairConfig(
        max_outer_iters=3,
        max_inner_steps=100,
        learning_rate=0.1,  # Increased from 0.01
        lambda_anchor=0.1,  # Reduced to allow more geometry optimization
        lambda_nn_preserve=0.1,  # Reduced
        delta_max=0.2,  # Increased to allow more movement
        geometry_loss="cond",
    )

    optimizer = EmbeddingRepairOptimizer(repair_config)
    repair_results = []

    for token_data in tqdm(top_tokens, desc="Repairing"):
        token_id = token_data["token_id"]
        embedding = embeddings[token_id]

        # Get neighbors
        query_vec = embedding.reshape(1, -1)
        neighbors, _ = index.query(query_vec, k=k, exclude_self=True)

        # Repair
        result = optimizer.repair(
            embedding=embedding,
            neighbors=neighbors[0],
            all_embeddings=embeddings,
            index=index,
            k=k,
        )
        result.token_id = token_id

        repair_results.append({
            "token_id": token_id,
            "token_str": token_data["token_str"],
            "severity_before": token_data["severity"],
            "geometry_before": result.geometry_before,
            "geometry_after": result.geometry_after,
            "semantic_validation": result.semantic_validation,
            "delta_norm": result.delta_norm,
            "converged": result.converged,
            "iterations": result.iterations,
            "final_loss": result.final_loss,
        })

    # Analyze results
    logger.info("\n" + "=" * 60)
    logger.info("REPAIR VALIDATION RESULTS")
    logger.info("=" * 60)

    cond_improvements = []
    pr_improvements = []
    logdet_improvements = []
    jaccard_scores = []
    similarity_scores = []

    for r in repair_results:
        before = r["geometry_before"]
        after = r["geometry_after"]

        cond_imp = before["cond"] - after["cond"]
        pr_imp = after["pr"] - before["pr"]
        logdet_imp = after["logdet"] - before["logdet"]

        cond_improvements.append(cond_imp)
        pr_improvements.append(pr_imp)
        logdet_improvements.append(logdet_imp)
        jaccard_scores.append(r["semantic_validation"]["neighbor_jaccard"])
        similarity_scores.append(r["semantic_validation"]["similarity_to_original"])

    logger.info("\nGeometry Improvement (positive = better):")
    logger.info(f"  Condition number reduction: {np.mean(cond_improvements):.4f} +/- {np.std(cond_improvements):.4f}")
    logger.info(f"  Participation ratio increase: {np.mean(pr_improvements):.4f} +/- {np.std(pr_improvements):.4f}")
    logger.info(f"  Log-determinant increase: {np.mean(logdet_improvements):.4f} +/- {np.std(logdet_improvements):.4f}")

    logger.info("\nSemantic Preservation:")
    logger.info(f"  Neighbor Jaccard overlap: {np.mean(jaccard_scores):.4f} +/- {np.std(jaccard_scores):.4f}")
    logger.info(f"  Similarity to original: {np.mean(similarity_scores):.4f} +/- {np.std(similarity_scores):.4f}")

    # Check validation criteria
    mean_jaccard = np.mean(jaccard_scores)
    mean_similarity = np.mean(similarity_scores)
    mean_cond_imp = np.mean(cond_improvements)
    mean_pr_imp = np.mean(pr_improvements)

    # Check if embeddings actually moved
    embeddings_moved = mean_similarity < 0.9999

    logger.info("\n" + "-" * 60)
    logger.info("VALIDATION CRITERIA:")
    logger.info(f"  Embeddings moved: {'YES' if embeddings_moved else 'NO'} (similarity={mean_similarity:.4f})")
    logger.info(f"  Jaccard > 0.7: {'PASS' if mean_jaccard > 0.7 else 'FAIL'} ({mean_jaccard:.3f})")
    logger.info(f"  Cond decreases: {'PASS' if mean_cond_imp > 0 else 'FAIL'} ({mean_cond_imp:.3f})")
    logger.info(f"  PR increases: {'PASS' if mean_pr_imp > 0 else 'FAIL'} ({mean_pr_imp:.3f})")

    if embeddings_moved and mean_cond_imp <= 0:
        logger.info("\n" + "-" * 60)
        logger.info("NOTE: Embeddings moved but geometry didn't improve.")
        logger.info("This suggests the entire local region has poor geometry.")
        logger.info("More sophisticated approaches (e.g., moving toward well-conditioned")
        logger.info("tokens in embedding space) may be needed for effective repair.")

    # Per-token details
    logger.info("\nPer-token results:")
    for r in repair_results:
        token_repr = repr(r["token_str"])[:20]
        before = r["geometry_before"]
        after = r["geometry_after"]
        logger.info(f"  Token {r['token_id']} ({token_repr}):")
        logger.info(f"    cond: {before['cond']:.2f} -> {after['cond']:.2f} ({before['cond'] - after['cond']:+.2f})")
        logger.info(f"    pr:   {before['pr']:.2f} -> {after['pr']:.2f} ({after['pr'] - before['pr']:+.2f})")
        logger.info(f"    jaccard: {r['semantic_validation']['neighbor_jaccard']:.3f}")

    # Save results
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output = convert_numpy({
        "summary": {
            "n_tokens_repaired": len(repair_results),
            "mean_cond_improvement": float(np.mean(cond_improvements)),
            "mean_pr_improvement": float(np.mean(pr_improvements)),
            "mean_logdet_improvement": float(np.mean(logdet_improvements)),
            "mean_jaccard": float(np.mean(jaccard_scores)),
            "mean_similarity": float(np.mean(similarity_scores)),
            "validation_passed": {
                "jaccard_gt_0.7": mean_jaccard > 0.7,
                "cond_decreases": mean_cond_imp > 0,
                "pr_increases": mean_pr_imp > 0,
            },
        },
        "repair_results": repair_results,
        "config": {
            "n_tokens": n_tokens_to_repair,
            "n_diagnostic_samples": n_diagnostic_samples,
            "model": cfg.model.name,
            "repair_config": {
                "max_outer_iters": repair_config.max_outer_iters,
                "max_inner_steps": repair_config.max_inner_steps,
                "learning_rate": repair_config.learning_rate,
                "lambda_anchor": repair_config.lambda_anchor,
                "lambda_nn_preserve": repair_config.lambda_nn_preserve,
                "delta_max": repair_config.delta_max,
                "geometry_loss": repair_config.geometry_loss,
            },
        },
    })

    output_path = output_dir / "repair_validation_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    logger.info(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
