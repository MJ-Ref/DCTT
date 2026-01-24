#!/usr/bin/env python3
"""Cluster Repair Experiment for DCTT.

This experiment tests the cluster-level repair approach, which addresses
the key finding that single-token repair fails when entire neighborhoods
are pathological.

The approach:
1. Find high-severity tokens
2. Detect clusters using mutual k-NN graph
3. Jointly optimize all tokens in each cluster
4. Measure geometry improvement and semantic preservation

Usage:
    python experiments/run_cluster_repair.py model=qwen2_5_coder_7b
    python experiments/run_cluster_repair.py model=qwen2_5_coder_7b cluster_repair.n_top_tokens=100
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
    """Run cluster repair experiment."""
    logger.info("Starting DCTT Cluster Repair Experiment")
    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    # Get cluster repair config
    cluster_cfg = OmegaConf.to_container(cfg.get("cluster_repair", {}))
    n_top_tokens = cluster_cfg.get("n_top_tokens", 200)
    n_diagnostic_samples = cluster_cfg.get("n_diagnostic_samples", 5000)
    mutual_k = cluster_cfg.get("mutual_k", 20)
    min_cluster_size = cluster_cfg.get("min_cluster_size", 3)

    from dctt.tracking.reproducibility import set_all_seeds
    set_all_seeds(cfg.seed)

    # Load cached embeddings
    logger.info("Loading cached embeddings...")
    from dctt.embeddings.cache import EmbeddingCache

    cache_dir = Path(__file__).parent.parent / "outputs" / "embeddings"
    cache = EmbeddingCache(str(cache_dir))
    cache_key = cache.make_key(cfg.model.name, cfg.model.revision)

    if not cache.has(cache_key):
        logger.error(f"No cached embeddings found for {cfg.model.name}")
        logger.error("Run: python experiments/run_census.py model=qwen2_5_coder_7b first")
        return

    embeddings, metadata = cache.load(cache_key)
    vocab_size, embedding_dim = embeddings.shape
    logger.info(f"Loaded embeddings: {vocab_size} x {embedding_dim}")

    # Load tokenizer
    logger.info("Loading tokenizer...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.name,
        trust_remote_code=cfg.model.tokenizer.trust_remote_code,
    )

    # Build or load kNN index
    logger.info("Building/loading kNN index...")
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

    # Run diagnostics on sample to find high-severity tokens
    logger.info(f"Running diagnostics on {n_diagnostic_samples} tokens...")
    from dctt.metrics.stage1 import compute_stage1_metrics
    from dctt.metrics.stage2 import compute_stage2_metrics

    rng = np.random.default_rng(cfg.seed)
    sample_indices = rng.choice(vocab_size, size=min(n_diagnostic_samples, vocab_size), replace=False)
    sample_indices = np.sort(sample_indices)

    k = cfg.neighbors.k
    severities: dict[int, float] = {}
    token_metrics: dict[int, dict] = {}

    for token_id in tqdm(sample_indices, desc="Computing diagnostics"):
        token_id = int(token_id)
        query_vec = embeddings[token_id].reshape(1, -1)
        neighbors, distances = index.query(query_vec, k=k, exclude_self=True)
        neighbors = neighbors[0]
        distances = distances[0]

        # Stage 1
        stage1 = compute_stage1_metrics(token_id=token_id, distances=distances)

        # Stage 2
        stage2 = compute_stage2_metrics(
            embeddings=embeddings,
            token_id=token_id,
            neighbor_ids=neighbors,
        )

        # Simple severity score
        severity = np.log1p(stage2.cond) + np.log1p(k / (stage2.pr + 1e-10))
        severities[token_id] = float(severity)
        token_metrics[token_id] = {
            "cond": stage2.cond,
            "pr": stage2.pr,
            "logdet": stage2.logdet,
            "spread_q": stage1.spread_q,
        }

    # Get top-N severity tokens
    sorted_tokens = sorted(severities.items(), key=lambda x: -x[1])
    top_tokens = [t[0] for t in sorted_tokens[:n_top_tokens]]

    logger.info(f"\nTop {n_top_tokens} severity tokens:")
    for i, (tid, sev) in enumerate(sorted_tokens[:10]):
        try:
            token_str = repr(tokenizer.decode([tid]))[:30]
        except:
            token_str = f"<token_{tid}>"
        logger.info(f"  {i+1}. Token {tid}: {token_str}, severity={sev:.2f}")

    # Detect clusters
    logger.info(f"\nDetecting clusters (mutual_k={mutual_k}, min_size={min_cluster_size})...")
    from dctt.repair.cluster import find_pathological_clusters

    clustering_result = find_pathological_clusters(
        token_ids=top_tokens,
        severities=severities,
        embeddings=embeddings,
        index=index,
        mutual_k=mutual_k,
        min_cluster_size=min_cluster_size,
    )

    logger.info(f"Found {clustering_result.n_clusters} clusters")
    logger.info(f"Isolated tokens: {clustering_result.n_isolated}")

    for cluster in clustering_result.clusters[:5]:
        logger.info(f"  Cluster {cluster.cluster_id}: {cluster.n_tokens} tokens, "
                   f"mean_sev={cluster.mean_severity:.2f}, connectivity={cluster.internal_connectivity:.2f}")

    if clustering_result.n_clusters == 0:
        logger.warning("No clusters found. Consider adjusting mutual_k or min_cluster_size.")
        return

    # Repair clusters
    logger.info("\nRepairing clusters...")
    from dctt.repair.cluster_optimizer import ClusterRepairOptimizer
    from dctt.core.types import RepairConfig

    repair_config = RepairConfig(
        max_outer_iters=cfg.get("cluster_repair", {}).get("max_outer_iters", 3),
        max_inner_steps=cfg.get("cluster_repair", {}).get("max_inner_steps", 50),
        learning_rate=cfg.get("cluster_repair", {}).get("learning_rate", 0.05),
        lambda_anchor=cfg.get("cluster_repair", {}).get("lambda_anchor", 0.5),
        lambda_nn_preserve=cfg.get("cluster_repair", {}).get("lambda_nn_preserve", 0.1),
        delta_max=cfg.get("cluster_repair", {}).get("delta_max", 0.15),
        geometry_loss="cond",
    )

    optimizer = ClusterRepairOptimizer(repair_config)
    repair_results = []

    n_clusters_to_repair = min(5, clustering_result.n_clusters)  # Limit for speed

    for cluster in tqdm(clustering_result.clusters[:n_clusters_to_repair], desc="Repairing clusters"):
        result = optimizer.repair_cluster(
            cluster=cluster,
            embeddings=embeddings,
            index=index,
            k=k,
        )
        repair_results.append(result)

        logger.info(f"  Cluster {result.cluster_id} ({len(result.token_ids)} tokens):")
        logger.info(f"    cond: {result.geometry_before['cond']:.2f} -> {result.geometry_after['cond']:.2f}")
        logger.info(f"    pr:   {result.geometry_before['pr']:.2f} -> {result.geometry_after['pr']:.2f}")
        logger.info(f"    improved: {result.geometry_improved}, jaccard: {result.mean_jaccard:.3f}")

    # Analyze results
    logger.info("\n" + "=" * 60)
    logger.info("CLUSTER REPAIR RESULTS")
    logger.info("=" * 60)

    cond_improvements = []
    pr_improvements = []
    jaccard_scores = []
    similarity_scores = []
    improved_count = 0

    for r in repair_results:
        cond_imp = r.geometry_before["cond"] - r.geometry_after["cond"]
        pr_imp = r.geometry_after["pr"] - r.geometry_before["pr"]
        cond_improvements.append(cond_imp)
        pr_improvements.append(pr_imp)
        jaccard_scores.append(r.mean_jaccard)
        similarity_scores.append(r.mean_similarity)
        if r.geometry_improved:
            improved_count += 1

    logger.info("\nGeometry Improvement (positive = better):")
    logger.info(f"  Condition number reduction: {np.mean(cond_improvements):.4f} +/- {np.std(cond_improvements):.4f}")
    logger.info(f"  Participation ratio increase: {np.mean(pr_improvements):.4f} +/- {np.std(pr_improvements):.4f}")
    logger.info(f"  Clusters improved: {improved_count}/{len(repair_results)}")

    logger.info("\nSemantic Preservation:")
    logger.info(f"  Neighbor Jaccard overlap: {np.mean(jaccard_scores):.4f} +/- {np.std(jaccard_scores):.4f}")
    logger.info(f"  Similarity to original: {np.mean(similarity_scores):.4f} +/- {np.std(similarity_scores):.4f}")

    # Validation criteria
    mean_jaccard = np.mean(jaccard_scores)
    mean_cond_imp = np.mean(cond_improvements)
    mean_pr_imp = np.mean(pr_improvements)

    logger.info("\n" + "-" * 60)
    logger.info("VALIDATION CRITERIA:")
    logger.info(f"  Jaccard > 0.5: {'PASS' if mean_jaccard > 0.5 else 'FAIL'} ({mean_jaccard:.3f})")
    logger.info(f"  Cond decreases: {'PASS' if mean_cond_imp > 0 else 'FAIL'} ({mean_cond_imp:.3f})")
    logger.info(f"  PR increases: {'PASS' if mean_pr_imp > 0 else 'FAIL'} ({mean_pr_imp:.3f})")

    cluster_repair_works = mean_cond_imp > 0 or mean_pr_imp > 0

    if cluster_repair_works:
        logger.info("\n*** CLUSTER REPAIR SHOWS GEOMETRY IMPROVEMENT! ***")
    else:
        logger.info("\nCluster repair did not improve geometry. Consider:")
        logger.info("  - Larger clusters (lower min_cluster_size)")
        logger.info("  - More iterations")
        logger.info("  - Relocation to healthy regions instead")

    # Save results
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output = convert_numpy({
        "summary": {
            "n_clusters_found": clustering_result.n_clusters,
            "n_clusters_repaired": len(repair_results),
            "n_isolated_tokens": clustering_result.n_isolated,
            "mean_cond_improvement": float(np.mean(cond_improvements)),
            "mean_pr_improvement": float(np.mean(pr_improvements)),
            "mean_jaccard": float(np.mean(jaccard_scores)),
            "mean_similarity": float(np.mean(similarity_scores)),
            "clusters_improved": improved_count,
            "cluster_repair_works": cluster_repair_works,
        },
        "clusters": [
            {
                "cluster_id": c.cluster_id,
                "n_tokens": c.n_tokens,
                "mean_severity": c.mean_severity,
                "internal_connectivity": c.internal_connectivity,
            }
            for c in clustering_result.clusters[:n_clusters_to_repair]
        ],
        "repair_results": [
            {
                "cluster_id": r.cluster_id,
                "n_tokens": len(r.token_ids),
                "geometry_before": r.geometry_before,
                "geometry_after": r.geometry_after,
                "geometry_improved": r.geometry_improved,
                "mean_jaccard": r.mean_jaccard,
                "mean_similarity": r.mean_similarity,
                "converged": r.converged,
                "iterations": r.iterations,
            }
            for r in repair_results
        ],
        "config": {
            "n_top_tokens": n_top_tokens,
            "mutual_k": mutual_k,
            "min_cluster_size": min_cluster_size,
            "model": cfg.model.name,
        },
    })

    output_path = output_dir / "cluster_repair_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    logger.info(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
