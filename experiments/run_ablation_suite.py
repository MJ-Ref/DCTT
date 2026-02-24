#!/usr/bin/env python3
"""Run ablation suite for DCTT.

This experiment runs a comprehensive ablation study covering:
1. Stage 1 only vs Stage 1+2 vs Stage 1+2+3
2. Different neighborhood sizes k
3. Distance metric: cosine vs L2
4. Severity definition variants
5. Repair objective variants
6. Anchor strength sweep
7. Candidate selection methods

Usage:
    python experiments/run_ablation_suite.py model=qwen2_5_coder_7b
    python experiments/run_ablation_suite.py model=qwen2_5_coder_7b ablation=k_sweep
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from dctt.core.types import DiagnosticResult, TokenInfo, TokenType, FrequencyTier, RepairConfig
from dctt.metrics.stage1 import compute_stage1_metrics
from dctt.metrics.stage2 import compute_stage2_metrics
from dctt.neighbors.usearch_index import USearchIndex
from dctt.repair.optimizer import EmbeddingRepairOptimizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AblationResult:
    """Results from a single ablation run."""

    name: str
    config: dict[str, Any]
    metrics: dict[str, float]
    n_tokens: int
    n_repaired: int = 0


@dataclass
class AblationSuiteResult:
    """Results from full ablation suite."""

    ablations: list[AblationResult] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)


def run_k_sweep_ablation(
    cfg: DictConfig,
    embeddings: np.ndarray,
    token_infos: list[TokenInfo],
) -> list[AblationResult]:
    """Ablation over neighborhood size k.

    Args:
        cfg: Configuration.
        embeddings: Normalized embeddings.
        token_infos: Token information.

    Returns:
        List of ablation results.
    """
    k_values = [25, 50, 75, 100, 150, 200]
    results = []

    for k in k_values:
        logger.info(f"Running ablation with k={k}")

        # Build index
        index = USearchIndex(
            connectivity=cfg.compute.index.connectivity,
            expansion_add=cfg.compute.index.expansion_add,
            expansion_search=cfg.compute.index.expansion_search,
        )
        index.build(embeddings, metric=cfg.neighbors.metric, seed=cfg.seed)

        # Compute metrics for sample tokens
        sample_size = min(500, len(token_infos))
        rng = np.random.default_rng(cfg.seed)
        sample_indices = rng.choice(len(token_infos), size=sample_size, replace=False)

        pr_values = []
        cond_values = []
        dim95_values = []

        for idx in tqdm(sample_indices, desc=f"k={k}"):
            token_id = token_infos[idx].token_id

            # Query neighbors (2D input required)
            query_vec = embeddings[token_id].reshape(1, -1)
            neighbor_ids, distances = index.query(query_vec, k=k, exclude_self=True)
            neighbor_ids = neighbor_ids[0]

            # Stage 2 metrics
            stage2 = compute_stage2_metrics(
                embeddings=embeddings,
                token_id=token_id,
                neighbor_ids=neighbor_ids,
            )

            pr_values.append(stage2.pr)
            cond_values.append(stage2.cond)
            dim95_values.append(stage2.dim95)

        results.append(AblationResult(
            name=f"k_{k}",
            config={"k": k},
            metrics={
                "mean_pr": float(np.mean(pr_values)),
                "std_pr": float(np.std(pr_values)),
                "mean_cond": float(np.mean(cond_values)),
                "std_cond": float(np.std(cond_values)),
                "mean_dim95": float(np.mean(dim95_values)),
                "std_dim95": float(np.std(dim95_values)),
            },
            n_tokens=sample_size,
        ))

    return results


def run_stage_ablation(
    cfg: DictConfig,
    embeddings: np.ndarray,
    index: USearchIndex,
    token_infos: list[TokenInfo],
) -> list[AblationResult]:
    """Ablation over diagnostic stages.

    Args:
        cfg: Configuration.
        embeddings: Normalized embeddings.
        index: kNN index.
        token_infos: Token information.

    Returns:
        List of ablation results.
    """
    results = []
    k = cfg.neighbors.k

    sample_size = min(500, len(token_infos))
    rng = np.random.default_rng(cfg.seed)
    sample_indices = rng.choice(len(token_infos), size=sample_size, replace=False)

    # Stage 1 only
    stage1_severities = []
    stage2_severities = []

    for idx in tqdm(sample_indices, desc="Stage ablation"):
        token_id = token_infos[idx].token_id

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

        # Stage 1 severity (based on spread_q and mean distance)
        stage1_sev = stage1.spread_q + stage1.mu_k
        stage1_severities.append(stage1_sev)

        # Stage 1+2 severity
        stage2_sev = stage2.cond - stage2.pr + stage1.spread_q
        stage2_severities.append(stage2_sev)

    results.append(AblationResult(
        name="stage1_only",
        config={"stages": [1]},
        metrics={
            "mean_severity": float(np.mean(stage1_severities)),
            "std_severity": float(np.std(stage1_severities)),
            "severity_range": float(np.max(stage1_severities) - np.min(stage1_severities)),
        },
        n_tokens=sample_size,
    ))

    results.append(AblationResult(
        name="stage1_stage2",
        config={"stages": [1, 2]},
        metrics={
            "mean_severity": float(np.mean(stage2_severities)),
            "std_severity": float(np.std(stage2_severities)),
            "severity_range": float(np.max(stage2_severities) - np.min(stage2_severities)),
        },
        n_tokens=sample_size,
    ))

    return results


def run_repair_loss_ablation(
    cfg: DictConfig,
    embeddings: np.ndarray,
    index: USearchIndex,
    token_infos: list[TokenInfo],
) -> list[AblationResult]:
    """Ablation over repair loss functions.

    Args:
        cfg: Configuration.
        embeddings: Normalized embeddings.
        index: kNN index.
        token_infos: Token information.

    Returns:
        List of ablation results.
    """
    loss_variants = [
        {"name": "cond_only", "geometry_loss": "cond"},
        {"name": "logdet_only", "geometry_loss": "logdet"},
        {"name": "pr_only", "geometry_loss": "pr"},
        {"name": "combined", "geometry_loss": "combined"},
    ]

    results = []
    k = cfg.neighbors.k

    # Select tokens to repair (high severity)
    logger.info("Selecting high-severity tokens for repair ablation...")
    severities = []

    for token_info in tqdm(token_infos[:1000], desc="Computing severity"):
        token_id = token_info.token_id

        query_vec = embeddings[token_id].reshape(1, -1)
        neighbor_ids, _ = index.query(query_vec, k=k, exclude_self=True)
        neighbor_ids = neighbor_ids[0]

        stage2 = compute_stage2_metrics(
            embeddings=embeddings,
            token_id=token_id,
            neighbor_ids=neighbor_ids,
        )
        severities.append((token_id, stage2.cond - stage2.pr))

    severities.sort(key=lambda x: x[1], reverse=True)
    repair_token_ids = [s[0] for s in severities[:10]]

    for variant in loss_variants:
        logger.info(f"Running repair ablation: {variant['name']}")

        repair_config = RepairConfig(
            max_outer_iters=3,
            max_inner_steps=50,
            learning_rate=cfg.repair.get("learning_rate", 0.1),
            lambda_anchor=cfg.repair.get("lambda_anchor", 0.1),
            lambda_nn_preserve=cfg.repair.get("lambda_nn_preserve", 0.1),
            delta_max=cfg.repair.get("delta_max", 0.2),
            geometry_loss=variant["geometry_loss"],
        )

        optimizer = EmbeddingRepairOptimizer(repair_config)

        pre_metrics = []
        post_metrics = []

        for token_id in tqdm(repair_token_ids, desc=f"Repair {variant['name']}"):
            embedding = embeddings[token_id]

            # Get neighbors
            neighbors, _ = index.query_single(
                embedding,
                k=k,
                exclude_self=True,
                self_index=token_id,
            )

            # Get pre-repair metrics
            pre_stage2 = compute_stage2_metrics(
                embeddings=embeddings,
                token_id=token_id,
                neighbor_ids=neighbors,
            )
            pre_metrics.append({"cond": pre_stage2.cond, "pr": pre_stage2.pr})

            # Repair
            result = optimizer.repair(
                embedding=embedding,
                neighbors=neighbors,
                all_embeddings=embeddings,
                index=index,
                k=k,
                token_id=token_id,
            )

            # Get post-repair metrics
            repaired_embedding = result.repaired_embedding
            new_neighbors, _ = index.query_single(
                repaired_embedding,
                k=k,
                exclude_self=True,
                self_index=token_id,
            )
            repaired_embeddings = embeddings.copy()
            repaired_embeddings[token_id] = repaired_embedding
            post_stage2 = compute_stage2_metrics(
                embeddings=repaired_embeddings,
                token_id=token_id,
                neighbor_ids=new_neighbors,
            )
            post_metrics.append({"cond": post_stage2.cond, "pr": post_stage2.pr})

        # Compute improvements
        cond_improvements = [
            pre["cond"] - post["cond"]
            for pre, post in zip(pre_metrics, post_metrics)
        ]
        pr_improvements = [
            post["pr"] - pre["pr"]
            for pre, post in zip(pre_metrics, post_metrics)
        ]

        results.append(AblationResult(
            name=variant["name"],
            config=variant,
            metrics={
                "mean_cond_improvement": float(np.mean(cond_improvements)),
                "mean_pr_improvement": float(np.mean(pr_improvements)),
                "pct_cond_improved": float(np.mean([c > 0 for c in cond_improvements])),
                "pct_pr_improved": float(np.mean([p > 0 for p in pr_improvements])),
            },
            n_tokens=len(repair_token_ids),
            n_repaired=len(repair_token_ids),
        ))

    return results


def run_anchor_sweep_ablation(
    cfg: DictConfig,
    embeddings: np.ndarray,
    index: USearchIndex,
    token_infos: list[TokenInfo],
) -> list[AblationResult]:
    """Ablation over anchor strength lambda.

    Args:
        cfg: Configuration.
        embeddings: Normalized embeddings.
        index: kNN index.
        token_infos: Token information.

    Returns:
        List of ablation results.
    """
    lambda_values = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0]
    results = []
    k = cfg.neighbors.k

    # Select tokens to repair
    severities = []
    for token_info in token_infos[:500]:
        token_id = token_info.token_id

        query_vec = embeddings[token_id].reshape(1, -1)
        neighbor_ids, _ = index.query(query_vec, k=k, exclude_self=True)
        neighbor_ids = neighbor_ids[0]

        stage2 = compute_stage2_metrics(
            embeddings=embeddings,
            token_id=token_id,
            neighbor_ids=neighbor_ids,
        )
        severities.append((token_id, stage2.cond))

    severities.sort(key=lambda x: x[1], reverse=True)
    repair_token_ids = [s[0] for s in severities[:5]]

    for lambda_anchor in lambda_values:
        logger.info(f"Running anchor sweep: lambda={lambda_anchor}")

        repair_config = RepairConfig(
            max_outer_iters=3,
            max_inner_steps=50,
            learning_rate=cfg.repair.get("learning_rate", 0.1),
            lambda_anchor=lambda_anchor,
            lambda_nn_preserve=cfg.repair.get("lambda_nn_preserve", 0.1),
            delta_max=cfg.repair.get("delta_max", 0.2),
            geometry_loss="cond",
        )

        optimizer = EmbeddingRepairOptimizer(repair_config)

        deltas = []
        cond_improvements = []

        for token_id in repair_token_ids:
            original = embeddings[token_id].copy()

            # Get neighbors
            neighbors, _ = index.query_single(
                original,
                k=k,
                exclude_self=True,
                self_index=token_id,
            )

            # Get pre-repair cond
            pre_stage2 = compute_stage2_metrics(
                embeddings=embeddings,
                token_id=token_id,
                neighbor_ids=neighbors,
            )
            pre_cond = pre_stage2.cond

            # Repair
            result = optimizer.repair(
                embedding=original,
                neighbors=neighbors,
                all_embeddings=embeddings,
                index=index,
                k=k,
                token_id=token_id,
            )

            delta = np.linalg.norm(result.repaired_embedding - original)
            deltas.append(delta)

            # Get post-repair cond
            new_neighbors, _ = index.query_single(
                result.repaired_embedding,
                k=k,
                exclude_self=True,
                self_index=token_id,
            )
            repaired_embeddings = embeddings.copy()
            repaired_embeddings[token_id] = result.repaired_embedding
            post_stage2 = compute_stage2_metrics(
                embeddings=repaired_embeddings,
                token_id=token_id,
                neighbor_ids=new_neighbors,
            )
            post_cond = post_stage2.cond

            cond_improvements.append(pre_cond - post_cond)

        results.append(AblationResult(
            name=f"lambda_{lambda_anchor}",
            config={"lambda_anchor": lambda_anchor},
            metrics={
                "mean_delta": float(np.mean(deltas)),
                "max_delta": float(np.max(deltas)),
                "mean_cond_improvement": float(np.mean(cond_improvements)),
            },
            n_tokens=len(repair_token_ids),
            n_repaired=len(repair_token_ids),
        ))

    return results


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main entry point for ablation suite."""
    logger.info("Starting ablation suite")
    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    # Set seeds
    from dctt.tracking.reproducibility import set_all_seeds
    set_all_seeds(cfg.seed)

    # Initialize W&B if enabled
    run = None
    if cfg.wandb.enabled:
        from dctt.tracking.wandb_utils import init_wandb_from_hydra
        run = init_wandb_from_hydra(cfg, tags=["ablation_suite"])

    try:
        # Load cached embeddings or extract
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
            embeddings, _ = normalize_embeddings(embeddings_raw, return_norms=True)
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

        # Get token info with real strings
        logger.info("Building token info...")
        token_infos = []
        for token_id in range(vocab_size):
            try:
                token_str = tokenizer.decode([token_id])
            except:
                token_str = f"<token_{token_id}>"

            # Simple token type classification
            if token_str.strip() == "":
                token_type = TokenType.WHITESPACE
            elif token_str.isalpha():
                token_type = TokenType.FULL_WORD
            elif token_str.isdigit():
                token_type = TokenType.NUMERIC
            elif token_str in "{}[]()<>":
                token_type = TokenType.CODE_SYMBOL
            elif not token_str.isalnum() and len(token_str.strip()) <= 2:
                token_type = TokenType.PUNCTUATION
            else:
                token_type = TokenType.SUBWORD

            token_infos.append(TokenInfo(
                token_id=token_id,
                token_str=token_str,
                token_type=token_type,
                frequency=1.0,
                frequency_tier=FrequencyTier.MID,
                norm=1.0,
            ))

        # Run ablations
        all_results = AblationSuiteResult()

        # K sweep
        logger.info("Running k-sweep ablation...")
        k_results = run_k_sweep_ablation(cfg, embeddings, token_infos)
        all_results.ablations.extend(k_results)

        # Stage ablation
        logger.info("Running stage ablation...")
        stage_results = run_stage_ablation(cfg, embeddings, index, token_infos)
        all_results.ablations.extend(stage_results)

        # Repair loss ablation
        logger.info("Running repair loss ablation...")
        repair_results = run_repair_loss_ablation(cfg, embeddings, index, token_infos)
        all_results.ablations.extend(repair_results)

        # Anchor sweep
        logger.info("Running anchor sweep ablation...")
        anchor_results = run_anchor_sweep_ablation(cfg, embeddings, index, token_infos)
        all_results.ablations.extend(anchor_results)

        # Summary
        all_results.summary = {
            "n_ablations": len(all_results.ablations),
            "ablation_names": [r.name for r in all_results.ablations],
        }

        # Log to tracker
        if run is not None:
            import wandb
            for result in all_results.ablations:
                for metric_name, metric_value in result.metrics.items():
                    wandb.log({f"{result.name}/{metric_name}": metric_value})

        # Save results
        output_dir = Path(cfg.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results_dict = {
            "ablations": [
                {
                    "name": r.name,
                    "config": r.config,
                    "metrics": r.metrics,
                    "n_tokens": r.n_tokens,
                    "n_repaired": r.n_repaired,
                }
                for r in all_results.ablations
            ],
            "summary": all_results.summary,
        }

        with open(output_dir / "ablation_results.json", "w") as f:
            json.dump(results_dict, f, indent=2)

        logger.info(f"Results saved to {output_dir}")
        logger.info(f"Completed {len(all_results.ablations)} ablations")

    finally:
        if run is not None:
            from dctt.tracking.wandb_utils import finish_run
            finish_run()


if __name__ == "__main__":
    main()
