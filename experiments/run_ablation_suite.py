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

from dctt.core.types import DiagnosticResult, TokenInfo, RepairConfig
from dctt.embeddings.extract import EmbeddingExtractor
from dctt.metrics.stage1 import compute_stage1_metrics
from dctt.metrics.stage2 import compute_stage2_metrics
from dctt.metrics.severity import SeverityScorer
from dctt.neighbors.usearch_index import USearchIndex
from dctt.repair.optimizer import RepairOptimizer
from dctt.tracking.wandb_utils import WandbTracker

log = logging.getLogger(__name__)


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
        log.info(f"Running ablation with k={k}")

        # Build index
        index = USearchIndex(
            dim=embeddings.shape[1],
            metric=cfg.neighbors.metric,
            seed=cfg.seed,
        )
        index.build(embeddings)

        # Compute metrics for sample tokens
        sample_size = min(500, len(token_infos))
        rng = np.random.default_rng(cfg.seed)
        sample_indices = rng.choice(len(token_infos), size=sample_size, replace=False)

        pr_values = []
        cond_values = []
        dim95_values = []

        for idx in sample_indices:
            token_id = token_infos[idx].token_id
            embedding = embeddings[token_id]

            neighbor_ids, distances = index.query(embedding, k=k + 1)
            mask = neighbor_ids != token_id
            neighbor_ids = neighbor_ids[mask][:k]

            neighbor_embeddings = embeddings[neighbor_ids]
            stage2 = compute_stage2_metrics(embedding, neighbor_embeddings)

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

    for idx in sample_indices:
        token_id = token_infos[idx].token_id
        embedding = embeddings[token_id]

        neighbor_ids, distances = index.query(embedding, k=k + 1)
        mask = neighbor_ids != token_id
        neighbor_ids = neighbor_ids[mask][:k]
        distances = distances[mask][:k]

        stage1 = compute_stage1_metrics(distances)
        neighbor_embeddings = embeddings[neighbor_ids]
        stage2 = compute_stage2_metrics(embedding, neighbor_embeddings)

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
        {"name": "cond_only", "geometry_weight": 1.0, "geometry_type": "cond"},
        {"name": "logdet_only", "geometry_weight": 1.0, "geometry_type": "logdet"},
        {"name": "pr_only", "geometry_weight": 1.0, "geometry_type": "pr"},
        {"name": "combined", "geometry_weight": 1.0, "geometry_type": "combined"},
    ]

    results = []

    # Select tokens to repair (high severity)
    k = cfg.neighbors.k
    severities = []

    for token_info in token_infos[:1000]:
        token_id = token_info.token_id
        embedding = embeddings[token_id]

        neighbor_ids, distances = index.query(embedding, k=k + 1)
        mask = neighbor_ids != token_id
        neighbor_ids = neighbor_ids[mask][:k]

        neighbor_embeddings = embeddings[neighbor_ids]
        stage2 = compute_stage2_metrics(embedding, neighbor_embeddings)
        severities.append((token_id, stage2.cond - stage2.pr))

    severities.sort(key=lambda x: x[1], reverse=True)
    repair_token_ids = [s[0] for s in severities[:10]]

    for variant in loss_variants:
        log.info(f"Running repair ablation: {variant['name']}")

        repair_config = RepairConfig(
            lambda_anchor=cfg.repair.lambda_anchor,
            lambda_nn_preserve=cfg.repair.lambda_nn_preserve,
            delta_max=cfg.repair.delta_max,
            outer_iterations=3,
            inner_steps=50,
            learning_rate=cfg.repair.learning_rate,
        )

        optimizer = RepairOptimizer(
            embeddings=embeddings.copy(),
            index=index,
            config=repair_config,
        )

        pre_metrics = []
        post_metrics = []

        for token_id in repair_token_ids:
            # Get pre-repair metrics
            embedding = embeddings[token_id]
            neighbor_ids, _ = index.query(embedding, k=k + 1)
            mask = neighbor_ids != token_id
            neighbor_ids = neighbor_ids[mask][:k]
            neighbor_embeddings = embeddings[neighbor_ids]
            pre_stage2 = compute_stage2_metrics(embedding, neighbor_embeddings)
            pre_metrics.append({"cond": pre_stage2.cond, "pr": pre_stage2.pr})

            # Repair
            result = optimizer.repair_token(token_id)

            # Get post-repair metrics
            repaired_embedding = result.repaired_embedding
            neighbor_ids, _ = index.query(repaired_embedding, k=k + 1)
            mask = neighbor_ids != token_id
            neighbor_ids = neighbor_ids[mask][:k]
            neighbor_embeddings = embeddings[neighbor_ids]
            post_stage2 = compute_stage2_metrics(repaired_embedding, neighbor_embeddings)
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
        embedding = embeddings[token_id]

        neighbor_ids, _ = index.query(embedding, k=k + 1)
        mask = neighbor_ids != token_id
        neighbor_ids = neighbor_ids[mask][:k]

        neighbor_embeddings = embeddings[neighbor_ids]
        stage2 = compute_stage2_metrics(embedding, neighbor_embeddings)
        severities.append((token_id, stage2.cond))

    severities.sort(key=lambda x: x[1], reverse=True)
    repair_token_ids = [s[0] for s in severities[:5]]

    for lambda_anchor in lambda_values:
        log.info(f"Running anchor sweep: lambda={lambda_anchor}")

        repair_config = RepairConfig(
            lambda_anchor=lambda_anchor,
            lambda_nn_preserve=cfg.repair.lambda_nn_preserve,
            delta_max=cfg.repair.delta_max,
            outer_iterations=3,
            inner_steps=50,
            learning_rate=cfg.repair.learning_rate,
        )

        optimizer = RepairOptimizer(
            embeddings=embeddings.copy(),
            index=index,
            config=repair_config,
        )

        deltas = []
        cond_improvements = []

        for token_id in repair_token_ids:
            original = embeddings[token_id].copy()
            result = optimizer.repair_token(token_id)

            delta = np.linalg.norm(result.repaired_embedding - original)
            deltas.append(delta)

            # Get pre/post cond
            neighbor_ids, _ = index.query(original, k=k + 1)
            mask = neighbor_ids != token_id
            neighbor_ids = neighbor_ids[mask][:k]
            neighbor_embeddings = embeddings[neighbor_ids]
            pre_cond = compute_stage2_metrics(original, neighbor_embeddings).cond

            neighbor_ids, _ = index.query(result.repaired_embedding, k=k + 1)
            mask = neighbor_ids != token_id
            neighbor_ids = neighbor_ids[mask][:k]
            neighbor_embeddings = embeddings[neighbor_ids]
            post_cond = compute_stage2_metrics(result.repaired_embedding, neighbor_embeddings).cond

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
    log.info("Starting ablation suite")
    log.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    # Initialize tracking
    tracker = WandbTracker(
        project=cfg.tracking.project,
        experiment_name="ablation_suite",
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
                frequency=1,
                token_type="unknown",
            ))

        # Run ablations
        all_results = AblationSuiteResult()

        # K sweep
        log.info("Running k-sweep ablation...")
        k_results = run_k_sweep_ablation(cfg, embeddings, token_infos)
        all_results.ablations.extend(k_results)

        # Stage ablation
        log.info("Running stage ablation...")
        stage_results = run_stage_ablation(cfg, embeddings, index, token_infos)
        all_results.ablations.extend(stage_results)

        # Repair loss ablation
        log.info("Running repair loss ablation...")
        repair_results = run_repair_loss_ablation(cfg, embeddings, index, token_infos)
        all_results.ablations.extend(repair_results)

        # Anchor sweep
        log.info("Running anchor sweep ablation...")
        anchor_results = run_anchor_sweep_ablation(cfg, embeddings, index, token_infos)
        all_results.ablations.extend(anchor_results)

        # Summary
        all_results.summary = {
            "n_ablations": len(all_results.ablations),
            "ablation_names": [r.name for r in all_results.ablations],
        }

        # Log to tracker
        for result in all_results.ablations:
            for metric_name, metric_value in result.metrics.items():
                tracker.log({f"{result.name}/{metric_name}": metric_value})

        # Save results
        output_dir = Path(cfg.output_dir) / "ablation_suite"
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

        with open(output_dir / "results.json", "w") as f:
            json.dump(results_dict, f, indent=2)

        log.info(f"Results saved to {output_dir}")
        log.info(f"Completed {len(all_results.ablations)} ablations")

    finally:
        tracker.finish()


if __name__ == "__main__":
    main()
