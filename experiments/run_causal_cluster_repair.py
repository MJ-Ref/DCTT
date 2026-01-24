#!/usr/bin/env python3
"""Causal Cluster Repair Experiment for DCTT.

This experiment tests whether cluster-level repair causally improves
token behavior by:
1. Selecting high-severity token clusters for treatment
2. Creating matched controls from low-severity tokens
3. Applying cluster repair to treatment, placebo to controls
4. Evaluating outcomes (geometry + simulated stress tests)
5. Computing causal effect with bootstrap CIs

Usage:
    python experiments/run_causal_cluster_repair.py model=qwen2_5_coder_7b
    python experiments/run_causal_cluster_repair.py model=qwen2_5_coder_7b causal.n_treatment_clusters=10
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import hydra
import numpy as np
from numpy.typing import NDArray
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CausalConfig:
    """Configuration for causal experiment."""

    n_treatment_clusters: int = 10  # Number of clusters to repair
    n_diagnostic_samples: int = 5000  # Sample for finding high-severity tokens
    n_top_tokens: int = 500  # Top severity tokens for clustering
    mutual_k: int = 50  # k for mutual k-NN graph
    min_cluster_size: int = 2  # Minimum cluster size
    n_bootstrap: int = 1000  # Bootstrap samples for CIs
    confidence_level: float = 0.95  # CI level


@dataclass
class TokenOutcome:
    """Outcome for a single token."""

    token_id: int
    group: str  # "treatment" or "control"
    geometry_before: dict[str, float] = field(default_factory=dict)
    geometry_after: dict[str, float] = field(default_factory=dict)
    failure_rate_before: float = 0.0
    failure_rate_after: float = 0.0
    severity: float = 0.0


@dataclass
class CausalResult:
    """Result of causal analysis."""

    # Sample sizes
    n_treatment: int
    n_control: int

    # Mean outcomes
    treatment_failure_before: float
    treatment_failure_after: float
    control_failure_before: float
    control_failure_after: float

    # Causal effects
    ate: float  # Average Treatment Effect
    ate_ci_low: float
    ate_ci_high: float

    # Difference-in-differences
    did: float
    did_ci_low: float
    did_ci_high: float

    # Geometry changes
    treatment_cond_change: float
    control_cond_change: float

    # Statistical significance
    significant: bool
    p_value: float


def simulate_failure_rate(
    cond: float,
    pr: float,
    logdet: float,
    severity: float,
    noise_std: float = 0.1,
    rng: np.random.Generator | None = None,
) -> float:
    """Simulate failure rate based on geometry metrics.

    For validation, we create synthetic failures that have known
    relationship with geometry. In production, replace with actual
    stress test results.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Higher cond, lower pr -> higher failure rate
    geometry_score = (
        0.3 * np.log1p(cond) / 5 +  # Normalize cond contribution
        0.3 * (1 - pr / 50) +  # Low PR -> higher failure
        0.2 * max(0, -logdet / 500) +  # Low logdet -> higher failure
        0.2 * severity / 10  # Severity contribution
    )

    base_prob = np.clip(geometry_score, 0, 1)
    noise = rng.normal(0, noise_std)
    return float(np.clip(base_prob + noise, 0, 1))


def apply_placebo_repair(
    embeddings: NDArray[np.float64],
    token_ids: list[int],
    delta_max: float = 0.15,
    rng: np.random.Generator | None = None,
) -> NDArray[np.float64]:
    """Apply placebo repair (random perturbation, same budget).

    This ensures controls go through same optimization process
    but without geometry-targeted changes.
    """
    if rng is None:
        rng = np.random.default_rng()

    repaired = embeddings[token_ids].copy()

    for i in range(len(token_ids)):
        # Random perturbation in tangent space
        random_dir = rng.normal(0, 1, repaired[i].shape)
        # Project to tangent space
        random_dir = random_dir - np.dot(random_dir, repaired[i]) * repaired[i]
        random_dir = random_dir / (np.linalg.norm(random_dir) + 1e-10)

        # Apply small random step
        step_size = rng.uniform(0, delta_max)
        repaired[i] = repaired[i] + step_size * random_dir

        # Re-normalize
        repaired[i] = repaired[i] / np.linalg.norm(repaired[i])

    return repaired


def compute_bootstrap_ci(
    treatment_before: NDArray[np.float64],
    treatment_after: NDArray[np.float64],
    control_before: NDArray[np.float64],
    control_after: NDArray[np.float64],
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, float, float, float, float]:
    """Compute bootstrap CIs for ATE and DiD.

    Returns:
        (ate, ate_ci_low, ate_ci_high, did, did_ci_low, did_ci_high)
    """
    if rng is None:
        rng = np.random.default_rng()

    n_treatment = len(treatment_before)
    n_control = len(control_before)

    ate_samples = []
    did_samples = []

    for _ in range(n_bootstrap):
        # Bootstrap sample treatment
        t_idx = rng.choice(n_treatment, size=n_treatment, replace=True)
        t_before = treatment_before[t_idx].mean()
        t_after = treatment_after[t_idx].mean()

        # Bootstrap sample control
        c_idx = rng.choice(n_control, size=n_control, replace=True)
        c_before = control_before[c_idx].mean()
        c_after = control_after[c_idx].mean()

        # ATE: difference in post-treatment outcomes
        ate = t_after - c_after
        ate_samples.append(ate)

        # DiD: (treatment change) - (control change)
        did = (t_after - t_before) - (c_after - c_before)
        did_samples.append(did)

    ate_samples = np.array(ate_samples)
    did_samples = np.array(did_samples)

    alpha = 1 - confidence_level
    ate_ci = np.percentile(ate_samples, [alpha/2 * 100, (1 - alpha/2) * 100])
    did_ci = np.percentile(did_samples, [alpha/2 * 100, (1 - alpha/2) * 100])

    return (
        float(ate_samples.mean()),
        float(ate_ci[0]),
        float(ate_ci[1]),
        float(did_samples.mean()),
        float(did_ci[0]),
        float(did_ci[1]),
    )


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
    """Run causal cluster repair experiment."""
    logger.info("Starting DCTT Causal Cluster Repair Experiment")
    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    # Get causal config
    causal_cfg = CausalConfig(
        n_treatment_clusters=cfg.get("causal", {}).get("n_treatment_clusters", 10),
        n_diagnostic_samples=cfg.get("cluster_repair", {}).get("n_diagnostic_samples", 5000),
        n_top_tokens=cfg.get("cluster_repair", {}).get("n_top_tokens", 500),
        mutual_k=cfg.get("cluster_repair", {}).get("mutual_k", 50),
        min_cluster_size=cfg.get("cluster_repair", {}).get("min_cluster_size", 2),
        n_bootstrap=cfg.get("causal", {}).get("n_bootstrap", 1000),
        confidence_level=cfg.get("causal", {}).get("confidence_level", 0.95),
    )

    from dctt.tracking.reproducibility import set_all_seeds
    set_all_seeds(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

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
    logger.info(f"Running diagnostics on {causal_cfg.n_diagnostic_samples} tokens...")
    from dctt.metrics.stage1 import compute_stage1_metrics
    from dctt.metrics.stage2 import compute_stage2_metrics

    sample_indices = rng.choice(
        vocab_size, size=min(causal_cfg.n_diagnostic_samples, vocab_size), replace=False
    )
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

    # Get top-N severity tokens for clustering
    sorted_tokens = sorted(severities.items(), key=lambda x: -x[1])
    top_tokens = [t[0] for t in sorted_tokens[:causal_cfg.n_top_tokens]]

    # Get low-severity tokens for control pool
    low_severity_tokens = [t[0] for t in sorted_tokens[-causal_cfg.n_top_tokens:]]

    logger.info(f"Top {causal_cfg.n_top_tokens} severity tokens selected")
    logger.info(f"Low severity control pool: {len(low_severity_tokens)} tokens")

    # Detect clusters
    logger.info(f"Detecting clusters (mutual_k={causal_cfg.mutual_k}, min_size={causal_cfg.min_cluster_size})...")
    from dctt.repair.cluster import find_pathological_clusters

    clustering_result = find_pathological_clusters(
        token_ids=top_tokens,
        severities=severities,
        embeddings=embeddings,
        index=index,
        mutual_k=causal_cfg.mutual_k,
        min_cluster_size=causal_cfg.min_cluster_size,
    )

    logger.info(f"Found {clustering_result.n_clusters} clusters")
    logger.info(f"Isolated tokens: {clustering_result.n_isolated}")

    if clustering_result.n_clusters == 0:
        logger.warning("No clusters found. Consider adjusting parameters.")
        return

    # Select treatment clusters
    n_treatment = min(causal_cfg.n_treatment_clusters, clustering_result.n_clusters)
    treatment_clusters = clustering_result.clusters[:n_treatment]
    treatment_token_ids = []
    for cluster in treatment_clusters:
        treatment_token_ids.extend(cluster.token_ids)

    logger.info(f"Treatment group: {n_treatment} clusters, {len(treatment_token_ids)} tokens")

    # Create matched control group (same size, low severity)
    control_token_ids = low_severity_tokens[:len(treatment_token_ids)]
    logger.info(f"Control group: {len(control_token_ids)} tokens")

    # Compute pre-repair outcomes for all tokens
    logger.info("Computing pre-repair outcomes...")
    treatment_outcomes: list[TokenOutcome] = []
    control_outcomes: list[TokenOutcome] = []

    for token_id in treatment_token_ids:
        if token_id not in token_metrics:
            # Compute metrics if not already done
            query_vec = embeddings[token_id].reshape(1, -1)
            neighbors, distances = index.query(query_vec, k=k, exclude_self=True)
            stage2 = compute_stage2_metrics(
                embeddings=embeddings,
                token_id=token_id,
                neighbor_ids=neighbors[0],
            )
            token_metrics[token_id] = {
                "cond": stage2.cond,
                "pr": stage2.pr,
                "logdet": stage2.logdet,
            }
            severities[token_id] = float(np.log1p(stage2.cond) + np.log1p(k / (stage2.pr + 1e-10)))

        metrics = token_metrics[token_id]
        failure_rate = simulate_failure_rate(
            cond=metrics["cond"],
            pr=metrics["pr"],
            logdet=metrics["logdet"],
            severity=severities[token_id],
            rng=rng,
        )

        treatment_outcomes.append(TokenOutcome(
            token_id=token_id,
            group="treatment",
            geometry_before=metrics.copy(),
            severity=severities[token_id],
            failure_rate_before=failure_rate,
        ))

    for token_id in control_token_ids:
        if token_id not in token_metrics:
            query_vec = embeddings[token_id].reshape(1, -1)
            neighbors, distances = index.query(query_vec, k=k, exclude_self=True)
            stage2 = compute_stage2_metrics(
                embeddings=embeddings,
                token_id=token_id,
                neighbor_ids=neighbors[0],
            )
            token_metrics[token_id] = {
                "cond": stage2.cond,
                "pr": stage2.pr,
                "logdet": stage2.logdet,
            }
            severities[token_id] = float(np.log1p(stage2.cond) + np.log1p(k / (stage2.pr + 1e-10)))

        metrics = token_metrics[token_id]
        failure_rate = simulate_failure_rate(
            cond=metrics["cond"],
            pr=metrics["pr"],
            logdet=metrics["logdet"],
            severity=severities[token_id],
            rng=rng,
        )

        control_outcomes.append(TokenOutcome(
            token_id=token_id,
            group="control",
            geometry_before=metrics.copy(),
            severity=severities[token_id],
            failure_rate_before=failure_rate,
        ))

    # Apply cluster repair to treatment group
    logger.info("Applying cluster repair to treatment group...")
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

    # Create a copy of embeddings for repair
    repaired_embeddings = embeddings.copy()

    for cluster in tqdm(treatment_clusters, desc="Repairing treatment clusters"):
        result = optimizer.repair_cluster(
            cluster=cluster,
            embeddings=repaired_embeddings,
            index=index,
            k=k,
        )

        # Update repaired embeddings
        for i, tid in enumerate(result.token_ids):
            repaired_embeddings[tid] = result.repaired_embeddings[i]

    # Apply placebo repair to control group
    logger.info("Applying placebo repair to control group...")
    placebo_embeddings = apply_placebo_repair(
        embeddings=embeddings,
        token_ids=control_token_ids,
        delta_max=repair_config.delta_max,
        rng=rng,
    )

    # Update control embeddings in repaired array
    for i, tid in enumerate(control_token_ids):
        repaired_embeddings[tid] = placebo_embeddings[i]

    # Rebuild index with repaired embeddings for post-repair metrics
    logger.info("Rebuilding index with repaired embeddings...")
    repaired_index = USearchIndex(
        connectivity=cfg.compute.index.connectivity,
        expansion_add=cfg.compute.index.expansion_add,
        expansion_search=cfg.compute.index.expansion_search,
    )
    repaired_index.build(repaired_embeddings, metric=cfg.neighbors.metric, seed=cfg.seed)

    # Compute post-repair outcomes
    logger.info("Computing post-repair outcomes...")

    for outcome in tqdm(treatment_outcomes, desc="Treatment post-repair"):
        token_id = outcome.token_id
        query_vec = repaired_embeddings[token_id].reshape(1, -1)
        neighbors, _ = repaired_index.query(query_vec, k=k, exclude_self=True)

        stage2 = compute_stage2_metrics(
            embeddings=repaired_embeddings,
            token_id=token_id,
            neighbor_ids=neighbors[0],
        )

        outcome.geometry_after = {
            "cond": stage2.cond,
            "pr": stage2.pr,
            "logdet": stage2.logdet,
        }

        # New severity based on repaired geometry
        new_severity = np.log1p(stage2.cond) + np.log1p(k / (stage2.pr + 1e-10))

        outcome.failure_rate_after = simulate_failure_rate(
            cond=stage2.cond,
            pr=stage2.pr,
            logdet=stage2.logdet,
            severity=new_severity,
            rng=rng,
        )

    for outcome in tqdm(control_outcomes, desc="Control post-repair"):
        token_id = outcome.token_id
        query_vec = repaired_embeddings[token_id].reshape(1, -1)
        neighbors, _ = repaired_index.query(query_vec, k=k, exclude_self=True)

        stage2 = compute_stage2_metrics(
            embeddings=repaired_embeddings,
            token_id=token_id,
            neighbor_ids=neighbors[0],
        )

        outcome.geometry_after = {
            "cond": stage2.cond,
            "pr": stage2.pr,
            "logdet": stage2.logdet,
        }

        new_severity = np.log1p(stage2.cond) + np.log1p(k / (stage2.pr + 1e-10))

        outcome.failure_rate_after = simulate_failure_rate(
            cond=stage2.cond,
            pr=stage2.pr,
            logdet=stage2.logdet,
            severity=new_severity,
            rng=rng,
        )

    # Compute causal effects
    logger.info("Computing causal effects with bootstrap CIs...")

    treatment_before = np.array([o.failure_rate_before for o in treatment_outcomes])
    treatment_after = np.array([o.failure_rate_after for o in treatment_outcomes])
    control_before = np.array([o.failure_rate_before for o in control_outcomes])
    control_after = np.array([o.failure_rate_after for o in control_outcomes])

    ate, ate_ci_low, ate_ci_high, did, did_ci_low, did_ci_high = compute_bootstrap_ci(
        treatment_before=treatment_before,
        treatment_after=treatment_after,
        control_before=control_before,
        control_after=control_after,
        n_bootstrap=causal_cfg.n_bootstrap,
        confidence_level=causal_cfg.confidence_level,
        rng=rng,
    )

    # Compute geometry changes
    treatment_cond_before = np.mean([o.geometry_before["cond"] for o in treatment_outcomes])
    treatment_cond_after = np.mean([o.geometry_after["cond"] for o in treatment_outcomes])
    control_cond_before = np.mean([o.geometry_before["cond"] for o in control_outcomes])
    control_cond_after = np.mean([o.geometry_after["cond"] for o in control_outcomes])

    # Statistical significance: check if CI excludes zero
    significant = (did_ci_low > 0) or (did_ci_high < 0)

    # P-value approximation from bootstrap
    did_samples = []
    for _ in range(causal_cfg.n_bootstrap):
        t_idx = rng.choice(len(treatment_before), size=len(treatment_before), replace=True)
        c_idx = rng.choice(len(control_before), size=len(control_before), replace=True)
        t_change = treatment_after[t_idx].mean() - treatment_before[t_idx].mean()
        c_change = control_after[c_idx].mean() - control_before[c_idx].mean()
        did_samples.append(t_change - c_change)

    # Two-tailed p-value: proportion of samples crossing zero
    did_samples = np.array(did_samples)
    if did < 0:
        p_value = 2 * (did_samples >= 0).mean()
    else:
        p_value = 2 * (did_samples <= 0).mean()

    causal_result = CausalResult(
        n_treatment=len(treatment_outcomes),
        n_control=len(control_outcomes),
        treatment_failure_before=float(treatment_before.mean()),
        treatment_failure_after=float(treatment_after.mean()),
        control_failure_before=float(control_before.mean()),
        control_failure_after=float(control_after.mean()),
        ate=ate,
        ate_ci_low=ate_ci_low,
        ate_ci_high=ate_ci_high,
        did=did,
        did_ci_low=did_ci_low,
        did_ci_high=did_ci_high,
        treatment_cond_change=float(treatment_cond_after - treatment_cond_before),
        control_cond_change=float(control_cond_after - control_cond_before),
        significant=significant,
        p_value=float(p_value),
    )

    # Print results
    logger.info("\n" + "=" * 60)
    logger.info("CAUSAL CLUSTER REPAIR RESULTS")
    logger.info("=" * 60)

    logger.info(f"\nSample Sizes:")
    logger.info(f"  Treatment: {causal_result.n_treatment} tokens ({n_treatment} clusters)")
    logger.info(f"  Control: {causal_result.n_control} tokens")

    logger.info(f"\nFailure Rates (simulated):")
    logger.info(f"  Treatment before: {causal_result.treatment_failure_before:.3f}")
    logger.info(f"  Treatment after:  {causal_result.treatment_failure_after:.3f}")
    logger.info(f"  Control before:   {causal_result.control_failure_before:.3f}")
    logger.info(f"  Control after:    {causal_result.control_failure_after:.3f}")

    logger.info(f"\nGeometry Changes (condition number):")
    logger.info(f"  Treatment: {causal_result.treatment_cond_change:+.3f}")
    logger.info(f"  Control:   {causal_result.control_cond_change:+.3f}")

    logger.info(f"\nCausal Effects:")
    logger.info(f"  ATE (Average Treatment Effect): {causal_result.ate:.4f}")
    logger.info(f"    95% CI: [{causal_result.ate_ci_low:.4f}, {causal_result.ate_ci_high:.4f}]")
    logger.info(f"  DiD (Difference-in-Differences): {causal_result.did:.4f}")
    logger.info(f"    95% CI: [{causal_result.did_ci_low:.4f}, {causal_result.did_ci_high:.4f}]")
    logger.info(f"  P-value: {causal_result.p_value:.4f}")
    logger.info(f"  Significant: {'YES' if causal_result.significant else 'NO'}")

    logger.info("\n" + "-" * 60)
    logger.info("INTERPRETATION:")

    if causal_result.did < 0:
        logger.info("  Cluster repair REDUCES failure rate relative to placebo!")
        logger.info(f"  Treatment failure dropped by {-causal_result.did:.3f} more than control")
    else:
        logger.info("  Cluster repair did not reduce failure rate more than placebo")

    if causal_result.treatment_cond_change < 0 and causal_result.control_cond_change >= 0:
        logger.info("  Geometry improved in treatment but not control (as expected)")

    # Save results
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output = convert_numpy({
        "summary": {
            "n_treatment": causal_result.n_treatment,
            "n_control": causal_result.n_control,
            "n_clusters": n_treatment,
            "treatment_failure_before": causal_result.treatment_failure_before,
            "treatment_failure_after": causal_result.treatment_failure_after,
            "control_failure_before": causal_result.control_failure_before,
            "control_failure_after": causal_result.control_failure_after,
            "ate": causal_result.ate,
            "ate_ci": [causal_result.ate_ci_low, causal_result.ate_ci_high],
            "did": causal_result.did,
            "did_ci": [causal_result.did_ci_low, causal_result.did_ci_high],
            "p_value": causal_result.p_value,
            "significant": causal_result.significant,
            "treatment_cond_change": causal_result.treatment_cond_change,
            "control_cond_change": causal_result.control_cond_change,
        },
        "treatment_outcomes": [
            {
                "token_id": o.token_id,
                "severity": o.severity,
                "geometry_before": o.geometry_before,
                "geometry_after": o.geometry_after,
                "failure_before": o.failure_rate_before,
                "failure_after": o.failure_rate_after,
            }
            for o in treatment_outcomes
        ],
        "control_outcomes": [
            {
                "token_id": o.token_id,
                "severity": o.severity,
                "geometry_before": o.geometry_before,
                "geometry_after": o.geometry_after,
                "failure_before": o.failure_rate_before,
                "failure_after": o.failure_rate_after,
            }
            for o in control_outcomes
        ],
        "config": {
            "n_treatment_clusters": causal_cfg.n_treatment_clusters,
            "n_diagnostic_samples": causal_cfg.n_diagnostic_samples,
            "mutual_k": causal_cfg.mutual_k,
            "min_cluster_size": causal_cfg.min_cluster_size,
            "n_bootstrap": causal_cfg.n_bootstrap,
            "model": cfg.model.name,
        },
    })

    output_path = output_dir / "causal_cluster_repair_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    logger.info(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
