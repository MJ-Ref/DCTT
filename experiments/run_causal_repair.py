"""Causal Repair Experiment.

This script implements the causal repair experiment that:
1. Selects high-severity tokens for repair
2. Creates matched control groups
3. Applies repairs to treatment group
4. Evaluates outcomes vs controls

Usage:
    python experiments/run_causal_repair.py model=qwen2_5_coder_7b
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
    """Run causal repair experiment."""
    logger.info("Starting DCTT Causal Repair Experiment")

    # Override with causal_repair experiment config
    cfg = OmegaConf.merge(cfg, OmegaConf.load("configs/experiment/causal_repair.yaml"))

    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    from dctt.tracking.reproducibility import set_all_seeds
    set_all_seeds(cfg.seed)

    # Initialize W&B
    run = None
    if cfg.wandb.enabled:
        from dctt.tracking.wandb_utils import init_wandb_from_hydra
        run = init_wandb_from_hydra(cfg, tags=["causal_repair"])

    try:
        # Load embeddings
        logger.info("Loading embeddings")
        from dctt.embeddings.cache import EmbeddingCache

        cache = EmbeddingCache(cfg.output_dir + "/embeddings")
        cache_key = cache.make_key(cfg.model.name, cfg.model.revision)

        if not cache.has(cache_key):
            logger.error("Embeddings not found. Run census experiment first.")
            return

        embeddings, metadata = cache.load(cache_key)
        vocab_size, embedding_dim = embeddings.shape
        logger.info(f"Loaded embeddings: {vocab_size} x {embedding_dim}")

        # Load diagnostic results
        logger.info("Loading diagnostic results")
        from dctt.tracking.artifacts import load_results_json
        from dctt.core.types import DiagnosticResult, TokenInfo, Stage1Result, Stage2Result
        from dctt.core.types import TokenType, FrequencyTier

        results_path = Path(cfg.output_dir) / "diagnostic_results.json"
        if not results_path.exists():
            logger.error("Diagnostic results not found. Run census experiment first.")
            return

        results_data = load_results_json(results_path)

        # Reconstruct DiagnosticResult objects (simplified)
        results = []
        for data in results_data:
            token_info = TokenInfo(
                token_id=data["token_info"]["token_id"],
                token_str=data["token_info"]["token_str"],
                token_type=TokenType[data["token_info"]["token_type"]],
                frequency=data["token_info"]["frequency"],
                frequency_tier=FrequencyTier[data["token_info"]["frequency_tier"]],
                norm=data["token_info"]["norm"],
            )

            s1 = Stage1Result(
                token_id=data["stage1"]["token_id"],
                mu_k=data["stage1"]["mu_k"],
                med_k=data["stage1"]["med_k"],
                spread_q=data["stage1"]["spread_q"],
                fail=data["stage1"]["fail"],
            )

            s2 = Stage2Result(
                token_id=data["stage2"]["token_id"],
                dim95=data["stage2"]["dim95"],
                pr=data["stage2"]["pr"],
                cond=data["stage2"]["cond"],
                logdet=data["stage2"]["logdet"],
                anisotropy=data["stage2"]["anisotropy"],
                fail=data["stage2"]["fail"],
            )

            result = DiagnosticResult(
                token_info=token_info,
                stage1=s1,
                stage2=s2,
                severity=data.get("severity", 0.0),
                consistency=data.get("consistency", 1.0),
            )
            if "bucket" in data and data["bucket"]:
                result.bucket = (
                    FrequencyTier[data["bucket"][0]],
                    TokenType[data["bucket"][1]],
                )
            results.append(result)

        logger.info(f"Loaded {len(results)} diagnostic results")

        # Select repair candidates
        logger.info("Selecting repair candidates")
        from dctt.repair.candidate import CandidateSelector, CandidateSelectionConfig

        exp_cfg = cfg.experiment
        selector_config = CandidateSelectionConfig(
            top_n=exp_cfg.candidates.top_n,
            min_consistency=exp_cfg.candidates.min_consistency,
            min_severity_percentile=exp_cfg.candidates.min_severity_percentile,
        )
        selector = CandidateSelector(selector_config)
        candidates, controls = selector.select_with_matched_controls(results)

        logger.info(f"Selected {len(candidates)} candidates and {len(controls)} controls")

        # Build kNN index for repair
        logger.info("Building kNN index")
        from dctt.neighbors.usearch_index import USearchIndex

        index = USearchIndex(
            connectivity=cfg.compute.index.connectivity,
            expansion_add=cfg.compute.index.expansion_add,
            expansion_search=cfg.compute.index.expansion_search,
        )
        index.build(embeddings, metric=cfg.neighbors.metric, seed=cfg.seed)

        # Apply repairs
        logger.info("Applying repairs to candidates")
        from dctt.repair.optimizer import EmbeddingRepairOptimizer
        from dctt.core.types import RepairConfig

        repair_config = RepairConfig(
            max_outer_iters=exp_cfg.repair.max_outer_iters,
            max_inner_steps=exp_cfg.repair.max_inner_steps,
            learning_rate=exp_cfg.repair.learning_rate,
            lambda_anchor=exp_cfg.repair.lambda_anchor,
            lambda_nn_preserve=exp_cfg.repair.lambda_nn_preserve,
            delta_max=exp_cfg.repair.delta_max,
            geometry_loss=exp_cfg.repair.geometry_loss,
        )

        optimizer = EmbeddingRepairOptimizer(repair_config)
        repair_results = []

        for candidate in tqdm(candidates, desc="Repairing embeddings"):
            token_id = candidate.token_id
            embedding = embeddings[token_id]

            # Get neighbors
            neighbors, _ = index.query_single(
                embedding,
                k=cfg.neighbors.k,
                exclude_self=True,
                self_index=token_id,
            )

            # Repair
            result = optimizer.repair(
                embedding=embedding,
                neighbors=neighbors,
                all_embeddings=embeddings,
                index=index,
                k=cfg.neighbors.k,
                token_id=token_id,
            )
            result.token_id = token_id
            repair_results.append(result)

        logger.info(f"Repaired {len(repair_results)} embeddings")

        # Compute improvement statistics
        improvements = {
            "cond": [],
            "pr": [],
            "logdet": [],
        }

        for result in repair_results:
            for metric in improvements:
                if metric in result.geometry_before and metric in result.geometry_after:
                    before = result.geometry_before[metric]
                    after = result.geometry_after[metric]
                    if metric == "cond":
                        # Lower is better
                        improvements[metric].append(before - after)
                    else:
                        # Higher is better
                        improvements[metric].append(after - before)

        logger.info("Improvement statistics:")
        for metric, values in improvements.items():
            if values:
                mean_imp = np.mean(values)
                std_imp = np.std(values)
                logger.info(f"  {metric}: {mean_imp:.4f} +/- {std_imp:.4f}")

        # Save repair results
        output_dir = Path(cfg.output_dir)
        repair_dir = output_dir / "repairs"
        repair_dir.mkdir(parents=True, exist_ok=True)

        import json
        repair_data = [r.to_dict() for r in repair_results]
        with open(repair_dir / "repair_results.json", "w") as f:
            json.dump(repair_data, f, indent=2, default=str)

        logger.info(f"Repair results saved to {repair_dir}")

        # Log to W&B
        if run is not None:
            from dctt.tracking.wandb_utils import log_metrics

            log_metrics({
                "repair/n_candidates": len(candidates),
                "repair/n_controls": len(controls),
                "repair/mean_cond_improvement": np.mean(improvements["cond"]) if improvements["cond"] else 0,
                "repair/mean_pr_improvement": np.mean(improvements["pr"]) if improvements["pr"] else 0,
            })

    finally:
        if run is not None:
            from dctt.tracking.wandb_utils import finish_run
            finish_run()


if __name__ == "__main__":
    main()
