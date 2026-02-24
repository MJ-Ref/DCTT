"""Integration smoke tests for experiment scripts."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
from omegaconf import OmegaConf

from dctt.core.types import FrequencyTier, TokenInfo, TokenType
from dctt.neighbors.usearch_index import USearchIndex
from experiments import (
    run_ablation_suite,
    run_predictive_validity,
    run_repair_validation,
)


class DummyTokenizer:
    """Minimal tokenizer stub for experiment smoke tests."""

    pad_token_id = 0
    eos_token_id = 0

    def decode(self, token_ids: list[int], skip_special_tokens: bool = False) -> str:
        _ = skip_special_tokens
        token_id = int(token_ids[0])
        if token_id % 4 == 0:
            return "{"
        if token_id % 4 == 1:
            return "token"
        if token_id % 4 == 2:
            return "123"
        return " "


def _make_embeddings(
    n_tokens: int,
    dim: int,
    seed: int,
    non_negative: bool = False,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    embeddings = rng.standard_normal((n_tokens, dim))
    if non_negative:
        embeddings = np.abs(embeddings) + 0.1
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return (embeddings / norms).astype(np.float64)


def _patch_embedding_cache(
    monkeypatch: pytest.MonkeyPatch,
    embeddings: np.ndarray,
    cache_key: str,
) -> None:
    class DummyCache:
        def __init__(self, cache_dir: str) -> None:
            self.cache_dir = cache_dir

        def make_key(self, model_name: str, revision: str | None) -> str:
            _ = (model_name, revision)
            return cache_key

        def has(self, key: str) -> bool:
            return key == cache_key

        def load(self, key: str) -> tuple[np.ndarray, dict]:
            assert key == cache_key
            return embeddings, {"source": "test"}

        def save(self, key: str, values: np.ndarray, info: dict) -> None:
            _ = (key, values, info)

    monkeypatch.setattr("dctt.embeddings.cache.EmbeddingCache", DummyCache)


@pytest.mark.integration
def test_repair_validation_main_smoke_no_model_downloads(
    tmp_path: pytest.TempPathFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Repair validation main runs end-to-end with cached synthetic embeddings."""
    embeddings = _make_embeddings(n_tokens=64, dim=24, seed=1)
    _patch_embedding_cache(monkeypatch, embeddings, cache_key="repair_validation_smoke")
    monkeypatch.setattr(
        "transformers.AutoTokenizer.from_pretrained",
        lambda *args, **kwargs: DummyTokenizer(),
    )

    cfg = OmegaConf.create({
        "seed": 7,
        "output_dir": str(tmp_path),
        "model": {
            "name": "dummy/model",
            "revision": "main",
            "tokenizer": {"trust_remote_code": False},
        },
        "compute": {
            "index": {
                "connectivity": 8,
                "expansion_add": 16,
                "expansion_search": 16,
            }
        },
        "neighbors": {"k": 10, "metric": "ip"},
        "repair_validation": {"n_tokens": 2, "n_diagnostic_samples": 20},
    })

    run_repair_validation.main.__wrapped__(cfg)

    result_path = tmp_path / "repair_validation_results.json"
    assert result_path.exists()

    payload = json.loads(result_path.read_text())
    assert payload["summary"]["n_tokens_repaired"] == 2


@pytest.mark.integration
def test_predictive_validity_main_smoke_no_model_downloads(
    tmp_path: pytest.TempPathFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Predictive validity main runs with simulated failures and no model loads."""
    embeddings = _make_embeddings(n_tokens=72, dim=20, seed=2)
    _patch_embedding_cache(monkeypatch, embeddings, cache_key="predictive_validity_smoke")
    monkeypatch.setattr(
        "transformers.AutoTokenizer.from_pretrained",
        lambda *args, **kwargs: DummyTokenizer(),
    )

    cfg = OmegaConf.create({
        "seed": 13,
        "output_dir": str(tmp_path),
        "model": {
            "name": "dummy/model",
            "revision": "main",
            "torch_dtype": "float32",
            "trust_remote_code": False,
            "low_cpu_mem_usage": True,
            "tokenizer": {"trust_remote_code": False},
        },
        "compute": {
            "device": "cpu",
            "index": {
                "connectivity": 8,
                "expansion_add": 16,
                "expansion_search": 16,
            },
        },
        "neighbors": {"k": 10, "metric": "ip"},
        "wandb": {"enabled": False},
        "experiment": {"tokens": {"mode": "sample", "sample_size": 30}},
        "predictive_validity": {
            "n_bootstrap": 10,
            "failure_threshold": 0.3,
            "use_simulated_failures": True,
        },
        "stress_test": {"n_prompts": 2},
    })

    run_predictive_validity.main.__wrapped__(cfg)

    result_path = tmp_path / "predictive_validity_results.json"
    assert result_path.exists()

    payload = json.loads(result_path.read_text())
    assert payload["config"]["use_simulated_failures"] is True
    assert payload["summary"]["n_tokens"] > 0


@pytest.mark.integration
def test_ablation_post_metrics_use_repaired_embedding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Post-repair metrics are computed with repaired token embeddings."""
    embeddings = _make_embeddings(n_tokens=32, dim=16, seed=4, non_negative=True)

    token_infos = [
        TokenInfo(
            token_id=i,
            token_str=f"tok_{i}",
            token_type=TokenType.SUBWORD,
            frequency=1.0,
            frequency_tier=FrequencyTier.MID,
            norm=1.0,
        )
        for i in range(32)
    ]

    index = USearchIndex()
    index.build(embeddings, metric="ip", seed=42)

    cfg = OmegaConf.create({
        "seed": 0,
        "neighbors": {"k": 8},
        "repair": {
            "learning_rate": 0.1,
            "lambda_anchor": 0.1,
            "lambda_nn_preserve": 0.1,
            "delta_max": 0.2,
        },
    })

    class DummyOptimizer:
        def __init__(self, config: object) -> None:
            self.config = config

        def repair(
            self,
            embedding: np.ndarray,
            neighbors: np.ndarray,
            all_embeddings: np.ndarray,
            index: USearchIndex,
            k: int = 50,
            token_id: int | None = None,
        ) -> SimpleNamespace:
            _ = (neighbors, all_embeddings, index, k, token_id)
            repaired = embedding.copy()
            repaired[0] = -abs(repaired[0])
            repaired = repaired / np.linalg.norm(repaired)
            return SimpleNamespace(repaired_embedding=repaired)

    monkeypatch.setattr(run_ablation_suite, "EmbeddingRepairOptimizer", DummyOptimizer)

    seen_repaired_for_stage2: dict[str, bool] = {"value": False}

    def fake_stage2(
        embeddings: np.ndarray,
        token_id: int,
        neighbor_ids: np.ndarray,
    ) -> SimpleNamespace:
        _ = neighbor_ids
        if embeddings[token_id, 0] < 0:
            seen_repaired_for_stage2["value"] = True
        return SimpleNamespace(
            cond=float(1.0 + abs(embeddings[token_id, 0])),
            pr=float(1.0 + abs(embeddings[token_id, 1])),
        )

    monkeypatch.setattr(run_ablation_suite, "compute_stage2_metrics", fake_stage2)

    run_ablation_suite.run_repair_loss_ablation(
        cfg=cfg,
        embeddings=embeddings,
        index=index,
        token_infos=token_infos,
    )

    assert seen_repaired_for_stage2["value"] is True
