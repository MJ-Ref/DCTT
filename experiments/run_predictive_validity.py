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
    """Simulate stress test failures using confounds only.

    This is intended only for smoke tests when real model stress tests
    are unavailable. Failures are generated from coarse confounds
    (frequency tier and token type) plus noise, so geometry features
    are not leaked directly into labels.

    Do not use simulated failures for research claims.
    """
    rng = np.random.default_rng(seed)
    failures: dict[int, float] = {}

    tier_base = {
        FrequencyTier.HIGH: 0.06,
        FrequencyTier.MID: 0.12,
        FrequencyTier.LOW: 0.20,
    }
    type_adjustment = {
        TokenType.WHITESPACE: 0.02,
        TokenType.FULL_WORD: 0.03,
        TokenType.SUBWORD: 0.04,
        TokenType.PUNCTUATION: 0.08,
        TokenType.NUMERIC: 0.07,
        TokenType.CODE_SYMBOL: 0.10,
        TokenType.SPECIAL: 0.05,
        TokenType.UNKNOWN: 0.05,
    }

    for result in diagnostic_results:
        base_prob = tier_base[result.token_info.frequency_tier]
        base_prob += type_adjustment.get(result.token_info.token_type, 0.05)
        noise = rng.normal(0.0, 0.05)
        failures[result.token_id] = float(np.clip(base_prob + noise, 0.0, 1.0))

    return failures


def _norms_cache_path(cache_dir: Path, cache_key: str) -> Path:
    """Return norms cache path aligned to embedding cache key."""
    return cache_dir / f"{cache_key}.norms.npy"


def _load_cached_norms(
    cache_dir: Path,
    cache_key: str,
    expected_vocab_size: int,
) -> np.ndarray | None:
    """Load cached raw embedding norms if available and shape-compatible."""
    path = _norms_cache_path(cache_dir, cache_key)
    if not path.exists():
        return None
    try:
        norms = np.load(path)
    except Exception as exc:
        logger.warning("Failed to load norms cache %s: %s", path, exc)
        return None
    if norms.ndim != 1 or int(norms.shape[0]) != int(expected_vocab_size):
        logger.warning(
            "Ignoring norms cache with unexpected shape %s (expected %d,)",
            norms.shape,
            expected_vocab_size,
        )
        return None
    return norms.astype(np.float64)


def _save_cached_norms(
    cache_dir: Path,
    cache_key: str,
    norms: np.ndarray,
) -> None:
    """Persist raw embedding norms for later confound-aware analysis."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = _norms_cache_path(cache_dir, cache_key)
    np.save(path, norms.astype(np.float64))


def _resolve_path(repo_root: Path, candidate: str) -> Path:
    """Resolve potentially-relative path against repository root."""
    path = Path(candidate).expanduser()
    if path.is_absolute():
        return path
    return (repo_root / path).resolve()


def _load_frequency_counts(
    *,
    path: Path,
    vocab_size: int,
    allow_length_mismatch: bool = True,
) -> np.ndarray | None:
    """Load token frequency counts vector from supported formats.

    Supported:
    - `.npy`: dense vector with length == vocab_size (or auto pad/truncate
      when allow_length_mismatch=True)
    - `.json`: dict/list payload. Accepted forms:
      - {"counts": [..]}
      - {"counts": {"<token_id>": count, ...}}
      - [..]
    """
    if not path.exists():
        logger.warning("Frequency counts path does not exist: %s", path)
        return None

    try:
        def _align_dense(values: np.ndarray) -> np.ndarray | None:
            if values.ndim != 1:
                logger.warning(
                    "Invalid frequency vector shape %s at %s (expected 1D)",
                    values.shape,
                    path,
                )
                return None

            clipped = np.clip(values.astype(np.float64), 0.0, None)
            n_current = int(clipped.shape[0])
            n_target = int(vocab_size)
            if n_current == n_target:
                return clipped
            if not allow_length_mismatch:
                logger.warning(
                    "Frequency vector length %d does not match vocab size %d at %s; "
                    "strict alignment is enabled so this file is rejected.",
                    n_current,
                    n_target,
                    path,
                )
                return None
            if n_current < n_target:
                logger.warning(
                    "Frequency vector length %d < vocab size %d at %s; padding trailing zeros.",
                    n_current,
                    n_target,
                    path,
                )
                padded = np.zeros(n_target, dtype=np.float64)
                padded[:n_current] = clipped
                return padded

            logger.warning(
                "Frequency vector length %d > vocab size %d at %s; truncating extras.",
                n_current,
                n_target,
                path,
            )
            return clipped[:n_target]

        if path.suffix.lower() == ".npy":
            values = np.load(path)
            return _align_dense(values)

        if path.suffix.lower() == ".json":
            with path.open() as f:
                payload = json.load(f)

            raw_counts = payload.get("counts", payload) if isinstance(payload, dict) else payload
            counts = np.zeros(int(vocab_size), dtype=np.float64)

            if isinstance(raw_counts, list):
                aligned = _align_dense(np.asarray(raw_counts, dtype=np.float64))
                if aligned is None:
                    return None
                counts = aligned
            elif isinstance(raw_counts, dict):
                for key, value in raw_counts.items():
                    try:
                        token_id = int(key)
                        if 0 <= token_id < int(vocab_size):
                            counts[token_id] = max(float(value), 0.0)
                    except Exception:
                        continue
            else:
                logger.warning("Unsupported frequency JSON payload at %s", path)
                return None

            return np.clip(counts.astype(np.float64), 0.0, None)
    except Exception as exc:
        logger.warning("Failed to load frequency counts from %s: %s", path, exc)
        return None

    logger.warning("Unsupported frequency counts extension for %s", path)
    return None


def _frequency_tier_from_values(
    value: float,
    q20: float,
    q80: float,
) -> FrequencyTier:
    """Map scalar frequency signal to HIGH/MID/LOW tier via log quantiles."""
    log_val = np.log(max(float(value), 0.0) + 1.0)
    if log_val >= q80:
        return FrequencyTier.HIGH
    if log_val <= q20:
        return FrequencyTier.LOW
    return FrequencyTier.MID


class _StressModelAdapter:
    """Adapter exposing generation and candidate-scoring helpers for stress tests."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        default_max_new_tokens: int,
    ) -> None:
        import torch

        self.model = model
        self.tokenizer = tokenizer
        self.default_max_new_tokens = int(default_max_new_tokens)
        self.torch = torch
        self.device = getattr(model, "device", torch.device("cpu"))

    def _on_device(self, tensor):
        if str(self.device) == "cpu":
            return tensor
        return tensor.to(self.device)

    def __call__(self, prompt: str, max_new_tokens: int | None = None) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt")
        if str(self.device) != "cpu":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

        generate_kwargs = {
            "max_new_tokens": (
                int(max_new_tokens)
                if max_new_tokens is not None
                else self.default_max_new_tokens
            ),
            "do_sample": False,
        }
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id
        if pad_token_id is not None:
            generate_kwargs["pad_token_id"] = pad_token_id

        with self.torch.no_grad():
            outputs = self.model.generate(**inputs, **generate_kwargs)

        prompt_len = int(inputs["input_ids"].shape[1])
        return self.tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True)

    def score_options(self, prompt: str, options: list[str]) -> dict[str, Any]:
        """Score candidate continuations by mean token log-probability."""
        prompt_ids = self.tokenizer(
            prompt,
            add_special_tokens=False,
        )["input_ids"]
        if not prompt_ids:
            fallback = self.tokenizer.bos_token_id
            if fallback is None:
                fallback = self.tokenizer.eos_token_id
            if fallback is not None:
                prompt_ids = [int(fallback)]

        scores: dict[str, float] = {}
        for option in options:
            option_ids = self.tokenizer(
                option,
                add_special_tokens=False,
            )["input_ids"]
            if not option_ids:
                scores[str(option)] = float("-inf")
                continue

            full_ids = list(prompt_ids) + list(option_ids)
            input_ids = self.torch.tensor([full_ids], dtype=self.torch.long)
            input_ids = self._on_device(input_ids)

            with self.torch.no_grad():
                logits = self.model(input_ids=input_ids).logits[0]
                log_probs = self.torch.log_softmax(logits, dim=-1)

            start = len(prompt_ids)
            positions = list(range(start, len(full_ids)))
            if not positions:
                scores[str(option)] = float("-inf")
                continue

            prev_positions = self.torch.tensor(
                [pos - 1 for pos in positions],
                dtype=self.torch.long,
                device=log_probs.device,
            )
            next_tokens = self.torch.tensor(
                [full_ids[pos] for pos in positions],
                dtype=self.torch.long,
                device=log_probs.device,
            )
            token_log_probs = log_probs[prev_positions, next_tokens]
            if token_log_probs.numel() == 0:
                scores[str(option)] = float("-inf")
            else:
                scores[str(option)] = float(token_log_probs.mean().item())

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        best_option = ranked[0][0] if ranked else ""
        best_score = float(ranked[0][1]) if ranked else float("-inf")
        second_score = float(ranked[1][1]) if len(ranked) > 1 else float("-inf")
        margin = (
            best_score - second_score
            if len(ranked) > 1
            else float("inf")
        )

        return {
            "best_option": best_option,
            "best_score": best_score,
            "margin": float(margin),
            "scores": {key: float(value) for key, value in scores.items()},
        }


def _create_model_fn(
    model: Any,
    tokenizer: Any,
    default_max_new_tokens: int,
) -> Any:
    """Create model adapter for stress tests."""
    return _StressModelAdapter(
        model=model,
        tokenizer=tokenizer,
        default_max_new_tokens=default_max_new_tokens,
    )


def _resolve_runtime_device(
    requested_device: str,
    torch_module: Any,
) -> str:
    """Resolve runtime device with availability-aware fallback."""
    requested = str(requested_device).strip().lower()
    has_cuda = bool(torch_module.cuda.is_available())
    backends = getattr(torch_module, "backends", None)
    mps_backend = getattr(backends, "mps", None) if backends is not None else None
    has_mps = bool(mps_backend is not None and mps_backend.is_available())

    if requested == "auto":
        if has_mps:
            return "mps"
        if has_cuda:
            return "cuda"
        return "cpu"

    if requested == "mps":
        if has_mps:
            return "mps"
        fallback = "cuda" if has_cuda else "cpu"
        logger.warning(
            "Requested device 'mps' unavailable; falling back to '%s'.",
            fallback,
        )
        return fallback

    if requested == "cuda":
        if has_cuda:
            return "cuda"
        fallback = "mps" if has_mps else "cpu"
        logger.warning(
            "Requested device 'cuda' unavailable; falling back to '%s'.",
            fallback,
        )
        return fallback

    if requested == "cpu":
        return "cpu"

    logger.warning(
        "Unknown compute.device '%s'; falling back to auto selection.",
        requested_device,
    )
    if has_mps:
        return "mps"
    if has_cuda:
        return "cuda"
    return "cpu"


def run_stress_tests(
    cfg: DictConfig,
    diagnostic_results: list[DiagnosticResult],
    model: Any,
    tokenizer: Any,
) -> tuple[dict[int, float], dict[str, Any]]:
    """Run actual stress tests to get failure rates."""
    from dctt.stress_tests.forced_minimal_pair import (
        ForcedTokenMinimalPairTest,
        build_minimal_pair_control_map,
    )
    from dctt.stress_tests.runner import RunnerConfig, StressTestRunner

    control_map = build_minimal_pair_control_map(
        diagnostic_results,
        seed=int(cfg.seed),
    )

    stress_cfg = cfg.get("stress_test", cfg.get("stress_tests", {}))
    n_cases = int(stress_cfg.get("n_prompts", stress_cfg.get("n_samples", 10)))
    max_new_tokens = int(stress_cfg.get("max_new_tokens", 32))
    scoring_mode = str(stress_cfg.get("scoring_mode", "generation"))
    min_logprob_margin = float(stress_cfg.get("min_logprob_margin", 0.0))

    tests = [
        ForcedTokenMinimalPairTest(
            control_map=control_map,
            seed=int(cfg.seed),
            scoring_mode=scoring_mode,
            min_logprob_margin=min_logprob_margin,
        )
    ]

    runner = StressTestRunner(
        tests=tests,
        config=RunnerConfig(n_cases_per_token=n_cases),
    )

    tokens = [
        (result.token_id, result.token_info.token_str)
        for result in diagnostic_results
    ]
    model_fn = _create_model_fn(
        model,
        tokenizer,
        default_max_new_tokens=max_new_tokens,
    )

    logger.info(
        "Running stress tests on %d tokens with %d cases/token",
        len(tokens),
        n_cases,
    )
    results = runner.run(tokens=tokens, model_fn=model_fn)
    failure_rates = runner.compute_overall_failure_rate(results)

    pair_results = results.get("forced_token_minimal_pair", [])
    target_rates = [float(item.failure_rate) for item in pair_results]
    control_rates = [
        float(item.details.get("control_failure_rate", 0.0))
        for item in pair_results
    ]
    gaps = [
        float(item.details.get("failure_gap", 0.0))
        for item in pair_results
    ]
    target_margins = [
        float(case["target_margin"])
        for item in pair_results
        for case in item.details.get("cases", [])
        if (
            case.get("target_margin") is not None
            and np.isfinite(float(case["target_margin"]))
        )
    ]
    control_margins = [
        float(case["control_margin"])
        for item in pair_results
        for case in item.details.get("cases", [])
        if (
            case.get("control_margin") is not None
            and np.isfinite(float(case["control_margin"]))
        )
    ]

    summary = {
        "mode": "forced_token_minimal_pair",
        "scoring_mode": scoring_mode,
        "min_logprob_margin": min_logprob_margin,
        "n_tokens": len(pair_results),
        "n_cases_per_token": n_cases,
        "mean_target_failure_rate": float(np.mean(target_rates)) if target_rates else 0.0,
        "mean_control_failure_rate": float(np.mean(control_rates)) if control_rates else 0.0,
        "mean_failure_gap": float(np.mean(gaps)) if gaps else 0.0,
        "pct_positive_gap": float(np.mean([g > 0 for g in gaps])) if gaps else 0.0,
        "pct_negative_gap": float(np.mean([g < 0 for g in gaps])) if gaps else 0.0,
        "pct_zero_gap": float(np.mean([g == 0 for g in gaps])) if gaps else 0.0,
        "mean_target_margin": float(np.mean(target_margins)) if target_margins else 0.0,
        "mean_control_margin": float(np.mean(control_margins)) if control_margins else 0.0,
    }
    return failure_rates, summary


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
        repo_root = Path(__file__).parent.parent

        # Load cached embeddings
        from dctt.embeddings import extract_embeddings
        from dctt.embeddings.normalize import normalize_embeddings
        from dctt.embeddings.cache import EmbeddingCache

        cache_dir = repo_root / "outputs" / "embeddings"
        cache = EmbeddingCache(str(cache_dir))
        cache_key = cache.make_key(cfg.model.name, cfg.model.revision)
        raw_norms: np.ndarray | None = None

        if cache.has(cache_key):
            logger.info("Loading embeddings from cache")
            cached_payload = cache.load(cache_key)
            if cached_payload is None:
                logger.warning("Embedding cache exists but failed to load; recomputing.")
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
                raw_norms = norms.astype(np.float64)
                _save_cached_norms(cache_dir, cache_key, raw_norms)
            else:
                embeddings, _metadata = cached_payload
                raw_norms = _load_cached_norms(
                    cache_dir,
                    cache_key,
                    expected_vocab_size=int(embeddings.shape[0]),
                )
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
            raw_norms = norms.astype(np.float64)
            _save_cached_norms(cache_dir, cache_key, raw_norms)

        vocab_size, embedding_dim = embeddings.shape
        logger.info(f"Embeddings: {vocab_size} x {embedding_dim}")

        # Load tokenizer
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            cfg.model.name,
            trust_remote_code=cfg.model.tokenizer.trust_remote_code,
        )

        # Build or load index
        index_cache_dir = repo_root / "outputs" / "indices"
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

        predictive_cfg = cfg.get("predictive_validity", {})
        fail_on_proxy_confounds = bool(
            predictive_cfg.get("fail_on_proxy_confounds", False)
        )
        strict_frequency_alignment = bool(
            predictive_cfg.get("strict_frequency_alignment", True)
        )
        if fail_on_proxy_confounds:
            # Publication-mode runs must not silently remap frequency vectors.
            strict_frequency_alignment = True

        # Resolve norm confound source
        norm_source = "cached_raw_norms"
        if raw_norms is None or int(raw_norms.shape[0]) != int(vocab_size):
            if fail_on_proxy_confounds:
                raise RuntimeError(
                    "Raw embedding norms unavailable for confound controls. "
                    "Recompute embeddings and ensure norms cache is written."
                )
            raw_norms = np.ones(int(vocab_size), dtype=np.float64)
            norm_source = "unit_proxy"
            logger.warning(
                "Using proxy norm confound (all ones); set "
                "predictive_validity.fail_on_proxy_confounds=true to forbid this."
            )

        # Resolve frequency confound source
        frequency_source = "token_id_proxy"
        frequency_counts_path = predictive_cfg.get("token_frequency_counts_path", None)
        frequency_values: np.ndarray | None = None
        resolved_frequency_path: str | None = None
        if frequency_counts_path:
            candidate = _resolve_path(repo_root, str(frequency_counts_path))
            loaded = _load_frequency_counts(
                path=candidate,
                vocab_size=int(vocab_size),
                allow_length_mismatch=not strict_frequency_alignment,
            )
            if loaded is not None:
                frequency_values = loaded
                frequency_source = "corpus_counts"
                resolved_frequency_path = str(candidate)
            else:
                logger.warning(
                    "Falling back to proxy frequency confound; failed to load %s",
                    candidate,
                )

        if frequency_values is None:
            if fail_on_proxy_confounds:
                raise RuntimeError(
                    "Token frequency counts unavailable for confound controls. "
                    "Provide predictive_validity.token_frequency_counts_path with "
                    "vocab-aligned counts when strict_frequency_alignment=true."
                )
            frequency_values = 1.0 / (np.arange(int(vocab_size), dtype=np.float64) + 1.0)
            logger.warning(
                "Using proxy frequency confound from token rank; set "
                "predictive_validity.fail_on_proxy_confounds=true to forbid this."
            )

        log_all_freq = np.log1p(np.clip(frequency_values, 0.0, None))
        q20 = float(np.quantile(log_all_freq, 0.20))
        q80 = float(np.quantile(log_all_freq, 0.80))

        # Build token info with classification and confound features
        logger.info("Building token info...")
        token_infos = []
        for token_id in range(vocab_size):
            try:
                token_str = tokenizer.decode([token_id])
            except:
                token_str = f"<token_{token_id}>"

            token_type = classify_token_type(token_str)
            token_frequency = float(frequency_values[token_id])
            freq_tier = _frequency_tier_from_values(token_frequency, q20=q20, q80=q80)

            token_infos.append(TokenInfo(
                token_id=token_id,
                token_str=token_str,
                token_type=token_type,
                frequency=token_frequency,
                frequency_tier=freq_tier,
                norm=float(raw_norms[token_id]),
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
        use_simulated = cfg.get("predictive_validity", {}).get(
            "use_simulated_failures",
            False,
        )
        stress_cfg = cfg.get("stress_test", cfg.get("stress_tests", {}))
        stress_scoring_mode = str(stress_cfg.get("scoring_mode", "generation"))
        min_logprob_margin = float(stress_cfg.get("min_logprob_margin", 0.0))

        minimal_pair_summary: dict[str, Any] = {
            "mode": "simulated",
            "scoring_mode": stress_scoring_mode,
            "min_logprob_margin": min_logprob_margin,
            "n_tokens": len(diagnostic_results),
            "n_cases_per_token": 0,
            "mean_target_failure_rate": 0.0,
            "mean_control_failure_rate": 0.0,
            "mean_failure_gap": 0.0,
            "pct_positive_gap": 0.0,
            "pct_negative_gap": 0.0,
            "pct_zero_gap": 0.0,
            "mean_target_margin": 0.0,
            "mean_control_margin": 0.0,
        }

        if use_simulated:
            logger.warning(
                "Using simulated stress test failures (smoke-test mode only)."
            )
            stress_test_results = simulate_stress_test_failures(diagnostic_results, seed=cfg.seed)
        else:
            # Load model and run actual stress tests
            logger.info("Loading model for stress tests...")
            from transformers import AutoModelForCausalLM
            import torch

            device = _resolve_runtime_device(str(cfg.compute.device), torch)

            torch_dtype = getattr(torch, str(cfg.model.torch_dtype), None)
            if torch_dtype is None:
                torch_dtype = torch.float32 if device == "cpu" else torch.float16
            if device == "mps" and torch_dtype == torch.bfloat16:
                torch_dtype = torch.float16

            model = AutoModelForCausalLM.from_pretrained(
                cfg.model.name,
                torch_dtype=torch_dtype,
                trust_remote_code=cfg.model.trust_remote_code,
                low_cpu_mem_usage=cfg.model.get("low_cpu_mem_usage", True),
            )
            model.to(device)
            model.eval()
            stress_test_results, minimal_pair_summary = run_stress_tests(
                cfg,
                diagnostic_results,
                model,
                tokenizer,
            )

        # Run comprehensive predictive validity analysis
        logger.info("Running predictive validity analysis...")
        analyzer = PredictiveValidityAnalyzer(
            n_bootstrap=cfg.get("predictive_validity", {}).get("n_bootstrap", 100),
            random_state=cfg.seed,
            failure_threshold=cfg.get("predictive_validity", {}).get("failure_threshold", 0.3),
            strict_mode=bool(cfg.get("predictive_validity", {}).get("strict_evaluation", True)),
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
            "stress_test_summary": minimal_pair_summary,
            "config": {
                "model": cfg.model.name,
                "n_bootstrap": cfg.get("predictive_validity", {}).get("n_bootstrap", 100),
                "use_simulated_failures": use_simulated,
                "stress_test_design": "forced_token_minimal_pair" if not use_simulated else "simulated",
                "stress_test_scoring_mode": stress_scoring_mode,
                "min_logprob_margin": min_logprob_margin,
                "frequency_confound_source": frequency_source,
                "frequency_counts_path": resolved_frequency_path,
                "norm_confound_source": norm_source,
                "fail_on_proxy_confounds": fail_on_proxy_confounds,
                "strict_frequency_alignment": strict_frequency_alignment,
                "strict_evaluation": bool(
                    cfg.get("predictive_validity", {}).get("strict_evaluation", True)
                ),
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
