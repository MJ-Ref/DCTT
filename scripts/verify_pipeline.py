#!/usr/bin/env python3
"""Verify DCTT pipeline with synthetic data.

This script validates the entire diagnostic and repair pipeline
using synthetic embeddings, without requiring a model download.
"""

import sys
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def create_synthetic_embeddings(vocab_size: int = 1000, dim: int = 256, seed: int = 42):
    """Create synthetic normalized embeddings with some pathological tokens."""
    rng = np.random.default_rng(seed)

    # Base embeddings - random unit vectors
    embeddings = rng.standard_normal((vocab_size, dim))
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    # Create some pathological tokens (collapsed local geometry)
    # These will have neighbors that are all in one direction
    for i in range(10):  # First 10 tokens are pathological
        direction = rng.standard_normal(dim)
        direction = direction / np.linalg.norm(direction)
        # Make neighbors of this token very similar
        for j in range(1, 6):
            neighbor_idx = (i + j) % vocab_size
            embeddings[neighbor_idx] = embeddings[i] + 0.01 * direction * j
            embeddings[neighbor_idx] /= np.linalg.norm(embeddings[neighbor_idx])

    return embeddings


def test_stage1_metrics():
    """Test Stage 1 metrics computation."""
    print("\n=== Testing Stage 1 Metrics ===")
    from dctt.metrics.stage1 import compute_stage1_metrics

    # Simulate kNN distances
    distances = np.array([0.1, 0.12, 0.15, 0.18, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5])
    token_id = 0

    result = compute_stage1_metrics(distances, token_id)

    print(f"  μ_k (mean distance): {result.mu_k:.4f}")
    print(f"  med_k (median distance): {result.med_k:.4f}")
    print(f"  spread_q (q90/q10 ratio): {result.spread_q:.4f}")

    assert result.mu_k > 0, "Mean distance should be positive"
    assert result.spread_q >= 1, "Spread ratio should be >= 1"
    print("  ✓ Stage 1 metrics pass")
    return True


def test_stage2_metrics():
    """Test Stage 2 metrics computation."""
    print("\n=== Testing Stage 2 Metrics ===")
    from dctt.metrics.stage2 import compute_stage2_metrics

    # Create synthetic embeddings
    rng = np.random.default_rng(42)
    vocab_size = 100
    d = 64
    k = 50

    embeddings = rng.standard_normal((vocab_size, d))
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    token_id = 0
    neighbor_ids = np.arange(1, k + 1)

    result = compute_stage2_metrics(embeddings, token_id, neighbor_ids)

    print(f"  dim95: {result.dim95}")
    print(f"  PR (participation ratio): {result.pr:.4f}")
    print(f"  cond (condition number): {result.cond:.4f}")
    print(f"  logdet: {result.logdet:.4f}")
    print(f"  anisotropy: {result.anisotropy:.4f}")

    assert 1 <= result.pr <= d, f"PR should be in [1, {d}], got {result.pr}"
    assert result.cond >= 1, f"Condition number should be >= 1, got {result.cond}"
    assert result.dim95 >= 1, f"Effective dimension should be >= 1, got {result.dim95}"
    print("  ✓ Stage 2 metrics pass")
    return True


def test_usearch_index():
    """Test USearch index building and querying."""
    print("\n=== Testing USearch Index ===")
    from dctt.neighbors.usearch_index import USearchIndex

    # Create synthetic embeddings
    embeddings = create_synthetic_embeddings(vocab_size=500, dim=128)

    # Build index
    index = USearchIndex(connectivity=32, expansion_add=128, expansion_search=64)
    index.build(embeddings, metric="cos", seed=42)

    print(f"  Index built with {len(embeddings)} vectors")

    # Query - exclude_self=True handles self-removal by distance threshold
    query = embeddings[0:1]  # Query expects 2D array
    neighbor_ids, distances = index.query(query, k=10, exclude_self=True)

    # Get first result
    neighbor_ids = neighbor_ids[0]
    distances = distances[0]

    print(f"  Query returned {len(neighbor_ids)} neighbors")
    print(f"  Nearest neighbor distance: {distances[0]:.4f}")
    print(f"  Farthest neighbor distance: {distances[-1]:.4f}")

    assert len(neighbor_ids) == 10, "Should return 10 neighbors"
    assert distances[0] <= distances[-1], "Distances should be sorted"
    print("  ✓ USearch index pass")
    return True


def test_severity_scoring():
    """Test severity scoring."""
    print("\n=== Testing Severity Scoring ===")
    from dctt.core.types import (
        DiagnosticResult, TokenInfo, Stage1Result, Stage2Result,
        TokenType, FrequencyTier
    )
    from dctt.metrics.severity import SeverityScorer

    # Create mock diagnostic results
    results = []
    for i in range(100):
        # Vary metrics to create different severities
        cond = 10 + i * 2  # Higher for later tokens
        pr = 50 - i * 0.3  # Lower for later tokens

        token_info = TokenInfo(
            token_id=i,
            token_str=f"token_{i}",
            token_type=TokenType.SUBWORD,
            frequency=1000.0 - i * 5,
            frequency_tier=FrequencyTier.MID,
            norm=1.0,
        )
        stage1 = Stage1Result(
            token_id=i,
            mu_k=0.1 + i * 0.001,
            med_k=0.1,
            spread_q=1.5 + i * 0.01,
            lof=None,
        )
        stage2 = Stage2Result(
            token_id=i,
            dim95=max(5, 30 - i // 5),
            pr=max(1, pr),
            cond=cond,
            logdet=-50 - i,
            anisotropy=1.5 + i * 0.05,
            eigenvalues=None,
        )

        results.append(DiagnosticResult(
            token_info=token_info,
            stage1=stage1,
            stage2=stage2,
            severity=0.0,
            bucket=(FrequencyTier.MID, TokenType.SUBWORD),
        ))

    # Fit scorer and compute severities
    scorer = SeverityScorer()
    scorer.fit(results)
    severities = scorer.compute_severity_batch(results)

    # Update results with computed severities
    for r, sev in zip(results, severities):
        r.severity = sev

    print(f"  Scored {len(results)} tokens")
    print(f"  Min severity: {min(severities):.4f}")
    print(f"  Max severity: {max(severities):.4f}")
    print(f"  Mean severity: {np.mean(severities):.4f}")

    # Later tokens should have higher severity (worse geometry)
    assert severities[-1] > severities[0], "Later tokens should have higher severity"
    print("  ✓ Severity scoring pass")
    return True


def test_repair_optimizer():
    """Test repair optimizer."""
    print("\n=== Testing Repair Optimizer ===")
    from dctt.core.types import RepairConfig
    from dctt.repair.optimizer import EmbeddingRepairOptimizer
    from dctt.neighbors.usearch_index import USearchIndex
    from dctt.metrics.stage2 import compute_stage2_metrics

    # Create embeddings with a pathological token
    embeddings = create_synthetic_embeddings(vocab_size=200, dim=64)

    # Build index
    index = USearchIndex(connectivity=32)
    index.build(embeddings, metric="cos", seed=42)

    # Get pre-repair metrics for token 0 (pathological)
    token_id = 0
    query = embeddings[token_id:token_id+1]
    neighbor_ids, _ = index.query(query, k=50, exclude_self=True)
    neighbor_ids = neighbor_ids[0]

    pre_metrics = compute_stage2_metrics(embeddings, token_id, neighbor_ids)
    print(f"  Pre-repair cond: {pre_metrics.cond:.4f}")
    print(f"  Pre-repair PR: {pre_metrics.pr:.4f}")

    # Repair - use correct API fields
    config = RepairConfig(
        lambda_anchor=1.0,
        lambda_nn_preserve=0.1,
        delta_max=0.1,
        max_outer_iters=3,
        max_inner_steps=20,
        learning_rate=0.01,
    )

    optimizer = EmbeddingRepairOptimizer(config)
    result = optimizer.repair(
        embedding=embeddings[token_id],
        neighbors=neighbor_ids,
        all_embeddings=embeddings,
        index=index,
        k=50,
    )

    # Get post-repair metrics
    query = result.repaired_embedding.reshape(1, -1)
    neighbor_ids_post, _ = index.query(query, k=50, exclude_self=True)
    neighbor_ids_post = neighbor_ids_post[0]

    # Need to create temp embeddings with repaired token
    temp_embeddings = embeddings.copy()
    temp_embeddings[token_id] = result.repaired_embedding

    post_metrics = compute_stage2_metrics(temp_embeddings, token_id, neighbor_ids_post)
    print(f"  Post-repair cond: {post_metrics.cond:.4f}")
    print(f"  Post-repair PR: {post_metrics.pr:.4f}")
    print(f"  Embedding delta: {np.linalg.norm(result.repaired_embedding - embeddings[token_id]):.4f}")
    print(f"  Converged: {result.converged}")

    assert result.iterations > 0, "Optimizer should run"
    print("  ✓ Repair optimizer pass")
    return True


def test_full_pipeline():
    """Test full diagnostic pipeline on synthetic data."""
    print("\n=== Testing Full Pipeline ===")
    from dctt.core.types import TokenInfo, DiagnosticResult, TokenType, FrequencyTier
    from dctt.metrics.stage1 import compute_stage1_metrics
    from dctt.metrics.stage2 import compute_stage2_metrics
    from dctt.metrics.severity import SeverityScorer
    from dctt.neighbors.usearch_index import USearchIndex

    # Create synthetic embeddings
    vocab_size = 500
    dim = 128
    k = 50

    embeddings = create_synthetic_embeddings(vocab_size=vocab_size, dim=dim)

    # Build index
    print("  Building index...")
    index = USearchIndex(connectivity=32)
    index.build(embeddings, metric="cos", seed=42)

    # Run diagnostics on sample
    sample_size = 100
    results = []

    print(f"  Running diagnostics on {sample_size} tokens...")
    for token_id in range(sample_size):
        embedding = embeddings[token_id:token_id+1]

        # Query neighbors
        neighbor_ids, distances = index.query(embedding, k=k, exclude_self=True)
        neighbor_ids = neighbor_ids[0]
        distances = distances[0]

        # Stage 1
        stage1 = compute_stage1_metrics(distances, token_id)

        # Stage 2
        stage2 = compute_stage2_metrics(embeddings, token_id, neighbor_ids)

        token_info = TokenInfo(
            token_id=token_id,
            token_str=f"token_{token_id}",
            token_type=TokenType.SUBWORD,
            frequency=1000.0,
            frequency_tier=FrequencyTier.MID,
            norm=1.0,
        )

        results.append(DiagnosticResult(
            token_info=token_info,
            stage1=stage1,
            stage2=stage2,
            severity=0.0,
            bucket=(FrequencyTier.MID, TokenType.SUBWORD),
        ))

    # Score severity
    print("  Computing severity scores...")
    scorer = SeverityScorer()
    scorer.fit(results)
    severities = scorer.compute_severity_batch(results)

    # Update results with computed severities
    for r, sev in zip(results, severities):
        r.severity = sev

    # Verify metrics are valid
    for r in results:
        assert 1 <= r.stage2.pr <= dim, f"PR out of range: {r.stage2.pr}"
        assert r.stage2.cond >= 1, f"Cond out of range: {r.stage2.cond}"
        assert r.stage2.dim95 >= 1, f"dim95 out of range: {r.stage2.dim95}"

    # Sort by severity
    results.sort(key=lambda x: x.severity, reverse=True)

    print(f"\n  Top 5 highest severity tokens:")
    for r in results[:5]:
        print(f"    Token {r.token_id}: severity={r.severity:.4f}, cond={r.stage2.cond:.2f}, PR={r.stage2.pr:.2f}")

    print(f"\n  Top 5 lowest severity tokens:")
    for r in results[-5:]:
        print(f"    Token {r.token_id}: severity={r.severity:.4f}, cond={r.stage2.cond:.2f}, PR={r.stage2.pr:.2f}")

    # Pathological tokens (0-9) should have higher severity
    pathological_severities = [r.severity for r in results if r.token_id < 10]
    other_severities = [r.severity for r in results if r.token_id >= 10]

    if pathological_severities and other_severities:
        avg_path = np.mean(pathological_severities)
        avg_other = np.mean(other_severities)
        print(f"\n  Avg severity (pathological tokens 0-9): {avg_path:.4f}")
        print(f"  Avg severity (other tokens): {avg_other:.4f}")

    print("  ✓ Full pipeline pass")
    return True


def main():
    """Run all verification tests."""
    print("=" * 60)
    print("DCTT Pipeline Verification")
    print("=" * 60)

    tests = [
        ("Stage 1 Metrics", test_stage1_metrics),
        ("Stage 2 Metrics", test_stage2_metrics),
        ("USearch Index", test_usearch_index),
        ("Severity Scoring", test_severity_scoring),
        ("Repair Optimizer", test_repair_optimizer),
        ("Full Pipeline", test_full_pipeline),
    ]

    passed = 0
    failed = 0

    for name, test_fn in tests:
        try:
            if test_fn():
                passed += 1
        except Exception as e:
            print(f"\n  ✗ {name} FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
