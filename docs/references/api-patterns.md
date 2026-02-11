# API Patterns

Current API signatures for DCTT components. Keep this updated when APIs change.

## Neighbor Index

```python
from dctt.neighbors import USearchIndex

# Construction
index = USearchIndex(
    metric="ip",           # Inner product (= cosine for normalized)
    connectivity=16,       # HNSW M parameter
    expansion_add=128,     # ef_construction
    expansion_search=64    # ef_search
)

# Adding embeddings
index.add(embeddings)  # Shape: (n_tokens, d)

# Querying - MUST be 2D input
distances, indices = index.query(
    embeddings[token_ids],  # Shape: (n_queries, d)
    k=50,
    exclude_self=True       # Remove self from results
)
```

## Metrics

```python
from dctt.metrics import compute_stage1_metrics, compute_stage2_metrics

# Stage 1: Fast screening
stage1 = compute_stage1_metrics(distances)
# Returns: {"mean_dist": float, "median_dist": float, "spread_q": float}

# Stage 2: Spectral geometry
stage2 = compute_stage2_metrics(embeddings, neighbor_indices)
# Returns: {"dim95": float, "pr": float, "cond": float, "logdet": float, "anisotropy": float}
```

## Severity Scoring

```python
from dctt.metrics import SeverityScorer

scorer = SeverityScorer()

# Fit on metrics DataFrame (learns bucket statistics)
scorer.fit(metrics_df)

# Compute severity scores
severity = scorer.compute_severity(metrics_df)
# Returns: np.ndarray of severity scores
```

## Repair

### Single-Token (Deprecated - doesn't improve geometry)

```python
from dctt.repair import EmbeddingRepairOptimizer, RepairConfig

config = RepairConfig(
    max_outer_iters=3,      # Outer loops (neighbor recomputation)
    max_inner_steps=50,     # Inner gradient steps
    learning_rate=0.1,
    lambda_anchor=0.1,      # Anchor loss weight
    lambda_nn_preserve=0.1, # Neighbor preservation weight
    delta_max=0.2           # Max movement from original
)

optimizer = EmbeddingRepairOptimizer(config)
result = optimizer.repair(embedding, index, token_id)
```

### Cluster-Level (Recommended)

```python
from dctt.repair.cluster import PathologicalClusterDetector
from dctt.repair.cluster_optimizer import ClusterRepairOptimizer

# Detect clusters
detector = PathologicalClusterDetector(
    mutual_k=50,           # k for mutual kNN graph
    min_cluster_size=2     # Minimum tokens per cluster
)
clusters = detector.detect(token_ids, embeddings, index)

# Repair clusters
optimizer = ClusterRepairOptimizer(
    max_outer_iters=3,
    max_inner_steps=50,
    learning_rate=0.05,
    lambda_anchor=0.5,
    delta_max=0.15
)
result = optimizer.repair_cluster(cluster, embeddings, index)
```

## Evaluation

```python
from dctt.evaluation.predictive import (
    compute_model_comparison,
    compute_feature_ablation,
    compute_bucket_analysis
)

# Model comparison with bootstrap CIs
comparison = compute_model_comparison(
    metrics_df,
    failure_rates,
    n_bootstrap=1000
)

# Feature ablation
ablation = compute_feature_ablation(metrics_df, failure_rates)

# Within-bucket analysis
bucket_analysis = compute_bucket_analysis(metrics_df, failure_rates)
```

## Common Gotchas

1. **index.query() needs 2D input** - Use `embeddings[ids]` not `embeddings[id]`
2. **SeverityScorer needs fit() first** - Call `fit()` before `compute_severity()`
3. **Class is EmbeddingRepairOptimizer** - Not `RepairOptimizer`
4. **Config uses max_outer_iters** - Not `outer_iters` or `n_outer`
