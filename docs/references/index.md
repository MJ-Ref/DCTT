# Reference Documentation

Quick lookup for implementation details and experimental results.

## Documents

| Document | Purpose |
|----------|---------|
| [API Patterns](api-patterns.md) | Current function signatures, common gotchas |
| [Key Metrics](key-metrics.md) | Reference numbers from all experiments |

## Quick Reference

### Most Common API Calls

```python
# Index
index = USearchIndex(metric="ip", connectivity=16, expansion_add=128, expansion_search=64)
distances, indices = index.query(embeddings_2d, k=50, exclude_self=True)

# Metrics
stage1 = compute_stage1_metrics(distances)
stage2 = compute_stage2_metrics(embeddings, indices)

# Severity
scorer = SeverityScorer()
scorer.fit(metrics_df)
severity = scorer.compute_severity(metrics_df)
```

### Key Numbers

- Strict predictive gate verdict: **FAIL** (20 runs, 4 models)
- Pooled geometry-minus-baseline delta: **-0.128** (95% CI [-0.168, -0.088])
- Pooled full-minus-baseline delta: **-0.012** (95% CI [-0.023, -0.000])
- Cond reduction: **0.427 ± 0.157**
- Jaccard overlap: **0.836 ± 0.030**
- Treatment effect: **-0.269** (vs control +0.036)
