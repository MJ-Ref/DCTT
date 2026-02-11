# Key Metrics

Reference numbers from DCTT experiments. Use these for consistency across documentation.

## Model: Qwen2.5-Coder-7B

### Census Results

| Metric | Value |
|--------|-------|
| Total tokens | 152,064 |
| Embedding dimensions | 3,584 |
| Flagged tokens | 3,325 (2.19%) |
| Processing speed | ~330 tokens/sec |
| Census time | 7.5 minutes |

### Predictive Validity (RQ1)

| Model | AUC | 95% CI |
|-------|-----|--------|
| Baseline (freq + type) | 0.534 | [0.48, 0.59] |
| Geometry only | 0.803 | [0.76, 0.85] |
| Full model | 0.803 | [0.76, 0.85] |
| **Improvement** | **+0.269** | |

Top features by importance:
1. severity (0.15)
2. logdet (0.12)
3. cond (0.08)
4. pr (0.06)

### Cluster Repair (RQ2 - Mechanistic)

| Metric | Value |
|--------|-------|
| Clusters found | 69 |
| Clusters repaired | 5 |
| Improvement rate | 100% (5/5) |
| Condition reduction | 0.427 ± 0.157 |
| PR change | -0.335 ± 0.851 |
| Jaccard overlap | 0.836 ± 0.030 |
| Similarity to original | 0.992 |

Per-cluster results:

| Cluster | Tokens | Cond Change | Jaccard |
|---------|--------|-------------|---------|
| 22 | 3 | 3.94 → 3.26 (-17%) | 0.819 |
| 34 | 2 | 4.77 → 4.52 (-5%) | 0.818 |
| 35 | 2 | 3.58 → 3.30 (-8%) | 0.803 |
| 23 | 3 | 2.96 → 2.52 (-15%) | 0.887 |
| 24 | 3 | 4.25 → 3.76 (-12%) | 0.852 |

### Causal Experiment

| Metric | Treatment | Control |
|--------|-----------|---------|
| n (tokens) | 13 | 13 |
| n (clusters) | 5 | - |
| Cond before | 3.90 | 2.85 |
| Cond after | 3.63 | 2.89 |
| **Cond change** | **-0.269** | **+0.036** |
| Failure before | 0.378 | 0.218 |
| Failure after | 0.372 | 0.193 |

Statistical tests:
- ATE: 0.179 (not meaningful - reflects baseline difference)
- DiD: +0.022, p = 0.81 (not significant)

### High-Severity Token Examples

| Rank | Token | Category | Severity |
|------|-------|----------|----------|
| 1 | `))):\n` | Nested punctuation | 3.54 |
| 2 | ` ...\\` | Escape fragment | 3.50 |
| 3 | `"For` | Quote + word | 3.44 |
| 4 | `("` | Bracket + quote | 3.37 |
| 5 | `...'` | Ellipsis + CJK | 3.34 |
| 6 | `'` | CJK punctuation | 3.31 |
| 7 | `0` | Full-width digit | 3.30 |
| 8 | `),\r\n` | Bracket + CRLF | 3.27 |
| 9 | `',\r\r\n` | Quote + double CRLF | 3.27 |
| 10 | `))))\n\n` | Deep nesting | 3.26 |

## Metric Definitions

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| PR | (Σλ)² / Σ(λ²) | Effective dimensionality |
| cond | (λ₁ + ε) / (λₘ + ε) | Conditioning (lower = better) |
| logdet | Σ log(λᵢ + ε) | Local volume |
| dim95 | min k: Σλ[:k] ≥ 0.95 Σλ | Dimensions for 95% variance |
| severity | mean(z-scores) | Aggregated pathology |
