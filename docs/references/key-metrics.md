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

### Predictive Validity (RQ1, Final Strict Package)

Strict package (forced-token minimal-pair, real labels, confound alignment):
- Total runs: 20 (4 models x 5 seeds/model)
- Gate verdict: FAIL

Pooled effects:

| Effect | Mean | 95% CI | Positive Runs |
|--------|------|--------|---------------|
| Geometry - Baseline | -0.128 | [-0.168, -0.088] | 1/20 |
| Full - Baseline | -0.012 | [-0.023, -0.000] | 4/20 |

Per-model geometry-minus-baseline means:

| Model | Mean | 95% CI | Positive Runs |
|-------|------|--------|---------------|
| qwen2_5_coder_7b | -0.211 | [-0.256, -0.166] | 0/5 |
| qwen2_5_7b | -0.164 | [-0.229, -0.099] | 0/5 |
| mistral_7b | -0.074 | [-0.168, +0.020] | 0/5 |
| tinyllama_1_1b | -0.063 | [-0.160, +0.034] | 1/5 |

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

### High-Severity Token Outputs

Current saved census artifact uses placeholder token strings (`token_<id>`) and `UNKNOWN` token type.
Use token IDs + severity for claim-bearing diagnostics until a decoded-token census artifact is produced.

## Metric Definitions

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| PR | (Σλ)² / Σ(λ²) | Effective dimensionality |
| cond | (λ₁ + ε) / (λₘ + ε) | Conditioning (lower = better) |
| logdet | Σ log(λᵢ + ε) | Local volume |
| dim95 | min k: Σλ[:k] ≥ 0.95 Σλ | Dimensions for 95% variance |
| severity | mean(z-scores) | Aggregated pathology |
