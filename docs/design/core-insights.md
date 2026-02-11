# Core Insights

Critical discoveries made during DCTT development.

## 1. Centered Covariance Makes Single-Token Optimization Gradient-Free

**This is the most important mathematical insight in the project.**

### The Problem

When computing local geometry metrics from neighbor displacements:

```python
displacements = neighbors - token        # Shape: (k, d)
centered = displacements - displacements.mean(axis=0)  # Centering
cov = centered.T @ centered / k          # Covariance
```

If you optimize only the token embedding while keeping neighbors fixed:
1. All displacement vectors shift by the same delta
2. Centering removes this uniform shift entirely
3. The covariance matrix is unchanged
4. **Gradient with respect to token = 0**

### Why This Matters

This explains why single-token repair "worked" (moved embeddings, preserved semantics) but didn't improve geometry. The optimizer was chasing a gradient that didn't exist for the metrics we cared about.

### The Solution: Cluster-Level Repair

When multiple tokens in a neighborhood move together:
- The centered covariance DOES change
- Different tokens shift by different amounts
- The mean shift doesn't cancel everything
- Optimization can now improve geometry

**Result:** Cluster repair achieves 10-17% condition number reduction where single-token repair achieved 0%.

## 2. Pathological Neighborhoods Are Uniform

High-severity tokens tend to cluster together. A token with bad geometry usually has neighbors with equally bad geometry.

**Implication:** You can't escape a bad neighborhood by moving slightly - you need to move into an entirely different region, or move the whole neighborhood together.

## 3. Geometry Predicts Beyond Confounds

Initial concern: Maybe severity just correlates with token frequency or type.

**Finding:** Geometry metrics (cond, PR, logdet) predict failures even after controlling for:
- Log frequency
- Token type (code, punctuation, etc.)
- Embedding norm
- Simple density (mean kNN distance)

AUC improvement: 0.80 (geometry) vs 0.53 (baseline confounds only)

## 4. Mechanistic vs Behavioral Evidence

Two different claims require different evidence:

| Claim Type | Evidence Required | DCTT Status |
|------------|------------------|-------------|
| Mechanistic | Intervention changes geometry as predicted | ✅ Supported |
| Behavioral | Intervention improves downstream performance | ❌ Not yet |

The causal experiment shows treatment (cluster repair) reduces condition number by 0.27 vs control (placebo) increasing by 0.04. This supports the mechanistic claim.

Behavioral evidence requires running actual model inference with repaired embeddings - not yet implemented.

## 5. Stress Tests Don't Isolate Tokens

Current stress test prompts don't force the target token to be the cause of failure. A prompt might fail for reasons unrelated to the specific token being tested.

**Needed:** Forced-token decoding, minimal pairs, or logprob margins on required tokens.
