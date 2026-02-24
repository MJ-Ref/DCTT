# Claim Boundaries

What DCTT can and cannot claim based on current evidence.

## Supported Claims ✅

### 1. "Cluster-level repair improves local geometry relative to placebo"

**Evidence:**
- Treatment condition number change: -0.269
- Control condition number change: +0.036
- 5/5 clusters improved (100%)
- Semantic preservation: Jaccard = 0.836

**Strength:** Strong for mechanistic claim. Treatment moves geometry in intended direction.

### 2. "Single-token local optimization is insufficient when neighborhoods are uniformly pathological"

**Evidence:**
- Single-token repair: geometry unchanged despite embeddings moving
- Mathematical explanation: centered covariance makes gradient zero
- Cluster repair fixes this by moving multiple tokens

**Strength:** Strong. Both empirical and theoretical support.

### 3. "Under strict controls, geometry-only predictive signal is negative across tested models"

**Evidence:**
- Final strict package: 20 runs across qwen2_5_coder_7b, qwen2_5_7b, mistral_7b, tinyllama_1_1b
- Pooled geometry-minus-baseline delta: -0.128 (95% CI [-0.168, -0.088])
- Positive geometry-minus-baseline runs: 1/20
- Each sweep-level gate verdict: FAIL

**Strength:** Strong for a constrained negative predictive claim on the tested setup.

---

## NOT Supported Claims ❌

### 1. "Geometry metrics predict token-level failures beyond frequency/type confounds"

**Why not:**
- Final strict package (20 runs) remains negative:
- qwen2_5_coder_7b: delta mean -0.211 (0/5 positive)
- qwen2_5_7b: delta mean -0.164 (0/5 positive)
- mistral_7b: delta mean -0.074 (0/5 positive)
- tinyllama_1_1b: delta mean -0.063 (1/5 positive)
- Pooled geometry-minus-baseline CI excludes zero on the negative side

**What's needed:**
- A different predictive endpoint definition (label redesign), not more runs of the same setup
- Pre-registered alternative features and ablations for conditional/interaction effects
- Clear separation of exploratory vs confirmatory predictive claims

### 2. "Repair causally improves downstream code/math behavior"

**Why not:**
- DiD (difference-in-differences) not significant (p = 0.81)
- DiD has wrong sign (+0.02 instead of negative)
- ATE = 0.179 reflects baseline severity difference, not treatment effect
- Outcomes are simulated, not from real model inference

**What's needed:**
- Real stress tests with model inference
- Embedding injection into model forward pass
- Larger sample (n=5 clusters insufficient)
- Better matching on continuous confounds

### 3. "Pathologies are consistent across model families"

**Why not:**
- Current strict replication includes four models but only one endpoint family
- Cross-family consistency is established for negative predictive signal, not for all pathology taxonomies

**What's needed:**
- Model-family expansion beyond current tested set
- Cross-model token taxonomy alignment and shared-flag analyses
- Additional tasks/endpoints to test transportability

### 4. "Geometry metrics are the best predictors of token failures"

**Why not:**
- Haven't compared against all alternatives
- TDA metrics (Stage 3) not fully evaluated
- Other spectral measures might work better

**What's needed:**
- Comprehensive feature comparison
- Information gain analysis
- Possibly better metrics exist

---

## Claim Language Guide

**Use this language:**
- "Strict real-label predictive-validity runs are negative for geometry-only models"
- "Cluster repair improves geometry vs placebo"
- "Single-token repair is insufficient for uniform pathologies"
- "This paper reports a rigorous negative predictive finding with mechanistic positive evidence"

**Avoid this language:**
- "Repair fixes token failures" (behavioral not shown)
- "Geometry causes failures" (correlation, not demonstrated causation)
- "This works for all LLMs" (only tested one model)

## Hard Pivot Manuscript Plan

Given finalized strict negative gates, use:
- `docs/design/predictive_negative_reframing.md`
- `docs/design/hard_pivot_publication_strategy.md`
- `outputs/sweeps/predictive_validity/HARD_PIVOT_REPORT.md`
