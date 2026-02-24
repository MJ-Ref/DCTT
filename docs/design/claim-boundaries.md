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

---

## NOT Supported Claims ❌

### 1. "Geometry metrics predict token-level failures beyond frequency/type confounds"

**Why not:**
- Multi-seed real-label sweep on qwen2_5_coder_7b: delta mean -0.005 (1/2 runs positive)
- Multi-seed real-label sweep on qwen2_5_7b: delta mean -0.111 (1/2 runs positive)
- Geometry-only performance is unstable and negative on average

**What's needed:**
- Larger real-label sample sizes and stress-test budgets
- Cross-seed replication
- Stronger token-failure labels (e.g., forced-choice/logprob margins)

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
- Replication currently covers only two Qwen-family models
- No Llama, Mistral, or other model comparisons

**What's needed:**
- Run census on 2-3 additional models
- Compare flagged token distributions
- Test if same tokens flagged across models

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
- "Real-label predictive-validity runs are currently negative for geometry-only models"
- "Cluster repair improves geometry vs placebo"
- "Single-token repair is insufficient for uniform pathologies"

**Avoid this language:**
- "Repair fixes token failures" (behavioral not shown)
- "Geometry causes failures" (correlation, not demonstrated causation)
- "This works for all LLMs" (only tested one model)
