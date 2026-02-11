# Quality Assessment

Current state of DCTT components and what needs improvement.

## Component Grades

| Component | Grade | Notes |
|-----------|-------|-------|
| Core types | A | Stable, well-tested |
| Embedding extraction | A | Works with multiple models |
| USearch index | A | Fast, M3-optimized |
| Stage 1 metrics | A | Simple, reliable |
| Stage 2 metrics | A | Core contribution, validated |
| Severity scoring | A | Bucketed z-scores working |
| Single-token repair | C | Works but doesn't improve geometry |
| Cluster repair | A | Key breakthrough, validated |
| Stress tests | C | Prompts don't isolate tokens |
| Predictive validity | A | Strong results with CIs |
| Causal framework | B | Mechanistic OK, behavioral incomplete |
| Documentation | B+ | Good coverage, could use more examples |
| Test coverage | B | 45 tests, some gaps in repair |

## What's Working Well

### Diagnostics Pipeline
- Full census completes in 7.5 minutes on 152k tokens
- Metrics are stable across k values
- Severity scoring handles frequency/type confounds
- Predictive validity shows clear signal

### Cluster Repair
- 100% improvement rate on tested clusters
- High semantic preservation (Jaccard 0.84)
- Minimal embedding movement required
- Addresses core mathematical limitation

### Reproducibility
- Hydra configs capture all parameters
- W&B tracking enabled
- Seeds locked throughout
- Results files include full metadata

## What Needs Work

### Stress Tests (Priority: High)
**Problem:** Current prompts don't force target token inclusion
**Impact:** Can't make behavioral causal claims
**Solution:** Forced-token decoding or minimal pairs

### Causal Behavioral Evidence (Priority: High)
**Problem:** DiD not significant, outcomes simulated
**Impact:** Can't claim repair improves performance
**Solution:** Real model inference with embedding injection

### Multi-Model Validation (Priority: Medium)
**Problem:** Only tested on Qwen2.5-Coder-7B
**Impact:** Results may not generalize
**Solution:** Run on Llama, Mistral

### Confound Matching (Priority: Medium)
**Problem:** Matching on (tier, type) only, not continuous
**Impact:** Residual confounding possible
**Solution:** Propensity score matching

### Stage 3 TDA (Priority: Low)
**Problem:** Implemented but not validated
**Impact:** Missing potentially better metrics
**Solution:** Full evaluation against Stage 2

## Technical Debt

| Issue | Location | Severity |
|-------|----------|----------|
| Hardcoded paths in some scripts | experiments/*.py | Low |
| Missing type hints in repair | src/dctt/repair/ | Low |
| Duplicate metric computation | census vs validation | Low |
| No integration tests | tests/ | Medium |

## Next Quality Goals

1. Add integration test for full pipeline
2. Implement forced-token stress tests
3. Add type hints to repair module
4. Remove hardcoded paths
