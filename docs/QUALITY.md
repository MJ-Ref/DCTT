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
| Stress tests | B | Forced-token minimal-pair design implemented |
| Predictive validity | B- | Strict multi-seed real-label sweeps negative vs confound baselines |
| Causal framework | B | Mechanistic OK, behavioral incomplete |
| Documentation | B+ | Good coverage, could use more examples |
| Test coverage | A- | 66 tests including integration smoke coverage |

## What's Working Well

### Diagnostics Pipeline
- Full census completes in 7.5 minutes on 152k tokens
- Metrics are stable across k values
- Severity scoring handles frequency/type confounds
- Predictive-validity pipeline is reproducible end-to-end

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

### Predictive Validity Signal (Priority: High)
**Problem:** In strict real-label runs, geometry-only underperforms confound baselines across both tested models
**Impact:** RQ1 claim is not yet publication-strength
**Solution:** Run a gated rescue sprint (higher power + calibrated scoring) and pivot claim language if gate fails

### Causal Behavioral Evidence (Priority: High)
**Problem:** DiD not significant, outcomes simulated
**Impact:** Can't claim repair improves performance
**Solution:** Real model inference with embedding injection

### Multi-Model Validation (Priority: Medium)
**Problem:** Replicated across seeds on two Qwen models, but no cross-family validation
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
| Protocol lock is output-local unless versioned in tracked config | outputs/sweeps/* | Medium |

## Next Quality Goals

1. Execute predictive rescue gate (`configs/experiment/predictive_rescue.yaml` + `scripts/evaluate_predictive_gate.py`)
2. Add cross-family replication (Llama/Mistral class models)
3. Tune stress-test scoring with model-specific forced-choice/logprob margins
4. Add type hints to repair module and remove remaining hardcoded paths
