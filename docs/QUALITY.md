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
| Predictive validity | B- | Final strict 20-run package is negative vs confound baselines |
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
**Problem:** In the finalized strict package (20 runs, 4 models), geometry-only underperforms confound baselines
**Impact:** Positive predictive claim is falsified for the current endpoint/protocol
**Solution:** Hard-pivot manuscript framing to mechanistic-positive + rigorous negative predictive result

### Causal Behavioral Evidence (Priority: High)
**Problem:** DiD not significant, outcomes simulated
**Impact:** Can't claim repair improves performance
**Solution:** Real model inference with embedding injection

### Multi-Model Validation (Priority: Medium)
**Problem:** Cross-family 5-seed runs are complete and still negative
**Impact:** Negative predictive finding is robust across tested model families
**Solution:** Stop repeated runs of same endpoint; move effort to endpoint redesign if predictive line is revisited

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
| Legacy scripts still fall back to latest-run heuristics if lock not provided | scripts/generate_paper_figures.py | Medium |

## Next Quality Goals

1. Complete manuscript hard pivot using `outputs/sweeps/predictive_validity/HARD_PIVOT_REPORT.md`
2. Separate confirmatory negative results from exploratory future predictive work
3. Define next-generation predictive endpoint before any new GPU sweep
4. Add type hints to repair module and remove remaining hardcoded paths
5. Keep `configs/paper/publication_assets_lock.yaml` updated when claim-bearing artifacts change
