# NeurIPS Action Plan

**Status:** COMPLETE (Steps 1-5)
**Source:** Mock review feedback in `Feedback.md`
**Completed:** 2026-01-24

---

## Overview

This plan addressed major issues from a mock NeurIPS review that rated the initial submission as "weak reject / borderline."

## Steps Completed

### Step 1: Make End-to-End Story Runnable ✅

**Goal:** One command produces census + stress tests + predictive validity + causal comparison

**Changes:**
- Fixed `compute_stage1_metrics` and `compute_stage2_metrics` API calls
- Fixed `USearchIndex` constructor parameters
- Fixed `index.query` to use 2D input and `exclude_self`
- Fixed `SeverityScorer` usage pattern
- Fixed `EmbeddingRepairOptimizer` class name
- All experiment scripts synced with current APIs

### Step 2: Prove RQ1 - Diagnostic Validity ✅

**Goal:** Show incremental predictive power of geometry metrics

**Deliverables:**
- Predictive validity analysis with bootstrap CIs
- Model comparison: Baseline AUC 0.53 → Geometry AUC 0.80
- Feature ablation showing severity and logdet as top features
- Within-bucket analysis confirming geometry predicts in all strata

**Key Finding:** Geometry alone matches full model - confounds add nothing beyond geometry.

### Step 3: Redesign Repair for Cluster-Level ✅

**Goal:** Fix the gradient-free optimization problem

**Implementation:**
- `src/dctt/repair/cluster.py` - Mutual kNN graph clustering
- `src/dctt/repair/cluster_optimizer.py` - Joint optimization
- `experiments/run_cluster_repair.py`

**Results:**
- 69 clusters detected
- 5/5 clusters improved (100%)
- Condition reduction: 0.427 ± 0.157
- Jaccard: 0.836 (excellent semantic preservation)

**Key Finding:** Cluster-level repair WORKS - centered covariance changes when multiple tokens move.

### Step 4: Causal Experiment ✅

**Goal:** Compare cluster repair to placebo control

**Implementation:**
- `experiments/run_causal_cluster_repair.py`
- Treatment: top-severity clusters with real repair
- Control: low-severity tokens with placebo (random perturbation)
- Bootstrap CIs for ATE and DiD

**Results:**
- Treatment cond change: -0.269
- Control cond change: +0.036
- Mechanistic claim supported
- Behavioral claim NOT supported (DiD p=0.81)

### Step 5: Paper Packaging ✅

**Goal:** Generate publication-ready figures and tables

**Implementation:**
- `scripts/generate_paper_figures.py`

**Artifacts:**
- `fig1_predictive_validity.pdf/png`
- `fig2_cluster_repair.pdf/png`
- `fig3_causal_geometry.pdf/png`
- `table1_main_results.txt`
- `table2_flagged_tokens.txt`

---

## Updated Assessment

**Before:** Weak reject / borderline
**After:** Borderline (mechanistic claim supported, behavioral needs work)

## Remaining Work (Not in Original Plan)

1. **Multi-model comparison** - Llama, Mistral
2. **Real stress tests** - Actual model inference
3. **Paper writing** - Assemble into submission

---

## Decision Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-01-23 | Use cluster repair over single-token | Mathematical impossibility of single-token gradient |
| 2026-01-24 | Accept mechanistic-only claim | DiD not significant for behavioral |
| 2026-01-24 | Skip real stress tests for now | Requires embedding injection engineering |
| 2026-01-24 | Proceed to paper packaging | All diagnostics complete |
