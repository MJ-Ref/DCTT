# CLAUDE.md - Project Guidelines for Claude Code

## Project Overview

DCTT (Discrete-to-Continuous Transition Testing) is a research framework for diagnosing token-level embedding geometry pathologies in LLMs. The core contribution is diagnostic + causal validation; repairs are exploratory.

## Key Guidelines

### Documentation Verification (Critical)
After completing any milestone or feature implementation, always verify that related documentation has been updated before marking the task complete:
- `README.md` - Project status, experimental results, code examples
- `Feedback.md` - Mock review status and addressed issues
- Plan file - Step completion status and next priorities
- `SESSION_HANDOFF.md` - Current state for session continuity

### Gradient and Optimization Code
When implementing optimization algorithms or gradient-based code:
1. Verify that normalization operations don't cancel gradient updates
2. Verify that centering operations preserve necessary dependencies
3. **Critical insight:** Centered covariance makes single-token optimization gradient-free when neighbors are fixed - use cluster-level repair instead

### Multi-File Changes
For multi-file changes, create a mental checklist of affected files at the start and verify each is complete before finishing the task.

## API Patterns (Current)

```python
# USearch index
from dctt.neighbors import USearchIndex
index = USearchIndex(metric="ip", connectivity=16, expansion_add=128, expansion_search=64)
index.add(embeddings)
distances, indices = index.query(embeddings_2d, k=50, exclude_self=True)

# Metrics
from dctt.metrics import compute_stage1_metrics, compute_stage2_metrics
stage1 = compute_stage1_metrics(distances)
stage2 = compute_stage2_metrics(embeddings, indices)

# Severity scoring
from dctt.metrics import SeverityScorer
scorer = SeverityScorer()
scorer.fit(metrics_df)
severity = scorer.compute_severity(metrics_df)

# Repair
from dctt.repair import EmbeddingRepairOptimizer, RepairConfig
config = RepairConfig(max_outer_iters=3, max_inner_steps=50, ...)
optimizer = EmbeddingRepairOptimizer(config)
result = optimizer.repair(embedding, index, token_id)
```

## Key Metrics (Reference Numbers)

| Metric | Value | Context |
|--------|-------|---------|
| Geometry AUC | 0.803 | vs Baseline 0.534 |
| Cond reduction | 0.427 ± 0.157 | Cluster repair |
| Jaccard overlap | 0.836 ± 0.030 | Semantic preservation |
| Treatment effect | -0.269 | vs Control +0.036 |

## Claim Boundaries

**Supported:**
- "Geometry metrics predict failures beyond confounds"
- "Cluster-level repair improves geometry vs placebo"

**NOT Supported:**
- "Repair causally improves behavior" (DiD not significant)

## Commands

```bash
# Run experiments
python experiments/run_census.py model=qwen2_5_coder_7b
python experiments/run_cluster_repair.py model=qwen2_5_coder_7b
python experiments/run_causal_cluster_repair.py model=qwen2_5_coder_7b

# Generate figures
python scripts/generate_paper_figures.py

# Testing
pytest tests/ -v
python scripts/verify_pipeline.py
```

## File Structure

```
src/dctt/           # Core library (48 modules)
experiments/        # Experiment scripts (8 files)
scripts/            # Utility scripts
configs/            # Hydra configuration
tests/              # Test suite (45 tests)
outputs/            # Generated artifacts
```
