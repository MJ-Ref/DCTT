# Session Handoff

**Last Updated:** 2026-01-24

---

## Current State

All 5 steps of the NeurIPS action plan are complete. See `docs/plans/completed/neurips-action-plan.md` for details.

### What's Done
- Steps 1-5 complete (API sync, predictive validity, cluster repair, causal experiment, paper figures)
- All figures generated in `outputs/figures/`
- Documentation restructured to follow Codex pattern

### What's Next
1. Multi-model comparison (Llama, Mistral)
2. Paper writing

---

## Quick Links

| For... | See |
|--------|-----|
| Key numbers | `docs/references/key-metrics.md` |
| API patterns | `docs/references/api-patterns.md` |
| Claim boundaries | `docs/design/claim-boundaries.md` |
| Architecture | `docs/design/architecture.md` |
| Quality status | `docs/QUALITY.md` |

---

## Critical Gotcha

**Centered covariance makes single-token optimization gradient-free.**

Full explanation: `docs/design/core-insights.md`

Short version: When computing geometry from centered displacements, moving one token while neighbors stay fixed gives zero gradient. Use cluster repair instead.

---

## Commands

```bash
# Core experiments
python experiments/run_census.py model=qwen2_5_coder_7b
python experiments/run_cluster_repair.py model=qwen2_5_coder_7b
python experiments/run_causal_cluster_repair.py model=qwen2_5_coder_7b

# Generate figures
python scripts/generate_paper_figures.py

# Tests
pytest tests/ -v
```

---

## Git State

- Branch: `main`
- Status: Clean (after pending commit)
- Remote: `origin/main`
