# CLAUDE.md

DCTT: Discrete-to-Continuous Transition Testing for LLM embedding geometry.

## Quick Start

```bash
# Run experiments
python experiments/run_census.py model=qwen2_5_coder_7b
python experiments/run_cluster_repair.py model=qwen2_5_coder_7b
python scripts/generate_paper_figures.py

# Testing
pytest tests/ -v
```

## Documentation Map

| Need to know... | Go to |
|-----------------|-------|
| How the pipeline works | `docs/design/architecture.md` |
| Why single-token repair failed | `docs/design/core-insights.md` |
| What claims are supported | `docs/design/claim-boundaries.md` |
| Current API signatures | `docs/references/api-patterns.md` |
| Reference numbers/metrics | `docs/references/key-metrics.md` |
| What's working/broken | `docs/QUALITY.md` |
| Completed work | `docs/plans/completed/` |
| Session state | `SESSION_HANDOFF.md` |

## Critical Context

1. **Centered covariance = zero gradient for single token** → Use cluster repair
2. **Mechanistic claim supported** (geometry improves) / **Behavioral NOT supported** (DiD p=0.81)
3. **Key numbers:** Geometry AUC 0.80, Cond reduction 0.43, Jaccard 0.84

## File Locations

```
src/dctt/          # Core library
experiments/       # Experiment scripts
scripts/           # Utilities (figure generation, etc.)
configs/           # Hydra configuration
outputs/           # Generated artifacts
docs/              # Deep documentation
```

## Before Ending a Session

Update `SESSION_HANDOFF.md` with current state.
