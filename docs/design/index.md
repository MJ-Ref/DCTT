# Design Documentation

Architectural decisions and technical insights for DCTT.

## Documents

| Document | Purpose |
|----------|---------|
| [Architecture](architecture.md) | Pipeline stages, module structure, key design decisions |
| [Core Insights](core-insights.md) | Critical discoveries made during development |
| [Claim Boundaries](claim-boundaries.md) | What the research can and cannot claim |

## Key Takeaways

1. **Staged pipeline** - Fast screening (Stage 1) → Spectral geometry (Stage 2) → Severity scoring
2. **k×k Gram trick** - 50,000x speedup over d×d covariance
3. **Centered covariance insight** - Explains why single-token repair fails
4. **Cluster repair solution** - Move multiple tokens together to change geometry
