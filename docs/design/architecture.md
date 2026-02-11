# DCTT Architecture

## Overview

DCTT diagnoses token-level embedding geometry pathologies in LLMs through a staged pipeline.

## Pipeline Stages

```
Stage 0: Embedding Extraction
    ↓
Stage 1: Fast Screening (kNN distances)
    ↓
Stage 2: Spectral Geometry (displacement covariance)
    ↓
Stage 3: TDA Metrics (optional, persistent homology)
    ↓
Severity Scoring (robust z-scores within buckets)
    ↓
Repair (cluster-level optimization)
```

## Module Structure

```
src/dctt/
├── core/           # Types, registry, exceptions
│   ├── types.py         # TokenMetrics, RepairResult, etc.
│   ├── registry.py      # Component registration
│   └── exceptions.py    # Custom exceptions
│
├── embeddings/     # Extraction and normalization
│   ├── extract.py       # Model embedding extraction
│   ├── normalize.py     # L2 normalization
│   └── cache.py         # Embedding caching
│
├── neighbors/      # kNN index
│   ├── usearch_index.py # USearch HNSW implementation
│   └── query.py         # Neighbor queries
│
├── metrics/        # Diagnostic metrics
│   ├── stage1.py        # Mean/median distance, spread
│   ├── stage2.py        # PR, cond, logdet, dim95
│   ├── stage3.py        # TDA metrics (optional)
│   └── severity.py      # Bucketed z-score severity
│
├── repair/         # Embedding repair
│   ├── optimizer.py     # Single-token repair (negative result)
│   ├── cluster.py       # Cluster detection
│   ├── cluster_optimizer.py  # Cluster repair (positive result)
│   └── losses.py        # Geometry and anchor losses
│
├── stress_tests/   # Token stress tests
│   ├── code_syntax.py   # Code parsing tests
│   └── math_format.py   # Math formatting tests
│
├── evaluation/     # Statistical analysis
│   ├── predictive.py    # Predictive validity
│   └── matched_controls.py  # Confound matching
│
└── tracking/       # Experiment tracking
    ├── wandb_utils.py   # W&B integration
    └── artifacts.py     # Artifact management
```

## Key Design Decisions

### 1. Inner Product Metric
USearch uses `metric="ip"` (inner product). For L2-normalized vectors, this equals cosine similarity.

### 2. k×k Gram Matrix Optimization
Instead of computing d×d covariance (expensive for d=3584), we compute k×k Gram matrix:
- `G = X @ X.T` where X is k×d displacements
- Eigenvalues of G match non-zero eigenvalues of X.T @ X
- 50,000x speedup for typical k=50, d=3584

### 3. Tangent Space Projection
Gradient descent on unit sphere requires projecting gradients to tangent space:
```python
grad_tangent = grad - (grad @ embedding) * embedding
```

### 4. Cluster-Level Repair
Single-token repair failed due to centered covariance. See `docs/design/core-insights.md`.

## Configuration

Hydra-based configuration in `configs/`:
- `config.yaml` - Root config with defaults
- `model/*.yaml` - Model-specific settings
- `compute/*.yaml` - Hardware settings
- `experiment/*.yaml` - Experiment configs
