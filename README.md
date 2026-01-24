# DCTT: Discrete-to-Continuous Transition Testing

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

**A rigorous framework for diagnosing and repairing token-level embedding geometry pathologies in Large Language Models.**

DCTT provides tools to identify tokens with problematic local geometry in LLM embedding spaces and apply minimal, targeted repairs that improve downstream performance on code and math tasks.

## Key Features

- **Multi-stage diagnostics**: Fast screening (Stage 1) followed by detailed spectral analysis (Stage 2)
- **Spectral geometry metrics**: Participation ratio, condition number, effective dimension, log-determinant
- **Constrained repair optimization**: Fix geometry while preserving semantics
- **Causal validation**: Stress tests and matched control experiments
- **Apple Silicon optimized**: USearch HNSW with ARM NEON acceleration
- **Reproducibility first**: W&B tracking, config snapshots, seed management

## Installation

```bash
# Clone the repository
git clone https://github.com/MJ-Ref/DCTT.git
cd DCTT

# Install with development dependencies
pip install -e ".[dev]"

# For Apple Silicon with MLX support
pip install -e ".[dev,mlx]"

# For cloud GPU with Modal
pip install -e ".[dev,modal]"
```

### Requirements

- Python 3.11+
- PyTorch 2.3+
- 16GB+ RAM (96GB recommended for full vocabulary analysis)

## Quick Start

### 1. Extract Embeddings

```bash
# Extract and normalize embeddings from a model
dctt extract --model Qwen/Qwen2.5-Coder-7B --output outputs/embeddings.npy
```

### 2. Run Diagnostic Census

```bash
# Analyze geometry for all tokens
python experiments/run_census.py model=qwen2_5_coder_7b

# Or sample 1000 tokens for quick analysis
python experiments/run_census.py model=qwen2_5_coder_7b \
    experiment.tokens.mode=sample \
    experiment.tokens.sample_size=1000
```

### 3. Run Causal Repair Experiment

```bash
# Repair high-severity tokens and compare to matched controls
python experiments/run_causal_repair.py model=qwen2_5_coder_7b
```

## Architecture

```
dctt/
├── core/           # Types, exceptions, registry pattern
├── embeddings/     # Extraction, normalization, caching
├── neighbors/      # USearch HNSW index (M3-optimized)
├── metrics/        # Stage 1 & 2 diagnostic metrics
├── repair/         # Constrained optimization repair
├── stress_tests/   # Code syntax, math formatting tests
├── evaluation/     # Statistical analysis, causal inference
└── tracking/       # W&B integration, reproducibility
```

## Diagnostic Pipeline

### Stage 1: Basic Local Outlier Checks
Fast, cheap metrics for initial screening:
- `μ_k`: Mean k-NN distance
- `med_k`: Median k-NN distance
- `spread_q`: Quantile spread ratio (q90/q10)

### Stage 2: Spectral Geometry Analysis
Core contribution - displacement matrix-based metrics:
- `dim95`: Effective dimension at 95% variance
- `PR`: Participation ratio (eigenvalue spread)
- `cond`: Local condition number
- `logdet`: Log-determinant of covariance
- `anisotropy`: Dominant direction ratio

### Severity Scoring
Robust z-scores within (frequency_tier, token_type) buckets ensure fair comparison across token categories.

## Configuration

DCTT uses [Hydra](https://hydra.cc/) for configuration management:

```bash
# Override model
python experiments/run_census.py model=llama3_8b

# Override compute settings
python experiments/run_census.py compute=modal_gpu

# Override multiple settings
python experiments/run_census.py \
    model=qwen2_5_coder_7b \
    neighbors.k=100 \
    seed=123
```

### Key Configuration Files

| File | Purpose |
|------|---------|
| `configs/config.yaml` | Root configuration |
| `configs/model/*.yaml` | Model-specific settings |
| `configs/compute/*.yaml` | Hardware optimization |
| `configs/pipeline/*.yaml` | Pipeline stages |
| `configs/experiment/*.yaml` | Experiment configurations |

## Metrics Reference

### Participation Ratio (PR)
```
PR = (Σλ)² / Σ(λ²)
```
Measures effective dimensionality. PR = d for uniform eigenvalues, PR ≈ 1 for single dominant direction.

### Condition Number
```
cond = (λ₁ + ε) / (λₘ + ε)
```
High condition numbers indicate ill-conditioned local geometry that may cause gradient instabilities.

### Log-Determinant
```
logdet = Σ log(λᵢ + ε)
```
Measures local "volume" or dispersion. Very negative values indicate collapsed geometry.

## Repair Method

The repair optimizer uses projected gradient descent:

```python
for outer_iter in range(max_outer_iters):
    neighbors = knn(index, embedding, k)  # Recompute neighbors

    for step in range(inner_steps):
        loss = geometry_loss + λ_anchor * anchor_loss + λ_nn * nn_preserve_loss
        embedding = embedding - lr * gradient(loss)
        embedding = project_to_constraints(embedding)  # Unit norm, max delta
```

Constraints ensure:
- Unit L2 norm (for cosine distance)
- Maximum change bounded by `delta_max`

## Development

```bash
# Run tests
make test

# Run fast tests only
make test-fast

# Lint and format
make format
make lint

# Type check
make typecheck

# Install pre-commit hooks
make pre-commit
```

## Project Status

This is a research codebase under active development. Current status:

- [x] Core types and abstractions
- [x] Stage 1 & 2 metrics
- [x] USearch index integration
- [x] Repair optimizer
- [x] Stress test framework
- [x] Statistical evaluation
- [x] W&B integration
- [x] Benchmark wrappers (HumanEval, GSM8k)
- [x] Stage 3 TDA metrics
- [x] Paper figures generation

## Citation

If you use DCTT in your research, please cite:

```bibtex
@software{dctt2024,
  title = {DCTT: Discrete-to-Continuous Transition Testing for LLM Embedding Geometry},
  year = {2024},
  url = {https://github.com/MJ-Ref/DCTT}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- [USearch](https://github.com/unum-cloud/usearch) for fast HNSW indexing
- [Weights & Biases](https://wandb.ai/) for experiment tracking
- [Hydra](https://hydra.cc/) for configuration management
