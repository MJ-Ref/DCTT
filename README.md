# DCTT: Discrete-to-Continuous Transition Testing

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

**A rigorous framework for diagnosing token-level embedding geometry pathologies in Large Language Models.**

DCTT provides tools to identify tokens with problematic local geometry in LLM embedding spaces. The core contribution is a **diagnostic and causal validation framework**; repair methods are exploratory, with a current finding that single-token local optimization is insufficient when entire neighborhoods exhibit pathological geometry.

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

## Repair Methods

### Single-Token Repair (Negative Result)

The initial repair optimizer uses projected gradient descent with tangent space projection:

```python
for outer_iter in range(max_outer_iters):
    neighbors = knn(index, embedding, k)  # Recompute neighbors
    for step in range(inner_steps):
        loss = geometry_loss + λ_anchor * anchor_loss + λ_nn * nn_preserve_loss
        grad_tangent = gradient - (gradient · embedding) * embedding
        embedding = embedding - lr * grad_tangent
        embedding = project_to_constraints(embedding)  # Unit norm, max delta
```

**Finding:** Single-token local optimization preserves semantics (Jaccard > 0.7) but does NOT improve geometry metrics. This occurs because centered displacement covariance makes the loss independent of a single moving token when neighbors are fixed.

### Cluster-Level Repair (Positive Result) ✓

To address the single-token limitation, we implemented **cluster-level repair** that jointly optimizes connected components of pathological tokens:

```bash
python experiments/run_cluster_repair.py model=qwen2_5_coder_7b \
    cluster_repair.mutual_k=50 cluster_repair.min_cluster_size=2
```

**Algorithm:**
1. Build mutual k-NN graph on high-severity tokens
2. Find connected components (clusters)
3. Jointly optimize all tokens in each cluster
4. Centered covariance CAN change when multiple reference points shift together

**Results on Qwen2.5-Coder-7B:**

| Metric | Value | Status |
|--------|-------|--------|
| Clusters found | 69 | With mutual_k=50, min_size=2 |
| Clusters improved | **5/5 (100%)** | All clusters show improvement |
| Condition number reduction | **0.427 ± 0.157** | 10-17% improvement |
| Jaccard overlap | **0.836 ± 0.030** | Excellent semantic preservation |
| Similarity to original | **0.992** | Minimal movement required |

**Key Finding:** Cluster-level repair successfully improves geometry (condition number decreases) while preserving semantics, validating the hypothesis that pathological tokens need to move together.

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

## Experimental Results

### Qwen2.5-Coder-7B Analysis

Full vocabulary census on 152,064 tokens (3,584 dimensions):

| Metric | Result |
|--------|--------|
| Tokens analyzed | 152,064 |
| Flagged tokens (poor geometry) | 3,325 (2.19%) |
| Processing speed | ~330 tokens/sec |
| Total census time | 7.5 minutes |

**High-severity token outputs in the current census artifact:**
- The saved `diagnostic_results.json` currently uses placeholder token strings (`token_<id>`)
- Token type is `UNKNOWN` in that artifact
- Lexical examples should be treated as pending until census writes decoded tokenizer strings

### Single-Token Repair Validation

| Criterion | Status | Value |
|-----------|--------|-------|
| Embeddings moved | ✓ YES | similarity = 0.98 |
| Semantic preservation | ✓ PASS | Jaccard 0.75 - 1.0 |
| Geometry improved | ✗ NO | cond/PR unchanged |

**Finding:** Single-token repair preserves semantics but doesn't improve geometry when neighborhoods are uniformly pathological.

### Cluster-Level Repair Validation ✓

| Criterion | Status | Value |
|-----------|--------|-------|
| Clusters detected | ✓ YES | 69 clusters found |
| Geometry improved | **✓ YES** | cond reduced 0.43 ± 0.16 |
| Semantic preservation | ✓ PASS | Jaccard = 0.836 |
| Improvement rate | ✓ 100% | 5/5 clusters improved |

### Causal Experiment Results

| Claim | Status | Evidence |
|-------|--------|----------|
| Geometry improves vs placebo | **✓ Supported** | Treatment cond -0.27 vs control +0.04 |
| Behavior improves causally | ✗ Not yet | DiD not significant (p=0.81), simulated outcomes |

**Supported Claim:** "Cluster-level repair improves local geometry (cond) relative to placebo with minimal embedding movement."

**Not Yet Supported:** "Repair causally improves downstream behavior." Requires real stress tests with model inference, better matching, and larger samples.

### Predictive Validity Artifact Status

Final strict real-label package (forced-token minimal-pair, `logprob_choice`, no proxy confounds):
- Sweeps: `2026-02-24_06-43-58`, `2026-02-24_07-31-10`, `2026-02-24_07-34-45`
- Total runs: `20` (4 models x 5 seeds/model)
- Strict gate verdict: `FAIL`
- Pooled geometry-minus-baseline delta: `-0.128153` (95% CI `[-0.168367, -0.087938]`, positive `1/20`)
- Pooled full-minus-baseline delta: `-0.011735` (95% CI `[-0.023119, -0.000351]`, positive `4/20`)

Per-model geometry-minus-baseline delta means:
- `qwen2_5_coder_7b`: `-0.211062` (5 seeds, positive `0/5`)
- `qwen2_5_7b`: `-0.164317` (5 seeds, positive `0/5`)
- `mistral_7b`: `-0.073882` (5 seeds, positive `0/5`)
- `tinyllama_1_1b`: `-0.063350` (5 seeds, positive `1/5`)

Current interpretation: under strict controls, geometry-only predictive signal is consistently negative versus confound baselines. The predictive claim is retired; the project pivots to mechanistic intervention + rigorous negative predictive evidence.

Reproduce with:

```bash
# Build/update counts vector (one-time per tokenizer/corpus).
# For Qwen strict runs, align to model output vocab size:
python experiments/build_token_frequency_counts.py \
  --model-name Qwen/Qwen2.5-Coder-7B \
  --input-root /path/to/corpus \
  --output configs/confounds/qwen2_5_coder_7b_repo_counts_aligned.npy \
  --target-vocab-size 152064

python experiments/run_predictive_validity_sweep.py \
  --models qwen2_5_coder_7b,qwen2_5_7b \
  --seeds 70,71,72 \
  --sample-size 100 \
  --n-prompts 2 \
  --max-new-tokens 8 \
  --n-bootstrap 30 \
  --scoring-mode logprob_choice \
  --compute-device cuda \
  --frequency-counts-path configs/confounds/qwen2_5_coder_7b_repo_counts_aligned.npy \
  --fail-on-proxy-confounds

# Gate decision (PASS/FAIL):
python scripts/evaluate_predictive_gate.py \
  --sweep-results outputs/sweeps/predictive_validity/<run_stamp>/sweep_results.json \
  --output-json outputs/sweeps/predictive_validity/<run_stamp>/gate_evaluation.json \
  --output-markdown outputs/sweeps/predictive_validity/<run_stamp>/gate_evaluation.md

# Full-power cross-family rescue launcher (5 seeds/model, per-model confound files):
python scripts/launch_cross_family_rescue.py \
  --config configs/experiment/cross_family_rescue.yaml

# Wait/pull/finalize modal sweep artifacts once a stamp is known:
python scripts/finalize_modal_predictive_sweep.py \
  --stamp <run_stamp> \
  --wait \
  --min-runs-per-model 5

# Build consolidated hard-pivot evidence report from finalized sweeps:
python scripts/build_hard_pivot_report.py
```

Final pivot artifacts:
- `outputs/sweeps/predictive_validity/HARD_PIVOT_REPORT.json`
- `outputs/sweeps/predictive_validity/HARD_PIVOT_REPORT.md`
- `docs/design/predictive_negative_reframing.md`
- `docs/design/hard_pivot_publication_strategy.md`

Camera-ready figures/tables (locked and reproducible):

```bash
# Uses repo-tracked lock + deterministic diagram spec + manifest hashes
python scripts/generate_paper_figures.py \
  --paper-lock configs/paper/publication_assets_lock.yaml \
  --pipeline-spec figures_src/pipeline_diagram_spec.yaml \
  --strict-lock
```

Generated publication assets include:
- `outputs/figures/fig0_pipeline_diagram.svg`
- `outputs/figures/fig0_pipeline_diagram.pdf`
- `outputs/figures/fig1_predictive_validity.pdf`
- `outputs/figures/fig2_cluster_repair.pdf`
- `outputs/figures/fig3_causal_geometry.pdf`
- `outputs/figures/fig4_model_replication.pdf`
- `outputs/figures/table1_main_results.txt`
- `outputs/figures/table2_flagged_tokens.txt`
- `outputs/figures/table3_model_replication.txt`
- `outputs/figures/PUBLICATION_MANIFEST.json`
- `outputs/figures/PUBLICATION_MANIFEST.md`

## Project Status

This is a research codebase under active development. Current status:

- [x] Core types and abstractions
- [x] Stage 1 & 2 metrics
- [x] USearch index integration
- [x] Single-token repair optimizer
- [x] Stress test framework
- [x] Statistical evaluation
- [x] W&B integration
- [x] Benchmark wrappers (HumanEval, GSM8k)
- [x] Stage 3 TDA metrics
- [x] Paper figures generation
- [x] Full census on Qwen2.5-Coder-7B
- [x] Single-token repair validation (negative result)
- [x] **Cluster-level repair** (positive result - geometry improves!)
- [x] Forced-token minimal-pair stress tests
- [x] Predictive-validity analysis pipeline (real-label runs complete)
- [x] **Causal experiment framework** (mechanistic claim validated)
- [x] Cross-family replication (Mistral, TinyLlama; strict negative)
- [x] Full-power cross-family rescue sweep (5 seeds/model; strict negative)
- [x] Hard pivot evidence report (20-run strict aggregate)
- [ ] Causal behavioral evidence (needs real stress tests)

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
