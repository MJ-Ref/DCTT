#!/usr/bin/env python3
"""Generate publication-quality plots for DCTT research.

This script produces figures for:
1. Pipeline diagram (Stage 0-3 + repair loop)
2. Metric distributions by bucket (cond, PR, dim95)
3. Severity vs failure rate scatter (with frequency controls)
4. Causal results (repaired vs matched control deltas with CI)
5. Compute scaling plot (time vs vocab size)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def setup_matplotlib():
    """Configure matplotlib for publication-quality figures."""
    import matplotlib.pyplot as plt

    # Use a clean style
    plt.style.use("seaborn-v0_8-whitegrid")

    # Configure for publication
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
    })

    return plt


def plot_metric_distributions(
    results_path: Path,
    output_dir: Path,
) -> None:
    """Plot metric distributions by frequency tier and token type.

    Args:
        results_path: Path to diagnostic results JSON.
        output_dir: Output directory for figures.
    """
    plt = setup_matplotlib()
    import matplotlib.patches as mpatches

    with open(results_path) as f:
        results = json.load(f)

    # Extract metrics by bucket
    buckets: dict[str, dict[str, list[float]]] = {}
    metrics = ["cond", "pr", "logdet", "dim95"]

    # Handle both list format and dict format with "tokens" key
    token_list = results if isinstance(results, list) else results.get("tokens", [])

    for token_result in token_list:
        # Get bucket - handle both tuple/list and string formats
        bucket_raw = token_result.get("bucket", ["UNKNOWN", "UNKNOWN"])
        if isinstance(bucket_raw, (list, tuple)):
            bucket = f"{bucket_raw[0]}_{bucket_raw[1]}"
        else:
            bucket = str(bucket_raw)

        if bucket not in buckets:
            buckets[bucket] = {m: [] for m in metrics}

        stage2 = token_result.get("stage2", {})
        for m in metrics:
            if m in stage2:
                buckets[bucket][m].append(stage2[m])

    if not buckets:
        print("No bucket data found in results")
        return

    # Create figure with subplots for each metric
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()

    colors = plt.cm.Set2(np.linspace(0, 1, len(buckets)))

    for ax, metric in zip(axes, metrics):
        data = []
        labels = []
        for bucket_name, bucket_data in sorted(buckets.items()):
            if bucket_data[metric]:
                data.append(bucket_data[metric])
                labels.append(bucket_name)

        if data:
            bp = ax.boxplot(data, labels=labels, patch_artist=True)
            for patch, color in zip(bp["boxes"], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

        ax.set_title(f"{metric.upper()} by Bucket")
        ax.set_xlabel("Bucket")
        ax.set_ylabel(metric.upper())
        ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    output_path = output_dir / "metric_distributions.png"
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def plot_severity_vs_failure(
    diagnostic_path: Path,
    stress_test_path: Path,
    output_dir: Path,
) -> None:
    """Plot severity score vs stress test failure rate.

    Args:
        diagnostic_path: Path to diagnostic results.
        stress_test_path: Path to stress test results.
        output_dir: Output directory.
    """
    plt = setup_matplotlib()

    with open(diagnostic_path) as f:
        diagnostics = json.load(f)

    with open(stress_test_path) as f:
        stress_tests = json.load(f)

    # Build token_id -> failure_rate map
    failure_rates = {}
    for result in stress_tests.get("results", []):
        token_id = result.get("token_id")
        if token_id is not None:
            failure_rates[token_id] = result.get("failure_rate", 0)

    # Collect severity vs failure data
    severities = []
    failures = []
    frequencies = []

    for token in diagnostics.get("tokens", []):
        token_id = token.get("token_id")
        severity = token.get("severity")
        freq = token.get("token_info", {}).get("frequency", 1)

        if token_id in failure_rates and severity is not None:
            severities.append(severity)
            failures.append(failure_rates[token_id])
            frequencies.append(np.log(freq + 1))

    if not severities:
        print("No matching severity/failure data found")
        return

    fig, ax = plt.subplots(figsize=(8, 6))

    scatter = ax.scatter(
        severities,
        failures,
        c=frequencies,
        cmap="viridis",
        alpha=0.6,
        s=20,
    )

    plt.colorbar(scatter, label="log(frequency + 1)")

    ax.set_xlabel("Severity Score")
    ax.set_ylabel("Failure Rate")
    ax.set_title("Severity vs Stress Test Failure Rate\n(colored by frequency)")

    # Add trend line
    z = np.polyfit(severities, failures, 1)
    p = np.poly1d(z)
    x_line = np.linspace(min(severities), max(severities), 100)
    ax.plot(x_line, p(x_line), "r--", alpha=0.8, label="Linear fit")
    ax.legend()

    output_path = output_dir / "severity_vs_failure.png"
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def plot_causal_effects(
    causal_results_path: Path,
    output_dir: Path,
) -> None:
    """Plot causal repair effects with confidence intervals.

    Args:
        causal_results_path: Path to causal analysis results.
        output_dir: Output directory.
    """
    plt = setup_matplotlib()

    with open(causal_results_path) as f:
        results = json.load(f)

    # Extract treatment and control deltas
    treatment_deltas = results.get("treatment_deltas", [])
    control_deltas = results.get("control_deltas", [])
    ate = results.get("ate", 0)
    ate_ci = results.get("ate_ci", [ate, ate])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Distribution comparison
    ax1 = axes[0]
    if treatment_deltas and control_deltas:
        ax1.hist(
            treatment_deltas,
            bins=30,
            alpha=0.6,
            label="Treatment (Repaired)",
            color="steelblue",
        )
        ax1.hist(
            control_deltas,
            bins=30,
            alpha=0.6,
            label="Control (Placebo)",
            color="coral",
        )
        ax1.axvline(
            np.mean(treatment_deltas),
            color="steelblue",
            linestyle="--",
            linewidth=2,
        )
        ax1.axvline(
            np.mean(control_deltas),
            color="coral",
            linestyle="--",
            linewidth=2,
        )

    ax1.set_xlabel("Change in Failure Rate")
    ax1.set_ylabel("Count")
    ax1.set_title("Distribution of Failure Rate Changes")
    ax1.legend()

    # Right: ATE with CI
    ax2 = axes[1]
    categories = ["Treatment\nEffect"]
    values = [ate]
    errors = [[ate - ate_ci[0]], [ate_ci[1] - ate]]

    bars = ax2.bar(categories, values, color="steelblue", alpha=0.7)
    ax2.errorbar(
        categories,
        values,
        yerr=errors,
        fmt="none",
        color="black",
        capsize=10,
        capthick=2,
    )
    ax2.axhline(0, color="gray", linestyle="--")
    ax2.set_ylabel("Average Treatment Effect (ATE)")
    ax2.set_title(f"Causal Effect of Repair\nATE = {ate:.4f} [{ate_ci[0]:.4f}, {ate_ci[1]:.4f}]")

    plt.tight_layout()
    output_path = output_dir / "causal_effects.png"
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def plot_compute_scaling(
    timing_results_path: Path,
    output_dir: Path,
) -> None:
    """Plot compute scaling with vocabulary size.

    Args:
        timing_results_path: Path to timing results.
        output_dir: Output directory.
    """
    plt = setup_matplotlib()

    with open(timing_results_path) as f:
        results = json.load(f)

    vocab_sizes = results.get("vocab_sizes", [])
    index_times = results.get("index_build_times", [])
    stage1_times = results.get("stage1_times", [])
    stage2_times = results.get("stage2_times", [])

    if not vocab_sizes:
        print("No timing data found")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Absolute times
    ax1 = axes[0]
    ax1.plot(vocab_sizes, index_times, "o-", label="Index Build", linewidth=2)
    ax1.plot(vocab_sizes, stage1_times, "s-", label="Stage 1", linewidth=2)
    ax1.plot(vocab_sizes, stage2_times, "^-", label="Stage 2", linewidth=2)
    ax1.set_xlabel("Vocabulary Size")
    ax1.set_ylabel("Time (seconds)")
    ax1.set_title("Compute Time vs Vocabulary Size")
    ax1.legend()
    ax1.set_xscale("log")
    ax1.set_yscale("log")

    # Right: Per-token times
    ax2 = axes[1]
    per_token_index = [t / v for t, v in zip(index_times, vocab_sizes)]
    per_token_stage1 = [t / v for t, v in zip(stage1_times, vocab_sizes)]
    per_token_stage2 = [t / v for t, v in zip(stage2_times, vocab_sizes)]

    ax2.plot(vocab_sizes, per_token_index, "o-", label="Index Build", linewidth=2)
    ax2.plot(vocab_sizes, per_token_stage1, "s-", label="Stage 1", linewidth=2)
    ax2.plot(vocab_sizes, per_token_stage2, "^-", label="Stage 2", linewidth=2)
    ax2.set_xlabel("Vocabulary Size")
    ax2.set_ylabel("Time per Token (seconds)")
    ax2.set_title("Per-Token Compute Time")
    ax2.legend()
    ax2.set_xscale("log")

    plt.tight_layout()
    output_path = output_dir / "compute_scaling.png"
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def plot_pipeline_diagram(output_dir: Path) -> None:
    """Generate pipeline diagram showing DCTT stages.

    Args:
        output_dir: Output directory.
    """
    plt = setup_matplotlib()
    import matplotlib.patches as mpatches
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis("off")

    # Define boxes for each stage
    stages = [
        {"name": "Stage 0\nPreprocessing", "x": 1, "y": 5, "color": "#E8E8E8"},
        {"name": "Stage 1\nBasic Outliers", "x": 4, "y": 5, "color": "#B3D9FF"},
        {"name": "Stage 2\nSpectral Geometry", "x": 7, "y": 5, "color": "#99CC99"},
        {"name": "Stage 3\nAdvanced (opt)", "x": 10, "y": 5, "color": "#FFE5B4"},
        {"name": "Severity\nScoring", "x": 7, "y": 2.5, "color": "#DDA0DD"},
        {"name": "Repair\nOptimization", "x": 10, "y": 2.5, "color": "#FFB6C1"},
    ]

    for stage in stages:
        box = FancyBboxPatch(
            (stage["x"] - 1, stage["y"] - 0.7),
            2,
            1.4,
            boxstyle="round,pad=0.05,rounding_size=0.2",
            facecolor=stage["color"],
            edgecolor="black",
            linewidth=1.5,
        )
        ax.add_patch(box)
        ax.text(
            stage["x"],
            stage["y"],
            stage["name"],
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
        )

    # Add arrows
    arrows = [
        ((2, 5), (3, 5)),  # Stage 0 -> Stage 1
        ((5, 5), (6, 5)),  # Stage 1 -> Stage 2
        ((8, 5), (9, 5)),  # Stage 2 -> Stage 3
        ((7, 4.3), (7, 3.2)),  # Stage 2 -> Severity
        ((8, 2.5), (9, 2.5)),  # Severity -> Repair
        ((10, 4.3), (10, 3.2)),  # Stage 3 -> Repair (dashed)
    ]

    for i, (start, end) in enumerate(arrows):
        style = "dashed" if i == len(arrows) - 1 else "solid"
        arrow = FancyArrowPatch(
            start,
            end,
            arrowstyle="->",
            mutation_scale=15,
            linestyle=style,
            color="black",
            linewidth=1.5,
        )
        ax.add_patch(arrow)

    # Add labels for components
    ax.text(1, 6.5, "Extract\nEmbeddings", ha="center", fontsize=8)
    ax.text(4, 6.5, "μ_k, spread_q\nLOF", ha="center", fontsize=8)
    ax.text(7, 6.5, "cond, PR\nlogdet, dim95", ha="center", fontsize=8)
    ax.text(10, 6.5, "TDA, MLE\n(optional)", ha="center", fontsize=8)

    # Title
    ax.text(7, 7.5, "DCTT Pipeline Overview", ha="center", fontsize=14, fontweight="bold")

    output_path = output_dir / "pipeline_diagram.png"
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate DCTT publication figures")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("outputs/runs/latest"),
        help="Directory containing result files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/figures"),
        help="Output directory for figures",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Generate all figures",
    )
    parser.add_argument(
        "--pipeline",
        action="store_true",
        help="Generate pipeline diagram",
    )
    parser.add_argument(
        "--distributions",
        action="store_true",
        help="Generate metric distribution plots",
    )
    parser.add_argument(
        "--severity",
        action="store_true",
        help="Generate severity vs failure plot",
    )
    parser.add_argument(
        "--causal",
        action="store_true",
        help="Generate causal effects plot",
    )
    parser.add_argument(
        "--scaling",
        action="store_true",
        help="Generate compute scaling plot",
    )

    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Generate requested plots
    if args.all or args.pipeline:
        plot_pipeline_diagram(args.output_dir)

    if args.all or args.distributions:
        diag_path = args.results_dir / "diagnostic_results.json"
        if diag_path.exists():
            plot_metric_distributions(diag_path, args.output_dir)
        else:
            print(f"Skipping distributions: {diag_path} not found")

    if args.all or args.severity:
        diag_path = args.results_dir / "diagnostic_results.json"
        stress_path = args.results_dir / "stress_test_results.json"
        if diag_path.exists() and stress_path.exists():
            plot_severity_vs_failure(diag_path, stress_path, args.output_dir)
        else:
            print("Skipping severity plot: required files not found")

    if args.all or args.causal:
        causal_path = args.results_dir / "causal_results.json"
        if causal_path.exists():
            plot_causal_effects(causal_path, args.output_dir)
        else:
            print(f"Skipping causal plot: {causal_path} not found")

    if args.all or args.scaling:
        timing_path = args.results_dir / "timing_results.json"
        if timing_path.exists():
            plot_compute_scaling(timing_path, args.output_dir)
        else:
            print(f"Skipping scaling plot: {timing_path} not found")


if __name__ == "__main__":
    main()
