#!/usr/bin/env python3
"""Generate figures and tables for DCTT paper.

This script produces publication-ready figures and tables:
1. Metric distributions by token type/frequency tier
2. Predictive validity comparison (baseline vs geometry)
3. Cluster repair results (geometry improvement)
4. Qualitative examples of flagged tokens

Usage:
    python scripts/generate_paper_figures.py
"""

from __future__ import annotations

import argparse
import heapq
import hashlib
import json
import logging
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style for publication
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.figsize': (6, 4),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})


def _resolve_path(path_value: str | Path, repo_root: Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else (repo_root / path).resolve()


def _load_json(path: Path) -> dict | None:
    try:
        with path.open(encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        logger.warning("Skipping unreadable JSON file %s: %s", path, exc)
        return None


def _load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    if not isinstance(payload, dict):
        raise RuntimeError(f"YAML payload is not a mapping: {path}")
    return payload


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sanitize_manifest_source_path(path_str: str) -> Path:
    base = path_str.split("#", 1)[0].strip()
    return Path(base)


def _git_metadata(repo_root: Path) -> dict:
    def _run_git(args: list[str], fallback: str = "unknown") -> str:
        try:
            out = subprocess.check_output(
                ["git", *args],
                cwd=repo_root,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
            return out or fallback
        except Exception:
            return fallback

    return {
        "branch": _run_git(["rev-parse", "--abbrev-ref", "HEAD"]),
        "commit": _run_git(["rev-parse", "HEAD"]),
        "is_dirty": _run_git(["status", "--porcelain"], fallback="") != "",
    }


def _predictive_proxy_from_sweep(
    sweep_payload: dict,
    preferred_models: tuple[str, ...] = ("qwen2_5_coder_7b",),
) -> dict | None:
    """Build predictive-validity proxy metrics from sweep aggregate means."""
    aggregate = sweep_payload.get("aggregate", {})
    if not aggregate:
        return None

    model_name = None
    for candidate in preferred_models:
        if candidate in aggregate:
            model_name = candidate
            break
    if model_name is None:
        model_name = sorted(aggregate.keys())[0]

    row = aggregate.get(model_name, {})
    if not row:
        return None

    def _auc_block(mean_key: str, std_key: str) -> dict:
        mean = float(row.get(mean_key, 0.0))
        std = float(row.get(std_key, 0.0))
        return {
            "auc": mean,
            "auc_ci": [
                max(0.0, mean - std),
                min(1.0, mean + std),
            ],
        }

    return {
        "config": {
            "model": model_name,
            "use_simulated_failures": False,
            "source": "predictive_validity_sweep_aggregate",
            "n_runs": int(row.get("n_runs", 0)),
        },
        "model_comparison": {
            "baseline": _auc_block("baseline_auc_mean", "baseline_auc_std"),
            "geometry": _auc_block("geometry_auc_mean", "geometry_auc_std"),
            "full": _auc_block("full_auc_mean", "full_auc_std"),
        },
        "summary": {},
    }


def load_results(output_dir: Path) -> dict:
    """Load all experiment results."""
    results = {}
    results_paths = {}
    predictive_runs: list[dict] = []

    def load_json_safe(path: Path):
        try:
            with path.open() as f:
                return json.load(f)
        except Exception as exc:
            logger.warning(f"Skipping unreadable JSON file {path}: {exc}")
            return None

    # Find most recent run directories
    runs_dir = output_dir / "runs"
    if runs_dir.exists():
        date_dirs = sorted(runs_dir.iterdir(), reverse=True)
        for date_dir in date_dirs[:5]:  # Check last 5 days
            for run_dir in sorted(date_dir.iterdir(), reverse=True):
                # Load diagnostic results
                diag_path = run_dir / "diagnostic_results.json"
                if diag_path.exists() and "diagnostic" not in results:
                    payload = load_json_safe(diag_path)
                    if payload is not None:
                        results["diagnostic"] = payload
                        results_paths["diagnostic"] = str(diag_path)
                        logger.info(f"Loaded diagnostic results from {diag_path}")

                # Load cluster repair results
                cluster_path = run_dir / "cluster_repair_results.json"
                if cluster_path.exists() and "cluster_repair" not in results:
                    payload = load_json_safe(cluster_path)
                    if payload is not None:
                        results["cluster_repair"] = payload
                        results_paths["cluster_repair"] = str(cluster_path)
                        logger.info(f"Loaded cluster repair from {cluster_path}")

                # Load causal results
                causal_path = run_dir / "causal_cluster_repair_results.json"
                if causal_path.exists() and "causal" not in results:
                    payload = load_json_safe(causal_path)
                    if payload is not None:
                        results["causal"] = payload
                        results_paths["causal"] = str(causal_path)
                        logger.info(f"Loaded causal results from {causal_path}")

                # Load predictive validity results
                pv_path = run_dir / "predictive_validity_results.json"
                if pv_path.exists():
                    payload = load_json_safe(pv_path)
                    if payload is not None:
                        model_name = payload.get("config", {}).get("model", "unknown")
                        predictive_runs.append({
                            "model": model_name,
                            "path": str(pv_path),
                            "run_dir": str(run_dir),
                            "payload": payload,
                        })
                        if "predictive_validity" not in results:
                            results["predictive_validity"] = payload
                            results_paths["predictive_validity"] = str(pv_path)
                            logger.info(f"Loaded predictive validity from {pv_path}")

                # Load single-token repair validation
                rv_path = run_dir / "repair_validation_results.json"
                if rv_path.exists() and "repair_validation" not in results:
                    payload = load_json_safe(rv_path)
                    if payload is not None:
                        results["repair_validation"] = payload
                        results_paths["repair_validation"] = str(rv_path)
                        logger.info(f"Loaded repair validation from {rv_path}")

    if results_paths:
        results["_paths"] = results_paths
    if predictive_runs:
        results["predictive_validity_runs"] = predictive_runs

    # Load predictive-validity sweep aggregate with protocol lock priority.
    sweep_root = output_dir / "sweeps" / "predictive_validity"
    if sweep_root.exists():
        lock_path = sweep_root / "PROTOCOL_LOCK.json"
        if lock_path.exists():
            lock_payload = load_json_safe(lock_path)
            if lock_payload is None:
                raise RuntimeError(f"Unreadable protocol lock file: {lock_path}")

            locked_sweep = lock_payload.get("sweep_results_path")
            if not isinstance(locked_sweep, str) or not locked_sweep:
                raise RuntimeError(
                    f"Protocol lock missing sweep_results_path: {lock_path}"
                )

            locked_path = Path(locked_sweep)
            if not locked_path.is_absolute():
                locked_path = (sweep_root / locked_path).resolve()

            locked_payload = load_json_safe(locked_path)
            if locked_payload is None or not locked_payload.get("aggregate"):
                raise RuntimeError(
                    "Protocol lock points to missing/invalid sweep aggregate: "
                    f"{locked_path}"
                )

            results["predictive_validity_sweep"] = locked_payload
            results_paths["predictive_validity_sweep"] = str(locked_path)
            results["_predictive_protocol_lock"] = lock_payload
            proxy = _predictive_proxy_from_sweep(locked_payload)
            if proxy is not None:
                results["predictive_validity"] = proxy
                results_paths["predictive_validity"] = (
                    f"{locked_path}#aggregate:{proxy['config']['model']}"
                )
            logger.info(
                "Loaded predictive validity sweep from protocol lock: %s",
                locked_path,
            )
            return results

        sweep_candidates = sorted(
            sweep_root.glob("*/sweep_results.json"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        best_sweep_payload = None
        best_sweep_path = None
        best_score = None
        for sweep_path in sweep_candidates:
            payload = load_json_safe(sweep_path)
            if payload is None:
                continue
            aggregate = payload.get("aggregate", {})
            if not aggregate:
                continue

            total_runs = int(
                sum(int(row.get("n_runs", 0)) for row in aggregate.values())
            )
            n_models = int(len(aggregate))
            mtime = float(sweep_path.stat().st_mtime)
            score = (total_runs, n_models, mtime)
            if best_score is None or score > best_score:
                best_score = score
                best_sweep_payload = payload
                best_sweep_path = sweep_path

        if best_sweep_payload is not None and best_sweep_path is not None:
            results["predictive_validity_sweep"] = best_sweep_payload
            results_paths["predictive_validity_sweep"] = str(best_sweep_path)
            proxy = _predictive_proxy_from_sweep(best_sweep_payload)
            if proxy is not None:
                results["predictive_validity"] = proxy
                results_paths["predictive_validity"] = (
                    f"{best_sweep_path}#aggregate:{proxy['config']['model']}"
                )
            logger.info(
                "Loaded predictive validity sweep from %s (score=%s). "
                "No protocol lock found.",
                best_sweep_path,
                best_score,
            )

    return results


def _attach_predictive_sweep_from_protocol_lock(
    *,
    results: dict,
    results_paths: dict,
    sweep_root: Path,
    protocol_lock_path: Path,
    strict_lock: bool,
) -> bool:
    lock_payload = _load_json(protocol_lock_path)
    if lock_payload is None:
        if strict_lock:
            raise RuntimeError(f"Unreadable protocol lock file: {protocol_lock_path}")
        logger.warning("Skipping unreadable protocol lock file: %s", protocol_lock_path)
        return False

    locked_sweep = lock_payload.get("sweep_results_path")
    if not isinstance(locked_sweep, str) or not locked_sweep:
        if strict_lock:
            raise RuntimeError(
                f"Protocol lock missing sweep_results_path: {protocol_lock_path}"
            )
        logger.warning(
            "Protocol lock missing sweep_results_path: %s", protocol_lock_path
        )
        return False

    locked_path = Path(locked_sweep)
    if not locked_path.is_absolute():
        locked_path = (sweep_root / locked_path).resolve()

    locked_payload = _load_json(locked_path)
    if locked_payload is None or not locked_payload.get("aggregate"):
        if strict_lock:
            raise RuntimeError(
                "Protocol lock points to missing/invalid sweep aggregate: "
                f"{locked_path}"
            )
        logger.warning(
            "Protocol lock points to missing/invalid sweep aggregate: %s", locked_path
        )
        return False

    results["predictive_validity_sweep"] = locked_payload
    results["_predictive_protocol_lock"] = lock_payload
    results_paths["predictive_protocol_lock"] = str(protocol_lock_path)
    results_paths["predictive_validity_sweep"] = str(locked_path)
    proxy = _predictive_proxy_from_sweep(locked_payload)
    if proxy is not None:
        results["predictive_validity"] = proxy
        results_paths["predictive_validity"] = (
            f"{locked_path}#aggregate:{proxy['config']['model']}"
        )
    logger.info(
        "Loaded predictive validity sweep from protocol lock: %s",
        locked_path,
    )
    return True


def load_results_from_paper_lock(
    *,
    output_dir: Path,
    paper_lock_path: Path,
    strict_lock: bool = True,
) -> dict:
    repo_root = output_dir.parent
    lock_payload = _load_yaml(paper_lock_path)
    artifacts = lock_payload.get("artifacts", {})
    if not isinstance(artifacts, dict):
        raise RuntimeError(
            f"Paper lock missing 'artifacts' mapping: {paper_lock_path}"
        )

    def _maybe_load_json(
        result_key: str,
        artifact_key: str,
        *,
        required: bool,
    ) -> None:
        rel_path = artifacts.get(artifact_key)
        if not rel_path:
            if required and strict_lock:
                raise RuntimeError(
                    f"Paper lock missing required artifact '{artifact_key}': "
                    f"{paper_lock_path}"
                )
            return

        artifact_path = _resolve_path(rel_path, repo_root)
        payload = _load_json(artifact_path)
        if payload is None:
            if required and strict_lock:
                raise RuntimeError(f"Unreadable required artifact: {artifact_path}")
            return

        results[result_key] = payload
        results_paths[result_key] = str(artifact_path)
        logger.info("Loaded %s from lock: %s", result_key, artifact_path)

    results: dict = {"_paper_lock": str(paper_lock_path)}
    results_paths: dict = {"paper_lock": str(paper_lock_path)}

    _maybe_load_json("diagnostic", "diagnostic_results", required=True)
    _maybe_load_json("cluster_repair", "cluster_repair_results", required=True)
    _maybe_load_json("causal", "causal_cluster_repair_results", required=True)
    _maybe_load_json("hard_pivot_report", "hard_pivot_report", required=False)

    sweep_root = output_dir / "sweeps" / "predictive_validity"
    protocol_lock_rel = artifacts.get("predictive_protocol_lock")
    protocol_lock_path = (
        _resolve_path(protocol_lock_rel, repo_root)
        if protocol_lock_rel
        else sweep_root / "PROTOCOL_LOCK.json"
    )

    _attach_predictive_sweep_from_protocol_lock(
        results=results,
        results_paths=results_paths,
        sweep_root=sweep_root,
        protocol_lock_path=protocol_lock_path,
        strict_lock=strict_lock,
    )

    results["_paths"] = results_paths
    return results


def _latest_predictive_runs_by_model(results: dict) -> list[dict]:
    """Get latest predictive-validity run per model."""
    runs = results.get("predictive_validity_runs", [])
    if not runs:
        return []

    latest: dict[str, dict] = {}
    for run in runs:
        model = run.get("model", "unknown")
        if model not in latest:
            latest[model] = run
    return list(latest.values())


def figure0_pipeline_diagram(output_dir: Path, spec_path: Path) -> None:
    """Figure 0: Deterministic pipeline diagram from source spec."""
    if not spec_path.exists():
        logger.warning("Pipeline diagram spec not found: %s", spec_path)
        return

    spec = _load_yaml(spec_path)
    stages = spec.get("stages", [])
    edges = spec.get("edges", [])
    layout = spec.get("layout", {})
    footer = spec.get("footer", {})

    if not stages:
        logger.warning("Pipeline diagram spec has no stages: %s", spec_path)
        return

    xlim = layout.get("xlim", [0.0, 14.0])
    ylim = layout.get("ylim", [0.0, 8.0])

    fig, ax = plt.subplots(figsize=(12.5, 7.0))
    ax.set_xlim(float(xlim[0]), float(xlim[1]))
    ax.set_ylim(float(ylim[0]), float(ylim[1]))
    ax.axis("off")

    node_map: dict[str, dict] = {}
    for stage in stages:
        stage_id = str(stage["id"])
        x = float(stage["x"])
        y = float(stage["y"])
        w = float(stage["width"])
        h = float(stage["height"])
        color = str(stage.get("color", "#E8E8E8"))
        label = str(stage.get("label", stage_id))
        detail = str(stage.get("detail", "")).strip()

        node_map[stage_id] = {"x": x, "y": y, "w": w, "h": h}
        box = FancyBboxPatch(
            (x - w / 2.0, y - h / 2.0),
            w,
            h,
            boxstyle="round,pad=0.06,rounding_size=0.18",
            facecolor=color,
            edgecolor="black",
            linewidth=1.3,
        )
        ax.add_patch(box)
        ax.text(
            x,
            y + 0.14,
            label,
            ha="center",
            va="center",
            fontsize=9.5,
            fontweight="bold",
            color="#1F2933",
        )
        if detail:
            ax.text(
                x,
                y - 0.36,
                detail,
                ha="center",
                va="center",
                fontsize=8.1,
                color="#273444",
            )

    for edge in edges:
        src = node_map.get(str(edge.get("from", "")))
        dst = node_map.get(str(edge.get("to", "")))
        if src is None or dst is None:
            continue

        dx = dst["x"] - src["x"]
        dy = dst["y"] - src["y"]
        if abs(dx) >= abs(dy):
            sx = src["x"] + np.sign(dx) * (src["w"] / 2.0)
            sy = src["y"]
            ex = dst["x"] - np.sign(dx) * (dst["w"] / 2.0)
            ey = dst["y"]
        else:
            sx = src["x"]
            sy = src["y"] + np.sign(dy) * (src["h"] / 2.0)
            ex = dst["x"]
            ey = dst["y"] - np.sign(dy) * (dst["h"] / 2.0)

        style = str(edge.get("style", "solid"))
        arrow = FancyArrowPatch(
            (sx, sy),
            (ex, ey),
            arrowstyle="->",
            mutation_scale=13,
            linewidth=1.5,
            color="#212121",
            linestyle="--" if style == "dashed" else "-",
            connectionstyle="arc3,rad=0.0",
        )
        ax.add_patch(arrow)

    title = str(spec.get("title", "DCTT Pipeline"))
    subtitle = str(spec.get("subtitle", "")).strip()
    ax.text(
        (float(xlim[0]) + float(xlim[1])) / 2.0,
        float(ylim[1]) - 0.35,
        title,
        ha="center",
        va="center",
        fontsize=14,
        fontweight="bold",
    )
    if subtitle:
        ax.text(
            (float(xlim[0]) + float(xlim[1])) / 2.0,
            float(ylim[1]) - 0.75,
            subtitle,
            ha="center",
            va="center",
            fontsize=9.8,
            color="#4B5563",
        )

    footer_text = str(footer.get("text", "")).strip()
    if footer_text:
        ax.text(
            float(footer.get("x", 7.0)),
            float(footer.get("y", 0.8)),
            footer_text,
            ha="center",
            va="center",
            fontsize=8.5,
            color="#374151",
        )

    plt.tight_layout()
    fig.savefig(output_dir / "fig0_pipeline_diagram.svg")
    fig.savefig(output_dir / "fig0_pipeline_diagram.pdf")
    fig.savefig(output_dir / "fig0_pipeline_diagram.png")
    plt.close()
    logger.info("Saved Figure 0: Pipeline diagram (svg/pdf/png)")


def figure1_predictive_validity(results: dict, output_dir: Path) -> None:
    """Figure 1: Predictive validity - geometry vs baseline."""
    hard_pivot = results.get("hard_pivot_report", {})
    pooled = hard_pivot.get("pooled", {})
    if pooled:
        labels = ["Geometry - Baseline", "Full - Baseline"]
        means = [
            float(pooled.get("geometry_minus_baseline_mean", 0.0)),
            float(pooled.get("full_minus_baseline_mean", 0.0)),
        ]
        ci_blocks = [
            pooled.get("geometry_minus_baseline_ci", [0.0, 0.0]),
            pooled.get("full_minus_baseline_ci", [0.0, 0.0]),
        ]
        errors = np.array(
            [
                [means[0] - float(ci_blocks[0][0]), float(ci_blocks[0][1]) - means[0]],
                [means[1] - float(ci_blocks[1][0]), float(ci_blocks[1][1]) - means[1]],
            ]
        ).T

        fig, ax = plt.subplots(figsize=(5.6, 4.0))
        bars = ax.bar(
            labels,
            means,
            color=["#D6604D", "#4393C3"],
            edgecolor="black",
            linewidth=0.6,
        )
        ax.errorbar(labels, means, yerr=errors, fmt="none", color="black", capsize=5)
        ax.axhline(0.0, color="gray", linestyle="--", linewidth=1.0, alpha=0.7)
        ax.set_ylabel("AUC Delta")
        ax.set_title("Strict Predictive Deltas (Pooled Across 20 Runs)")
        ax.set_ylim(min(-0.28, min(means) - 0.04), max(0.06, max(means) + 0.04))

        pos_geom = int(pooled.get("geometry_positive_runs", 0))
        pos_full = int(pooled.get("full_positive_runs", 0))
        total_runs = int(hard_pivot.get("total_runs", 0))
        for idx, (bar, mean) in enumerate(zip(bars, means)):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                mean + (0.012 if mean >= 0 else -0.012),
                f"{mean:+.3f}",
                ha="center",
                va="bottom" if mean >= 0 else "top",
                fontsize=10,
                fontweight="bold",
            )
            positive = pos_geom if idx == 0 else pos_full
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                ax.get_ylim()[0] + 0.01,
                f"positive: {positive}/{total_runs}",
                ha="center",
                va="bottom",
                fontsize=8,
                color="#374151",
            )

        verdict = hard_pivot.get("verdicts", {}).get("strict_predictive_gate", "unknown")
        ax.text(
            0.5,
            0.95,
            f"Gate verdict: {verdict}",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=9,
            color="#7F1D1D" if str(verdict).upper() == "FAIL" else "#065F46",
            fontweight="bold",
        )

        plt.tight_layout()
        fig.savefig(output_dir / "fig1_predictive_validity.pdf")
        fig.savefig(output_dir / "fig1_predictive_validity.png")
        plt.close()
        logger.info("Saved Figure 1: Hard pivot pooled predictive deltas")
        return

    if "predictive_validity" not in results:
        logger.warning("No predictive validity results found")
        return

    pv = results["predictive_validity"]
    model_comp = pv.get("model_comparison", {})
    use_simulated = pv.get("config", {}).get("use_simulated_failures", False)

    # Extract data
    models = ["Baseline\n(confounds)", "Geometry\nOnly", "Full\nModel"]
    aucs = [
        model_comp.get("baseline", {}).get("auc", 0),
        model_comp.get("geometry", {}).get("auc", 0),
        model_comp.get("full", {}).get("auc", 0),
    ]

    # CI bars
    baseline_ci = model_comp.get("baseline", {}).get("auc_ci", [0, 0])
    geometry_ci = model_comp.get("geometry", {}).get("auc_ci", [0, 0])
    full_ci = model_comp.get("full", {}).get("auc_ci", [0, 0])

    errors = [
        [aucs[0] - baseline_ci[0], baseline_ci[1] - aucs[0]],
        [aucs[1] - geometry_ci[0], geometry_ci[1] - aucs[1]],
        [aucs[2] - full_ci[0], full_ci[1] - aucs[2]],
    ]
    errors = np.array(errors).T

    # Create figure
    fig, ax = plt.subplots(figsize=(5, 4))

    colors = ['#95a5a6', '#3498db', '#2ecc71']
    bars = ax.bar(models, aucs, color=colors, edgecolor='black', linewidth=0.5)
    ax.errorbar(models, aucs, yerr=errors, fmt='none', color='black', capsize=5)

    # Add value labels
    for bar, auc in zip(bars, aucs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{auc:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_ylabel('AUC-ROC')
    ax.set_ylim(0.4, 1.0)
    ax.set_title('Predictive Validity: Geometry Metrics vs Confound Baselines')
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random')

    # Add improvement annotation
    improvement = aucs[1] - aucs[0]
    ax.annotate(f'{improvement:+.3f}', xy=(1, aucs[1]), xytext=(1.3, aucs[1] - 0.05),
                fontsize=10, color='#e74c3c', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#e74c3c'))

    if use_simulated:
        ax.text(
            0.5,
            0.03,
            'Warning: labels are simulated in this run',
            transform=ax.transAxes,
            ha='center',
            va='bottom',
            fontsize=8,
            color='#c0392b',
            fontstyle='italic',
        )

    plt.tight_layout()
    fig.savefig(output_dir / "fig1_predictive_validity.pdf")
    fig.savefig(output_dir / "fig1_predictive_validity.png")
    plt.close()
    logger.info("Saved Figure 1: Predictive validity")


def figure4_model_replication(results: dict, output_dir: Path) -> None:
    """Figure 4: Multi-model predictive validity replication."""
    hard_pivot = results.get("hard_pivot_report", {})
    per_model = hard_pivot.get("per_model", {})
    if per_model:
        model_labels = sorted(per_model.keys())
        geom_mean = [
            float(per_model[m].get("geometry_minus_baseline_mean", 0.0))
            for m in model_labels
        ]
        full_mean = [
            float(per_model[m].get("full_minus_baseline_mean", 0.0))
            for m in model_labels
        ]

        def _err(ci: list[float], mean: float) -> tuple[float, float]:
            return (
                mean - float(ci[0]),
                float(ci[1]) - mean,
            )

        geom_err = np.array(
            [
                _err(per_model[m].get("geometry_minus_baseline_ci", [0.0, 0.0]), geom_mean[i])
                for i, m in enumerate(model_labels)
            ]
        ).T
        full_err = np.array(
            [
                _err(per_model[m].get("full_minus_baseline_ci", [0.0, 0.0]), full_mean[i])
                for i, m in enumerate(model_labels)
            ]
        ).T

        x = np.arange(len(model_labels))
        width = 0.36
        fig, ax = plt.subplots(figsize=(max(7.0, len(model_labels) * 2.2), 4.6))
        ax.bar(
            x - width / 2.0,
            geom_mean,
            width,
            yerr=geom_err,
            capsize=4,
            color="#D6604D",
            edgecolor="black",
            linewidth=0.6,
            label="Geometry - Baseline (mean ± 95% CI)",
        )
        ax.bar(
            x + width / 2.0,
            full_mean,
            width,
            yerr=full_err,
            capsize=4,
            color="#4393C3",
            edgecolor="black",
            linewidth=0.6,
            label="Full - Baseline (mean ± 95% CI)",
        )

        ax.axhline(0.0, linestyle="--", color="gray", linewidth=1.0, alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels([m[:22] for m in model_labels], rotation=15, ha="right")
        ax.set_ylabel("AUC Delta")
        ax.set_title("Cross-Family Strict Predictive Deltas (5 Seeds/Model)")
        ax.legend(loc="lower left")
        ax.set_ylim(min(-0.30, min(geom_mean) - 0.03), max(0.08, max(full_mean) + 0.03))

        for idx, model in enumerate(model_labels):
            n_runs = int(per_model[model].get("n_runs", 0))
            ax.text(
                x[idx],
                ax.get_ylim()[1] - 0.015,
                f"n={n_runs}",
                ha="center",
                va="top",
                fontsize=8,
                color="#1F2937",
            )

        plt.tight_layout()
        fig.savefig(output_dir / "fig4_model_replication.pdf")
        fig.savefig(output_dir / "fig4_model_replication.png")
        plt.close()
        logger.info("Saved Figure 4: Hard pivot cross-family replication")
        return

    sweep = results.get("predictive_validity_sweep")
    if sweep:
        aggregate = sweep.get("aggregate", {})
        if not aggregate:
            logger.warning("Predictive sweep payload missing aggregate data")
            return

        model_labels = sorted(aggregate.keys())
        baseline_auc = [float(aggregate[m].get("baseline_auc_mean", 0.0)) for m in model_labels]
        geometry_auc = [float(aggregate[m].get("geometry_auc_mean", 0.0)) for m in model_labels]
        baseline_std = [float(aggregate[m].get("baseline_auc_std", 0.0)) for m in model_labels]
        geometry_std = [float(aggregate[m].get("geometry_auc_std", 0.0)) for m in model_labels]
        n_runs = [int(aggregate[m].get("n_runs", 0)) for m in model_labels]

        x = np.arange(len(model_labels))
        width = 0.35
        fig, ax = plt.subplots(figsize=(max(6, len(model_labels) * 2.4), 4.4))

        ax.bar(
            x - width / 2,
            baseline_auc,
            width,
            yerr=baseline_std,
            capsize=4,
            color="#95a5a6",
            edgecolor="black",
            linewidth=0.5,
            label="Baseline AUC (mean±sd)",
        )
        ax.bar(
            x + width / 2,
            geometry_auc,
            width,
            yerr=geometry_std,
            capsize=4,
            color="#3498db",
            edgecolor="black",
            linewidth=0.5,
            label="Geometry AUC (mean±sd)",
        )

        for idx, run_count in enumerate(n_runs):
            ax.text(
                x[idx],
                min(0.98, max(baseline_auc[idx], geometry_auc[idx]) + 0.04),
                f"n={run_count}",
                ha="center",
                va="bottom",
                fontsize=8,
                color="#2c3e50",
            )

        ax.set_xticks(x)
        ax.set_xticklabels([m[:20] for m in model_labels], rotation=15, ha="right")
        ax.set_ylabel("AUC-ROC")
        ax.set_ylim(0.2, 1.0)
        ax.set_title("Model Replication: Predictive Validity Sweep Means")
        ax.legend(loc="lower right")
        ax.axhline(0.5, linestyle="--", color="gray", alpha=0.5)

        plt.tight_layout()
        fig.savefig(output_dir / "fig4_model_replication.pdf")
        fig.savefig(output_dir / "fig4_model_replication.png")
        plt.close()
        logger.info("Saved Figure 4: Model replication (sweep aggregate)")
        return

    runs = _latest_predictive_runs_by_model(results)
    if not runs:
        logger.warning("No predictive-validity runs found for replication figure")
        return

    model_labels = []
    baseline_auc = []
    geometry_auc = []
    real_label_flags = []

    for run in runs:
        payload = run.get("payload", {})
        cfg = payload.get("config", {})
        model = cfg.get("model", "unknown")
        model_labels.append(model.split("/")[-1][:20])
        model_comp = payload.get("model_comparison", {})
        baseline_auc.append(float(model_comp.get("baseline", {}).get("auc", 0.0)))
        geometry_auc.append(float(model_comp.get("geometry", {}).get("auc", 0.0)))
        real_label_flags.append(not bool(cfg.get("use_simulated_failures", False)))

    x = np.arange(len(model_labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(6, len(model_labels) * 2.2), 4.2))

    ax.bar(
        x - width / 2,
        baseline_auc,
        width,
        color="#95a5a6",
        edgecolor="black",
        linewidth=0.5,
        label="Baseline AUC",
    )
    ax.bar(
        x + width / 2,
        geometry_auc,
        width,
        color="#3498db",
        edgecolor="black",
        linewidth=0.5,
        label="Geometry AUC",
    )

    for idx, is_real in enumerate(real_label_flags):
        if not is_real:
            ax.text(
                x[idx],
                min(0.98, max(baseline_auc[idx], geometry_auc[idx]) + 0.03),
                "sim",
                ha="center",
                va="bottom",
                fontsize=8,
                color="#c0392b",
                fontweight="bold",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(model_labels, rotation=15, ha="right")
    ax.set_ylabel("AUC-ROC")
    ax.set_ylim(0.4, 1.0)
    ax.set_title("Model Replication: Predictive Validity Across Models")
    ax.legend(loc="lower right")
    ax.axhline(0.5, linestyle="--", color="gray", alpha=0.5)

    plt.tight_layout()
    fig.savefig(output_dir / "fig4_model_replication.pdf")
    fig.savefig(output_dir / "fig4_model_replication.png")
    plt.close()
    logger.info("Saved Figure 4: Model replication")


def figure2_cluster_repair_geometry(results: dict, output_dir: Path) -> None:
    """Figure 2: Cluster repair geometry improvement."""
    if "cluster_repair" not in results:
        logger.warning("No cluster repair results found")
        return

    cr = results["cluster_repair"]
    repair_results = cr.get("repair_results", [])

    if not repair_results:
        return

    # Extract before/after condition numbers
    clusters = [f"C{r['cluster_id']}" for r in repair_results]
    cond_before = [r["geometry_before"]["cond"] for r in repair_results]
    cond_after = [r["geometry_after"]["cond"] for r in repair_results]
    jaccards = [r["mean_jaccard"] for r in repair_results]

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Left: Condition number before/after
    x = np.arange(len(clusters))
    width = 0.35

    bars1 = ax1.bar(x - width/2, cond_before, width, label='Before', color='#e74c3c', alpha=0.8)
    bars2 = ax1.bar(x + width/2, cond_after, width, label='After', color='#27ae60', alpha=0.8)

    ax1.set_xlabel('Cluster')
    ax1.set_ylabel('Condition Number')
    ax1.set_title('Geometry Improvement: Condition Number')
    ax1.set_xticks(x)
    ax1.set_xticklabels(clusters)
    ax1.legend()

    # Add improvement percentages
    for i, (b, a) in enumerate(zip(cond_before, cond_after)):
        pct = (b - a) / b * 100
        ax1.annotate(f'-{pct:.0f}%', xy=(i, a), xytext=(i, a - 0.3),
                    ha='center', fontsize=8, color='#27ae60', fontweight='bold')

    # Right: Semantic preservation (Jaccard)
    colors = ['#3498db' if j > 0.7 else '#e74c3c' for j in jaccards]
    bars3 = ax2.bar(clusters, jaccards, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)

    ax2.axhline(y=0.7, color='gray', linestyle='--', alpha=0.7, label='Threshold (0.7)')
    ax2.set_xlabel('Cluster')
    ax2.set_ylabel('Jaccard Overlap')
    ax2.set_title('Semantic Preservation: Neighbor Overlap')
    ax2.set_ylim(0, 1)
    ax2.legend()

    # Add value labels
    for bar, j in zip(bars3, jaccards):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{j:.2f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    fig.savefig(output_dir / "fig2_cluster_repair.pdf")
    fig.savefig(output_dir / "fig2_cluster_repair.png")
    plt.close()
    logger.info("Saved Figure 2: Cluster repair geometry")


def figure3_causal_geometry(results: dict, output_dir: Path) -> None:
    """Figure 3: Causal experiment - geometry changes."""
    if "causal" not in results:
        logger.warning("No causal results found")
        return

    causal = results["causal"]
    summary = causal.get("summary", {})

    # Create figure
    fig, ax = plt.subplots(figsize=(5, 4))

    groups = ['Treatment\n(Cluster Repair)', 'Control\n(Placebo)']
    changes = [
        summary.get("treatment_cond_change", 0),
        summary.get("control_cond_change", 0),
    ]

    colors = ['#27ae60' if c < 0 else '#e74c3c' for c in changes]
    bars = ax.bar(groups, changes, color=colors, edgecolor='black', linewidth=0.5)

    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.set_ylabel('Condition Number Change')
    ax.set_title('Mechanistic Result: Geometry Change by Group')

    # Add value labels
    for bar, change in zip(bars, changes):
        va = 'bottom' if change >= 0 else 'top'
        offset = 0.02 if change >= 0 else -0.02
        ax.text(bar.get_x() + bar.get_width()/2, change + offset,
                f'{change:+.3f}', ha='center', va=va, fontsize=11, fontweight='bold')

    # Add interpretation
    ax.text(0.5, 0.95, 'Geometry improves in treatment only',
            transform=ax.transAxes, ha='center', va='top',
            fontsize=10, style='italic', color='#27ae60')

    plt.tight_layout()
    fig.savefig(output_dir / "fig3_causal_geometry.pdf")
    fig.savefig(output_dir / "fig3_causal_geometry.png")
    plt.close()
    logger.info("Saved Figure 3: Causal geometry changes")


def table1_main_results(results: dict, output_dir: Path) -> None:
    """Table 1: Main experimental results."""
    pv_model = (
        results.get("predictive_validity", {})
        .get("config", {})
        .get("model", "unknown")
        .split("/")[-1]
    )

    lines = []
    lines.append("=" * 70)
    lines.append(f"TABLE 1: DCTT Main Experimental Results ({pv_model})")
    lines.append("=" * 70)
    lines.append("")

    # Predictive validity
    lines.append("A. PREDICTIVE VALIDITY (RQ1)")
    lines.append("-" * 50)
    hard_pivot = results.get("hard_pivot_report", {})
    pooled = hard_pivot.get("pooled", {})
    if pooled:
        geom_mean = float(pooled.get("geometry_minus_baseline_mean", 0.0))
        geom_ci = pooled.get("geometry_minus_baseline_ci", [0.0, 0.0])
        geom_pos = int(pooled.get("geometry_positive_runs", 0))
        full_mean = float(pooled.get("full_minus_baseline_mean", 0.0))
        full_ci = pooled.get("full_minus_baseline_ci", [0.0, 0.0])
        full_pos = int(pooled.get("full_positive_runs", 0))
        total_runs = int(hard_pivot.get("total_runs", 0))
        verdict = hard_pivot.get("verdicts", {}).get("strict_predictive_gate", "unknown")

        lines.append("  Label source:          REAL stress tests (strict package)")
        lines.append(
            "  Geometry-baseline:     "
            f"{geom_mean:+.3f} (95% CI [{float(geom_ci[0]):+.3f}, {float(geom_ci[1]):+.3f}])"
        )
        lines.append(
            "  Full-baseline:         "
            f"{full_mean:+.3f} (95% CI [{float(full_ci[0]):+.3f}, {float(full_ci[1]):+.3f}])"
        )
        lines.append(f"  Positive geometry runs:{geom_pos}/{total_runs}")
        lines.append(f"  Positive full runs:    {full_pos}/{total_runs}")
        lines.append(f"  Strict gate verdict:   {verdict}")
    elif "predictive_validity" in results:
        pv = results["predictive_validity"]
        mc = pv.get("model_comparison", {})
        use_simulated = pv.get("config", {}).get("use_simulated_failures", False)
        lines.append(f"  Label source:          {'SIMULATED (smoke-test only)' if use_simulated else 'REAL stress tests'}")
        lines.append(f"  Baseline (confounds):  AUC = {mc.get('baseline', {}).get('auc', 0):.3f}")
        lines.append(f"  Geometry only:         AUC = {mc.get('geometry', {}).get('auc', 0):.3f}")
        lines.append(f"  Full model:            AUC = {mc.get('full', {}).get('auc', 0):.3f}")
        improvement = mc.get('geometry', {}).get('auc', 0) - mc.get('baseline', {}).get('auc', 0)
        lines.append(f"  Improvement:           {improvement:+.3f}")
    lines.append("")

    # Cluster repair
    lines.append("B. CLUSTER REPAIR (RQ2 - Mechanistic)")
    lines.append("-" * 50)
    if "cluster_repair" in results:
        cr = results["cluster_repair"]
        summary = cr.get("summary", {})
        repair_results = cr.get("repair_results", [])
        cond_improvements = []
        jaccards = []
        for item in repair_results:
            before = item.get("geometry_before", {}).get("cond")
            after = item.get("geometry_after", {}).get("cond")
            if before is not None and after is not None:
                cond_improvements.append(before - after)
            jaccard = item.get("mean_jaccard")
            if jaccard is not None:
                jaccards.append(jaccard)

        cond_mean = float(np.mean(cond_improvements)) if cond_improvements else summary.get("mean_cond_improvement", 0.0)
        cond_std = float(np.std(cond_improvements)) if cond_improvements else 0.0
        jaccard_mean = float(np.mean(jaccards)) if jaccards else summary.get("mean_jaccard", 0.0)
        jaccard_std = float(np.std(jaccards)) if jaccards else 0.0
        clusters_improved = sum(1 for item in repair_results if item.get("geometry_improved"))
        total_repaired = len(repair_results) or summary.get("n_clusters_repaired", 0)

        lines.append(f"  Clusters found:        {summary.get('n_clusters_found', 0)}")
        lines.append(f"  Clusters repaired:     {summary.get('n_clusters_repaired', 0)}")
        lines.append(f"  Cond improvement:      {cond_mean:.3f} +/- {cond_std:.3f}")
        lines.append(f"  Jaccard overlap:       {jaccard_mean:.3f} +/- {jaccard_std:.3f}")
        if total_repaired:
            pct = 100 * clusters_improved / total_repaired
            lines.append(f"  Improvement rate:      {clusters_improved}/{total_repaired} ({pct:.0f}%)")
    lines.append("")

    # Causal experiment
    lines.append("C. CAUSAL EXPERIMENT (Treatment vs Control)")
    lines.append("-" * 50)
    if "causal" in results:
        causal = results["causal"]
        summary = causal.get("summary", {})
        lines.append(f"  Treatment n:           {summary.get('n_treatment', 0)} tokens ({summary.get('n_clusters', 0)} clusters)")
        lines.append(f"  Control n:             {summary.get('n_control', 0)} tokens")
        lines.append(f"  Treatment cond change: {summary.get('treatment_cond_change', 0):+.3f}")
        lines.append(f"  Control cond change:   {summary.get('control_cond_change', 0):+.3f}")
        lines.append(f"  Difference:            {summary.get('treatment_cond_change', 0) - summary.get('control_cond_change', 0):+.3f}")
    lines.append("")

    # Claims summary
    lines.append("D. SUPPORTED CLAIMS")
    lines.append("-" * 50)
    predictive_supported = False
    if "predictive_validity" in results:
        pv = results["predictive_validity"]
        use_simulated = pv.get("config", {}).get("use_simulated_failures", False)
        baseline_auc = pv.get("model_comparison", {}).get("baseline", {}).get("auc", 0.0)
        geometry_auc = pv.get("model_comparison", {}).get("geometry", {}).get("auc", 0.0)
        predictive_supported = (not use_simulated) and geometry_auc > baseline_auc

    cluster_supported = False
    if "cluster_repair" in results:
        repair_results = results["cluster_repair"].get("repair_results", [])
        cluster_supported = any(item.get("geometry_improved") for item in repair_results)

    behavioral_supported = False
    if "causal" in results:
        summary = results["causal"].get("summary", {})
        behavioral_supported = bool(summary.get("significant", False)) and summary.get("did", 0.0) < 0

    lines.append(f"  [{'YES' if predictive_supported else 'NO'}] Geometry predicts failures beyond confounds")
    lines.append(f"  [{'YES' if cluster_supported else 'NO'}] Cluster repair improves geometry vs placebo")
    lines.append(f"  [{'YES' if behavioral_supported else 'NO'}] Behavioral causal improvement")
    lines.append("")
    lines.append("=" * 70)

    table_text = "\n".join(lines)

    with (output_dir / "table1_main_results.txt").open("w") as f:
        f.write(table_text)

    print(table_text)
    logger.info("Saved Table 1: Main results")


def table2_flagged_tokens(results: dict, output_dir: Path) -> None:
    """Table 2: Examples of high-severity flagged tokens."""
    diagnostic = results.get("diagnostic", [])
    if not diagnostic:
        logger.warning("No diagnostic results found for Table 2")
        return

    top_results = heapq.nlargest(
        10,
        diagnostic,
        key=lambda item: item.get("severity", 0.0),
    )

    lines = []
    lines.append("=" * 70)
    lines.append("TABLE 2: High-Severity Flagged Tokens (Top 10)")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"{'Rank':<6}{'TokenID':<9}{'Token':<20}{'Type':<15}{'Severity':<10}")
    lines.append("-" * 70)

    placeholder_count = 0
    for i, item in enumerate(top_results, 1):
        token_info = item.get("token_info", {})
        token_id = token_info.get("token_id", -1)
        token = str(token_info.get("token_str", ""))
        token_type = str(token_info.get("token_type", "UNKNOWN"))
        severity = float(item.get("severity", 0.0))

        if token.startswith("token_"):
            placeholder_count += 1

        display_token = repr(token)[1:-1][:18]
        lines.append(f"{i:<6}{token_id:<9}{display_token:<20}{token_type:<15}{severity:<10.2f}")

    lines.append("")
    if placeholder_count == len(top_results):
        lines.append("Note: token strings are placeholder IDs in this census output.")
        lines.append("      Regenerate census with real tokenizer metadata for lexical examples.")
    else:
        lines.append("Note: Top tokens are computed directly from diagnostic severity scores.")
    lines.append("=" * 70)

    table_text = "\n".join(lines)

    with (output_dir / "table2_flagged_tokens.txt").open("w") as f:
        f.write(table_text)

    print(table_text)
    logger.info("Saved Table 2: Flagged tokens")


def table3_model_replication(results: dict, output_dir: Path) -> None:
    """Table 3: Latest predictive-validity results by model."""
    hard_pivot = results.get("hard_pivot_report", {})
    per_model = hard_pivot.get("per_model", {})
    if per_model:
        lines = []
        lines.append("=" * 124)
        lines.append("TABLE 3: Predictive Validity Replication (Hard Pivot, Strict 20-Run Package)")
        lines.append("=" * 124)
        lines.append("")
        lines.append(
            f"{'Model':<24}{'Runs':<8}{'GeomDelta (95% CI)':<38}"
            f"{'FullDelta (95% CI)':<38}{'PosGeom':<8}{'PosFull':<8}"
        )
        lines.append("-" * 124)
        for model in sorted(per_model.keys()):
            row = per_model[model]
            n_runs = int(row.get("n_runs", 0))
            g_mean = float(row.get("geometry_minus_baseline_mean", 0.0))
            g_ci = row.get("geometry_minus_baseline_ci", [0.0, 0.0])
            g_pos = int(row.get("geometry_positive_runs", 0))
            f_mean = float(row.get("full_minus_baseline_mean", 0.0))
            f_ci = row.get("full_minus_baseline_ci", [0.0, 0.0])
            f_pos = int(row.get("full_positive_runs", 0))
            g_cell = f"{g_mean:+.3f} [{float(g_ci[0]):+.3f}, {float(g_ci[1]):+.3f}]"
            f_cell = f"{f_mean:+.3f} [{float(f_ci[0]):+.3f}, {float(f_ci[1]):+.3f}]"
            lines.append(
                f"{model:<24}{n_runs:<8}{g_cell:<38}{f_cell:<38}{f'{g_pos}/{n_runs}':<8}{f'{f_pos}/{n_runs}':<8}"
            )

        pooled = hard_pivot.get("pooled", {})
        if pooled:
            total_runs = int(hard_pivot.get("total_runs", 0))
            g_mean = float(pooled.get("geometry_minus_baseline_mean", 0.0))
            g_ci = pooled.get("geometry_minus_baseline_ci", [0.0, 0.0])
            g_pos = int(pooled.get("geometry_positive_runs", 0))
            f_mean = float(pooled.get("full_minus_baseline_mean", 0.0))
            f_ci = pooled.get("full_minus_baseline_ci", [0.0, 0.0])
            f_pos = int(pooled.get("full_positive_runs", 0))
            g_cell = f"{g_mean:+.3f} [{float(g_ci[0]):+.3f}, {float(g_ci[1]):+.3f}]"
            f_cell = f"{f_mean:+.3f} [{float(f_ci[0]):+.3f}, {float(f_ci[1]):+.3f}]"
            lines.append("-" * 124)
            lines.append(
                f"{'POOLED':<24}{total_runs:<8}{g_cell:<38}{f_cell:<38}{f'{g_pos}/{total_runs}':<8}{f'{f_pos}/{total_runs}':<8}"
            )

        lines.append("")
        lines.append(
            "Strict gate verdict: "
            f"{hard_pivot.get('verdicts', {}).get('strict_predictive_gate', 'unknown')}"
        )
        lines.append("=" * 124)

        table_text = "\n".join(lines)
        with (output_dir / "table3_model_replication.txt").open("w") as f:
            f.write(table_text)

        print(table_text)
        logger.info("Saved Table 3: Model replication (hard pivot)")
        return

    sweep = results.get("predictive_validity_sweep")
    if sweep:
        aggregate = sweep.get("aggregate", {})
        records = sweep.get("records", [])
        if aggregate:
            lines = []
            lines.append("=" * 112)
            lines.append("TABLE 3: Predictive Validity Replication (Sweep Aggregate)")
            lines.append("=" * 112)
            lines.append("")
            lines.append(
                f"{'Model':<30}{'Runs':<8}{'Baseline(mean±sd)':<24}{'Geometry(mean±sd)':<24}{'Delta(mean±sd)':<20}{'PosDelta':<8}"
            )
            lines.append("-" * 112)
            for model_name in sorted(aggregate.keys()):
                row = aggregate[model_name]
                baseline_cell = (
                    f"{float(row.get('baseline_auc_mean', 0.0)):.3f}"
                    f"±{float(row.get('baseline_auc_std', 0.0)):.3f}"
                )
                geometry_cell = (
                    f"{float(row.get('geometry_auc_mean', 0.0)):.3f}"
                    f"±{float(row.get('geometry_auc_std', 0.0)):.3f}"
                )
                delta_cell = (
                    f"{float(row.get('delta_mean', 0.0)):+.3f}"
                    f"±{float(row.get('delta_std', 0.0)):.3f}"
                )
                lines.append(
                    f"{model_name:<30}"
                    f"{int(row.get('n_runs', 0)):<8d}"
                    f"{baseline_cell:<24}"
                    f"{geometry_cell:<24}"
                    f"{delta_cell:<20}"
                    f"{int(row.get('n_positive_delta', 0))}/{int(row.get('n_runs', 0)):<8d}"
                )

            if records:
                lines.append("")
                lines.append("Per-run:")
                lines.append("-" * 112)
                lines.append(
                    f"{'Model':<30}{'Seed':<8}{'Baseline':<12}{'Geometry':<12}{'Full':<12}{'Delta':<12}{'N':<8}"
                )
                for row in records:
                    lines.append(
                        f"{str(row.get('model_override', 'unknown')):<30}{int(row.get('seed', 0)):<8d}"
                        f"{float(row.get('baseline_auc', 0.0)):<12.3f}"
                        f"{float(row.get('geometry_auc', 0.0)):<12.3f}"
                        f"{float(row.get('full_auc', 0.0)):<12.3f}"
                        f"{float(row.get('delta_geometry_minus_baseline', 0.0)):<12.3f}"
                        f"{int(row.get('n_tokens', 0)):<8d}"
                    )

            lines.append("")
            lines.append("Note: Aggregate rows are preferred for publication claims.")
            lines.append("=" * 112)

            table_text = "\n".join(lines)
            with (output_dir / "table3_model_replication.txt").open("w") as f:
                f.write(table_text)

            print(table_text)
            logger.info("Saved Table 3: Model replication (sweep aggregate)")
            return

    runs = _latest_predictive_runs_by_model(results)
    if not runs:
        logger.warning("No predictive-validity runs found for Table 3")
        return

    lines = []
    lines.append("=" * 96)
    lines.append("TABLE 3: Predictive Validity Replication (Latest Run Per Model)")
    lines.append("=" * 96)
    lines.append("")
    lines.append(
        f"{'Model':<30}{'LabelSource':<15}{'BaselineAUC':<12}{'GeometryAUC':<12}{'Delta':<10}{'N':<8}"
    )
    lines.append("-" * 96)

    for run in runs:
        payload = run.get("payload", {})
        cfg = payload.get("config", {})
        model_name = cfg.get("model", "unknown")
        model_short = model_name.split("/")[-1][:28]
        use_sim = bool(cfg.get("use_simulated_failures", False))
        label_source = "simulated" if use_sim else "real"

        model_comp = payload.get("model_comparison", {})
        baseline_auc = float(model_comp.get("baseline", {}).get("auc", 0.0))
        geometry_auc = float(model_comp.get("geometry", {}).get("auc", 0.0))
        delta = geometry_auc - baseline_auc
        n_tokens = int(payload.get("summary", {}).get("n_tokens", 0))
        lines.append(
            f"{model_short:<30}{label_source:<15}{baseline_auc:<12.3f}{geometry_auc:<12.3f}{delta:<10.3f}{n_tokens:<8d}"
        )

    lines.append("")
    lines.append("Note: Use rows with LabelSource=real for publication claims.")
    lines.append("=" * 96)

    table_text = "\n".join(lines)
    with (output_dir / "table3_model_replication.txt").open("w") as f:
        f.write(table_text)

    print(table_text)
    logger.info("Saved Table 3: Model replication")


def _collect_generated_outputs(figures_dir: Path) -> list[Path]:
    generated: list[Path] = []
    for pattern in ("fig*.svg", "fig*.pdf", "fig*.png", "table*.txt"):
        generated.extend(sorted(figures_dir.glob(pattern)))
    return generated


def write_publication_manifest(
    *,
    repo_root: Path,
    figures_dir: Path,
    results: dict,
    paper_lock_path: Path | None,
    strict_lock: bool,
) -> tuple[Path, Path]:
    results_paths = results.get("_paths", {})
    source_artifacts = []
    for name, raw_path in sorted(results_paths.items()):
        artifact_path = _sanitize_manifest_source_path(str(raw_path))
        if not artifact_path.is_absolute():
            artifact_path = (repo_root / artifact_path).resolve()

        exists = artifact_path.exists()
        source_artifacts.append(
            {
                "name": name,
                "path": str(artifact_path),
                "exists": exists,
                "sha256": _file_sha256(artifact_path) if exists else None,
            }
        )

    generated_outputs = []
    for output_path in _collect_generated_outputs(figures_dir):
        generated_outputs.append(
            {
                "path": str(output_path.resolve()),
                "sha256": _file_sha256(output_path),
                "size_bytes": output_path.stat().st_size,
            }
        )

    manifest = {
        "manifest_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "repo": {
            "root": str(repo_root),
            **_git_metadata(repo_root),
        },
        "paper_lock": {
            "path": str(paper_lock_path.resolve()) if paper_lock_path else None,
            "sha256": (
                _file_sha256(paper_lock_path.resolve())
                if paper_lock_path and paper_lock_path.exists()
                else None
            ),
            "strict_lock": strict_lock,
        },
        "source_artifacts": source_artifacts,
        "generated_outputs": generated_outputs,
        "predictive_protocol_lock": results.get("_predictive_protocol_lock", {}),
    }

    json_path = figures_dir / "PUBLICATION_MANIFEST.json"
    md_path = figures_dir / "PUBLICATION_MANIFEST.md"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    lines = [
        "# Publication Manifest",
        "",
        f"- Generated at (UTC): `{manifest['generated_at_utc']}`",
        f"- Git branch: `{manifest['repo']['branch']}`",
        f"- Git commit: `{manifest['repo']['commit']}`",
        f"- Git dirty: `{manifest['repo']['is_dirty']}`",
        f"- Strict lock mode: `{strict_lock}`",
        "",
        "## Source Artifacts",
        "",
        "| Name | Path | Exists | SHA256 |",
        "|---|---|---:|---|",
    ]
    for row in source_artifacts:
        lines.append(
            f"| {row['name']} | `{row['path']}` | "
            f"{'yes' if row['exists'] else 'no'} | `{row['sha256'] or 'n/a'}` |"
        )
    lines.extend(
        [
            "",
            "## Generated Outputs",
            "",
            "| Path | Size (bytes) | SHA256 |",
            "|---|---:|---|",
        ]
    )
    for row in generated_outputs:
        lines.append(
            f"| `{row['path']}` | {row['size_bytes']} | `{row['sha256']}` |"
        )
    lines.append("")
    with md_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    logger.info("Saved publication manifest: %s and %s", json_path, md_path)
    return json_path, md_path


def main():
    """Generate all paper figures and tables."""
    parser = argparse.ArgumentParser(
        description="Generate publication figures/tables from locked DCTT artifacts."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent.parent / "outputs",
        help="Root output directory containing runs/ and sweeps/ directories.",
    )
    parser.add_argument(
        "--paper-lock",
        type=Path,
        default=Path(__file__).parent.parent / "configs" / "paper" / "publication_assets_lock.yaml",
        help="Path to publication lock YAML.",
    )
    parser.add_argument(
        "--strict-lock",
        action="store_true",
        help="Require all lock artifacts to be present and readable.",
    )
    parser.add_argument(
        "--pipeline-spec",
        type=Path,
        default=Path(__file__).parent.parent / "figures_src" / "pipeline_diagram_spec.yaml",
        help="Path to deterministic pipeline diagram spec YAML.",
    )
    parser.add_argument(
        "--skip-manifest",
        action="store_true",
        help="Skip publication manifest generation.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Output directory: {figures_dir}")

    paper_lock_path = args.paper_lock.resolve()
    if paper_lock_path.exists():
        results = load_results_from_paper_lock(
            output_dir=output_dir,
            paper_lock_path=paper_lock_path,
            strict_lock=args.strict_lock,
        )
        logger.info("Loaded results using paper lock: %s", paper_lock_path)
    else:
        if args.strict_lock:
            raise RuntimeError(f"Strict lock mode requested but lock not found: {paper_lock_path}")
        logger.warning(
            "Paper lock not found (%s). Falling back to heuristic latest-run loading.",
            paper_lock_path,
        )
        results = load_results(output_dir)
        paper_lock_path = None

    if not results:
        logger.warning("No results found. Run experiments first.")
        return

    logger.info(f"Found results for: {list(results.keys())}")

    # Generate figures
    figure0_pipeline_diagram(figures_dir, args.pipeline_spec.resolve())
    figure1_predictive_validity(results, figures_dir)
    figure2_cluster_repair_geometry(results, figures_dir)
    figure3_causal_geometry(results, figures_dir)
    figure4_model_replication(results, figures_dir)

    # Generate tables
    table1_main_results(results, figures_dir)
    table2_flagged_tokens(results, figures_dir)
    table3_model_replication(results, figures_dir)

    if not args.skip_manifest:
        write_publication_manifest(
            repo_root=Path(__file__).parent.parent.resolve(),
            figures_dir=figures_dir,
            results=results,
            paper_lock_path=paper_lock_path,
            strict_lock=args.strict_lock,
        )

    logger.info(f"\nAll figures and tables saved to {figures_dir}")


if __name__ == "__main__":
    main()
