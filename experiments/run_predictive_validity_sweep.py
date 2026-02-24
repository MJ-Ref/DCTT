#!/usr/bin/env python3
"""Run multi-model, multi-seed predictive-validity sweeps.

This script orchestrates repeated real-label predictive-validity runs and
writes aggregate artifacts for acceptance tracking.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


def _parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_int_csv(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _as_finite_float(value: Any, default: float = 0.0) -> float:
    """Convert value to float and guard against NaN/Inf in artifacts."""
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(parsed):
        return default
    return parsed


def _latest_run_dir(repo_root: Path) -> Path | None:
    runs_root = repo_root / "outputs" / "runs"
    if not runs_root.exists():
        return None

    run_dirs = [
        path
        for path in runs_root.glob("*/*")
        if path.is_dir()
    ]
    if not run_dirs:
        return None
    return max(run_dirs, key=lambda path: path.stat().st_mtime)


def _extract_run_dir_from_logs(stdout: str, stderr: str) -> Path | None:
    pattern = re.compile(r"Results saved to (.+)")
    text = f"{stdout}\n{stderr}"
    matches = pattern.findall(text)
    if not matches:
        return None
    return Path(matches[-1].strip())


def _run_single(
    repo_root: Path,
    *,
    model: str,
    seed: int,
    sample_size: int,
    n_prompts: int,
    max_new_tokens: int,
    n_bootstrap: int,
    scoring_mode: str,
    min_logprob_margin: float,
    compute_device: str | None,
) -> dict[str, Any]:
    before_latest = _latest_run_dir(repo_root)
    cmd = [
        sys.executable,
        str(repo_root / "experiments" / "run_predictive_validity.py"),
        f"model={model}",
        f"seed={seed}",
        "predictive_validity.use_simulated_failures=false",
        "experiment.tokens.mode=sample",
        f"experiment.tokens.sample_size={sample_size}",
        f"stress_test.n_prompts={n_prompts}",
        f"stress_test.max_new_tokens={max_new_tokens}",
        f"stress_test.scoring_mode={scoring_mode}",
        f"stress_test.min_logprob_margin={min_logprob_margin}",
        f"predictive_validity.n_bootstrap={n_bootstrap}",
        "wandb.enabled=false",
    ]
    if compute_device:
        cmd.append(f"compute.device={compute_device}")

    proc = subprocess.run(
        cmd,
        cwd=repo_root,
        capture_output=True,
        text=True,
    )

    run_dir = _extract_run_dir_from_logs(proc.stdout, proc.stderr)
    if run_dir is None:
        after_latest = _latest_run_dir(repo_root)
        if after_latest is not None and after_latest != before_latest:
            run_dir = after_latest

    if proc.returncode != 0:
        raise RuntimeError(
            "Predictive-validity run failed.\n"
            f"model={model} seed={seed}\n"
            f"stderr_tail:\n{proc.stderr[-4000:]}"
        )

    if run_dir is None:
        raise RuntimeError(
            f"Could not identify run directory for model={model} seed={seed}"
        )

    result_path = run_dir / "predictive_validity_results.json"
    if not result_path.exists():
        raise RuntimeError(f"Missing result file: {result_path}")

    payload = json.loads(result_path.read_text())
    model_comp = payload.get("model_comparison", {})
    baseline_auc = float(model_comp.get("baseline", {}).get("auc", 0.0))
    geometry_auc = float(model_comp.get("geometry", {}).get("auc", 0.0))
    full_auc = float(model_comp.get("full", {}).get("auc", 0.0))
    delta = geometry_auc - baseline_auc

    stress_summary = payload.get("stress_test_summary", {})
    return {
        "model_override": model,
        "seed": seed,
        "run_dir": str(run_dir),
        "result_path": str(result_path),
        "baseline_auc": baseline_auc,
        "geometry_auc": geometry_auc,
        "full_auc": full_auc,
        "delta_geometry_minus_baseline": delta,
        "n_tokens": int(payload.get("summary", {}).get("n_tokens", 0)),
        "positive_rate": _as_finite_float(
            payload.get("summary", {}).get("positive_rate", 0.0),
        ),
        "scoring_mode": stress_summary.get("scoring_mode", "unknown"),
        "mean_failure_gap": _as_finite_float(
            stress_summary.get("mean_failure_gap", 0.0),
        ),
        "mean_target_margin": _as_finite_float(
            stress_summary.get("mean_target_margin", 0.0),
        ),
        "mean_control_margin": _as_finite_float(
            stress_summary.get("mean_control_margin", 0.0),
        ),
    }


def _aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_model: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_model.setdefault(record["model_override"], []).append(record)

    summary: dict[str, Any] = {}
    for model, model_records in by_model.items():
        baseline = np.array([row["baseline_auc"] for row in model_records], dtype=float)
        geometry = np.array([row["geometry_auc"] for row in model_records], dtype=float)
        full = np.array([row["full_auc"] for row in model_records], dtype=float)
        delta = np.array(
            [row["delta_geometry_minus_baseline"] for row in model_records],
            dtype=float,
        )
        summary[model] = {
            "n_runs": int(len(model_records)),
            "baseline_auc_mean": float(np.mean(baseline)),
            "baseline_auc_std": float(np.std(baseline)),
            "geometry_auc_mean": float(np.mean(geometry)),
            "geometry_auc_std": float(np.std(geometry)),
            "full_auc_mean": float(np.mean(full)),
            "full_auc_std": float(np.std(full)),
            "delta_mean": float(np.mean(delta)),
            "delta_std": float(np.std(delta)),
            "n_positive_delta": int(np.sum(delta > 0.0)),
            "n_non_positive_delta": int(np.sum(delta <= 0.0)),
        }
    return summary


def _write_markdown(
    output_dir: Path,
    *,
    config: dict[str, Any],
    records: list[dict[str, Any]],
    aggregate: dict[str, Any],
) -> None:
    lines = []
    lines.append("# Predictive Validity Sweep Summary")
    lines.append("")
    lines.append("## Config")
    lines.append("")
    lines.append(f"- Models: `{', '.join(config['models'])}`")
    lines.append(f"- Seeds: `{', '.join(str(s) for s in config['seeds'])}`")
    lines.append(f"- Sample size: `{config['sample_size']}`")
    lines.append(f"- Prompts per token: `{config['n_prompts']}`")
    lines.append(f"- Scoring mode: `{config['scoring_mode']}`")
    lines.append(f"- Min logprob margin: `{config['min_logprob_margin']}`")
    if config.get("compute_device"):
        lines.append(f"- Compute device override: `{config['compute_device']}`")
    lines.append("")
    lines.append("## Aggregate")
    lines.append("")
    lines.append("| Model | Runs | Baseline AUC (mean±sd) | Geometry AUC (mean±sd) | Delta (mean±sd) | Positive Delta |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for model, row in aggregate.items():
        lines.append(
            f"| {model} | {row['n_runs']} | "
            f"{row['baseline_auc_mean']:.3f} ± {row['baseline_auc_std']:.3f} | "
            f"{row['geometry_auc_mean']:.3f} ± {row['geometry_auc_std']:.3f} | "
            f"{row['delta_mean']:.3f} ± {row['delta_std']:.3f} | "
            f"{row['n_positive_delta']}/{row['n_runs']} |"
        )

    lines.append("")
    lines.append("## Per-Run")
    lines.append("")
    lines.append("| Model | Seed | Baseline AUC | Geometry AUC | Full AUC | Delta | Tokens | Run Dir |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---|")
    for row in records:
        lines.append(
            f"| {row['model_override']} | {row['seed']} | {row['baseline_auc']:.3f} | "
            f"{row['geometry_auc']:.3f} | {row['full_auc']:.3f} | "
            f"{row['delta_geometry_minus_baseline']:.3f} | {row['n_tokens']} | "
            f"`{row['run_dir']}` |"
        )

    (output_dir / "sweep_summary.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        default="qwen2_5_coder_7b,qwen2_5_7b",
        help="Comma-separated model config names.",
    )
    parser.add_argument(
        "--seeds",
        default="42,43,44",
        help="Comma-separated seeds.",
    )
    parser.add_argument("--sample-size", type=int, default=300)
    parser.add_argument("--n-prompts", type=int, default=6)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--n-bootstrap", type=int, default=50)
    parser.add_argument("--scoring-mode", default="logprob_choice")
    parser.add_argument("--min-logprob-margin", type=float, default=0.0)
    parser.add_argument(
        "--compute-device",
        default=None,
        help="Optional compute.device override for run_predictive_validity.py.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output dir for sweep summary artifacts.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    models = _parse_csv(args.models)
    seeds = _parse_int_csv(args.seeds)
    if not models:
        raise ValueError("No models provided.")
    if not seeds:
        raise ValueError("No seeds provided.")

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_dir = repo_root / "outputs" / "sweeps" / "predictive_validity" / stamp
    output_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    for model in models:
        for seed in seeds:
            print(f"[run] model={model} seed={seed}", flush=True)
            record = _run_single(
                repo_root,
                model=model,
                seed=seed,
                sample_size=args.sample_size,
                n_prompts=args.n_prompts,
                max_new_tokens=args.max_new_tokens,
                n_bootstrap=args.n_bootstrap,
                scoring_mode=args.scoring_mode,
                min_logprob_margin=args.min_logprob_margin,
                compute_device=args.compute_device,
            )
            records.append(record)
            print(
                f"[done] model={model} seed={seed} "
                f"delta={record['delta_geometry_minus_baseline']:+.3f} "
                f"run={record['run_dir']}",
                flush=True,
            )

    aggregate = _aggregate(records)
    config = {
        "models": models,
        "seeds": seeds,
        "sample_size": int(args.sample_size),
        "n_prompts": int(args.n_prompts),
        "max_new_tokens": int(args.max_new_tokens),
        "n_bootstrap": int(args.n_bootstrap),
        "scoring_mode": str(args.scoring_mode),
        "min_logprob_margin": float(args.min_logprob_margin),
        "compute_device": args.compute_device,
    }

    payload = {
        "config": config,
        "records": records,
        "aggregate": aggregate,
    }
    (output_dir / "sweep_results.json").write_text(json.dumps(payload, indent=2))
    _write_markdown(output_dir, config=config, records=records, aggregate=aggregate)

    print(f"\nSweep artifacts written to: {output_dir}")
    for model, row in aggregate.items():
        print(
            f"- {model}: delta_mean={row['delta_mean']:+.3f} "
            f"(positive={row['n_positive_delta']}/{row['n_runs']})"
        )


if __name__ == "__main__":
    main()
