#!/usr/bin/env python3
"""Evaluate go/no-go gate for predictive-validity sweep outputs.

Example:
    python scripts/evaluate_predictive_gate.py \
      --sweep-results outputs/sweeps/predictive_validity/2026-02-24_06-04-00/sweep_results.json \
      --output-json outputs/sweeps/predictive_validity/2026-02-24_06-04-00/gate_evaluation.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import t


def _load_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def _delta_stats(values: list[float], confidence_level: float) -> dict[str, Any]:
    arr = np.asarray(values, dtype=float)
    n = int(arr.size)
    if n == 0:
        return {
            "n": 0,
            "mean": None,
            "std": None,
            "ci_low": None,
            "ci_high": None,
        }

    mean = float(np.mean(arr))
    if n == 1:
        return {
            "n": 1,
            "mean": mean,
            "std": 0.0,
            "ci_low": None,
            "ci_high": None,
        }

    std = float(np.std(arr, ddof=1))
    sem = std / np.sqrt(float(n))
    crit = float(t.ppf((1.0 + confidence_level) / 2.0, df=n - 1))
    half_width = crit * sem
    return {
        "n": n,
        "mean": mean,
        "std": std,
        "ci_low": float(mean - half_width),
        "ci_high": float(mean + half_width),
    }


def _fmt(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _render_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Predictive Gate Evaluation")
    lines.append("")
    lines.append(f"- Verdict: **{report['verdict']}**")
    lines.append(f"- Sweep: `{report['sweep_results_path']}`")
    lines.append(f"- Confidence level: `{report['confidence_level']}`")
    lines.append(f"- Min runs/model: `{report['criteria']['min_runs_per_model']}`")
    lines.append(
        "- Require positive per-model CI lower bound: "
        f"`{report['criteria']['require_positive_delta_per_model']}`"
    )
    lines.append(
        "- Require positive pooled CI lower bound: "
        f"`{report['criteria']['require_positive_pooled_delta']}`"
    )
    lines.append("")
    lines.append("## Per-model")
    lines.append("")
    lines.append("| Model | n | mean delta | ci_low | ci_high |")
    lines.append("|---|---:|---:|---:|---:|")
    for model, stats in sorted(report["per_model"].items()):
        lines.append(
            f"| {model} | {stats['n']} | {_fmt(stats['mean'])} | "
            f"{_fmt(stats['ci_low'])} | {_fmt(stats['ci_high'])} |"
        )
    lines.append("")
    lines.append("## Pooled")
    lines.append("")
    pooled = report["pooled"]
    lines.append(
        f"- n={pooled['n']}, mean={_fmt(pooled['mean'])}, "
        f"ci=[{_fmt(pooled['ci_low'])}, {_fmt(pooled['ci_high'])}]"
    )
    if report["reasons"]:
        lines.append("")
        lines.append("## Gate Fail Reasons")
        lines.append("")
        for reason in report["reasons"]:
            lines.append(f"- {reason}")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sweep-results",
        required=True,
        help="Path to sweep_results.json",
    )
    parser.add_argument(
        "--confidence-level",
        type=float,
        default=0.95,
        help="Confidence level for mean-delta t-intervals.",
    )
    parser.add_argument(
        "--min-runs-per-model",
        type=int,
        default=5,
        help="Minimum required runs per model for gate pass.",
    )
    parser.add_argument(
        "--require-positive-delta-per-model",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require each model's CI lower bound for delta_mean to be > 0.",
    )
    parser.add_argument(
        "--require-positive-pooled-delta",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require pooled CI lower bound across all runs to be > 0.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional JSON output path for gate report.",
    )
    parser.add_argument(
        "--output-markdown",
        default=None,
        help="Optional markdown output path for gate report.",
    )
    args = parser.parse_args()

    sweep_path = Path(args.sweep_results).expanduser().resolve()
    if not sweep_path.exists():
        raise FileNotFoundError(f"Missing sweep results: {sweep_path}")

    payload = _load_json(sweep_path)
    records = payload.get("records", [])
    if not records:
        raise RuntimeError(f"Sweep has no records: {sweep_path}")

    deltas_by_model: dict[str, list[float]] = {}
    all_deltas: list[float] = []
    for row in records:
        model = str(row.get("model_override", "unknown"))
        delta = float(row.get("delta_geometry_minus_baseline", 0.0))
        deltas_by_model.setdefault(model, []).append(delta)
        all_deltas.append(delta)

    per_model = {
        model: _delta_stats(values, confidence_level=float(args.confidence_level))
        for model, values in deltas_by_model.items()
    }
    pooled = _delta_stats(all_deltas, confidence_level=float(args.confidence_level))

    reasons: list[str] = []
    for model, stats in sorted(per_model.items()):
        if int(stats["n"]) < int(args.min_runs_per_model):
            reasons.append(
                f"{model}: only {stats['n']} runs (min required {args.min_runs_per_model})"
            )
        if bool(args.require_positive_delta_per_model):
            ci_low = stats["ci_low"]
            if ci_low is None:
                reasons.append(f"{model}: CI unavailable (need at least 2 runs)")
            elif float(ci_low) <= 0.0:
                reasons.append(
                    f"{model}: CI lower bound <= 0 ({float(ci_low):.3f})"
                )

    if bool(args.require_positive_pooled_delta):
        pooled_ci_low = pooled["ci_low"]
        if pooled_ci_low is None:
            reasons.append("pooled: CI unavailable (need at least 2 runs)")
        elif float(pooled_ci_low) <= 0.0:
            reasons.append(
                f"pooled: CI lower bound <= 0 ({float(pooled_ci_low):.3f})"
            )

    verdict = "PASS" if not reasons else "FAIL"
    report = {
        "verdict": verdict,
        "sweep_results_path": str(sweep_path),
        "confidence_level": float(args.confidence_level),
        "criteria": {
            "min_runs_per_model": int(args.min_runs_per_model),
            "require_positive_delta_per_model": bool(args.require_positive_delta_per_model),
            "require_positive_pooled_delta": bool(args.require_positive_pooled_delta),
        },
        "per_model": per_model,
        "pooled": pooled,
        "reasons": reasons,
    }

    print(f"Predictive gate verdict: {verdict}")
    for model, stats in sorted(per_model.items()):
        print(
            f"- {model}: n={stats['n']} mean={_fmt(stats['mean'])} "
            f"ci=[{_fmt(stats['ci_low'])}, {_fmt(stats['ci_high'])}]"
        )
    print(
        f"- pooled: n={pooled['n']} mean={_fmt(pooled['mean'])} "
        f"ci=[{_fmt(pooled['ci_low'])}, {_fmt(pooled['ci_high'])}]"
    )
    if reasons:
        print("Reasons:")
        for reason in reasons:
            print(f"  - {reason}")

    if args.output_json:
        output_json = Path(args.output_json).expanduser().resolve()
        output_json.parent.mkdir(parents=True, exist_ok=True)
        with output_json.open("w") as f:
            json.dump(report, f, indent=2)
        print(f"Wrote JSON report: {output_json}")

    if args.output_markdown:
        output_md = Path(args.output_markdown).expanduser().resolve()
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(_render_markdown(report))
        print(f"Wrote markdown report: {output_md}")


if __name__ == "__main__":
    main()
