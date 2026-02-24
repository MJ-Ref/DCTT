#!/usr/bin/env python3
"""Build a consolidated hard-pivot evidence report from strict sweep artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats


def _parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _t_ci(values: np.ndarray, confidence_level: float) -> list[float]:
    n = int(values.size)
    if n <= 0:
        return [0.0, 0.0]
    if n == 1:
        v = float(values[0])
        return [v, v]

    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1))
    alpha = 1.0 - confidence_level
    tcrit = float(stats.t.ppf(1.0 - alpha / 2.0, df=n - 1))
    half_width = float(tcrit * std / np.sqrt(n))
    return [mean - half_width, mean + half_width]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _collect_sweep_data(
    sweep_root: Path,
    stamps: list[str],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    all_records: list[dict[str, Any]] = []
    per_sweep: dict[str, Any] = {}
    per_model_records: dict[str, list[dict[str, Any]]] = {}

    for stamp in stamps:
        sweep_dir = sweep_root / stamp
        sweep_path = sweep_dir / "sweep_results.json"
        gate_path = sweep_dir / "gate_evaluation.json"
        if not sweep_path.exists():
            raise FileNotFoundError(f"Missing sweep results: {sweep_path}")
        if not gate_path.exists():
            raise FileNotFoundError(f"Missing gate evaluation: {gate_path}")

        sweep_payload = _load_json(sweep_path)
        gate_payload = _load_json(gate_path)
        records = sweep_payload.get("records", [])
        for row in records:
            enriched = dict(row)
            enriched["sweep_stamp"] = stamp
            baseline = float(row.get("baseline_auc", 0.0))
            full_auc = float(row.get("full_auc", 0.0))
            enriched["delta_full_minus_baseline"] = full_auc - baseline
            all_records.append(enriched)
            model_name = str(row.get("model_override", "unknown"))
            per_model_records.setdefault(model_name, []).append(enriched)

        per_sweep[stamp] = {
            "sweep_results_path": str(sweep_path),
            "gate_evaluation_path": str(gate_path),
            "gate_verdict": str(gate_payload.get("verdict", "UNKNOWN")),
            "gate_pooled": gate_payload.get("pooled", {}),
            "aggregate": sweep_payload.get("aggregate", {}),
            "n_records": int(len(records)),
        }

    return all_records, per_sweep, per_model_records


def _build_report(
    *,
    all_records: list[dict[str, Any]],
    per_sweep: dict[str, Any],
    per_model_records: dict[str, list[dict[str, Any]]],
    confidence_level: float,
) -> dict[str, Any]:
    geom_delta = np.array(
        [float(r.get("delta_geometry_minus_baseline", 0.0)) for r in all_records],
        dtype=float,
    )
    full_delta = np.array(
        [float(r.get("delta_full_minus_baseline", 0.0)) for r in all_records],
        dtype=float,
    )

    per_model_summary: dict[str, Any] = {}
    for model_name, rows in sorted(per_model_records.items()):
        model_geom_delta = np.array(
            [float(r.get("delta_geometry_minus_baseline", 0.0)) for r in rows],
            dtype=float,
        )
        model_full_delta = np.array(
            [float(r.get("delta_full_minus_baseline", 0.0)) for r in rows],
            dtype=float,
        )
        per_model_summary[model_name] = {
            "n_runs": int(model_geom_delta.size),
            "geometry_minus_baseline_mean": float(np.mean(model_geom_delta)),
            "geometry_minus_baseline_ci": _t_ci(model_geom_delta, confidence_level),
            "geometry_positive_runs": int(np.sum(model_geom_delta > 0.0)),
            "full_minus_baseline_mean": float(np.mean(model_full_delta)),
            "full_minus_baseline_ci": _t_ci(model_full_delta, confidence_level),
            "full_positive_runs": int(np.sum(model_full_delta > 0.0)),
        }

    pooled_geom_ci = _t_ci(geom_delta, confidence_level)
    pooled_full_ci = _t_ci(full_delta, confidence_level)
    strict_predictive_pass = bool(pooled_geom_ci[0] > 0.0)

    report = {
        "confidence_level": float(confidence_level),
        "sweeps": per_sweep,
        "total_runs": int(len(all_records)),
        "models": sorted(per_model_summary.keys()),
        "pooled": {
            "geometry_minus_baseline_mean": float(np.mean(geom_delta)),
            "geometry_minus_baseline_ci": pooled_geom_ci,
            "geometry_positive_runs": int(np.sum(geom_delta > 0.0)),
            "full_minus_baseline_mean": float(np.mean(full_delta)),
            "full_minus_baseline_ci": pooled_full_ci,
            "full_positive_runs": int(np.sum(full_delta > 0.0)),
        },
        "per_model": per_model_summary,
        "verdicts": {
            "strict_predictive_gate": "PASS" if strict_predictive_pass else "FAIL",
            "hard_pivot_recommended": not strict_predictive_pass,
        },
        "interpretation": {
            "summary": (
                "Geometry-only predictive signal remains negative under strict controls."
                if not strict_predictive_pass
                else "Geometry-only predictive signal is positive under strict controls."
            ),
            "paper_position": (
                "Mechanistic intervention + rigorous negative predictive result"
                if not strict_predictive_pass
                else "Mechanistic intervention + supported predictive signal"
            ),
        },
    }
    return report


def _write_markdown(report: dict[str, Any], output_path: Path) -> None:
    pooled = report["pooled"]
    lines = []
    lines.append("# Hard Pivot Evidence Report")
    lines.append("")
    lines.append(f"- Confidence level: `{report['confidence_level']:.2f}`")
    lines.append(f"- Total runs: `{report['total_runs']}`")
    lines.append(f"- Models: `{', '.join(report['models'])}`")
    lines.append(f"- Strict predictive gate verdict: **{report['verdicts']['strict_predictive_gate']}**")
    lines.append(
        f"- Recommended paper position: `{report['interpretation']['paper_position']}`"
    )
    lines.append("")
    lines.append("## Pooled Effects")
    lines.append("")
    lines.append(
        f"- Geometry - baseline: mean `{pooled['geometry_minus_baseline_mean']:.6f}`, "
        f"CI `{pooled['geometry_minus_baseline_ci'][0]:.6f}` to "
        f"`{pooled['geometry_minus_baseline_ci'][1]:.6f}`, "
        f"positive runs `{pooled['geometry_positive_runs']}/{report['total_runs']}`"
    )
    lines.append(
        f"- Full - baseline: mean `{pooled['full_minus_baseline_mean']:.6f}`, "
        f"CI `{pooled['full_minus_baseline_ci'][0]:.6f}` to "
        f"`{pooled['full_minus_baseline_ci'][1]:.6f}`, "
        f"positive runs `{pooled['full_positive_runs']}/{report['total_runs']}`"
    )
    lines.append("")
    lines.append("## Per-Model Effects")
    lines.append("")
    lines.append("| Model | n | Geometry-baseline mean | Geometry CI | Full-baseline mean | Full CI |")
    lines.append("|---|---:|---:|---|---:|---|")
    for model_name, row in report["per_model"].items():
        lines.append(
            f"| {model_name} | {row['n_runs']} | "
            f"{row['geometry_minus_baseline_mean']:.6f} | "
            f"[{row['geometry_minus_baseline_ci'][0]:.6f}, {row['geometry_minus_baseline_ci'][1]:.6f}] | "
            f"{row['full_minus_baseline_mean']:.6f} | "
            f"[{row['full_minus_baseline_ci'][0]:.6f}, {row['full_minus_baseline_ci'][1]:.6f}] |"
        )
    lines.append("")
    lines.append("## Sweep Gate Status")
    lines.append("")
    for stamp, row in report["sweeps"].items():
        lines.append(
            f"- `{stamp}`: `{row['gate_verdict']}` "
            f"(records={row['n_records']})"
        )
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(report["interpretation"]["summary"])
    lines.append("")
    lines.append(
        "This artifact is intended to lock the pivot narrative to computed results."
    )
    output_path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sweep-stamps",
        default="2026-02-24_06-43-58,2026-02-24_07-31-10,2026-02-24_07-34-45",
        help="Comma-separated sweep stamps to aggregate.",
    )
    parser.add_argument(
        "--sweep-root",
        default="outputs/sweeps/predictive_validity",
        help="Root containing per-stamp sweep folders.",
    )
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument(
        "--output-json",
        default="outputs/sweeps/predictive_validity/HARD_PIVOT_REPORT.json",
    )
    parser.add_argument(
        "--output-markdown",
        default="outputs/sweeps/predictive_validity/HARD_PIVOT_REPORT.md",
    )
    args = parser.parse_args()

    sweep_root = Path(args.sweep_root).resolve()
    stamps = _parse_csv(args.sweep_stamps)
    if not stamps:
        raise ValueError("No sweep stamps provided.")

    all_records, per_sweep, per_model_records = _collect_sweep_data(
        sweep_root=sweep_root,
        stamps=stamps,
    )
    if not all_records:
        raise RuntimeError("No records found in provided sweeps.")

    report = _build_report(
        all_records=all_records,
        per_sweep=per_sweep,
        per_model_records=per_model_records,
        confidence_level=float(args.confidence_level),
    )

    output_json = Path(args.output_json).resolve()
    output_md = Path(args.output_markdown).resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2))
    _write_markdown(report, output_md)

    print(f"Wrote hard pivot JSON: {output_json}")
    print(f"Wrote hard pivot markdown: {output_md}")
    print(f"Strict predictive gate: {report['verdicts']['strict_predictive_gate']}")


if __name__ == "__main__":
    main()
