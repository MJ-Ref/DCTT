#!/usr/bin/env python3
"""Verify submission package integrity against a hash-pinned lock file.

Checks performed:
1) Every artifact listed in lock exists and matches SHA256.
2) Hard-pivot claim checks match HARD_PIVOT_REPORT.json (within tolerance).
3) PUBLICATION_MANIFEST.json is internally consistent (all listed hashes resolve).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import yaml


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_path(raw_path: str, repo_root: Path) -> Path:
    p = Path(raw_path)
    return p if p.is_absolute() else (repo_root / p).resolve()


def as_float(value: Any) -> float:
    return float(value)


def check_close(name: str, actual: float, expected: float, tol: float, errors: list[str]) -> None:
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=tol):
        errors.append(
            f"{name}: expected {expected:.12g}, got {actual:.12g} (tol={tol})"
        )


def validate_manifest(manifest_path: Path, errors: list[str]) -> None:
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover
        errors.append(f"Failed to parse manifest JSON {manifest_path}: {exc}")
        return

    sections = ["source_artifacts", "generated_outputs"]
    for section in sections:
        rows = manifest.get(section, [])
        if not isinstance(rows, list):
            errors.append(f"Manifest section '{section}' is not a list")
            continue

        for row in rows:
            path_str = row.get("path")
            expected_hash = row.get("sha256")
            exists_flag = bool(row.get("exists", True))
            if not isinstance(path_str, str) or not path_str:
                errors.append(f"Manifest row missing path in section '{section}': {row}")
                continue
            if not isinstance(expected_hash, str) or not expected_hash:
                errors.append(
                    f"Manifest row missing sha256 in section '{section}': {path_str}"
                )
                continue

            file_path = Path(path_str)
            if not file_path.exists():
                if exists_flag:
                    errors.append(f"Manifest path missing: {file_path}")
                continue

            actual_hash = sha256_file(file_path)
            if actual_hash != expected_hash:
                errors.append(
                    f"Manifest hash mismatch for {file_path}: expected {expected_hash}, got {actual_hash}"
                )


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify hard-pivot submission package")
    parser.add_argument(
        "--lock",
        type=Path,
        default=Path("configs/paper/submission_package_lock.yaml"),
        help="Path to submission package lock YAML",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="Repository root for resolving relative paths",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only print failure details",
    )
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    lock_path = resolve_path(str(args.lock), repo_root)

    if not lock_path.exists():
        print(f"ERROR: lock file not found: {lock_path}")
        return 1

    try:
        lock = yaml.safe_load(lock_path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        print(f"ERROR: failed to parse lock file {lock_path}: {exc}")
        return 1

    errors: list[str] = []
    artifact_index: dict[str, Path] = {}

    artifacts = lock.get("artifacts", [])
    if not isinstance(artifacts, list) or not artifacts:
        errors.append("Lock file must define non-empty 'artifacts' list")
    else:
        for item in artifacts:
            if not isinstance(item, dict):
                errors.append(f"Invalid artifact entry (not mapping): {item}")
                continue
            artifact_id = item.get("id")
            raw_path = item.get("path")
            expected_hash = item.get("sha256")
            if not isinstance(artifact_id, str) or not artifact_id:
                errors.append(f"Artifact missing id: {item}")
                continue
            if not isinstance(raw_path, str) or not raw_path:
                errors.append(f"Artifact '{artifact_id}' missing path")
                continue

            file_path = resolve_path(raw_path, repo_root)
            artifact_index[artifact_id] = file_path
            if not file_path.exists():
                errors.append(f"Artifact missing: {artifact_id} -> {file_path}")
                continue

            if expected_hash is not None:
                if not isinstance(expected_hash, str) or not expected_hash:
                    errors.append(f"Artifact '{artifact_id}' has invalid sha256 field")
                    continue
                actual_hash = sha256_file(file_path)
                if actual_hash != expected_hash:
                    errors.append(
                        f"Artifact hash mismatch {artifact_id}: expected {expected_hash}, got {actual_hash}"
                    )

    hard_pivot_path = artifact_index.get("hard_pivot_report_json")
    claim_checks = lock.get("claim_checks", {})
    tolerances = lock.get("tolerances", {})
    float_tol = float(tolerances.get("float_abs", 1e-9))

    if hard_pivot_path and hard_pivot_path.exists() and isinstance(claim_checks, dict):
        try:
            hard_pivot = json.loads(hard_pivot_path.read_text(encoding="utf-8"))
        except Exception as exc:
            errors.append(f"Failed to parse hard pivot report JSON: {exc}")
            hard_pivot = {}

        verdict = hard_pivot.get("verdicts", {}).get("strict_predictive_gate")
        expected_verdict = claim_checks.get("strict_predictive_gate")
        if expected_verdict is not None and verdict != expected_verdict:
            errors.append(
                f"strict_predictive_gate mismatch: expected {expected_verdict}, got {verdict}"
            )

        total_runs = hard_pivot.get("total_runs")
        expected_runs = claim_checks.get("total_runs")
        if expected_runs is not None and int(total_runs) != int(expected_runs):
            errors.append(f"total_runs mismatch: expected {expected_runs}, got {total_runs}")

        models = hard_pivot.get("models", [])
        expected_model_count = claim_checks.get("model_count")
        if expected_model_count is not None and len(models) != int(expected_model_count):
            errors.append(
                f"model_count mismatch: expected {expected_model_count}, got {len(models)}"
            )

        pooled = hard_pivot.get("pooled", {})
        numeric_checks = [
            ("pooled_geometry_minus_baseline_mean", pooled.get("geometry_minus_baseline_mean")),
            ("pooled_full_minus_baseline_mean", pooled.get("full_minus_baseline_mean")),
        ]
        for key, actual_value in numeric_checks:
            if key in claim_checks and actual_value is not None:
                check_close(key, as_float(actual_value), as_float(claim_checks[key]), float_tol, errors)

        int_checks = [
            ("pooled_geometry_positive_runs", pooled.get("geometry_positive_runs")),
            ("pooled_full_positive_runs", pooled.get("full_positive_runs")),
        ]
        for key, actual_value in int_checks:
            if key in claim_checks and actual_value is not None and int(actual_value) != int(claim_checks[key]):
                errors.append(f"{key} mismatch: expected {claim_checks[key]}, got {actual_value}")

        ci_checks = [
            ("pooled_geometry_ci", pooled.get("geometry_minus_baseline_ci")),
            ("pooled_full_ci", pooled.get("full_minus_baseline_ci")),
        ]
        for key, actual_ci in ci_checks:
            expected_ci = claim_checks.get(key)
            if expected_ci is None:
                continue
            if not isinstance(actual_ci, list) or len(actual_ci) != 2:
                errors.append(f"{key} malformed in hard pivot report: {actual_ci}")
                continue
            check_close(f"{key}[0]", as_float(actual_ci[0]), as_float(expected_ci[0]), float_tol, errors)
            check_close(f"{key}[1]", as_float(actual_ci[1]), as_float(expected_ci[1]), float_tol, errors)

    verify_cfg = lock.get("verification", {})
    if isinstance(verify_cfg, dict) and verify_cfg.get("require_manifest_consistency", False):
        manifest_raw = verify_cfg.get("manifest_path")
        if manifest_raw is None:
            manifest_path = artifact_index.get("publication_manifest_json")
        else:
            manifest_path = resolve_path(str(manifest_raw), repo_root)
        if manifest_path is None:
            errors.append("Manifest consistency required but no manifest path provided")
        elif not manifest_path.exists():
            errors.append(f"Manifest consistency required but file missing: {manifest_path}")
        else:
            validate_manifest(manifest_path, errors)

    if not args.quiet:
        print(f"Lock file: {lock_path}")
        print(f"Artifacts checked: {len(artifacts) if isinstance(artifacts, list) else 0}")

    if errors:
        print("\nFAIL: submission package verification failed")
        for err in errors:
            print(f"- {err}")
        return 1

    print("\nPASS: submission package verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
