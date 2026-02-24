#!/usr/bin/env python3
"""Freeze submission package artifacts and refresh lock hashes.

Workflow:
1. Regenerate camera-ready assets with strict lock mode.
2. Recompute SHA256 for all artifacts listed in submission lock.
3. Update submission lock hashes and timestamp.
4. Verify package integrity.
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_cmd(cmd: list[str], cwd: Path) -> None:
    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed ({result.returncode}): {' '.join(cmd)}")


def resolve_path(raw_path: str, repo_root: Path) -> Path:
    p = Path(raw_path)
    return p if p.is_absolute() else (repo_root / p).resolve()


def assert_git_clean(repo_root: Path) -> None:
    out = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=repo_root, text=True
    ).strip()
    if out:
        raise RuntimeError("Working tree must be clean before freezing package")


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze DCTT submission package")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="Repository root",
    )
    parser.add_argument(
        "--paper-lock",
        type=Path,
        default=Path("configs/paper/publication_assets_lock.yaml"),
        help="Paper lock used for figure generation",
    )
    parser.add_argument(
        "--submission-lock",
        type=Path,
        default=Path("configs/paper/submission_package_lock.yaml"),
        help="Submission package lock to refresh",
    )
    parser.add_argument(
        "--pipeline-spec",
        type=Path,
        default=Path("figures_src/pipeline_diagram_spec.yaml"),
        help="Pipeline spec for figure generation",
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Allow running on a dirty working tree",
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip final verification pass",
    )
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    paper_lock = resolve_path(str(args.paper_lock), repo_root)
    submission_lock = resolve_path(str(args.submission_lock), repo_root)
    pipeline_spec = resolve_path(str(args.pipeline_spec), repo_root)

    if not args.allow_dirty:
        assert_git_clean(repo_root)

    if not submission_lock.exists():
        raise RuntimeError(f"Submission lock not found: {submission_lock}")

    print("[1/4] Regenerating camera-ready assets...")
    run_cmd(
        [
            sys.executable,
            "scripts/generate_paper_figures.py",
            "--paper-lock",
            str(paper_lock),
            "--pipeline-spec",
            str(pipeline_spec),
            "--strict-lock",
        ],
        cwd=repo_root,
    )

    print("[2/4] Loading submission lock...")
    lock = yaml.safe_load(submission_lock.read_text(encoding="utf-8")) or {}
    artifacts = lock.get("artifacts", [])
    if not isinstance(artifacts, list) or not artifacts:
        raise RuntimeError("submission lock has empty or invalid 'artifacts' list")

    print("[3/4] Refreshing artifact hashes...")
    for entry in artifacts:
        if not isinstance(entry, dict):
            raise RuntimeError(f"Invalid artifact entry: {entry}")
        artifact_id = entry.get("id")
        raw_path = entry.get("path")
        if not isinstance(artifact_id, str) or not artifact_id:
            raise RuntimeError(f"Artifact missing id: {entry}")
        if not isinstance(raw_path, str) or not raw_path:
            raise RuntimeError(f"Artifact '{artifact_id}' missing path")

        path = resolve_path(raw_path, repo_root)
        if not path.exists():
            raise RuntimeError(f"Artifact missing: {artifact_id} -> {path}")

        if "sha256" in entry:
            entry["sha256"] = sha256_file(path)

    lock["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    submission_lock.write_text(
        yaml.safe_dump(lock, sort_keys=False, default_flow_style=False),
        encoding="utf-8",
    )

    if not args.skip_verify:
        print("[4/4] Verifying submission package...")
        run_cmd(
            [
                sys.executable,
                "scripts/verify_submission_package.py",
                "--lock",
                str(submission_lock),
            ],
            cwd=repo_root,
        )

    print("Done: submission package frozen and lock refreshed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
