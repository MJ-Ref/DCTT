#!/usr/bin/env python3
"""Wait for a Modal predictive-validity sweep and finalize local artifacts.

This script waits for `sweep_results.json` under a Modal volume path, pulls the
finished sweep locally, evaluates the predictive gate, updates protocol lock,
and regenerates figures/tables.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def _run(cmd: list[str], *, cwd: Path, check: bool = True) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True)
    if check and proc.returncode != 0:
        raise RuntimeError(
            "Command failed:\n"
            f"$ {' '.join(cmd)}\n"
            f"exit={proc.returncode}\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )
    return proc


def _remote_sweep_ready(repo_root: Path, *, volume_name: str, remote_dir: str) -> bool:
    proc = _run(
        ["modal", "volume", "ls", volume_name, remote_dir],
        cwd=repo_root,
        check=False,
    )
    if proc.returncode != 0:
        return False
    return "sweep_results.json" in proc.stdout


def _wait_for_remote(
    repo_root: Path,
    *,
    volume_name: str,
    remote_dir: str,
    poll_seconds: int,
    timeout_seconds: int,
) -> None:
    start = time.monotonic()
    while True:
        if _remote_sweep_ready(
            repo_root,
            volume_name=volume_name,
            remote_dir=remote_dir,
        ):
            print(f"[ready] Found sweep_results.json in {remote_dir}")
            return

        elapsed = int(time.monotonic() - start)
        print(f"[wait] {elapsed}s elapsed; waiting for {remote_dir}/sweep_results.json")
        if timeout_seconds > 0 and elapsed >= timeout_seconds:
            raise TimeoutError(
                f"Timed out after {timeout_seconds}s waiting for {remote_dir}/sweep_results.json"
            )
        time.sleep(poll_seconds)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stamp", required=True, help="Sweep timestamp dir, e.g. 2026-02-24_06-43-58")
    parser.add_argument("--wait", action="store_true", help="Wait until remote sweep_results.json exists.")
    parser.add_argument("--poll-seconds", type=int, default=30, help="Polling interval while waiting.")
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=0,
        help="Wait timeout in seconds (0 = no timeout).",
    )
    parser.add_argument("--volume-name", default="dctt-predictive-artifacts")
    parser.add_argument("--remote-root", default="/predictive_validity")
    parser.add_argument(
        "--local-root",
        default="outputs/sweeps/predictive_validity",
        help="Local root where sweep dir will be downloaded.",
    )
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--min-runs-per-model", type=int, default=5)
    parser.add_argument(
        "--require-positive-delta-per-model",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--require-positive-pooled-delta",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--protocol-name", default="predictive_acceptance_v1")
    parser.add_argument(
        "--lock-output-path",
        default="outputs/sweeps/predictive_validity/PROTOCOL_LOCK.json",
    )
    parser.add_argument(
        "--skip-figures",
        action="store_true",
        help="Skip scripts/generate_paper_figures.py refresh step.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    local_root = (repo_root / args.local_root).resolve()
    remote_dir = f"{args.remote_root.rstrip('/')}/{args.stamp}"
    local_sweep_dir = local_root / args.stamp

    # Fast fail if modal CLI is unavailable.
    _run(["modal", "--help"], cwd=repo_root, check=True)

    if args.wait:
        _wait_for_remote(
            repo_root,
            volume_name=args.volume_name,
            remote_dir=remote_dir,
            poll_seconds=max(1, int(args.poll_seconds)),
            timeout_seconds=max(0, int(args.timeout_seconds)),
        )
    elif not _remote_sweep_ready(
        repo_root,
        volume_name=args.volume_name,
        remote_dir=remote_dir,
    ):
        raise FileNotFoundError(
            f"Remote sweep not ready: {remote_dir}/sweep_results.json\n"
            "Re-run with --wait to block until artifacts are committed."
        )

    local_root.mkdir(parents=True, exist_ok=True)
    _run(
        [
            "modal",
            "volume",
            "get",
            args.volume_name,
            remote_dir,
            str(local_root),
            "--force",
        ],
        cwd=repo_root,
        check=True,
    )
    if not (local_sweep_dir / "sweep_results.json").exists():
        raise FileNotFoundError(
            f"Downloaded sweep dir missing sweep_results.json: {local_sweep_dir}"
        )

    gate_json = local_sweep_dir / "gate_evaluation.json"
    gate_md = local_sweep_dir / "gate_evaluation.md"
    _run(
        [
            sys.executable,
            str(repo_root / "scripts" / "evaluate_predictive_gate.py"),
            "--sweep-results",
            str(local_sweep_dir / "sweep_results.json"),
            "--confidence-level",
            str(args.confidence_level),
            "--min-runs-per-model",
            str(args.min_runs_per_model),
            "--output-json",
            str(gate_json),
            "--output-markdown",
            str(gate_md),
            (
                "--require-positive-delta-per-model"
                if args.require_positive_delta_per_model
                else "--no-require-positive-delta-per-model"
            ),
            (
                "--require-positive-pooled-delta"
                if args.require_positive_pooled_delta
                else "--no-require-positive-pooled-delta"
            ),
        ],
        cwd=repo_root,
        check=True,
    )

    _run(
        [
            sys.executable,
            str(repo_root / "scripts" / "lock_predictive_protocol.py"),
            "--sweep-dir",
            str(local_sweep_dir),
            "--protocol-name",
            str(args.protocol_name),
            "--output-lock-path",
            str((repo_root / args.lock_output_path).resolve()),
        ],
        cwd=repo_root,
        check=True,
    )

    if not args.skip_figures:
        _run(
            [sys.executable, str(repo_root / "scripts" / "generate_paper_figures.py")],
            cwd=repo_root,
            check=True,
        )

    gate_payload = json.loads(gate_json.read_text())
    print(f"[done] Sweep: {local_sweep_dir}")
    print(f"[done] Gate verdict: {gate_payload.get('verdict', 'UNKNOWN')}")
    print(f"[done] Gate report: {gate_md}")
    print(
        f"[done] Protocol lock: {(repo_root / args.lock_output_path).resolve()}"
    )
    if not args.skip_figures:
        print(f"[done] Figures refreshed under {(repo_root / 'outputs' / 'figures').resolve()}")


if __name__ == "__main__":
    main()
