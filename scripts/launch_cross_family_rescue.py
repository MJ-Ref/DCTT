#!/usr/bin/env python3
"""Launch cross-family predictive rescue sweeps on Modal from a YAML config."""

from __future__ import annotations

import argparse
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


APP_ID_RE = re.compile(r"\b(ap-[A-Za-z0-9]+)\b")


def _run(cmd: list[str], *, cwd: Path, check: bool = True) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True)
    if check and proc.returncode != 0:
        raise RuntimeError(
            "Command failed:\n"
            f"$ {shlex.join(cmd)}\n"
            f"exit={proc.returncode}\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )
    return proc


def _csv_ints(values: list[int]) -> str:
    return ",".join(str(int(v)) for v in values)


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Expected mapping at top-level in {path}")
    return payload


def _validate_config(config: dict[str, Any]) -> None:
    if "common" not in config or not isinstance(config["common"], dict):
        raise ValueError("Config missing required `common` mapping.")
    if "jobs" not in config or not isinstance(config["jobs"], list) or not config["jobs"]:
        raise ValueError("Config missing required non-empty `jobs` list.")

    common = config["common"]
    seeds = common.get("seeds")
    if not isinstance(seeds, list) or not seeds:
        raise ValueError("Config `common.seeds` must be a non-empty list.")

    required_common = [
        "sample_size",
        "n_prompts",
        "max_new_tokens",
        "n_bootstrap",
        "scoring_mode",
        "min_logprob_margin",
        "compute_device",
    ]
    for key in required_common:
        if key not in common:
            raise ValueError(f"Config missing `common.{key}`.")

    for idx, job in enumerate(config["jobs"]):
        if not isinstance(job, dict):
            raise ValueError(f"Config job #{idx} must be a mapping.")
        if "model" not in job:
            raise ValueError(f"Config job #{idx} missing `model`.")
        if "frequency_counts_path" not in job:
            raise ValueError(f"Config job #{idx} missing `frequency_counts_path`.")


def _build_modal_cmd(
    *,
    model: str,
    frequency_counts_path: str,
    common: dict[str, Any],
    detach: bool,
) -> list[str]:
    cmd = ["modal", "run"]
    if detach:
        cmd.append("--detach")
    cmd.extend(
        [
            "experiments/modal_predictive_sweep.py",
            "--models",
            model,
            "--seeds",
            _csv_ints([int(v) for v in common["seeds"]]),
            "--sample-size",
            str(int(common["sample_size"])),
            "--n-prompts",
            str(int(common["n_prompts"])),
            "--max-new-tokens",
            str(int(common["max_new_tokens"])),
            "--n-bootstrap",
            str(int(common["n_bootstrap"])),
            "--scoring-mode",
            str(common["scoring_mode"]),
            "--min-logprob-margin",
            str(float(common["min_logprob_margin"])),
            "--compute-device",
            str(common["compute_device"]),
            "--frequency-counts-path",
            str(frequency_counts_path),
        ]
    )
    if bool(common.get("fail_on_proxy_confounds", False)):
        cmd.append("--fail-on-proxy-confounds")
    return cmd


def _parse_app_id(text: str) -> str | None:
    match = APP_ID_RE.search(text)
    return match.group(1) if match else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/experiment/cross_family_rescue.yaml",
        help="YAML config with common sweep params and per-model jobs.",
    )
    parser.add_argument(
        "--detach",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Launch detached Modal jobs (default true).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print launch commands without executing.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    config_path = (repo_root / args.config).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    config = _load_yaml(config_path)
    _validate_config(config)
    common = config["common"]
    jobs = config["jobs"]

    print(f"[config] {config_path}")
    print(f"[jobs] {len(jobs)}")

    launches: list[dict[str, str]] = []
    for job in jobs:
        model = str(job["model"])
        counts_path = str(job["frequency_counts_path"])
        cmd = _build_modal_cmd(
            model=model,
            frequency_counts_path=counts_path,
            common=common,
            detach=bool(args.detach),
        )
        print(f"\n[launch:{model}] $ {shlex.join(cmd)}")
        if args.dry_run:
            continue

        proc = _run(cmd, cwd=repo_root, check=True)
        combined = f"{proc.stdout}\n{proc.stderr}"
        app_id = _parse_app_id(combined)
        launches.append(
            {
                "model": model,
                "app_id": app_id or "unknown",
                "counts_path": counts_path,
            }
        )
        print(combined.strip())

    if args.dry_run:
        return

    print("\n[summary]")
    for launch in launches:
        print(
            f"- model={launch['model']} app_id={launch['app_id']} "
            f"counts={launch['counts_path']}"
        )
    print("\nUse scripts/finalize_modal_predictive_sweep.py after each run stamp is committed.")


if __name__ == "__main__":
    main()
