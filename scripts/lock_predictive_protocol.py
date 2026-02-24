#!/usr/bin/env python3
"""Lock predictive-validity paper artifacts to a specific sweep result.

Usage:
    python scripts/lock_predictive_protocol.py \
      --sweep-dir outputs/sweeps/predictive_validity/2026-02-23_23-21-40 \
      --protocol-name predictive_acceptance_v1
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def _load_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sweep-dir",
        required=True,
        help="Directory containing sweep_results.json.",
    )
    parser.add_argument(
        "--protocol-name",
        default="predictive_acceptance_v1",
        help="Human-readable protocol name.",
    )
    parser.add_argument(
        "--output-lock-path",
        default="outputs/sweeps/predictive_validity/PROTOCOL_LOCK.json",
        help="Where to write protocol lock manifest.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    sweep_dir = (repo_root / args.sweep_dir).resolve()
    sweep_results = sweep_dir / "sweep_results.json"
    if not sweep_results.exists():
        raise FileNotFoundError(f"Missing sweep_results.json: {sweep_results}")

    payload = _load_json(sweep_results)
    aggregate = payload.get("aggregate", {})
    if not aggregate:
        raise RuntimeError(f"Sweep has no aggregate content: {sweep_results}")

    config = payload.get("config", {})
    total_runs = int(sum(int(row.get("n_runs", 0)) for row in aggregate.values()))
    n_models = int(len(aggregate))

    lock_payload = {
        "protocol_name": str(args.protocol_name),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "sweep_dir": str(sweep_dir),
        "sweep_results_path": str(sweep_results),
        "summary": {
            "n_models": n_models,
            "total_runs": total_runs,
            "models": sorted(aggregate.keys()),
        },
        "expected_config": config,
    }

    lock_path = (repo_root / args.output_lock_path).resolve()
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w") as f:
        json.dump(lock_payload, f, indent=2)

    print(f"Protocol lock written: {lock_path}")
    print(f"Locked sweep: {sweep_results}")
    print(f"Models: {', '.join(sorted(aggregate.keys()))}")
    print(f"Total runs: {total_runs}")


if __name__ == "__main__":
    main()
