#!/usr/bin/env python3
"""Modal entrypoint for cloud GPU predictive-validity sweeps."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import modal

APP_NAME = "dctt-predictive-sweep"
REPO_REMOTE_DIR = "/root/DCTT"
VOLUME_MOUNT = "/vol"
SWEEP_ROOT = f"{VOLUME_MOUNT}/predictive_validity"
HF_CACHE_ROOT = f"{VOLUME_MOUNT}/hf_cache"

repo_root = Path(__file__).resolve().parent.parent

app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git")
    .pip_install(
        "torch==2.3.1",
        index_url="https://download.pytorch.org/whl/cu121",
        extra_index_url="https://pypi.org/simple",
    )
    .add_local_file(
        local_path=str(repo_root / "pyproject.toml"),
        remote_path=f"{REPO_REMOTE_DIR}/pyproject.toml",
        copy=True,
    )
    .add_local_file(
        local_path=str(repo_root / "README.md"),
        remote_path=f"{REPO_REMOTE_DIR}/README.md",
        copy=True,
    )
    .add_local_dir(
        local_path=str(repo_root / "src"),
        remote_path=f"{REPO_REMOTE_DIR}/src",
        copy=True,
    )
    .add_local_dir(
        local_path=str(repo_root / "configs"),
        remote_path=f"{REPO_REMOTE_DIR}/configs",
        copy=True,
    )
    .add_local_dir(
        local_path=str(repo_root / "experiments"),
        remote_path=f"{REPO_REMOTE_DIR}/experiments",
        copy=True,
    )
    .add_local_dir(
        local_path=str(repo_root / "scripts"),
        remote_path=f"{REPO_REMOTE_DIR}/scripts",
        copy=True,
    )
    .run_commands(
        "python -m pip install --upgrade pip",
        f"cd {REPO_REMOTE_DIR} && pip install -e .[modal]",
    )
)

artifact_volume = modal.Volume.from_name("dctt-predictive-artifacts", create_if_missing=True)
hf_secret = modal.Secret.from_name(
    "dctt-hf",
    required_keys=["HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"],
)


def _resolve_local_path(candidate: str) -> Path:
    """Resolve a user-provided local path relative to repository root."""
    path = Path(candidate).expanduser()
    if path.is_absolute():
        return path
    return (repo_root / path).resolve()


def _stage_frequency_counts_to_volume(local_candidate: str) -> str:
    """Upload local frequency counts file to shared volume and return remote path."""
    local_path = _resolve_local_path(local_candidate)
    if not local_path.exists() or not local_path.is_file():
        raise FileNotFoundError(
            f"Frequency counts file not found: {local_path}. "
            "Provide a valid local .npy/.json path."
        )

    remote_rel = f"confounds/{local_path.name}"
    with artifact_volume.batch_upload(force=True) as batch:
        batch.put_file(str(local_path), remote_rel)

    return f"{VOLUME_MOUNT}/{remote_rel}"


@app.function(
    image=image,
    gpu="A100",
    timeout=4 * 60 * 60,
    cpu=8,
    memory=32768,
    secrets=[hf_secret],
    volumes={VOLUME_MOUNT: artifact_volume},
)
def run_modal_sweep(  # noqa: PLR0913
    models: str = "qwen2_5_coder_7b,qwen2_5_7b",
    seeds: str = "42,43",
    sample_size: int = 120,
    n_prompts: int = 3,
    max_new_tokens: int = 16,
    n_bootstrap: int = 30,
    scoring_mode: str = "logprob_choice",
    min_logprob_margin: float = 0.0,
    compute_device: str = "cuda",
    frequency_counts_path: str = "",
    fail_on_proxy_confounds: bool = False,
) -> dict[str, Any]:
    """Run predictive-validity sweep on Modal GPU and return aggregate metrics."""
    stamp = datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = f"{SWEEP_ROOT}/{stamp}"
    resolved_frequency_counts_path = frequency_counts_path.strip()
    if resolved_frequency_counts_path:
        path_obj = Path(resolved_frequency_counts_path)
        if not path_obj.is_absolute():
            repo_candidate = (Path(REPO_REMOTE_DIR) / path_obj).resolve()
            if repo_candidate.exists():
                resolved_frequency_counts_path = str(repo_candidate)
            else:
                # Fallback for staged artifacts uploaded by local entrypoint.
                staged_candidate = (Path(VOLUME_MOUNT) / "confounds" / path_obj.name).resolve()
                if staged_candidate.exists():
                    resolved_frequency_counts_path = str(staged_candidate)

    cmd = [
        sys.executable,
        f"{REPO_REMOTE_DIR}/experiments/run_predictive_validity_sweep.py",
        "--models",
        models,
        "--seeds",
        seeds,
        "--sample-size",
        str(sample_size),
        "--n-prompts",
        str(n_prompts),
        "--max-new-tokens",
        str(max_new_tokens),
        "--n-bootstrap",
        str(n_bootstrap),
        "--scoring-mode",
        scoring_mode,
        "--min-logprob-margin",
        str(min_logprob_margin),
        "--compute-device",
        compute_device,
        "--output-dir",
        output_dir,
    ]
    if resolved_frequency_counts_path:
        cmd.extend(["--frequency-counts-path", resolved_frequency_counts_path])
    if fail_on_proxy_confounds:
        cmd.append("--fail-on-proxy-confounds")

    env = {
        **os.environ,
        "HF_HOME": HF_CACHE_ROOT,
        "HUGGINGFACE_HUB_CACHE": f"{HF_CACHE_ROOT}/hub",
        "TOKENIZERS_PARALLELISM": "false",
    }

    proc = subprocess.run(
        cmd,
        cwd=REPO_REMOTE_DIR,
        env=env,
        text=True,
        capture_output=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "Modal predictive sweep failed.\n"
            f"cmd={' '.join(cmd)}\n"
            f"stdout_tail:\n{proc.stdout[-4000:]}\n"
            f"stderr_tail:\n{proc.stderr[-4000:]}"
        )

    result_path = Path(output_dir) / "sweep_results.json"
    if not result_path.exists():
        raise RuntimeError(f"Expected sweep result not found: {result_path}")

    payload = json.loads(result_path.read_text())
    artifact_volume.commit()

    return {
        "output_dir": output_dir,
        "aggregate": payload.get("aggregate", {}),
        "records": payload.get("records", []),
        "stdout_tail": proc.stdout[-2000:],
    }


@app.local_entrypoint()
def main(  # noqa: PLR0913
    models: str = "qwen2_5_coder_7b,qwen2_5_7b",
    seeds: str = "42,43",
    sample_size: int = 120,
    n_prompts: int = 3,
    max_new_tokens: int = 16,
    n_bootstrap: int = 30,
    scoring_mode: str = "logprob_choice",
    min_logprob_margin: float = 0.0,
    compute_device: str = "cuda",
    frequency_counts_path: str = "",
    fail_on_proxy_confounds: bool = False,
) -> None:
    """Run Modal sweep and print compact summary."""
    staged_frequency_counts_path = frequency_counts_path
    if frequency_counts_path and not frequency_counts_path.startswith(f"{VOLUME_MOUNT}/"):
        staged_frequency_counts_path = _stage_frequency_counts_to_volume(frequency_counts_path)
        print(
            f"[staged] frequency counts: {frequency_counts_path} -> "
            f"{staged_frequency_counts_path}"
        )

    result = run_modal_sweep.remote(
        models=models,
        seeds=seeds,
        sample_size=sample_size,
        n_prompts=n_prompts,
        max_new_tokens=max_new_tokens,
        n_bootstrap=n_bootstrap,
        scoring_mode=scoring_mode,
        min_logprob_margin=min_logprob_margin,
        compute_device=compute_device,
        frequency_counts_path=staged_frequency_counts_path,
        fail_on_proxy_confounds=fail_on_proxy_confounds,
    )
    print(json.dumps(result, indent=2))
