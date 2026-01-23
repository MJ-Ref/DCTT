"""Reproducibility infrastructure for DCTT.

This module provides utilities for ensuring reproducible experiments:
- Random seed management
- Configuration hashing
- Run manifests
"""

from __future__ import annotations

import hashlib
import json
import platform
import random
import subprocess
import sys
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from omegaconf import DictConfig


@dataclass
class RunManifest:
    """Complete reproducibility manifest for a run.

    Captures all information needed to reproduce an experiment.
    """

    # Identity
    run_id: str
    timestamp: str

    # Code version
    git_commit: str
    git_branch: str
    git_dirty: bool
    code_hash: str

    # Model identity
    model_name: str
    model_revision: str
    tokenizer_hash: str
    embedding_matrix_hash: str

    # Index identity
    index_type: str
    index_config_hash: str
    index_seed: int

    # Configuration
    config: dict

    # Environment
    python_version: str
    package_versions: dict = field(default_factory=dict)
    platform_info: str = ""
    device: str = ""

    # Random seeds
    numpy_seed: int = 42
    torch_seed: int = 42
    random_seed: int = 42

    def save(self, path: Path) -> None:
        """Save manifest as JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2, default=str)

    @classmethod
    def load(cls, path: Path) -> "RunManifest":
        """Load manifest from JSON."""
        with open(path) as f:
            data = json.load(f)
        return cls(**data)


def generate_run_id() -> str:
    """Generate a unique run identifier."""
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    random_suffix = hashlib.sha256(
        str(random.random()).encode()
    ).hexdigest()[:8]
    return f"{timestamp}_{random_suffix}"


def get_git_commit() -> str:
    """Get current git commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def get_git_branch() -> str:
    """Get current git branch."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def is_git_dirty() -> bool:
    """Check if git working directory has uncommitted changes."""
    try:
        result = subprocess.run(
            ["git", "diff", "--quiet"],
            capture_output=True,
        )
        return result.returncode != 0
    except FileNotFoundError:
        return False


def hash_directory(path: str) -> str:
    """Compute hash of all Python files in directory."""
    hasher = hashlib.sha256()
    for file in sorted(Path(path).rglob("*.py")):
        hasher.update(file.read_bytes())
    return hasher.hexdigest()[:16]


def hash_embeddings(embeddings: np.ndarray) -> str:
    """Compute stable hash of embedding matrix."""
    return hashlib.sha256(embeddings.tobytes()).hexdigest()[:16]


def get_package_versions() -> dict[str, str]:
    """Get versions of key packages."""
    packages = [
        "numpy", "scipy", "torch", "transformers",
        "usearch", "faiss", "wandb", "hydra-core",
    ]
    versions = {}
    for pkg in packages:
        try:
            import importlib
            mod = importlib.import_module(pkg.replace("-", "_"))
            versions[pkg] = getattr(mod, "__version__", "unknown")
        except ImportError:
            versions[pkg] = "not installed"
    return versions


def set_all_seeds(seed: int) -> None:
    """Set all random seeds for reproducibility.

    Args:
        seed: Random seed to use.
    """
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # Enable deterministic algorithms where possible
        torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass


def create_run_manifest(
    config: "DictConfig",
    model_name: str,
    model_revision: str,
    embeddings: np.ndarray,
    tokenizer_hash: str,
    index_type: str,
    index_config_hash: str,
    seed: int = 42,
) -> RunManifest:
    """Create a run manifest from current state.

    Args:
        config: Hydra configuration.
        model_name: Model identifier.
        model_revision: Model revision.
        embeddings: Embedding matrix.
        tokenizer_hash: Hash of tokenizer.
        index_type: Type of kNN index.
        index_config_hash: Hash of index configuration.
        seed: Random seed.

    Returns:
        Populated RunManifest.
    """
    from omegaconf import OmegaConf

    return RunManifest(
        run_id=generate_run_id(),
        timestamp=datetime.utcnow().isoformat(),
        git_commit=get_git_commit(),
        git_branch=get_git_branch(),
        git_dirty=is_git_dirty(),
        code_hash=hash_directory("src/dctt") if Path("src/dctt").exists() else "unknown",
        model_name=model_name,
        model_revision=model_revision,
        tokenizer_hash=tokenizer_hash,
        embedding_matrix_hash=hash_embeddings(embeddings),
        index_type=index_type,
        index_config_hash=index_config_hash,
        index_seed=seed,
        config=OmegaConf.to_container(config, resolve=True),
        python_version=sys.version,
        package_versions=get_package_versions(),
        platform_info=platform.platform(),
        device=str(config.get("compute", {}).get("device", "cpu")),
        numpy_seed=seed,
        torch_seed=seed,
        random_seed=seed,
    )
