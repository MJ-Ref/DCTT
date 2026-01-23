"""Embedding caching utilities.

This module provides caching functionality for extracted embeddings,
avoiding redundant extraction from models.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from dctt.embeddings.extract import EmbeddingInfo

logger = logging.getLogger(__name__)


class EmbeddingCache:
    """Cache for extracted embeddings.

    Stores embeddings on disk with metadata for validation.
    Uses content-based hashing to detect stale caches.

    Example:
        >>> cache = EmbeddingCache("./outputs/embeddings")
        >>> key = cache.make_key("Qwen/Qwen2.5-7B", "main")
        >>> if cache.has(key):
        ...     embeddings, info = cache.load(key)
        ... else:
        ...     embeddings, tokenizer = extract_embeddings(...)
        ...     cache.save(key, embeddings, info)
    """

    def __init__(self, cache_dir: str | Path) -> None:
        """Initialize cache.

        Args:
            cache_dir: Directory to store cached embeddings.
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def make_key(
        self,
        model_name: str,
        revision: str = "main",
        embedding_source: str = "auto",
    ) -> str:
        """Create cache key from model parameters.

        Args:
            model_name: HuggingFace model identifier.
            revision: Model revision.
            embedding_source: Embedding extraction source.

        Returns:
            Cache key string.
        """
        # Create a unique key based on parameters
        key_data = f"{model_name}:{revision}:{embedding_source}"
        return hashlib.sha256(key_data.encode()).hexdigest()[:16]

    def _get_paths(self, key: str) -> tuple[Path, Path]:
        """Get paths for embeddings and metadata files."""
        emb_path = self.cache_dir / f"{key}.npy"
        meta_path = self.cache_dir / f"{key}.meta.json"
        return emb_path, meta_path

    def has(self, key: str) -> bool:
        """Check if cache contains embeddings for key.

        Args:
            key: Cache key.

        Returns:
            True if cached embeddings exist.
        """
        emb_path, meta_path = self._get_paths(key)
        return emb_path.exists() and meta_path.exists()

    def load(
        self, key: str
    ) -> tuple[NDArray[np.float64], dict] | None:
        """Load cached embeddings.

        Args:
            key: Cache key.

        Returns:
            Tuple of (embeddings, metadata) or None if not found.
        """
        emb_path, meta_path = self._get_paths(key)

        if not (emb_path.exists() and meta_path.exists()):
            return None

        try:
            # Load embeddings
            embeddings = np.load(emb_path)

            # Load metadata
            with open(meta_path) as f:
                metadata = json.load(f)

            # Verify hash
            actual_hash = hashlib.sha256(embeddings.tobytes()).hexdigest()[:16]
            if metadata.get("hash") != actual_hash:
                logger.warning(f"Cache hash mismatch for {key}, invalidating")
                return None

            logger.info(f"Loaded cached embeddings: {key}")
            return embeddings.astype(np.float64), metadata

        except Exception as e:
            logger.warning(f"Failed to load cache {key}: {e}")
            return None

    def save(
        self,
        key: str,
        embeddings: NDArray[np.float64],
        info: "EmbeddingInfo | dict",
    ) -> None:
        """Save embeddings to cache.

        Args:
            key: Cache key.
            embeddings: Embedding matrix to cache.
            info: Embedding metadata (EmbeddingInfo or dict).
        """
        emb_path, meta_path = self._get_paths(key)

        try:
            # Save embeddings
            np.save(emb_path, embeddings)

            # Prepare metadata
            if hasattr(info, "to_dict"):
                metadata = info.to_dict()
            else:
                metadata = dict(info)

            # Ensure hash is included
            if "hash" not in metadata:
                metadata["hash"] = hashlib.sha256(
                    embeddings.tobytes()
                ).hexdigest()[:16]

            # Save metadata
            with open(meta_path, "w") as f:
                json.dump(metadata, f, indent=2)

            logger.info(f"Saved embeddings to cache: {key}")

        except Exception as e:
            logger.error(f"Failed to save cache {key}: {e}")
            raise

    def invalidate(self, key: str) -> None:
        """Remove cached embeddings.

        Args:
            key: Cache key to invalidate.
        """
        emb_path, meta_path = self._get_paths(key)
        if emb_path.exists():
            emb_path.unlink()
        if meta_path.exists():
            meta_path.unlink()
        logger.info(f"Invalidated cache: {key}")

    def clear(self) -> None:
        """Clear all cached embeddings."""
        for path in self.cache_dir.glob("*.npy"):
            path.unlink()
        for path in self.cache_dir.glob("*.meta.json"):
            path.unlink()
        logger.info("Cleared embedding cache")

    def list_cached(self) -> list[dict]:
        """List all cached embeddings with metadata.

        Returns:
            List of metadata dictionaries for cached embeddings.
        """
        cached = []
        for meta_path in sorted(self.cache_dir.glob("*.meta.json")):
            try:
                with open(meta_path) as f:
                    metadata = json.load(f)
                metadata["cache_key"] = meta_path.stem
                cached.append(metadata)
            except Exception as e:
                logger.warning(f"Failed to read {meta_path}: {e}")
        return cached
