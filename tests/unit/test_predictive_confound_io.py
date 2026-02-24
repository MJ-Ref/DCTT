"""Unit tests for predictive-validity confound I/O helpers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.run_predictive_validity import (
    _load_cached_norms,
    _load_frequency_counts,
    _save_cached_norms,
)


def test_load_frequency_counts_npy(tmp_path: Path) -> None:
    path = tmp_path / "counts.npy"
    np.save(path, np.array([5, 4, 3, 2, 1], dtype=np.int64))

    loaded = _load_frequency_counts(path=path, vocab_size=5)
    assert loaded is not None
    assert loaded.shape == (5,)
    assert np.allclose(loaded, np.array([5, 4, 3, 2, 1], dtype=np.float64))


def test_load_frequency_counts_json_list(tmp_path: Path) -> None:
    path = tmp_path / "counts.json"
    path.write_text(json.dumps({"counts": [1, 2, 3]}))

    loaded = _load_frequency_counts(path=path, vocab_size=3)
    assert loaded is not None
    assert np.allclose(loaded, np.array([1.0, 2.0, 3.0]))


def test_load_frequency_counts_json_dict(tmp_path: Path) -> None:
    path = tmp_path / "counts.json"
    path.write_text(json.dumps({"counts": {"0": 10, "2": 4}}))

    loaded = _load_frequency_counts(path=path, vocab_size=4)
    assert loaded is not None
    assert np.allclose(loaded, np.array([10.0, 0.0, 4.0, 0.0]))


def test_load_frequency_counts_short_json_list_pads_zeros(tmp_path: Path) -> None:
    path = tmp_path / "counts.json"
    path.write_text(json.dumps({"counts": [1, 2]}))

    loaded = _load_frequency_counts(path=path, vocab_size=3)
    assert loaded is not None
    assert np.allclose(loaded, np.array([1.0, 2.0, 0.0]))


def test_load_frequency_counts_long_npy_truncates(tmp_path: Path) -> None:
    path = tmp_path / "counts.npy"
    np.save(path, np.array([9, 8, 7, 6], dtype=np.int64))

    loaded = _load_frequency_counts(path=path, vocab_size=2)
    assert loaded is not None
    assert np.allclose(loaded, np.array([9.0, 8.0]))


def test_save_and_load_cached_norms(tmp_path: Path) -> None:
    cache_key = "abc123"
    norms = np.array([1.2, 0.8, 2.1], dtype=np.float64)
    _save_cached_norms(tmp_path, cache_key, norms)

    loaded = _load_cached_norms(tmp_path, cache_key, expected_vocab_size=3)
    assert loaded is not None
    assert np.allclose(loaded, norms)
