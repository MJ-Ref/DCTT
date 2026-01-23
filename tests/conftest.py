"""Pytest configuration and shared fixtures."""

import numpy as np
import pytest
from pathlib import Path


# Seed for reproducibility
RANDOM_SEED = 42


@pytest.fixture(scope="session")
def random_state() -> np.random.Generator:
    """Seeded random generator for reproducible tests."""
    return np.random.default_rng(RANDOM_SEED)


@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """Path to test data directory."""
    return Path(__file__).parent / "data"


@pytest.fixture(scope="session")
def output_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Temporary output directory for test artifacts."""
    return tmp_path_factory.mktemp("outputs")


@pytest.fixture
def sample_embeddings(random_state: np.random.Generator) -> np.ndarray:
    """Generate sample normalized embeddings for testing.

    Returns embeddings matrix of shape (1000, 128) with L2-normalized rows.
    """
    n_tokens = 1000
    embedding_dim = 128
    embeddings = random_state.standard_normal((n_tokens, embedding_dim))
    # L2 normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / norms


@pytest.fixture
def sample_embeddings_small(random_state: np.random.Generator) -> np.ndarray:
    """Smaller embedding matrix for fast unit tests."""
    n_tokens = 100
    embedding_dim = 32
    embeddings = random_state.standard_normal((n_tokens, embedding_dim))
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / norms


@pytest.fixture
def degenerate_embeddings(random_state: np.random.Generator) -> np.ndarray:
    """Embeddings with intentional degeneracies for testing detection.

    Creates a mix of:
    - Normal embeddings
    - Near-duplicate clusters
    - Low-rank neighborhoods
    """
    n_tokens = 200
    embedding_dim = 64

    embeddings = random_state.standard_normal((n_tokens, embedding_dim))

    # Create a cluster of near-duplicates (tokens 50-60)
    base_vector = embeddings[50].copy()
    for i in range(51, 60):
        embeddings[i] = base_vector + random_state.standard_normal(embedding_dim) * 0.01

    # Create low-rank neighborhood (tokens 100-120 lie in 2D subspace)
    u = random_state.standard_normal(embedding_dim)
    v = random_state.standard_normal(embedding_dim)
    u = u / np.linalg.norm(u)
    v = v - np.dot(v, u) * u  # Orthogonalize
    v = v / np.linalg.norm(v)
    for i in range(100, 120):
        alpha, beta = random_state.standard_normal(2)
        embeddings[i] = alpha * u + beta * v

    # Normalize all
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / norms


@pytest.fixture
def mock_token_frequencies(random_state: np.random.Generator) -> np.ndarray:
    """Generate Zipf-distributed token frequencies."""
    n_tokens = 1000
    ranks = np.arange(1, n_tokens + 1)
    # Zipf's law: frequency ~ 1/rank
    frequencies = 1.0 / ranks
    # Add some noise
    frequencies *= random_state.uniform(0.8, 1.2, n_tokens)
    return frequencies


@pytest.fixture
def sample_neighbors(random_state: np.random.Generator) -> np.ndarray:
    """Sample neighbor indices for testing.

    Returns array of shape (100, 50) with neighbor indices.
    """
    n_queries = 100
    k = 50
    n_tokens = 1000
    # Generate random neighbors (in practice these come from kNN index)
    neighbors = np.zeros((n_queries, k), dtype=np.int64)
    for i in range(n_queries):
        # Exclude self and sample k neighbors
        candidates = list(range(n_tokens))
        candidates.remove(i)
        neighbors[i] = random_state.choice(candidates, size=k, replace=False)
    return neighbors


# Markers
def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "gpu: marks tests requiring GPU/MPS")
    config.addinivalue_line("markers", "integration: marks integration tests")
    config.addinivalue_line("markers", "property: marks property-based tests")
