"""Property-based tests for metric numerical stability and correctness."""

import numpy as np
import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from dctt.metrics.stage2 import (
    compute_participation_ratio,
    compute_condition_number,
    compute_effective_dimension,
    compute_log_determinant,
    compute_anisotropy,
    compute_eigenvalues,
    compute_local_covariance,
)


# Custom strategies for generating test data
positive_floats = st.floats(min_value=1e-10, max_value=1e6, allow_nan=False, allow_infinity=False)


@st.composite
def eigenvalue_arrays(draw, min_size: int = 2, max_size: int = 200):
    """Generate arrays of positive eigenvalues.

    Note: We use min_value=1e-4 to avoid numerical issues with epsilon
    regularization (1e-10) in metric implementations. Very small eigenvalues
    cause relative errors in scale-invariant properties.
    """
    size = draw(st.integers(min_value=min_size, max_value=max_size))
    elements = st.floats(min_value=1e-4, max_value=1e6, allow_nan=False, allow_infinity=False)
    arr = draw(arrays(dtype=np.float64, shape=size, elements=elements))
    return np.sort(arr)[::-1]  # Descending order


@st.composite
def covariance_matrices(draw, min_dim: int = 5, max_dim: int = 50):
    """Generate valid positive semi-definite covariance matrices."""
    d = draw(st.integers(min_value=min_dim, max_value=max_dim))
    k = draw(st.integers(min_value=d + 5, max_value=d * 3))

    # Generate random data and compute its covariance
    elements = st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False)
    data = draw(arrays(dtype=np.float64, shape=(k, d), elements=elements))

    # Center and compute covariance
    data_centered = data - data.mean(axis=0)
    cov = (data_centered.T @ data_centered) / (k - 1)

    return cov


class TestParticipationRatioProperties:
    """Property tests for participation ratio."""

    @given(eigenvalues=eigenvalue_arrays())
    @settings(max_examples=200)
    def test_pr_bounded_by_dimension(self, eigenvalues: np.ndarray) -> None:
        """PR must be between 1 and d (dimension)."""
        pr = compute_participation_ratio(eigenvalues)
        assert 1.0 - 1e-6 <= pr <= len(eigenvalues) + 1e-6

    @given(eigenvalues=eigenvalue_arrays(), scale=st.floats(min_value=0.5, max_value=2.0))
    @settings(max_examples=200)
    def test_pr_scale_invariant(self, eigenvalues: np.ndarray, scale: float) -> None:
        """PR should be scale-invariant (homogeneous of degree 0).

        Note: The implementation uses PR = (Σλ)² / (Σλ² + eps), where eps
        provides numerical stability but breaks exact scale invariance when
        Σλ² is comparable to eps. We test that PR is approximately scale
        invariant for well-conditioned inputs.
        """
        # Skip cases where eigenvalue squared sum is close to epsilon
        # (scale invariance breaks when eps dominates the denominator)
        sum_sq = np.sum(eigenvalues ** 2)
        assume(sum_sq > 1e-6)  # Well above eps=1e-10
        assume(sum_sq * scale ** 2 > 1e-6)

        pr1 = compute_participation_ratio(eigenvalues)
        pr2 = compute_participation_ratio(eigenvalues * scale)
        # Allow 0.1% relative error for numerical stability
        assert abs(pr1 - pr2) < 1e-3 * max(pr1, 1)

    @given(d=st.integers(min_value=2, max_value=100))
    @settings(max_examples=50)
    def test_uniform_eigenvalues_max_pr(self, d: int) -> None:
        """Uniform eigenvalues should give PR = d."""
        eigenvalues = np.ones(d)
        pr = compute_participation_ratio(eigenvalues)
        assert abs(pr - d) < 1e-6

    @given(d=st.integers(min_value=10, max_value=100))
    @settings(max_examples=50)
    def test_single_dominant_min_pr(self, d: int) -> None:
        """Single dominant eigenvalue should give PR ~ 1."""
        eigenvalues = np.array([1e6] + [1e-6] * (d - 1))
        pr = compute_participation_ratio(eigenvalues)
        assert pr < 1.1


class TestConditionNumberProperties:
    """Property tests for condition number."""

    @given(eigenvalues=eigenvalue_arrays())
    @settings(max_examples=200)
    def test_cond_at_least_one(self, eigenvalues: np.ndarray) -> None:
        """Condition number is always >= 1."""
        cond = compute_condition_number(eigenvalues, m_min=1)
        assert cond >= 1.0 - 1e-6

    @given(eigenvalues=eigenvalue_arrays())
    @settings(max_examples=200)
    def test_cond_finite(self, eigenvalues: np.ndarray) -> None:
        """Condition number should always be finite."""
        cond = compute_condition_number(eigenvalues, m_min=1)
        assert np.isfinite(cond)


class TestEffectiveDimensionProperties:
    """Property tests for effective dimension."""

    @given(eigenvalues=eigenvalue_arrays(), threshold=st.floats(min_value=0.5, max_value=0.99))
    @settings(max_examples=200)
    def test_dim_bounded(self, eigenvalues: np.ndarray, threshold: float) -> None:
        """Effective dimension is between 1 and d."""
        dim = compute_effective_dimension(eigenvalues, variance_threshold=threshold)
        assert 1 <= dim <= len(eigenvalues)

    @given(eigenvalues=eigenvalue_arrays())
    @settings(max_examples=200)
    def test_dim_monotonic_in_threshold(self, eigenvalues: np.ndarray) -> None:
        """Higher variance threshold requires more dimensions."""
        dim_low = compute_effective_dimension(eigenvalues, variance_threshold=0.5)
        dim_high = compute_effective_dimension(eigenvalues, variance_threshold=0.99)
        assert dim_low <= dim_high


class TestLogDeterminantProperties:
    """Property tests for log-determinant."""

    @given(eigenvalues=eigenvalue_arrays())
    @settings(max_examples=200)
    def test_logdet_finite(self, eigenvalues: np.ndarray) -> None:
        """Log-determinant should always be finite."""
        logdet = compute_log_determinant(eigenvalues)
        assert np.isfinite(logdet)

    @given(eigenvalues=eigenvalue_arrays(), scale=st.floats(min_value=0.5, max_value=2.0))
    @settings(max_examples=200)
    def test_logdet_scaling(self, eigenvalues: np.ndarray, scale: float) -> None:
        """logdet(scale * C) = logdet(C) + d * log(scale).

        Note: Due to epsilon regularization (log(λ + ε)), this property holds
        approximately when eigenvalues are well above epsilon.
        """
        d = len(eigenvalues)
        logdet1 = compute_log_determinant(eigenvalues)
        logdet2 = compute_log_determinant(eigenvalues * scale)
        expected_diff = d * np.log(scale)
        # Use tolerance proportional to dimension and absolute value
        tolerance = max(0.01, 1e-3 * d)
        assert abs(logdet2 - logdet1 - expected_diff) < tolerance


class TestAnisotropyProperties:
    """Property tests for anisotropy."""

    @given(eigenvalues=eigenvalue_arrays())
    @settings(max_examples=200)
    def test_anisotropy_at_least_one(self, eigenvalues: np.ndarray) -> None:
        """Anisotropy is always >= 1 when eigenvalues are well above epsilon.

        Note: Due to epsilon regularization in the denominator (mean + eps),
        we use a small tolerance below 1.0.
        """
        eigenvalues = np.sort(eigenvalues)[::-1]  # Ensure descending
        aniso = compute_anisotropy(eigenvalues)
        # Allow small tolerance for numerical precision
        assert aniso >= 1.0 - 1e-4

    @given(eigenvalues=eigenvalue_arrays())
    @settings(max_examples=200)
    def test_anisotropy_finite(self, eigenvalues: np.ndarray) -> None:
        """Anisotropy should always be finite."""
        eigenvalues = np.sort(eigenvalues)[::-1]
        aniso = compute_anisotropy(eigenvalues)
        assert np.isfinite(aniso)

    @given(d=st.integers(min_value=2, max_value=100))
    @settings(max_examples=50)
    def test_uniform_anisotropy_one(self, d: int) -> None:
        """Uniform eigenvalues give anisotropy = 1."""
        eigenvalues = np.ones(d)
        aniso = compute_anisotropy(eigenvalues)
        assert abs(aniso - 1.0) < 1e-6


class TestEigenvalueProperties:
    """Property tests for eigenvalue computation."""

    @given(cov=covariance_matrices())
    @settings(max_examples=100)
    def test_eigenvalues_non_negative(self, cov: np.ndarray) -> None:
        """Eigenvalues of covariance are non-negative."""
        eigenvalues = compute_eigenvalues(cov)
        assert np.all(eigenvalues >= 0)

    @given(cov=covariance_matrices())
    @settings(max_examples=100)
    def test_eigenvalues_descending(self, cov: np.ndarray) -> None:
        """Eigenvalues are in descending order."""
        eigenvalues = compute_eigenvalues(cov)
        assert np.all(eigenvalues[:-1] >= eigenvalues[1:])

    @given(cov=covariance_matrices())
    @settings(max_examples=100)
    def test_eigenvalue_count(self, cov: np.ndarray) -> None:
        """Number of eigenvalues equals matrix dimension."""
        eigenvalues = compute_eigenvalues(cov)
        assert len(eigenvalues) == cov.shape[0]


class TestCovarianceProperties:
    """Property tests for covariance computation."""

    @given(
        k=st.integers(min_value=10, max_value=100),
        d=st.integers(min_value=5, max_value=50),
    )
    @settings(max_examples=50)
    def test_covariance_symmetric(self, k: int, d: int) -> None:
        """Covariance matrix is symmetric."""
        rng = np.random.default_rng(42)
        displacement = rng.standard_normal((k, d))
        cov = compute_local_covariance(displacement)
        assert np.allclose(cov, cov.T)

    @given(
        k=st.integers(min_value=10, max_value=100),
        d=st.integers(min_value=5, max_value=50),
    )
    @settings(max_examples=50)
    def test_covariance_shape(self, k: int, d: int) -> None:
        """Covariance has shape (k, k) by default for efficiency."""
        rng = np.random.default_rng(42)
        displacement = rng.standard_normal((k, d))
        # Default: k×k (same non-zero eigenvalues as d×d, but O(k³) vs O(d³))
        cov = compute_local_covariance(displacement, use_small_matrix=True)
        assert cov.shape == (k, k)
        # Can also get d×d if needed
        cov_full = compute_local_covariance(displacement, use_small_matrix=False)
        assert cov_full.shape == (d, d)
