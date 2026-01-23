"""Unit tests for Stage 2 spectral geometry metrics."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from dctt.metrics.stage2 import (
    compute_displacement_matrix,
    compute_local_covariance,
    compute_eigenvalues,
    compute_effective_dimension,
    compute_participation_ratio,
    compute_condition_number,
    compute_log_determinant,
    compute_anisotropy,
    compute_stage2_metrics,
)


class TestDisplacementMatrix:
    """Tests for displacement matrix computation."""

    def test_shape(self, sample_embeddings_small: np.ndarray) -> None:
        """Displacement matrix has correct shape."""
        k = 20
        neighbors = np.arange(1, k + 1)  # Token 0's neighbors: 1 to k
        displacement = compute_displacement_matrix(
            sample_embeddings_small, token_id=0, neighbor_ids=neighbors
        )
        assert displacement.shape == (k, sample_embeddings_small.shape[1])

    def test_values(self) -> None:
        """Displacement is correctly computed as neighbor - token."""
        embeddings = np.array([
            [1.0, 0.0, 0.0],  # token 0
            [0.0, 1.0, 0.0],  # token 1
            [0.0, 0.0, 1.0],  # token 2
        ])
        neighbors = np.array([1, 2])
        displacement = compute_displacement_matrix(embeddings, token_id=0, neighbor_ids=neighbors)
        expected = np.array([
            [-1.0, 1.0, 0.0],   # token 1 - token 0
            [-1.0, 0.0, 1.0],   # token 2 - token 0
        ])
        assert_allclose(displacement, expected)


class TestLocalCovariance:
    """Tests for local covariance computation."""

    def test_shape(self) -> None:
        """Covariance matrix has correct shape."""
        k, d = 50, 32
        displacement = np.random.randn(k, d)
        covariance = compute_local_covariance(displacement)
        assert covariance.shape == (d, d)

    def test_symmetric(self) -> None:
        """Covariance matrix is symmetric."""
        k, d = 50, 32
        displacement = np.random.randn(k, d)
        covariance = compute_local_covariance(displacement)
        assert_allclose(covariance, covariance.T)

    def test_positive_semidefinite(self) -> None:
        """Covariance matrix is positive semi-definite."""
        k, d = 50, 32
        displacement = np.random.randn(k, d)
        covariance = compute_local_covariance(displacement)
        eigenvalues = np.linalg.eigvalsh(covariance)
        # All eigenvalues should be >= 0 (with numerical tolerance)
        assert np.all(eigenvalues >= -1e-10)


class TestEigenvalues:
    """Tests for eigenvalue computation."""

    def test_descending_order(self) -> None:
        """Eigenvalues are returned in descending order."""
        covariance = np.diag([1.0, 5.0, 3.0, 2.0, 4.0])
        eigenvalues = compute_eigenvalues(covariance)
        assert np.all(eigenvalues[:-1] >= eigenvalues[1:])

    def test_non_negative(self) -> None:
        """All eigenvalues are non-negative."""
        rng = np.random.default_rng(42)
        data = rng.standard_normal((100, 20))
        covariance = data.T @ data / 99
        eigenvalues = compute_eigenvalues(covariance)
        assert np.all(eigenvalues >= 0)

    def test_known_values(self) -> None:
        """Eigenvalues match known diagonal matrix."""
        values = np.array([5.0, 3.0, 1.0, 0.5])
        covariance = np.diag(values)
        eigenvalues = compute_eigenvalues(covariance)
        assert_allclose(eigenvalues, np.sort(values)[::-1], rtol=1e-10)


class TestEffectiveDimension:
    """Tests for effective dimension computation."""

    def test_uniform_eigenvalues(self) -> None:
        """Uniform eigenvalues require all dimensions for 95% variance."""
        d = 100
        eigenvalues = np.ones(d)
        dim95 = compute_effective_dimension(eigenvalues, variance_threshold=0.95)
        assert dim95 == 95  # Need 95 of 100 equal eigenvalues for 95% variance

    def test_single_dominant(self) -> None:
        """Single dominant eigenvalue gives dim95 = 1."""
        eigenvalues = np.array([100.0, 0.1, 0.1, 0.1])
        dim95 = compute_effective_dimension(eigenvalues, variance_threshold=0.95)
        assert dim95 == 1

    def test_minimum_one(self) -> None:
        """Effective dimension is at least 1."""
        eigenvalues = np.array([1.0])
        dim95 = compute_effective_dimension(eigenvalues, variance_threshold=0.95)
        assert dim95 >= 1


class TestParticipationRatio:
    """Tests for participation ratio computation."""

    def test_uniform_eigenvalues_max_pr(self) -> None:
        """Uniform eigenvalues give PR = d."""
        d = 50
        eigenvalues = np.ones(d)
        pr = compute_participation_ratio(eigenvalues)
        assert_allclose(pr, d, rtol=1e-10)

    def test_single_dominant_min_pr(self) -> None:
        """Single dominant eigenvalue gives PR ~ 1."""
        eigenvalues = np.array([1000.0] + [0.001] * 99)
        pr = compute_participation_ratio(eigenvalues)
        # PR should be close to 1
        assert pr < 1.1

    def test_bounded(self, sample_embeddings_small: np.ndarray) -> None:
        """PR is bounded between 1 and d."""
        d = sample_embeddings_small.shape[1]
        displacement = sample_embeddings_small[:50] - sample_embeddings_small[0]
        covariance = compute_local_covariance(displacement)
        eigenvalues = compute_eigenvalues(covariance)
        pr = compute_participation_ratio(eigenvalues)
        assert 1.0 <= pr <= d + 0.01  # Small tolerance for numerical issues

    def test_scale_invariant(self) -> None:
        """PR is scale-invariant."""
        eigenvalues = np.array([4.0, 2.0, 1.0, 0.5])
        pr1 = compute_participation_ratio(eigenvalues)
        pr2 = compute_participation_ratio(eigenvalues * 100)
        assert_allclose(pr1, pr2, rtol=1e-6)


class TestConditionNumber:
    """Tests for condition number computation."""

    def test_identity_cond_one(self) -> None:
        """Identity-like eigenvalues give condition number ~ 1."""
        eigenvalues = np.ones(50)
        cond = compute_condition_number(eigenvalues, m_min=10)
        assert_allclose(cond, 1.0, rtol=1e-6)

    def test_high_cond_for_ill_conditioned(self) -> None:
        """Ill-conditioned eigenvalues give high condition number."""
        eigenvalues = np.array([1e6] + [1e-3] * 49)
        cond = compute_condition_number(eigenvalues, m_min=10)
        assert cond > 1e6  # Should be very high

    def test_at_least_one(self) -> None:
        """Condition number is at least 1."""
        eigenvalues = np.array([1.0, 1.0, 1.0])
        cond = compute_condition_number(eigenvalues, m_min=1)
        assert cond >= 1.0


class TestLogDeterminant:
    """Tests for log-determinant computation."""

    def test_identity_matrix(self) -> None:
        """Log-det of identity is 0."""
        eigenvalues = np.ones(10)
        logdet = compute_log_determinant(eigenvalues)
        assert_allclose(logdet, 0.0, atol=1e-6)

    def test_scaled_identity(self) -> None:
        """Log-det of scaled identity is d * log(scale)."""
        d = 10
        scale = 2.0
        eigenvalues = np.ones(d) * scale
        logdet = compute_log_determinant(eigenvalues)
        expected = d * np.log(scale)
        assert_allclose(logdet, expected, rtol=1e-6)

    def test_degenerate_low_logdet(self) -> None:
        """Degenerate (near-zero) eigenvalues give very low logdet."""
        eigenvalues = np.array([1.0] + [1e-10] * 49)
        logdet = compute_log_determinant(eigenvalues)
        # Should be strongly negative due to near-zero eigenvalues
        assert logdet < -400


class TestAnisotropy:
    """Tests for anisotropy computation."""

    def test_uniform_eigenvalues_anisotropy_one(self) -> None:
        """Uniform eigenvalues give anisotropy = 1."""
        eigenvalues = np.ones(50)
        aniso = compute_anisotropy(eigenvalues)
        assert_allclose(aniso, 1.0, rtol=1e-6)

    def test_high_anisotropy(self) -> None:
        """Dominant eigenvalue gives high anisotropy."""
        eigenvalues = np.array([100.0] + [1.0] * 99)
        aniso = compute_anisotropy(eigenvalues)
        # anisotropy = 100 / (mean of 100, 1, 1, ..., 1) = 100 / ((100 + 99) / 100) ~ 50
        assert aniso > 10


class TestComputeStage2Metrics:
    """Integration tests for full Stage 2 metric computation."""

    def test_returns_stage2_result(
        self, sample_embeddings_small: np.ndarray
    ) -> None:
        """Returns Stage2Result with all metrics."""
        from dctt.core.types import Stage2Result

        k = 20
        neighbors = np.arange(1, k + 1)
        result = compute_stage2_metrics(
            embeddings=sample_embeddings_small,
            token_id=0,
            neighbor_ids=neighbors,
        )
        assert isinstance(result, Stage2Result)
        assert result.token_id == 0
        assert result.dim95 >= 1
        assert result.pr >= 1.0
        assert result.cond >= 1.0
        assert isinstance(result.logdet, float)
        assert result.anisotropy >= 1.0

    def test_eigenvalues_optional(
        self, sample_embeddings_small: np.ndarray
    ) -> None:
        """Eigenvalues included only when requested."""
        k = 20
        neighbors = np.arange(1, k + 1)

        result_no_eig = compute_stage2_metrics(
            sample_embeddings_small, 0, neighbors, return_eigenvalues=False
        )
        assert result_no_eig.eigenvalues is None

        result_with_eig = compute_stage2_metrics(
            sample_embeddings_small, 0, neighbors, return_eigenvalues=True
        )
        assert result_with_eig.eigenvalues is not None
        assert len(result_with_eig.eigenvalues) == sample_embeddings_small.shape[1]

    def test_degenerate_neighborhood_detected(
        self, degenerate_embeddings: np.ndarray
    ) -> None:
        """Degenerate neighborhoods show characteristic metric patterns."""
        # Token 55 is in the near-duplicate cluster (50-60)
        k = 15
        # Use other tokens in the cluster as neighbors
        neighbors = np.array([50, 51, 52, 53, 54, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65])

        result = compute_stage2_metrics(
            degenerate_embeddings, token_id=55, neighbor_ids=neighbors
        )

        # Near-duplicates should have:
        # - Low effective dimension (neighbors are nearly identical)
        # - Low participation ratio
        # - Potentially high condition number

        # Compare to a "normal" token (token 0)
        normal_neighbors = np.arange(1, k + 1)
        normal_result = compute_stage2_metrics(
            degenerate_embeddings, token_id=0, neighbor_ids=normal_neighbors
        )

        # PR should be lower for degenerate token
        assert result.pr < normal_result.pr

    def test_to_dict_serialization(
        self, sample_embeddings_small: np.ndarray
    ) -> None:
        """Result can be serialized to dictionary."""
        k = 20
        neighbors = np.arange(1, k + 1)
        result = compute_stage2_metrics(
            sample_embeddings_small, 0, neighbors, return_eigenvalues=True
        )

        d = result.to_dict(include_eigenvalues=False)
        assert "token_id" in d
        assert "dim95" in d
        assert "pr" in d
        assert "cond" in d
        assert "eigenvalues" not in d

        d_with_eig = result.to_dict(include_eigenvalues=True)
        assert "eigenvalues" in d_with_eig
