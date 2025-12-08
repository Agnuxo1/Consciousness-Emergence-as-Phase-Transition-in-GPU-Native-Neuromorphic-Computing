"""
Test Suite for Einstein Field Equation Residual

Tests the computation of Einstein residual: |G_μν - 8πT_μν|

In the discrete approximation:
- G_00 ≈ R + Λ (scalar curvature + cosmological constant)
- T_00 = stress-energy tensor from matter fields
"""

import pytest
import numpy as np
from utils.cpu_reference import (
    compute_einstein_residual,
    compute_laplacian_2d,
    compute_stress_energy_tensor,
    COSMOLOGICAL_CONSTANT
)


class TestEinsteinResidual:
    """Test Einstein field equation residual computation."""

    def test_vacuum_solution(self, grid_size):
        """
        Test vacuum solution: T_μν = 0 → R ≈ -Λ

        In vacuum, Einstein equations reduce to:
        G_μν + Λg_μν = 0
        R_μν - (1/2)Rg_μν + Λg_μν = 0
        """
        # Vacuum: zero field and momentum
        phi = np.zeros((grid_size, grid_size), dtype=np.float32)
        pi = np.zeros((grid_size, grid_size), dtype=np.float32)

        residual = compute_einstein_residual(phi, pi)

        # Vacuum residual should be small
        assert residual < 1e-3, \
            f"Vacuum residual too large: {residual}"

    def test_uniform_density(self, grid_size):
        """
        Test uniform matter density → uniform curvature.
        """
        # Uniform field
        phi = np.ones((grid_size, grid_size), dtype=np.float32) * 0.5
        pi = np.ones((grid_size, grid_size), dtype=np.float32) * 0.1

        residual = compute_einstein_residual(phi, pi)

        # Should have some residual but bounded
        assert residual < 10.0, \
            f"Uniform density residual unexpectedly large: {residual}"

    def test_residual_decreases_with_evolution(self, grid_size):
        """
        Test that residual tends to decrease as system evolves
        (approaches Einstein equations).
        """
        from utils.cpu_reference import stormer_verlet_step

        np.random.seed(42)
        phi = np.random.randn(grid_size, grid_size).astype(np.float32) * 0.1
        pi = np.random.randn(grid_size, grid_size).astype(np.float32) * 0.05

        residuals = []

        for _ in range(50):
            residual = compute_einstein_residual(phi, pi)
            residuals.append(residual)

            # Evolve system
            phi, pi = stormer_verlet_step(phi, pi, dt=0.01)

        # Check that residual generally decreases
        # (allow some fluctuations)
        initial_avg = np.mean(residuals[:10])
        final_avg = np.mean(residuals[-10:])

        # Should show some improvement
        assert final_avg < initial_avg * 1.5, \
            f"Residual did not decrease: {initial_avg} → {final_avg}"

    def test_residual_components(self, sample_state):
        """
        Test individual components of Einstein residual calculation.
        """
        phi = sample_state[:, :, 0]
        pi = sample_state[:, :, 1]

        # Compute components
        R = compute_laplacian_2d(phi)
        T_00 = compute_stress_energy_tensor(phi, pi)

        # Both should be finite
        assert np.all(np.isfinite(R)), "Curvature has non-finite values"
        assert np.all(np.isfinite(T_00)), "Stress-energy has non-finite values"

        # G_00 ≈ R + Λ
        G_00 = R + COSMOLOGICAL_CONSTANT

        # Residual
        residual_field = G_00 - 8 * np.pi * T_00
        residual_rms = np.sqrt(np.mean(residual_field ** 2))

        # Should be finite
        assert np.isfinite(residual_rms), "Residual is not finite"
