"""
Test Suite for Laplacian / Curvature Computation

Tests the discrete Laplacian operator which approximates the Ricci scalar
curvature in the computational network.
"""

import pytest
import numpy as np
from utils.cpu_reference import compute_laplacian_2d


class TestLaplacianBasics:
    """Test basic Laplacian properties."""

    def test_laplacian_of_constant_is_zero(self, constant_field):
        """
        Test: ∇²(constant) = 0

        The Laplacian of a constant field should be exactly zero.
        """
        laplacian = compute_laplacian_2d(constant_field)

        assert np.allclose(laplacian, 0.0, atol=1e-10), \
            f"Laplacian of constant field non-zero: max = {np.max(np.abs(laplacian))}"

    def test_laplacian_of_linear_is_zero(self, linear_field):
        """
        Test: ∇²(ax + by + c) = 0

        The Laplacian of a linear field should be zero (second derivatives vanish).
        """
        laplacian = compute_laplacian_2d(linear_field)

        # Allow small numerical error at boundaries
        interior = laplacian[2:-2, 2:-2]

        assert np.allclose(interior, 0.0, atol=1e-6), \
            f"Laplacian of linear field non-zero: max = {np.max(np.abs(interior))}"

    def test_laplacian_of_gaussian(self, gaussian_field):
        """
        Test: ∇²(exp(-r²/2σ²)) has known analytical form

        For a Gaussian: ∇²G = G * (r²/σ² - 2) / σ²
        At the peak (r=0): ∇²G = -2G/σ²  (negative)
        """
        laplacian = compute_laplacian_2d(gaussian_field)

        # At center, Laplacian should be negative
        center_y, center_x = gaussian_field.shape[0] // 2, gaussian_field.shape[1] // 2
        laplacian_center = laplacian[center_y, center_x]

        assert laplacian_center < 0, \
            f"Laplacian at Gaussian peak should be negative, got {laplacian_center}"


class TestLaplacianConvergence:
    """Test convergence of discrete Laplacian to continuous."""

    @pytest.mark.slow
    def test_convergence_with_grid_refinement(self):
        """
        Test: Discrete Laplacian converges to analytical as grid is refined.

        For a quadratic field φ = x² + y², analytical ∇²φ = 4.
        Test convergence for grids: 16, 32, 64, 128.
        """
        errors = []

        for size in [16, 32, 64, 128]:
            # Create quadratic field
            x = np.linspace(-1, 1, size)
            y = np.linspace(-1, 1, size)
            X, Y = np.meshgrid(x, y)
            field = (X**2 + Y**2).astype(np.float32)

            # Analytical Laplacian
            analytical = np.ones_like(field) * 4.0

            # Computed Laplacian
            computed = compute_laplacian_2d(field, periodic=False)

            # Error in interior (avoid boundaries)
            interior_slice = slice(2, -2)
            error = np.max(np.abs(computed[interior_slice, interior_slice] -
                                  analytical[interior_slice, interior_slice]))
            errors.append(error)

        # Error should decrease with grid refinement (O(h²) convergence)
        # Check that error at 128 is much smaller than at 16
        assert errors[-1] < 0.1 * errors[0], \
            f"Convergence not observed: errors = {errors}"


class TestPeriodicBoundaryConditions:
    """Test periodic boundary conditions."""

    def test_periodic_left_right(self, grid_size):
        """
        Test that left and right boundaries are connected periodically.
        """
        field = np.zeros((grid_size, grid_size), dtype=np.float32)

        # Create spike at left edge
        field[:, 0] = 1.0

        laplacian = compute_laplacian_2d(field, periodic=True)

        # At x=0, left neighbor wraps to x=grid_size-1
        # Laplacian at (y, 0) should include contribution from (y, grid_size-1)
        assert laplacian[grid_size//2, 0] != 0.0, \
            "Periodic boundary not working (left-right)"

    def test_periodic_top_bottom(self, grid_size):
        """
        Test that top and bottom boundaries are connected periodically.
        """
        field = np.zeros((grid_size, grid_size), dtype=np.float32)

        # Create spike at top edge
        field[0, :] = 1.0

        laplacian = compute_laplacian_2d(field, periodic=True)

        # At y=0, up neighbor wraps to y=grid_size-1
        assert laplacian[0, grid_size//2] != 0.0, \
            "Periodic boundary not working (top-bottom)"
