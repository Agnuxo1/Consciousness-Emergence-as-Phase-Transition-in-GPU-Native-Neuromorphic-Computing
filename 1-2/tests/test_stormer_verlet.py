"""
Test Suite for Störmer-Verlet Symplectic Integrator

Tests the symplectic integration method used for Hamiltonian dynamics.
"""

import pytest
import numpy as np
from utils.cpu_reference import stormer_verlet_step, compute_total_energy


class TestStormerVerlet:
    """Test Störmer-Verlet integration properties."""

    def test_harmonic_oscillator(self):
        """
        Test on 1D harmonic oscillator: φ'' = -ω²φ

        Analytical solution: φ(t) = A*cos(ωt) + B*sin(ωt)
        """
        omega = 2.0
        dt = 0.01
        num_steps = int(2 * np.pi / omega / dt)  # One period

        # Initial conditions: φ=1, φ'=0
        phi = np.array([[1.0]], dtype=np.float32)
        pi = np.array([[0.0]], dtype=np.float32)

        positions = [phi[0, 0]]

        for _ in range(num_steps):
            # For harmonic oscillator: force = -ω²φ
            # But Störmer-Verlet uses full Hamiltonian
            phi, pi = stormer_verlet_step(phi, pi, lambda_cosmo=0.0, dt=dt)
            positions.append(phi[0, 0])

        # After one period, should return close to initial position
        final_phi = phi[0, 0]

        # Allow some numerical drift
        assert abs(final_phi - 1.0) < 0.1, \
            f"Harmonic oscillator drift too large: {final_phi} vs 1.0"

    def test_energy_bounded(self, grid_size):
        """
        Test that energy remains bounded (symplectic property).
        """
        np.random.seed(42)
        phi = np.random.randn(grid_size, grid_size).astype(np.float32) * 0.1
        pi = np.random.randn(grid_size, grid_size).astype(np.float32) * 0.05

        E0 = compute_total_energy(phi, pi)
        energies = [E0]

        dt = 0.01
        for _ in range(200):
            phi, pi = stormer_verlet_step(phi, pi, dt=dt)
            E = compute_total_energy(phi, pi)
            energies.append(E)

        energies = np.array(energies)

        # Energy should oscillate but not grow unboundedly
        max_energy = np.max(energies)
        min_energy = np.min(energies)

        # Bounded oscillation
        assert max_energy < 2 * abs(E0) + 1.0, \
            f"Energy grew unboundedly: {E0} → {max_energy}"
        assert min_energy > -2 * abs(E0) - 1.0, \
            f"Energy dropped unboundedly: {E0} → {min_energy}"

    def test_second_order_accuracy(self):
        """
        Test that integrator has O(dt²) global error.
        """
        # Simple test: integrate for fixed time with different dt
        omega = 1.0
        T = 1.0  # Total time

        phi_exact = np.cos(omega * T)  # Analytical solution

        errors = []
        dts = [0.1, 0.05, 0.025]

        for dt in dts:
            num_steps = int(T / dt)
            phi = np.array([[1.0]], dtype=np.float32)
            pi = np.array([[0.0]], dtype=np.float32)

            for _ in range(num_steps):
                phi, pi = stormer_verlet_step(phi, pi, lambda_cosmo=0.0, dt=dt)

            error = abs(phi[0, 0] - phi_exact)
            errors.append(error)

        # Error should decrease roughly as dt²
        # Check that halving dt reduces error by ~4x
        ratio1 = errors[0] / errors[1]
        ratio2 = errors[1] / errors[2]

        # Allow some tolerance (2-6x reduction)
        assert 2.0 < ratio1 < 6.0, \
            f"Not second-order accurate: ratio={ratio1}"

    def test_symplectic_volume_preservation(self, grid_size):
        """
        Test symplectic property: phase space volume preserved.

        This is tested indirectly through long-term energy stability.
        """
        np.random.seed(42)
        phi = np.random.randn(grid_size, grid_size).astype(np.float32) * 0.1
        pi = np.random.randn(grid_size, grid_size).astype(np.float32) * 0.05

        E0 = compute_total_energy(phi, pi)

        # Long integration
        dt = 0.01
        for _ in range(1000):
            phi, pi = stormer_verlet_step(phi, pi, dt=dt)

        E_final = compute_total_energy(phi, pi)

        # Energy drift should be bounded (characteristic of symplectic integrators)
        drift = abs(E_final - E0) / (abs(E0) + 1e-10)

        assert drift < 0.1, \
            f"Long-term energy drift {drift:.4f} too large (not symplectic)"
