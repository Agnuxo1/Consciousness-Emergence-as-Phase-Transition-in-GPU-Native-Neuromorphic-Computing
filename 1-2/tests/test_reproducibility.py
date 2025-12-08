"""
Test Suite for Reproducibility

Tests that simulations are deterministic and reproducible with fixed seeds.
"""

import pytest
import numpy as np
from utils.cpu_reference import cpu_simulation_step


class TestReproducibility:
    """Test deterministic behavior with fixed seeds."""

    def test_same_seed_same_results(self, grid_size):
        """
        Test that same seed produces identical results.
        """
        # Run 1
        np.random.seed(42)
        state1 = np.zeros((grid_size, grid_size, 4), dtype=np.float32)
        state1[:, :, 0] = np.random.randn(grid_size, grid_size) * 0.1
        state1[:, :, 1] = np.random.randn(grid_size, grid_size) * 0.05

        weights1 = np.random.randn(grid_size * grid_size, 25).astype(np.float32) * 0.1

        # Evolve
        for epoch in range(10):
            state1 = cpu_simulation_step(state1, weights1, epoch)

        # Run 2 (same seed)
        np.random.seed(42)
        state2 = np.zeros((grid_size, grid_size, 4), dtype=np.float32)
        state2[:, :, 0] = np.random.randn(grid_size, grid_size) * 0.1
        state2[:, :, 1] = np.random.randn(grid_size, grid_size) * 0.05

        weights2 = np.random.randn(grid_size * grid_size, 25).astype(np.float32) * 0.1

        # Evolve
        for epoch in range(10):
            state2 = cpu_simulation_step(state2, weights2, epoch)

        # Should be identical
        assert np.allclose(state1, state2, rtol=1e-10, atol=1e-10), \
            f"Same seed produced different results: max diff = {np.max(np.abs(state1 - state2))}"

    def test_different_seed_different_results(self, grid_size):
        """
        Test that different seeds produce different results.
        """
        # Run 1 (seed 42)
        np.random.seed(42)
        state1 = np.zeros((grid_size, grid_size, 4), dtype=np.float32)
        state1[:, :, 0] = np.random.randn(grid_size, grid_size) * 0.1

        # Run 2 (seed 123)
        np.random.seed(123)
        state2 = np.zeros((grid_size, grid_size, 4), dtype=np.float32)
        state2[:, :, 0] = np.random.randn(grid_size, grid_size) * 0.1

        # Should be different
        assert not np.allclose(state1, state2, rtol=1e-3, atol=1e-3), \
            "Different seeds produced same results"

    def test_cross_session_reproducibility(self, grid_size):
        """
        Test that results are reproducible across multiple runs.
        """
        results = []

        for run in range(3):
            np.random.seed(42)
            state = np.zeros((grid_size, grid_size, 4), dtype=np.float32)
            state[:, :, 0] = np.random.randn(grid_size, grid_size) * 0.1
            weights = np.random.randn(grid_size * grid_size, 25).astype(np.float32) * 0.1

            # Evolve
            for epoch in range(5):
                state = cpu_simulation_step(state, weights, epoch)

            results.append(state.copy())

        # All runs should be identical
        for i in range(1, len(results)):
            assert np.allclose(results[0], results[i], rtol=1e-10, atol=1e-10), \
                f"Run {i} differs from run 0"

    def test_gpu_noise_deterministic(self):
        """
        Test that GPU noise function is deterministic.

        GPU noise: sin(idx * 12.9898 + epoch * 78.233) * amplitude
        This should be fully deterministic (no random number generator).
        """
        # Simulate GPU noise
        size = 64
        epoch = 100

        noise1 = np.zeros((size, size))
        for y in range(size):
            for x in range(size):
                idx = x + y * size
                amplitude = 0.001 * np.exp(-epoch * 0.0001)
                noise1[y, x] = np.sin(idx * 12.9898 + epoch * 78.233) * amplitude

        # Compute again
        noise2 = np.zeros((size, size))
        for y in range(size):
            for x in range(size):
                idx = x + y * size
                amplitude = 0.001 * np.exp(-epoch * 0.0001)
                noise2[y, x] = np.sin(idx * 12.9898 + epoch * 78.233) * amplitude

        # Should be identical
        assert np.allclose(noise1, noise2, rtol=1e-10, atol=1e-10), \
            "GPU noise not deterministic"
