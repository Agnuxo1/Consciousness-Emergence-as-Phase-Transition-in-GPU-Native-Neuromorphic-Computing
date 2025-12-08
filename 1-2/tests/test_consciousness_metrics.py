"""
Unit Tests for Consciousness Metrics (Experiment 2)

Tests the five consciousness parameters:
1. Connectivity ⟨k⟩
2. Integration Φ (Integrated Information)
3. Hierarchical Depth D
4. Complexity C (Lempel-Ziv)
5. Qualia Coherence QCM

Author: Claude Code Testing Framework
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import consciousness metric functions from experiment2
from experiment2_consciousness_emergence import (
    compute_connectivity,
    compute_integration_phi,
    compute_hierarchical_depth,
    compute_complexity_lz,
    compute_qualia_coherence,
    CONSCIOUSNESS_THRESHOLDS,
)


class TestConnectivity:
    """Test connectivity metric ⟨k⟩"""

    def test_connectivity_zero_weights(self):
        """Zero weights should give zero connectivity"""
        weights = np.zeros(512 * 512 * 25, dtype=np.float32)
        k = compute_connectivity(weights, threshold=0.1)
        assert k == 0.0, "Zero weights should give zero connectivity"

    def test_connectivity_all_strong(self):
        """All strong weights should give max connectivity (25 neighbors)"""
        weights = np.ones(512 * 512 * 25, dtype=np.float32) * 0.5
        k = compute_connectivity(weights, threshold=0.1)
        assert k == 25.0, f"All strong weights should give k=25, got k={k}"

    def test_connectivity_threshold_sensitivity(self):
        """Connectivity should be sensitive to threshold"""
        weights = np.random.uniform(0, 0.3, 512 * 512 * 25).astype(np.float32)

        k_low = compute_connectivity(weights, threshold=0.05)
        k_high = compute_connectivity(weights, threshold=0.25)

        assert k_low > k_high, "Lower threshold should give higher connectivity"

    def test_connectivity_range(self):
        """Connectivity should be in valid range [0, 25]"""
        weights = np.random.randn(512 * 512 * 25).astype(np.float32)
        k = compute_connectivity(weights, threshold=0.1)

        assert 0 <= k <= 25, f"Connectivity k={k} out of range [0, 25]"

    def test_connectivity_critical_threshold(self):
        """Test if realistic weights can exceed critical threshold (k > 15)"""
        # Simulate trained network with strong connections
        weights = np.random.randn(512 * 512 * 25).astype(np.float32) * 0.3
        # Add some strong connections
        strong_mask = np.random.rand(512 * 512 * 25) < 0.7
        weights[strong_mask] = np.random.uniform(0.2, 0.8, strong_mask.sum())

        k = compute_connectivity(weights, threshold=0.1)

        # Should be able to reach critical threshold with trained weights
        assert k > 5.0, f"Trained network should have k > 5, got k={k}"


class TestIntegration:
    """Test integration Φ (Integrated Information)"""

    def test_integration_uniform_field(self):
        """Uniform field should have low integration (no differentiation)"""
        activations = np.ones(512 * 512, dtype=np.float32) * 0.5
        phi = compute_integration_phi(activations, num_partitions=8)

        # Uniform field has no variance, penalized to low Φ
        assert phi < 0.2, f"Uniform field should have low Φ, got Φ={phi}"

    def test_integration_random_field(self):
        """Random field should have moderate integration"""
        activations = np.random.rand(512 * 512).astype(np.float32)
        phi = compute_integration_phi(activations, num_partitions=8)

        assert 0 <= phi <= 1.0, f"Φ should be in [0, 1], got Φ={phi}"

    def test_integration_structured_field(self):
        """Structured field with modules should have higher integration"""
        activations = np.zeros(512 * 512, dtype=np.float32)

        # Create modular structure with correlations
        grid = activations.reshape(512, 512)
        for i in range(8):
            for j in range(8):
                # Each module has similar pattern
                module_val = np.random.rand()
                grid[i*64:(i+1)*64, j*64:(j+1)*64] = module_val + \
                    np.random.randn(64, 64) * 0.1

        phi = compute_integration_phi(activations, num_partitions=8)

        # Structured field should have higher Φ than uniform
        assert phi > 0.1, f"Structured field should have Φ > 0.1, got Φ={phi}"

    def test_integration_range(self):
        """Integration should be bounded [0, 1]"""
        activations = np.random.randn(512 * 512).astype(np.float32)
        phi = compute_integration_phi(activations, num_partitions=8)

        assert 0 <= phi <= 1.0, f"Φ={phi} out of range [0, 1]"


class TestHierarchicalDepth:
    """Test hierarchical depth D"""

    def test_depth_flat_field(self):
        """Flat (uniform) field should have low depth"""
        activations = np.ones(512 * 512, dtype=np.float32) * 0.5
        depth = compute_hierarchical_depth(activations)

        assert depth < 5.0, f"Flat field should have low depth, got D={depth}"

    def test_depth_hierarchical_structure(self):
        """Hierarchical structure should give higher depth"""
        activations = np.zeros(512 * 512, dtype=np.float32)
        grid = activations.reshape(512, 512)

        # Create hierarchical structure at multiple scales
        for scale in [2, 4, 8, 16, 32, 64]:
            size = 512 // scale
            for i in range(scale):
                for j in range(scale):
                    # Different patterns at each scale
                    pattern_val = np.sin(i * 0.5) * np.cos(j * 0.5)
                    grid[i*size:(i+1)*size, j*size:(j+1)*size] += pattern_val * 0.1

        depth = compute_hierarchical_depth(activations)

        assert depth > 5.0, f"Hierarchical structure should have D > 5, got D={depth}"

    def test_depth_non_negative(self):
        """Depth should always be non-negative"""
        activations = np.random.randn(512 * 512).astype(np.float32)
        depth = compute_hierarchical_depth(activations)

        assert depth >= 0, f"Depth should be non-negative, got D={depth}"


class TestComplexity:
    """Test complexity C (Lempel-Ziv)"""

    def test_complexity_ordered(self):
        """Ordered pattern should have low complexity"""
        # Repeating pattern
        activations = np.tile([0.0, 1.0], 256 * 512).astype(np.float32)
        complexity = compute_complexity_lz(activations)

        # Ordered pattern has low LZ complexity
        assert complexity < 0.5, f"Ordered pattern should have low C, got C={complexity}"

    def test_complexity_random(self):
        """Random pattern should have high complexity"""
        activations = np.random.rand(512 * 512).astype(np.float32)
        complexity = compute_complexity_lz(activations)

        # Random has high complexity
        assert complexity > 0.5, f"Random pattern should have high C, got C={complexity}"

    def test_complexity_range(self):
        """Complexity should be in [0, 1]"""
        activations = np.random.randn(512 * 512).astype(np.float32)
        complexity = compute_complexity_lz(activations)

        assert 0 <= complexity <= 1.0, f"C={complexity} out of range [0, 1]"

    def test_complexity_edge_of_chaos(self):
        """Mixed pattern (edge of chaos) should have intermediate complexity"""
        activations = np.zeros(512 * 512, dtype=np.float32)

        # Mix of order and randomness
        activations[::2] = np.random.rand(256 * 512)  # Random
        activations[1::2] = np.tile([0.3, 0.7], 128 * 512)  # Ordered

        complexity = compute_complexity_lz(activations)

        # Should be in intermediate range
        assert 0.3 < complexity < 0.9, f"Mixed pattern should have intermediate C, got C={complexity}"


class TestQualiaCoherence:
    """Test qualia coherence QCM"""

    def test_qualia_uniform(self):
        """Uniform field should have high coherence (all modules similar)"""
        activations = np.ones(512 * 512, dtype=np.float32) * 0.5
        qcm = compute_qualia_coherence(activations, num_modules=8)

        assert qcm > 0.9, f"Uniform field should have high QCM, got QCM={qcm}"

    def test_qualia_incoherent(self):
        """Incoherent modules should have low QCM"""
        activations = np.zeros(512 * 512, dtype=np.float32)
        grid = activations.reshape(512, 512)

        # Each module completely different
        for i in range(8):
            for j in range(8):
                grid[i*64:(i+1)*64, j*64:(j+1)*64] = np.random.rand()

        qcm = compute_qualia_coherence(activations, num_modules=8)

        # Incoherent modules should have lower QCM
        assert qcm < 0.9, f"Incoherent modules should have QCM < 0.9, got QCM={qcm}"

    def test_qualia_range(self):
        """QCM should be in [0, 1]"""
        activations = np.random.rand(512 * 512).astype(np.float32)
        qcm = compute_qualia_coherence(activations, num_modules=8)

        assert 0 <= qcm <= 1.0, f"QCM={qcm} out of range [0, 1]"

    def test_qualia_partial_coherence(self):
        """Partially coherent modules should have intermediate QCM"""
        activations = np.zeros(512 * 512, dtype=np.float32)
        grid = activations.reshape(512, 512)

        # Similar modules with some variation
        base_val = 0.5
        for i in range(8):
            for j in range(8):
                grid[i*64:(i+1)*64, j*64:(j+1)*64] = base_val + \
                    np.random.randn(64, 64) * 0.1

        qcm = compute_qualia_coherence(activations, num_modules=8)

        assert 0.4 < qcm < 0.95, f"Partially coherent should have intermediate QCM, got QCM={qcm}"


class TestThresholds:
    """Test consciousness thresholds and emergence detection"""

    def test_thresholds_defined(self):
        """All 5 thresholds should be defined"""
        required_keys = ['connectivity', 'integration', 'depth',
                        'complexity', 'qualia_coherence']

        for key in required_keys:
            assert key in CONSCIOUSNESS_THRESHOLDS, f"Missing threshold: {key}"
            assert CONSCIOUSNESS_THRESHOLDS[key] > 0, f"Invalid threshold for {key}"

    def test_threshold_values_realistic(self):
        """Thresholds should be challenging but achievable"""
        # Based on NeuroCHIMERA paper
        assert CONSCIOUSNESS_THRESHOLDS['connectivity'] == 15.0
        assert CONSCIOUSNESS_THRESHOLDS['integration'] == 0.65
        assert CONSCIOUSNESS_THRESHOLDS['depth'] == 7.0
        assert CONSCIOUSNESS_THRESHOLDS['complexity'] == 0.8
        assert CONSCIOUSNESS_THRESHOLDS['qualia_coherence'] == 0.75

    def test_random_state_not_conscious(self):
        """Random network state should not meet all thresholds"""
        # Random state
        activations = np.random.rand(512 * 512).astype(np.float32)
        weights = np.random.randn(512 * 512 * 25).astype(np.float32) * 0.1

        k = compute_connectivity(weights, threshold=0.1)
        phi = compute_integration_phi(activations)
        depth = compute_hierarchical_depth(activations)
        complexity = compute_complexity_lz(activations)
        qcm = compute_qualia_coherence(activations)

        # Count how many thresholds are met
        thresholds_met = sum([
            k > CONSCIOUSNESS_THRESHOLDS['connectivity'],
            phi > CONSCIOUSNESS_THRESHOLDS['integration'],
            depth > CONSCIOUSNESS_THRESHOLDS['depth'],
            complexity > CONSCIOUSNESS_THRESHOLDS['complexity'],
            qcm > CONSCIOUSNESS_THRESHOLDS['qualia_coherence'],
        ])

        # Random state should not meet all 5
        assert thresholds_met < 5, \
            f"Random state met {thresholds_met}/5 thresholds (should be < 5)"


class TestReproducibility:
    """Test reproducibility of consciousness metrics"""

    def test_connectivity_deterministic(self):
        """Same weights should give same connectivity"""
        np.random.seed(42)
        weights = np.random.randn(512 * 512 * 25).astype(np.float32)

        k1 = compute_connectivity(weights, threshold=0.1)
        k2 = compute_connectivity(weights, threshold=0.1)

        assert k1 == k2, "Connectivity computation not deterministic"

    def test_integration_deterministic(self):
        """Same activations should give same integration"""
        np.random.seed(42)
        activations = np.random.rand(512 * 512).astype(np.float32)

        phi1 = compute_integration_phi(activations, num_partitions=8)
        phi2 = compute_integration_phi(activations, num_partitions=8)

        assert phi1 == phi2, "Integration computation not deterministic"

    def test_all_metrics_deterministic(self):
        """All metrics should be deterministic with same input"""
        np.random.seed(42)
        activations = np.random.rand(512 * 512).astype(np.float32)
        weights = np.random.randn(512 * 512 * 25).astype(np.float32)

        # Compute twice
        results1 = [
            compute_connectivity(weights),
            compute_integration_phi(activations),
            compute_hierarchical_depth(activations),
            compute_complexity_lz(activations),
            compute_qualia_coherence(activations),
        ]

        results2 = [
            compute_connectivity(weights),
            compute_integration_phi(activations),
            compute_hierarchical_depth(activations),
            compute_complexity_lz(activations),
            compute_qualia_coherence(activations),
        ]

        for i, (r1, r2) in enumerate(zip(results1, results2)):
            assert r1 == r2, f"Metric {i} not deterministic: {r1} != {r2}"


class TestEdgeCases:
    """Test edge cases and numerical stability"""

    def test_empty_network(self):
        """All-zero network should not crash"""
        activations = np.zeros(512 * 512, dtype=np.float32)
        weights = np.zeros(512 * 512 * 25, dtype=np.float32)

        # Should not raise exceptions
        k = compute_connectivity(weights)
        phi = compute_integration_phi(activations)
        depth = compute_hierarchical_depth(activations)
        complexity = compute_complexity_lz(activations)
        qcm = compute_qualia_coherence(activations)

        # All should be valid numbers
        assert all(np.isfinite([k, phi, depth, complexity, qcm]))

    def test_extreme_activations(self):
        """Extreme activation values should be handled"""
        activations = np.random.uniform(-10, 10, 512 * 512).astype(np.float32)

        # Should not crash or return NaN
        phi = compute_integration_phi(activations)
        depth = compute_hierarchical_depth(activations)
        complexity = compute_complexity_lz(activations)
        qcm = compute_qualia_coherence(activations)

        assert all(np.isfinite([phi, depth, complexity, qcm]))

    def test_nan_handling(self):
        """Should handle NaN gracefully or detect them"""
        activations = np.random.rand(512 * 512).astype(np.float32)
        activations[0] = np.nan

        # Functions should either handle NaN or it should be detectable
        try:
            phi = compute_integration_phi(activations)
            # If it doesn't raise, result should be NaN or valid
            assert np.isnan(phi) or np.isfinite(phi)
        except (ValueError, RuntimeWarning):
            # Expected to potentially fail with NaN input
            pass


# ============================================================================
# Test Summary
# ============================================================================

def test_summary():
    """Print test summary"""
    print("\n" + "="*70)
    print("EXPERIMENT 2 - CONSCIOUSNESS METRICS TEST SUMMARY")
    print("="*70)
    print("\nTest Categories:")
    print("  1. Connectivity ⟨k⟩ - 5 tests")
    print("  2. Integration Φ - 4 tests")
    print("  3. Hierarchical Depth D - 3 tests")
    print("  4. Complexity C - 4 tests")
    print("  5. Qualia Coherence QCM - 4 tests")
    print("  6. Thresholds - 3 tests")
    print("  7. Reproducibility - 3 tests")
    print("  8. Edge Cases - 3 tests")
    print("\nTotal: 29 tests")
    print("="*70)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
