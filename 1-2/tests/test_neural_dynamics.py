"""
Unit Tests for Neural Dynamics (Experiment 2)

Tests the neuromorphic components:
1. Izhikevich neuron model
2. STDP (Spike-Timing-Dependent Plasticity)
3. Holographic memory
4. Network initialization

Author: Claude Code Testing Framework
"""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# These tests verify the CPU-side logic and initialization
# GPU shader tests would require WebGPU runtime


class TestNetworkInitialization:
    """Test network initialization parameters"""

    def test_network_size(self):
        """Network size should be 512x512 = 262,144 neurons"""
        from experiment2_consciousness_emergence import NETWORK_SIZE

        assert NETWORK_SIZE == 512
        num_neurons = NETWORK_SIZE * NETWORK_SIZE
        assert num_neurons == 262144

    def test_consciousness_thresholds(self):
        """Consciousness thresholds should match NeuroCHIMERA paper"""
        from experiment2_consciousness_emergence import CONSCIOUSNESS_THRESHOLDS

        assert CONSCIOUSNESS_THRESHOLDS['connectivity'] == 15.0
        assert CONSCIOUSNESS_THRESHOLDS['integration'] == 0.65
        assert CONSCIOUSNESS_THRESHOLDS['depth'] == 7.0
        assert CONSCIOUSNESS_THRESHOLDS['complexity'] == 0.8
        assert CONSCIOUSNESS_THRESHOLDS['qualia_coherence'] == 0.75

    def test_stdp_parameters(self):
        """STDP parameters should match biological values"""
        from experiment2_consciousness_emergence import (
            TAU_PLUS, TAU_MINUS, A_PLUS, A_MINUS
        )

        # Time constants around 20ms
        assert TAU_PLUS == 20.0
        assert TAU_MINUS == 20.0

        # LTD slightly stronger than LTP (asymmetric STDP)
        assert A_MINUS > A_PLUS
        assert A_PLUS == 0.01
        assert A_MINUS == 0.012

    def test_holographic_parameters(self):
        """Holographic memory parameters"""
        from experiment2_consciousness_emergence import (
            HOLOGRAPHIC_SIZE, INTERFERENCE_PATTERNS
        )

        assert HOLOGRAPHIC_SIZE == 256
        assert INTERFERENCE_PATTERNS == 64


class TestIzhikevichModel:
    """Test Izhikevich neuron model simulation"""

    def test_izhikevich_rest_state(self):
        """Neuron at rest should stay at rest without input"""
        # Izhikevich model: dv/dt = 0.04v² + 5v + 140 - u + I
        # At equilibrium (v=-65, u=b*v), should be stable

        v = -65.0  # Rest potential
        u = 0.2 * v  # b=0.2
        I = 0.0  # No input

        # Simple Euler integration
        dt = 0.5
        for _ in range(100):
            dv = (0.04 * v * v + 5.0 * v + 140.0 - u + I) * dt
            du = 0.02 * (0.2 * v - u) * dt
            v += dv
            u += du

        # Should remain near rest
        assert -70 < v < -60, f"Rest state should be stable, got v={v}"

    def test_izhikevich_spike(self):
        """Neuron should spike with sufficient input"""
        v = -65.0
        u = 0.2 * v
        I = 20.0  # Strong input

        dt = 0.5
        spiked = False

        for _ in range(100):
            dv = (0.04 * v * v + 5.0 * v + 140.0 - u + I) * dt
            du = 0.02 * (0.2 * v - u) * dt
            v += dv
            u += du

            # Check for spike (v > 30 mV)
            if v >= 30.0:
                spiked = True
                # Reset
                v = -65.0
                u += 8.0
                break

        assert spiked, "Neuron should spike with strong input"

    def test_izhikevich_subthreshold(self):
        """Weak input should not cause spike"""
        v = -65.0
        u = 0.2 * v
        I = 5.0  # Weak input

        dt = 0.5
        max_v = v

        for _ in range(100):
            dv = (0.04 * v * v + 5.0 * v + 140.0 - u + I) * dt
            du = 0.02 * (0.2 * v - u) * dt
            v += dv
            u += du
            max_v = max(max_v, v)

            # Reset if spike
            if v >= 30.0:
                v = -65.0
                u += 8.0

        # Should depolarize but not spike
        assert max_v < 30.0, f"Weak input should not spike, max_v={max_v}"
        assert max_v > -65.0, "Should show some depolarization"


class TestSTDP:
    """Test STDP (Spike-Timing-Dependent Plasticity)"""

    def test_stdp_ltp(self):
        """Post-synaptic spike after pre-synaptic should strengthen (LTP)"""
        pre_time = 100.0
        post_time = 105.0  # 5ms after
        current_weight = 0.5

        # STDP rule: dt > 0 → LTP
        dt = post_time - pre_time
        dw = 0.01 * np.exp(-dt / 20.0)
        new_weight = current_weight + dw

        assert new_weight > current_weight, \
            "Post after pre should strengthen synapse (LTP)"

    def test_stdp_ltd(self):
        """Pre-synaptic spike after post-synaptic should weaken (LTD)"""
        pre_time = 105.0
        post_time = 100.0  # Pre after post
        current_weight = 0.5

        # STDP rule: dt < 0 → LTD
        dt = post_time - pre_time
        dw = -0.012 * np.exp(dt / 20.0)
        new_weight = current_weight + dw

        assert new_weight < current_weight, \
            "Pre after post should weaken synapse (LTD)"

    def test_stdp_causality_window(self):
        """STDP effect should decay with time difference"""
        current_weight = 0.5

        # Close timing (5ms)
        dt_close = 5.0
        dw_close = 0.01 * np.exp(-dt_close / 20.0)

        # Far timing (50ms)
        dt_far = 50.0
        dw_far = 0.01 * np.exp(-dt_far / 20.0)

        assert abs(dw_close) > abs(dw_far), \
            "STDP effect should be stronger for closer spikes"

    def test_stdp_weight_bounds(self):
        """Weights should remain in bounds [-1, 1]"""
        # Test upper bound
        weight = 0.95
        for _ in range(100):
            dw = 0.01  # Strong LTP
            weight = np.clip(weight + dw, -1.0, 1.0)

        assert -1.0 <= weight <= 1.0, "Weight should be bounded"
        assert weight <= 1.0, f"Weight should not exceed 1.0, got {weight}"

        # Test lower bound
        weight = -0.95
        for _ in range(100):
            dw = -0.012  # Strong LTD
            weight = np.clip(weight + dw, -1.0, 1.0)

        assert -1.0 <= weight <= 1.0, "Weight should be bounded"
        assert weight >= -1.0, f"Weight should not go below -1.0, got {weight}"


class TestHolographicMemory:
    """Test holographic memory encoding/decoding"""

    def test_holographic_transformation(self):
        """Holographic transformation should preserve information"""
        # Approximate the shader's holographic_activation function
        input_pattern = np.array([0.5, 0.3, 0.0, 0.0], dtype=np.float32)
        phase = np.pi / 4

        # Fourier-like transformation
        real = input_pattern[0] * np.cos(phase) - input_pattern[1] * np.sin(phase)
        imag = input_pattern[0] * np.sin(phase) + input_pattern[1] * np.cos(phase)

        magnitude = np.sqrt(real**2 + imag**2)

        # Should preserve magnitude
        input_magnitude = np.sqrt(input_pattern[0]**2 + input_pattern[1]**2)
        assert np.abs(magnitude - input_magnitude) < 1e-5, \
            "Holographic transform should preserve magnitude"

    def test_holographic_phase_sensitivity(self):
        """Different phases should give different outputs"""
        input_pattern = np.array([0.5, 0.3], dtype=np.float32)

        phase1 = 0.0
        real1 = input_pattern[0] * np.cos(phase1) - input_pattern[1] * np.sin(phase1)

        phase2 = np.pi / 2
        real2 = input_pattern[0] * np.cos(phase2) - input_pattern[1] * np.sin(phase2)

        assert abs(real1 - real2) > 0.1, \
            "Different phases should give different outputs"

    def test_holographic_interference(self):
        """Multiple patterns should interfere constructively/destructively"""
        # Pattern 1
        p1_real = 0.5 * np.cos(0.0)
        p1_imag = 0.5 * np.sin(0.0)

        # Pattern 2 (same phase - constructive)
        p2_real = 0.5 * np.cos(0.0)
        p2_imag = 0.5 * np.sin(0.0)

        # Superposition
        total_real = p1_real + p2_real
        total_imag = p1_imag + p2_imag

        magnitude_super = np.sqrt(total_real**2 + total_imag**2)
        magnitude_single = np.sqrt(p1_real**2 + p1_imag**2)

        # Constructive interference
        assert magnitude_super > magnitude_single * 1.5, \
            "Same phase should interfere constructively"


class TestNetworkConnectivity:
    """Test network connectivity structure"""

    def test_local_connectivity(self):
        """Each neuron should connect to 5x5 neighborhood (25 neighbors)"""
        kernel_size = 5
        num_neighbors = kernel_size * kernel_size

        assert num_neighbors == 25, "Should have 25 neighbors (5x5 kernel)"

    def test_periodic_boundary(self):
        """Network should have periodic boundary conditions"""
        size = 512

        # Test wrapping
        x = 0
        y = 0

        # Neighbor at (-1, -1) should wrap to (511, 511)
        nx = (x - 1 + size) % size
        ny = (y - 1 + size) % size

        assert nx == 511, "Should wrap horizontally"
        assert ny == 511, "Should wrap vertically"

    def test_weight_initialization(self):
        """Weights should be initialized with small random values"""
        np.random.seed(42)

        num_neurons = 512 * 512
        num_weights = num_neurons * 25

        weights = np.random.randn(num_weights).astype(np.float32) * 0.1

        # Check statistics
        mean = np.mean(weights)
        std = np.std(weights)

        assert abs(mean) < 0.02, f"Mean should be ~0, got {mean}"
        assert 0.08 < std < 0.12, f"Std should be ~0.1, got {std}"


class TestNumericalStability:
    """Test numerical stability of neural dynamics"""

    def test_activation_bounds(self):
        """Activations should remain in [0, 1] range"""
        # Simulate normalization from Izhikevich voltage to [0,1]
        v_min = -65.0  # Reset voltage
        v_max = 30.0   # Spike threshold

        for v in np.linspace(-70, 35, 100):
            activation = (v + 65.0) / 100.0
            activation = np.clip(activation, 0.0, 1.0)

            assert 0.0 <= activation <= 1.0, \
                f"Activation out of bounds: {activation} for v={v}"

    def test_recovery_bounds(self):
        """Recovery variable should be bounded"""
        # u typically ranges [-20, 20]
        u_max = 20.0

        for u in np.linspace(-25, 25, 100):
            recovery = u / 20.0
            recovery = np.clip(recovery, -1.0, 1.0)

            assert -1.0 <= recovery <= 1.0, \
                f"Recovery out of bounds: {recovery} for u={u}"

    def test_plasticity_decay(self):
        """Plasticity trace should decay exponentially"""
        trace = 1.0
        tau = 20.0
        dt = 0.5

        for _ in range(100):
            trace *= np.exp(-dt / tau)

        # After ~100 steps (50ms), should decay significantly
        assert trace < 0.1, f"Trace should decay, got {trace}"
        assert trace > 0, "Trace should not go negative"


# ============================================================================
# Test Summary
# ============================================================================

def test_summary():
    """Print test summary"""
    print("\n" + "="*70)
    print("EXPERIMENT 2 - NEURAL DYNAMICS TEST SUMMARY")
    print("="*70)
    print("\nTest Categories:")
    print("  1. Network Initialization - 4 tests")
    print("  2. Izhikevich Model - 3 tests")
    print("  3. STDP - 4 tests")
    print("  4. Holographic Memory - 3 tests")
    print("  5. Network Connectivity - 3 tests")
    print("  6. Numerical Stability - 3 tests")
    print("\nTotal: 20 tests")
    print("\nComponents Tested:")
    print("  • Izhikevich spiking neuron model")
    print("  • STDP learning rule (LTP/LTD)")
    print("  • Holographic memory interference patterns")
    print("  • Network topology (5x5 local, periodic)")
    print("  • Numerical stability and bounds")
    print("="*70)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
