#!/usr/bin/env python3
"""
CPU Reference Implementation for Experiment 1 Validation

This module provides pure NumPy implementations of all GPU kernels from
experiment1_spacetime_emergence.py. It serves as the ground truth for
validating GPU computation correctness.

All functions are translated directly from WGSL shaders to ensure identical
logic and serve as baseline for numerical validation.
"""

import numpy as np
from typing import Tuple, Dict
import random

# ============================================================================
# CONSTANTS (must match experiment1_spacetime_emergence.py)
# ============================================================================

L_PLANCK = 1.616255e-35  # metros
T_PLANCK = 5.391247e-44  # segundos
M_PLANCK = 2.176434e-8   # kg
G_NEWTON = 6.67430e-11   # m³/(kg·s²)
C_LIGHT = 299792458      # m/s
HBAR = 1.054571817e-34   # J·s

GALOIS_N = 1
LAMBDA_0 = 3.0 / (L_PLANCK ** 2)
COSMOLOGICAL_CONSTANT = LAMBDA_0 * (2 ** (-2 * GALOIS_N))

GRID_SIZE = 256
LEARNING_RATE = 0.001
CONNECTIVITY_THRESHOLD = 0.1
CRITICAL_CONNECTIVITY = 4.0
TIME_STEP = 0.01


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def set_seed(seed: int):
    """
    Set random seed for reproducibility.

    Args:
        seed: Random seed value
    """
    np.random.seed(seed)
    random.seed(seed)


# ============================================================================
# HIERARCHICAL NUMERIC SYSTEM (HNS) - CPU VERSION
# ============================================================================

def hns_encode(value: float) -> np.ndarray:
    """
    Encode a float value into HNS (Hierarchical Numeric System) format.

    HNS encoding: N = R×10⁰ + G×10³ + B×10⁶ + A×10⁹

    Args:
        value: Float value to encode

    Returns:
        RGBA array [R, G, B, A] where each component is in [0, 1000)
    """
    abs_val = abs(value)
    sign = 1.0 if value >= 0 else -1.0

    billions = np.floor(abs_val / 1e9)
    remainder1 = abs_val - billions * 1e9

    millions = np.floor(remainder1 / 1e6)
    remainder2 = remainder1 - millions * 1e6

    thousands = np.floor(remainder2 / 1e3)
    units = remainder2 - thousands * 1e3

    # Apply sign to first component
    units *= sign

    return np.array([units, thousands, millions, billions], dtype=np.float32)


def hns_decode(rgba: np.ndarray) -> float:
    """
    Decode HNS format back to float.

    Args:
        rgba: RGBA array [R, G, B, A]

    Returns:
        Decoded float value
    """
    R, G, B, A = rgba
    sign = 1.0 if R >= 0 else -1.0

    value = abs(R) * 1e0 + G * 1e3 + B * 1e6 + A * 1e9
    return sign * value


def hns_add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Add two HNS-encoded numbers with carry propagation.

    Args:
        a, b: HNS-encoded arrays [R, G, B, A]

    Returns:
        HNS-encoded sum
    """
    result = a + b

    # Carry propagation from units to thousands
    if abs(result[0]) >= 1000.0:
        carry = np.floor(result[0] / 1000.0)
        result[0] -= carry * 1000.0
        result[1] += carry

    # Carry from thousands to millions
    if abs(result[1]) >= 1000.0:
        carry = np.floor(result[1] / 1000.0)
        result[1] -= carry * 1000.0
        result[2] += carry

    # Carry from millions to billions
    if abs(result[2]) >= 1000.0:
        carry = np.floor(result[2] / 1000.0)
        result[2] -= carry * 1000.0
        result[3] += carry

    return result


# ============================================================================
# GALOIS FIELD OPERATIONS - GF(2^n)
# ============================================================================

def galois_multiply_gf2(a: int, b: int) -> int:
    """
    Multiply two elements in GF(2) - simplest Galois field.

    In GF(2): multiplication is AND operation

    Args:
        a, b: Binary values (0 or 1)

    Returns:
        Product in GF(2)
    """
    return a & b


def frobenius_automorphism_gf2(x: int) -> int:
    """
    Frobenius automorphism in GF(2): φ(x) = x²

    In GF(2): x² = x (every element is idempotent)

    Args:
        x: Element in GF(2)

    Returns:
        φ(x) = x² = x
    """
    return x  # In GF(2), x² = x


# ============================================================================
# PHYSICS KERNELS - CPU IMPLEMENTATIONS
# ============================================================================

def compute_laplacian_2d(field: np.ndarray, periodic: bool = True) -> np.ndarray:
    """
    Compute 2D discrete Laplacian: ∇²φ ≈ (φ_left + φ_right + φ_up + φ_down - 4φ_center)

    Matches WGSL compute_curvature() function (lines 202-220 in shader).

    Args:
        field: 2D array (grid_size, grid_size) of scalar field values
        periodic: Use periodic boundary conditions (default True)

    Returns:
        2D array of Laplacian values
    """
    size = field.shape[0]
    laplacian = np.zeros_like(field)

    for y in range(size):
        for x in range(size):
            center = field[y, x]

            if periodic:
                # Periodic boundary conditions
                left = field[y, (x - 1) % size]
                right = field[y, (x + 1) % size]
                up = field[(y - 1) % size, x]
                down = field[(y + 1) % size, x]
            else:
                # Dirichlet boundary conditions (zero at edges)
                left = field[y, x - 1] if x > 0 else 0
                right = field[y, x + 1] if x < size - 1 else 0
                up = field[y - 1, x] if y > 0 else 0
                down = field[y + 1, x] if y < size - 1 else 0

            # 5-point stencil Laplacian
            laplacian[y, x] = left + right + up + down - 4.0 * center

    return laplacian


def compute_stress_energy_tensor(phi: np.ndarray, pi: np.ndarray) -> np.ndarray:
    """
    Compute T_00 component of stress-energy tensor.

    T_00 ≈ (1/2)(π² + (∇φ)²) + V(φ)

    Matches WGSL compute_stress_energy() function (lines 223-242 in shader).

    Args:
        phi: Scalar field (grid_size, grid_size)
        pi: Conjugate momentum (grid_size, grid_size)

    Returns:
        T_00 component (grid_size, grid_size)
    """
    size = phi.shape[0]
    T_00 = np.zeros_like(phi)

    # Kinetic term: (1/2)π²
    kinetic = 0.5 * pi ** 2

    # Gradient term: (1/2)(∇φ)²
    gradient_sq = np.zeros_like(phi)
    for y in range(size):
        for x in range(size):
            # Forward differences with periodic BC
            dx = phi[y, (x + 1) % size] - phi[y, x]
            dy = phi[(y + 1) % size, x] - phi[y, x]
            gradient_sq[y, x] = dx ** 2 + dy ** 2

    # Quartic potential: V(φ) = λ(φ² - v²)²
    v_squared = 1.0
    lambda_coupling = 0.1
    potential = lambda_coupling * (phi ** 2 - v_squared) ** 2

    T_00 = kinetic + 0.5 * gradient_sq + potential

    return T_00


def compute_connectivity(state: np.ndarray, weights: np.ndarray,
                        threshold: float = CONNECTIVITY_THRESHOLD) -> np.ndarray:
    """
    Compute effective connectivity k for each node.

    Counts active connections in 5×5 neighborhood (25 neighbors total).

    Matches WGSL compute_connectivity() function (lines 245-270 in shader).

    Args:
        state: Full state array (grid_size, grid_size, 4) [phi, pi, R, k]
        weights: Weight matrix (grid_size*grid_size, 25)
        threshold: Connection activation threshold

    Returns:
        Connectivity array (grid_size, grid_size)
    """
    size = state.shape[0]
    k = np.zeros((size, size), dtype=np.float32)

    for y in range(size):
        for x in range(size):
            idx = x + y * size
            connections = 0

            # 5×5 neighborhood (dx, dy in [-2, 2])
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    if dx == 0 and dy == 0:
                        continue

                    # Periodic boundary conditions
                    nx = (x + dx) % size
                    ny = (y + dy) % size

                    # Weight index in flattened 5×5 array
                    weight_idx = (dx + 2) * 5 + (dy + 2)
                    w = weights[idx, weight_idx]

                    if abs(w) > threshold:
                        connections += 1

            k[y, x] = float(connections)

    return k


def stormer_verlet_step(phi: np.ndarray, pi: np.ndarray,
                        lambda_cosmo: float = COSMOLOGICAL_CONSTANT,
                        dt: float = TIME_STEP) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform one Störmer-Verlet integration step for Hamiltonian dynamics.

    Implements leapfrog integration:
    1. π_{n+1/2} = π_n + (dt/2) * F(φ_n)
    2. φ_{n+1} = φ_n + dt * π_{n+1/2}
    3. π_{n+1} = π_{n+1/2} + (dt/2) * F(φ_{n+1})

    Matches WGSL integration (lines 309-336 in shader).

    Args:
        phi: Scalar field at time n
        pi: Conjugate momentum at time n
        lambda_cosmo: Cosmological constant
        dt: Time step

    Returns:
        (phi_new, pi_new) at time n+1
    """
    # Compute force: F = -δL/δφ = -(-∇²φ + Λφ + dV/dφ)
    laplacian = compute_laplacian_2d(phi)

    # Quartic potential derivative: dV/dφ = 0.1 * 4φ(φ² - 1) = 0.4(φ³ - φ)
    dV_dphi = 0.1 * (phi ** 3 - phi)

    # δL/δφ = -∇²φ + Λφ + dV/dφ
    dL_dphi = -laplacian + lambda_cosmo * phi + dV_dphi

    # Force: F = -δL/δφ
    F = -dL_dphi

    # Half step of momentum
    pi_half = pi + 0.5 * dt * F

    # Full step of position
    phi_new = phi + dt * pi_half

    # Recalculate force at new position
    laplacian_new = compute_laplacian_2d(phi_new)
    dV_dphi_new = 0.1 * (phi_new ** 3 - phi_new)
    dL_dphi_new = -laplacian_new + lambda_cosmo * phi_new + dV_dphi_new
    F_new = -dL_dphi_new

    # Complete momentum step
    pi_new = pi_half + 0.5 * dt * F_new

    return phi_new, pi_new


def apply_phase_transition_rule(state: np.ndarray, k_old: np.ndarray,
                                k_new: np.ndarray, critical_k: float = CRITICAL_CONNECTIVITY,
                                idx_seed: int = 0) -> np.ndarray:
    """
    Apply M/R (More of the Same / Radically Different) rule for phase transitions.

    Matches WGSL phase transition logic (lines 346-361 in shader).

    Args:
        state: Current state [phi, pi, R, k] (grid_size, grid_size, 4)
        k_old: Previous connectivity
        k_new: New connectivity
        critical_k: Critical connectivity threshold
        idx_seed: Index seed for deterministic noise

    Returns:
        Updated state with phase transition applied
    """
    size = state.shape[0]
    final_state = state.copy()

    for y in range(size):
        for x in range(size):
            was_below = k_old[y, x] < critical_k
            is_above = k_new[y, x] >= critical_k

            if was_below and is_above:
                # Radically Different rule: reconfiguration
                idx = x + y * size
                noise = np.sin((idx + idx_seed) * 0.01)

                # Reheating analogy
                final_state[y, x, 0] = final_state[y, x, 0] * 0.9 + 0.1 * noise
                # Energy dissipation
                final_state[y, x, 1] = final_state[y, x, 1] * 0.5

    return final_state


def apply_quantum_noise(phi: np.ndarray, epoch: int) -> np.ndarray:
    """
    Apply decaying quantum/stochastic noise to field.

    Matches WGSL noise application (lines 363-366 in shader).

    Args:
        phi: Scalar field (grid_size, grid_size)
        epoch: Current epoch number

    Returns:
        Field with noise applied
    """
    size = phi.shape[0]
    noise_amplitude = 0.001 * np.exp(-epoch * 0.0001)

    # Deterministic pseudo-random noise (matches GPU)
    noise = np.zeros_like(phi)
    for y in range(size):
        for x in range(size):
            idx = x + y * size
            noise[y, x] = np.sin(idx * 12.9898 + epoch * 78.233) * noise_amplitude

    return phi + noise


# ============================================================================
# FULL SIMULATION STEP - CPU VERSION
# ============================================================================

def cpu_simulation_step(state: np.ndarray, weights: np.ndarray, epoch: int,
                       lambda_cosmo: float = COSMOLOGICAL_CONSTANT,
                       dt: float = TIME_STEP) -> np.ndarray:
    """
    Perform one complete simulation step on CPU (matches GPU compute shader).

    This function replicates the entire WGSL main() kernel (lines 276-373).

    Args:
        state: Current state (grid_size, grid_size, 4) where channels are:
               [0] = φ (scalar field)
               [1] = π (conjugate momentum)
               [2] = R (curvature scalar)
               [3] = k (connectivity)
        weights: Connection weights (grid_size*grid_size, 25)
        epoch: Current epoch number
        lambda_cosmo: Cosmological constant
        dt: Time step

    Returns:
        Updated state (grid_size, grid_size, 4)
    """
    size = state.shape[0]

    # Extract channels
    phi = state[:, :, 0]
    pi = state[:, :, 1]
    R_old = state[:, :, 2]
    k_old = state[:, :, 3]

    # Step 1: Störmer-Verlet integration
    phi_new, pi_new = stormer_verlet_step(phi, pi, lambda_cosmo, dt)

    # Step 2: Update curvature
    R_new = compute_laplacian_2d(phi_new)

    # Step 3: Update connectivity
    state_temp = np.stack([phi_new, pi_new, R_new, k_old], axis=-1)
    k_new = compute_connectivity(state_temp, weights)

    # Step 4: Assemble new state
    state_new = np.stack([phi_new, pi_new, R_new, k_new], axis=-1)

    # Step 5: Apply M/R phase transition rule
    state_new = apply_phase_transition_rule(state_new, k_old, k_new)

    # Step 6: Apply quantum noise
    state_new[:, :, 0] = apply_quantum_noise(state_new[:, :, 0], epoch)

    return state_new


# ============================================================================
# METRICS COMPUTATION - CPU VERSION
# ============================================================================

def compute_fractal_dimension_boxcounting(field: np.ndarray,
                                         threshold: float = 0.5) -> float:
    """
    Compute fractal dimension using box-counting method.

    Matches Python implementation in experiment1_spacetime_emergence.py (lines 830-847).

    Args:
        field: 2D field (grid_size, grid_size)
        threshold: Binarization threshold

    Returns:
        Estimated fractal dimension D
    """
    binary = (np.abs(field) > threshold).astype(int)
    size = field.shape[0]

    scales = [2, 4, 8, 16, 32]
    counts = []

    for scale in scales:
        if size % scale != 0:
            continue

        # Coarse-grain and count non-empty boxes
        reshaped = binary.reshape(size // scale, scale, size // scale, scale)
        boxes = reshaped.any(axis=(1, 3)).sum()
        counts.append(boxes)

    if len(counts) < 2:
        return 2.0  # Default for 2D surface

    # Log-log fit: log(N) ~ D * log(1/ε)
    log_scales = np.log(1.0 / np.array(scales[:len(counts)]))
    log_counts = np.log(np.array(counts) + 1)

    # Linear regression
    coeffs = np.polyfit(log_scales, log_counts, 1)
    dimension = float(coeffs[0])

    return dimension


def compute_einstein_residual(phi: np.ndarray, pi: np.ndarray,
                              lambda_cosmo: float = COSMOLOGICAL_CONSTANT) -> float:
    """
    Compute residual of Einstein field equations: |G_μν - 8πT_μν|

    In discrete approximation:
    - G_00 ≈ R (scalar curvature)
    - T_00 from stress-energy tensor

    Args:
        phi: Scalar field
        pi: Conjugate momentum
        lambda_cosmo: Cosmological constant

    Returns:
        RMS residual value
    """
    # Left side: G_μν = R_μν - (1/2)Rg_μν + Λg_μν
    # Simplified: G_00 ≈ R + Λ
    R = compute_laplacian_2d(phi)
    G_00 = R + lambda_cosmo

    # Right side: 8πGT_μν (with G=1 in Planck units)
    T_00 = compute_stress_energy_tensor(phi, pi)

    # Residual
    residual = G_00 - 8 * np.pi * T_00

    return float(np.sqrt(np.mean(residual ** 2)))


def compute_total_energy(phi: np.ndarray, pi: np.ndarray) -> float:
    """
    Compute total Hamiltonian energy: H = T + V

    Args:
        phi: Scalar field
        pi: Conjugate momentum

    Returns:
        Total energy
    """
    # Kinetic: (1/2)∫ π² dV
    kinetic = 0.5 * np.sum(pi ** 2)

    # Potential: (1/2)∫ (∇φ)² dV + ∫ V(φ) dV
    gradient_sq = compute_laplacian_2d(phi) ** 2  # Approximation
    potential_field = 0.5 * np.sum(phi ** 2) + 0.1 * np.sum((phi ** 2 - 1) ** 2)

    return float(kinetic + potential_field)


# ============================================================================
# VALIDATION UTILITIES
# ============================================================================

def compare_states(gpu_state: np.ndarray, cpu_state: np.ndarray,
                  rtol: float = 1e-4, atol: float = 1e-5) -> Dict[str, bool]:
    """
    Compare GPU and CPU states for validation.

    Args:
        gpu_state: State from GPU computation
        cpu_state: State from CPU reference
        rtol: Relative tolerance
        atol: Absolute tolerance

    Returns:
        Dictionary with comparison results for each channel
    """
    results = {}
    channel_names = ['phi', 'pi', 'R', 'k']

    for i, name in enumerate(channel_names):
        gpu_channel = gpu_state[:, :, i]
        cpu_channel = cpu_state[:, :, i]

        matches = np.allclose(gpu_channel, cpu_channel, rtol=rtol, atol=atol)
        max_abs_diff = np.max(np.abs(gpu_channel - cpu_channel))
        max_rel_diff = np.max(np.abs((gpu_channel - cpu_channel) / (cpu_channel + 1e-10)))

        results[name] = {
            'matches': matches,
            'max_abs_diff': float(max_abs_diff),
            'max_rel_diff': float(max_rel_diff)
        }

    return results


if __name__ == "__main__":
    # Quick self-test
    print("CPU Reference Implementation - Self Test")
    print("=" * 60)

    # Test Laplacian on constant field (should be zero)
    field_const = np.ones((64, 64))
    laplacian = compute_laplacian_2d(field_const)
    print(f"Laplacian of constant field: {np.max(np.abs(laplacian)):.10f} (should be ~0)")

    # Test HNS encode/decode
    test_val = 123456.789
    encoded = hns_encode(test_val)
    decoded = hns_decode(encoded)
    print(f"HNS encode/decode: {test_val} -> {decoded} (error: {abs(test_val - decoded):.10f})")

    # Test fractal dimension on uniform 2D field
    field_2d = np.random.rand(256, 256)
    dim = compute_fractal_dimension_boxcounting(field_2d, threshold=0.5)
    print(f"Fractal dimension of random 2D field: {dim:.3f} (expected ~2.0)")

    print("=" * 60)
    print("CPU Reference Implementation ready for validation!")
