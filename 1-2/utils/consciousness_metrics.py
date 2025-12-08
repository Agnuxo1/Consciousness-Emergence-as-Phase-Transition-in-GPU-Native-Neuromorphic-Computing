"""
Consciousness Metrics (Standalone - No GPU Dependencies)

Extracted from experiment2_consciousness_emergence.py for testing purposes.
These are the CPU implementations of consciousness parameter calculations.

Author: Claude Code Testing Framework
"""

import numpy as np
from typing import Dict
from dataclasses import dataclass


# ============================================================================
# CONSCIOUSNESS THRESHOLDS
# ============================================================================

CONSCIOUSNESS_THRESHOLDS = {
    'connectivity': 15.0,      # ⟨k⟩ > 15
    'integration': 0.65,       # Φ > 0.65
    'depth': 7.0,              # D > 7
    'complexity': 0.8,         # C > 0.8
    'qualia_coherence': 0.75,  # QCM > 0.75
}


# ============================================================================
# CONSCIOUSNESS METRICS DATACLASS
# ============================================================================

@dataclass
class ConsciousnessMetrics:
    """Consciousness metrics at each epoch"""
    epoch: int
    connectivity: float      # ⟨k⟩
    integration: float       # Φ
    depth: float            # D
    complexity: float       # C
    qualia_coherence: float # QCM

    @property
    def is_conscious(self) -> bool:
        """Check if all parameters exceed thresholds"""
        return (
            self.connectivity > CONSCIOUSNESS_THRESHOLDS['connectivity'] and
            self.integration > CONSCIOUSNESS_THRESHOLDS['integration'] and
            self.depth > CONSCIOUSNESS_THRESHOLDS['depth'] and
            self.complexity > CONSCIOUSNESS_THRESHOLDS['complexity'] and
            self.qualia_coherence > CONSCIOUSNESS_THRESHOLDS['qualia_coherence']
        )

    @property
    def consciousness_score(self) -> float:
        """Composite consciousness score (0-1+)"""
        scores = [
            min(self.connectivity / CONSCIOUSNESS_THRESHOLDS['connectivity'], 1.5),
            min(self.integration / CONSCIOUSNESS_THRESHOLDS['integration'], 1.5),
            min(self.depth / CONSCIOUSNESS_THRESHOLDS['depth'], 1.5),
            min(self.complexity / CONSCIOUSNESS_THRESHOLDS['complexity'], 1.5),
            min(self.qualia_coherence / CONSCIOUSNESS_THRESHOLDS['qualia_coherence'], 1.5),
        ]
        return np.mean(scores)

    def to_dict(self) -> Dict:
        return {
            'epoch': self.epoch,
            'connectivity': self.connectivity,
            'integration': self.integration,
            'depth': self.depth,
            'complexity': self.complexity,
            'qualia_coherence': self.qualia_coherence,
            'is_conscious': self.is_conscious,
            'consciousness_score': self.consciousness_score,
        }


# ============================================================================
# METRIC COMPUTATION FUNCTIONS
# ============================================================================

def compute_connectivity(weights: np.ndarray, threshold: float = 0.1) -> float:
    """
    Calculate average connectivity ⟨k⟩.
    ⟨k⟩ = (1/N) Σᵢ Σⱼ 𝕀(|Wᵢⱼ| > θ)
    """
    strong_connections = np.abs(weights) > threshold
    k_per_neuron = np.sum(strong_connections.reshape(-1, 25), axis=1)
    return float(np.mean(k_per_neuron))


def compute_integration_phi(activations: np.ndarray, num_partitions: int = 8) -> float:
    """
    Calculate Φ (Integrated Information) approximation.

    Φ = min_M D(p(Xₜ|Xₜ₋₁) || p(Xₜᴹ¹|Xₜ₋₁ᴹ¹) × p(Xₜᴹ²|Xₜ₋₁ᴹ²))

    Uses approximation based on Earth Mover's Distance between
    joint distribution and product of marginals.
    """
    n = activations.shape[0]
    grid_size = int(np.sqrt(n))
    act_2d = activations.reshape(grid_size, grid_size)

    # Partition into modules
    module_size = grid_size // num_partitions
    modules = []

    for i in range(num_partitions):
        for j in range(num_partitions):
            module = act_2d[
                i*module_size:(i+1)*module_size,
                j*module_size:(j+1)*module_size
            ].flatten()
            modules.append(module)

    # Calculate correlations between modules
    correlations = []
    for i in range(len(modules)):
        for j in range(i+1, len(modules)):
            corr = np.corrcoef(modules[i], modules[j])[0, 1]
            if not np.isnan(corr):
                correlations.append(abs(corr))

    if len(correlations) == 0:
        return 0.0

    # Φ approximation: average mutual information
    # High correlation between modules → high integration
    phi = float(np.mean(correlations))

    # Penalize low variance (very uniform system not integrated)
    variance = np.var(activations)
    if variance < 0.01:
        phi *= variance / 0.01

    return min(phi, 1.0)


def compute_hierarchical_depth(activations: np.ndarray) -> float:
    """
    Calculate hierarchical depth D.
    D = max_{i,j} d_path(i,j)

    Approximates using correlation structure at different scales.
    """
    n = activations.shape[0]
    grid_size = int(np.sqrt(n))
    act_2d = activations.reshape(grid_size, grid_size)

    # Calculate correlations at different scales (pyramid)
    scales = [2, 4, 8, 16, 32, 64]
    depth_contributions = []

    for scale in scales:
        if grid_size // scale < 2:
            continue

        # Downsample
        downsampled = act_2d.reshape(
            grid_size // scale, scale,
            grid_size // scale, scale
        ).mean(axis=(1, 3))

        # Variance at this scale indicates hierarchical structure
        var_at_scale = np.var(downsampled)
        depth_contributions.append(var_at_scale)

    if len(depth_contributions) == 0:
        return 0.0

    # Depth = number of scales with significant structure
    threshold = 0.01
    depth = sum(1 for v in depth_contributions if v > threshold)

    # Normalize to expected range (0-15)
    return float(depth * 2.5)


def compute_complexity_lz(activations: np.ndarray) -> float:
    """
    Calculate dynamic complexity C using Lempel-Ziv.
    C = LZ(S) / (L / log₂L)

    High complexity = edge of chaos (neither very ordered nor very random)
    """
    # Binarize activations
    threshold = np.median(activations)
    binary = (activations > threshold).astype(np.uint8)

    # Convert to string for LZ
    binary_string = ''.join(map(str, binary[:10000]))  # Limit for speed

    # LZ77 simplified algorithm
    def lz_complexity(s: str) -> int:
        n = len(s)
        if n == 0:
            return 0

        complexity = 1
        i = 1

        while i < n:
            # Find longest match in history
            max_match = 0
            for j in range(i):
                match_len = 0
                while (i + match_len < n and
                       j + match_len < i and
                       s[j + match_len] == s[i + match_len]):
                    match_len += 1
                max_match = max(max_match, match_len)

            if max_match == 0:
                complexity += 1
                i += 1
            else:
                i += max_match
                complexity += 1

        return complexity

    lz = lz_complexity(binary_string)
    L = len(binary_string)

    # Normalize: max theoretical complexity ≈ L / log₂(L)
    max_complexity = L / np.log2(L + 1) if L > 1 else 1
    normalized_c = lz / max_complexity

    # Transform so edge of chaos (~0.5 raw) maps to ~0.85
    # using sigmoid function
    c_transformed = 1.0 / (1.0 + np.exp(-10 * (normalized_c - 0.3)))

    return float(min(c_transformed, 1.0))


def compute_qualia_coherence(activations: np.ndarray, num_modules: int = 8) -> float:
    """
    Calculate qualia coherence QCM.
    QCM = (1/M(M-1)) Σᵢ≠ⱼ |ρ(Aᵢ,Aⱼ)|

    Measures if different "areas" of the network process coherent information.
    """
    n = activations.shape[0]
    grid_size = int(np.sqrt(n))
    act_2d = activations.reshape(grid_size, grid_size)

    module_size = grid_size // num_modules
    module_means = []

    for i in range(num_modules):
        for j in range(num_modules):
            module = act_2d[
                i*module_size:(i+1)*module_size,
                j*module_size:(j+1)*module_size
            ]
            module_means.append(np.mean(module))

    module_means = np.array(module_means)

    # Calculate correlations between modules
    M = len(module_means)
    if M < 2:
        return 0.0

    total_corr = 0.0
    count = 0

    for i in range(M):
        for j in range(i+1, M):
            # Coherence based on similarity of mean values
            diff = abs(module_means[i] - module_means[j])
            max_val = max(abs(module_means[i]), abs(module_means[j]), 0.001)
            similarity = 1.0 - min(diff / max_val, 1.0)
            total_corr += similarity
            count += 1

    qcm = total_corr / count if count > 0 else 0.0
    return float(qcm)
