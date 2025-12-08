"""
Pytest Configuration and Fixtures for Experiment 1 Tests

This module provides shared fixtures for all test suites, including:
- GPU device initialization
- Sample states and fields
- CPU reference implementations
- Validation utilities
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.cpu_reference import *

# ============================================================================
# SESSION-SCOPED FIXTURES (initialized once per test session)
# ============================================================================

@pytest.fixture(scope="session")
def random_seed():
    """Fixed random seed for reproducibility."""
    seed = 42
    np.random.seed(seed)
    return seed


@pytest.fixture(scope="session")
def grid_size():
    """Standard grid size for tests."""
    return 64  # Smaller than 256 for faster tests


@pytest.fixture(scope="session")
def test_constants():
    """Physical constants for tests."""
    return {
        'L_PLANCK': L_PLANCK,
        'COSMOLOGICAL_CONSTANT': COSMOLOGICAL_CONSTANT,
        'CRITICAL_CONNECTIVITY': CRITICAL_CONNECTIVITY,
        'TIME_STEP': TIME_STEP,
        'LEARNING_RATE': LEARNING_RATE,
    }


# ============================================================================
# FUNCTION-SCOPED FIXTURES (created for each test)
# ============================================================================

@pytest.fixture
def zero_field(grid_size):
    """Zero field for testing boundary conditions."""
    return np.zeros((grid_size, grid_size), dtype=np.float32)


@pytest.fixture
def constant_field(grid_size):
    """Constant field (should have zero Laplacian)."""
    return np.ones((grid_size, grid_size), dtype=np.float32) * 2.5


@pytest.fixture
def linear_field(grid_size):
    """Linear gradient field (should have zero Laplacian)."""
    x = np.linspace(0, 1, grid_size)
    y = np.linspace(0, 1, grid_size)
    X, Y = np.meshgrid(x, y)
    return (X + Y).astype(np.float32)


@pytest.fixture
def gaussian_field(grid_size):
    """Gaussian field (known Laplacian)."""
    x = np.linspace(-3, 3, grid_size)
    y = np.linspace(-3, 3, grid_size)
    X, Y = np.meshgrid(x, y)
    R2 = X**2 + Y**2
    sigma = 1.0
    return np.exp(-R2 / (2 * sigma**2)).astype(np.float32)


@pytest.fixture
def random_field(grid_size, random_seed):
    """Random field for general testing."""
    np.random.seed(random_seed)
    return np.random.randn(grid_size, grid_size).astype(np.float32) * 0.1


@pytest.fixture
def sample_state(grid_size, random_seed):
    """Sample 4-channel state [phi, pi, R, k]."""
    np.random.seed(random_seed)
    state = np.zeros((grid_size, grid_size, 4), dtype=np.float32)

    # phi: scalar field
    state[:, :, 0] = np.random.randn(grid_size, grid_size) * 0.1

    # pi: conjugate momentum
    state[:, :, 1] = np.random.randn(grid_size, grid_size) * 0.05

    # R: curvature (computed from phi)
    state[:, :, 2] = compute_laplacian_2d(state[:, :, 0])

    # k: connectivity (random initial values)
    state[:, :, 3] = np.random.rand(grid_size, grid_size) * 10.0

    return state


@pytest.fixture
def sample_weights(grid_size, random_seed):
    """Sample weight matrix for connectivity tests."""
    np.random.seed(random_seed)
    num_nodes = grid_size * grid_size
    num_neighbors = 25  # 5×5 neighborhood

    # Initialize with small random weights
    weights = np.random.randn(num_nodes, num_neighbors).astype(np.float32) * 0.1

    # Ensure some weights exceed threshold
    mask = np.random.rand(num_nodes, num_neighbors) > 0.7
    weights[mask] *= 5.0  # Amplify some weights

    return weights


# ============================================================================
# ANALYTICAL SOLUTIONS FOR VALIDATION
# ============================================================================

@pytest.fixture
def harmonic_oscillator_solution():
    """
    Analytical solution for harmonic oscillator: φ'' = -ω²φ
    Solution: φ(t) = A*cos(ωt) + B*sin(ωt)
    """
    def solution(t, A=1.0, B=0.0, omega=1.0):
        return A * np.cos(omega * t) + B * np.sin(omega * t)

    def momentum(t, A=1.0, B=0.0, omega=1.0):
        return -A * omega * np.sin(omega * t) + B * omega * np.cos(omega * t)

    return {'phi': solution, 'pi': momentum}


# ============================================================================
# TOLERANCE UTILITIES
# ============================================================================

@pytest.fixture
def tolerance_config():
    """Standard tolerance configuration for numerical comparisons."""
    return {
        'rtol': 1e-4,  # Relative tolerance
        'atol': 1e-5,  # Absolute tolerance
        'energy_drift_max': 0.01,  # 1% energy drift allowed
    }


# ============================================================================
# VALIDATION HELPERS
# ============================================================================

@pytest.fixture
def assert_fields_close():
    """Helper function to assert two fields are numerically close."""
    def _assert_close(field1, field2, rtol=1e-4, atol=1e-5, field_name="field"):
        if not np.allclose(field1, field2, rtol=rtol, atol=atol):
            max_diff = np.max(np.abs(field1 - field2))
            rel_diff = np.max(np.abs((field1 - field2) / (field2 + 1e-10)))
            raise AssertionError(
                f"{field_name} mismatch:\n"
                f"  Max abs difference: {max_diff:.10f}\n"
                f"  Max rel difference: {rel_diff:.10f}\n"
                f"  Tolerance: rtol={rtol}, atol={atol}"
            )
    return _assert_close


@pytest.fixture
def compute_relative_error():
    """Compute relative error between two arrays."""
    def _rel_error(computed, expected):
        return np.max(np.abs((computed - expected) / (expected + 1e-10)))
    return _rel_error


# ============================================================================
# SKIP MARKERS FOR CONDITIONAL TESTS
# ============================================================================

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "gpu: mark test as requiring GPU hardware"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow-running"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )


# ============================================================================
# GPU FIXTURES (optional, may fail if GPU not available)
# ============================================================================

@pytest.fixture(scope="session")
def gpu_available():
    """Check if GPU is available for testing."""
    try:
        import wgpu
        adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
        device = adapter.request_device_sync()
        return True
    except Exception as e:
        return False


@pytest.fixture
def skip_if_no_gpu(gpu_available):
    """Skip test if GPU is not available."""
    if not gpu_available:
        pytest.skip("GPU not available")
