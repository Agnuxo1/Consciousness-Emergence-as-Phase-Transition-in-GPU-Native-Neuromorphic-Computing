"""
Scaling Benchmark for Experiment 1

Tests algorithmic complexity and scaling behavior with grid size.
Expected: O(N²) where N = grid_size
"""

import numpy as np
import json
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.cpu_reference import cpu_simulation_step


def benchmark_scaling(grid_sizes=[32, 64, 128, 256, 512], num_epochs=50):
    """
    Benchmark scaling behavior.

    Args:
        grid_sizes: List of grid sizes
        num_epochs: Number of epochs per test

    Returns:
        Scaling analysis results
    """
    print("=" * 80)
    print("SCALING BENCHMARK - Algorithmic Complexity")
    print("=" * 80)
    print()

    results = {
        'benchmark_type': 'scaling',
        'num_epochs': num_epochs,
        'measurements': []
    }

    for grid_size in grid_sizes:
        print(f"Grid size: {grid_size}×{grid_size} ({grid_size**2:,} nodes)")

        # Initialize
        np.random.seed(42)
        state = np.zeros((grid_size, grid_size, 4), dtype=np.float32)
        state[:, :, 0] = np.random.randn(grid_size, grid_size) * 0.1
        state[:, :, 1] = np.random.randn(grid_size, grid_size) * 0.05

        weights = np.random.randn(grid_size * grid_size, 25).astype(np.float32) * 0.1

        # Benchmark
        start_time = time.time()

        for epoch in range(num_epochs):
            state = cpu_simulation_step(state, weights, epoch)

        elapsed = time.time() - start_time

        epochs_per_sec = num_epochs / elapsed if elapsed > 0 else 0
        time_per_node_us = (elapsed / num_epochs) / (grid_size ** 2) * 1e6

        measurement = {
            'grid_size': grid_size,
            'num_nodes': grid_size * grid_size,
            'total_time_seconds': float(elapsed),
            'epochs_per_second': float(epochs_per_sec),
            'time_per_node_microseconds': float(time_per_node_us)
        }

        results['measurements'].append(measurement)

        print(f"  Time: {elapsed:.2f}s | Epochs/sec: {epochs_per_sec:.1f} | "
              f"Time/node: {time_per_node_us:.3f} μs")
        print()

    # Analyze scaling
    print("=" * 80)
    print("SCALING ANALYSIS")
    print("=" * 80)

    # Fit to O(N^p)
    sizes = np.array([m['grid_size'] for m in results['measurements']])
    times = np.array([m['total_time_seconds'] for m in results['measurements']])

    # Log-log fit
    log_sizes = np.log(sizes)
    log_times = np.log(times)
    coeffs = np.polyfit(log_sizes, log_times, 1)
    exponent = coeffs[0]

    results['scaling_analysis'] = {
        'fitted_exponent': float(exponent),
        'expected_exponent': 2.0,  # O(N²)
        'scaling_type': f"O(N^{exponent:.2f})"
    }

    print(f"Fitted exponent: {exponent:.3f}")
    print(f"Expected: 2.0 (O(N²))")

    if 1.8 <= exponent <= 2.2:
        print("✓ Scaling matches expected O(N²)")
    else:
        print("⚠ Scaling differs from expected")

    print("=" * 80)

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run scaling benchmark")
    parser.add_argument("--epochs", type=int, default=50,
                       help="Number of epochs per grid size")
    parser.add_argument("--grid-sizes", type=int, nargs="+",
                       default=[32, 64, 128, 256],
                       help="Grid sizes to test")
    parser.add_argument("--output", type=str, default="benchmark_scaling_results.json",
                       help="Output JSON file")

    args = parser.parse_args()

    results = benchmark_scaling(
        grid_sizes=args.grid_sizes,
        num_epochs=args.epochs
    )

    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {args.output}")
    sys.exit(0)
