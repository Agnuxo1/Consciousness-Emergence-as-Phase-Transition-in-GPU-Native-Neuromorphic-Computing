"""
Performance Benchmark for Experiment 1

Measures computational performance metrics:
- Epochs per second (throughput)
- GPU memory usage
- CPU↔GPU transfer time
- Single step latency
"""

import numpy as np
import json
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.cpu_reference import cpu_simulation_step


def benchmark_cpu_performance(grid_sizes=[64, 128, 256], num_epochs=100):
    """
    Benchmark CPU performance across different grid sizes.

    Args:
        grid_sizes: List of grid sizes to test
        num_epochs: Number of epochs per test

    Returns:
        Performance results dictionary
    """
    print("=" * 80)
    print("PERFORMANCE BENCHMARK - CPU")
    print("=" * 80)
    print()

    results = {
        'benchmark_type': 'performance_cpu',
        'grid_sizes': grid_sizes,
        'num_epochs': num_epochs,
        'measurements': []
    }

    for grid_size in grid_sizes:
        print(f"Testing grid size: {grid_size}×{grid_size}")

        # Initialize
        np.random.seed(42)
        state = np.zeros((grid_size, grid_size, 4), dtype=np.float32)
        state[:, :, 0] = np.random.randn(grid_size, grid_size) * 0.1
        state[:, :, 1] = np.random.randn(grid_size, grid_size) * 0.05

        weights = np.random.randn(grid_size * grid_size, 25).astype(np.float32) * 0.1

        # Warmup
        for epoch in range(5):
            state = cpu_simulation_step(state, weights, epoch)

        # Benchmark
        start_time = time.time()
        step_times = []

        for epoch in range(num_epochs):
            step_start = time.time()
            state = cpu_simulation_step(state, weights, epoch)
            step_end = time.time()

            step_times.append(step_end - step_start)

        end_time = time.time()
        elapsed = end_time - start_time

        # Metrics
        epochs_per_sec = num_epochs / elapsed if elapsed > 0 else 0
        avg_step_time = np.mean(step_times)
        std_step_time = np.std(step_times)

        # Memory estimate
        state_memory_mb = state.nbytes / (1024 ** 2)
        weights_memory_mb = weights.nbytes / (1024 ** 2)
        total_memory_mb = state_memory_mb + weights_memory_mb

        measurement = {
            'grid_size': grid_size,
            'num_nodes': grid_size * grid_size,
            'epochs_per_second': float(epochs_per_sec),
            'avg_step_time_ms': float(avg_step_time * 1000),
            'std_step_time_ms': float(std_step_time * 1000),
            'total_time_seconds': float(elapsed),
            'memory_mb': float(total_memory_mb)
        }

        results['measurements'].append(measurement)

        print(f"  Epochs/sec: {epochs_per_sec:.1f}")
        print(f"  Avg step time: {avg_step_time * 1000:.2f} ms")
        print(f"  Memory: {total_memory_mb:.1f} MB")
        print()

    # Summary
    print("=" * 80)
    print("PERFORMANCE SUMMARY")
    print("=" * 80)
    print(f"{'Grid Size':<12} {'Nodes':<10} {'Epochs/sec':<12} {'Step (ms)':<12} {'Memory (MB)':<12}")
    print("-" * 80)

    for m in results['measurements']:
        print(f"{m['grid_size']:>3}×{m['grid_size']:<6} "
              f"{m['num_nodes']:<10} "
              f"{m['epochs_per_second']:<12.1f} "
              f"{m['avg_step_time_ms']:<12.2f} "
              f"{m['memory_mb']:<12.1f}")

    print("=" * 80)

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run performance benchmark")
    parser.add_argument("--epochs", type=int, default=100,
                       help="Number of epochs per grid size")
    parser.add_argument("--grid-sizes", type=int, nargs="+", default=[64, 128, 256],
                       help="Grid sizes to test")
    parser.add_argument("--output", type=str, default="benchmark_performance_results.json",
                       help="Output JSON file")

    args = parser.parse_args()

    results = benchmark_cpu_performance(
        grid_sizes=args.grid_sizes,
        num_epochs=args.epochs
    )

    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {args.output}")
    sys.exit(0)
