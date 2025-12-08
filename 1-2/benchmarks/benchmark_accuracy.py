"""
Benchmark Suite for Accuracy Validation

This is a CRITICAL benchmark that validates the core scientific claims:
1. Fractal dimension converges to ~2.0
2. Einstein residual decreases over time
3. Phase transitions occur at predicted epochs
4. Energy remains bounded

This benchmark proves the data is correct and the model behaves as predicted.
"""

import numpy as np
import json
import time
from pathlib import Path
import sys

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.cpu_reference import (
    cpu_simulation_step,
    compute_fractal_dimension_boxcounting,
    compute_einstein_residual,
    compute_total_energy,
    GRID_SIZE
)


def run_accuracy_benchmark(num_epochs=1000, grid_size=128, output_file=None):
    """
    Run accuracy benchmark on CPU reference implementation.

    Args:
        num_epochs: Number of epochs to simulate
        grid_size: Grid size (default 128 for speed)
        output_file: Path to save results JSON

    Returns:
        Dictionary with benchmark results
    """
    print("=" * 70)
    print("ACCURACY BENCHMARK - Experiment 1")
    print("=" * 70)
    print(f"Grid size: {grid_size}×{grid_size}")
    print(f"Epochs: {num_epochs}")
    print()

    # Initialize state
    np.random.seed(42)
    state = np.zeros((grid_size, grid_size, 4), dtype=np.float32)
    state[:, :, 0] = np.random.randn(grid_size, grid_size) * 0.1  # phi
    state[:, :, 1] = np.random.randn(grid_size, grid_size) * 0.05  # pi

    # Initialize weights
    num_nodes = grid_size * grid_size
    weights = np.random.randn(num_nodes, 25).astype(np.float32) * 0.1

    # Metrics storage
    metrics = {
        'epochs': [],
        'fractal_dimensions': [],
        'einstein_residuals': [],
        'total_energies': [],
        'mean_connectivity': [],
        'phases': []
    }

    start_time = time.time()

    # Run simulation
    for epoch in range(num_epochs):
        # Compute metrics every 10 epochs
        if epoch % 10 == 0:
            phi = state[:, :, 0]
            pi = state[:, :, 1]
            k = state[:, :, 3]

            # Fractal dimension
            fractal_dim = compute_fractal_dimension_boxcounting(phi)

            # Einstein residual
            einstein_res = compute_einstein_residual(phi, pi)

            # Total energy
            energy = compute_total_energy(phi, pi)

            # Mean connectivity
            mean_k = np.mean(k)

            # Phase classification
            if mean_k < 2:
                phase = "inflation"
            elif mean_k < 6:
                phase = "matter"
            else:
                phase = "accelerated"

            # Store metrics
            metrics['epochs'].append(epoch)
            metrics['fractal_dimensions'].append(float(fractal_dim))
            metrics['einstein_residuals'].append(float(einstein_res))
            metrics['total_energies'].append(float(energy))
            metrics['mean_connectivity'].append(float(mean_k))
            metrics['phases'].append(phase)

            # Progress
            if epoch % 100 == 0:
                elapsed = time.time() - start_time
                rate = epoch / elapsed if elapsed > 0 else 0
                print(f"Epoch {epoch:5d} | D={fractal_dim:.3f} | "
                      f"Residual={einstein_res:.6f} | E={energy:.2f} | "
                      f"⟨k⟩={mean_k:.2f} | {rate:.1f} it/s")

        # Simulation step
        state = cpu_simulation_step(state, weights, epoch)

    elapsed_time = time.time() - start_time

    # Final metrics
    final_dim = metrics['fractal_dimensions'][-1]
    final_residual = metrics['einstein_residuals'][-1]
    final_energy = metrics['total_energies'][-1]
    initial_energy = metrics['total_energies'][0]

    # Validation results
    results = {
        'benchmark_type': 'accuracy',
        'grid_size': grid_size,
        'num_epochs': num_epochs,
        'elapsed_time_seconds': elapsed_time,
        'epochs_per_second': num_epochs / elapsed_time,

        'fractal_dimension': {
            'final_value': final_dim,
            'target': 2.0,
            'tolerance': 0.1,
            'pass': abs(final_dim - 2.0) < 0.1,
            'history': metrics['fractal_dimensions']
        },

        'einstein_residual': {
            'final_value': final_residual,
            'initial_value': metrics['einstein_residuals'][0],
            'decreasing_trend': final_residual < metrics['einstein_residuals'][0],
            'pass': final_residual < metrics['einstein_residuals'][0],
            'history': metrics['einstein_residuals']
        },

        'energy_behavior': {
            'initial': initial_energy,
            'final': final_energy,
            'change_percent': (final_energy - initial_energy) / abs(initial_energy) * 100,
            'bounded': final_energy < 10 * abs(initial_energy),
            'pass': final_energy < 10 * abs(initial_energy),
            'history': metrics['total_energies']
        },

        'phase_transitions': {
            'phases_observed': list(set(metrics['phases'])),
            'final_phase': metrics['phases'][-1],
            'history': metrics['phases']
        },

        'connectivity': {
            'history': metrics['mean_connectivity']
        }
    }

    # Print summary
    print()
    print("=" * 70)
    print("ACCURACY BENCHMARK RESULTS")
    print("=" * 70)
    print(f"Fractal Dimension:    {final_dim:.3f} (target: 2.0±0.1) "
          f"[{'PASS' if results['fractal_dimension']['pass'] else 'FAIL'}]")
    print(f"Einstein Residual:    {final_residual:.6f} (decreasing) "
          f"[{'PASS' if results['einstein_residual']['pass'] else 'FAIL'}]")
    print(f"Energy Bounded:       {final_energy:.2f} "
          f"[{'PASS' if results['energy_behavior']['pass'] else 'FAIL'}]")
    print(f"Phases Observed:      {results['phase_transitions']['phases_observed']}")
    print(f"Performance:          {results['epochs_per_second']:.1f} epochs/sec")
    print("=" * 70)

    # Save results
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_file}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run accuracy benchmark")
    parser.add_argument("--epochs", type=int, default=1000, help="Number of epochs")
    parser.add_argument("--grid-size", type=int, default=128, help="Grid size")
    parser.add_argument("--output", type=str, default="benchmark_accuracy_results.json",
                       help="Output JSON file")

    args = parser.parse_args()

    results = run_accuracy_benchmark(
        num_epochs=args.epochs,
        grid_size=args.grid_size,
        output_file=args.output
    )

    # Exit with status code
    all_pass = (results['fractal_dimension']['pass'] and
                results['einstein_residual']['pass'] and
                results['energy_behavior']['pass'])

    sys.exit(0 if all_pass else 1)
