"""
Visualization: Energy Evolution

Plots the evolution of total energy over time from benchmark or audit results.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path


def plot_energy_evolution(data_file, output_file="energy_evolution.png"):
    """
    Plot energy evolution from benchmark/audit data.

    Args:
        data_file: JSON file with energy history
        output_file: Output PNG file
    """
    # Load data
    with open(data_file, 'r') as f:
        data = json.load(f)

    # Extract energy history
    if 'energy_analysis' in data and 'history' in data['energy_analysis']:
        energies = data['energy_analysis']['history']
        epochs = list(range(len(energies)))
    elif 'energy_behavior' in data and 'history' in data['energy_behavior']:
        energies = data['energy_behavior']['history']
        epochs = list(range(len(energies)))
    else:
        print("No energy history found in data file")
        return

    # Create plot
    plt.figure(figsize=(12, 6))

    plt.plot(epochs, energies, linewidth=2, color='#2E86AB')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Total Energy', fontsize=12)
    plt.title('Energy Evolution - Experiment 1', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)

    # Annotate initial and final
    if len(energies) > 0:
        plt.scatter([0], [energies[0]], color='green', s=100, zorder=5,
                   label=f'Initial: {energies[0]:.2f}')
        plt.scatter([epochs[-1]], [energies[-1]], color='red', s=100, zorder=5,
                   label=f'Final: {energies[-1]:.2f}')
        plt.legend()

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Energy evolution plot saved to: {output_file}")
    plt.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plot energy evolution")
    parser.add_argument("--input", type=str, default="benchmark_accuracy_results.json",
                       help="Input JSON file")
    parser.add_argument("--output", type=str, default="energy_evolution.png",
                       help="Output PNG file")

    args = parser.parse_args()

    if not Path(args.input).exists():
        print(f"ERROR: Input file not found: {args.input}")
        sys.exit(1)

    plot_energy_evolution(args.input, args.output)
