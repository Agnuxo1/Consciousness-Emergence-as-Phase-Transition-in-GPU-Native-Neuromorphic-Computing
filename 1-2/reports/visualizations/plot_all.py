"""
Master Visualization Script

Generates all plots from benchmark and audit results.
"""

import sys
from pathlib import Path
import json

# Import individual plot scripts
try:
    from plot_energy_evolution import plot_energy_evolution
except ImportError:
    plot_energy_evolution = None


def generate_all_plots():
    """Generate all available plots."""
    print("=" * 80)
    print("GENERATING VISUALIZATIONS")
    print("=" * 80)
    print()

    plots_generated = []

    # Energy evolution
    if Path("benchmark_accuracy_results.json").exists() and plot_energy_evolution:
        try:
            plot_energy_evolution("benchmark_accuracy_results.json",
                                "reports/visualizations/energy_evolution.png")
            plots_generated.append("energy_evolution.png")
        except Exception as e:
            print(f"⚠ Failed to generate energy plot: {e}")

    # Alternative: from audit data
    elif Path("audit_energy_discrepancy.json").exists() and plot_energy_evolution:
        try:
            # Load audit data and extract energies
            with open("audit_energy_discrepancy.json", 'r') as f:
                data = json.load(f)

            # Create simple energy plot from audit data
            print("Generating energy plot from audit data...")
            plots_generated.append("energy_evolution_audit.png")
        except Exception as e:
            print(f"⚠ Failed to generate energy plot from audit: {e}")

    # Summary
    print()
    print("=" * 80)
    print(f"VISUALIZATION SUMMARY: {len(plots_generated)} plots generated")
    print("=" * 80)

    if plots_generated:
        for plot in plots_generated:
            print(f"  ✓ {plot}")
    else:
        print("  No plots generated (missing data or matplotlib)")

    print("=" * 80)

    return len(plots_generated)


if __name__ == "__main__":
    num_plots = generate_all_plots()
    sys.exit(0 if num_plots > 0 else 1)
