#!/usr/bin/env python3
"""
Master Audit Runner for Experiment 1

Executes dual independent audit:
- Approach A: Theoretical Physics Validation
- Approach B: Numerical/Computational Validation

Usage:
    python run_dual_audit.py                    # Run both audits
    python run_dual_audit.py --physics-only     # Physics audit only
    python run_dual_audit.py --numerical-only   # Numerical audit only
"""

import subprocess
import sys
import json
import time
from pathlib import Path


def run_audit(script_name, args=None):
    """
    Run a single audit script.

    Args:
        script_name: Name of audit script
        args: Additional arguments

    Returns:
        (exit_code, elapsed_time)
    """
    cmd = [sys.executable, f"audits/{script_name}"]
    if args:
        cmd.extend(args)

    print(f"Running: {script_name}")
    print("-" * 80)
    start = time.time()
    result = subprocess.run(cmd, cwd=Path(__file__).parent)
    elapsed = time.time() - start
    print("-" * 80)

    return result.returncode, elapsed


def run_dual_audit(physics_only=False, numerical_only=False, output_file=None):
    """
    Run dual independent audit.

    Args:
        physics_only: Run physics audit only
        numerical_only: Run numerical audit only
        output_file: Save combined results

    Returns:
        Exit code
    """
    print("=" * 80)
    print("EXPERIMENT 1 - DUAL INDEPENDENT AUDIT")
    print("=" * 80)
    print()
    print("Two complementary approaches:")
    print("  Approach A: Theoretical Physics Validation")
    print("  Approach B: Numerical/Computational Validation")
    print()

    audits = []

    # Energy Discrepancy Audit (CRITICAL - always run)
    print("\n" + "=" * 80)
    print("CRITICAL AUDIT: ENERGY DISCREPANCY RESOLUTION")
    print("=" * 80)
    print()
    args = ["--epochs", "2000", "--grid-size", "128",
            "--output", "audit_energy_discrepancy.json"]
    exit_code, elapsed = run_audit("audit_energy_discrepancy.py", args)
    audits.append({
        'name': 'Energy Discrepancy Resolution',
        'approach': 'Both',
        'status': 'COMPLETE' if exit_code == 0 else 'FAILED',
        'elapsed_seconds': elapsed
    })

    # Approach A: Physics (if it exists and not numerical_only)
    approach_a_script = Path(__file__).parent / "audits" / "audit_approach_a_physics.py"
    if approach_a_script.exists() and not numerical_only:
        print("\n" + "=" * 80)
        print("APPROACH A: THEORETICAL PHYSICS VALIDATION")
        print("=" * 80)
        print()
        exit_code, elapsed = run_audit("audit_approach_a_physics.py")
        audits.append({
            'name': 'Physics Validation',
            'approach': 'A',
            'status': 'COMPLETE' if exit_code == 0 else 'FAILED',
            'elapsed_seconds': elapsed
        })

    # Approach B: Numerical (if it exists and not physics_only)
    approach_b_script = Path(__file__).parent / "audits" / "audit_approach_b_numerical.py"
    if approach_b_script.exists() and not physics_only:
        print("\n" + "=" * 80)
        print("APPROACH B: NUMERICAL/COMPUTATIONAL VALIDATION")
        print("=" * 80)
        print()
        exit_code, elapsed = run_audit("audit_approach_b_numerical.py")
        audits.append({
            'name': 'Numerical Validation',
            'approach': 'B',
            'status': 'COMPLETE' if exit_code == 0 else 'FAILED',
            'elapsed_seconds': elapsed
        })

    # Summary
    print()
    print("=" * 80)
    print("DUAL AUDIT RESULTS SUMMARY")
    print("=" * 80)
    for a in audits:
        status_symbol = "✓" if a['status'] == 'COMPLETE' else "✗"
        print(f"{status_symbol} {a['name']:40s} [Approach {a['approach']:4s}]: "
              f"{a['status']:8s} ({a['elapsed_seconds']:.1f}s)")
    print("=" * 80)

    # Save results
    if output_file:
        with open(output_file, 'w') as f:
            json.dump({
                'audits': audits,
                'all_complete': all(a['status'] == 'COMPLETE' for a in audits)
            }, f, indent=2)
        print(f"\nResults saved to: {output_file}")

    all_complete = all(a['status'] == 'COMPLETE' for a in audits)
    return 0 if all_complete else 1


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Run dual independent audit for Experiment 1"
    )

    parser.add_argument("--physics-only", action="store_true",
                       help="Run physics audit only")
    parser.add_argument("--numerical-only", action="store_true",
                       help="Run numerical audit only")
    parser.add_argument("-o", "--output", type=str, default="audit_results.json",
                       help="Output JSON file for combined results")

    args = parser.parse_args()

    exit_code = run_dual_audit(
        physics_only=args.physics_only,
        numerical_only=args.numerical_only,
        output_file=args.output
    )

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
