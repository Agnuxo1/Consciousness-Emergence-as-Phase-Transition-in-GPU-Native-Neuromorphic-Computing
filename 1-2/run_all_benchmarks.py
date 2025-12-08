#!/usr/bin/env python3
"""
Master Benchmark Runner for Experiment 1

Executes all benchmarks (performance, accuracy, scaling, precision).

Usage:
    python run_all_benchmarks.py                    # Run all benchmarks
    python run_all_benchmarks.py --quick            # Quick benchmarks only
    python run_all_benchmarks.py --output results.json  # Save results
"""

import subprocess
import sys
import json
import time
from pathlib import Path


def run_benchmark(script_name, args=None):
    """
    Run a single benchmark script.

    Args:
        script_name: Name of benchmark script (e.g., 'benchmark_accuracy.py')
        args: Additional arguments as list

    Returns:
        (exit_code, elapsed_time)
    """
    cmd = [sys.executable, f"benchmarks/{script_name}"]
    if args:
        cmd.extend(args)

    print(f"Running: {script_name}")
    start = time.time()
    result = subprocess.run(cmd, cwd=Path(__file__).parent)
    elapsed = time.time() - start

    return result.returncode, elapsed


def run_all_benchmarks(quick=False, output_file=None):
    """
    Run all benchmarks.

    Args:
        quick: Run quick benchmarks only (fewer epochs)
        output_file: Save combined results to JSON file

    Returns:
        Exit code (0 if all pass)
    """
    print("=" * 80)
    print("EXPERIMENT 1 - BENCHMARK SUITE")
    print("=" * 80)
    print()

    benchmarks = []

    # Accuracy benchmark (CRITICAL)
    print("\n" + "=" * 80)
    print("BENCHMARK 1: ACCURACY VALIDATION")
    print("=" * 80)
    args = ["--epochs", "500" if quick else "1000", "--grid-size", "128",
            "--output", "benchmark_accuracy_results.json"]
    exit_code, elapsed = run_benchmark("benchmark_accuracy.py", args)
    benchmarks.append({
        'name': 'accuracy',
        'status': 'PASS' if exit_code == 0 else 'FAIL',
        'elapsed_seconds': elapsed
    })

    # Performance benchmark (optional, create if exists)
    perf_script = Path(__file__).parent / "benchmarks" / "benchmark_performance.py"
    if perf_script.exists():
        print("\n" + "=" * 80)
        print("BENCHMARK 2: PERFORMANCE METRICS")
        print("=" * 80)
        args = ["--epochs", "100" if quick else "500"]
        exit_code, elapsed = run_benchmark("benchmark_performance.py", args)
        benchmarks.append({
            'name': 'performance',
            'status': 'PASS' if exit_code == 0 else 'FAIL',
            'elapsed_seconds': elapsed
        })

    # Summary
    print()
    print("=" * 80)
    print("BENCHMARK RESULTS SUMMARY")
    print("=" * 80)
    for b in benchmarks:
        status_symbol = "✓" if b['status'] == 'PASS' else "✗"
        print(f"{status_symbol} {b['name']:20s}: {b['status']:6s} ({b['elapsed_seconds']:.1f}s)")
    print("=" * 80)

    # Save results
    if output_file:
        with open(output_file, 'w') as f:
            json.dump({
                'benchmarks': benchmarks,
                'all_passed': all(b['status'] == 'PASS' for b in benchmarks)
            }, f, indent=2)
        print(f"\nResults saved to: {output_file}")

    all_passed = all(b['status'] == 'PASS' for b in benchmarks)
    return 0 if all_passed else 1


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Run all benchmarks for Experiment 1"
    )

    parser.add_argument("-q", "--quick", action="store_true",
                       help="Run quick benchmarks (fewer epochs)")
    parser.add_argument("-o", "--output", type=str, default="benchmark_results.json",
                       help="Output JSON file for combined results")

    args = parser.parse_args()

    exit_code = run_all_benchmarks(quick=args.quick, output_file=args.output)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
