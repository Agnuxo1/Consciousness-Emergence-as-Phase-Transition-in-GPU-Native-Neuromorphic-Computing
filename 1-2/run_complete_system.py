"""
Master Runner: Complete Testing System for Both Experiments

Runs comprehensive validation for:
- Experiment 1: Spacetime Emergence
- Experiment 2: Consciousness Emergence

Includes tests, benchmarks, audits, and final report generation.

Author: Claude Code Testing Framework
"""

import sys
import subprocess
import time
import json
from pathlib import Path
from typing import Dict, List


def run_command(cmd: List[str], description: str, timeout: int = 3600) -> Dict:
    """
    Run a command and capture results.

    Args:
        cmd: Command and arguments
        description: Human-readable description
        timeout: Timeout in seconds

    Returns:
        Results dictionary
    """
    print(f"\n{'='*70}")
    print(f"RUNNING: {description}")
    print(f"{'='*70}")
    print(f"Command: {' '.join(cmd)}")
    print()

    start_time = time.time()

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding='utf-8',
            errors='replace'
        )

        elapsed = time.time() - start_time

        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)

        success = result.returncode == 0

        print(f"\n{'='*70}")
        if success:
            print(f"✓ {description} - PASSED ({elapsed:.1f}s)")
        else:
            print(f"✗ {description} - FAILED ({elapsed:.1f}s)")
        print(f"{'='*70}")

        return {
            'description': description,
            'command': ' '.join(cmd),
            'success': success,
            'returncode': result.returncode,
            'elapsed_time': elapsed,
            'stdout': result.stdout,
            'stderr': result.stderr,
        }

    except subprocess.TimeoutExpired:
        elapsed = time.time() - start_time
        print(f"\n✗ {description} - TIMEOUT after {elapsed:.1f}s")
        return {
            'description': description,
            'command': ' '.join(cmd),
            'success': False,
            'returncode': -1,
            'elapsed_time': elapsed,
            'error': 'Timeout',
        }

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n✗ {description} - ERROR: {e}")
        return {
            'description': description,
            'command': ' '.join(cmd),
            'success': False,
            'returncode': -1,
            'elapsed_time': elapsed,
            'error': str(e),
        }


def main():
    """Run complete validation system."""
    print("""
================================================================================
COMPLETE VALIDATION SYSTEM - EXPERIMENTS 1 & 2
================================================================================

This script runs the entire testing/benchmarking/auditing system for:
  • Experiment 1: Spacetime Emergence from Network Connectivity
  • Experiment 2: Consciousness Emergence as Phase Transition

Components:
  1. System verification
  2. Unit tests (Experiments 1 & 2)
  3. Benchmarks (Experiments 1 & 2)
  4. Independent audits (dual-approach validation)
  5. Final comprehensive report

Estimated time: 4-6 hours
================================================================================
""")

    input("Press ENTER to start, or CTRL+C to cancel...")

    start_time_total = time.time()
    results = []

    # ========================================================================
    # PHASE 1: System Verification
    # ========================================================================
    print("\n\n")
    print("#" * 70)
    print("# PHASE 1: SYSTEM VERIFICATION")
    print("#" * 70)

    results.append(run_command(
        [sys.executable, 'VERIFY_SYSTEM.py'],
        'System Verification',
        timeout=60
    ))

    # ========================================================================
    # PHASE 2: Unit Tests - Experiment 1
    # ========================================================================
    print("\n\n")
    print("#" * 70)
    print("# PHASE 2: UNIT TESTS - EXPERIMENT 1")
    print("#" * 70)

    exp1_tests = [
        'test_hamiltonian.py',
        'test_hns_precision.py',
        'test_laplacian.py',
        'test_einstein_residual.py',
        'test_stormer_verlet.py',
        'test_phase_detection.py',
        'test_reproducibility.py',
    ]

    for test_file in exp1_tests:
        results.append(run_command(
            [sys.executable, 'run_all_tests.py', '--test', test_file, '-v'],
            f'Experiment 1 - {test_file}',
            timeout=600
        ))

    # ========================================================================
    # PHASE 3: Unit Tests - Experiment 2
    # ========================================================================
    print("\n\n")
    print("#" * 70)
    print("# PHASE 3: UNIT TESTS - EXPERIMENT 2")
    print("#" * 70)

    exp2_tests = [
        'test_consciousness_metrics.py',
        'test_phase_transition.py',
        'test_neural_dynamics.py',
    ]

    for test_file in exp2_tests:
        results.append(run_command(
            [sys.executable, 'run_all_tests.py', '--test', test_file, '-v'],
            f'Experiment 2 - {test_file}',
            timeout=600
        ))

    # ========================================================================
    # PHASE 4: Benchmarks - Experiment 1
    # ========================================================================
    print("\n\n")
    print("#" * 70)
    print("# PHASE 4: BENCHMARKS - EXPERIMENT 1")
    print("#" * 70)

    results.append(run_command(
        [sys.executable, 'benchmarks/benchmark_accuracy.py',
         '--epochs', '1000', '--grid-size', '128'],
        'Experiment 1 - Accuracy Benchmark',
        timeout=7200
    ))

    results.append(run_command(
        [sys.executable, 'benchmarks/benchmark_performance.py',
         '--iterations', '100'],
        'Experiment 1 - Performance Benchmark',
        timeout=1800
    ))

    results.append(run_command(
        [sys.executable, 'benchmarks/benchmark_scaling.py'],
        'Experiment 1 - Scaling Benchmark',
        timeout=1800
    ))

    # ========================================================================
    # PHASE 5: Benchmarks - Experiment 2
    # ========================================================================
    print("\n\n")
    print("#" * 70)
    print("# PHASE 5: BENCHMARKS - EXPERIMENT 2")
    print("#" * 70)

    results.append(run_command(
        [sys.executable, 'benchmarks/benchmark_consciousness_emergence.py',
         '--epochs', '10000', '--network-size', '512'],
        'Experiment 2 - Consciousness Emergence Benchmark',
        timeout=7200
    ))

    # ========================================================================
    # PHASE 6: Audits - Experiment 1
    # ========================================================================
    print("\n\n")
    print("#" * 70)
    print("# PHASE 6: AUDITS - EXPERIMENT 1")
    print("#" * 70)

    results.append(run_command(
        [sys.executable, 'audits/audit_energy_discrepancy.py',
         '--epochs', '500', '--grid-size', '64'],
        'Experiment 1 - Energy Discrepancy Audit',
        timeout=3600
    ))

    results.append(run_command(
        [sys.executable, 'audits/audit_approach_a_physics.py'],
        'Experiment 1 - Physics Validation (Approach A)',
        timeout=3600
    ))

    results.append(run_command(
        [sys.executable, 'audits/audit_approach_b_numerical.py'],
        'Experiment 1 - Numerical Validation (Approach B)',
        timeout=3600
    ))

    results.append(run_command(
        [sys.executable, 'audits/audit_comparator.py'],
        'Experiment 1 - Audit Comparator',
        timeout=300
    ))

    # ========================================================================
    # PHASE 7: Audits - Experiment 2
    # ========================================================================
    print("\n\n")
    print("#" * 70)
    print("# PHASE 7: AUDITS - EXPERIMENT 2")
    print("#" * 70)

    results.append(run_command(
        [sys.executable, 'audits/audit_experiment2_neuroscience.py'],
        'Experiment 2 - Neuroscience Validation (Approach A)',
        timeout=1800
    ))

    # ========================================================================
    # PHASE 8: Final Report Generation
    # ========================================================================
    print("\n\n")
    print("#" * 70)
    print("# PHASE 8: FINAL REPORT GENERATION")
    print("#" * 70)

    results.append(run_command(
        [sys.executable, 'generate_final_report.py'],
        'Generate Final Comprehensive Report',
        timeout=300
    ))

    # ========================================================================
    # Summary
    # ========================================================================
    elapsed_total = time.time() - start_time_total

    print("\n\n")
    print("="*70)
    print("COMPLETE SYSTEM VALIDATION - FINAL SUMMARY")
    print("="*70)

    successful = sum(1 for r in results if r['success'])
    failed = len(results) - successful

    print(f"\nTotal Tasks: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Success Rate: {successful/len(results)*100:.1f}%")
    print(f"Total Time: {elapsed_total/3600:.2f} hours")

    print("\n" + "-"*70)
    print("Task Results:")
    print("-"*70)

    for i, r in enumerate(results, 1):
        status = "✓ PASS" if r['success'] else "✗ FAIL"
        print(f"{i:2d}. [{status}] {r['description']:50s} ({r['elapsed_time']:6.1f}s)")

    print("="*70)

    if failed == 0:
        print("\n✓✓✓ ALL VALIDATION TASKS PASSED ✓✓✓")
        print("\nBoth Experiment 1 and Experiment 2 are scientifically valid!")
        exit_code = 0
    else:
        print(f"\n✗✗✗ {failed} TASK(S) FAILED ✗✗✗")
        print("\nPlease review the failed tasks above.")
        exit_code = 1

    print("="*70)

    # Save summary
    summary = {
        'total_tasks': len(results),
        'successful': successful,
        'failed': failed,
        'success_rate': successful / len(results),
        'total_time_seconds': elapsed_total,
        'total_time_hours': elapsed_total / 3600,
        'tasks': results,
    }

    with open('validation_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nValidation summary saved to: validation_summary.json")

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
