#!/usr/bin/env python3
"""
System Verification Script

Verifies that all components of the testing/benchmarking/audit system
are properly installed and ready to execute.
"""

import sys
from pathlib import Path


def check_file_exists(filepath, description):
    """Check if a file exists and report status."""
    exists = Path(filepath).exists()
    status = "[OK]" if exists else "[MISSING]"
    print(f"{status} {description:50s} {filepath}")
    return exists


def verify_system():
    """
    Verify all system components.

    Returns:
        True if all critical components exist, False otherwise
    """
    print("=" * 80)
    print("SYSTEM VERIFICATION - Experiments 1 & 2 Testing Framework")
    print("=" * 80)
    print()

    all_ok = True

    # ========================================================================
    # Core Infrastructure
    # ========================================================================
    print("CORE INFRASTRUCTURE")
    print("-" * 80)

    files = [
        ("utils/cpu_reference.py", "CPU Reference Implementation"),
        ("utils/__init__.py", "Utils package init"),
        ("tests/conftest.py", "Pytest fixtures"),
        ("tests/__init__.py", "Tests package init"),
        ("benchmarks/__init__.py", "Benchmarks package init"),
        ("audits/__init__.py", "Audits package init"),
        ("reports/__init__.py", "Reports package init"),
        ("requirements_testing.txt", "Dependencies file"),
    ]

    for filepath, desc in files:
        if not check_file_exists(filepath, desc):
            all_ok = False

    print()

    # ========================================================================
    # Unit Tests - Experiment 1
    # ========================================================================
    print("UNIT TESTS - EXPERIMENT 1")
    print("-" * 80)

    tests_exp1 = [
        ("tests/test_hamiltonian.py", "Hamiltonian tests (CRITICAL)"),
        ("tests/test_hns_precision.py", "HNS precision tests"),
        ("tests/test_laplacian.py", "Laplacian tests"),
        ("tests/test_einstein_residual.py", "Einstein residual tests"),
        ("tests/test_stormer_verlet.py", "Störmer-Verlet tests"),
        ("tests/test_phase_detection.py", "Phase detection tests"),
        ("tests/test_reproducibility.py", "Reproducibility tests"),
    ]

    for filepath, desc in tests_exp1:
        if not check_file_exists(filepath, desc):
            all_ok = False

    print()

    # ========================================================================
    # Unit Tests - Experiment 2
    # ========================================================================
    print("UNIT TESTS - EXPERIMENT 2")
    print("-" * 80)

    tests_exp2 = [
        ("tests/test_consciousness_metrics.py", "Consciousness metrics tests (CRITICAL)"),
        ("tests/test_phase_transition.py", "Phase transition tests (CRITICAL)"),
        ("tests/test_neural_dynamics.py", "Neural dynamics tests"),
    ]

    for filepath, desc in tests_exp2:
        if not check_file_exists(filepath, desc):
            all_ok = False

    print()

    # ========================================================================
    # Benchmarks
    # ========================================================================
    print("BENCHMARKS")
    print("-" * 80)

    benchmarks = [
        ("benchmarks/benchmark_accuracy.py", "Exp1: Accuracy benchmark (CRITICAL)"),
        ("benchmarks/benchmark_performance.py", "Exp1: Performance benchmark"),
        ("benchmarks/benchmark_scaling.py", "Exp1: Scaling benchmark"),
        ("benchmarks/benchmark_consciousness_emergence.py", "Exp2: Consciousness benchmark (CRITICAL)"),
    ]

    for filepath, desc in benchmarks:
        if not check_file_exists(filepath, desc):
            all_ok = False

    print()

    # ========================================================================
    # Audits
    # ========================================================================
    print("AUDITS")
    print("-" * 80)

    audits = [
        ("audits/audit_energy_discrepancy.py", "Exp1: Energy discrepancy audit (CRITICAL)"),
        ("audits/audit_approach_a_physics.py", "Exp1: Physics validation audit"),
        ("audits/audit_approach_b_numerical.py", "Exp1: Numerical validation audit"),
        ("audits/audit_comparator.py", "Exp1: Audit comparator"),
        ("audits/audit_experiment2_neuroscience.py", "Exp2: Neuroscience validation audit"),
    ]

    for filepath, desc in audits:
        if not check_file_exists(filepath, desc):
            all_ok = False

    print()

    # ========================================================================
    # Master Runners
    # ========================================================================
    print("MASTER RUNNERS")
    print("-" * 80)

    runners = [
        ("run_all_tests.py", "Test runner"),
        ("run_all_benchmarks.py", "Benchmark runner"),
        ("run_dual_audit.py", "Audit runner"),
        ("generate_final_report.py", "Report generator"),
        ("run_complete_system.py", "Complete system runner (both experiments)"),
        ("RUN_COMPLETE_VALIDATION.bat", "Complete validation script (Windows)"),
    ]

    for filepath, desc in runners:
        if not check_file_exists(filepath, desc):
            all_ok = False

    print()

    # ========================================================================
    # Documentation
    # ========================================================================
    print("DOCUMENTATION")
    print("-" * 80)

    docs = [
        ("COMPREHENSIVE_TESTING_GUIDE.md", "Comprehensive testing guide (MAIN)"),
        ("TESTING_README.md", "Testing guide (Exp1)"),
        ("IMPLEMENTATION_SUMMARY.md", "Implementation summary"),
        ("COMPLETION_REPORT.md", "Completion report"),
    ]

    for filepath, desc in docs:
        if not check_file_exists(filepath, desc):
            all_ok = False

    print()

    # ========================================================================
    # Visualizations
    # ========================================================================
    print("VISUALIZATIONS")
    print("-" * 80)

    viz = [
        ("reports/visualizations/plot_energy_evolution.py", "Energy plot"),
        ("reports/visualizations/plot_all.py", "Master plot script"),
    ]

    for filepath, desc in viz:
        if not check_file_exists(filepath, desc):
            all_ok = False

    print()

    # ========================================================================
    # Python Imports
    # ========================================================================
    print("PYTHON IMPORTS")
    print("-" * 80)

    import_checks = [
        ("numpy", "NumPy"),
        ("pytest", "pytest"),
    ]

    for module, desc in import_checks:
        try:
            __import__(module)
            print(f"[OK] {desc:50s} installed")
        except ImportError:
            print(f"[MISSING] {desc:50s} NOT INSTALLED")
            all_ok = False

    print()

    # ========================================================================
    # Summary
    # ========================================================================
    print("=" * 80)
    if all_ok:
        print("[SUCCESS] SYSTEM VERIFICATION COMPLETE - ALL COMPONENTS READY")
    else:
        print("[FAILED] SYSTEM VERIFICATION FAILED - MISSING COMPONENTS")
    print("=" * 80)

    return all_ok


def main():
    """Main verification routine."""
    all_ok = verify_system()

    if all_ok:
        print()
        print("NEXT STEPS:")
        print("1. Run tests:      python run_all_tests.py --verbose")
        print("2. Run benchmarks: python run_all_benchmarks.py")
        print("3. Run audits:     python run_dual_audit.py")
        print("4. Generate report: python generate_final_report.py")
        print()
        print("OR run everything: RUN_COMPLETE_VALIDATION.bat")
        print()
        sys.exit(0)
    else:
        print()
        print("ERROR: System verification failed. Some components are missing.")
        print("Please check the output above for details.")
        print()
        sys.exit(1)


if __name__ == "__main__":
    main()
