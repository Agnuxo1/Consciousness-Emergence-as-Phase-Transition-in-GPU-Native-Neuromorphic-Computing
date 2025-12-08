#!/usr/bin/env python3
"""
Master Test Runner for Experiment 1

Executes all unit and integration tests using pytest.

Usage:
    python run_all_tests.py                    # Run all tests
    python run_all_tests.py --verbose          # Verbose output
    python run_all_tests.py --coverage         # With coverage report
"""

import subprocess
import sys
from pathlib import Path


def run_tests(verbose=False, coverage=False, specific_test=None):
    """
    Run pytest with specified options.

    Args:
        verbose: Enable verbose output
        coverage: Enable coverage reporting
        specific_test: Run specific test file (e.g., 'test_hamiltonian.py')

    Returns:
        Exit code from pytest
    """
    print("=" * 80)
    print("EXPERIMENT 1 - TEST SUITE")
    print("=" * 80)
    print()

    # Build pytest command
    cmd = ["pytest"]

    if verbose:
        cmd.append("-v")
    else:
        cmd.append("-q")

    if coverage:
        cmd.extend(["--cov=.", "--cov-report=html", "--cov-report=term"])

    # Add test directory or specific test
    if specific_test:
        cmd.append(f"tests/{specific_test}")
    else:
        cmd.append("tests/")

    # Additional pytest options
    cmd.extend([
        "--tb=short",  # Short traceback format
        "-ra",  # Show summary of all test outcomes
    ])

    print(f"Running command: {' '.join(cmd)}")
    print()

    # Run pytest
    result = subprocess.run(cmd, cwd=Path(__file__).parent)

    print()
    print("=" * 80)
    if result.returncode == 0:
        print("✓ ALL TESTS PASSED")
    else:
        print("✗ SOME TESTS FAILED")
    print("=" * 80)

    return result.returncode


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Run all tests for Experiment 1",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_all_tests.py                           # Run all tests
  python run_all_tests.py --verbose                 # Verbose output
  python run_all_tests.py --coverage                # With coverage
  python run_all_tests.py --test test_hamiltonian.py  # Specific test
"""
    )

    parser.add_argument("-v", "--verbose", action="store_true",
                       help="Verbose output")
    parser.add_argument("-c", "--coverage", action="store_true",
                       help="Generate coverage report")
    parser.add_argument("-t", "--test", type=str,
                       help="Run specific test file")

    args = parser.parse_args()

    exit_code = run_tests(
        verbose=args.verbose,
        coverage=args.coverage,
        specific_test=args.test
    )

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
