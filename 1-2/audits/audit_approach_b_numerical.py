"""
Audit Approach B: Numerical/Computational Validation

Validates numerical correctness:
1. HNS precision measurement
2. Fractal dimension algorithm
3. Reproducibility
4. Numerical stability
"""

import numpy as np
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.cpu_reference import (
    hns_encode,
    hns_decode,
    hns_add,
    compute_fractal_dimension_boxcounting,
    cpu_simulation_step
)


def audit_numerical_validation():
    """
    Perform numerical/computational validation.

    Returns:
        Audit report dictionary
    """
    print("=" * 80)
    print("AUDIT APPROACH B: NUMERICAL/COMPUTATIONAL VALIDATION")
    print("=" * 80)
    print()

    report = {
        'audit_type': 'numerical_computational',
        'approach': 'B',
        'validations': []
    }

    # ========================================================================
    # VALIDATION 1: HNS Precision
    # ========================================================================
    print("VALIDATION 1: HNS Precision System")
    print("-" * 80)

    # Test accumulation
    n_ops = 10000
    small_value = 0.0001

    # Float32 accumulation
    sum_float32 = np.float32(0.0)
    for _ in range(n_ops):
        sum_float32 += np.float32(small_value)

    expected = n_ops * small_value
    error_float32 = abs(sum_float32 - expected)

    # HNS accumulation
    sum_hns = hns_encode(0.0)
    small_hns = hns_encode(small_value)
    for _ in range(n_ops):
        sum_hns = hns_add(sum_hns, small_hns)

    decoded = hns_decode(sum_hns)
    error_hns = abs(decoded - expected)

    improvement = error_float32 / (error_hns + 1e-20)

    validation_1 = {
        'name': 'HNS Precision',
        'operations': n_ops,
        'float32_error': float(error_float32),
        'hns_error': float(error_hns),
        'improvement_factor': float(improvement),
        'hns_threshold': 1e-10,
        'pass': error_hns < 1e-10 and error_hns < error_float32
    }

    report['validations'].append(validation_1)

    print(f"  Operations: {n_ops}")
    print(f"  Float32 error: {error_float32:.10e}")
    print(f"  HNS error: {error_hns:.10e}")
    print(f"  Improvement: {improvement:.1f}x")
    print(f"  Status: {'✓ PASS' if validation_1['pass'] else '✗ FAIL'}")
    print()

    # ========================================================================
    # VALIDATION 2: Fractal Dimension Algorithm
    # ========================================================================
    print("VALIDATION 2: Fractal Dimension Algorithm")
    print("-" * 80)

    # Test on 2D uniform field (should give D ≈ 2)
    grid_size = 256
    np.random.seed(42)
    field_2d = np.random.rand(grid_size, grid_size).astype(np.float32)

    dimension = compute_fractal_dimension_boxcounting(field_2d, threshold=0.5)

    validation_2 = {
        'name': 'Fractal Dimension Algorithm',
        'test_field': '2D random',
        'computed_dimension': float(dimension),
        'expected_dimension': 2.0,
        'tolerance': 0.2,
        'pass': abs(dimension - 2.0) < 0.2
    }

    report['validations'].append(validation_2)

    print(f"  Test field: 2D random")
    print(f"  Computed D: {dimension:.3f}")
    print(f"  Expected D: 2.0 ± 0.2")
    print(f"  Status: {'✓ PASS' if validation_2['pass'] else '✗ FAIL'}")
    print()

    # ========================================================================
    # VALIDATION 3: Reproducibility
    # ========================================================================
    print("VALIDATION 3: Reproducibility")
    print("-" * 80)

    grid_size = 64

    # Run 1
    np.random.seed(42)
    state1 = np.zeros((grid_size, grid_size, 4), dtype=np.float32)
    state1[:, :, 0] = np.random.randn(grid_size, grid_size) * 0.1
    weights1 = np.random.randn(grid_size * grid_size, 25).astype(np.float32) * 0.1

    for epoch in range(10):
        state1 = cpu_simulation_step(state1, weights1, epoch)

    # Run 2 (same seed)
    np.random.seed(42)
    state2 = np.zeros((grid_size, grid_size, 4), dtype=np.float32)
    state2[:, :, 0] = np.random.randn(grid_size, grid_size) * 0.1
    weights2 = np.random.randn(grid_size * grid_size, 25).astype(np.float32) * 0.1

    for epoch in range(10):
        state2 = cpu_simulation_step(state2, weights2, epoch)

    # Compare
    max_diff = np.max(np.abs(state1 - state2))

    validation_3 = {
        'name': 'Reproducibility',
        'seed': 42,
        'steps': 10,
        'max_difference': float(max_diff),
        'threshold': 1e-10,
        'pass': max_diff < 1e-10
    }

    report['validations'].append(validation_3)

    print(f"  Seed: 42")
    print(f"  Steps: 10")
    print(f"  Max difference: {max_diff:.10e}")
    print(f"  Status: {'✓ PASS' if validation_3['pass'] else '✗ FAIL'}")
    print()

    # ========================================================================
    # VALIDATION 4: Numerical Stability
    # ========================================================================
    print("VALIDATION 4: Numerical Stability")
    print("-" * 80)

    grid_size = 128
    np.random.seed(42)
    state = np.zeros((grid_size, grid_size, 4), dtype=np.float32)
    state[:, :, 0] = np.random.randn(grid_size, grid_size) * 0.1
    state[:, :, 1] = np.random.randn(grid_size, grid_size) * 0.05
    weights = np.random.randn(grid_size * grid_size, 25).astype(np.float32) * 0.1

    max_values = []
    for epoch in range(100):
        state = cpu_simulation_step(state, weights, epoch)
        max_val = np.max(np.abs(state))
        max_values.append(max_val)

        if not np.all(np.isfinite(state)):
            break

    all_finite = np.all([np.isfinite(v) for v in max_values])
    max_value = max(max_values) if max_values else float('inf')
    bounded = max_value < 100.0

    validation_4 = {
        'name': 'Numerical Stability',
        'steps': 100,
        'all_finite': bool(all_finite),
        'max_value': float(max_value),
        'bounded_threshold': 100.0,
        'pass': all_finite and bounded
    }

    report['validations'].append(validation_4)

    print(f"  Steps: 100")
    print(f"  All finite: {all_finite}")
    print(f"  Max value: {max_value:.2f}")
    print(f"  Status: {'✓ PASS' if validation_4['pass'] else '✗ FAIL'}")
    print()

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("=" * 80)
    print("AUDIT SUMMARY - APPROACH B")
    print("=" * 80)

    passed = sum(1 for v in report['validations'] if v.get('pass', False))
    total = len(report['validations'])

    report['summary'] = {
        'validations_passed': passed,
        'validations_total': total,
        'pass_rate': passed / total if total > 0 else 0,
        'overall_verdict': 'VALID' if passed == total else 'PARTIAL'
    }

    print(f"Validations passed: {passed}/{total}")
    print(f"Pass rate: {report['summary']['pass_rate']*100:.0f}%")
    print(f"Overall verdict: {report['summary']['overall_verdict']}")
    print("=" * 80)

    return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run numerical validation audit")
    parser.add_argument("--output", type=str, default="audit_numerical_validation.json",
                       help="Output JSON file")

    args = parser.parse_args()

    report = audit_numerical_validation()

    with open(args.output, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\nAudit report saved to: {args.output}")

    sys.exit(0 if report['summary']['overall_verdict'] == 'VALID' else 1)
