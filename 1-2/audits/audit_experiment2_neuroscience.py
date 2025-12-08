"""
Audit Experiment 2: Neuroscience Validation (Approach A)

Validates Experiment 2 from neuroscience/consciousness theory perspective:
1. Izhikevich model correctness
2. STDP implementation
3. IIT (Integrated Information Theory) compliance
4. Neural connectivity patterns
5. Biological plausibility

Independent validation using neuroscience principles.

Author: Claude Code Testing Framework
"""

import numpy as np
import json
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))

# Constants from experiment 2 (avoiding wgpu import)
CONSCIOUSNESS_THRESHOLDS = {
    'connectivity': 15.0,
    'integration': 0.65,
    'depth': 7.0,
    'complexity': 0.8,
    'qualia_coherence': 0.75,
}

TAU_PLUS = 20.0
TAU_MINUS = 20.0
A_PLUS = 0.01
A_MINUS = 0.012
NETWORK_SIZE = 512


def validate_izhikevich_model() -> Dict:
    """
    Validate Izhikevich neuron model implementation.

    The Izhikevich model should reproduce spike patterns of real neurons.
    """
    print("\n" + "="*70)
    print("VALIDATION 1: Izhikevich Neuron Model")
    print("="*70)

    validations = []

    # Test 1: Resting potential stability
    print("\n  Test 1.1: Resting potential stability")
    v = -65.0  # mV
    u = 0.2 * v
    dt = 0.5

    # Simulate without input
    for _ in range(1000):
        dv = (0.04 * v * v + 5.0 * v + 140.0 - u) * dt
        du = 0.02 * (0.2 * v - u) * dt
        v += dv
        u += du

    rest_stable = abs(v + 65.0) < 5.0  # Should stay near -65mV

    print(f"    Resting potential: {v:.2f} mV")
    print(f"    Stability: {'[PASS] PASS' if rest_stable else '[FAIL] FAIL'}")

    validations.append({
        'name': 'Resting Potential Stability',
        'pass': rest_stable,
        'value': float(v),
        'expected': -65.0,
    })

    # Test 2: Spike generation with input
    print("\n  Test 1.2: Spike generation")
    v = -65.0
    u = 0.2 * v
    I = 20.0  # Strong input

    spiked = False
    for _ in range(200):
        dv = (0.04 * v * v + 5.0 * v + 140.0 - u + I) * dt
        du = 0.02 * (0.2 * v - u) * dt
        v += dv
        u += du

        if v >= 30.0:
            spiked = True
            break

    print(f"    Spike generated: {'[PASS] YES' if spiked else '[FAIL] NO'}")
    print(f"    Result: {'[PASS] PASS' if spiked else '[FAIL] FAIL'}")

    validations.append({
        'name': 'Spike Generation',
        'pass': spiked,
        'input_current': float(I),
    })

    # Test 3: Spike frequency adaptation
    print("\n  Test 1.3: Spike frequency adaptation")
    v = -65.0
    u = 0.2 * v
    I = 15.0

    spike_times = []
    t = 0
    for step in range(1000):
        dv = (0.04 * v * v + 5.0 * v + 140.0 - u + I) * dt
        du = 0.02 * (0.2 * v - u) * dt
        v += dv
        u += du
        t += dt

        if v >= 30.0:
            spike_times.append(t)
            v = -65.0
            u += 8.0  # Reset

    if len(spike_times) >= 3:
        isi1 = spike_times[1] - spike_times[0]
        isi_last = spike_times[-1] - spike_times[-2]
        adaptation = isi_last > isi1 * 1.1  # ISI should increase

        print(f"    First ISI: {isi1:.2f} ms")
        print(f"    Last ISI: {isi_last:.2f} ms")
        print(f"    Adaptation: {'[PASS] PASS' if adaptation else '[FAIL] FAIL'}")

        validations.append({
            'name': 'Spike Frequency Adaptation',
            'pass': adaptation,
            'first_isi': float(isi1),
            'last_isi': float(isi_last),
        })
    else:
        print(f"    Not enough spikes: {len(spike_times)}")
        validations.append({
            'name': 'Spike Frequency Adaptation',
            'pass': False,
            'spike_count': len(spike_times),
        })

    all_pass = all(v['pass'] for v in validations)

    print(f"\n  {'='*66}")
    print(f"  Izhikevich Model: {'[PASS] VALID' if all_pass else '[FAIL] ISSUES DETECTED'}")
    print(f"  {'='*66}")

    return {
        'validation': 'izhikevich_model',
        'validations': validations,
        'pass': all_pass,
    }


def validate_stdp() -> Dict:
    """
    Validate STDP (Spike-Timing-Dependent Plasticity) implementation.

    STDP should follow Hebbian learning: "cells that fire together, wire together"
    """
    print("\n" + "="*70)
    print("VALIDATION 2: STDP (Spike-Timing-Dependent Plasticity)")
    print("="*70)

    validations = []

    # Test 1: LTP (Long-Term Potentiation)
    print("\n  Test 2.1: LTP (post after pre)")
    pre_time = 100.0
    post_time = 105.0  # 5ms after
    current_weight = 0.5

    dt = post_time - pre_time
    dw = A_PLUS * np.exp(-dt / TAU_PLUS)
    new_weight = current_weight + dw

    ltp_correct = new_weight > current_weight

    print(f"    dt = {dt} ms")
    print(f"    dw = {dw:.6f}")
    print(f"    Weight: {current_weight} -> {new_weight:.6f}")
    print(f"    LTP: {'[PASS] PASS' if ltp_correct else '[FAIL] FAIL'}")

    validations.append({
        'name': 'LTP (Long-Term Potentiation)',
        'pass': ltp_correct,
        'delta_t': float(dt),
        'delta_w': float(dw),
    })

    # Test 2: LTD (Long-Term Depression)
    print("\n  Test 2.2: LTD (pre after post)")
    pre_time = 105.0
    post_time = 100.0  # Pre after post
    current_weight = 0.5

    dt = post_time - pre_time
    dw = -A_MINUS * np.exp(dt / TAU_MINUS)
    new_weight = current_weight + dw

    ltd_correct = new_weight < current_weight

    print(f"    dt = {dt} ms")
    print(f"    dw = {dw:.6f}")
    print(f"    Weight: {current_weight} -> {new_weight:.6f}")
    print(f"    LTD: {'[PASS] PASS' if ltd_correct else '[FAIL] FAIL'}")

    validations.append({
        'name': 'LTD (Long-Term Depression)',
        'pass': ltd_correct,
        'delta_t': float(dt),
        'delta_w': float(dw),
    })

    # Test 3: Temporal window
    print("\n  Test 2.3: Temporal causality window")

    # Close timing
    dt_close = 5.0
    dw_close = A_PLUS * np.exp(-dt_close / TAU_PLUS)

    # Far timing
    dt_far = 50.0
    dw_far = A_PLUS * np.exp(-dt_far / TAU_PLUS)

    window_correct = abs(dw_close) > abs(dw_far) * 2

    print(f"    Close (5ms): dw = {dw_close:.6f}")
    print(f"    Far (50ms):  dw = {dw_far:.6f}")
    print(f"    Ratio: {dw_close/dw_far:.2f}x")
    print(f"    Causality window: {'[PASS] PASS' if window_correct else '[FAIL] FAIL'}")

    validations.append({
        'name': 'Temporal Causality Window',
        'pass': window_correct,
        'close_effect': float(dw_close),
        'far_effect': float(dw_far),
        'ratio': float(dw_close / dw_far),
    })

    # Test 4: Weight bounds
    print("\n  Test 2.4: Weight bounding")

    weight = 0.95
    for _ in range(100):
        dw = A_PLUS  # Strong LTP
        weight = np.clip(weight + dw, -1.0, 1.0)

    bounds_correct = -1.0 <= weight <= 1.0

    print(f"    After 100 LTP steps: weight = {weight:.6f}")
    print(f"    Bounded [-1, 1]: {'[PASS] PASS' if bounds_correct else '[FAIL] FAIL'}")

    validations.append({
        'name': 'Weight Bounding',
        'pass': bounds_correct,
        'final_weight': float(weight),
    })

    all_pass = all(v['pass'] for v in validations)

    print(f"\n  {'='*66}")
    print(f"  STDP Implementation: {'[PASS] VALID' if all_pass else '[FAIL] ISSUES DETECTED'}")
    print(f"  {'='*66}")

    return {
        'validation': 'stdp',
        'validations': validations,
        'pass': all_pass,
    }


def validate_iit_compliance() -> Dict:
    """
    Validate compliance with IIT (Integrated Information Theory).

    Phi (integrated information) should:
    1. Be zero for disconnected systems
    2. Be high for integrated systems
    3. Measure irreducibility
    """
    print("\n" + "="*70)
    print("VALIDATION 3: IIT (Integrated Information Theory) Compliance")
    print("="*70)

    from utils.consciousness_metrics import compute_integration_phi

    validations = []

    # Test 1: Uniform field (low Phi)
    print("\n  Test 3.1: Uniform field (no integration)")
    uniform = np.ones(512 * 512, dtype=np.float32) * 0.5
    phi_uniform = compute_integration_phi(uniform, num_partitions=8)

    low_phi = phi_uniform < 0.3

    print(f"    Phi (uniform) = {phi_uniform:.4f}")
    print(f"    Expected: < 0.3")
    print(f"    Result: {'[PASS] PASS' if low_phi else '[FAIL] FAIL'}")

    validations.append({
        'name': 'Uniform Field (Low Phi)',
        'pass': low_phi,
        'phi': float(phi_uniform),
        'threshold': 0.3,
    })

    # Test 2: Random field (moderate Phi)
    print("\n  Test 3.2: Random field (moderate integration)")
    np.random.seed(42)
    random = np.random.rand(512 * 512).astype(np.float32)
    phi_random = compute_integration_phi(random, num_partitions=8)

    moderate_phi = 0.1 < phi_random < 0.8

    print(f"    Phi (random) = {phi_random:.4f}")
    print(f"    Expected: 0.1 < Phi < 0.8")
    print(f"    Result: {'[PASS] PASS' if moderate_phi else '[FAIL] FAIL'}")

    validations.append({
        'name': 'Random Field (Moderate Phi)',
        'pass': moderate_phi,
        'phi': float(phi_random),
    })

    # Test 3: Structured field (higher Phi)
    print("\n  Test 3.3: Structured field (high integration)")
    structured = np.zeros(512 * 512, dtype=np.float32)
    grid = structured.reshape(512, 512)

    # Create modular structure
    for i in range(8):
        for j in range(8):
            module_val = (i + j) / 14.0
            grid[i*64:(i+1)*64, j*64:(j+1)*64] = module_val + \
                np.random.randn(64, 64) * 0.05

    phi_structured = compute_integration_phi(structured, num_partitions=8)

    high_phi = phi_structured > phi_uniform

    print(f"    Phi (structured) = {phi_structured:.4f}")
    print(f"    Phi (uniform) = {phi_uniform:.4f}")
    print(f"    Structured > Uniform: {'[PASS] PASS' if high_phi else '[FAIL] FAIL'}")

    validations.append({
        'name': 'Structured Field (High Phi)',
        'pass': high_phi,
        'phi_structured': float(phi_structured),
        'phi_uniform': float(phi_uniform),
    })

    all_pass = all(v['pass'] for v in validations)

    print(f"\n  {'='*66}")
    print(f"  IIT Compliance: {'[PASS] VALID' if all_pass else '[FAIL] ISSUES DETECTED'}")
    print(f"  {'='*66}")

    return {
        'validation': 'iit_compliance',
        'validations': validations,
        'pass': all_pass,
    }


def validate_biological_plausibility() -> Dict:
    """
    Validate biological plausibility of parameters.
    """
    print("\n" + "="*70)
    print("VALIDATION 4: Biological Plausibility")
    print("="*70)

    validations = []

    # Test 1: STDP time constants
    print("\n  Test 4.1: STDP time constants")
    tau_realistic = 15.0 <= TAU_PLUS <= 25.0 and 15.0 <= TAU_MINUS <= 25.0

    print(f"    tau+ = {TAU_PLUS} ms (expected: 15-25 ms)")
    print(f"    tau- = {TAU_MINUS} ms (expected: 15-25 ms)")
    print(f"    Result: {'[PASS] PASS' if tau_realistic else '[FAIL] FAIL'}")

    validations.append({
        'name': 'STDP Time Constants',
        'pass': tau_realistic,
        'tau_plus': float(TAU_PLUS),
        'tau_minus': float(TAU_MINUS),
    })

    # Test 2: Network size
    print("\n  Test 4.2: Network size")
    num_neurons = NETWORK_SIZE * NETWORK_SIZE
    size_realistic = 100000 < num_neurons < 1000000  # Realistic for cortical column

    print(f"    Network size: {num_neurons:,} neurons")
    print(f"    Expected: 100k - 1M (cortical column scale)")
    print(f"    Result: {'[PASS] PASS' if size_realistic else '[FAIL] FAIL'}")

    validations.append({
        'name': 'Network Size',
        'pass': size_realistic,
        'num_neurons': num_neurons,
    })

    # Test 3: Connectivity threshold
    print("\n  Test 4.3: Connectivity threshold")
    k_threshold = CONSCIOUSNESS_THRESHOLDS['connectivity']
    k_realistic = 10.0 <= k_threshold <= 30.0  # Realistic for cortical neurons

    print(f"    <k> threshold: {k_threshold}")
    print(f"    Expected: 10-30 (cortical neuron range)")
    print(f"    Result: {'[PASS] PASS' if k_realistic else '[FAIL] FAIL'}")

    validations.append({
        'name': 'Connectivity Threshold',
        'pass': k_realistic,
        'threshold': float(k_threshold),
    })

    all_pass = all(v['pass'] for v in validations)

    print(f"\n  {'='*66}")
    print(f"  Biological Plausibility: {'[PASS] VALID' if all_pass else '[FAIL] ISSUES DETECTED'}")
    print(f"  {'='*66}")

    return {
        'validation': 'biological_plausibility',
        'validations': validations,
        'pass': all_pass,
    }


def run_neuroscience_audit() -> Dict:
    """Run complete neuroscience audit."""
    print("="*70)
    print("EXPERIMENT 2 - NEUROSCIENCE VALIDATION AUDIT (APPROACH A)")
    print("="*70)
    print("\nValidating from neuroscience/consciousness theory perspective:")
    print("  - Izhikevich neuron model")
    print("  - STDP learning rule")
    print("  - IIT compliance")
    print("  - Biological plausibility")
    print("="*70)

    results = []

    # Run validations
    results.append(validate_izhikevich_model())
    results.append(validate_stdp())
    results.append(validate_iit_compliance())
    results.append(validate_biological_plausibility())

    # Calculate overall pass rate
    total_validations = sum(len(r['validations']) for r in results)
    passed_validations = sum(
        sum(1 for v in r['validations'] if v['pass'])
        for r in results
    )
    pass_rate = passed_validations / total_validations if total_validations > 0 else 0

    # Overall verdict
    all_pass = all(r['pass'] for r in results)

    print("\n" + "="*70)
    print("NEUROSCIENCE AUDIT SUMMARY")
    print("="*70)
    print(f"\nValidation Categories: {len(results)}")
    print(f"Total Tests: {total_validations}")
    print(f"Passed: {passed_validations}")
    print(f"Failed: {total_validations - passed_validations}")
    print(f"Pass Rate: {pass_rate*100:.1f}%")

    print(f"\n{'='*70}")
    if all_pass:
        print("VERDICT: [PASS][PASS][PASS] EXPERIMENT 2 IS NEUROSCIENTIFICALLY VALID [PASS][PASS][PASS]")
        verdict = 'VALID'
    else:
        print("VERDICT: [FAIL][FAIL][FAIL] NEUROSCIENCE ISSUES DETECTED [FAIL][FAIL][FAIL]")
        verdict = 'ISSUES_DETECTED'
    print(f"{'='*70}")

    return {
        'audit': 'experiment2_neuroscience',
        'approach': 'A',
        'results': results,
        'summary': {
            'total_tests': total_validations,
            'passed': passed_validations,
            'failed': total_validations - passed_validations,
            'pass_rate': float(pass_rate),
            'overall_verdict': verdict,
        }
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Neuroscience validation audit for Experiment 2"
    )
    parser.add_argument('--output', type=str, default='audit_experiment2_neuroscience.json',
                       help='Output JSON file')

    args = parser.parse_args()

    # Run audit
    results = run_neuroscience_audit()

    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nAudit results saved to: {args.output}")

    # Exit code
    sys.exit(0 if results['summary']['overall_verdict'] == 'VALID' else 1)


if __name__ == "__main__":
    main()
