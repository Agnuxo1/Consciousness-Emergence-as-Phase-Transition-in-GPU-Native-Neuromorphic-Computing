#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fixed experiment runner that handles Unicode properly on Windows.
Executes all 6 experiments with proper encoding configuration.
"""
import os
import sys
import subprocess
import json
from pathlib import Path
from datetime import datetime, timezone

# Fix Windows console encoding FIRST
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    # Also set environment variable for subprocesses
    os.environ['PYTHONIOENCODING'] = 'utf-8'

BASE_DIR = Path(__file__).parent.parent
RESULTS_DIR = BASE_DIR / "release" / "benchmarks"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

EXPERIMENTS = {
    1: {
        'path': '1-2/experiment1_spacetime_emergence.py',
        'name': 'Spacetime Emergence',
        'timeout': 300
    },
    2: {
        'path': '1-2/experiment2_consciousness_emergence.py',
        'name': 'Consciousness Emergence',
        'timeout': 300
    },
    3: {
        'path': '3-4/Experiment_Genesis_1.py',
        'name': 'Genesis 1',
        'timeout': 120
    },
    4: {
        'path': '3-4/Experiment_Genesis_2.py',
        'name': 'Genesis 2',
        'timeout': 120
    },
    5: {
        'path': '5-6/benchmark_experiment_1.py',
        'name': 'Benchmark 1',
        'timeout': 120
    },
    6: {
        'path': '5-6/benchmark_experiment_2.py',
        'name': 'Benchmark 2',
        'timeout': 120
    }
}

def run_experiment(exp_num):
    """Run single experiment with proper encoding."""
    exp = EXPERIMENTS[exp_num]
    exp_path = BASE_DIR / exp['path']

    print(f"\n{'='*80}")
    print(f"RUNNING EXPERIMENT {exp_num}: {exp['name']}")
    print(f"{'='*80}")
    print(f"Script: {exp_path}")

    if not exp_path.exists():
        print(f"ERROR: Script not found!")
        return None

    # Set up environment with UTF-8
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    env['PYTHONUTF8'] = '1'

    # Run with timeout
    try:
        result = subprocess.run(
            [sys.executable, '-u', str(exp_path)],
            capture_output=True,
            text=True,
            timeout=exp['timeout'],
            cwd=exp_path.parent,
            env=env,
            encoding='utf-8',
            errors='replace'
        )

        # Save result
        timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        output = {
            'experiment': exp_num,
            'name': exp['name'],
            'timestamp': timestamp,
            'returncode': result.returncode,
            'stdout': result.stdout[-2000:] if result.stdout else "",
            'stderr': result.stderr[-2000:] if result.stderr else "",
            'summary': extract_metrics(result.stdout)
        }

        result_file = RESULTS_DIR / f"results-exp-{exp_num}-{timestamp}.json"
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        print(f"\n{'='*80}")
        if result.returncode == 0:
            print(f"SUCCESS: Experiment {exp_num} completed")
        else:
            print(f"FAILED: Experiment {exp_num} returned code {result.returncode}")
        print(f"Result saved to: {result_file}")
        print(f"{'='*80}")

        return output

    except subprocess.TimeoutExpired:
        print(f"TIMEOUT: Experiment {exp_num} exceeded {exp['timeout']}s")
        return None
    except Exception as e:
        print(f"ERROR: {e}")
        return None

def extract_metrics(stdout):
    """Extract metrics from stdout if possible."""
    if not stdout:
        return None

    metrics = {}

    # Look for magnetization
    if 'Magnetization' in stdout:
        for line in stdout.split('\n'):
            if 'Magnetization' in line:
                try:
                    parts = line.split('Magnetization=')[1].split(',')[0]
                    metrics['magnetization'] = float(parts)
                except:
                    pass

    # Look for accuracy
    if 'Accuracy' in stdout or 'Acc:' in stdout:
        for line in stdout.split('\n'):
            if 'Acc:' in line:
                try:
                    parts = line.split('Acc:')[1].split()[0]
                    metrics['accuracy'] = float(parts)
                except:
                    pass

    # Look for duration
    if 'Duration' in stdout:
        for line in stdout.split('\n'):
            if 'Duration' in line:
                try:
                    parts = line.split('Duration:')[1].split('s')[0]
                    metrics['duration_s'] = float(parts)
                except:
                    pass

    return metrics if metrics else None

def main():
    print("\n")
    print("="*80)
    print(" "*20 + "NEUROCHIMERA EXPERIMENT RUNNER")
    print(" "*25 + "All 6 Experiments")
    print("="*80)

    results = {}
    success_count = 0

    for exp_num in range(1, 7):
        result = run_experiment(exp_num)
        if result:
            results[exp_num] = result
            if result['returncode'] == 0:
                success_count += 1

    # Summary
    print("\n\n")
    print("="*80)
    print(" "*30 + "FINAL SUMMARY")
    print("="*80)
    print(f"\nTotal Experiments: 6")
    print(f"Successful: {success_count}/6 ({success_count/6*100:.1f}%)")
    print(f"Failed: {6-success_count}/6")

    print("\n" + "-"*80)
    print(f"{'Exp':<5} {'Name':<30} {'Status':<15} {'Metrics':<30}")
    print("-"*80)

    for exp_num in range(1, 7):
        exp = EXPERIMENTS[exp_num]
        if exp_num in results:
            r = results[exp_num]
            status = "SUCCESS" if r['returncode'] == 0 else "FAILED"
            metrics = r.get('summary', {}) or {}
            metrics_str = ', '.join([f"{k}={v:.4f}" for k, v in metrics.items()]) if metrics else "N/A"
        else:
            status = "NOT RUN"
            metrics_str = "N/A"

        print(f"{exp_num:<5} {exp['name']:<30} {status:<15} {metrics_str:<30}")

    print("-"*80)
    print(f"\nResults saved in: {RESULTS_DIR}")
    print("="*80)

    return 0 if success_count == 6 else 1

if __name__ == '__main__':
    sys.exit(main())
