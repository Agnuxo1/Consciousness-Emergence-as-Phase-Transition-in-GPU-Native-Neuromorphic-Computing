#!/usr/bin/env python3
"""Run one of the six experiments and collect results.

Usage:
  python publish/run_experiment.py --exp 1

This script maps experiment numbers to the corresponding script files in the
repository, runs them, captures stdout/stderr and any produced `benchmark_summary.json`,
and writes a result JSON to `release/benchmarks/`.

It will also optionally upload the result as a W&B artifact when `WANDB_API_KEY` is
available in the environment.
"""
import argparse
import json
import os
import subprocess
from datetime import datetime

EXP_MAP = {
    1: '1-2/experiment1_spacetime_emergence.py',
    2: '1-2/experiment2_consciousness_emergence.py',
    3: '3-4/Experiment_Genesis_1.py',
    4: '3-4/Experiment_Genesis_2.py',
    5: '5-6/benchmark_experiment_1.py',
    6: '5-6/benchmark_experiment_2.py',
}

RESULT_DIR = os.path.join('release', 'benchmarks')
os.makedirs(RESULT_DIR, exist_ok=True)


def run_script(script_path):
    if not os.path.exists(script_path):
        raise FileNotFoundError(f'Experiment script not found: {script_path}')
    proc = subprocess.run(['python', script_path], capture_output=True, text=True)
    return proc.returncode, proc.stdout, proc.stderr


def collect_summary(exp):
    # Common file produced by some benchmarks
    candidate = os.path.join('benchmarks', 'benchmark_summary.json')
    if os.path.exists(candidate):
        try:
            with open(candidate, 'r', encoding='utf8') as fh:
                return json.load(fh)
        except Exception:
            return {'error': 'failed to parse benchmark_summary.json'}
    # fall back to per-experiment summary
    candidate2 = os.path.join('release', 'benchmarks', f'ep{exp}_summary.json')
    if os.path.exists(candidate2):
        try:
            with open(candidate2, 'r', encoding='utf8') as fh:
                return json.load(fh)
        except Exception:
            return {'error': 'failed to parse ep summary'}
    return None


def write_result(exp, rc, stdout, stderr, summary):
    timestamp = datetime.utcnow().isoformat() + 'Z'
    out = {
        'experiment': exp,
        'timestamp': timestamp,
        'returncode': rc,
        'stdout_tail': stdout[-400:],
        'stderr_tail': stderr[-400:],
        'summary': summary,
    }
    fname = os.path.join(RESULT_DIR, f'results-exp-{exp}-{timestamp}.json')
    with open(fname, 'w', encoding='utf8') as fh:
        json.dump(out, fh, indent=2)
    return fname


def upload_to_wandb(file_path, project='lareliquia-angulo'):
    key = os.getenv('WANDB_API_KEY')
    if not key:
        print('WANDB_API_KEY not found; skipping W&B upload')
        return
    try:
        import wandb
    except Exception:
        print('wandb not installed; skipping W&B upload')
        return
    wandb.login(key=key)
    run = wandb.init(project=project, job_type='experiment-upload')
    art = wandb.Artifact(f'neurochimera-experiment-results', type='experiment')
    art.add_file(file_path)
    run.log_artifact(art)
    run.finish()
    print('Uploaded result to W&B:', file_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=int, required=True, choices=list(EXP_MAP.keys()))
    args = parser.parse_args()
    script = EXP_MAP[args.exp]
    print('Running experiment', args.exp, '->', script)
    rc, out, err = run_script(script)
    summary = collect_summary(args.exp)
    result_file = write_result(args.exp, rc, out, err, summary)
    print('Result written to', result_file)
    upload_to_wandb(result_file)


if __name__ == '__main__':
    main()
