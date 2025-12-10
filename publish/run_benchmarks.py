#!/usr/bin/env python3
"""Run repository benchmarks and collect results.

This script wraps the repository's benchmark runner(s), stores results in
`release/benchmarks/results.json` and (optionally) uploads the results as an
artifact to Weights & Biases when `WANDB_API_KEY` is available in the environment.

It is safe to run locally; the CI workflow included in `.github/workflows/benchmarks.yml`
will call this script on a schedule once `WANDB_API_KEY` is configured in GitHub
Secrets.
"""
import os
import json
import subprocess
from datetime import datetime

try:
    import wandb
except Exception:
    wandb = None


RESULT_DIR = os.path.join('release', 'benchmarks')
os.makedirs(RESULT_DIR, exist_ok=True)


def run_local_benchmarks():
    """Calls the existing `run_all_benchmarks.py` script and captures output."""
    script = os.path.join(os.getcwd(), 'run_all_benchmarks.py')
    if not os.path.exists(script):
        raise FileNotFoundError('run_all_benchmarks.py not found in repository root')

    timestamp = datetime.utcnow().isoformat() + 'Z'
    result_file = os.path.join(RESULT_DIR, f'results_{timestamp}.json')

    print('Running benchmarks (this may take some time)...')
    # Run the benchmark script and capture JSON output if it supports it.
    # Many repo benchmark scripts will generate files; we call them and then
    # collect summary metrics from known output locations. If `run_all_benchmarks.py`
    # already writes results, we preserve them.
    proc = subprocess.run(['python', script], capture_output=True, text=True)
    stdout = proc.stdout
    stderr = proc.stderr
    exitcode = proc.returncode

    summary = {
        'timestamp': timestamp,
        'exitcode': exitcode,
        'stdout_tail': stdout[-800:],
        'stderr_tail': stderr[-800:]
    }

    # Try to find existing benchmark summary files produced by the script
    candidate = os.path.join('benchmarks', 'benchmark_summary.json')
    if os.path.exists(candidate):
        try:
            with open(candidate, 'r', encoding='utf8') as fh:
                summary['benchmark_summary'] = json.load(fh)
        except Exception:
            summary['benchmark_summary'] = {'error': 'failed to parse benchmark_summary.json'}

    with open(result_file, 'w', encoding='utf8') as fh:
        json.dump(summary, fh, indent=2)

    print('Benchmark results written to', result_file)
    return result_file


def upload_to_wandb(result_file, project='lareliquia-angulo'):
    key = os.getenv('WANDB_API_KEY')
    if not key:
        print('WANDB_API_KEY not found; skipping W&B upload')
        return
    if wandb is None:
        print('wandb package not available; skipping W&B upload')
        return

    wandb.login(key=key)
    run = wandb.init(project=project, job_type='benchmark')
    artifact = wandb.Artifact('neurochimera-benchmarks', type='benchmark')
    artifact.add_file(result_file)
    run.log_artifact(artifact)
    run.finish()
    print('Uploaded benchmark artifact to W&B')


def main():
    result_file = run_local_benchmarks()
    upload_to_wandb(result_file)


if __name__ == '__main__':
    main()
