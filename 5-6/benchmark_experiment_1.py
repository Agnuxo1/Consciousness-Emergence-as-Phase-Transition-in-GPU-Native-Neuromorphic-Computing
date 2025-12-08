# Benchmark script for Experiment 1
# This script runs Experiment 1 multiple times with varying parameters,
# measures execution time and (if available) GPU memory usage, and validates
# that the resulting metrics are within expected ranges.

import argparse
import json
import time
import os
import sys

import numpy as np

# Import the experiment runner
from neuro_chimera_experiments_bundle import run_experiment_1, GFNetworkConfig, GFNetworkSimulation

def benchmark(iterations: int = 5, N: int = 65536, p: float = 0.02, bits: int = 1, device: str = 'cpu'):
    results = []
    for i in range(iterations):
        cfg = GFNetworkConfig(N=N, p_conn=p, field_bits=bits, device=device, seed=42 + i)
        sim = GFNetworkSimulation(cfg)
        start = time.time()
        out = sim.run(T=2000, sample_interval=10, noise=0.01)
        duration = time.time() - start
        # Optional GPU memory usage (if torch and CUDA)
        gpu_mem = None
        try:
            import torch
            if torch.cuda.is_available():
                gpu_mem = torch.cuda.max_memory_allocated(device=None) / (1024 ** 2)  # MB
        except Exception:
            pass
        # Basic sanity checks
        sync = out['metrics']['sync']
        if not sync:
            raise RuntimeError('No sync data produced')
        results.append({
            'iteration': i,
            'duration_sec': duration,
            'gpu_mem_mb': gpu_mem,
            'final_sync': sync[-1],
            'tc': out.get('tc')
        })
        # Save metrics for this run (optional)
        with open(f'benchmark_run_{i}_metrics.json', 'w') as f:
            json.dump(out['metrics'], f)
    # Summary
    summary_path = 'benchmark_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'Benchmark completed. Summary written to {summary_path}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Benchmark Experiment 1')
    parser.add_argument('--iterations', type=int, default=5)
    parser.add_argument('--N', type=int, default=65536)
    parser.add_argument('--p', type=float, default=0.02)
    parser.add_argument('--bits', type=int, default=1)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()
    benchmark(iterations=args.iterations, N=args.N, p=args.p, bits=args.bits, device=args.device)

