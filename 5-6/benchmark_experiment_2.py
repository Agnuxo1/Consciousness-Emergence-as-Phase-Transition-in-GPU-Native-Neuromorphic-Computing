# Benchmark script for Experiment 2
# Measures performance of RGBA-CHIMERA simulation and probe tasks.

import argparse
import json
import time
import numpy as np

from neuro_chimera_experiments_bundle import run_experiment_2, RGBACHIMERAConfig, RGBACHIMERASimulation

def benchmark(iterations: int = 5, modules: int = 512, pconn: float = 0.1, device: str = 'cpu'):
    results = []
    accuracies = []
    
    for i in range(iterations):
        cfg = RGBACHIMERAConfig(modules=modules, module_size=4, p_conn=pconn, device=device, seed=42 + i)
        sim = RGBACHIMERASimulation(cfg)
        
        start = time.time()
        # Reduce T for benchmark speed unless --full is requested (assumed T=100 from default)
        out = sim.run_probe_experiment(T=100)
        duration = time.time() - start
        
        # Check integrity
        acc = out['metacog_acc']
        if not (0.0 <= acc <= 1.0):
            raise ValueError(f"Invalid accuracy: {acc}")
            
        results.append({
            'iteration': i,
            'duration_sec': duration,
            'metacog_acc': acc,
            'memory_persistence': out['memory_persistence']
        })
        accuracies.append(acc)
        
        # Save run metrics
        with open(f'benchmark_exp2_run_{i}.json', 'w') as f:
            json.dump(out, f)

    # Summary
    summary = {
        'total_iterations': iterations,
        'avg_duration': float(np.mean([r['duration_sec'] for r in results])),
        'avg_accuracy': float(np.mean(accuracies)),
        'runs': results
    }
    
    with open('benchmark_exp2_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
        
    print(f"Benchmark Exp2 Done. Avg Duration: {summary['avg_duration']:.4f}s, Avg Acc: {summary['avg_accuracy']:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--iterations', type=int, default=5)
    parser.add_argument('--modules', type=int, default=512)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()
    benchmark(iterations=args.iterations, modules=args.modules, device=args.device)
