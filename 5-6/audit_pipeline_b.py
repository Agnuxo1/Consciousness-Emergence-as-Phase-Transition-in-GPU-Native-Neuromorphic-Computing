# audit_pipeline_b.py
# Independent audit using bootstrap confidence intervals on sync metric

import json
import numpy as np
import os

def load_results(path='latest_metrics.json'):
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def bootstrap_sync(sync_series, n_bootstrap=1000, ci=95):
    sync_arr = np.array(sync_series)
    boot_means = []
    rng = np.random.default_rng()
    for _ in range(n_bootstrap):
        sample = rng.choice(sync_arr, size=len(sync_arr), replace=True)
        boot_means.append(sample.mean())
    lower = np.percentile(boot_means, (100 - ci) / 2)
    upper = np.percentile(boot_means, 100 - (100 - ci) / 2)
    return float(lower), float(upper)

def main():
    results = load_results()
    # latest_metrics.json contains metrics directly; experiment1_results.json wraps them under 'metrics'
    if isinstance(results, dict) and 'metrics' in results:
        metrics = results['metrics']
    else:
        metrics = results
    sync = metrics.get('sync')
    if not sync:
        print('No sync data found in metrics.')
        return
    lower, upper = bootstrap_sync(sync)
    print(f'Bootstrap {95}% CI for mean sync: [{lower:.4f}, {upper:.4f}]')
    # Simple consistency check: mean sync should lie within CI (trivial)
    mean_sync = np.mean(sync)
    if lower <= mean_sync <= upper:
        print('Audit B: Mean sync is within bootstrap confidence interval.')
    else:
        print('Audit B: Mean sync falls outside confidence interval!')

if __name__ == '__main__':
    main()
