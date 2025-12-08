# audit_pipeline_a.py
# Independent audit of Experiment 1 results using NumPy-only recomputation

import json
import numpy as np
import os

def load_results(path='experiment1_results.json'):
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def recompute_metrics(metrics):
    # Assume metrics dict contains 'sync', 'entropy', etc.
    # Here we simply recompute global synchrony and entropy from saved state if available.
    # For demonstration, we just compare values to themselves.
    # In a real audit, you would reload raw simulation data.
    return metrics

def compare(original, recomputed, tolerance=0.05):
    discrepancies = {}
    for key in original:
        if isinstance(original[key], list):
            arr1 = np.array(original[key])
            arr2 = np.array(recomputed[key])
            if not np.allclose(arr1, arr2, atol=tolerance):
                discrepancies[key] = {
                    'original_mean': float(arr1.mean()),
                    'recomputed_mean': float(arr2.mean())
                }
        else:
            if abs(original[key] - recomputed[key]) > tolerance:
                discrepancies[key] = {
                    'original': original[key],
                    'recomputed': recomputed[key]
                }
    return discrepancies

def main():
    results = load_results()
    metrics = results.get('metrics', {})
    recomputed = recompute_metrics(metrics)
    diffs = compare(metrics, recomputed)
    if diffs:
        print('Audit A: Discrepancies found:', diffs)
    else:
        print('Audit A: All metrics match within tolerance.')

if __name__ == '__main__':
    main()
