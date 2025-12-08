# Audit script for Experiment 2
# Verifies Logic (performance > chance) and Statistics (confidence calibration)

import numpy as np
from neuro_chimera_experiments_bundle import RGBACHIMERAConfig, RGBACHIMERASimulation

def audit_experiment_2():
    print("Audit Experiment 2: Starting...")
    
    # 1. Logic Audit: Performance > Chance
    # We run the probe experiment multiple times.
    # The task is binary classification, so random chance is 0.5.
    # We expect the network (even small) to learn something -> acc > 0.55
    
    accuracies = []
    confidences = []
    
    for i in range(10):
        # Use a consistent seed for reproducibility of the audit itself, but different per iter
        cfg = RGBACHIMERAConfig(modules=256, module_size=4, p_conn=0.1, device='cpu', seed=1000+i)
        sim = RGBACHIMERASimulation(cfg)
        res = sim.run_probe_experiment(T=50) # T doesn't affect metacog proxy directly, but useful init
        accuracies.append(res['metacog_acc'])
        confidences.append(res['metacog_conf'])
        
    avg_acc = np.mean(accuracies)
    avg_conf = np.mean(confidences)
    
    print(f"Audit Logic: Avg Accuracy = {avg_acc:.4f} (Threshold > 0.55)")
    if avg_acc > 0.55:
        print("PASS: Logic Audit (Performance > Chance)")
    else:
        print("FAIL: Logic Audit (Performance <= Chance)")
        
    # 2. Statistical Audit: Calibration
    # For a reliable system, confidence should track accuracy.
    # We check if the difference is within a reasonable margin ("calibration error").
    ece_proxy = abs(avg_acc - avg_conf)
    print(f"Audit Stats: Calibration Error (Abs Diff) = {ece_proxy:.4f}")
    
    if ece_proxy < 0.15:
        print("PASS: Statistical Audit (Well Calibrated)")
    else:
        print("WARN: Statistical Audit (Poor Calibration)")

if __name__ == '__main__':
    audit_experiment_2()
