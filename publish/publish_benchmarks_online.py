#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Publish benchmark results online to W&B with detailed metrics and visualizations.
"""
import os
import sys
import json
from pathlib import Path
from datetime import datetime, timezone

# Fix Windows console encoding
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

BASE_DIR = Path(__file__).parent.parent
BENCHMARKS_DIR = BASE_DIR / "release" / "benchmarks"
WANDB_API_KEY = "b017394dfb1bfdbcaf122dcd20383d5ac9cb3bae"

def publish_to_wandb():
    """Publish all benchmark results to W&B with detailed visualizations."""
    print("=" * 80)
    print("PUBLISHING BENCHMARK RESULTS TO WEIGHTS & BIASES")
    print("=" * 80)

    try:
        import wandb
    except ImportError:
        print("✗ wandb not installed. Installing...")
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "wandb"])
        import wandb

    # Login to W&B
    wandb.login(key=WANDB_API_KEY)

    # Initialize run
    run = wandb.init(
        project="neurochimera-benchmarks",
        entity="lareliquia-angulo-agnuxo",
        name=f"benchmark-run-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}",
        tags=["benchmark", "neurochimera", "consciousness", "experiments"],
        notes="Complete benchmark results for 6 NeuroCHIMERA experiments"
    )

    # Collect all benchmark results
    experiment_results = {}
    experiment_info = {
        1: {
            "name": "Spacetime Emergence",
            "description": "Space-time emergence from computational network",
            "physics": "Einstein equations from connectivity"
        },
        2: {
            "name": "Consciousness Emergence",
            "description": "Consciousness as phase transition",
            "physics": "Free energy minimization"
        },
        3: {
            "name": "Genesis 1",
            "description": "Genesis simulation - computational substrate",
            "physics": "Emergent dynamics"
        },
        4: {
            "name": "Genesis 2",
            "description": "Genesis simulation - advanced patterns",
            "physics": "Magnetization and weight evolution"
        },
        5: {
            "name": "Benchmark 1",
            "description": "Performance benchmark 1",
            "physics": "Computational efficiency"
        },
        6: {
            "name": "Benchmark 2",
            "description": "Performance benchmark 2",
            "physics": "Scaling behavior"
        }
    }

    print("\nCollecting benchmark results...")
    for exp_num in range(1, 7):
        # Find latest result for this experiment
        results = list(BENCHMARKS_DIR.glob(f"results-exp-{exp_num}-*.json"))
        if not results:
            print(f"  ⚠ No results found for experiment {exp_num}")
            continue

        latest_result = max(results, key=lambda p: p.stat().st_mtime)

        with open(latest_result, encoding='utf-8') as f:
            data = json.load(f)

        experiment_results[exp_num] = {
            "file": latest_result.name,
            "data": data,
            "info": experiment_info[exp_num]
        }

        success = data.get("returncode") == 0
        status = "✓" if success else "✗"
        print(f"  {status} Experiment {exp_num}: {latest_result.name}")

    # Create summary table
    print("\nCreating W&B summary table...")
    columns = ["Exp", "Name", "Status", "Timestamp", "Return Code"]
    table_data = []

    for exp_num, result in experiment_results.items():
        data = result["data"]
        info = result["info"]
        status = "✓ Success" if data.get("returncode") == 0 else "✗ Failed"

        table_data.append([
            exp_num,
            info["name"],
            status,
            data.get("timestamp", "N/A"),
            data.get("returncode", -1)
        ])

    table = wandb.Table(columns=columns, data=table_data)
    wandb.log({"benchmark_summary": table})

    # Log individual experiment metrics
    print("\nLogging experiment metrics...")
    for exp_num, result in experiment_results.items():
        data = result["data"]
        info = result["info"]

        metrics = {
            f"exp_{exp_num}/return_code": data.get("returncode", -1),
            f"exp_{exp_num}/success": 1 if data.get("returncode") == 0 else 0,
            f"exp_{exp_num}/has_output": 1 if data.get("stdout_tail") else 0,
        }

        # Parse output for specific metrics if available
        stdout = data.get("stdout_tail", "")
        if "Magnetization" in stdout and exp_num == 4:
            # Extract magnetization value for experiment 4
            lines = stdout.split('\n')
            for line in reversed(lines):
                if "Magnetization" in line:
                    try:
                        mag_str = line.split("Magnetization=")[1].split(",")[0]
                        metrics[f"exp_{exp_num}/magnetization"] = float(mag_str)
                        break
                    except:
                        pass

        wandb.log(metrics)
        print(f"  ✓ Logged metrics for experiment {exp_num}")

    # Create detailed experiment info table
    info_columns = ["Exp", "Name", "Description", "Physics Framework"]
    info_data = []
    for exp_num, result in experiment_results.items():
        info = result["info"]
        info_data.append([
            exp_num,
            info["name"],
            info["description"],
            info["physics"]
        ])

    info_table = wandb.Table(columns=info_columns, data=info_data)
    wandb.log({"experiment_details": info_table})

    # Log summary statistics
    total_experiments = len(experiment_results)
    successful = sum(1 for r in experiment_results.values() if r["data"].get("returncode") == 0)
    success_rate = (successful / total_experiments * 100) if total_experiments > 0 else 0

    wandb.summary.update({
        "total_experiments": total_experiments,
        "successful_experiments": successful,
        "failed_experiments": total_experiments - successful,
        "success_rate": success_rate,
        "audit_timestamp": datetime.now(timezone.utc).isoformat()
    })

    print("\nSummary statistics:")
    print(f"  Total experiments: {total_experiments}")
    print(f"  Successful: {successful}")
    print(f"  Failed: {total_experiments - successful}")
    print(f"  Success rate: {success_rate:.1f}%")

    # Upload benchmark files as artifacts
    print("\nUploading benchmark artifacts...")
    artifact = wandb.Artifact(
        name="neurochimera-benchmark-results",
        type="benchmark-results",
        description="Complete benchmark results for NeuroCHIMERA experiments"
    )

    for result_file in BENCHMARKS_DIR.glob("results-exp-*.json"):
        artifact.add_file(str(result_file), name=f"benchmarks/{result_file.name}")

    run.log_artifact(artifact)
    print("  ✓ Uploaded all benchmark result files")

    # Finish run
    run_url = run.get_url()
    run.finish()

    print("\n" + "=" * 80)
    print("✓ BENCHMARK RESULTS PUBLISHED SUCCESSFULLY!")
    print("=" * 80)
    print(f"\nView results at: {run_url}")
    print(f"Project: https://wandb.ai/lareliquia-angulo-agnuxo/neurochimera-benchmarks")
    print("\n" + "=" * 80)

    return run_url

def main():
    print("\n")
    print("█" * 80)
    print("█  NEUROCHIMERA ONLINE BENCHMARK PUBLICATION SYSTEM")
    print("█" * 80)
    print("\n")

    if not BENCHMARKS_DIR.exists():
        print(f"✗ Benchmarks directory not found: {BENCHMARKS_DIR}")
        return 1

    # Count available results
    result_files = list(BENCHMARKS_DIR.glob("results-exp-*.json"))
    print(f"Found {len(result_files)} benchmark result files\n")

    # Publish to W&B
    try:
        url = publish_to_wandb()

        print("\n🎉 SUCCESS! Your benchmarks are now publicly available online!")
        print("\nShare this link with collaborators:")
        print(f"  {url}")

        return 0
    except Exception as e:
        print(f"\n✗ Error publishing benchmarks: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
