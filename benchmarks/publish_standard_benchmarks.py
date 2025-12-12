#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Publish standard benchmark results to W&B with leaderboard-style visualizations.
"""
import json
import sys
from pathlib import Path

# Fix Windows encoding
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

BASE_DIR = Path(__file__).parent.parent
RESULTS_DIR = BASE_DIR / "release" / "benchmarks" / "standard"
import os
WANDB_API_KEY = os.getenv('WANDB_API_KEY')
if not WANDB_API_KEY:
    raise SystemExit('WANDB_API_KEY not found in environment. Set WANDB_API_KEY env var or run `wandb login`.')

def publish_to_wandb():
    """Publish standard benchmark results to W&B."""
    print("=" * 80)
    print("PUBLISHING STANDARD BENCHMARKS TO W&B")
    print("=" * 80)

    try:
        import wandb
    except ImportError:
        print("Installing wandb...")
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "wandb"])
        import wandb

    # Login
    wandb.login(key=WANDB_API_KEY)

    # Load latest results
    result_files = list(RESULTS_DIR.glob("standard_benchmarks_*.json"))
    if not result_files:
        print("No results found!")
        return None

    latest = max(result_files, key=lambda p: p.stat().st_mtime)
    print(f"\nLoading: {latest.name}")

    with open(latest, 'r', encoding='utf-8') as f:
        results = json.load(f)

    # Initialize run
    run = wandb.init(
        project="neurochimera-standard-benchmarks",
        entity="lareliquia-angulo-agnuxo",
        name=f"standard-benchmarks-{latest.stem.split('_')[-1]}",
        tags=["standard-benchmarks", "cifar10", "imdb", "papers-with-code", "ml-leaderboard"],
        notes="Standard ML benchmarks with real datasets for leaderboard submission"
    )

    print("\nPublishing results...")

    # Log metrics
    if 'cifar10' in results:
        cifar = results['cifar10']
        wandb.log({
            "cifar10/accuracy": cifar['metrics']['accuracy'],
            "cifar10/top1_error": cifar['metrics']['top1_error'],
            "cifar10/train_time_s": cifar['metrics']['train_time_seconds'],
            "cifar10/inference_time_ms": cifar['metrics']['avg_inference_time_ms'],
            "cifar10/throughput_samples_per_sec": cifar['metrics']['throughput_samples_per_sec'],
            "cifar10/parameters": cifar['parameters']
        })

        # Per-class accuracy table
        if 'per_class_accuracy' in cifar:
            class_data = [[cls, acc] for cls, acc in cifar['per_class_accuracy'].items()]
            table = wandb.Table(
                columns=["Class", "Accuracy (%)"],
                data=class_data
            )
            wandb.log({"cifar10/per_class_accuracy": table})

    if 'imdb' in results:
        imdb = results['imdb']
        wandb.log({
            "imdb/accuracy": imdb['metrics']['accuracy'],
            "imdb/precision": imdb['metrics']['precision'],
            "imdb/recall": imdb['metrics']['recall'],
            "imdb/f1_score": imdb['metrics']['f1_score'],
            "imdb/train_time_s": imdb['metrics']['train_time_seconds'],
            "imdb/inference_time_ms": imdb['metrics']['inference_time_ms'],
            "imdb/parameters": imdb['parameters'],
            "imdb/vocabulary_size": imdb['vocabulary_size']
        })

    if 'regression' in results:
        reg = results['regression']
        wandb.log({
            "regression/r2_score": reg['metrics']['r2_score'],
            "regression/rmse": reg['metrics']['rmse'],
            "regression/mae": reg['metrics']['mae'],
            "regression/mse": reg['metrics']['mse'],
            "regression/train_time_s": reg['metrics']['train_time_seconds'],
            "regression/parameters": reg['parameters']
        })

    # Summary table
    summary_data = []
    if 'cifar10' in results:
        summary_data.append([
            "CIFAR-10",
            "Image Classification",
            f"{results['cifar10']['metrics']['accuracy']:.2f}%",
            f"{results['cifar10']['parameters']:,}"
        ])
    if 'imdb' in results:
        summary_data.append([
            "IMDb",
            "Sentiment Analysis",
            f"{results['imdb']['metrics']['accuracy']:.2f}%",
            f"{results['imdb']['parameters']:,}"
        ])
    if 'regression' in results:
        summary_data.append([
            "Regression",
            "Synthetic Data",
            f"R²={results['regression']['metrics']['r2_score']:.4f}",
            f"{results['regression']['parameters']:,}"
        ])

    summary_table = wandb.Table(
        columns=["Benchmark", "Task", "Score", "Parameters"],
        data=summary_data
    )
    wandb.log({"summary": summary_table})

    # Save as artifact
    artifact = wandb.Artifact(
        name="standard-benchmark-results",
        type="benchmark-results",
        description="Standard ML benchmark results for Papers with Code submission"
    )

    artifact.add_file(str(latest), name=f"results/{latest.name}")

    # Add leaderboard tables if they exist
    leaderboard_dir = BASE_DIR / "benchmarks" / "leaderboards"
    if leaderboard_dir.exists():
        for md_file in leaderboard_dir.glob("*.md"):
            artifact.add_file(str(md_file), name=f"leaderboards/{md_file.name}")

    run.log_artifact(artifact)

    # Summary
    wandb.summary.update({
        "total_benchmarks": len(results),
        "has_cifar10": "cifar10" in results,
        "has_imdb": "imdb" in results,
        "has_regression": "regression" in results
    })

    url = run.url
    run.finish()

    print("\n" + "=" * 80)
    print("✅ PUBLISHED SUCCESSFULLY TO W&B")
    print("=" * 80)
    print(f"\nView at: {url}")
    print(f"Project: https://wandb.ai/lareliquia-angulo-agnuxo/neurochimera-standard-benchmarks")
    print("\n" + "=" * 80)

    return url

if __name__ == "__main__":
    try:
        url = publish_to_wandb()
        print(f"\n✅ Success! Results published to:\n{url}")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
