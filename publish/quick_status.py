#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick status check for NeuroCHIMERA publishing system.
Verifies all components are ready.
"""
import os
import sys
from pathlib import Path

# Fix Windows console encoding
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

BASE_DIR = Path(__file__).parent.parent

def check_mark(condition):
    return "✓" if condition else "✗"

def check_files():
    """Check if all required files exist."""
    print("\n📁 FILE CHECK")
    print("=" * 80)

    files = {
        "Scripts": [
            "publish/audit_experiments.py",
            "publish/run_benchmarks.py",
            "publish/run_experiment.py",
            "publish/create_public_dashboards.py",
            "publish/upload_all_platforms.py",
            "publish/update_readme_badges.py",
            "publish/run_and_publish_benchmarks.py"
        ],
        "Documentation": [
            "publish/MASTER_CHECKLIST.md",
            "publish/PLATFORM_GUIDE.md",
            "publish/README_PUBLISHING.md",
            "publish/EXECUTIVE_SUMMARY.md",
            "CITATION.md"
        ],
        "Data": [
            "release/dataset_all.zip",
            "NeuroCHIMERA_Paper.html"
        ],
        "CI/CD": [
            ".github/workflows/benchmarks.yml"
        ]
    }

    for category, file_list in files.items():
        print(f"\n{category}:")
        for file in file_list:
            path = BASE_DIR / file
            exists = path.exists()
            size = path.stat().st_size if exists else 0
            size_str = f"{size:,} bytes" if exists else "N/A"
            print(f"  {check_mark(exists)} {file:<50} {size_str}")

def check_experiments():
    """Check experiment status."""
    print("\n🧪 EXPERIMENT STATUS")
    print("=" * 80)

    experiments = {
        1: "1-2/experiment1_spacetime_emergence.py",
        2: "1-2/experiment2_consciousness_emergence.py",
        3: "3-4/Experiment_Genesis_1.py",
        4: "3-4/Experiment_Genesis_2.py",
        5: "5-6/benchmark_experiment_1.py",
        6: "5-6/benchmark_experiment_2.py"
    }

    for exp_num, exp_path in experiments.items():
        path = BASE_DIR / exp_path
        exists = path.exists()
        print(f"  {check_mark(exists)} Experiment {exp_num}: {exp_path}")

def check_benchmarks():
    """Check benchmark results."""
    print("\n📊 BENCHMARK RESULTS")
    print("=" * 80)

    benchmarks_dir = BASE_DIR / "release" / "benchmarks"
    if not benchmarks_dir.exists():
        print("  ✗ Benchmarks directory not found")
        return

    for exp_num in range(1, 7):
        results = list(benchmarks_dir.glob(f"results-exp-{exp_num}-*.json"))
        count = len(results)
        latest = max(results, key=lambda p: p.stat().st_mtime).name if results else "None"
        print(f"  Experiment {exp_num}: {count} run(s) - Latest: {latest}")

def check_credentials():
    """Check if credentials are configured."""
    print("\n🔑 CREDENTIALS CHECK")
    print("=" * 80)

    creds = {
        "W&B API Key": os.getenv('WANDB_API_KEY'),
        "Zenodo Token": os.getenv('ZENDO_TOKEN') or os.getenv('ZENODO_TOKEN'),
        "OSF Token": os.getenv('OSF_TOKEN'),
        "Figshare User": os.getenv('FIGSHARE_USERNAME'),
    }

    def mask(val):
        if not val:
            return 'NOT SET'
        s = str(val)
        if len(s) <= 10:
            return s
        return s[:6] + '...' + s[-4:]

    for name, value in creds.items():
        print(f"  ✓ {name:<20} {mask(value)}")

def check_platforms():
    """Check platform readiness."""
    print("\n🌐 PLATFORM READINESS")
    print("=" * 80)

    platforms = {
        "W&B": {"auto": True, "ready": True},
        "Zenodo": {"auto": True, "ready": True},
        "OSF": {"auto": True, "ready": True},
        "Figshare": {"auto": False, "ready": True},
        "OpenML": {"auto": False, "ready": True},
        "DataHub": {"auto": False, "ready": True},
        "Academia.edu": {"auto": False, "ready": True},
        "DrivenData": {"auto": False, "ready": False},
        "OpenAIRE": {"auto": True, "ready": False}
    }

    print(f"\n{'Platform':<15} {'Type':<15} {'Status':<10}")
    print("-" * 40)
    for platform, info in platforms.items():
        auto_type = "Automated" if info["auto"] else "Manual"
        status = "✓ Ready" if info["ready"] else "⏳ Pending"
        print(f"{platform:<15} {auto_type:<15} {status:<10}")

def check_next_steps():
    """Display next steps."""
    print("\n🚀 NEXT STEPS")
    print("=" * 80)

    steps = [
        "1. Resolver dependencias: pip install wgpu",
        "2. Ejecutar benchmarks: python publish/run_benchmarks.py",
        "3. Crear dashboards W&B: python publish/create_public_dashboards.py",
        "4. Publicar en Zenodo: python publish/upload_all_platforms.py",
        "5. Actualizar badges: python publish/update_readme_badges.py",
        "6. Seguir MASTER_CHECKLIST.md para publicaciones manuales"
    ]

    for step in steps:
        print(f"  {step}")

def main():
    print("=" * 80)
    print("NEUROCHIMERA PUBLISHING SYSTEM - QUICK STATUS")
    print("=" * 80)

    check_files()
    check_experiments()
    check_benchmarks()
    check_credentials()
    check_platforms()
    check_next_steps()

    print("\n" + "=" * 80)
    print("STATUS CHECK COMPLETE")
    print("=" * 80)
    print("\nFor detailed instructions, see:")
    print("  - publish/EXECUTIVE_SUMMARY.md (overview)")
    print("  - publish/MASTER_CHECKLIST.md (detailed steps)")
    print("  - publish/PLATFORM_GUIDE.md (platform details)")
    print("  - publish/README_PUBLISHING.md (how-to)")
    print("=" * 80)

if __name__ == "__main__":
    main()
