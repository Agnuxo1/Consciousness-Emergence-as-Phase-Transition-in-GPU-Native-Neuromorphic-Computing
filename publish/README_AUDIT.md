Reproducible audit & benchmarking instructions
=============================================

This document explains how external auditors and benchmarkers can reproduce the experiments
and run automated benchmarks. It includes quick commands, links to artifacts and how to
interpret results produced by the scheduled CI job.

Quick reproduction (local)
1. Clone repository and create environment:
```powershell
git clone https://github.com/Agnuxo1/Consciousness-Emergence-as-Phase-Transition-in-GPU-Native-Neuromorphic-Computing.git
cd Consciousness-Emergence-as-Phase-Transition-in-GPU-Native-Neuromorphic-Computing
conda env create -f environment.yml
conda activate neurochimera-experiments
pip install -r requirements_experiments.txt
```

2. Prepare data (release artifacts are included in `release/` or fetch dataset from Zenodo):
```powershell
# Option A: use release artifacts included in the repository
ls release

# Option B: download from Zenodo (dataset DOI)
curl -L -o dataset_all.zip "https://zenodo.org/record/17872411/files/dataset_all.zip"
unzip dataset_all.zip -d dataset
```

3. Run benchmarks locally
```powershell
python publish/run_benchmarks.py
```

CI / Scheduled benchmarks
- The repository contains a scheduled GitHub Actions workflow `.github/workflows/benchmarks.yml`
  that runs weekly and stores benchmark outputs as build artifacts under the workflow run.
- When `WANDB_API_KEY` is configured as a repository secret, benchmark results are also uploaded
  to the W&B project `lareliquia-angulo` as artifacts for easy browsing and comparison.

Audit checklist for external reviewers
- Verify code integrity: run `python -m flake8` or similar linter if desired.
- Re-run benchmarks locally and compare `release/benchmarks/*.json` produced locally with the
  CI artifacts attached to the latest workflow run.
- Verify dataset contents and checksums against Zenodo record.

Contact & support
- For questions about reproducing experiments, contact Francisco (GitHub: @Agnuxo1).
