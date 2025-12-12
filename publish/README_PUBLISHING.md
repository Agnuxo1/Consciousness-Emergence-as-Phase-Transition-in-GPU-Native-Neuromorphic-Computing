# NeuroCHIMERA Publishing & Benchmarking Guide

Complete guide for auditing, benchmarking, and publishing NeuroCHIMERA experiments to multiple platforms.

## Quick Start

### 1. Complete Pipeline (Automated)

```bash
# Run everything: audit, benchmarks, dashboards, uploads
python publish/run_and_publish_benchmarks.py
```

### 2. Individual Steps

```bash
# Step 1: Audit all experiments
python publish/audit_experiments.py

# Step 2: Run all benchmarks
python publish/run_benchmarks.py

# Step 3: Create W&B dashboards
python publish/create_public_dashboards.py

# Step 4: Upload to all platforms
python publish/upload_all_platforms.py
```

### 3. Individual Experiment

```bash
# Run single experiment
python publish/run_experiment.py --exp 1
```

## Current Status

Based on the latest audit (`release/audit_report_*.md`):

### ✓ Experiments Status
- **6/6 experiments** passed syntax and structure checks
- **All benchmark scripts** present and valid
- **8/8 benchmark scripts** available

### ⚠ Known Issues
- **Experiments 1-2**: Missing `wgpu` dependency
  ```bash
  pip install wgpu
  ```

- **Experiments 5-6**: Missing `neuro_chimera_experiments_bundle`
  ```bash
  # This appears to be a custom package - check if it's needed or mock it
  ```

### ✓ Recent Executions
- Experiment 3: 2 successful runs
- Experiment 4: 2 successful runs
- Experiments 1-2: Runs attempted (dependency issues)

## Publishing Platforms

### Automated Platforms

#### 1. Weights & Biases (W&B) ✓
```bash
export WANDB_API_KEY="<YOUR_WANDB_API_KEY>"
python publish/create_public_dashboards.py
```

**Access**: https://wandb.ai/lareliquia-angulo

**Features**:
- Live experiment tracking
- Public dashboards
- Benchmark comparisons
- Artifact versioning

#### 2. Zenodo ✓ (Draft)
```bash
python publish/upload_all_platforms.py
```

**Access**: https://zenodo.org/me/uploads

**What happens**:
- Creates draft deposition
- Uploads dataset_all.zip
- Uploads paper HTML
- Uploads benchmark results
- Generates DOI (after manual publish)

**Next steps**:
1. Visit Zenodo uploads page
2. Find draft deposition
3. Review metadata
4. Click "Publish"
5. Copy DOI for citations

#### 3. Open Science Framework (OSF) ✓
```bash
python publish/upload_all_platforms.py
```

**Access**: https://osf.io/

**Token**: `<YOUR_OSF_TOKEN>`

**What happens**:
- Creates public project
- Adds metadata and tags
- Provides project URL

**Manual file upload**:
```bash
pip install osfclient
export OSF_TOKEN="<YOUR_OSF_TOKEN>"
osf -p <project_id> upload release/ /data/
```

### Manual Upload Platforms

#### 4. Figshare 📤
**Access**: https://figshare.com/

**FTP Credentials**:
- Username: `5292188`
- Password: `<FIGSHARE_PASSWORD>`

**Option A - Web Interface**:
1. Visit https://figshare.com/account/home
2. Click "Upload" → "Create a new item"
3. Select item type: "Dataset"
4. Upload `release/dataset_all.zip`
5. Add metadata:
   - Title: "NeuroCHIMERA Consciousness Emergence Experiments"
   - Authors: V.F. Veselov, Francisco Angulo de Lafuente
   - Categories: Artificial Intelligence, Neuroscience
   - Keywords: consciousness, phase transition, neuromorphic, GPU, benchmark
   - Description: [Copy from paper abstract]
6. Click "Publish"

**Option B - FTP Upload**:
```bash
# Using lftp (Linux/Mac)
lftp -u 5292188,'<FIGSHARE_PASSWORD>' ftp://figshare.com
cd uploads
put release/dataset_all.zip
put release/benchmarks/*.json
bye
```

**Option C - Python API**:
```python
import requests
import json

# Upload using Figshare API
# See: https://docs.figshare.com/
```

#### 5. OpenML 📤
**Access**: https://www.openml.org/

**Preparation**:
```bash
python publish/upload_all_platforms.py
# This creates release/openml_export/ with ARFF files
```

**Programmatic registration via script**
```bash
# Create an OpenML API key at https://www.openml.org/profile
export OPENML_API_KEY="<YOUR_OPENML_API_KEY>"
python publish/openml_dataset_registration.py --input release/openml_export/ --name NeuroCHIMERA --description "NeuroCHIMERA experiment datasets" --version 1.0 --license CC-BY-4.0
```

This script will convert CSV/JSON to ARFF if needed and call the OpenML client to create and publish a dataset programmatically. The tool prints the created dataset ID and a link to the dataset on OpenML when available.

**Automated**: You can also add `OPENML_API_KEY` to your environment and run the full pipeline:
```bash
export OPENML_API_KEY="<YOUR_OPENML_API_KEY>"
python publish/upload_all_platforms.py
```
The pipeline will automatically register datasets on OpenML when the API key is available.

**Manual Upload**:
1. Visit https://www.openml.org/
2. Login to your account
3. For each experiment (1-6):
   - Click "Upload" → "Dataset"
   - Upload from `release/openml_export/`
   - Fill metadata:
     - Name: NeuroCHIMERA Experiment N
     - Description: [From experiment docstring]
     - Format: ARFF
     - Tags: consciousness, neuromorphic, GPU, benchmark, phase-transition
   - Click "Submit"

#### 6. DataHub 📤
#### 7. Hugging Face Hub (Datasets) ✓
**Access**: https://huggingface.co/

**Preparation**:
```bash
# Create a write token in your Hugging Face account: https://huggingface.co/settings/tokens
export HF_TOKEN="<YOUR_HF_TOKEN>"
``` 

**Upload via script**:
```bash
python publish/upload_to_hf.py --repo-id Agnuxo/neurochimera-dataset --repo-type dataset --path release --commit "Initial upload"
```

**Notes**: This creates a dataset repository containing `release/` files. Consider splitting large files and use LFS for big assets.

**Automated**: If you prefer to upload as part of the full pipeline, ensure `HF_TOKEN` is set and run:
```bash
export HF_TOKEN="<YOUR_HF_TOKEN>"
python publish/upload_all_platforms.py
```
The `upload_all_platforms.py` script will upload to Hugging Face automatically when `HF_TOKEN` is present in the environment.

**Access**: https://datahub.io/

**Preparation**:
```bash
python publish/upload_all_platforms.py
# Creates release/datahub_export/ with datapackage.json
```

**Upload via CLI**:
```bash
# Install DataHub CLI
npm install -g data-cli

# Login
data login

# Publish dataset
cd release/datahub_export
data push neurochimera-experiments

# Or push with explicit name
data push neurochimera-experiments --dataset=neurochimera-experiments
```

**Upload via Web**:
1. Visit https://datahub.io/
2. Click "Publish Data"
3. Upload `release/datahub_export/datapackage.json`
4. Confirm dataset structure
5. Publish

#### 7. Academia.edu 📤
**Access**: https://www.academia.edu/

**Gravatar**: https://gravatar.com/ecofa

**Preparation**:
```bash
python publish/upload_all_platforms.py
# Creates release/academia_export/
```

**Manual Upload**:
1. Visit https://www.academia.edu/
2. Click "Upload" in top menu
3. Upload paper:
   - File: `release/academia_export/NeuroCHIMERA_Paper.html`
   - Title: NeuroCHIMERA: Consciousness Emergence as Phase Transition in Neuromorphic GPU-Native Computing
   - Authors: V.F. Veselov, Francisco Angulo de Lafuente
   - Abstract: [Copy from paper]
   - Keywords: consciousness, phase transition, neuromorphic computing, GPU, artificial consciousness
   - Discipline: Computer Science > Artificial Intelligence
4. Upload supplementary materials:
   - File: `release/academia_export/supplementary_materials.zip`
   - Type: "Supplemental Material"
5. Click "Publish"

#### 8. DrivenData (Optional) 📤
**Access**: https://www.drivendata.org/users/Francisco_Angulo_Lafuente/

**Purpose**: Create public benchmark challenge

**Steps**:
1. Contact DrivenData: https://www.drivendata.org/contact/
2. Propose challenge:
   - Title: "NeuroCHIMERA Consciousness Emergence Benchmark"
   - Description: Reproduce and improve upon consciousness emergence experiments
   - Prize: Community recognition, co-authorship on follow-up paper
   - Timeline: 3-6 months
3. Provide dataset and baseline results
4. Define evaluation metrics
5. Set up leaderboard

**Challenge Structure**:
- Track 1: Reproduce experiments (verification)
- Track 2: Improve metrics (innovation)
- Track 3: Novel architectures (creativity)

### Automatic Platforms

#### 9. OpenAIRE ✓ (Automatic)
**Access**: https://explore.openaire.eu/

**What happens**:
- Automatically harvests from Zenodo after publication
- No manual action required
- Wait 24-48 hours after Zenodo publication

**Verification**:
1. Wait 48 hours after Zenodo publication
2. Search for "NeuroCHIMERA" at https://explore.openaire.eu/
3. Verify dataset appears in results
4. Check metadata accuracy

## File Structure

```
release/
├── audit_report_TIMESTAMP.json     # Audit results (JSON)
├── audit_report_TIMESTAMP.md       # Audit results (Markdown)
├── upload_report_TIMESTAMP.json    # Upload status
├── pipeline_summary_TIMESTAMP.json # Complete pipeline status
├── benchmarks/
│   ├── results-exp-1-TIMESTAMP.json
│   ├── results-exp-2-TIMESTAMP.json
│   ├── results-exp-3-TIMESTAMP.json
│   ├── results-exp-4-TIMESTAMP.json
│   ├── results-exp-5-TIMESTAMP.json
│   └── results-exp-6-TIMESTAMP.json
├── openml_export/
│   ├── openml_metadata.json
│   └── experiment_*.arff
├── datahub_export/
│   ├── datapackage.json
│   └── benchmarks/
└── academia_export/
    ├── NeuroCHIMERA_Paper.html
    └── supplementary_materials.zip
```

## Platform URLs Reference

### Your Accounts
- **W&B**: https://wandb.ai/lareliquia-angulo
- **Zenodo**: https://zenodo.org/me/uploads
- **OSF**: https://osf.io/myprojects
- **Figshare**: https://figshare.com/account/home
- **OpenML**: https://www.openml.org/auth/profile-page
- **DataHub**: https://datahub.io/
- **Academia.edu**: https://www.academia.edu/
- **DrivenData**: https://www.drivendata.org/users/Francisco_Angulo_Lafuente/
- **Gravatar**: https://gravatar.com/ecofa

### Search & Discovery
- **OpenAIRE**: https://explore.openaire.eu/
- **Signate** (Japan): https://signate.jp/
- **Zindi** (Africa): https://zindi.africa/

## Badges for README

After publishing, add these badges to your README:

```markdown
[![W&B](https://img.shields.io/badge/Weights%20%26%20Biases-Dashboard-yellow)](https://wandb.ai/lareliquia-angulo/neurochimera-full-experiments)

[![Zenodo](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXX)

[![OSF](https://img.shields.io/badge/OSF-Project-blue)](https://osf.io/PROJECT_ID/)

[![Figshare](https://img.shields.io/badge/Figshare-Dataset-orange)](https://figshare.com/articles/dataset/...)

[![OpenML](https://img.shields.io/badge/OpenML-Experiments-green)](https://www.openml.org/search?type=data&q=neurochimera)
```

## Citation Template

After obtaining DOIs, use this citation:

```bibtex
@dataset{veselov2025neurochimera,
  author = {Veselov, V. F. and Angulo de Lafuente, Francisco},
  title = {NeuroCHIMERA: Consciousness Emergence as Phase Transition in Neuromorphic GPU-Native Computing},
  year = {2025},
  publisher = {Zenodo},
  doi = {10.5281/zenodo.XXXXX},
  url = {https://doi.org/10.5281/zenodo.XXXXX}
}
```

## Troubleshooting

### Issue: Missing dependencies
```bash
# Install all required dependencies
pip install -r requirements_experiments.txt

# Install specific missing packages
pip install wgpu  # For experiments 1-2
```

### Issue: W&B upload fails
```bash
# Check API key
echo $WANDB_API_KEY

# Re-login
wandb login <YOUR_WANDB_API_KEY>
```

### Issue: Zenodo upload fails
- Check token validity at https://zenodo.org/account/settings/applications/
- Ensure file size < 50GB per file
- Try splitting large files

### Issue: FTP to Figshare fails
- Verify credentials: Username `5292188`, Password `<FIGSHARE_PASSWORD>`
- Check firewall settings for FTP
- Try web interface instead

### Issue: Experiments don't run
```bash
# Check audit report
cat release/audit_report_*.md

# Install missing dependencies
pip install <missing_package>

# Run with verbose output
python publish/run_experiment.py --exp 1 2>&1 | tee experiment_1_output.log
```

## Support

For questions or issues:
- GitHub Issues: https://github.com/Agnuxo1/Consciousness-Emergence-as-Phase-Transition-in-GPU-Native-Neuromorphic-Computing/issues

## Authentication & Secrets (PowerShell / Bash)

Set environment variables before running the scripts. DO NOT hard-code tokens in scripts or commit them to git.

Bash (Linux / macOS):
```bash
export WANDB_API_KEY="<YOUR_WANDB_API_KEY>"
export ZENDO_TOKEN="<YOUR_ZENODO_TOKEN>"
export OSF_TOKEN="<YOUR_OSF_TOKEN>" # optional
export FIGSHARE_USERNAME="5292188"
export FIGSHARE_PASSWORD="<FIGSHARE_PASSWORD>"
export HF_TOKEN="<YOUR_HUGGINGFACE_TOKEN>"
export OPENML_API_KEY="<YOUR_OPENML_API_KEY>"
```

PowerShell (Windows):
```powershell
$env:WANDB_API_KEY = '<YOUR_WANDB_API_KEY>'
$env:ZENDO_TOKEN = '<YOUR_ZENODO_TOKEN>'
$env:OSF_TOKEN = '<YOUR_OSF_TOKEN>' # optional
$env:FIGSHARE_USERNAME = '5292188'
$env:FIGSHARE_PASSWORD = '<FIGSHARE_PASSWORD>'
$env:HF_TOKEN = '<YOUR_HUGGINGFACE_TOKEN>'
$env:OPENML_API_KEY = '<YOUR_OPENML_API_KEY>'
```

Helpful hints:
- Use GitHub Secrets to store tokens for CI (Actions). See `README_SECRETS.md` for guidance.
- For HF token: create via https://huggingface.co/settings/tokens with `repo` scope.
- For OSF: create personal access token via https://osf.io/settings/tokens/ and mark `publish` (if needed).
- For Zenodo: create token at https://zenodo.org/account/settings/applications/ and give `deposit:write` permission.
- Contact: Francisco Angulo de Lafuente

## License

All code: GPL-3.0
All data: CC-BY-4.0
Paper: CC-BY-4.0

---

Last updated: 2025-12-09
