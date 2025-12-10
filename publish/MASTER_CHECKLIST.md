# NeuroCHIMERA Publication Master Checklist

Complete checklist for publishing 6 experiments across all platforms with maximum visibility.

## Phase 1: Preparation & Audit ✓

### 1.1 Audit Experiments
- [x] Run audit script: `python publish/audit_experiments.py`
- [x] Review audit reports in `release/audit_report_*.md`
- [x] Fix dependency issues
  - [ ] Install `wgpu` for experiments 1-2
  - [ ] Resolve `neuro_chimera_experiments_bundle` for experiments 5-6
- [x] Verify all 6 experiments pass syntax checks

### 1.2 Generate Documentation
- [x] Create platform guide: `publish/PLATFORM_GUIDE.md`
- [x] Create publishing README: `publish/README_PUBLISHING.md`
- [x] Generate CITATION.md
- [x] Create audit checklist for external reviewers

### 1.3 Prepare Datasets
- [x] Verify `release/dataset_all.zip` exists and is complete
- [x] Generate checksums for data integrity
- [ ] Create dataset README with descriptions

---

## Phase 2: Run Benchmarks

### 2.1 Local Benchmark Execution
- [ ] Install missing dependencies:
  ```bash
  pip install wgpu  # For experiments 1-2
  pip install -r requirements_experiments.txt
  ```

- [ ] Run all benchmarks:
  ```bash
  python publish/run_benchmarks.py
  ```

- [ ] Verify benchmark results:
  - [ ] Experiment 1: Spacetime emergence
  - [ ] Experiment 2: Consciousness emergence
  - [ ] Experiment 3: Genesis 1 ✓ (already successful)
  - [ ] Experiment 4: Genesis 2 ✓ (already successful)
  - [ ] Experiment 5: Benchmark 1
  - [ ] Experiment 6: Benchmark 2

### 2.2 Generate Benchmark Reports
- [ ] Review `release/benchmarks/results-exp-*.json`
- [ ] Create summary visualization
- [ ] Document any anomalies or failures

---

## Phase 3: Automated Platform Uploads

### 3.1 Weights & Biases (W&B) ✓
**Priority**: HIGH | **Automated**: YES | **Public**: YES

- [ ] Set API key:
  ```bash
  export WANDB_API_KEY="b017394dfb1bfdbcaf122dcd20383d5ac9cb3bae"
  ```

- [ ] Create dashboards:
  ```bash
  python publish/create_public_dashboards.py
  ```

- [ ] Verify dashboards:
  - [ ] Main dashboard: https://wandb.ai/lareliquia-angulo/neurochimera-full-experiments
  - [ ] Performance dashboard: https://wandb.ai/lareliquia-angulo/neurochimera-full-experiments-performance
  - [ ] All 6 experiments visible
  - [ ] Benchmark results uploaded as artifacts

- [ ] Make projects public:
  - [ ] Visit project settings
  - [ ] Change visibility to "Public"
  - [ ] Enable "Allow anyone to view"

- [ ] Copy dashboard URLs for paper

**Status**: ⏳ Pending
**URL**: https://wandb.ai/lareliquia-angulo

---

### 3.2 Zenodo ✓
**Priority**: HIGH | **Automated**: DRAFT | **DOI**: YES

- [ ] Run upload script:
  ```bash
  python publish/upload_all_platforms.py
  ```

- [ ] Review draft deposition at: https://zenodo.org/me/uploads

- [ ] Verify uploaded files:
  - [ ] dataset_all.zip (complete dataset)
  - [ ] NeuroCHIMERA_Paper.html
  - [ ] benchmark_results_*.zip

- [ ] Review metadata:
  - [ ] Title: NeuroCHIMERA: Consciousness Emergence as Phase Transition...
  - [ ] Authors: V.F. Veselov, Francisco Angulo de Lafuente
  - [ ] Keywords: consciousness, phase transition, neuromorphic, GPU, benchmark
  - [ ] License: CC-BY-4.0
  - [ ] Publication date: 2025

- [ ] Publish deposition:
  - [ ] Click "Publish" button
  - [ ] Confirm publication (irreversible)
  - [ ] **Copy DOI** (e.g., 10.5281/zenodo.XXXXX)

- [ ] Update DOI in:
  - [ ] README.md
  - [ ] CITATION.md
  - [ ] Paper
  - [ ] publish/update_readme_badges.py

**Status**: ⏳ Pending
**URL**: https://zenodo.org/me/uploads
**DOI**: `10.5281/zenodo.17872411` (update after publish)

---

### 3.3 Open Science Framework (OSF) ✓
**Priority**: MEDIUM | **Automated**: PARTIAL | **DOI**: YES

- [ ] Run upload script (already creates project):
  ```bash
  python publish/upload_all_platforms.py
  ```

- [ ] Note project ID from output

- [ ] Upload files manually:
  ```bash
  pip install osfclient
  export OSF_TOKEN="KSAPimE65LQJ648xovRICXTSKHSnQT2xRgunNM1QHf6tu3eI81x1Z7b0vHduNJFTFgVKhL"

  # Upload dataset
  osf -p <PROJECT_ID> upload release/dataset_all.zip /data/

  # Upload benchmarks
  osf -p <PROJECT_ID> upload release/benchmarks/ /benchmarks/

  # Upload paper
  osf -p <PROJECT_ID> upload NeuroCHIMERA_Paper.html /paper/
  ```

- [ ] Verify project:
  - [ ] Visit https://osf.io/<PROJECT_ID>
  - [ ] Check all files uploaded
  - [ ] Project is public
  - [ ] Metadata complete

- [ ] Create OSF Preprint (optional):
  - [ ] Visit https://osf.io/preprints/
  - [ ] Upload paper
  - [ ] Link to OSF project

- [ ] Copy project URL for citations

**Status**: ⏳ Pending
**URL**: https://osf.io/ (project URL TBD)
**Token**: KSAPimE65LQJ648xovRICXTSKHSnQT2xRgunNM1QHf6tu3eI81x1Z7b0vHduNJFTFgVKhL

---

## Phase 4: Manual Platform Uploads

### 4.1 Figshare 📤
**Priority**: MEDIUM | **Manual**: YES | **DOI**: YES

**Credentials**:
- Username: `5292188`
- Password: `$GNJmzWHcQL6XSS`

**Option A - Web Upload**:
- [ ] Visit https://figshare.com/
- [ ] Click "Upload" → "Create new item"
- [ ] Select "Dataset"
- [ ] Upload files:
  - [ ] release/dataset_all.zip
  - [ ] release/benchmarks/*.json (as ZIP)
  - [ ] NeuroCHIMERA_Paper.html
- [ ] Add metadata:
  - [ ] Title: NeuroCHIMERA Consciousness Emergence Experiments
  - [ ] Authors: V.F. Veselov, Francisco Angulo de Lafuente
  - [ ] Categories: Artificial Intelligence, Neuroscience, Computer Science
  - [ ] Keywords: consciousness, phase transition, neuromorphic, GPU, benchmark
  - [ ] Description: [Copy from paper abstract]
  - [ ] License: CC-BY-4.0
  - [ ] Funding: [If applicable]
- [ ] Add references:
  - [ ] Link to GitHub repository
  - [ ] Link to Zenodo DOI
- [ ] Publish
- [ ] **Copy Figshare DOI**
- [ ] Update URLs in badges script

**Option B - FTP Upload**:
```bash
lftp -u 5292188,'$GNJmzWHcQL6XSS' ftp://figshare.com
cd uploads
put release/dataset_all.zip
put release/benchmarks/*.json
bye
```

**Status**: ⏳ Pending
**URL**: https://figshare.com/ (item URL TBD)
**FTP**: ftp://figshare.com

---

### 4.2 OpenML 📤
**Priority**: LOW | **Manual**: YES | **DOI**: NO

- [ ] Prepare exports:
  ```bash
  python publish/upload_all_platforms.py
  # Creates release/openml_export/
  ```

- [ ] Visit https://www.openml.org/
- [ ] Login to account

For each experiment (1-6):
- [ ] Experiment 1: Spacetime Emergence
  - [ ] Click "Upload" → "Dataset"
  - [ ] Upload ARFF file from `release/openml_export/`
  - [ ] Add metadata:
    - Name: NeuroCHIMERA Experiment 1 - Spacetime Emergence
    - Description: Space-time emergence from computational network
    - Format: ARFF
    - Tags: consciousness, neuromorphic, GPU, spacetime, phase-transition
    - Default target: [if applicable]
  - [ ] Submit

- [ ] Experiment 2: Consciousness Emergence
  - [ ] [Repeat process]

- [ ] Experiment 3: Genesis 1
  - [ ] [Repeat process]

- [ ] Experiment 4: Genesis 2
  - [ ] [Repeat process]

- [ ] Experiment 5: Benchmark 1
  - [ ] [Repeat process]

- [ ] Experiment 6: Benchmark 2
  - [ ] [Repeat process]

- [ ] Copy OpenML dataset IDs
- [ ] Update URLs in badges script

**Status**: ⏳ Pending
**URL**: https://www.openml.org/

---

### 4.3 DataHub 📤
**Priority**: LOW | **Manual**: YES | **DOI**: NO

- [ ] Prepare package:
  ```bash
  python publish/upload_all_platforms.py
  # Creates release/datahub_export/
  ```

- [ ] Install DataHub CLI:
  ```bash
  npm install -g data-cli
  ```

- [ ] Login:
  ```bash
  data login
  # Follow prompts
  ```

- [ ] Publish dataset:
  ```bash
  cd release/datahub_export
  data push neurochimera-experiments
  ```

- [ ] Verify publication:
  - [ ] Visit https://datahub.io/neurochimera-experiments
  - [ ] Check datapackage.json rendered correctly
  - [ ] All resources accessible
  - [ ] Preview working

- [ ] Copy DataHub URL
- [ ] Update badges script

**Alternative - Web Upload**:
- [ ] Visit https://datahub.io/
- [ ] Click "Publish Data"
- [ ] Upload `datapackage.json`
- [ ] Confirm and publish

**Status**: ⏳ Pending
**URL**: https://datahub.io/

---

### 4.4 Academia.edu 📤
**Priority**: LOW | **Manual**: YES | **DOI**: NO

- [ ] Prepare export:
  ```bash
  python publish/upload_all_platforms.py
  # Creates release/academia_export/
  ```

- [ ] Visit https://www.academia.edu/
- [ ] Login to account

- [ ] Upload paper:
  - [ ] Click "Upload" in top menu
  - [ ] Upload file: `release/academia_export/NeuroCHIMERA_Paper.html`
  - [ ] Add metadata:
    - [ ] Title: NeuroCHIMERA: Consciousness Emergence as Phase Transition in Neuromorphic GPU-Native Computing
    - [ ] Authors: V.F. Veselov, Francisco Angulo de Lafuente
    - [ ] Co-authors: [Add if applicable]
    - [ ] Abstract: [Copy from paper]
    - [ ] Keywords: consciousness, phase transition, neuromorphic computing, GPU, artificial consciousness
    - [ ] Discipline: Computer Science → Artificial Intelligence
    - [ ] Subdiscipline: Computational Neuroscience
  - [ ] Upload supplementary:
    - [ ] File: `release/academia_export/supplementary_materials.zip`
    - [ ] Type: "Supplemental Material"
  - [ ] Add external links:
    - [ ] GitHub repository
    - [ ] Zenodo DOI
    - [ ] W&B dashboard
  - [ ] Publish

- [ ] Update profile:
  - [ ] Add research interests: consciousness, AI, neuromorphic computing
  - [ ] Link to Gravatar: https://gravatar.com/ecofa

- [ ] Copy Academia.edu paper URL

**Status**: ⏳ Pending
**URL**: https://www.academia.edu/

---

### 4.5 DrivenData (Optional) 📤
**Priority**: OPTIONAL | **Manual**: YES | **DOI**: NO

This is optional but highly recommended for community engagement.

- [ ] Visit https://www.drivendata.org/contact/

- [ ] Propose challenge:
  - [ ] Challenge title: "NeuroCHIMERA Consciousness Emergence Benchmark"
  - [ ] Description: Reproduce and improve consciousness emergence experiments
  - [ ] Dataset: Link to Zenodo
  - [ ] Baseline: Provide current benchmark results
  - [ ] Evaluation metrics:
    - Accuracy of spacetime emergence prediction
    - Consciousness metric (Φ) correlation
    - Computational efficiency
    - Novel insights
  - [ ] Timeline: 3-6 months
  - [ ] Prize: Co-authorship, community recognition

- [ ] Provide materials:
  - [ ] Dataset
  - [ ] Baseline code
  - [ ] Evaluation script
  - [ ] Documentation

- [ ] Set up leaderboard

**Status**: ⏳ Pending (Optional)
**URL**: https://www.drivendata.org/users/Francisco_Angulo_Lafuente/

---

## Phase 5: Automatic Platforms

### 5.1 OpenAIRE ✓
**Priority**: MEDIUM | **Automatic**: YES | **DOI**: NO

This happens automatically via Zenodo integration.

- [ ] Wait 24-48 hours after Zenodo publication

- [ ] Verify indexing:
  - [ ] Visit https://explore.openaire.eu/
  - [ ] Search for "NeuroCHIMERA"
  - [ ] Verify dataset appears
  - [ ] Check metadata accuracy

- [ ] If not indexed after 48h:
  - [ ] Contact OpenAIRE support
  - [ ] Verify Zenodo metadata is complete
  - [ ] Check OpenAIRE eligibility guidelines

**Status**: ⏳ Pending (automatic after Zenodo)
**URL**: https://explore.openaire.eu/

---

## Phase 6: Documentation & Citations

### 6.1 Update Repository Documentation

- [ ] Update README.md:
  - [ ] Add badge section (run `python publish/update_readme_badges.py`)
  - [ ] Add DOIs
  - [ ] Add platform links
  - [ ] Update citation instructions

- [ ] Update CITATION.md:
  - [ ] Add Zenodo DOI
  - [ ] Add Figshare DOI (if applicable)
  - [ ] Add OSF project ID
  - [ ] Update all citation formats

- [ ] Create DATASETS.md:
  - [ ] List all 6 experiments
  - [ ] Link to each platform
  - [ ] Include DOIs
  - [ ] Add download instructions

### 6.2 Add Badges

Run badge updater script:
```bash
python publish/update_readme_badges.py
```

Update platform IDs in script, then re-run to add to README.

### 6.3 Update Paper

- [ ] Add acknowledgments:
  - [ ] Platform acknowledgments (Zenodo, W&B, etc.)
  - [ ] Dataset availability statement
  - [ ] Link to all platforms

- [ ] Add Data Availability section:
  ```
  All data and code are publicly available:
  - Code: https://github.com/Agnuxo1/...
  - Dataset: https://doi.org/10.5281/zenodo.XXXXX
  - Benchmarks: https://wandb.ai/lareliquia-angulo/...
  - Supplementary: https://osf.io/XXXXX
  ```

---

## Phase 7: CI/CD & Automation

### 7.1 GitHub Actions

- [ ] Verify workflow file exists: `.github/workflows/benchmarks.yml`

- [ ] Add repository secrets:
  - [ ] `WANDB_API_KEY`: b017394dfb1bfdbcaf122dcd20383d5ac9cb3bae
  - [ ] `ZENODO_TOKEN`: lDYsHSupjRQXYxMAMihKn5lQwamqnsBliy0kwXbdUBg4VmxxuePbXxCpq2iw
  - [ ] `OSF_TOKEN`: KSAPimE65LQJ648xovRICXTSKHSnQT2xRgunNM1QHf6tu3eI81x1Z7b0vHduNJFTFgVKhL

- [ ] Test workflow manually:
  - [ ] Go to Actions tab
  - [ ] Select "Scheduled Benchmarks"
  - [ ] Click "Run workflow"
  - [ ] Monitor execution
  - [ ] Verify artifacts uploaded
  - [ ] Check W&B for new runs

- [ ] Schedule weekly runs:
  - [ ] Workflow runs every Monday at 04:00 UTC (already configured)
  - [ ] Monitor first few runs
  - [ ] Adjust schedule if needed

### 7.2 Automated Updates

Create GitHub Action to update badges automatically (optional):
```yaml
name: Update Badges
on:
  schedule:
    - cron: '0 0 1 * *'  # Monthly
  workflow_dispatch: {}
jobs:
  update-badges:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Update badges
        run: python publish/update_readme_badges.py
      - name: Commit changes
        run: |
          git config --local user.email "action@github.com"
          git config --local user.name "GitHub Action"
          git add README.md CITATION.md
          git commit -m "Update badges and citations" || exit 0
          git push
```

---

## Phase 8: Verification & Testing

### 8.1 Download & Reproduce Test

Have external collaborator test reproducibility:

- [ ] Clone repository
- [ ] Follow README instructions
- [ ] Download dataset from Zenodo
- [ ] Run benchmarks
- [ ] Compare results with published benchmarks
- [ ] Document any issues

### 8.2 Link Verification

Test all public links:
- [ ] W&B dashboard accessible
- [ ] Zenodo DOI resolves
- [ ] OSF project loads
- [ ] Figshare dataset downloadable
- [ ] OpenML datasets visible
- [ ] DataHub package renders
- [ ] Academia.edu paper viewable
- [ ] GitHub repository public
- [ ] All badges link correctly

### 8.3 Citation Test

- [ ] Test BibTeX citation imports to reference manager
- [ ] Verify DOI citations work in LaTeX
- [ ] Test APA format in Word
- [ ] Check IEEE format

---

## Phase 9: Community Outreach

### 9.1 Announce Publication

- [ ] Twitter/X:
  ```
  Excited to share NeuroCHIMERA - our work on consciousness emergence
  as phase transition in GPU-native neuromorphic computing! 🧠⚡

  📊 Interactive benchmarks: [W&B URL]
  📦 Open dataset: [Zenodo DOI]
  💻 Code: [GitHub URL]
  📄 Paper: [Paper URL]

  All experiments fully reproducible! #AI #Consciousness #Neuroscience
  ```

- [ ] Reddit:
  - [ ] r/MachineLearning
  - [ ] r/artificial
  - [ ] r/Neuroscience
  - [ ] r/compsci

- [ ] LinkedIn:
  - [ ] Professional announcement
  - [ ] Link to all platforms
  - [ ] Highlight reproducibility

- [ ] Hacker News:
  - [ ] Submit GitHub repository
  - [ ] Submit paper
  - [ ] Engage with comments

### 9.2 Email Researchers

Contact researchers in:
- Consciousness studies
- Neuromorphic computing
- GPU computing
- Phase transitions in complex systems

Template:
```
Subject: NeuroCHIMERA: Open Dataset for Consciousness Emergence Research

Dear [Name],

I wanted to share our recent work on consciousness emergence as a phase
transition in neuromorphic computing that might interest you.

We've made everything fully open and reproducible:
- Dataset with DOI: [Zenodo]
- Interactive benchmarks: [W&B]
- Complete code: [GitHub]

All 6 experiments are documented and can be reproduced locally.

Best regards,
Francisco Angulo de Lafuente
```

### 9.3 Submit to Conferences/Workshops

- [ ] NeurIPS (Datasets & Benchmarks track)
- [ ] ICML (Benchmarks track)
- [ ] ICLR
- [ ] Consciousness-related conferences
- [ ] Neuromorphic computing workshops

---

## Phase 10: Maintenance

### 10.1 Regular Updates

Monthly tasks:
- [ ] Review W&B metrics for anomalies
- [ ] Check download counts on platforms
- [ ] Respond to issues/questions on GitHub
- [ ] Monitor citation counts
- [ ] Update documentation as needed

### 10.2 Version Control

When making updates:
- [ ] Increment version number
- [ ] Update changelog
- [ ] Create new Zenodo version (links to previous)
- [ ] Update all platform references
- [ ] Re-run benchmarks to verify compatibility

---

## Summary Statistics

### Platforms Checklist
- [ ] W&B (Automated) ✓
- [ ] Zenodo (Automated draft) ✓
- [ ] OSF (Semi-automated) ✓
- [ ] Figshare (Manual) 📤
- [ ] OpenML (Manual) 📤
- [ ] DataHub (Manual) 📤
- [ ] Academia.edu (Manual) 📤
- [ ] DrivenData (Optional) 📤
- [ ] OpenAIRE (Automatic) ✓

**Total**: 9 platforms (8 required, 1 optional)

### Expected Reach
- **Immediate**: GitHub, W&B (developers, ML engineers)
- **1 week**: Zenodo, OSF (researchers, academics)
- **1 month**: Figshare, OpenML, DataHub (data scientists)
- **Ongoing**: OpenAIRE, Academia.edu (broad academic community)
- **Long-term**: DrivenData (community challenges, innovation)

### Time Estimate
- Phase 1-2 (Prep & Benchmarks): 2-4 hours
- Phase 3 (Automated uploads): 30 minutes
- Phase 4 (Manual uploads): 2-3 hours
- Phase 5-10 (Docs, CI/CD, outreach): 2-4 hours

**Total**: 1-2 days of focused work

---

## Quick Reference

### Credentials
```
W&B API: b017394dfb1bfdbcaf122dcd20383d5ac9cb3bae
Zenodo Token: lDYsHSupjRQXYxMAMihKn5lQwamqnsBliy0kwXbdUBg4VmxxuePbXxCpq2iw
OSF Token: KSAPimE65LQJ648xovRICXTSKHSnQT2xRgunNM1QHf6tu3eI81x1Z7b0vHduNJFTFgVKhL
Figshare User: 5292188
Figshare Pass: $GNJmzWHcQL6XSS
```

### Key Commands
```bash
# Full pipeline
python publish/run_and_publish_benchmarks.py

# Individual steps
python publish/audit_experiments.py
python publish/run_benchmarks.py
python publish/create_public_dashboards.py
python publish/upload_all_platforms.py
python publish/update_readme_badges.py

# Single experiment
python publish/run_experiment.py --exp 1
```

### Important URLs
```
GitHub: https://github.com/Agnuxo1/Consciousness-Emergence-as-Phase-Transition-in-GPU-Native-Neuromorphic-Computing
W&B: https://wandb.ai/lareliquia-angulo
Zenodo: https://zenodo.org/me/uploads
OSF: https://osf.io/
Figshare: https://figshare.com/
OpenML: https://www.openml.org/
DataHub: https://datahub.io/
Academia: https://www.academia.edu/
DrivenData: https://www.drivendata.org/users/Francisco_Angulo_Lafuente/
Gravatar: https://gravatar.com/ecofa
```

---

**Last Updated**: 2025-12-09
**Version**: 1.0
**Status**: Ready for execution
