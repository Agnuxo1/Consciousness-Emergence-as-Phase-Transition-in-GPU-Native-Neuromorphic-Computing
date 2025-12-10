# NeuroCHIMERA Multi-Platform Publication Guide

Complete guide for publishing experiments, benchmarks, datasets, and paper across all scientific platforms.

## Quick Start

```bash
# Run comprehensive upload to all platforms
python publish/upload_all_platforms.py

# Or upload to specific platforms individually
python publish/wandb_upload.py
python publish/zenodo_upload.py
```

## Platforms Overview

### 1. Weights & Biases (W&B) ✓ AUTOMATED
**Purpose**: Live experiment tracking, benchmark visualization, interactive dashboards

**Access**: https://wandb.ai/lareliquia-angulo

**What gets uploaded**:
- Real-time experiment metrics
- Benchmark results as artifacts
- Interactive comparison dashboards
- Model performance visualizations

**Benefits**:
- Public dashboards for community viewing
- Automatic versioning
- Collaborative experiment tracking
- Integration with CI/CD

**Status**: Fully automated via `upload_all_platforms.py`

---

### 2. Zenodo ✓ AUTOMATED (Draft)
**Purpose**: Permanent DOI, long-term archival, dataset hosting

**Access**: https://zenodo.org/me/uploads

**What gets uploaded**:
- Complete dataset (dataset_all.zip)
- Benchmark results
- Paper HTML
- Code repository snapshot

**Benefits**:
- Permanent DOI for citations
- CERN-backed long-term preservation
- Automatic OpenAIRE integration
- Versioning support

**Status**: Automated draft creation - requires manual publish

**Manual steps**:
1. Visit https://zenodo.org/me/uploads
2. Find draft deposition
3. Review metadata
4. Click "Publish"
5. Copy DOI for paper citations

---

### 3. Open Science Framework (OSF) ✓ AUTOMATED
**Purpose**: Research workflow management, preprints, project documentation

**Access**: https://osf.io/

**Token**: KSAPimE65LQJ648xovRICXTSKHSnQT2xRgunNM1QHf6tu3eI81x1Z7b0vHduNJFTFgVKhL

**What gets uploaded**:
- Project overview
- Experiment documentation
- Links to datasets
- Preregistration (optional)

**Benefits**:
- Complete research lifecycle management
- Preprint hosting
- Collaboration tools
- Integration with other services

**Status**: Project creation automated - file upload requires manual steps or osf-cli

**Manual steps**:
```bash
# Install OSF CLI
pip install osfclient

# Upload files
osf -p <project_id> upload release/benchmarks/ /benchmarks/
osf -p <project_id> upload NeuroCHIMERA_Paper.html /paper/
```

---

### 4. Figshare ⚠ MANUAL
**Purpose**: Dataset sharing, visualizations, posters, presentations

**Access**: https://figshare.com/

**FTP Credentials**:
- Username: 5292188
- Password: $GNJmzWHcQL6XSS

**What to upload**:
- Dataset files
- Benchmark visualizations
- Presentation slides
- Supplementary figures

**Benefits**:
- DOI for datasets
- Good for media-rich content
- Discipline-specific collections
- Citation tracking

**Manual upload options**:

**Option A - Web Interface**:
1. Visit https://figshare.com/
2. Click "Upload"
3. Drag dataset_all.zip
4. Add metadata (title, authors, description, tags)
5. Publish

**Option B - FTP**:
```bash
# Using lftp
lftp -u 5292188,'$GNJmzWHcQL6XSS' ftp://figshare.com
cd uploads
put release/dataset_all.zip
put release/benchmarks/*.json
bye
```

**Option C - API**:
```bash
pip install figshare
# Use Python API with credentials above
```

---

### 5. OpenML ⚠ MANUAL
**Purpose**: Machine learning experiment registry, reproducible ML

**Access**: https://www.openml.org/auth/profile-page

**What to upload**:
- Benchmark datasets in ARFF format
- Experiment configurations
- Model performance metrics

**Benefits**:
- ML-specific platform
- Automatic benchmark comparisons
- Community challenges
- Reproducibility tools

**Preparation**: Run `upload_all_platforms.py` to generate OpenML-compatible exports

**Manual steps**:
1. Visit https://www.openml.org/
2. Login to your account
3. Click "Upload" → "Dataset"
4. Use files from `release/openml_export/`
5. Add metadata:
   - Name: NeuroCHIMERA Experiment N
   - Description: [Copy from metadata file]
   - Tags: consciousness, neuromorphic, GPU, benchmark
6. Repeat for all 6 experiments

---

### 6. DataHub ⚠ MANUAL
**Purpose**: Data package repository, version control for data

**Access**: https://datahub.io/

**What to upload**:
- Data packages (datapackage.json)
- CSV/JSON datasets
- Benchmark results

**Benefits**:
- Git-like versioning for data
- Automatic validation
- Data quality checks
- Public data portal

**Preparation**: Run `upload_all_platforms.py` to generate datapackage.json

**Manual steps**:
```bash
# Install DataHub CLI
npm install -g data-cli

# Login
data login

# Publish
cd release/datahub_export
data push
```

---

### 7. Academia.edu ⚠ MANUAL
**Purpose**: Academic social network, paper sharing, researcher profiles

**Access**: https://www.academia.edu/

**Gravatar**: https://gravatar.com/ecofa

**What to upload**:
- Paper HTML
- Supplementary materials
- Research interests
- Profile updates

**Benefits**:
- Academic networking
- Paper view statistics
- Researcher discovery
- Citation notifications

**Manual steps**:
1. Visit https://www.academia.edu/
2. Click "Upload"
3. Upload `release/academia_export/NeuroCHIMERA_Paper.html`
4. Add metadata:
   - Title: NeuroCHIMERA: Consciousness Emergence as Phase Transition in Neuromorphic GPU-Native Computing
   - Authors: V.F. Veselov, Francisco Angulo de Lafuente
   - Abstract: [Copy from paper]
   - Keywords: consciousness, phase transition, neuromorphic, GPU
5. Upload supplementary_materials.zip as "Additional Files"
6. Publish

---

### 8. OpenAIRE ✓ AUTOMATIC
**Purpose**: European research infrastructure, aggregation

**Access**: https://explore.openaire.eu/

**What happens**:
- Automatically harvests from Zenodo
- Indexes your datasets and papers
- Connects to EU research funding
- Provides additional metrics

**Benefits**:
- European research visibility
- Funding compliance
- Cross-repository search
- Impact metrics

**Status**: Fully automatic once Zenodo is published

**No action required** - OpenAIRE will automatically index your Zenodo uploads within 24-48 hours.

---

### 9. DrivenData ⚠ MANUAL
**Purpose**: Data science competitions, social impact challenges

**Access**: https://www.drivendata.org/users/Francisco_Angulo_Lafuente/

**What to do**:
- Create consciousness benchmark challenge
- Invite community to reproduce experiments
- Compare approaches

**Benefits**:
- Competitive benchmarking
- Community engagement
- Novel approaches
- Visibility

**Manual steps**:
1. Contact DrivenData to propose challenge
2. Provide dataset and baseline benchmarks
3. Define evaluation metrics
4. Set up leaderboard

---

### 10. Signate & Zindi ⚠ OPTIONAL
**Purpose**: Regional ML competition platforms (Japan, Africa)

**Signate**: https://signate.jp/
**Zindi**: https://zindi.africa/

**Benefits**:
- Regional research communities
- Diverse perspectives
- Global collaboration

**Use if**: You want maximum global reach for benchmark challenges

---

## Automated Workflow

### Complete Upload Sequence

```bash
# 1. Ensure all experiments have run
python publish/run_benchmarks.py

# 2. Upload to all automated platforms
python publish/upload_all_platforms.py

# 3. Review generated report
cat release/upload_report_*.json

# 4. Complete manual uploads (see sections above)
```

### CI/CD Integration

The GitHub Actions workflow `.github/workflows/benchmarks.yml` automatically:
- Runs benchmarks weekly
- Uploads to W&B
- Creates artifacts
- Generates reports

**To enable**:
1. Add secrets to GitHub repository:
   - `WANDB_API_KEY`
   - `ZENODO_TOKEN`
   - `OSF_TOKEN`

2. Workflow runs automatically every Monday at 04:00 UTC

3. Manual trigger: GitHub → Actions → "Scheduled Benchmarks" → "Run workflow"

---

## Platform Comparison Matrix

| Platform | Automated | DOI | Versioning | API | Best For |
|----------|-----------|-----|------------|-----|----------|
| W&B | ✓ | ✗ | ✓ | ✓ | Live experiments |
| Zenodo | ✓* | ✓ | ✓ | ✓ | Permanent archive |
| OSF | ✓* | ✓ | ✓ | ✓ | Research workflow |
| Figshare | ✗ | ✓ | ✓ | ✓ | Rich media |
| OpenML | ✗ | ✗ | ✓ | ✓ | ML benchmarks |
| DataHub | ✗ | ✗ | ✓ | ✓ | Data packages |
| Academia | ✗ | ✗ | ✗ | ✗ | Networking |
| OpenAIRE | Auto | ✗ | ✗ | ✓ | EU visibility |

*Requires manual publish step

---

## Complete Checklist

### Phase 1: Automated Uploads ✓
- [ ] Run `python publish/upload_all_platforms.py`
- [ ] Verify W&B dashboard: https://wandb.ai/lareliquia-angulo
- [ ] Check Zenodo draft created
- [ ] Confirm OSF project created

### Phase 2: Manual Publications
- [ ] Publish Zenodo deposition (get DOI)
- [ ] Upload to Figshare via web or FTP
- [ ] Submit datasets to OpenML
- [ ] Publish to DataHub using CLI
- [ ] Upload paper to Academia.edu
- [ ] Update profile and add to Gravatar

### Phase 3: Competition Platforms (Optional)
- [ ] Consider DrivenData challenge
- [ ] Evaluate Signate for Japanese audience
- [ ] Evaluate Zindi for African audience

### Phase 4: Verification
- [ ] Verify OpenAIRE indexing (24-48h after Zenodo)
- [ ] Check all DOIs resolve correctly
- [ ] Test dataset downloads from each platform
- [ ] Verify public visibility of all uploads

### Phase 5: Documentation Updates
- [ ] Add DOIs to README.md
- [ ] Update paper with Zenodo badge
- [ ] Add platform links to repository
- [ ] Create citation guidance

---

## Platform URLs Summary

Quick reference for all platforms:

```
W&B Dashboard:      https://wandb.ai/lareliquia-angulo
Zenodo Upload:      https://zenodo.org/me/uploads
OSF Projects:       https://osf.io/myprojects
Figshare Upload:    https://figshare.com/account/home
OpenML Profile:     https://www.openml.org/auth/profile-page
DataHub:            https://datahub.io/
Academia.edu:       https://www.academia.edu/
DrivenData:         https://www.drivendata.org/users/Francisco_Angulo_Lafuente/
Gravatar:           https://gravatar.com/ecofa
OpenAIRE:           https://explore.openaire.eu/
```

---

## Troubleshooting

### Upload fails with authentication error
- Check API keys in environment variables
- Verify tokens haven't expired
- Review platform-specific credentials

### Dataset too large
- Use compression (already done for dataset_all.zip)
- Split into multiple parts if needed
- Consider Zenodo's 50GB limit per file

### DOI not generated
- Ensure Zenodo deposition is published (not draft)
- Wait up to 24 hours for DOI minting
- Check deposition status page

### OpenAIRE not indexing
- Verify Zenodo upload is public
- Wait 48 hours for harvesting
- Check OpenAIRE guidelines compatibility

---

## Support Contacts

- W&B: support@wandb.ai
- Zenodo: info@zenodo.org
- OSF: support@osf.io
- Figshare: support@figshare.com
- OpenML: https://www.openml.org/contact

---

## Citation Template

Once DOIs are generated, use this citation format:

```
Veselov, V.F., & Angulo de Lafuente, F. (2025).
NeuroCHIMERA: Consciousness Emergence as Phase Transition in
Neuromorphic GPU-Native Computing [Dataset].
Zenodo. https://doi.org/10.5281/zenodo.XXXXX

Benchmark results available at: https://wandb.ai/lareliquia-angulo
Code repository: https://github.com/Agnuxo1/Consciousness-Emergence-as-Phase-Transition-in-GPU-Native-Neuromorphic-Computing
```

---

Last updated: 2025-12-09
