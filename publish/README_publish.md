Publication and Automated Upload Instructions
===========================================

This folder contains automation scripts and instructions to publish the NeuroCHIMERA paper, datasets,
slides and experimental artifacts to common open-science platforms (Zenodo, W&B, Figshare, OpenML,
OSF, DataHub). The scripts are templates that use API tokens supplied via environment variables or
GitHub Actions secrets. Do NOT commit API keys to the repository.

High-level workflow
- Prepare release assets in `release/` (manuscript PDF, HTML, slides.zip, dataset.zip, models/)
- Create a Git tag and GitHub Release (recommended). The GitHub release can be automatically linked
  to Zenodo for DOI minting (see Zenodo docs).
- Use the included scripts to upload artifacts to W&B, Zenodo, Figshare, OpenML and OSF. Each script
  expects a token in an environment variable; see the examples below.

Required environment variables (examples)
- `WANDB_API_KEY` — W&B API key
- `ZENDO_TOKEN` — Zenodo API token
- `FIGSHARE_TOKEN` — Figshare personal token
- `OPENML_API_KEY` — OpenML API key
- `OSF_TOKEN` — OSF personal access token
- `FTP_USER`, `FTP_PASS` — FTP credentials (if using FTP uploads)

Quick start (local)
1. Create the release assets (zip slides, datasets):
   ```powershell
   cd "D:\Experiment_Genesis_Veselov-Angulo"
   mkdir release
   Compress-Archive -Path .\Imagen\* -DestinationPath release\slides.zip -Force
   Compress-Archive -Path .\1-2\* -DestinationPath release\dataset_1-2.zip -Force
   Compress-Archive -Path .\3-4\* -DestinationPath release\dataset_3-4.zip -Force
   Compress-Archive -Path .\5-6\* -DestinationPath release\dataset_5-6.zip -Force
   ```

2. Populate secrets (locally via environment variables) and run an uploader script, for example:
   ```powershell
   $env:WANDB_API_KEY = 'YOUR_KEY'
   python .\publish\upload_to_wandb.py --project lareliquia-angulo --artifact release\slides.zip --name neurochimera-slides
   ```

3. For automated CI: add the required secrets to GitHub repository `Settings -> Secrets` with the
   names shown above. A sample GitHub Actions workflow `.github/workflows/publish.yml` is included.

Safety notes
- Never commit tokens to the repository. Use GitHub Secrets for CI.
- Verify dataset licensing before public release.

If you want me to run uploads, add the tokens to GitHub Secrets and confirm which platforms I should publish to.
