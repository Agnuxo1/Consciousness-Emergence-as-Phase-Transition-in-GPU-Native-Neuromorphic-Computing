Storing API keys and secrets (GitHub Secrets)
============================================

For the automated publishing workflow to run securely, add the following secrets to your GitHub repository
(`Settings -> Secrets -> Actions`):

- `WANDB_API_KEY` — Weights & Biases API key
- `ZENDO_TOKEN` — Zenodo personal access token (if using API)
- `FIGSHARE_TOKEN` — Figshare token
- `OPENML_API_KEY` — OpenML API key
- `OSF_TOKEN` — OSF personal access token (optional)
- `FTP_USER`, `FTP_PASS` — FTP credentials (optional)

How to add secrets
1. Go to your repository on GitHub.
2. Click `Settings` -> `Secrets` -> `Actions` -> `New repository secret`.
3. Add the secret name and value, and save.

Do NOT commit tokens to the repository or paste them into issues or PRs.
