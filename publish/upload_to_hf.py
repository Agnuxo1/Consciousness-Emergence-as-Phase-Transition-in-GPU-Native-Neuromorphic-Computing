#!/usr/bin/env python3
"""Upload a repository/dataset to Hugging Face Hub.

This script expects `HF_TOKEN` in the environment. It can create a repo (dataset or model) and push files.

Usage:
    python upload_to_hf.py --repo-id Agnuxo/neurochimera-dataset --repo-type dataset --path release --commit "Initial upload"
"""
import argparse
import os
from pathlib import Path
from huggingface_hub import HfApi, Repository, create_repo


def repo_exists(api, repo_id, repo_type):
    try:
        api.repo_info(repo_id)
        return True
    except Exception:
        return False


def create_and_push_repo(repo_id, repo_type, path, token, commit_message="Initial upload"):
    api = HfApi()
    if not repo_exists(api, repo_id, repo_type):
        create_repo(repo_id, repo_type=repo_type, token=token, private=False)

    # Initialize a local repo and push
    local_repo_dir = Path('/tmp') / repo_id.split('/')[-1]
    if local_repo_dir.exists():
        # clear dir
        for p in local_repo_dir.iterdir():
            if p.is_file():
                p.unlink()
            elif p.is_dir():
                import shutil
                shutil.rmtree(p)
    else:
        local_repo_dir.mkdir(parents=True)

    # Copy files into the local repo
    from shutil import copytree, copy2
    src = Path(path)
    if src.is_file():
        copy2(src, local_repo_dir / src.name)
    elif src.is_dir():
        # copy contents
        for item in src.iterdir():
            if item.is_file():
                copy2(item, local_repo_dir / item.name)
            elif item.is_dir():
                copytree(item, local_repo_dir / item.name)

    repo = Repository(local_repo_dir, clone_from=repo_id, token=token)
    repo.git_add(".")
    repo.git_commit(commit_message)
    repo.git_push()

    return api.repo_info(repo_id, token=token)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo-id', required=True, help='Hugging Face repo id (e.g. user/repo)')
    parser.add_argument('--repo-type', default='dataset', choices=['dataset', 'model', 'space', 'dataset-dv'], help='Repo type')
    parser.add_argument('--path', required=True, help='Path to local file or dir to upload')
    parser.add_argument('--commit', default='Initial upload', help='Commit message')

    args = parser.parse_args()

    hf_token = os.getenv('HF_TOKEN') or os.getenv('HUGGINGFACE_TOKEN')
    if not hf_token:
        raise SystemExit('HF_TOKEN not found in environment. Set HF_TOKEN or HUGGINGFACE_TOKEN env var.')

    repo_info = create_and_push_repo(args.repo_id, args.repo_type, args.path, hf_token, args.commit)
    print('Uploaded to Hugging Face:', f'https://huggingface.co/{args.repo_id}')


if __name__ == '__main__':
    main()
