#!/usr/bin/env python3
"""Upload an artifact (dataset or model) to Weights & Biases.

This script expects `WANDB_API_KEY` in the environment or configured via `wandb login`.
Usage:
    python upload_to_wandb.py --project lareliquia-angulo --artifact release/slides.zip --name neurochimera-slides
"""
import argparse
import os
import wandb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', required=True)
    parser.add_argument('--entity', default=None)
    parser.add_argument('--artifact', required=True, help='Path to file to upload')
    parser.add_argument('--name', required=True, help='Artifact name')
    parser.add_argument('--type', default='dataset', help='Artifact type (dataset/model/code)')
    args = parser.parse_args()

    api_key = os.getenv('WANDB_API_KEY')
    if not api_key:
        raise SystemExit('WANDB_API_KEY not found in environment. Set it or run `wandb login`.')

    wandb.login(key=api_key)
    run = wandb.init(project=args.project, entity=args.entity or None, job_type='publish')
    artifact = wandb.Artifact(args.name, type=args.type)
    artifact.add_file(args.artifact)
    run.log_artifact(artifact)
    run.finish()
    print('Uploaded', args.artifact, 'to W&B project', args.project)


if __name__ == '__main__':
    main()
