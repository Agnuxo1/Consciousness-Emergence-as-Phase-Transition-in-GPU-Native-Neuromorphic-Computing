#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive platform upload script for NeuroCHIMERA experiments.
Uploads datasets, benchmarks, and paper to all configured platforms.

Platforms supported:
- Weights & Biases (W&B)
- Zenodo
- Figshare
- OpenML
- Open Science Framework (OSF)
- DataHub
- Academia.edu (manual - provides export)
- OpenAIRE (via Zenodo integration)
"""
import os
import json
import sys
import subprocess
from pathlib import Path
from datetime import datetime, timezone
import zipfile
import shutil

# Fix Windows console encoding
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Configuration from environment
WANDB_API_KEY = os.getenv('WANDB_API_KEY')
ZENODO_TOKEN = os.getenv('ZENDO_TOKEN') or os.getenv('ZENODO_TOKEN')
FIGSHARE_USERNAME = os.getenv('FIGSHARE_USERNAME')
FIGSHARE_PASSWORD = os.getenv('FIGSHARE_PASSWORD')
OSF_TOKEN = os.getenv('OSF_TOKEN')

BASE_DIR = Path(__file__).parent.parent
RELEASE_DIR = BASE_DIR / "release"
BENCHMARKS_DIR = RELEASE_DIR / "benchmarks"
PAPER_HTML = BASE_DIR / "NeuroCHIMERA_Paper.html"

class PlatformUploader:
    def __init__(self):
        self.results = {}
        self.timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')

    def upload_to_wandb(self):
        """Upload all benchmark results and datasets to W&B."""
        print("\n=== UPLOADING TO WEIGHTS & BIASES ===")
        try:
            import wandb
            if not WANDB_API_KEY:
                raise SystemExit('WANDB_API_KEY not found in environment. Set WANDB_API_KEY env var or run `wandb login`.')
            wandb.login(key=WANDB_API_KEY)

            # Create comprehensive project
            run = wandb.init(
                project="neurochimera-full-experiments",
                name=f"complete-benchmark-{self.timestamp}",
                tags=["neurochimera", "consciousness", "phase-transition", "benchmark"],
                notes="Complete NeuroCHIMERA experiment suite with 6 experiments"
            )

            # Upload all benchmark results
            for exp_result in BENCHMARKS_DIR.glob("results-exp-*.json"):
                with open(exp_result) as f:
                    data = json.load(f)
                    exp_num = data.get('experiment', 0)

                    # Log metrics if available
                    if data.get('summary'):
                        wandb.log({
                            f"exp_{exp_num}/returncode": data['returncode'],
                            f"exp_{exp_num}/timestamp": data['timestamp'],
                            **{f"exp_{exp_num}/{k}": v for k, v in data['summary'].items() if isinstance(v, (int, float))}
                        })

            # Create artifacts for datasets
            dataset_artifact = wandb.Artifact(
                name="neurochimera-datasets",
                type="dataset",
                description="Complete datasets for 6 NeuroCHIMERA experiments",
                metadata={
                    "experiments": 6,
                    "authors": "V.F. Veselov & Francisco Angulo de Lafuente",
                    "year": 2025
                }
            )

            # Add release directory
            if RELEASE_DIR.exists():
                dataset_artifact.add_dir(str(RELEASE_DIR), name="release")

            # Add benchmark results
            benchmark_artifact = wandb.Artifact(
                name="neurochimera-benchmark-results",
                type="benchmark-results",
                description=f"Benchmark results from {self.timestamp}"
            )
            benchmark_artifact.add_dir(str(BENCHMARKS_DIR), name="benchmarks")

            # Upload paper HTML
            paper_artifact = wandb.Artifact(
                name="neurochimera-paper",
                type="paper",
                description="NeuroCHIMERA: Consciousness Emergence as Phase Transition"
            )
            if PAPER_HTML.exists():
                paper_artifact.add_file(str(PAPER_HTML), name="NeuroCHIMERA_Paper.html")

            # Log artifacts
            run.log_artifact(dataset_artifact)
            run.log_artifact(benchmark_artifact)
            run.log_artifact(paper_artifact)

            # Create summary table
            table = wandb.Table(columns=["Experiment", "Status", "Description"])
            experiments = [
                (1, "Spacetime Emergence", "Space-time emergence from computational network"),
                (2, "Consciousness Emergence", "Consciousness as phase transition"),
                (3, "Genesis 1", "Genesis experiment 1"),
                (4, "Genesis 2", "Genesis experiment 2"),
                (5, "Benchmark 1", "Benchmark experiment 1"),
                (6, "Benchmark 2", "Benchmark experiment 2"),
            ]
            for exp_num, name, desc in experiments:
                status_file = BENCHMARKS_DIR / f"results-exp-{exp_num}-{self.timestamp}.json"
                status = "✓ Complete" if status_file.exists() else "⧗ Pending"
                table.add_data(exp_num, status, desc)

            run.log({"experiment_summary": table})

            run.finish()

            self.results['wandb'] = {
                'status': 'success',
                'url': f"https://wandb.ai/lareliquia-angulo/neurochimera-full-experiments",
                'run_id': run.id
            }
            print(f"✓ W&B upload complete: {self.results['wandb']['url']}")

        except Exception as e:
            self.results['wandb'] = {'status': 'error', 'error': str(e)}
            print(f"✗ W&B upload failed: {e}")

    def upload_to_zenodo(self):
        """Upload to Zenodo with proper metadata."""
        print("\n=== UPLOADING TO ZENODO ===")
        try:
            import requests

            # Create new deposition
            headers = {"Content-Type": "application/json"}
            if not ZENODO_TOKEN:
                raise SystemExit('ZENDO_TOKEN (Zenodo access token) not found in environment. Set ZENDO_TOKEN env var.')
            params = {'access_token': ZENODO_TOKEN}

            r = requests.post(
                'https://zenodo.org/api/deposit/depositions',
                params=params,
                json={},
                headers=headers
            )
            r.raise_for_status()
            deposition = r.json()
            deposition_id = deposition['id']
            bucket_url = deposition['links']['bucket']

            print(f"Created Zenodo deposition: {deposition_id}")

            # Upload dataset zip
            dataset_zip = RELEASE_DIR / "dataset_all.zip"
            if dataset_zip.exists():
                with open(dataset_zip, 'rb') as f:
                    r = requests.put(
                        f"{bucket_url}/dataset_all.zip",
                        data=f,
                        params=params
                    )
                    r.raise_for_status()
                print("✓ Uploaded dataset_all.zip")

            # Upload paper HTML
            if PAPER_HTML.exists():
                with open(PAPER_HTML, 'rb') as f:
                    r = requests.put(
                        f"{bucket_url}/NeuroCHIMERA_Paper.html",
                        data=f,
                        params=params
                    )
                    r.raise_for_status()
                print("✓ Uploaded paper HTML")

            # Create benchmark results ZIP
            benchmark_zip = BENCHMARKS_DIR / f"benchmarks_{self.timestamp}.zip"
            with zipfile.ZipFile(benchmark_zip, 'w', zipfile.ZIP_DEFLATED) as zf:
                for result_file in BENCHMARKS_DIR.glob("results-exp-*.json"):
                    zf.write(result_file, result_file.name)

            with open(benchmark_zip, 'rb') as f:
                r = requests.put(
                    f"{bucket_url}/benchmark_results_{self.timestamp}.zip",
                    data=f,
                    params=params
                )
                r.raise_for_status()
            print(f"✓ Uploaded benchmark results")

            # Add metadata
            metadata = {
                'metadata': {
                    'title': 'NeuroCHIMERA: Consciousness Emergence as Phase Transition in Neuromorphic GPU-Native Computing',
                    'upload_type': 'dataset',
                    'description': '''Complete dataset and benchmark results for NeuroCHIMERA experiments.

                    This dataset includes:
                    - 6 complete experiments on consciousness emergence
                    - Benchmark results with performance metrics
                    - Implementation code and documentation
                    - Paper HTML version

                    Authors: V.F. Veselov & Francisco Angulo de Lafuente (2025)

                    The experiments demonstrate consciousness emergence as a phase transition
                    in GPU-native neuromorphic computing systems.''',
                    'creators': [
                        {'name': 'Veselov, V.F.', 'affiliation': 'Independent Researcher'},
                        {'name': 'Angulo de Lafuente, Francisco', 'affiliation': 'Independent Researcher', 'orcid': '0000-0000-0000-0000'}
                    ],
                    'keywords': [
                        'consciousness',
                        'phase transition',
                        'neuromorphic computing',
                        'GPU computing',
                        'artificial consciousness',
                        'computational neuroscience',
                        'benchmark'
                    ],
                    'access_right': 'open',
                    'license': 'cc-by-4.0',
                    'publication_date': datetime.now().strftime('%Y-%m-%d')
                }
            }

            r = requests.put(
                f'https://zenodo.org/api/deposit/depositions/{deposition_id}',
                params=params,
                data=json.dumps(metadata),
                headers=headers
            )
            r.raise_for_status()
            print("✓ Added metadata")

            # Publish (uncomment when ready)
            # r = requests.post(
            #     f'https://zenodo.org/api/deposit/depositions/{deposition_id}/actions/publish',
            #     params=params
            # )
            # r.raise_for_status()

            self.results['zenodo'] = {
                'status': 'success',
                'deposition_id': deposition_id,
                'url': f"https://zenodo.org/deposit/{deposition_id}",
                'note': 'Draft created - publish manually when ready'
            }
            print(f"✓ Zenodo upload complete (draft): https://zenodo.org/deposit/{deposition_id}")

        except Exception as e:
            self.results['zenodo'] = {'status': 'error', 'error': str(e)}
            print(f"✗ Zenodo upload failed: {e}")

    def upload_to_osf(self):
        """Upload to Open Science Framework."""
        print("\n=== UPLOADING TO OSF ===")
        try:
            import requests

            if not OSF_TOKEN:
                raise SystemExit('OSF_TOKEN not found in environment. Set OSF_TOKEN env var if you want to create an OSF project programmatically.')
            headers = {
                'Authorization': f'Bearer {OSF_TOKEN}',
                'Content-Type': 'application/json'
            }

            # Create new project
            project_data = {
                'data': {
                    'type': 'nodes',
                    'attributes': {
                        'title': 'NeuroCHIMERA: Consciousness Emergence Experiments',
                        'category': 'project',
                        'description': 'Complete experimental suite for consciousness emergence as phase transition',
                        'public': True,
                        'tags': ['consciousness', 'neuromorphic', 'GPU', 'phase-transition']
                    }
                }
            }

            r = requests.post(
                'https://api.osf.io/v2/nodes/',
                headers=headers,
                json=project_data
            )
            r.raise_for_status()
            project = r.json()
            project_id = project['data']['id']

            print(f"Created OSF project: {project_id}")

            # Note: File uploads to OSF require WaterButler API
            # This is a simplified version - full implementation would use osf-cli

            self.results['osf'] = {
                'status': 'success',
                'project_id': project_id,
                'url': f"https://osf.io/{project_id}",
                'note': 'Project created - upload files manually via web interface or osf-cli'
            }
            print(f"✓ OSF project created: https://osf.io/{project_id}")

        except Exception as e:
            self.results['osf'] = {'status': 'error', 'error': str(e)}
            print(f"✗ OSF upload failed: {e}")

    def upload_to_hf(self):
        """Upload release files to Hugging Face Hub as a dataset repo."""
        print("\n=== UPLOADING TO HUGGING FACE HUB ===")
        try:
            import subprocess
            hf_repo = "Agnuxo/neurochimera-dataset"
            hf_token = os.getenv('HF_TOKEN') or os.getenv('HUGGINGFACE_TOKEN')
            if not hf_token:
                raise SystemExit('HF_TOKEN not found in environment. Set HF_TOKEN or HUGGINGFACE_TOKEN env var.')
            cmd = [sys.executable, str(Path(__file__).parent / 'upload_to_hf.py'), '--repo-id', hf_repo, '--repo-type', 'dataset', '--path', str(RELEASE_DIR), '--commit', f'Upload {self.timestamp}']
            subprocess.run(cmd, check=True)
            self.results['huggingface'] = {
                'status': 'success',
                'repo': hf_repo,
                'url': f'https://huggingface.co/{hf_repo}'
            }
            print(f"✓ Hugging Face upload complete: https://huggingface.co/{hf_repo}")
        except Exception as e:
            self.results['huggingface'] = {'status': 'error', 'error': str(e)}
            print(f"✗ Hugging Face upload failed: {e}")

    def prepare_openml_export(self):
        """Prepare datasets for OpenML format."""
        print("\n=== PREPARING OPENML EXPORT ===")
        try:
            export_dir = RELEASE_DIR / "openml_export"
            export_dir.mkdir(exist_ok=True)

            # Create ARFF format files for each experiment
            # This is a template - actual conversion would depend on data format

            openml_metadata = {
                'name': 'NeuroCHIMERA-Experiments',
                'description': 'Consciousness emergence benchmark datasets',
                'format': 'ARFF',
                'experiments': 6,
                'url': 'https://github.com/Agnuxo1/Consciousness-Emergence-as-Phase-Transition-in-GPU-Native-Neuromorphic-Computing'
            }

            with open(export_dir / "openml_metadata.json", 'w') as f:
                json.dump(openml_metadata, f, indent=2)

            self.results['openml'] = {
                'status': 'prepared',
                'export_dir': str(export_dir),
                'note': 'Export prepared - upload manually via https://www.openml.org/'
            }
            print(f"✓ OpenML export prepared in {export_dir}")

        except Exception as e:
            self.results['openml'] = {'status': 'error', 'error': str(e)}
            print(f"✗ OpenML preparation failed: {e}")

    def register_openml_datasets(self):
        """Use the provided `openml_dataset_registration.py` to register datasets programmatically (optional)."""
        print("\n=== REGISTERING DATASETS ON OPENML ===")
        try:
            openml_api_key = os.getenv('OPENML_API_KEY')
            if not openml_api_key:
                raise SystemExit('OPENML_API_KEY not found in environment. Set it if you want to programmatically register datasets on OpenML.')
            import subprocess
            cmd = [sys.executable, str(Path(__file__).parent / 'openml_dataset_registration.py'), '--api-key', openml_api_key]
            subprocess.run(cmd, check=True)
            self.results['openml_register'] = {'status': 'success', 'note': 'Datasets registered on OpenML (check output)'}
            print('✓ OpenML registration executed (check logs for details)')
        except Exception as e:
            self.results['openml_register'] = {'status': 'error', 'error': str(e)}
            print(f"✗ OpenML registration failed: {e}")

    def prepare_datahub_export(self):
        """Prepare datasets for DataHub."""
        print("\n=== PREPARING DATAHUB EXPORT ===")
        try:
            export_dir = RELEASE_DIR / "datahub_export"
            export_dir.mkdir(exist_ok=True)

            # Create datapackage.json for DataHub
            datapackage = {
                "name": "neurochimera-experiments",
                "title": "NeuroCHIMERA Consciousness Emergence Experiments",
                "description": "Complete experimental suite demonstrating consciousness emergence as phase transition",
                "version": "1.0.0",
                "license": "CC-BY-4.0",
                "authors": [
                    {"name": "V.F. Veselov"},
                    {"name": "Francisco Angulo de Lafuente"}
                ],
                "keywords": ["consciousness", "neuromorphic", "GPU", "phase-transition", "benchmark"],
                "homepage": "https://github.com/Agnuxo1/Consciousness-Emergence-as-Phase-Transition-in-GPU-Native-Neuromorphic-Computing",
                "resources": []
            }

            # Add benchmark results as resources
            for result_file in BENCHMARKS_DIR.glob("results-exp-*.json"):
                datapackage['resources'].append({
                    "name": result_file.stem,
                    "path": f"benchmarks/{result_file.name}",
                    "format": "json",
                    "mediatype": "application/json"
                })

            with open(export_dir / "datapackage.json", 'w') as f:
                json.dump(datapackage, f, indent=2)

            # Copy benchmark results
            benchmarks_export = export_dir / "benchmarks"
            benchmarks_export.mkdir(exist_ok=True)
            for result_file in BENCHMARKS_DIR.glob("results-exp-*.json"):
                shutil.copy2(result_file, benchmarks_export / result_file.name)

            self.results['datahub'] = {
                'status': 'prepared',
                'export_dir': str(export_dir),
                'note': 'Export prepared - upload via https://datahub.io/ using datahub-cli'
            }
            print(f"✓ DataHub export prepared in {export_dir}")

        except Exception as e:
            self.results['datahub'] = {'status': 'error', 'error': str(e)}
            print(f"✗ DataHub preparation failed: {e}")

    def prepare_academia_export(self):
        """Prepare paper for Academia.edu upload."""
        print("\n=== PREPARING ACADEMIA.EDU EXPORT ===")
        try:
            export_dir = RELEASE_DIR / "academia_export"
            export_dir.mkdir(exist_ok=True)

            # Copy paper
            if PAPER_HTML.exists():
                shutil.copy2(PAPER_HTML, export_dir / "NeuroCHIMERA_Paper.html")

            # Create supplementary materials package
            supp_zip = export_dir / "supplementary_materials.zip"
            with zipfile.ZipFile(supp_zip, 'w', zipfile.ZIP_DEFLATED) as zf:
                # Add benchmark results
                for result_file in BENCHMARKS_DIR.glob("results-exp-*.json"):
                    zf.write(result_file, f"benchmarks/{result_file.name}")

                # Add README
                readme_content = """# NeuroCHIMERA Supplementary Materials

This package contains:
- Benchmark results from 6 experiments
- Dataset references
- Code repository links

Main repository: https://github.com/Agnuxo1/Consciousness-Emergence-as-Phase-Transition-in-GPU-Native-Neuromorphic-Computing
W&B Dashboard: https://wandb.ai/lareliquia-angulo
Zenodo Dataset: [DOI to be added]

Authors: V.F. Veselov & Francisco Angulo de Lafuente (2025)
"""
                zf.writestr("README.md", readme_content)

            self.results['academia'] = {
                'status': 'prepared',
                'export_dir': str(export_dir),
                'note': 'Upload manually to https://www.academia.edu/'
            }
            print(f"✓ Academia.edu export prepared in {export_dir}")

        except Exception as e:
            self.results['academia'] = {'status': 'error', 'error': str(e)}
            print(f"✗ Academia.edu preparation failed: {e}")

    def generate_report(self):
        """Generate comprehensive upload report."""
        report_path = RELEASE_DIR / f"upload_report_{self.timestamp}.json"

        report = {
            'timestamp': self.timestamp,
            'platforms': self.results,
            'summary': {
                'total_platforms': len(self.results),
                'successful': sum(1 for r in self.results.values() if r.get('status') == 'success'),
                'prepared': sum(1 for r in self.results.values() if r.get('status') == 'prepared'),
                'failed': sum(1 for r in self.results.values() if r.get('status') == 'error')
            }
        }

        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"\n=== UPLOAD REPORT ===")
        print(f"Report saved to: {report_path}")
        print(f"Successful uploads: {report['summary']['successful']}")
        print(f"Prepared for manual upload: {report['summary']['prepared']}")
        print(f"Failed: {report['summary']['failed']}")

        return report

def main():
    uploader = PlatformUploader()

    print("=" * 80)
    print("NeuroCHIMERA MULTI-PLATFORM UPLOAD UTILITY")
    print("=" * 80)

    # Automated uploads
    uploader.upload_to_wandb()
    uploader.upload_to_zenodo()
    uploader.upload_to_osf()
    # Optional: push to Hugging Face Hub (dataset repo)
    try:
        uploader.upload_to_hf()
    except SystemExit as e:
        print('Hugging Face upload skipped:', e)

    # Prepare exports for manual upload
    uploader.prepare_openml_export()
    # Optionally register on OpenML when API key is provided
    try:
        uploader.register_openml_datasets()
    except SystemExit as e:
        print('OpenML registration skipped:', e)
    uploader.prepare_datahub_export()
    uploader.prepare_academia_export()

    # Generate report
    report = uploader.generate_report()

    print("\n" + "=" * 80)
    print("NEXT STEPS:")
    print("=" * 80)
    print("1. Review Zenodo draft and publish: https://zenodo.org/me/uploads")
    print("2. Upload to OpenML: https://www.openml.org/")
    print("3. Upload to DataHub using datahub-cli")
    print("4. Upload paper to Academia.edu: https://www.academia.edu/")
    print("5. Figshare upload via FTP or web interface: https://figshare.com/")
    print("6. OpenAIRE indexing happens automatically via Zenodo")
    print("=" * 80)

if __name__ == '__main__':
    main()
