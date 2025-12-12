#!/usr/bin/env python3
"""
OpenML Dataset Registration for NeuroCHIMERA Experiments

This script registers all 6 NeuroCHIMERA experimental datasets on OpenML.
It converts experimental data to ARFF format and uploads with comprehensive metadata.
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import openml
import argparse

class OpenMLDatasetRegistrar:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv('OPENML_API_KEY')
        if not self.api_key:
            raise ValueError("OpenML API key required. Set OPENML_API_KEY environment variable.")

        openml.config.apikey = self.api_key
        self.base_dir = Path(__file__).parent.parent
        self.release_dir = self.base_dir / "release"
        self.openml_export_dir = self.release_dir / "openml_export"

        # Experiment configurations
        self.experiments = {
            1: {
                'name': 'NeuroCHIMERA-Exp1-Spacetime-Emergence',
                'description': '''Experiment 1: Space-Time Emergence from Computational Network

This dataset contains benchmark results from the first NeuroCHIMERA experiment demonstrating
space-time emergence in GPU-native neuromorphic computing systems.

Key metrics:
- Free Energy evolution over training epochs
- System Entropy measurements
- Stability metrics (dState convergence)
- Performance benchmarks across 5 runs

The experiment shows how computational networks can spontaneously generate
space-time structures through phase transitions in neuromorphic hardware.''',
                'data_file': '3-4/genesis_1_benchmark_data.csv',
                'attributes': ['Epoch', 'FreeEnergy', 'SystemEntropy', 'Stability_dState']
            },
            2: {
                'name': 'NeuroCHIMERA-Exp2-Consciousness-Emergence',
                'description': '''Experiment 2: Consciousness Emergence as Phase Transition

Complete benchmark suite for consciousness emergence demonstration in neuromorphic computing.
This experiment shows consciousness as an emergent property of phase transitions in GPU-native
neuromorphic systems.

Dataset includes:
- Consciousness emergence metrics
- Metacognitive accuracy measurements
- Memory persistence data
- Phase transition detection
- Benchmark performance across multiple runs

The data demonstrates how consciousness emerges from computational complexity
through critical phase transitions.''',
                'data_file': '5-6/benchmark_exp2_summary.json',
                'attributes': ['iteration', 'duration_sec', 'metacog_acc', 'memory_persistence']
            },
            3: {
                'name': 'NeuroCHIMERA-Exp3-Genesis-1',
                'description': '''Experiment 3: Genesis Experiment 1

Genesis series experiments exploring fundamental emergence patterns in neuromorphic computing.
This dataset contains the first genesis experiment data showing basic emergence phenomena.

Includes benchmark data with performance metrics and emergence indicators.''',
                'data_file': '3-4/genesis_1_benchmark_data.csv',
                'attributes': ['Epoch', 'FreeEnergy', 'SystemEntropy', 'Stability_dState']
            },
            4: {
                'name': 'NeuroCHIMERA-Exp4-Genesis-2',
                'description': '''Experiment 4: Genesis Experiment 2

Second genesis experiment with enhanced emergence patterns and consciousness precursors.
Contains comprehensive benchmark data showing advanced emergence characteristics.''',
                'data_file': '3-4/genesis_2_benchmark_data.csv',
                'attributes': ['Epoch', 'FreeEnergy', 'SystemEntropy', 'Stability_dState']
            },
            5: {
                'name': 'NeuroCHIMERA-Exp5-Benchmark-1',
                'description': '''Experiment 5: Benchmark Experiment 1

Standardized benchmark suite for neuromorphic consciousness emergence validation.
Contains performance metrics and validation data for the first benchmark experiment.''',
                'data_file': 'benchmark_summary.json',
                'attributes': ['iteration', 'duration_sec', 'gpu_mem_mb', 'final_sync', 'tc']
            },
            6: {
                'name': 'NeuroCHIMERA-Exp6-Benchmark-2',
                'description': '''Experiment 6: Benchmark Experiment 2

Final benchmark experiment demonstrating complete consciousness emergence validation.
Includes all performance metrics, consciousness indicators, and validation results.''',
                'data_file': '5-6/benchmark_exp2_summary.json',
                'attributes': ['iteration', 'duration_sec', 'metacog_acc', 'memory_persistence']
            }
        }

    def create_arff_from_csv(self, csv_path, attributes, experiment_name):
        """Convert CSV data to ARFF format."""
        try:
            df = pd.read_csv(csv_path)

            # Create ARFF content
            arff_content = f"@RELATION {experiment_name}\n\n"

            # Add attributes
            for attr in attributes:
                if attr in df.columns:
                    # Determine attribute type
                    sample_val = df[attr].iloc[0] if len(df) > 0 else 0
                    if isinstance(sample_val, (int, np.integer)):
                        attr_type = "INTEGER"
                    elif isinstance(sample_val, (float, np.floating)):
                        attr_type = "REAL"
                    else:
                        attr_type = "STRING"
                    arff_content += f"@ATTRIBUTE {attr} {attr_type}\n"

            arff_content += "\n@DATA\n"

            # Add data rows
            for _, row in df.iterrows():
                values = []
                for attr in attributes:
                    if attr in df.columns:
                        val = row[attr]
                        if pd.isna(val):
                            values.append("?")
                        elif isinstance(val, float):
                            values.append(f"{val:.6f}")
                        else:
                            values.append(str(val))
                    else:
                        values.append("0")  # Default value
                arff_content += ",".join(values) + "\n"

            return arff_content

        except Exception as e:
            print(f"Error creating ARFF from {csv_path}: {e}")
            return None

    def create_arff_from_json(self, json_path, attributes, experiment_name):
        """Convert JSON benchmark data to ARFF format."""
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)

            # Handle different JSON structures
            if isinstance(data, list):  # Array of iterations
                df = pd.DataFrame(data)
            elif 'runs' in data:  # Summary with runs
                df = pd.DataFrame(data['runs'])
            else:
                df = pd.DataFrame([data])

            # Create ARFF content
            arff_content = f"@RELATION {experiment_name}\n\n"

            # Add attributes
            for attr in attributes:
                if attr in df.columns:
                    sample_val = df[attr].iloc[0] if len(df) > 0 else 0
                    if isinstance(sample_val, (int, np.integer)):
                        attr_type = "INTEGER"
                    elif isinstance(sample_val, (float, np.floating)):
                        attr_type = "REAL"
                    else:
                        attr_type = "STRING"
                    arff_content += f"@ATTRIBUTE {attr} {attr_type}\n"

            arff_content += "\n@DATA\n"

            # Add data rows
            for _, row in df.iterrows():
                values = []
                for attr in attributes:
                    if attr in df.columns:
                        val = row[attr]
                        if pd.isna(val):
                            values.append("?")
                        elif isinstance(val, float):
                            values.append(f"{val:.6f}")
                        else:
                            values.append(str(val))
                    else:
                        values.append("0")
                arff_content += ",".join(values) + "\n"

            return arff_content

        except Exception as e:
            print(f"Error creating ARFF from {json_path}: {e}")
            return None

    def register_dataset(self, exp_num, exp_config):
        """Register a single dataset on OpenML."""
        print(f"\n=== REGISTERING EXPERIMENT {exp_num}: {exp_config['name']} ===")

        try:
            # Determine data source
            data_path = self.base_dir / exp_config['data_file']
            if not data_path.exists():
                print(f"Data file not found: {data_path}")
                return None

            # Create ARFF content
            if exp_config['data_file'].endswith('.csv'):
                arff_content = self.create_arff_from_csv(
                    data_path, exp_config['attributes'], exp_config['name']
                )
            else:  # JSON files
                arff_content = self.create_arff_from_json(
                    data_path, exp_config['attributes'], exp_config['name']
                )

            if not arff_content:
                print(f"Failed to create ARFF content for experiment {exp_num}")
                return None

            # Save ARFF file
            arff_filename = f"experiment_{exp_num}.arff"
            arff_path = self.openml_export_dir / arff_filename
            with open(arff_path, 'w') as f:
                f.write(arff_content)

            # Create comprehensive description
            full_description = f"""{exp_config['description']}

NeuroCHIMERA Project: Consciousness Emergence as Phase Transition in Neuromorphic GPU-Native Computing

Authors: V.F. Veselov & Francisco Angulo de Lafuente (2025)

This dataset is part of a comprehensive experimental suite demonstrating consciousness emergence
through phase transitions in GPU-native neuromorphic computing systems.

Key Features:
- Experimental benchmark data
- Performance metrics
- Emergence indicators
- Consciousness metrics (where applicable)

Related Links:
- Main Repository: https://github.com/Agnuxo1/Consciousness-Emergence-as-Phase-Transition-in-GPU-Native-Neuromorphic-Computing
- Zenodo Dataset: [DOI to be assigned]
- Weights & Biases Dashboard: https://wandb.ai/lareliquia-angulo/neurochimera-full-experiments
- Figshare Collection: [DOI to be assigned]
- Open Science Framework: [URL to be assigned]

License: CC-BY-4.0
Publication Date: {datetime.now().strftime('%Y-%m-%d')}
"""

            # Create OpenML dataset
            dataset = openml.datasets.create_dataset(  # type: ignore
                name=exp_config['name'],
                description=full_description,
                creator='V.F. Veselov; Francisco Angulo de Lafuente',
                contributor='NeuroCHIMERA Research Team',
                collection_date=datetime.now().strftime('%Y-%m-%d'),
                language='English',
                licence='CC-BY-4.0',
                default_target_attribute=None,  # No target attribute for benchmark data
                row_id_attribute=None,
                ignore_attribute=None,
                citation='Veselov, V.F. & Angulo de Lafuente, F. (2025). Consciousness Emergence as Phase Transition in Neuromorphic GPU-Native Computing.',
                original_data_url='https://github.com/Agnuxo1/Consciousness-Emergence-as-Phase-Transition-in-GPU-Native-Computing',
                paper_url='https://github.com/Agnuxo1/Consciousness-Emergence-as-Phase-Transition-in-GPU-Native-Neuromorphic-Computing',
                file=str(arff_path)  # type: ignore[arg-type]
            )

            # Publish the dataset
            dataset.publish()

            print(f"+ Successfully registered dataset: {dataset.name}")
            print(f"  Dataset ID: {dataset.dataset_id}")
            print(f"  OpenML URL: https://www.openml.org/d/{dataset.dataset_id}")

            return {
                'experiment': exp_num,
                'name': exp_config['name'],
                'dataset_id': dataset.dataset_id,
                'url': f"https://www.openml.org/d/{dataset.dataset_id}",
                'status': 'published'
            }

        except Exception as e:
            print(f"✗ Failed to register experiment {exp_num}: {e}")
            return {
                'experiment': exp_num,
                'name': exp_config['name'],
                'error': str(e),
                'status': 'failed'
            }

    def register_all_datasets(self):
        """Register all 6 datasets on OpenML."""
        print("Starting OpenML dataset registration for NeuroCHIMERA experiments...")
        if self.api_key:
            print(f"Using OpenML API key: {self.api_key[:8]}...")
        else:
            print("No API key available")

        # Ensure export directory exists
        self.openml_export_dir.mkdir(exist_ok=True)

        results = []
        for exp_num in range(1, 7):
            result = self.register_dataset(exp_num, self.experiments[exp_num])
            if result:
                results.append(result)

        # Save registration report
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_datasets': len(results),
            'successful': len([r for r in results if r.get('status') == 'published']),
            'failed': len([r for r in results if r.get('status') == 'failed']),
            'datasets': results
        }

        report_path = self.openml_export_dir / "openml_registration_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        print("\n=== REGISTRATION COMPLETE ===")
        print(f"Report saved to: {report_path}")
        print(f"Successful registrations: {report['successful']}/6")

        # Print dataset URLs
        print("\nRegistered Datasets:")
        for result in results:
            if result.get('status') == 'published':
                print(f"- Experiment {result['experiment']}: {result['url']}")
            else:
                print(f"- Experiment {result['experiment']}: FAILED - {result.get('error', 'Unknown error')}")

        return report

def main():
    parser = argparse.ArgumentParser(description='Register NeuroCHIMERA datasets on OpenML')
    parser.add_argument('--api-key', help='OpenML API key (or set OPENML_API_KEY env var)')
    args = parser.parse_args()

    try:
        registrar = OpenMLDatasetRegistrar(api_key=args.api_key)
        report = registrar.register_all_datasets()

        # Exit with success/failure code
        if report['successful'] == 6:
            print("\n+ All datasets successfully registered on OpenML!")
            exit(0)
        else:
            print(f"\n- {report['failed']} datasets failed to register")
            exit(1)

    except Exception as e:
        print(f"Error: {e}")
        exit(1)

if __name__ == '__main__':
    main()