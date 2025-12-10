#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive audit script for all 6 NeuroCHIMERA experiments.
Generates detailed audit reports with reproducibility checks.
"""
import os
import sys
import json
import subprocess
import hashlib
from pathlib import Path
from datetime import datetime, timezone

# Fix Windows console encoding
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

BASE_DIR = Path(__file__).parent.parent
EXPERIMENT_MAP = {
    1: '1-2/experiment1_spacetime_emergence.py',
    2: '1-2/experiment2_consciousness_emergence.py',
    3: '3-4/Experiment_Genesis_1.py',
    4: '3-4/Experiment_Genesis_2.py',
    5: '5-6/benchmark_experiment_1.py',
    6: '5-6/benchmark_experiment_2.py',
}

BENCHMARK_MAP = {
    1: ['1-2/benchmarks/benchmark_performance.py',
        '1-2/benchmarks/benchmark_accuracy.py',
        '1-2/benchmarks/benchmark_scaling.py'],
    2: ['1-2/benchmarks/benchmark_consciousness_emergence.py'],
    3: ['3-4/benchmark_genesis_1.py'],
    4: ['3-4/benchmark_genesis_2.py'],
    5: ['5-6/benchmark_experiment_1.py'],
    6: ['5-6/benchmark_experiment_2.py'],
}

class ExperimentAuditor:
    def __init__(self):
        self.results = {}
        self.timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')

    def compute_file_hash(self, filepath):
        """Compute SHA256 hash of file for integrity verification."""
        sha256_hash = hashlib.sha256()
        try:
            with open(filepath, "rb") as f:
                for byte_block in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(byte_block)
            return sha256_hash.hexdigest()
        except Exception as e:
            return f"ERROR: {e}"

    def check_dependencies(self, script_path):
        """Extract and verify dependencies from script."""
        dependencies = []
        try:
            with open(script_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip().startswith('import ') or line.strip().startswith('from '):
                        # Extract module name
                        parts = line.strip().split()
                        if len(parts) >= 2:
                            module = parts[1].split('.')[0]
                            if module not in dependencies and module not in ['os', 'sys', 'json']:
                                dependencies.append(module)
        except Exception as e:
            return {'error': str(e)}

        # Check if dependencies are installed
        installed = {}
        for dep in dependencies:
            try:
                __import__(dep)
                installed[dep] = True
            except ImportError:
                installed[dep] = False

        return {
            'required': dependencies,
            'installed': installed,
            'missing': [d for d, status in installed.items() if not status]
        }

    def audit_experiment(self, exp_num):
        """Comprehensive audit of a single experiment."""
        print(f"\n{'='*80}")
        print(f"AUDITING EXPERIMENT {exp_num}")
        print(f"{'='*80}")

        script_path = BASE_DIR / EXPERIMENT_MAP[exp_num]
        audit_data = {
            'experiment_number': exp_num,
            'timestamp': self.timestamp,
            'script_path': str(script_path),
            'checks': {}
        }

        # Check 1: File exists
        print(f"[1/7] Checking file existence...")
        if script_path.exists():
            audit_data['checks']['file_exists'] = True
            print(f"  ✓ File exists: {script_path}")
        else:
            audit_data['checks']['file_exists'] = False
            print(f"  ✗ File not found: {script_path}")
            return audit_data

        # Check 2: File integrity (hash)
        print(f"[2/7] Computing file hash...")
        file_hash = self.compute_file_hash(script_path)
        audit_data['checks']['file_hash'] = file_hash
        print(f"  ✓ SHA256: {file_hash[:16]}...")

        # Check 3: File size and line count
        print(f"[3/7] Analyzing file structure...")
        file_size = script_path.stat().st_size
        with open(script_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            line_count = len(lines)
            code_lines = len([l for l in lines if l.strip() and not l.strip().startswith('#')])

        audit_data['checks']['file_size_bytes'] = file_size
        audit_data['checks']['total_lines'] = line_count
        audit_data['checks']['code_lines'] = code_lines
        print(f"  ✓ Size: {file_size:,} bytes, {line_count} lines ({code_lines} code)")

        # Check 4: Dependencies
        print(f"[4/7] Checking dependencies...")
        deps = self.check_dependencies(script_path)
        audit_data['checks']['dependencies'] = deps
        if deps.get('missing'):
            print(f"  ⚠ Missing dependencies: {', '.join(deps['missing'])}")
        else:
            print(f"  ✓ All dependencies available")

        # Check 5: Syntax check
        print(f"[5/7] Validating Python syntax...")
        syntax_check = subprocess.run(
            [sys.executable, '-m', 'py_compile', str(script_path)],
            capture_output=True,
            text=True
        )
        audit_data['checks']['syntax_valid'] = syntax_check.returncode == 0
        if syntax_check.returncode == 0:
            print(f"  ✓ Syntax valid")
        else:
            print(f"  ✗ Syntax errors found")
            audit_data['checks']['syntax_errors'] = syntax_check.stderr

        # Check 6: Benchmark scripts
        print(f"[6/7] Checking benchmark scripts...")
        benchmark_audit = []
        if exp_num in BENCHMARK_MAP:
            for bench_path in BENCHMARK_MAP[exp_num]:
                bench_full = BASE_DIR / bench_path
                bench_data = {
                    'path': str(bench_full),
                    'exists': bench_full.exists()
                }
                if bench_full.exists():
                    bench_data['hash'] = self.compute_file_hash(bench_full)
                    print(f"  ✓ Benchmark: {bench_path}")
                else:
                    print(f"  ⚠ Missing benchmark: {bench_path}")
                benchmark_audit.append(bench_data)
        audit_data['checks']['benchmarks'] = benchmark_audit

        # Check 7: Recent execution results
        print(f"[7/7] Checking execution results...")
        results_dir = BASE_DIR / 'release' / 'benchmarks'
        recent_results = []
        if results_dir.exists():
            for result_file in results_dir.glob(f'results-exp-{exp_num}-*.json'):
                try:
                    with open(result_file, 'r') as f:
                        result_data = json.load(f)
                        recent_results.append({
                            'file': result_file.name,
                            'timestamp': result_data.get('timestamp'),
                            'returncode': result_data.get('returncode'),
                            'has_summary': result_data.get('summary') is not None
                        })
                except:
                    pass

        audit_data['checks']['recent_results'] = recent_results
        if recent_results:
            successful = [r for r in recent_results if r['returncode'] == 0]
            print(f"  ✓ Found {len(recent_results)} execution(s), {len(successful)} successful")
        else:
            print(f"  ⚠ No execution results found")

        # Overall assessment
        print(f"\n{'='*80}")
        critical_checks = ['file_exists', 'syntax_valid']
        all_critical_pass = all(audit_data['checks'].get(c, False) for c in critical_checks)

        if all_critical_pass:
            audit_data['status'] = 'PASS'
            print(f"EXPERIMENT {exp_num}: ✓ PASS")
        else:
            audit_data['status'] = 'FAIL'
            print(f"EXPERIMENT {exp_num}: ✗ FAIL")

        return audit_data

    def generate_audit_report(self, all_audits):
        """Generate comprehensive audit report."""
        report = {
            'audit_timestamp': self.timestamp,
            'total_experiments': 6,
            'experiments': all_audits,
            'summary': {
                'passed': sum(1 for a in all_audits.values() if a.get('status') == 'PASS'),
                'failed': sum(1 for a in all_audits.values() if a.get('status') == 'FAIL'),
                'total_missing_deps': 0,
                'total_benchmark_scripts': 0,
                'total_benchmark_missing': 0
            }
        }

        # Aggregate statistics
        for audit in all_audits.values():
            deps = audit['checks'].get('dependencies', {})
            if isinstance(deps, dict):
                report['summary']['total_missing_deps'] += len(deps.get('missing', []))

            benchmarks = audit['checks'].get('benchmarks', [])
            report['summary']['total_benchmark_scripts'] += len(benchmarks)
            report['summary']['total_benchmark_missing'] += sum(
                1 for b in benchmarks if not b.get('exists', False)
            )

        # Save report
        report_path = BASE_DIR / 'release' / f'audit_report_{self.timestamp}.json'
        report_path.parent.mkdir(exist_ok=True)

        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)

        return report, report_path

    def generate_markdown_report(self, all_audits):
        """Generate human-readable markdown audit report."""
        md_lines = [
            "# NeuroCHIMERA Experiment Audit Report",
            "",
            f"**Audit Date**: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC",
            f"**Total Experiments**: 6",
            "",
            "## Summary",
            ""
        ]

        passed = sum(1 for a in all_audits.values() if a.get('status') == 'PASS')
        failed = sum(1 for a in all_audits.values() if a.get('status') == 'FAIL')

        md_lines.extend([
            f"- **Passed**: {passed}/6",
            f"- **Failed**: {failed}/6",
            "",
            "## Individual Experiment Results",
            ""
        ])

        for exp_num in sorted(all_audits.keys()):
            audit = all_audits[exp_num]
            status_icon = "✓" if audit.get('status') == 'PASS' else "✗"

            md_lines.extend([
                f"### Experiment {exp_num}: {status_icon} {audit.get('status', 'UNKNOWN')}",
                "",
                f"**Script**: `{audit.get('script_path', 'N/A')}`",
                "",
                "**Checks**:",
                ""
            ])

            checks = audit.get('checks', {})

            # File checks
            md_lines.append(f"- File exists: {'✓' if checks.get('file_exists') else '✗'}")
            md_lines.append(f"- Syntax valid: {'✓' if checks.get('syntax_valid') else '✗'}")

            if checks.get('file_size_bytes'):
                md_lines.append(f"- File size: {checks['file_size_bytes']:,} bytes")
                md_lines.append(f"- Lines of code: {checks.get('code_lines', 0)}/{checks.get('total_lines', 0)}")

            if checks.get('file_hash'):
                md_lines.append(f"- SHA256: `{checks['file_hash'][:32]}...`")

            # Dependencies
            deps = checks.get('dependencies', {})
            if isinstance(deps, dict) and deps.get('missing'):
                md_lines.append(f"- ⚠ Missing dependencies: {', '.join(deps['missing'])}")
            else:
                md_lines.append(f"- Dependencies: All available")

            # Benchmarks
            benchmarks = checks.get('benchmarks', [])
            if benchmarks:
                existing = sum(1 for b in benchmarks if b.get('exists'))
                md_lines.append(f"- Benchmark scripts: {existing}/{len(benchmarks)} available")

            # Recent results
            results = checks.get('recent_results', [])
            if results:
                successful = sum(1 for r in results if r.get('returncode') == 0)
                md_lines.append(f"- Recent executions: {len(results)} ({successful} successful)")

            md_lines.append("")

        # Recommendations
        md_lines.extend([
            "## Recommendations",
            ""
        ])

        issues = []
        for exp_num, audit in all_audits.items():
            checks = audit.get('checks', {})
            deps = checks.get('dependencies', {})

            if isinstance(deps, dict) and deps.get('missing'):
                issues.append(f"- Install missing dependencies for Experiment {exp_num}: `pip install {' '.join(deps['missing'])}`")

            benchmarks = checks.get('benchmarks', [])
            missing_benchmarks = [b for b in benchmarks if not b.get('exists')]
            if missing_benchmarks:
                issues.append(f"- Restore missing benchmark scripts for Experiment {exp_num}")

        if issues:
            md_lines.extend(issues)
        else:
            md_lines.append("✓ No issues found - all experiments ready for execution")

        md_lines.extend([
            "",
            "## Reproducibility",
            "",
            "To reproduce these experiments:",
            "",
            "```bash",
            "# Install dependencies",
            "pip install -r requirements_experiments.txt",
            "",
            "# Run all experiments",
            "python publish/run_benchmarks.py",
            "",
            "# Run individual experiment",
            "python publish/run_experiment.py --exp 1",
            "```",
            "",
            "## Contact",
            "",
            "For questions about this audit:",
            "- Repository: https://github.com/Agnuxo1/Consciousness-Emergence-as-Phase-Transition-in-GPU-Native-Neuromorphic-Computing",
            "- Authors: V.F. Veselov & Francisco Angulo de Lafuente",
            ""
        ])

        md_report = '\n'.join(md_lines)
        md_path = BASE_DIR / 'release' / f'audit_report_{self.timestamp}.md'

        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(md_report)

        return md_report, md_path

def main():
    print("=" * 80)
    print("NEUROCHIMERA EXPERIMENT AUDIT SYSTEM")
    print("=" * 80)
    print("Performing comprehensive audit of all 6 experiments...")

    auditor = ExperimentAuditor()
    all_audits = {}

    # Audit each experiment
    for exp_num in range(1, 7):
        audit_result = auditor.audit_experiment(exp_num)
        all_audits[exp_num] = audit_result

    # Generate reports
    print(f"\n{'='*80}")
    print("GENERATING AUDIT REPORTS")
    print(f"{'='*80}")

    json_report, json_path = auditor.generate_audit_report(all_audits)
    md_report, md_path = auditor.generate_markdown_report(all_audits)

    print(f"✓ JSON report: {json_path}")
    print(f"✓ Markdown report: {md_path}")

    # Print summary
    print(f"\n{'='*80}")
    print("AUDIT SUMMARY")
    print(f"{'='*80}")
    print(f"Total Experiments: {json_report['total_experiments']}")
    print(f"Passed: {json_report['summary']['passed']}")
    print(f"Failed: {json_report['summary']['failed']}")
    print(f"Missing Dependencies: {json_report['summary']['total_missing_deps']}")
    print(f"Missing Benchmark Scripts: {json_report['summary']['total_benchmark_missing']}/{json_report['summary']['total_benchmark_scripts']}")

    # Return appropriate exit code
    if json_report['summary']['failed'] > 0:
        print(f"\n⚠ Some experiments failed audit checks")
        return 1
    else:
        print(f"\n✓ All experiments passed audit!")
        return 0

if __name__ == '__main__':
    sys.exit(main())
