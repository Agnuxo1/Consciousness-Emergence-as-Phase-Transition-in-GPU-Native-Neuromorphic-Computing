#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Update README with badges for all platforms where NeuroCHIMERA is published.
"""
import os
import sys
from pathlib import Path

# Fix Windows console encoding
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

BASE_DIR = Path(__file__).parent.parent
README_PATH = BASE_DIR / "README.md"

# Platform configuration (update these after publishing)
PLATFORMS = {
    'wandb': {
        'url': 'https://wandb.ai/lareliquia-angulo/neurochimera-full-experiments',
        'badge': '[![W&B Experiments](https://img.shields.io/badge/W%26B-Experiments-FFBE00?style=for-the-badge&logo=weightsandbiases)](https://wandb.ai/lareliquia-angulo/neurochimera-full-experiments)',
        'enabled': True
    },
    'zenodo': {
        'doi': '10.5281/zenodo.17873070',  # Draft DOI - will change after publish
        'badge': '[![Zenodo DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.17873070-blue?style=for-the-badge&logo=zenodo)](https://zenodo.org/deposit/17873070)',
        'enabled': True  # Draft uploaded
    },
    'osf': {
        'project_id': '9wg2n',  # Actual project ID
        'badge': '[![OSF Project](https://img.shields.io/badge/OSF-Project-blue?style=for-the-badge&logo=osf)](https://osf.io/9wg2n/)',
        'enabled': True  # Project created
    },
    'figshare': {
        'url': 'https://figshare.com/articles/dataset/NeuroCHIMERA/XXXXX',
        'badge': '[![Figshare Dataset](https://img.shields.io/badge/Figshare-Dataset-orange?style=for-the-badge&logo=figshare)](https://figshare.com/articles/dataset/NeuroCHIMERA/XXXXX)',
        'enabled': False  # Set to True after publishing
    },
    'openml': {
        'url': 'https://www.openml.org/search?type=data&q=neurochimera',
        'badge': '[![OpenML](https://img.shields.io/badge/OpenML-Datasets-green?style=for-the-badge&logo=openml)](https://www.openml.org/search?type=data&q=neurochimera)',
        'enabled': False  # Set to True after publishing
    },
    'datahub': {
        'url': 'https://datahub.io/neurochimera-experiments',
        'badge': '[![DataHub](https://img.shields.io/badge/DataHub-Package-purple?style=for-the-badge)](https://datahub.io/neurochimera-experiments)',
        'enabled': False  # Set to True after publishing
    },
    'github': {
        'url': 'https://github.com/Agnuxo1/Consciousness-Emergence-as-Phase-Transition-in-GPU-Native-Neuromorphic-Computing',
        'badge': '[![GitHub](https://img.shields.io/github/stars/Agnuxo1/Consciousness-Emergence-as-Phase-Transition-in-GPU-Native-Neuromorphic-Computing?style=for-the-badge&logo=github)](https://github.com/Agnuxo1/Consciousness-Emergence-as-Phase-Transition-in-GPU-Native-Neuromorphic-Computing)',
        'enabled': True
    },
    'license': {
        'badge': '[![License: GPL-3.0](https://img.shields.io/badge/License-GPL%203.0-blue.svg?style=for-the-badge)](https://www.gnu.org/licenses/gpl-3.0)',
        'enabled': True
    }
}

BADGE_SECTION_TEMPLATE = """
## 📊 Datasets & Benchmarks

{badges}

### Live Experiment Tracking
View real-time experiment metrics and benchmarks on our [Weights & Biases dashboard](https://wandb.ai/lareliquia-angulo/neurochimera-full-experiments).

### Published Datasets
- **Zenodo**: Permanent archive with DOI for citations
- **Figshare**: Rich media and visualization support
- **OpenML**: Machine learning benchmark repository
- **DataHub**: Versioned data packages
- **OSF**: Complete research workflow documentation

### Benchmark Results
All benchmark results are publicly available and reproducible. See [publish/README_PUBLISHING.md](publish/README_PUBLISHING.md) for instructions on running benchmarks locally.
"""

def generate_badges():
    """Generate badge markdown for enabled platforms."""
    badges = []

    for platform, config in PLATFORMS.items():
        if config.get('enabled', False):
            badges.append(config['badge'])

    return '\n'.join(badges)

def update_readme():
    """Update README.md with badge section."""
    if not README_PATH.exists():
        print(f"✗ README.md not found at {README_PATH}")
        return False

    with open(README_PATH, 'r', encoding='utf-8') as f:
        content = f.read()

    # Generate badges
    badges = generate_badges()

    # Create badge section
    badge_section = BADGE_SECTION_TEMPLATE.format(badges=badges)

    # Check if badge section already exists
    if "## 📊 Datasets & Benchmarks" in content:
        print("⚠ Badge section already exists in README")
        print("Replace manually or delete existing section first")
        return False

    # Find insertion point (after main title)
    lines = content.split('\n')
    insert_index = -1

    for i, line in enumerate(lines):
        if line.startswith('# ') and i > 0:
            # Insert after first heading
            insert_index = i + 1
            break

    if insert_index == -1:
        # Fallback: insert at beginning
        insert_index = 0

    # Insert badge section
    lines.insert(insert_index, badge_section)

    # Write updated content
    updated_content = '\n'.join(lines)

    with open(README_PATH, 'w', encoding='utf-8') as f:
        f.write(updated_content)

    print(f"✓ README.md updated with badges")
    print(f"✓ Enabled platforms: {sum(1 for p in PLATFORMS.values() if p.get('enabled'))}/{len(PLATFORMS)}")

    return True

def print_badge_preview():
    """Print preview of all badges."""
    print("\n" + "="*80)
    print("BADGE PREVIEW")
    print("="*80)

    for platform, config in PLATFORMS.items():
        status = "✓ ENABLED" if config.get('enabled') else "✗ DISABLED"
        print(f"\n{platform.upper()}: {status}")
        print(config['badge'])

    print("\n" + "="*80)

def print_manual_instructions():
    """Print manual update instructions."""
    print("\n" + "="*80)
    print("MANUAL BADGE UPDATE INSTRUCTIONS")
    print("="*80)
    print("\nAfter publishing to each platform, update this script:")
    print("\n1. Open: publish/update_readme_badges.py")
    print("\n2. Update PLATFORMS dictionary:")
    print("   - Set 'enabled': True for published platforms")
    print("   - Update DOI for Zenodo")
    print("   - Update project_id for OSF")
    print("   - Update URLs for Figshare, DataHub, etc.")
    print("\n3. Run this script again:")
    print("   python publish/update_readme_badges.py")
    print("\n" + "="*80)

def generate_citation():
    """Generate citation text."""
    citation = f"""
## 📖 How to Cite

If you use NeuroCHIMERA in your research, please cite:

### BibTeX
```bibtex
@dataset{{veselov2025neurochimera,
  author = {{Veselov, V. F. and Angulo de Lafuente, Francisco}},
  title = {{NeuroCHIMERA: Consciousness Emergence as Phase Transition in
           Neuromorphic GPU-Native Computing}},
  year = {{2025}},
  publisher = {{Zenodo}},
  doi = {{{PLATFORMS['zenodo']['doi']}}},
  url = {{https://doi.org/{PLATFORMS['zenodo']['doi']}}}
}}
```

### APA
Veselov, V. F., & Angulo de Lafuente, F. (2025). *NeuroCHIMERA: Consciousness
Emergence as Phase Transition in Neuromorphic GPU-Native Computing* [Dataset].
Zenodo. https://doi.org/{PLATFORMS['zenodo']['doi']}

### IEEE
V. F. Veselov and F. Angulo de Lafuente, "NeuroCHIMERA: Consciousness Emergence
as Phase Transition in Neuromorphic GPU-Native Computing," Zenodo, 2025,
doi: {PLATFORMS['zenodo']['doi']}.
"""
    return citation

def save_citation_file():
    """Save citation information to CITATION.md."""
    citation_path = BASE_DIR / "CITATION.md"
    citation_content = generate_citation()

    with open(citation_path, 'w', encoding='utf-8') as f:
        f.write(citation_content.strip())

    print(f"✓ CITATION.md created")

def main():
    print("="*80)
    print("NEUROCHIMERA README BADGE UPDATER")
    print("="*80)

    # Print preview
    print_badge_preview()

    # Print manual instructions
    print_manual_instructions()

    # Ask user to confirm
    print("\nWould you like to:")
    print("1. Preview badges only (current)")
    print("2. Update README.md with badges")
    print("3. Create CITATION.md file")
    print("4. Both 2 and 3")

    # For automation, we'll create citation file
    save_citation_file()

    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    print("1. Publish to platforms (see publish/README_PUBLISHING.md)")
    print("2. Update platform IDs in this script")
    print("3. Re-run this script to add badges to README")
    print("4. Commit changes to git")
    print("="*80)

if __name__ == '__main__':
    main()
