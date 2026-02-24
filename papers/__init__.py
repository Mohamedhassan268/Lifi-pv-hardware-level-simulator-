"""
papers/ — Paper Validation & Figure Generation Registry
========================================================

Each paper module provides:
  - PARAMS dict (locked to paper values)
  - TARGETS dict (expected results)
  - run_validation(output_dir) -> bool (generates figures + returns pass/fail)

Usage:
  from papers import PAPERS, run_paper, run_all_papers
"""

from papers import kadirvelu_2021
from papers import correa_2025
from papers import gonzalez_2024
from papers import sarwar_2017
from papers import oliveira_2024
from papers import xu_2024


PAPERS = {
    'kadirvelu2021': {
        'label': 'Kadirvelu et al. (2021)',
        'reference': 'IEEE Photonics Journal, 2021',
        'module': kadirvelu_2021,
        'run': kadirvelu_2021.run_validation,
    },
    'correa2025': {
        'label': 'Correa Morales et al. (2025)',
        'reference': 'Optics Express, 2025',
        'module': correa_2025,
        'run': correa_2025.run_validation,
    },
    'gonzalez2024': {
        'label': 'Gonzalez et al. (2024)',
        'reference': 'IEEE Access, 2024',
        'module': gonzalez_2024,
        'run': gonzalez_2024.run_validation,
    },
    'sarwar2017': {
        'label': 'Sarwar et al. (2017)',
        'reference': 'ICOCN 2017',
        'module': sarwar_2017,
        'run': sarwar_2017.run_validation,
    },
    'oliveira2024': {
        'label': 'Oliveira et al. (2024)',
        'reference': 'Light: Science & Applications, 2024',
        'module': oliveira_2024,
        'run': oliveira_2024.run_validation,
    },
    'xu2024': {
        'label': 'Xu et al. (2024)',
        'reference': 'EWSN 2024',
        'module': xu_2024,
        'run': xu_2024.run_validation,
    },
}


def run_paper(name, output_dir=None):
    """Run validation for a single paper."""
    if name not in PAPERS:
        raise ValueError(f"Unknown paper: {name}. Available: {list(PAPERS.keys())}")
    return PAPERS[name]['run'](output_dir)


def run_all_papers(base_dir=None):
    """Run validation for all papers. Returns dict of results."""
    import os
    if base_dir is None:
        base_dir = os.path.join('workspace', 'validation')
    results = {}
    for name, info in PAPERS.items():
        out = os.path.join(base_dir, name)
        try:
            results[name] = info['run'](out)
        except Exception as e:
            print(f"\n  ERROR in {name}: {e}")
            results[name] = False
    return results


def list_papers():
    """Return list of (name, label, reference) tuples."""
    return [(k, v['label'], v['reference']) for k, v in PAPERS.items()]
