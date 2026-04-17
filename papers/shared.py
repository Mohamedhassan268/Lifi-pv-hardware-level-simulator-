# papers/shared.py
"""
Shared utilities for paper validation scripts.

Provides common figure management, validation metric comparison,
and run_validation() boilerplate used across all 6 paper modules.

Usage:
    from papers.shared import save_figure, validate_metric, validation_header
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# =============================================================================
# FIGURE MANAGEMENT
# =============================================================================

# Publication-quality defaults
DEFAULT_DPI = 150
DEFAULT_FIGSIZE = (10, 6)


def save_figure(output_dir: str, filename: str, dpi: int = DEFAULT_DPI) -> str:
    """
    Save current matplotlib figure and close it.

    Args:
        output_dir: Directory to save into
        filename: Filename (e.g., 'fig6_transfer.png')
        dpi: Resolution (default 150)

    Returns:
        Full path to saved file
    """
    path = os.path.join(output_dir, filename)
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {path}")
    return path


def style_axis(ax, xlabel: str = '', ylabel: str = '', title: str = '',
               grid: bool = True, fontsize: int = 12):
    """Apply standard axis formatting."""
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=fontsize)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=fontsize)
    if title:
        ax.set_title(title, fontsize=fontsize + 2, fontweight='bold')
    if grid:
        ax.grid(True, alpha=0.3)


# =============================================================================
# VALIDATION METRICS
# =============================================================================

def validate_metric(computed, target, description: str,
                    threshold_pct: float = 20.0, unit: str = '') -> dict:
    """
    Compare a computed value against a paper target.

    Args:
        computed: Simulated value
        target: Expected value from paper
        description: Metric name for printing
        threshold_pct: Pass/fail threshold as percentage error
        unit: Optional unit string for display

    Returns:
        Dict with 'computed', 'target', 'error_pct', 'passed'
    """
    if target == 0:
        error_pct = 0.0 if computed == 0 else 100.0
    else:
        error_pct = abs(computed - target) / abs(target) * 100

    passed = error_pct <= threshold_pct
    status = 'PASS' if passed else 'FAIL'

    unit_str = f' {unit}' if unit else ''
    print(f"  {description}: {computed:.4g}{unit_str} "
          f"(target: {target:.4g}{unit_str}), "
          f"error: {error_pct:.1f}%  [{status}]")

    return {
        'description': description,
        'computed': computed,
        'target': target,
        'error_pct': error_pct,
        'passed': passed,
    }


def print_validation_summary(results: list) -> bool:
    """
    Print summary of multiple validation results.

    Args:
        results: List of dicts from validate_metric()

    Returns:
        True if all passed
    """
    n_pass = sum(1 for r in results if r['passed'])
    n_total = len(results)
    all_pass = n_pass == n_total

    print(f"\n  Summary: {n_pass}/{n_total} metrics passed")
    print(f"  Overall: {'PASS' if all_pass else 'REVIEW'}")
    return all_pass


# =============================================================================
# VALIDATION BOILERPLATE
# =============================================================================

def validation_header(paper_label: str, journal: str,
                      output_dir: str = None,
                      default_subdir: str = 'validation') -> str:
    """
    Print validation header and ensure output directory exists.

    Args:
        paper_label: e.g., "KADIRVELU et al. (2021) - SLIPT VALIDATION"
        journal: e.g., "IEEE Trans. Green Communications and Networking"
        output_dir: Explicit output dir (None = auto)
        default_subdir: Subdirectory name under workspace/

    Returns:
        Resolved output_dir path
    """
    if output_dir is None:
        output_dir = os.path.join('workspace', default_subdir)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'=' * 65}")
    print(f"  {paper_label}")
    print(f"  {journal}")
    print(f"{'=' * 65}")

    return output_dir


# =============================================================================
# PHYSICS HELPERS (commonly reimplemented across papers)
# =============================================================================

def lambertian_order(half_angle_deg: float) -> float:
    """Lambertian emission order: m = -ln(2) / ln(cos(theta))."""
    alpha_rad = np.deg2rad(half_angle_deg)
    return -np.log(2) / np.log(np.cos(alpha_rad))


def lambertian_channel_gain(m: float, distance_m: float,
                            rx_area_m2: float) -> float:
    """
    LOS Lambertian channel gain (on-axis).

    H = (m+1) / (2*pi*d^2) * A_rx
    """
    if distance_m <= 0:
        return 0.0
    return ((m + 1) * rx_area_m2) / (2 * np.pi * distance_m ** 2)


def beer_lambert_attenuation(distance_m: float,
                             humidity_rh: float = 0.3) -> float:
    """
    Beer-Lambert atmospheric attenuation factor (Correa 2025 model).

    alpha = 0.3 + 4.0 * max(RH - 0.3, 0)^1.5
    Returns: exp(-alpha * d)
    """
    alpha = 0.3 + 4.0 * max(humidity_rh - 0.3, 0) ** 1.5
    return np.exp(-alpha * distance_m)
