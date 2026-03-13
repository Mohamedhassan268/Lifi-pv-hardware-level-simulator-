# papers/pipeline_validation.py
"""
Pipeline Validation — Run all paper presets through the unified cosim pipeline
and compare results against paper targets.

Bridges the gap between standalone paper validation scripts (which use their
own physics models) and the generalized cosim framework.

Usage:
    from papers.pipeline_validation import validate_all, validate_preset
    results = validate_all(output_dir='workspace/validation_pipeline')
    result = validate_preset('kadirvelu2021')
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from cosim.system_config import SystemConfig
from cosim.python_engine import run_python_simulation


# =============================================================================
# PAPER-SPECIFIC METRIC EXTRACTORS
# =============================================================================

# Each paper has different validation metrics. These functions extract
# the relevant comparison points from the pipeline output.

_PAPER_METRICS = {
    'kadirvelu2021': {
        'label': 'Kadirvelu 2021',
        'metrics': ['channel_gain', 'P_rx_uW', 'I_ph_uA', 'BER'],
        'expected': {
            'channel_gain': 0.0345,   # G_ch at 32.5cm
            'P_rx_uW': 321.0,        # P_rx from link budget
            'I_ph_uA': 146.7,        # I_ph = R * P_rx
            'BER': 1.008e-3,         # Target from paper
        },
    },
    'gonzalez2024': {
        'label': 'González 2024',
        'metrics': ['BER'],
        'expected': {
            'BER': 0.0,              # Zero BER at 60cm
        },
    },
    'correa2025': {
        'label': 'Correa 2025',
        'metrics': ['BER'],
        'expected': {
            'BER': 0.01,             # Target from paper
        },
    },
    'sarwar2017': {
        'label': 'Sarwar 2017',
        'metrics': ['BER', 'data_rate_mbps'],
        'expected': {
            'BER': 1.6883e-3,
            'data_rate_mbps': 15.03,
        },
    },
    'oliveira2024': {
        'label': 'Oliveira 2024',
        'metrics': ['BER', 'data_rate_mbps'],
        'expected': {
            'BER': 3.4e-3,
            'data_rate_mbps': 21.3,
        },
    },
    'xu2024': {
        'label': 'Xu 2024',
        'metrics': ['BER'],
        'expected': {
            'BER': 0.10,
        },
    },
}


# =============================================================================
# SINGLE PRESET VALIDATION
# =============================================================================

def validate_preset(preset_name, verbose=True):
    """
    Run a preset through the cosim pipeline and compare against targets.

    Returns:
        dict with keys: preset, label, pipeline_result, comparisons, passed
    """
    cfg = SystemConfig.from_preset(preset_name)
    paper = _PAPER_METRICS.get(preset_name, {})
    label = paper.get('label', preset_name)

    if verbose:
        print(f"\n  [{preset_name}] {label}")
        print(f"    Topology: {cfg.rx_topology}, Modulation: {cfg.modulation}")

    # Run pipeline
    result = run_python_simulation(cfg)

    # Extract pipeline metrics
    pipeline_metrics = {
        'channel_gain': result.get('channel_gain', 0),
        'P_rx_uW': result.get('P_rx_avg_uW', 0),
        'I_ph_uA': result.get('I_ph_avg_uA', 0),
        'BER': result.get('ber', 1.0),
        'SNR_dB': result.get('snr_est_dB', 0),
        'data_rate_mbps': cfg.data_rate_bps / 1e6,
    }

    # Compare against expected
    expected = paper.get('expected', {})
    comparisons = []
    all_pass = True

    for metric_name in paper.get('metrics', ['BER']):
        got = pipeline_metrics.get(metric_name)
        exp = expected.get(metric_name)

        if exp is None or exp == 0:
            # No target — just report
            status = 'INFO'
            error_pct = None
        elif metric_name == 'BER':
            # BER comparison: check order of magnitude
            if got == 0 and exp == 0:
                status = 'PASS'
                error_pct = 0
            elif got == 0:
                status = 'PASS'  # Got perfect, target was nonzero
                error_pct = -100
            elif exp == 0:
                status = 'FAIL' if got > 0.01 else 'PASS'
                error_pct = float('inf')
            else:
                ratio = got / exp
                # BER within 1 order of magnitude = PASS
                status = 'PASS' if 0.1 <= ratio <= 10 else 'REVIEW'
                error_pct = abs(ratio - 1) * 100
                if status == 'REVIEW':
                    all_pass = False
        else:
            # Numeric comparison: 20% tolerance
            error_pct = abs(got - exp) / abs(exp) * 100 if exp != 0 else 0
            status = 'PASS' if error_pct < 20 else 'REVIEW'
            if status == 'REVIEW':
                all_pass = False

        comp = {
            'metric': metric_name,
            'pipeline': got,
            'expected': exp,
            'error_pct': error_pct,
            'status': status,
        }
        comparisons.append(comp)

        if verbose:
            exp_str = f"{exp:.4e}" if exp is not None else "N/A"
            got_str = f"{got:.4e}" if got is not None else "N/A"
            err_str = f"{error_pct:.1f}%" if error_pct is not None else ""
            print(f"    {metric_name:20s}  pipeline={got_str:>12s}  "
                  f"target={exp_str:>12s}  {err_str:>8s}  {status}")

    return {
        'preset': preset_name,
        'label': label,
        'pipeline_result': pipeline_metrics,
        'comparisons': comparisons,
        'passed': all_pass,
    }


# =============================================================================
# VALIDATE ALL PRESETS
# =============================================================================

def validate_all(output_dir=None, verbose=True):
    """
    Run all 7 presets through the pipeline and generate comparison report.

    Returns:
        dict: {preset_name: validation_result}
    """
    if output_dir is None:
        output_dir = os.path.join('workspace', 'validation_pipeline')
    os.makedirs(output_dir, exist_ok=True)

    if verbose:
        print("\n" + "=" * 70)
        print("  COSIM PIPELINE VALIDATION — All Paper Presets")
        print("=" * 70)

    results = {}
    presets = SystemConfig.list_presets()

    for name in presets:
        results[name] = validate_preset(name, verbose=verbose)

    # Summary
    n_pass = sum(1 for r in results.values() if r['passed'])
    n_total = len(results)

    if verbose:
        print("\n" + "=" * 70)
        print("  SUMMARY")
        print("=" * 70)
        for name, r in results.items():
            status = "PASS" if r['passed'] else "REVIEW"
            print(f"  {status:6s}  {r['label']}")
        print(f"\n  {n_pass}/{n_total} passed")

    # Generate comparison figures
    if output_dir:
        _plot_comparison_summary(results, output_dir)
        _plot_per_paper_details(results, output_dir)
        _plot_radar_summary(results, output_dir)

    return results


# =============================================================================
# CROSS-VALIDATION: STANDALONE vs PIPELINE
# =============================================================================

def cross_validate(output_dir=None, verbose=True):
    """
    Compare standalone paper validation metrics vs pipeline metrics.

    Returns:
        dict with per-paper comparison
    """
    if output_dir is None:
        output_dir = os.path.join('workspace', 'validation_pipeline')
    os.makedirs(output_dir, exist_ok=True)

    if verbose:
        print("\n" + "=" * 70)
        print("  CROSS-VALIDATION: Standalone Scripts vs Cosim Pipeline")
        print("=" * 70)

    comparisons = {}

    # Kadirvelu: compare channel gain and link budget
    cfg = SystemConfig.from_preset('kadirvelu2021')
    pipeline = run_python_simulation(
        SystemConfig(**{**cfg.to_dict(), 'n_bits': 50, 'simulation_engine': 'python'}))

    from papers.kadirvelu_2021 import optical_channel_gain, received_power_W
    standalone_G = optical_channel_gain()
    standalone_P = received_power_W()
    pipeline_G = pipeline['channel_gain']
    pipeline_P = pipeline['P_rx_avg_uW'] * 1e-6  # Convert back to W

    comparisons['kadirvelu2021'] = {
        'channel_gain': {'standalone': standalone_G, 'pipeline': pipeline_G,
                         'ratio': pipeline_G / standalone_G if standalone_G else 0},
        'P_rx_W': {'standalone': standalone_P, 'pipeline': pipeline_P,
                    'ratio': pipeline_P / standalone_P if standalone_P else 0},
    }

    if verbose:
        print(f"\n  [kadirvelu2021]")
        print(f"    Channel gain:  standalone={standalone_G:.6e}  pipeline={pipeline_G:.6e}  "
              f"ratio={pipeline_G/standalone_G:.4f}")
        print(f"    P_rx (W):      standalone={standalone_P:.6e}  pipeline={pipeline_P:.6e}  "
              f"ratio={pipeline_P/standalone_P:.4f}")

    # González: compare bandwidth
    from papers.gonzalez_2024 import bandwidth as gz_bandwidth
    gz_cfg = SystemConfig.from_preset('gonzalez2024')
    C_j = 14.5e-9
    R_sh = 200e3
    R_load = 220.0
    bw_standalone = gz_bandwidth(R_load, C_j, R_sh)

    comparisons['gonzalez2024'] = {
        'bandwidth_Hz': {'standalone': bw_standalone, 'pipeline': 'N/A (not computed)'},
    }
    if verbose:
        print(f"\n  [gonzalez2024]")
        print(f"    Bandwidth:     standalone={bw_standalone/1e3:.1f} kHz")

    # Correa: compare received power at nominal distance
    from papers.correa_2025 import compute_received_power
    cr_cfg = SystemConfig.from_preset('correa2025')
    cr_P_standalone = compute_received_power(3.0, 0.85, 66e-4, 0.50, m=1)
    cr_pipeline = run_python_simulation(
        SystemConfig(**{**cr_cfg.to_dict(), 'n_bits': 20, 'simulation_engine': 'python'}))
    cr_P_pipeline = cr_pipeline['P_rx_avg_uW'] * 1e-6

    comparisons['correa2025'] = {
        'P_rx_W': {'standalone': cr_P_standalone, 'pipeline': cr_P_pipeline,
                    'ratio': cr_P_pipeline / cr_P_standalone if cr_P_standalone else 0},
    }
    if verbose:
        print(f"\n  [correa2025]")
        print(f"    P_rx (W):      standalone={cr_P_standalone:.6e}  pipeline={cr_P_pipeline:.6e}  "
              f"ratio={cr_P_pipeline/cr_P_standalone:.4f}" if cr_P_standalone else
              f"    P_rx (W):      standalone=0  pipeline={cr_P_pipeline:.6e}")

    if output_dir:
        _plot_cross_validation(comparisons, output_dir)

    return comparisons


# =============================================================================
# FIGURES
# =============================================================================

def _plot_comparison_summary(results, output_dir):
    """Summary bar chart: pipeline BER vs target BER for each paper."""
    papers = []
    pipeline_bers = []
    target_bers = []
    colors = []

    for name, r in results.items():
        for comp in r['comparisons']:
            if comp['metric'] == 'BER':
                papers.append(r['label'])
                pipeline_bers.append(max(comp['pipeline'], 1e-6))
                target_bers.append(max(comp['expected'], 1e-6) if comp['expected'] else 1e-6)
                colors.append('#2ca02c' if comp['status'] == 'PASS' else '#d62728')
                break

    if not papers:
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(papers))
    w = 0.35

    ax.bar(x - w/2, pipeline_bers, w, label='Pipeline', color='steelblue', alpha=0.8)
    ax.bar(x + w/2, target_bers, w, label='Paper Target', color='coral', alpha=0.8)

    ax.set_yscale('log')
    ax.set_xticks(x)
    ax.set_xticklabels(papers, rotation=30, ha='right', fontsize=10)
    ax.set_ylabel('BER', fontsize=12)
    ax.set_title('Pipeline BER vs Paper Targets', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, which='both', alpha=0.3, axis='y')
    ax.set_ylim([1e-6, 1])

    plt.tight_layout()
    path = os.path.join(output_dir, 'pipeline_vs_targets_ber.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {path}")


def _plot_per_paper_details(results, output_dir):
    """Per-paper detail figures showing all metric comparisons."""
    for name, r in results.items():
        comps = r['comparisons']
        numeric = [(c['metric'], c['pipeline'], c['expected'], c['status'])
                   for c in comps if c['expected'] is not None and c['expected'] != 0]
        if not numeric:
            continue

        metrics = [m for m, _, _, _ in numeric]
        pipeline_vals = [p for _, p, _, _ in numeric]
        target_vals = [e for _, _, e, _ in numeric]
        statuses = [s for _, _, _, s in numeric]

        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(len(metrics))
        w = 0.35

        bars_p = ax.bar(x - w/2, pipeline_vals, w, label='Pipeline', color='steelblue', alpha=0.8)
        bars_t = ax.bar(x + w/2, target_vals, w, label='Target', color='coral', alpha=0.8)

        # Color-code by status
        for i, st in enumerate(statuses):
            color = '#2ca02c' if st == 'PASS' else '#d62728' if st in ('FAIL', 'REVIEW') else '#999999'
            bars_p[i].set_edgecolor(color)
            bars_p[i].set_linewidth(2)

        ax.set_xticks(x)
        ax.set_xticklabels(metrics, fontsize=10)
        ax.set_title(f"{r['label']} — Pipeline vs Target", fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')

        # Use log scale if values span orders of magnitude
        all_vals = [v for v in pipeline_vals + target_vals if v > 0]
        if all_vals and max(all_vals) / max(min(all_vals), 1e-30) > 100:
            ax.set_yscale('log')

        plt.tight_layout()
        path = os.path.join(output_dir, f'detail_{name}.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"    Saved: {path}")


def _plot_radar_summary(results, output_dir):
    """Radar chart showing pass/review/fail status for each paper."""
    papers = list(results.keys())
    labels = [results[p]['label'] for p in papers]
    n = len(papers)

    # Score: 1.0 for PASS, 0.5 for INFO, 0.0 for REVIEW/FAIL
    scores = []
    for p in papers:
        comps = results[p]['comparisons']
        if not comps:
            scores.append(0.5)
            continue
        sc = []
        for c in comps:
            if c['status'] == 'PASS':
                sc.append(1.0)
            elif c['status'] == 'INFO':
                sc.append(0.5)
            else:
                sc.append(0.0)
        scores.append(np.mean(sc))

    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    scores_closed = scores + [scores[0]]
    angles_closed = angles + [angles[0]]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.fill(angles_closed, scores_closed, alpha=0.25, color='steelblue')
    ax.plot(angles_closed, scores_closed, 'o-', color='steelblue', linewidth=2)

    ax.set_xticks(angles)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['0', '0.25', '0.5', '0.75', '1.0'], fontsize=8)
    ax.set_title('Pipeline Validation Radar', fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    path = os.path.join(output_dir, 'radar_summary.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {path}")


def _plot_cross_validation(comparisons, output_dir):
    """Cross-validation: standalone vs pipeline channel gain / power."""
    metrics = []
    standalone_vals = []
    pipeline_vals = []

    for paper, comps in comparisons.items():
        for metric, data in comps.items():
            if isinstance(data, dict) and isinstance(data.get('standalone'), (int, float)) \
                    and isinstance(data.get('pipeline'), (int, float)):
                metrics.append(f"{paper}\n{metric}")
                standalone_vals.append(data['standalone'])
                pipeline_vals.append(data['pipeline'])

    if not metrics:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(metrics))
    w = 0.35

    ax.bar(x - w/2, standalone_vals, w, label='Standalone Script', color='#4472C4')
    ax.bar(x + w/2, pipeline_vals, w, label='Cosim Pipeline', color='#ED7D31')

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=9)
    ax.set_ylabel('Value')
    ax.set_title('Cross-Validation: Standalone vs Pipeline', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    path = os.path.join(output_dir, 'cross_validation.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {path}")
