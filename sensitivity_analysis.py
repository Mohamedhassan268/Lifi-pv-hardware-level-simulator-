#!/usr/bin/env python3
"""
Parameter Sensitivity Analysis
===============================

Demonstrates the hardware-faithfulness of the simulator by sweeping
key material and system parameters through the full Python pipeline
and showing that outputs respond as physics predicts.

Sweeps across 3 representative presets:
    - Kadirvelu 2021 (OOK, full analog chain)
    - Sarwar 2017    (OFDM, direct topology)
    - Xu 2024        (BFSK, direct topology)

Generates publication-quality figures:
    1. Individual parameter sweeps (BER & SNR vs parameter)
    2. Tornado chart (normalized sensitivity ranking)
    3. Multi-preset consistency comparison
    4. Temperature cascade (Varshni -> bandgap -> dark current -> BER)

Usage:
    python sensitivity_analysis.py                    # All figures
    python sensitivity_analysis.py --preset kadirvelu2021  # Single preset
    python sensitivity_analysis.py --fig tornado      # Single figure

CLI integration:
    python cli.py sensitivity                         # All figures
    python cli.py sensitivity --preset sarwar2017     # Single preset
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from pathlib import Path
from dataclasses import asdict
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import warnings
import copy
import json
from scipy.special import erfc

# Project imports
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cosim.system_config import SystemConfig
from cosim.python_engine import run_python_simulation


# =============================================================================
# CONFIGURATION
# =============================================================================

# Presets to analyze
PRESETS = ['kadirvelu2021', 'sarwar2017', 'gonzalez2024']

PRESET_LABELS = {
    'kadirvelu2021': 'Kadirvelu 2021\n(OOK, 5 kbps)',
    'sarwar2017':    'Sarwar 2017\n(OFDM, 15 Mbps)',
    'gonzalez2024':  'Gonzalez 2024\n(Manchester, 4.8 kbps)',
}

PRESET_COLORS = {
    'kadirvelu2021': '#1f77b4',
    'sarwar2017':    '#2ca02c',
    'gonzalez2024':  '#d62728',
}

# Parameters to sweep: (field_name, display_name, unit, sweep_factors, category)
# sweep_factors are multipliers relative to baseline (1.0 = nominal)
SWEEP_PARAMS = [
    # Material / Physics parameters
    ('sc_responsivity',       'PV Responsivity',       'A/W',       np.linspace(0.3, 1.5, 13), 'material'),
    ('distance_m',            'TX-RX Distance',        'm',         np.linspace(0.5, 2.0, 13), 'system'),
    ('led_half_angle_deg',    'LED Half-Angle',        'deg',       np.linspace(3, 60, 13),     'material'),
    ('bias_current_A',        'LED Bias Current',      'A',         np.linspace(0.3, 3.0, 13), 'system'),
    ('temperature_K',         'Temperature',           'K',         np.linspace(250, 400, 13),  'material'),
    ('r_sense_ohm',           'Sense Resistance',      'Ohm',       np.linspace(0.3, 3.0, 13), 'system'),
    ('modulation_depth',      'Modulation Depth',      '',          np.linspace(0.1, 1.0, 13),  'system'),
    ('sc_area_cm2',           'PV Active Area',        'cm2',       np.linspace(0.3, 2.0, 13), 'material'),
]

# Noise-specific sweeps (only for presets with noise_enable=True)
NOISE_SWEEP_PARAMS = [
    ('ina_noise_nV_rtHz',     'Amplifier Noise',       'nV/rtHz',   np.linspace(0.2, 5.0, 13), 'system'),
]

# Plot styling
plt.rcParams.update({
    'font.size': 11, 'axes.labelsize': 12, 'axes.titlesize': 13,
    'legend.fontsize': 9, 'xtick.labelsize': 10, 'ytick.labelsize': 10,
    'figure.dpi': 150, 'savefig.dpi': 200, 'axes.grid': True, 'grid.alpha': 0.3,
    'font.family': 'serif',
})


# =============================================================================
# CORE SWEEP ENGINE
# =============================================================================

def _make_config(preset_name: str, overrides: dict = None) -> SystemConfig:
    """Load a preset and apply overrides for sensitivity sweep."""
    cfg = SystemConfig.from_preset(preset_name)
    d = cfg.to_dict()
    # Force Python engine and enable noise for sensitivity
    d['simulation_engine'] = 'python'
    d['noise_enable'] = True
    d['random_seed'] = 42
    if overrides:
        d.update(overrides)
    return SystemConfig(**d)


def _compute_link_snr(cfg) -> float:
    """
    Compute link-budget SNR from config using the noise model.

    This is a physics-based SNR that accounts for temperature, noise sources,
    received power, and responsivity — more faithful than the pipeline's
    variance-based estimate.
    """
    from cosim.channel import OpticalChannel
    from cosim.noise import NoiseModel

    channel = OpticalChannel.from_config(cfg)
    G_ch = channel.channel_gain()

    # Estimate P_tx from bias current and LED efficiency
    P_tx = cfg.led_gled * cfg.bias_current_A * cfg.modulation_depth
    P_rx = P_tx * G_ch
    I_ph = cfg.sc_responsivity * P_rx

    if I_ph <= 0:
        return -200.0

    noise_model = NoiseModel.from_config(cfg)
    bandwidth = cfg.data_rate_bps / 2

    noise_std = noise_model.total_noise_std(I_ph, bandwidth)
    if noise_std <= 0:
        return 200.0

    snr_linear = (I_ph ** 2) / (noise_std ** 2)
    return 10 * np.log10(max(snr_linear, 1e-30))


def sweep_parameter(preset_name: str, param_name: str,
                    values: np.ndarray, use_factors: bool = False) -> Dict:
    """
    Sweep a single parameter through the pipeline.

    Args:
        preset_name: Preset to use as baseline
        param_name: SystemConfig field to sweep
        values: Either absolute values or multiplier factors
        use_factors: If True, values are multipliers of the baseline value

    Returns:
        Dict with 'values', 'ber', 'snr_dB', 'P_rx_uW', 'baseline_value'
    """
    baseline_cfg = _make_config(preset_name)
    baseline_val = getattr(baseline_cfg, param_name)

    if use_factors:
        abs_values = baseline_val * values
    else:
        abs_values = values

    ber_list = []
    snr_list = []
    prx_list = []

    for val in abs_values:
        try:
            cfg = _make_config(preset_name, {param_name: float(val)})
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = run_python_simulation(cfg)

            ber_list.append(result['ber'])
            prx_list.append(result['P_rx_avg_uW'])

            # Use link-budget SNR (physics-based, temperature-aware)
            snr = _compute_link_snr(cfg)
            snr_list.append(snr)

        except Exception as e:
            ber_list.append(np.nan)
            snr_list.append(np.nan)
            prx_list.append(np.nan)

    return {
        'param': param_name,
        'values': abs_values,
        'ber': np.array(ber_list),
        'snr_dB': np.array(snr_list),
        'P_rx_uW': np.array(prx_list),
        'baseline_value': baseline_val,
    }


def compute_sensitivity(sweep_result: Dict) -> float:
    """
    Compute normalized sensitivity coefficient using SNR.

    S = |delta_SNR_dB| / |delta_param / param_nominal|

    SNR is used instead of BER because it responds reliably to parameter
    changes even when the demodulator is saturated (BER ~ 0 or ~ 0.5).

    Uses the steepest finite-difference slope around the baseline.
    """
    values = sweep_result['values']
    snr = sweep_result['snr_dB']
    baseline = sweep_result['baseline_value']

    # Find index closest to baseline
    idx_base = np.argmin(np.abs(values - baseline))
    snr_base = snr[idx_base]

    if np.isnan(snr_base) or baseline == 0:
        return 0.0

    # Compute max |dSNR| / |dparam/param| across neighbors
    max_S = 0.0
    for i in range(len(values)):
        if i == idx_base or np.isnan(snr[i]):
            continue
        dparam = (values[i] - baseline) / baseline
        dsnr = snr[i] - snr_base  # in dB, already log-scale
        if abs(dparam) > 1e-6:
            S = abs(dsnr / dparam)
            max_S = max(max_S, S)

    return max_S


# =============================================================================
# FIGURE 1: INDIVIDUAL PARAMETER SWEEPS
# =============================================================================

def plot_parameter_sweeps(preset_name: str, output_dir: Path,
                          sweep_results: Dict[str, Dict]) -> None:
    """Plot SNR and P_rx vs each swept parameter for one preset."""
    params_to_plot = [p for p in SWEEP_PARAMS if p[0] in sweep_results]
    n_params = len(params_to_plot)
    if n_params == 0:
        return

    n_cols = 2
    n_rows = (n_params + 1) // 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 3.5 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    label = PRESET_LABELS.get(preset_name, preset_name)
    color = PRESET_COLORS.get(preset_name, '#333333')

    for idx, (param_name, display_name, unit, _, category) in enumerate(params_to_plot):
        row, col = idx // 2, idx % 2
        ax = axes[row, col]
        sr = sweep_results[param_name]

        # SNR on left y-axis
        valid = ~np.isnan(sr['snr_dB'])
        if np.any(valid):
            ax.plot(sr['values'][valid], sr['snr_dB'][valid],
                   '-o', color=color, markersize=4, linewidth=2, label='SNR (dB)')

        # P_rx on right y-axis
        ax2 = ax.twinx()
        valid_prx = ~np.isnan(sr['P_rx_uW']) & (sr['P_rx_uW'] > 0)
        if np.any(valid_prx):
            ax2.plot(sr['values'][valid_prx], sr['P_rx_uW'][valid_prx],
                    '--s', color='gray', markersize=3, linewidth=1.5, alpha=0.6, label='P_rx (uW)')
            ax2.set_ylabel('P_rx (uW)', color='gray', fontsize=9)
            ax2.tick_params(axis='y', labelcolor='gray', labelsize=8)

        # Mark baseline
        ax.axvline(x=sr['baseline_value'], color='gray', linestyle='--',
                  linewidth=1.5, alpha=0.7, label=f'Nominal ({sr["baseline_value"]:.3g})')

        xlabel = f'{display_name} ({unit})' if unit else display_name
        ax.set_xlabel(xlabel, fontweight='bold')
        ax.set_ylabel('SNR (dB)', fontweight='bold', color=color)
        ax.tick_params(axis='y', labelcolor=color)
        ax.legend(loc='upper left', fontsize=8)

        # Category badge
        badge_color = '#2196F3' if category == 'material' else '#FF9800'
        badge_text = 'PHYSICS' if category == 'material' else 'DESIGN'
        ax.text(0.02, 0.95, badge_text, transform=ax.transAxes,
               fontsize=7, fontweight='bold', color='white',
               bbox=dict(boxstyle='round,pad=0.3', facecolor=badge_color, alpha=0.8),
               verticalalignment='top')

    # Hide unused axes
    for idx in range(n_params, n_rows * n_cols):
        row, col = idx // 2, idx % 2
        axes[row, col].set_visible(False)

    fig.suptitle(f'Parameter Sensitivity: {label.replace(chr(10), " ")}',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    fname = output_dir / f'sweeps_{preset_name}.png'
    plt.savefig(fname, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fname}")


# =============================================================================
# FIGURE 2: TORNADO CHART
# =============================================================================

def plot_tornado(sensitivities: Dict[str, Dict[str, float]],
                 output_dir: Path) -> None:
    """
    Tornado chart ranking parameters by sensitivity across presets.
    """
    # Gather all params and compute average sensitivity across presets
    all_params = set()
    for preset_sens in sensitivities.values():
        all_params.update(preset_sens.keys())

    param_avg = {}
    for param in all_params:
        vals = [sensitivities[p].get(param, 0) for p in PRESETS if p in sensitivities]
        param_avg[param] = np.mean(vals) if vals else 0

    # Sort by average sensitivity
    sorted_params = sorted(param_avg.keys(), key=lambda p: param_avg[p])

    # Map param names to display names
    display_map = {p[0]: p[1] for p in SWEEP_PARAMS + NOISE_SWEEP_PARAMS}
    category_map = {p[0]: p[4] for p in SWEEP_PARAMS + NOISE_SWEEP_PARAMS}

    fig, ax = plt.subplots(figsize=(12, max(5, 0.6 * len(sorted_params))))

    y_pos = np.arange(len(sorted_params))
    bar_height = 0.25

    for i, preset in enumerate(PRESETS):
        if preset not in sensitivities:
            continue
        vals = [sensitivities[preset].get(p, 0) for p in sorted_params]
        color = PRESET_COLORS.get(preset, '#333')
        short_label = preset.replace('2021', "'21").replace('2017', "'17").replace('2024', "'24")
        ax.barh(y_pos + (i - 1) * bar_height, vals, bar_height,
                color=color, alpha=0.85, label=short_label, edgecolor='white', linewidth=0.5)

    # Y-axis labels with category coloring
    labels = []
    for p in sorted_params:
        name = display_map.get(p, p)
        cat = category_map.get(p, 'system')
        labels.append(name)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)

    # Color y-tick labels by category
    for i, p in enumerate(sorted_params):
        cat = category_map.get(p, 'system')
        color = '#2196F3' if cat == 'material' else '#FF9800'
        ax.get_yticklabels()[i].set_color(color)
        ax.get_yticklabels()[i].set_fontweight('bold')

    ax.set_xlabel('Normalized Sensitivity |S|', fontweight='bold', fontsize=13)
    ax.set_title('Parameter Sensitivity Ranking (Tornado Chart)\n'
                 'Blue = Physics/Material parameter, Orange = System/Design parameter',
                 fontsize=13, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.axvline(x=1.0, color='gray', linestyle=':', linewidth=1, alpha=0.5)

    plt.tight_layout()
    fname = output_dir / 'tornado_chart.png'
    plt.savefig(fname, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fname}")


# =============================================================================
# FIGURE 3: MULTI-PRESET CONSISTENCY
# =============================================================================

def plot_multi_preset_comparison(all_sweeps: Dict[str, Dict[str, Dict]],
                                  output_dir: Path) -> None:
    """
    Show that key parameters affect all presets consistently.
    Pick 4 key parameters and show BER response across all 3 presets.
    """
    key_params = ['sc_responsivity', 'distance_m', 'temperature_K', 'modulation_depth']
    key_params = [p for p in key_params if any(p in all_sweeps.get(pr, {}) for pr in PRESETS)]

    if not key_params:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    display_map = {p[0]: (p[1], p[2]) for p in SWEEP_PARAMS}

    for idx, param in enumerate(key_params[:4]):
        ax = axes[idx]
        name, unit = display_map.get(param, (param, ''))

        for preset in PRESETS:
            if preset not in all_sweeps or param not in all_sweeps[preset]:
                continue
            sr = all_sweeps[preset][param]
            valid = ~np.isnan(sr['snr_dB'])
            if not np.any(valid):
                continue

            color = PRESET_COLORS[preset]
            short = preset.replace('2021', "'21").replace('2017', "'17").replace('2024', "'24")

            # Normalize x-axis to fraction of baseline for comparison
            x_norm = sr['values'][valid] / sr['baseline_value']
            # Normalize SNR to delta from baseline for cross-preset comparison
            idx_base = np.argmin(np.abs(sr['values'] - sr['baseline_value']))
            snr_base = sr['snr_dB'][idx_base]
            delta_snr = sr['snr_dB'][valid] - snr_base
            ax.plot(x_norm, delta_snr, '-o', color=color,
                   markersize=4, linewidth=2, label=short)

        ax.axvline(x=1.0, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
        ax.axhline(y=0, color='gray', linestyle=':', linewidth=1, alpha=0.5)
        ax.set_xlabel(f'{name} (normalized to baseline)', fontweight='bold')
        ax.set_ylabel('Delta SNR (dB)', fontweight='bold')
        ax.set_title(name, fontweight='bold')
        ax.legend(loc='best', fontsize=9)

    for idx in range(len(key_params), 4):
        axes[idx].set_visible(False)

    fig.suptitle('Multi-Preset Sensitivity Consistency\n'
                 'Same physics parameters produce consistent BER response across systems',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    fname = output_dir / 'multi_preset_comparison.png'
    plt.savefig(fname, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fname}")


# =============================================================================
# FIGURE 4: TEMPERATURE CASCADE
# =============================================================================

def plot_temperature_cascade(output_dir: Path) -> None:
    """
    Show how temperature change cascades through the physics:
    T -> Eg(T) via Varshni -> n_i(T) -> I_dark(T) -> noise -> BER
    """
    from materials.semiconductors import GAAS, SILICON

    temps = np.linspace(250, 400, 30)

    # Physics cascades
    eg_gaas = [GAAS.bandgap(T) for T in temps]
    eg_si = [SILICON.bandgap(T) for T in temps]
    ni_gaas = [GAAS.intrinsic_concentration(T) for T in temps]
    ni_si = [SILICON.intrinsic_concentration(T) for T in temps]

    # Run link-budget SNR at each temperature
    snr_per_preset = {}

    for preset in PRESETS:
        snrs = []
        for T in temps:
            try:
                cfg = _make_config(preset, {'temperature_K': float(T)})
                snr = _compute_link_snr(cfg)
                snrs.append(snr)
            except Exception:
                snrs.append(np.nan)
        snr_per_preset[preset] = np.array(snrs)

    # Plot 4-panel cascade
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: Bandgap vs Temperature (Varshni)
    ax = axes[0, 0]
    ax.plot(temps, eg_gaas, 'r-', linewidth=2, label='GaAs')
    ax.plot(temps, eg_si, 'b-', linewidth=2, label='Si')
    ax.set_xlabel('Temperature (K)', fontweight='bold')
    ax.set_ylabel('Bandgap (eV)', fontweight='bold')
    ax.set_title('(a) Varshni: Eg(T) = Eg(0) - aT^2/(T+b)', fontweight='bold')
    ax.legend()

    # Panel 2: Intrinsic carrier concentration
    ax = axes[0, 1]
    ax.semilogy(temps, ni_gaas, 'r-', linewidth=2, label='GaAs')
    ax.semilogy(temps, ni_si, 'b-', linewidth=2, label='Si')
    ax.set_xlabel('Temperature (K)', fontweight='bold')
    ax.set_ylabel('n_i (cm^-3)', fontweight='bold')
    ax.set_title('(b) Intrinsic Concentration: n_i ~ exp(-Eg/2kT)', fontweight='bold')
    ax.legend()

    # Panel 3: SNR vs Temperature
    ax = axes[1, 0]
    for preset in PRESETS:
        color = PRESET_COLORS[preset]
        short = preset.replace('2021', "'21").replace('2017', "'17").replace('2024', "'24")
        valid = ~np.isnan(snr_per_preset[preset]) & (snr_per_preset[preset] > -100)
        if not np.any(valid):
            continue
        ax.plot(temps[valid], snr_per_preset[preset][valid], '-o',
               color=color, markersize=3, linewidth=2, label=short)
    ax.set_xlabel('Temperature (K)', fontweight='bold')
    ax.set_ylabel('SNR (dB)', fontweight='bold')
    ax.set_title('(c) SNR vs Temperature (shot-noise limited)', fontweight='bold')
    ax.legend()

    # Panel 4: Analytical BER vs Temperature (from SNR)
    ax = axes[1, 1]
    for preset in PRESETS:
        color = PRESET_COLORS[preset]
        short = preset.replace('2021', "'21").replace('2017', "'17").replace('2024', "'24")
        valid = ~np.isnan(snr_per_preset[preset]) & (snr_per_preset[preset] > -100)
        if not np.any(valid):
            continue
        # Analytical BER from SNR (OOK approximation)
        snr_lin = 10 ** (snr_per_preset[preset][valid] / 10)
        ber_analytical = 0.5 * erfc(np.sqrt(snr_lin / 2))
        ber_analytical = np.maximum(ber_analytical, 1e-15)
        ax.semilogy(temps[valid], ber_analytical, '-o',
                   color=color, markersize=3, linewidth=2, label=short)
    ax.set_xlabel('Temperature (K)', fontweight='bold')
    ax.set_ylabel('BER (analytical)', fontweight='bold')
    ax.set_title('(d) BER: End-to-End Temperature Cascade', fontweight='bold')
    ax.axhline(y=1e-3, color='gray', linestyle='--', linewidth=1, alpha=0.7, label='BER=10^-3')
    ax.legend()

    fig.suptitle('Temperature Cascade: Material Physics -> System Performance\n'
                 'Demonstrates hardware-faithful parameter propagation',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    fname = output_dir / 'temperature_cascade.png'
    plt.savefig(fname, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fname}")


# =============================================================================
# SUMMARY TABLE
# =============================================================================

def generate_summary_table(sensitivities: Dict[str, Dict[str, float]],
                            output_dir: Path) -> None:
    """Generate a LaTeX-ready sensitivity table."""
    display_map = {p[0]: p[1] for p in SWEEP_PARAMS + NOISE_SWEEP_PARAMS}
    unit_map = {p[0]: p[2] for p in SWEEP_PARAMS + NOISE_SWEEP_PARAMS}
    category_map = {p[0]: p[4] for p in SWEEP_PARAMS + NOISE_SWEEP_PARAMS}

    all_params = set()
    for preset_sens in sensitivities.values():
        all_params.update(preset_sens.keys())

    # Sort by average sensitivity descending
    param_avg = {}
    for param in all_params:
        vals = [sensitivities[p].get(param, 0) for p in PRESETS if p in sensitivities]
        param_avg[param] = np.mean(vals) if vals else 0
    sorted_params = sorted(all_params, key=lambda p: param_avg[p], reverse=True)

    # Write CSV
    csv_path = output_dir / 'sensitivity_table.csv'
    with open(csv_path, 'w') as f:
        header = ['Parameter', 'Unit', 'Category']
        for preset in PRESETS:
            if preset in sensitivities:
                header.append(f'|S| ({preset})')
        header.append('|S| (avg)')
        f.write(','.join(header) + '\n')

        for param in sorted_params:
            name = display_map.get(param, param)
            unit = unit_map.get(param, '')
            cat = category_map.get(param, 'system')
            row = [name, unit, cat]
            for preset in PRESETS:
                if preset in sensitivities:
                    row.append(f'{sensitivities[preset].get(param, 0):.3f}')
            row.append(f'{param_avg[param]:.3f}')
            f.write(','.join(row) + '\n')

    print(f"  Saved: {csv_path}")

    # Write LaTeX
    tex_path = output_dir / 'sensitivity_table.tex'
    with open(tex_path, 'w') as f:
        n_presets = sum(1 for p in PRESETS if p in sensitivities)
        col_spec = 'l l c ' + ' '.join(['c'] * n_presets) + ' c'
        f.write(f'\\begin{{tabular}}{{{col_spec}}}\n')
        f.write('\\toprule\n')
        header_parts = ['Parameter', 'Unit', 'Type']
        for preset in PRESETS:
            if preset in sensitivities:
                short = preset.split('2')[0].title()
                header_parts.append(f'$|S|$ ({short})')
        header_parts.append('$|S|$ (Avg)')
        f.write(' & '.join(header_parts) + ' \\\\\n')
        f.write('\\midrule\n')

        for param in sorted_params:
            name = display_map.get(param, param)
            unit = unit_map.get(param, '--')
            cat = 'P' if category_map.get(param, 'system') == 'material' else 'D'
            parts = [name, unit, cat]
            for preset in PRESETS:
                if preset in sensitivities:
                    parts.append(f'{sensitivities[preset].get(param, 0):.2f}')
            parts.append(f'\\textbf{{{param_avg[param]:.2f}}}')
            f.write(' & '.join(parts) + ' \\\\\n')

        f.write('\\bottomrule\n')
        f.write('\\end{tabular}\n')

    print(f"  Saved: {tex_path}")


# =============================================================================
# MAIN ORCHESTRATOR
# =============================================================================

def run_sensitivity_analysis(presets: List[str] = None,
                              output_dir: str = None,
                              fig: str = 'all') -> str:
    """
    Run the full sensitivity analysis.

    Args:
        presets: List of preset names (default: all 3)
        output_dir: Output directory (auto-generated if None)
        fig: Which figure(s) to generate: 'all', 'sweeps', 'tornado',
             'comparison', 'temperature'

    Returns:
        Output directory path
    """
    if presets is None:
        presets = PRESETS

    if output_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = f'workspace/sensitivity_{timestamp}'

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("  PARAMETER SENSITIVITY ANALYSIS")
    print("  Hardware-Faithfulness Demonstration")
    print("=" * 70)
    print(f"\n  Presets:  {', '.join(presets)}")
    print(f"  Output:   {out}")
    print(f"  Figures:  {fig}")
    print()

    # ---- Run all sweeps ----
    all_sweeps = {}       # preset -> param -> sweep_result
    all_sensitivities = {}  # preset -> param -> S

    for preset in presets:
        print(f"\n--- Sweeping: {preset} ---")
        all_sweeps[preset] = {}
        all_sensitivities[preset] = {}

        # Determine which params to sweep
        params = list(SWEEP_PARAMS)
        baseline = _make_config(preset)
        if baseline.noise_enable or True:  # always enable noise for sensitivity
            params += NOISE_SWEEP_PARAMS

        for param_name, display_name, unit, default_values, category in params:
            baseline_val = getattr(baseline, param_name)

            # Skip params that are zero or not applicable
            if baseline_val == 0 and param_name not in ('temperature_K',):
                print(f"  Skip {display_name} (baseline=0)")
                continue

            # Compute absolute sweep values
            if param_name == 'temperature_K':
                values = default_values  # Already absolute
            elif param_name == 'distance_m':
                values = default_values  # Already absolute
            elif param_name == 'led_half_angle_deg':
                values = default_values  # Already absolute
            else:
                # Use factors as multipliers of baseline
                values = baseline_val * default_values

            print(f"  Sweeping {display_name} ({len(values)} points)...", end=' ', flush=True)
            sr = sweep_parameter(preset, param_name, values)
            all_sweeps[preset][param_name] = sr

            S = compute_sensitivity(sr)
            all_sensitivities[preset][param_name] = S
            print(f"|S| = {S:.3f}")

    # ---- Generate figures ----
    print(f"\n--- Generating Figures ---")

    if fig in ('all', 'sweeps'):
        for preset in presets:
            if preset in all_sweeps:
                plot_parameter_sweeps(preset, out, all_sweeps[preset])

    if fig in ('all', 'tornado'):
        plot_tornado(all_sensitivities, out)

    if fig in ('all', 'comparison'):
        plot_multi_preset_comparison(all_sweeps, out)

    if fig in ('all', 'temperature'):
        print("  Generating temperature cascade (may take a moment)...")
        plot_temperature_cascade(out)

    if fig in ('all', 'table'):
        generate_summary_table(all_sensitivities, out)

    # Save raw data as JSON
    raw_data = {}
    for preset in presets:
        raw_data[preset] = {}
        for param, sr in all_sweeps.get(preset, {}).items():
            raw_data[preset][param] = {
                'values': sr['values'].tolist(),
                'ber': [float(x) if not np.isnan(x) else None for x in sr['ber']],
                'snr_dB': [float(x) if not np.isnan(x) else None for x in sr['snr_dB']],
                'baseline': float(sr['baseline_value']),
                'sensitivity': float(all_sensitivities.get(preset, {}).get(param, 0)),
            }
    (out / 'sensitivity_data.json').write_text(json.dumps(raw_data, indent=2))
    print(f"  Saved: {out / 'sensitivity_data.json'}")

    print(f"\n{'=' * 70}")
    print(f"  SENSITIVITY ANALYSIS COMPLETE")
    print(f"  Output: {out}")
    print(f"{'=' * 70}\n")

    return str(out)


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Parameter Sensitivity Analysis for Hardware-Faithful LiFi-PV Simulator')
    parser.add_argument('--preset', nargs='+', default=None,
                       help='Preset(s) to analyze (default: all 3)')
    parser.add_argument('--fig', default='all',
                       choices=['all', 'sweeps', 'tornado', 'comparison', 'temperature', 'table'],
                       help='Which figure to generate')
    parser.add_argument('-o', '--output', default=None,
                       help='Output directory')
    args = parser.parse_args()

    run_sensitivity_analysis(
        presets=args.preset,
        output_dir=args.output,
        fig=args.fig,
    )
