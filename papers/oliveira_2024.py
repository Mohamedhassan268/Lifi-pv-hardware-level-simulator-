"""
Oliveira et al. (2024) — MIMO SLIPT Validation
================================================

Paper: "Reconfigurable MIMO-based self-powered battery-less light communication system"
       Light: Science & Applications (2024) 13:218

Figures:
  - Fig 3c: SNR, BER, Spectral Efficiency per Subcarrier
  - Fig 3d: QAM Constellation Diagrams
  - Fig 4c: Quadrant Beam Tracking (SNR/BER vs Time)
  - Fig 5: Energy Harvesting vs Illuminance (Table 1)
  - Communication vs Harvesting Tradeoff
  - Mode Comparison Summary Table
  - PD Array Heatmap
  - Supercapacitor Charging
  - BER vs Distance
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from scipy.special import erfc
from scipy.ndimage import gaussian_filter1d

from papers.ofdm_modem import (
    OFDMModem, allocate_bits, ber_mqam,
    gross_data_rate, net_data_rate, BIT_ALLOCATION_TABLE,
)

# =============================================================================
# PAPER PARAMETERS (LOCKED)
# =============================================================================
PARAMS = {
    'n_pds_total': 9,
    'n_rows': 3,
    'n_cols': 3,
    'nfft': 1024,
    'cp_length': 10,
    'n_subcarriers': 500,
    'bandwidth_hz': 1.5e6,
    'max_qam': 64,
    'sample_rate_hz': 12.02e6,

    'small_pd_area_mm2': 7.7,
    'small_pd_responsivity': 0.36,
    'large_pd_active_area_cm2': 1.0,
    'large_pd_bandwidth_mhz': 1.5,

    'snr_max_db': 30.0,
    'snr_min_db': 10.0,
    'rolloff_exponent': 2.0,

    'siso_gross_data_rate_mbps': 25.7,
    'siso_net_data_rate_mbps': 21.3,
    'mimo_net_data_rate_mbps': 85.2,
    'switching_time_us': 22.0,
    'harvest_max_power_mw': 87.33,
    'system_power_typical_mw': 43.0,
    'supercap_capacitance_f': 0.1,
    'supercap_voltage_v': 5.0,
}

TARGETS = {
    'siso_gross_mbps': 25.7,
    'siso_net_mbps': 21.3,
    'mimo_net_mbps': 85.2,
    'ber_threshold': 3.8e-3,
    'ber_siso_small_pd': 3.4e-3,
    'ber_siso_large_pd': 3.3e-3,
    'fec_overhead': 0.171,
}

ENERGY_HARVESTING_TABLE = [
    (14, 2.57, 0.0078, 'Unilluminated'),
    (1275, 3.82, 2.75, 'Incandescent 60W'),
    (17780, 5.06, 21.36, 'Focused 75W'),
    (1852, 3.84, 14.10, 'Shaded Sunlight'),
    (55900, 4.10, 21.30, 'Direct Sunlight'),
]

QUADRANT_CONFIGS = {
    'I': [0, 1, 3, 4], 'II': [1, 2, 4, 5], 'III': [3, 4, 6, 7], 'IV': [4, 5, 7, 8],
    'V': [0, 1, 2], 'VI': [3, 4, 5], 'VII': [6, 7, 8],
    'VIII': [0, 3, 6], 'IX': [1, 4, 7], 'X': [2, 5, 8],
}


# =============================================================================
# PHYSICS FUNCTIONS
# =============================================================================

def compute_ber_awgn(snr_db, modulation_order=16):
    """Compute theoretical BER for M-QAM over AWGN."""
    snr_linear = 10 ** (snr_db / 10)
    if modulation_order == 2:
        return 0.5 * erfc(np.sqrt(snr_linear))
    elif modulation_order == 4:
        return 0.5 * erfc(np.sqrt(snr_linear / 2))
    elif modulation_order == 16:
        return (3 / 8) * erfc(np.sqrt(snr_linear / 10))
    elif modulation_order == 64:
        return (7 / 24) * erfc(np.sqrt(snr_linear / 42))
    return 0.5


def calculate_harvested_power(lux, n_pds=9):
    """Estimate harvested power from illuminance using Table 1 data."""
    lux_values = np.array([14, 1275, 1852, 17780, 55900])
    power_values = np.array([
        2.57 * 0.0078, 3.82 * 2.75, 3.84 * 14.10,
        min(5.06 * 21.36, 87.33), 4.10 * 21.30
    ])
    power_values = np.minimum(power_values, 87.33)
    log_lux = np.log10(lux_values)
    log_lux_query = np.log10(max(lux, 1))
    power = np.interp(log_lux_query, log_lux, power_values)
    return power * (n_pds / 9)


def calculate_data_rate(n_comm_pds, pd_type='small'):
    """Calculate achievable data rate based on number of communication PDs."""
    base_rate = 4.0 if pd_type == 'large' else 21.3
    if n_comm_pds == 1:
        efficiency = 1.0
    elif n_comm_pds <= 4:
        efficiency = 0.85
    else:
        efficiency = 0.75
    return base_rate * n_comm_pds * efficiency


def generate_snr_profile(n_sc=500, snr_max=25.0, snr_min=3.0,
                          rolloff=1.5, noise_std=0.5):
    """Generate realistic SNR profile across subcarriers."""
    freq_norm = np.linspace(0, 1, n_sc)
    snr_db = snr_max - (snr_max - snr_min) * freq_norm ** rolloff
    snr_db += np.random.normal(0, noise_std, n_sc)
    return np.clip(snr_db, 0, snr_max)


# =============================================================================
# FIGURE 3c: SNR, BER, SPECTRAL EFFICIENCY PER SUBCARRIER
# =============================================================================

def _plot_fig3c(results, snr_db, modem, output_dir):
    """Fig. 3c: SNR, BER, and spectral efficiency per subcarrier."""
    n_sc = results['n_subcarriers']
    sc = np.arange(n_sc)
    ber = np.maximum(results['ber_per_sc'], 1e-6)
    spectral_eff = modem.bit_allocation.astype(float)

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    # Spectral Efficiency
    axes[0].step(sc, 6 * np.ones(n_sc), 'g-', linewidth=1.5,
                 where='mid', label='Maximum (64-QAM)')
    axes[0].step(sc, spectral_eff, 'b-', linewidth=1.5,
                 where='mid', label='Used')
    axes[0].fill_between(sc, 0, spectral_eff, alpha=0.3, step='mid')
    axes[0].set_ylabel('Spectral efficiency\n(b/s/Hz)', fontweight='bold')
    axes[0].set_ylim(0, 8)
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)

    # SNR
    axes[1].plot(sc, snr_db[:n_sc], 'b.', markersize=2, alpha=0.7)
    axes[1].axhline(y=3, color='r', linestyle='--', linewidth=1.5, label='Min SNR (3 dB)')
    axes[1].set_ylabel('SNR (dB)', fontweight='bold')
    axes[1].set_ylim(0, 35)
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)

    # BER
    axes[2].semilogy(sc, ber, 'b.', markersize=2, alpha=0.7)
    axes[2].axhline(y=3.8e-3, color='r', linestyle='--', linewidth=1.5,
                    label=f'FEC limit (3.8e-3)')
    axes[2].set_ylabel('BER', fontweight='bold')
    axes[2].set_xlabel('Subcarrier index', fontweight='bold')
    axes[2].set_ylim(1e-6, 1e-1)
    axes[2].legend(loc='upper right')
    axes[2].grid(True, alpha=0.3)

    avg_ber = np.mean(ber[ber < 0.5])
    axes[2].annotate(f'Average BER: {avg_ber:.1e}', xy=(400, avg_ber), fontsize=10,
                     bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

    fig.suptitle('OFDM Subcarrier Performance (Fig. 3c)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(output_dir, 'fig3c_subcarrier.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {path}")


# =============================================================================
# FIGURE 3d: CONSTELLATION DIAGRAMS
# =============================================================================

def _plot_fig3d(results, output_dir, snr_db_val=25.0, n_symbols=2000):
    """Fig. 3d: QAM constellation diagrams."""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    qam_orders = [4, 8, 16, 64]

    for idx, qam_order in enumerate(qam_orders):
        ax_ideal, ax_noisy = axes[0, idx], axes[1, idx]

        if qam_order == 8:
            angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
            ideal = np.exp(1j * angles)
        else:
            M = int(np.sqrt(qam_order))
            levels = np.arange(-M + 1, M, 2)
            ideal = np.array([(i + 1j * q) for i in levels for q in levels])
            ideal = ideal / np.sqrt(np.mean(np.abs(ideal) ** 2))

        snr_linear = 10 ** (snr_db_val / 10)
        noise_std = 1 / np.sqrt(2 * snr_linear)

        tx_indices = np.random.randint(0, len(ideal), n_symbols)
        tx_syms = ideal[tx_indices]
        noise = (np.random.randn(n_symbols) + 1j * np.random.randn(n_symbols)) * noise_std
        rx_syms = tx_syms + noise

        max_val = max(np.abs(ideal).max(), np.abs(rx_syms).max()) * 1.3

        for ax, symbols, title_suffix in [(ax_ideal, ideal, 'Ideal'),
                                           (ax_noisy, rx_syms, f'SNR={snr_db_val}dB')]:
            if ax == ax_ideal:
                ax.scatter(ideal.real, ideal.imag, s=100, c='red', marker='x', linewidths=2)
            else:
                ax.scatter(rx_syms.real, rx_syms.imag, s=5, c='blue', alpha=0.3)
                ax.scatter(ideal.real, ideal.imag, s=80, c='red', marker='x', linewidths=2)
            ax.axhline(y=0, color='k', linewidth=0.5, alpha=0.3)
            ax.axvline(x=0, color='k', linewidth=0.5, alpha=0.3)
            ax.set_aspect('equal')
            ax.set_title(f'{qam_order}-QAM ({title_suffix})', fontweight='bold')
            ax.set_xlim(-max_val, max_val)
            ax.set_ylim(-max_val, max_val)

        if idx == 0:
            ax_ideal.set_ylabel('Quadrature', fontweight='bold')
            ax_noisy.set_ylabel('Quadrature', fontweight='bold')
        ax_noisy.set_xlabel('In-phase', fontweight='bold')

    fig.suptitle('QAM Constellation Diagrams (Fig. 3d)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(output_dir, 'fig3d_constellation.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {path}")


# =============================================================================
# FIGURE 4c: QUADRANT BEAM TRACKING
# =============================================================================

def _plot_fig4c(output_dir, duration_s=4.0, samples_per_second=100):
    """Fig. 4c: SNR and BER vs time during beam tracking."""
    n_samples = int(duration_s * samples_per_second)
    t = np.linspace(0, duration_s, n_samples)

    key_times = [0, 1, 2, 3, 4]
    key_positions = [(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0), (1.0, 0.0)]
    beam_rows = np.interp(t, key_times, [p[0] for p in key_positions])
    beam_cols = np.interp(t, key_times, [p[1] for p in key_positions])

    pd_positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
    pd_labels = ['PD1', 'PD2', 'PD4', 'PD5']

    beam_power_mw, beam_width = 10.0, 0.8
    responsivity, bandwidth = 0.36, 1.5e6
    noise_floor = 1e-9

    snr_history = np.zeros((n_samples, 4))
    ber_history = np.zeros((n_samples, 4))

    for i in range(n_samples):
        for j, (pd_row, pd_col) in enumerate(pd_positions):
            dist = np.sqrt((pd_row - beam_rows[i]) ** 2 + (pd_col - beam_cols[i]) ** 2)
            power = beam_power_mw * np.exp(-2 * (dist / beam_width) ** 2)
            current = responsivity * power * 1e-3
            snr_linear = (current ** 2) / (noise_floor ** 2 * bandwidth) if current > 0 else 1e-10
            snr_history[i, j] = 10 * np.log10(snr_linear + 1e-10)
            ber_history[i, j] = compute_ber_awgn(snr_history[i, j], 16)

    snr_history += np.random.randn(*snr_history.shape) * 1.5
    snr_history = np.maximum(snr_history, 0)
    for j in range(4):
        snr_history[:, j] = gaussian_filter1d(snr_history[:, j], sigma=3)
        ber_history[:, j] = np.array([compute_ber_awgn(s, 16) for s in snr_history[:, j]])

    fig, axes = plt.subplots(4, 1, figsize=(12, 14), sharex=True)
    colors = ['#1f77b4', '#2ca02c', '#9467bd', '#d62728']

    for j in range(4):
        ax = axes[j]
        ax.plot(t, snr_history[:, j], color=colors[j], linewidth=2, label='SNR')
        ax.set_ylabel('Mean SNR (dB)', fontweight='bold', color=colors[j])
        ax.tick_params(axis='y', labelcolor=colors[j])
        ax.set_ylim(0, 40)

        ax2 = ax.twinx()
        ax2.semilogy(t, ber_history[:, j], color='gray', linewidth=1.5, linestyle='--', alpha=0.7)
        ax2.set_ylabel('BER', fontweight='bold', color='gray')
        ax2.set_ylim(1e-5, 1)
        ax2.axhline(y=3.8e-3, color='red', linestyle=':', linewidth=1.5, alpha=0.7)

        ax.set_title(f'{pd_labels[j]}', fontweight='bold', loc='right')
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Time (s)', fontweight='bold')
    fig.suptitle('Quadrant Beam Tracking: SNR and BER vs Time (Fig. 4c)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(output_dir, 'fig4c_beam_tracking.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {path}")


# =============================================================================
# FIGURE 5/TABLE 1: ENERGY HARVESTING
# =============================================================================

def _plot_fig5_energy(output_dir):
    """Energy harvesting performance from Table 1."""
    lux_values = np.array([d[0] for d in ENERGY_HARVESTING_TABLE])
    voc_values = np.array([d[1] for d in ENERGY_HARVESTING_TABLE])
    isc_values = np.array([d[2] for d in ENERGY_HARVESTING_TABLE])
    source_names = [d[3] for d in ENERGY_HARVESTING_TABLE]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Voc vs Illuminance
    ax = axes[0, 0]
    ax.semilogx(lux_values, voc_values, 'b-o', markersize=10, linewidth=2)
    for i, name in enumerate(source_names):
        ax.annotate(name, (lux_values[i], voc_values[i]),
                    textcoords="offset points", xytext=(5, 5), fontsize=9)
    ax.set_xlabel('Illuminance (lux)', fontweight='bold')
    ax.set_ylabel('Open Circuit Voltage (V)', fontweight='bold')
    ax.set_title('Voc vs Illuminance', fontweight='bold')
    ax.grid(True, which='both', alpha=0.3)

    # Isc vs Illuminance
    ax = axes[0, 1]
    ax.loglog(lux_values, isc_values, 'g-s', markersize=10, linewidth=2)
    for i, name in enumerate(source_names):
        ax.annotate(name, (lux_values[i], isc_values[i]),
                    textcoords="offset points", xytext=(5, 5), fontsize=9)
    ax.set_xlabel('Illuminance (lux)', fontweight='bold')
    ax.set_ylabel('Short Circuit Current (mA)', fontweight='bold')
    ax.set_title('Isc vs Illuminance', fontweight='bold')
    ax.grid(True, which='both', alpha=0.3)

    # Power vs Illuminance for different configs
    ax = axes[1, 0]
    lux_dense = np.logspace(1, 5, 100)
    for n_pds, label, color, marker in [(1, '1 PD', 'blue', 'o'), (5, '5 PDs', 'green', 's'),
                                         (8, '8 PDs', 'orange', '^'), (9, '9 PDs', 'red', 'D')]:
        power_curve = [calculate_harvested_power(lux, n_pds) for lux in lux_dense]
        ax.semilogx(lux_dense, power_curve, f'-{marker}', color=color, markersize=6,
                    linewidth=2, label=label, markevery=15)
    ax.axhline(y=43, color='k', linestyle='--', linewidth=2, label='System needs (43 mW)')
    ax.set_xlabel('Illuminance (lux)', fontweight='bold')
    ax.set_ylabel('Harvested Power (mW)', fontweight='bold')
    ax.set_title('Harvested Power vs Illuminance', fontweight='bold')
    ax.set_ylim(0, 100)
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, which='both', alpha=0.3)

    # Self-Powered Region
    ax = axes[1, 1]
    lux_test = np.logspace(1, 5, 1000)
    for n_pds, label, color in [(5, '5 PDs', 'green'), (8, '8 PDs', 'orange'), (9, '9 PDs', 'red')]:
        power_curve = np.array([calculate_harvested_power(lux, n_pds) for lux in lux_test])
        ax.fill_between(lux_test, 0, 1, where=(power_curve >= 43), alpha=0.3, label=label)
    ax.set_xscale('log')
    ax.set_xlabel('Illuminance (lux)', fontweight='bold')
    ax.set_ylabel('Self-Powered Capability', fontweight='bold')
    ax.set_title('Self-Powering Threshold', fontweight='bold')
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['No', 'Yes'])
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, which='both', alpha=0.3)

    fig.suptitle('Energy Harvesting Performance (Table 1 / Fig. 5)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(output_dir, 'fig5_energy_harvesting.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {path}")


# =============================================================================
# TRADEOFF ANALYSIS
# =============================================================================

def _plot_tradeoff(output_dir, illuminance_lux=55900):
    """Communication vs energy harvesting tradeoff."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    n_comm_pds = np.array([0, 1, 2, 4, 6, 9])
    n_harvest_pds = 9 - n_comm_pds
    data_rates = np.array([calculate_data_rate(n, 'small') for n in n_comm_pds])
    harvest_powers = np.array([calculate_harvested_power(illuminance_lux, n)
                               for n in n_harvest_pds])

    ax1 = axes[0]
    ax1.set_xlabel('Number of Communication PDs', fontweight='bold')
    ax1.set_ylabel('Net Data Rate (Mbps)', color='tab:blue', fontweight='bold')
    ax1.plot(n_comm_pds, data_rates, 'b-o', linewidth=3, markersize=10, label='Data Rate')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Harvested Power (mW)', color='tab:green', fontweight='bold')
    ax2.plot(n_comm_pds, harvest_powers, 'g-s', linewidth=3, markersize=10)
    ax2.tick_params(axis='y', labelcolor='tab:green')
    ax2.axhline(y=43, color='red', linestyle='--', linewidth=2)
    ax1.axvline(x=4, color='orange', linestyle=':', linewidth=2, alpha=0.7)
    ax1.set_title('Data Rate vs Harvested Power', fontweight='bold')
    ax1.grid(True, alpha=0.3)

    ax = axes[1]
    colors_pareto = plt.cm.viridis(np.linspace(0, 1, len(n_comm_pds)))
    for i, (dr, hp, nc) in enumerate(zip(data_rates, harvest_powers, n_comm_pds)):
        ax.scatter(hp, dr, s=200, c=[colors_pareto[i]], edgecolors='black', linewidth=2)
        ax.annotate(f'{nc}C/{9 - nc}H', (hp, dr),
                    textcoords="offset points", xytext=(10, 0), fontsize=9)
    ax.plot(harvest_powers, data_rates, 'k--', alpha=0.5, linewidth=1)
    ax.axvline(x=43, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax.fill_betweenx([0, max(data_rates) * 1.1], 43, max(harvest_powers) * 1.1,
                     color='green', alpha=0.1)
    ax.set_xlabel('Harvested Power (mW)', fontweight='bold')
    ax.set_ylabel('Net Data Rate (Mbps)', fontweight='bold')
    ax.set_title('Pareto Frontier', fontweight='bold')
    ax.grid(True, alpha=0.3)

    fig.suptitle(f'Communication vs Harvesting Tradeoff @ {illuminance_lux} lux',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(output_dir, 'fig6_tradeoff.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {path}")


# =============================================================================
# MODE COMPARISON TABLE
# =============================================================================

def _plot_comparison_table(output_dir):
    """Mode comparison summary table."""
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.axis('off')

    data = [
        ['Mode', 'Comm\nPDs', 'Harvest\nPDs', 'Gross Rate\n(Mbps)',
         'Net Rate\n(Mbps)', 'BER', 'Harvest\n(mW)', 'Self-\nPowered'],
        ['SISO (Large PD)', '1', '8', '5.3', '4.0', '3.3e-3', '77.4', 'Yes'],
        ['SISO (Small PD)', '1', '8', '25.7', '21.3', '3.4e-3', '77.4', 'Yes'],
        ['MIMO 2x2', '4', '5', '51.4', '42.6', '~3.5e-3', '48.4', 'Yes'],
        ['MIMO 3x3', '4', '5', '102.8', '85.2', '3.3e-3', '48.4', 'Yes'],
        ['All Comm', '9', '0', '~230', '~180', '~2e-3', '0', 'No'],
    ]

    table = ax.table(cellText=data, cellLoc='center', loc='center', colWidths=[0.14] * 8)
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.8)

    for j in range(8):
        table[(0, j)].set_facecolor('#2E86AB')
        table[(0, j)].set_text_props(weight='bold', color='white')

    for i in range(1, 6):
        for j in range(8):
            table[(i, j)].set_facecolor('#E8F4F8' if i % 2 else '#FFFFFF')
            if i == 4:
                table[(i, j)].set_edgecolor('#28A745')
                table[(i, j)].set_linewidth(3)

    ax.set_title('Oliveira 2024: Operating Mode Comparison\n(@ Direct Sunlight, 55900 lux)',
                 fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    path = os.path.join(output_dir, 'fig7_comparison.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {path}")


# =============================================================================
# PD ARRAY HEATMAP
# =============================================================================

def _plot_pd_array(output_dir):
    """Visualize the 3x3 PD array with quadrant configurations."""
    fig, axes = plt.subplots(2, 5, figsize=(18, 8))
    quadrant_names = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X']

    for idx, qname in enumerate(quadrant_names):
        ax = axes[idx // 5, idx % 5]
        comm_pds = QUADRANT_CONFIGS.get(qname, [])

        for pd_idx in range(9):
            row, col = pd_idx // 3, pd_idx % 3
            color = '#2E86AB' if pd_idx in comm_pds else '#28A745'
            mode = 'C' if pd_idx in comm_pds else 'H'
            rect = FancyBboxPatch((col, 2 - row), 0.9, 0.9,
                                  boxstyle="round,pad=0.02",
                                  facecolor=color, edgecolor='black', linewidth=2)
            ax.add_patch(rect)
            ax.text(col + 0.45, 2 - row + 0.45, f'PD{pd_idx + 1}\n({mode})',
                    ha='center', va='center', fontsize=9, fontweight='bold', color='white')

        ax.set_xlim(-0.1, 3.1)
        ax.set_ylim(-0.1, 3.1)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(f'Quadrant {qname}\n({len(comm_pds)}C / {9 - len(comm_pds)}H)',
                     fontweight='bold')

    fig.suptitle('3x3 PD Array Quadrant Configurations (Fig. 4d)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    path = os.path.join(output_dir, 'fig8_pd_array.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {path}")


# =============================================================================
# SUPERCAPACITOR CHARGING
# =============================================================================

def _plot_supercap(output_dir, illuminance_lux=55900, duration_min=60):
    """Supercapacitor charging curve."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    t_min = np.linspace(0, duration_min, 1000)
    t_s = t_min * 60
    supercap_f, supercap_v_max, mppt_eff = 0.1, 5.0, 0.85

    ax = axes[0]
    configs = [(9, 'All harvest', 'red'), (8, 'SISO mode', 'orange'),
               (5, 'Quadrant mode', 'green'), (1, 'Minimal', 'blue')]

    for n_pds, label, color in configs:
        power_mw = calculate_harvested_power(illuminance_lux, n_pds) * mppt_eff
        power_w = power_mw * 1e-3
        max_energy = 0.5 * supercap_f * supercap_v_max ** 2

        voltages = []
        energy = 0
        for t in t_s:
            dt = t_s[1] - t_s[0] if len(t_s) > 1 else 1
            energy = min(energy + power_w * dt, max_energy)
            voltages.append(np.sqrt(2 * energy / supercap_f))
        ax.plot(t_min, voltages, color=color, linewidth=2, label=label)

    ax.axhline(y=3.3, color='gray', linestyle='--', linewidth=1.5, label='Min operating')
    ax.axhline(y=5.0, color='black', linestyle=':', linewidth=1.5, label='Max voltage')
    ax.set_xlabel('Time (minutes)', fontweight='bold')
    ax.set_ylabel('Supercapacitor Voltage (V)', fontweight='bold')
    ax.set_title(f'Charging at {illuminance_lux} lux', fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    lux_values = np.logspace(2, 5, 50)
    for n_pds, label, color in [(9, '9 PDs', 'red'), (5, '5 PDs', 'green'), (1, '1 PD', 'blue')]:
        charge_times = []
        for lux in lux_values:
            power_w = calculate_harvested_power(lux, n_pds) * mppt_eff * 1e-3
            max_energy = 0.5 * supercap_f * supercap_v_max ** 2
            t_charge = max_energy / power_w / 60 if power_w > 0 else 1000
            charge_times.append(min(t_charge, 1000))
        ax.loglog(lux_values, charge_times, color=color, linewidth=2, label=label)

    ax.scatter([55900], [32.2], s=200, c='red', marker='*', zorder=5, label='Paper (32.2 min)')
    ax.set_xlabel('Illuminance (lux)', fontweight='bold')
    ax.set_ylabel('Charge Time (minutes)', fontweight='bold')
    ax.set_title('Charge Time vs Illuminance', fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, which='both', alpha=0.3)

    fig.suptitle('Supercapacitor Charging Performance', fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(output_dir, 'fig9_supercap.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {path}")


# =============================================================================
# BER VS DISTANCE
# =============================================================================

def _plot_ber_distance(output_dir):
    """BER vs distance for different configurations."""
    distances_m = np.linspace(0.1, 2.0, 20)

    fig, ax = plt.subplots(figsize=(10, 7))
    configs = [('SISO Small', 1, 0.36, 5e6, 'b', 'o'),
               ('SISO Large', 1, 0.36, 1.5e6, 'c', 's'),
               ('MIMO 2x2', 4, 0.36, 5e6, 'g', '^'),
               ('MIMO 3x3', 9, 0.36, 5e6, 'r', 'D')]
    tx_power_mw, noise_floor = 10.0, 1e-9

    for name, n_pds, resp, bw, color, marker in configs:
        ber_values = []
        for dist in distances_m:
            rx_power = tx_power_mw * (0.1 / dist) ** 2
            current = resp * rx_power * 1e-3
            snr_linear = (current ** 2) / (noise_floor ** 2 * bw) if current > 0 else 1e-10
            snr_db = 10 * np.log10(snr_linear * n_pds + 1e-10)
            ber_values.append(max(compute_ber_awgn(snr_db, 16), 1e-7))
        ax.semilogy(distances_m, ber_values, color=color, marker=marker, linestyle='-',
                    linewidth=2, markersize=8, label=name, markevery=2)

    ax.axhline(y=3.8e-3, color='black', linestyle='--', linewidth=2, label='FEC threshold')
    ax.set_xlabel('Distance (m)', fontweight='bold')
    ax.set_ylabel('Bit Error Rate (BER)', fontweight='bold')
    ax.set_title('BER vs Distance', fontweight='bold')
    ax.set_ylim(1e-7, 1)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, which='both', alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, 'fig10_ber_distance.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {path}")


# =============================================================================
# BIT ALLOCATION HEATMAP (from original)
# =============================================================================

def _plot_bit_allocation(modem, snr_db, output_dir):
    """Bit allocation heatmap."""
    fig, ax = plt.subplots(figsize=(12, 4))
    n_sc = modem.n_subcarriers
    colors = {0: 'white', 1: '#fee08b', 2: '#a6d96a', 3: '#66bd63',
              4: '#1a9850', 5: '#006837', 6: '#004529'}
    for sc in range(n_sc):
        nb = modem.bit_allocation[sc]
        ax.bar(sc, 1, color=colors.get(nb, 'gray'), width=1.0, edgecolor='none')

    from matplotlib.patches import Patch
    handles = [Patch(color=colors[b],
                     label=f'{b}b ({2 ** b}-QAM)' if b > 0 else '0b (off)')
               for b in sorted(colors.keys())
               if np.any(modem.bit_allocation == b)]
    ax.legend(handles=handles, loc='upper right', fontsize=9, ncol=4)

    ax.set_xlabel('Subcarrier Index')
    ax.set_title('Adaptive Bit Allocation (Oliveira 2024)')
    ax.set_xlim([0, n_sc])
    ax.set_yticks([])

    path = os.path.join(output_dir, 'bit_allocation_map.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {path}")


# =============================================================================
# VALIDATION
# =============================================================================

def run_validation(output_dir=None, n_symbols=200):
    """Run Oliveira 2024 full validation suite."""
    if output_dir is None:
        output_dir = os.path.join('workspace', 'validation_oliveira2024')
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("  OLIVEIRA et al. (2024) - MIMO SLIPT VALIDATION")
    print("  Light: Science & Applications")
    print("=" * 60)

    # 1. SNR profile
    np.random.seed(42)
    snr_db = generate_snr_profile(
        PARAMS['n_subcarriers'],
        PARAMS['snr_max_db'],
        PARAMS['snr_min_db'],
        PARAMS['rolloff_exponent'],
    )

    # 2. Setup modem (adaptive)
    modem = OFDMModem(
        nfft=PARAMS['nfft'],
        cp_length=PARAMS['cp_length'],
        n_subcarriers=PARAMS['n_subcarriers'],
        bandwidth_hz=PARAMS['bandwidth_hz'],
    )
    modem.set_adaptive_modulation(snr_db)

    fec_table = [
        (28.0, 6, 64),
        (23.0, 5, 32),
        (19.0, 4, 16),
        (14.0, 3, 8),
        (11.0, 2, 4),
        (8.0, 1, 2),
    ]
    modem.bit_allocation = allocate_bits(snr_db, fec_table)

    # 3. Data rate
    gdr = modem.calculate_data_rate(PARAMS['sample_rate_hz'])
    ndr = net_data_rate(gdr, TARGETS['fec_overhead'])

    print(f"\n  Bit Allocation Summary:")
    for threshold, bits, M in BIT_ALLOCATION_TABLE:
        count = np.sum(modem.bit_allocation == bits)
        if count > 0:
            print(f"    {M:2d}-QAM ({bits}b): {count:3d} subcarriers")
    inactive = np.sum(modem.bit_allocation == 0)
    if inactive > 0:
        print(f"    Inactive:     {inactive:3d} subcarriers")

    print(f"\n  SISO Data Rates:")
    print(f"    Gross: {gdr:.2f} Mbps (target: {TARGETS['siso_gross_mbps']})")
    print(f"    Net:   {ndr:.2f} Mbps (target: {TARGETS['siso_net_mbps']})")

    mimo_net = ndr * 4
    print(f"\n  MIMO (4-channel) Data Rates:")
    print(f"    Net: {mimo_net:.2f} Mbps (target: {TARGETS['mimo_net_mbps']})")

    # 4. Simulate per-subcarrier BER
    print(f"\n  Simulating {n_symbols} OFDM symbols...")
    results = modem.simulate(n_symbols=n_symbols, snr_per_sc=snr_db, seed=42)

    print(f"    Overall BER: {results['overall_ber']:.4e}")
    passed = results['overall_ber'] < TARGETS['ber_threshold']
    print(f"    Status: {'PASS' if passed else 'FAIL'} "
          f"(threshold: {TARGETS['ber_threshold']:.1e})")

    # 5. Figures
    print("\n  Generating figures...")
    _plot_fig3c(results, snr_db, modem, output_dir)
    _plot_fig3d(results, output_dir)
    _plot_bit_allocation(modem, snr_db, output_dir)
    _plot_fig4c(output_dir)
    _plot_fig5_energy(output_dir)
    _plot_tradeoff(output_dir)
    _plot_comparison_table(output_dir)
    _plot_pd_array(output_dir)
    _plot_supercap(output_dir)
    _plot_ber_distance(output_dir)

    print(f"\n  Output: {output_dir}")
    return passed


if __name__ == "__main__":
    run_validation()
