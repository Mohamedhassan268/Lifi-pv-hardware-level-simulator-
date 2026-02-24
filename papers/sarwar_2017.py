"""
Sarwar et al. (2017) - Paper Validation & Figure Generation
============================================================

Paper: "Visible Light Communication Using a Solar-Panel Receiver"
       ICOCN 2017

Figures:
  - Fig 4: BER vs subcarrier index
  - Fig 5: EVM vs subcarrier index
  - Fig 6: OFDM spectrum (PSD)
  - Fig 7: 16-QAM constellation diagram
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import signal as sig

from papers.ofdm_modem import OFDMModem, solar_panel_channel_response, ber_mqam

PARAMS = {
    'led_power_w': 3.0,
    'distance_m': 2.0,
    'panel_area_cm2': 7.5,
    'qam_order': 16,
    'n_subcarriers': 80,
    'n_fft': 256,
    'cp_len': 32,
    'sample_rate_hz': 15e6,
    'signal_bandwidth_hz': 5e6,
    'junction_capacitance_pf': 500,
    'shunt_resistance_kohm': 10,
}

TARGETS = {
    'data_rate_mbps': 15.03,
    'ber': 1.6883e-3,
    'fec_threshold': 3.8e-3,
}


def _plot_fig4_ber(results, output_dir):
    fig, ax = plt.subplots(figsize=(10, 6))
    sc = np.arange(results['n_subcarriers'])
    ber = np.maximum(results['ber_per_sc'], 1e-6)
    ax.semilogy(sc, ber, 'b-o', markersize=3, linewidth=1.5, label='Simulated')
    ax.axhline(TARGETS['ber'], color='r', linestyle='--', linewidth=1.5,
               label=f"Paper avg: {TARGETS['ber']:.4e}")
    ax.axhline(TARGETS['fec_threshold'], color='g', linestyle=':', linewidth=1.5,
               label=f"FEC threshold: {TARGETS['fec_threshold']:.1e}")
    ax.set_xlabel('Subcarrier Index', fontsize=12)
    ax.set_ylabel('BER', fontsize=12)
    ax.set_title('BER per Subcarrier (Sarwar 2017, Fig. 4)', fontsize=14, fontweight='bold')
    ax.set_xlim([0, results['n_subcarriers']])
    ax.set_ylim([1e-5, 0.1])
    ax.legend(fontsize=11); ax.grid(True, alpha=0.3)
    path = os.path.join(output_dir, 'fig4_ber_subcarrier.png')
    plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()
    print(f"    Saved: {path}")


def _plot_fig5_evm(results, output_dir):
    fig, ax = plt.subplots(figsize=(10, 6))
    sc = np.arange(results['n_subcarriers'])
    ax.plot(sc, results['evm_per_sc'], 'b-s', markersize=3, linewidth=1.5)
    ax.set_xlabel('Subcarrier Index', fontsize=12)
    ax.set_ylabel('EVM (%)', fontsize=12)
    ax.set_title('EVM per Subcarrier (Sarwar 2017, Fig. 5)', fontsize=14, fontweight='bold')
    ax.set_xlim([0, results['n_subcarriers']]); ax.grid(True, alpha=0.3)
    path = os.path.join(output_dir, 'fig5_evm_subcarrier.png')
    plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()
    print(f"    Saved: {path}")


def _plot_fig6_spectrum(results, output_dir):
    n_fft = PARAMS['n_fft']
    freq_frame = np.zeros(n_fft, dtype=complex)
    H = results['channel_H']
    for i in range(min(len(H), n_fft // 2)):
        freq_frame[i + 1] = H[i] * (np.random.randn() + 1j * np.random.randn())
    time_sig = np.tile(np.fft.ifft(freq_frame), 100)
    time_sig += 0.01 * np.random.randn(len(time_sig))
    freqs, psd = sig.welch(time_sig.real, PARAMS['sample_rate_hz'], nperseg=1024)
    psd_db = 10 * np.log10(psd + 1e-20)
    psd_db -= np.max(psd_db)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(freqs / 1e6, psd_db, 'b-', linewidth=1.5)
    ax.set_xlabel('Frequency (MHz)', fontsize=12)
    ax.set_ylabel('PSD (dB, normalized)', fontsize=12)
    ax.set_title('OFDM Spectrum (Sarwar 2017, Fig. 6)', fontsize=14, fontweight='bold')
    ax.set_xlim([0, 5]); ax.set_ylim([-70, 10]); ax.grid(True, alpha=0.3)
    path = os.path.join(output_dir, 'fig6_spectrum.png')
    plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()
    print(f"    Saved: {path}")


def _plot_fig7_constellation(results, output_dir):
    fig, ax = plt.subplots(figsize=(8, 8))
    rx = results['rx_symbols'][:2000]
    ax.scatter(rx.real, rx.imag, s=5, alpha=0.3, c='blue', label='Received')
    ideal = np.array([-3, -1, 1, 3]) / np.sqrt(10)
    for i in ideal:
        for q in ideal:
            ax.scatter(i, q, s=100, c='red', marker='+', linewidths=2)
    ax.set_xlabel('In-Phase', fontsize=12); ax.set_ylabel('Quadrature', fontsize=12)
    ax.set_title(f"16-QAM Constellation (BER={results['overall_ber']:.4e})",
                 fontsize=14, fontweight='bold')
    ax.set_xlim([-1.5, 1.5]); ax.set_ylim([-1.5, 1.5])
    ax.grid(True, alpha=0.3); ax.set_aspect('equal')
    path = os.path.join(output_dir, 'fig7_constellation.png')
    plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()
    print(f"    Saved: {path}")


def run_validation(output_dir=None, n_symbols=1000):
    if output_dir is None:
        output_dir = os.path.join('workspace', 'validation_sarwar2017')
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("  SARWAR et al. (2017) - OFDM VALIDATION")
    print("  ICOCN 2017")
    print("=" * 60)

    modem = OFDMModem(
        nfft=PARAMS['n_fft'], cp_length=PARAMS['cp_len'],
        n_subcarriers=PARAMS['n_subcarriers'],
        bandwidth_hz=PARAMS['signal_bandwidth_hz'],
    )
    modem.set_fixed_modulation(qam_order=PARAMS['qam_order'])

    H, snr_db = solar_panel_channel_response(
        PARAMS['n_subcarriers'], PARAMS['signal_bandwidth_hz'],
        PARAMS['junction_capacitance_pf'], PARAMS['shunt_resistance_kohm'],
    )

    print(f"\n  Running {n_symbols} OFDM symbols x {PARAMS['n_subcarriers']} subcarriers...")
    results = modem.simulate(n_symbols=n_symbols, snr_per_sc=snr_db,
                             channel_H=H, equalization='ZF', seed=42)

    data_rate = modem.calculate_data_rate(PARAMS['sample_rate_hz'])
    print(f"  Data rate: {data_rate:.2f} Mbps (target: {TARGETS['data_rate_mbps']})")
    print(f"  BER: {results['overall_ber']:.4e} (target: {TARGETS['ber']:.4e})")

    passed = results['overall_ber'] < TARGETS['fec_threshold']
    print(f"  Status: {'PASS' if passed else 'FAIL'} (below FEC {TARGETS['fec_threshold']:.1e})")

    print("\n  Generating figures...")
    _plot_fig4_ber(results, output_dir)
    _plot_fig5_evm(results, output_dir)
    _plot_fig6_spectrum(results, output_dir)
    _plot_fig7_constellation(results, output_dir)

    print(f"  Output: {output_dir}")
    return passed


if __name__ == "__main__":
    run_validation()
