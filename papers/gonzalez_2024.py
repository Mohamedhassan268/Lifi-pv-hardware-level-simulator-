"""
Gonzalez-Uriarte et al. (2024) - Paper Validation & Figure Generation
======================================================================

Paper: "Design and Implementation of a Low-Cost VLC Photovoltaic Panel-Based
        Receiver with off-the-Shelf Components"
       IEEE LATINCOM 2024

Figures:
  - Fig 3: Vpp vs Load
  - Fig 7: TX vs RX Manchester waveforms
  - Combined validation: BW + Amplifier/Slicer + BER vs interference
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import signal as sp_signal

PARAMS = {
    'C_j_nF': 14.5,
    'R_sh_kohm': 200.0,
    'responsivity': 0.4,
    'fig2_loads_ohm': [11.6, 100, 220, 1e3, 10e3, 128e3, 1e6],
    'fig2_bw_hz': [300e3, 100e3, 50e3, 10e3, 1e3, 500, 500],
    'fig3_V_pp_open_mV': 600.0,
    'fig3_V_pp_220_mV': 20.0,
    'R_load_ohm': 220.0,
    'notch_freq_hz': 100.0,
    'notch_Q': 30.0,
    'amp_gain': 165.0,
    'amp_rail_v': 3.3,
    'slicer_threshold_v': 1.65,
    'baud_rate': 4800,
}

TARGETS = {
    'bw_at_220_hz': 50e3,
    'V_pp_220_mV': 20.0,
    'ber_at_60cm': 0.0,
}


def bandwidth(R_load, C_j_F, R_sh):
    R_eq = (R_sh * R_load) / (R_sh + R_load) if R_load < 1e12 else R_sh
    return 1.0 / (2 * np.pi * R_eq * C_j_F)


def voltage_out(I_ph, R_load, R_sh):
    R_eq = (R_sh * R_load) / (R_sh + R_load) if R_load < 1e12 else R_sh
    return I_ph * R_eq


def manchester_encode(bits, samples_per_bit):
    sig = np.zeros(len(bits) * samples_per_bit)
    for i, b in enumerate(bits):
        start = i * samples_per_bit
        mid = start + samples_per_bit // 2
        end = start + samples_per_bit
        if b == 0:
            sig[mid:end] = 1.0
        else:
            sig[start:mid] = 1.0
    return sig


def manchester_decode(sig, samples_per_bit, threshold=0.5):
    binary = (sig > threshold).astype(float)
    n_bits = len(sig) // samples_per_bit
    bits = np.zeros(n_bits, dtype=int)
    for i in range(n_bits):
        q1 = i * samples_per_bit + samples_per_bit // 4
        q3 = i * samples_per_bit + 3 * samples_per_bit // 4
        if q3 < len(binary):
            bits[i] = 1 if binary[q1] > binary[q3] else 0
    return bits


def apply_notch(sig, fs, f0=100, Q=30):
    w0 = f0 / (fs / 2)
    if w0 >= 1.0:
        return sig
    b, a = sp_signal.iirnotch(w0, Q)
    out = sp_signal.filtfilt(b, a, sig)
    out = sp_signal.filtfilt(b, a, out)
    if fs > 200:
        b_hp, a_hp = sp_signal.butter(2, 50 / (fs / 2), btype='high')
        out = sp_signal.filtfilt(b_hp, a_hp, out)
    return out


def apply_amplifier(sig, gain=165, rail=3.3):
    return np.clip(sig * gain + rail / 2, 0, rail)


def apply_lowpass_rc(sig, t, R_load, C_j_F, R_sh):
    f_c = bandwidth(R_load, C_j_F, R_sh)
    fs = 1.0 / np.mean(np.diff(t))
    wn = f_c / (fs / 2)
    if wn >= 1.0:
        R_eq = (R_sh * R_load) / (R_sh + R_load)
        return sig * R_eq
    b, a = sp_signal.butter(1, wn, btype='low')
    R_eq = (R_sh * R_load) / (R_sh + R_load)
    return sp_signal.lfilter(b, a, sig * R_eq)


def sim_bandwidth_vs_load():
    C_j = PARAMS['C_j_nF'] * 1e-9
    R_sh = PARAMS['R_sh_kohm'] * 1e3
    R_loads = np.logspace(1, 7, 50)
    bws = np.array([bandwidth(R, C_j, R_sh) for R in R_loads])
    bw_220 = bandwidth(220, C_j, R_sh)
    return R_loads, bws, bw_220


def sim_time_domain(n_bits=200):
    C_j = PARAMS['C_j_nF'] * 1e-9
    R_sh = PARAMS['R_sh_kohm'] * 1e3
    R_load = PARAMS['R_load_ohm']
    baud = PARAMS['baud_rate']
    spb = 100
    fs = baud * spb
    np.random.seed(42)
    bits_tx = np.random.randint(0, 2, n_bits)
    sig_tx = manchester_encode(bits_tx, spb)
    t = np.arange(len(sig_tx)) / fs
    I_ph = sig_tx * 1e-3 * PARAMS['responsivity']
    V_pv = apply_lowpass_rc(I_ph, t, R_load, C_j, R_sh)
    interf = np.max(np.abs(V_pv)) * 0.5 * np.sin(2 * np.pi * 100 * t)
    V_noisy = V_pv + interf
    V_notch = apply_notch(V_noisy, fs)
    V_amp = apply_amplifier(V_notch, PARAMS['amp_gain'], PARAMS['amp_rail_v'])
    V_norm = V_amp / PARAMS['amp_rail_v']
    bits_rx = manchester_decode(V_norm, spb, threshold=0.5)
    n = min(len(bits_tx), len(bits_rx))
    errors = int(np.sum(bits_tx[:n] != bits_rx[:n]))
    ber = errors / n
    return {
        't': t, 'sig_tx': sig_tx, 'V_pv': V_pv, 'V_noisy': V_noisy,
        'V_notch': V_notch, 'V_amp': V_amp, 'bits_tx': bits_tx,
        'bits_rx': bits_rx, 'ber': ber, 'fs': fs, 'spb': spb,
    }


def sim_ber_vs_interference():
    C_j = PARAMS['C_j_nF'] * 1e-9
    R_sh = PARAMS['R_sh_kohm'] * 1e3
    R_load = PARAMS['R_load_ohm']
    baud = PARAMS['baud_rate']
    spb = 50; fs = baud * spb; n_bits = 500
    np.random.seed(42)
    bits_tx = np.random.randint(0, 2, n_bits)
    sig_tx = manchester_encode(bits_tx, spb)
    t = np.arange(len(sig_tx)) / fs
    I_ph = sig_tx * 1e-3 * PARAMS['responsivity']
    V_pv = apply_lowpass_rc(I_ph, t, R_load, C_j, R_sh)
    sig_amp = np.max(np.abs(V_pv))
    levels = np.linspace(0, 2, 10)
    bers = []
    for lv in levels:
        interf = lv * sig_amp * np.sin(2 * np.pi * 100 * t)
        V_n = apply_notch(V_pv + interf, fs)
        V_a = apply_amplifier(V_n, PARAMS['amp_gain'], PARAMS['amp_rail_v'])
        bits_rx = manchester_decode(V_a / PARAMS['amp_rail_v'], spb)
        n = min(len(bits_tx), len(bits_rx))
        bers.append(np.sum(bits_tx[:n] != bits_rx[:n]) / n)
    return levels, bers


# =============================================================================
# FIGURES
# =============================================================================

def _plot_all(output_dir, R_loads, bws, td, levels, bers):
    R_sh = PARAMS['R_sh_kohm'] * 1e3
    R_eq_220 = (R_sh * 220) / (R_sh + 220)
    I_ph = PARAMS['fig3_V_pp_220_mV'] * 1e-3 / R_eq_220
    V_oc_mV = PARAMS['fig3_V_pp_open_mV']

    # Fig 3: Vpp vs Load
    R_sweep = np.logspace(0.8, 7, 200)
    V_out_mV = np.array([
        min(I_ph * (R_sh * R) / (R_sh + R) * 1e3, V_oc_mV) for R in R_sweep
    ])
    paper_R = [11.6, 100, 220, 356, 1e3, 2e3, 5.18e3, 10e3, 128e3, 1e6]
    paper_Vpp = [1.0, 9, 20, 30, 80, 145, 330, 450, 590, 600]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(R_sweep, V_out_mV, '-', color='#4472C4', lw=2, label='Simulated')
    ax.plot(paper_R, paper_Vpp, 'o-', color='#4472C4', ms=7, lw=1,
            mfc='#4472C4', mec='#2F5496', label='Paper Data')
    ax.set_xscale('log')
    ax.set_xlabel('Load [Ohm]', fontsize=11)
    ax.set_ylabel('Vpp [mV]', fontsize=11)
    ax.set_title('Vpp vs Load of PV Panel (Fig. 3)', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3); ax.legend(fontsize=9)
    ax.set_ylim([0, 650]); ax.set_xlim([8, 2e6])
    plt.tight_layout()
    path = os.path.join(output_dir, 'fig3_vpp_vs_load.png')
    plt.savefig(path, dpi=200, bbox_inches='tight'); plt.close()
    print(f"    Saved: {path}")

    # Fig 7: TX vs RX Manchester waveforms
    fig7, (ax_rx, ax_tx) = plt.subplots(2, 1, figsize=(7, 5.5), sharex=True)
    fig7.suptitle('Received and Transmitted Signal (Fig. 7)', fontsize=13, fontweight='bold')
    n_show = int(4.5e-3 * td['fs'])
    t_ms = td['t'][:n_show] * 1e3
    ax_rx.plot(t_ms, td['V_amp'][:n_show], color='#C00000', linewidth=1.0)
    ax_rx.set_ylabel('Voltage [V]', fontsize=10)
    ax_rx.set_title('Received Signal', fontsize=10)
    ax_rx.set_ylim([-0.2, 3.6]); ax_rx.grid(True, alpha=0.3)
    ax_tx.plot(t_ms, td['sig_tx'][:n_show] * 3.3, color='#4472C4', linewidth=1.0)
    ax_tx.set_ylabel('Voltage [V]', fontsize=10)
    ax_tx.set_xlabel('Time [ms]', fontsize=10)
    ax_tx.set_title('Transmitted Signal', fontsize=10)
    ax_tx.set_ylim([-0.2, 3.6]); ax_tx.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, 'fig7_rx_tx_manchester.png')
    plt.savefig(path, dpi=200, bbox_inches='tight'); plt.close()
    print(f"    Saved: {path}")

    # Combined validation
    fig_c, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig_c.suptitle("Gonzalez-Uriarte 2024 - Validation Results", fontsize=14, fontweight='bold')

    ax = axes[0]
    ax.loglog(R_loads, bws, 'b-', lw=2, label='Simulated')
    ax.loglog(PARAMS['fig2_loads_ohm'], PARAMS['fig2_bw_hz'], 'ro', ms=8,
              mfc='white', mew=2, label='Paper Data')
    ax.axhline(50e3, color='green', ls='--', alpha=0.5, label='Target BW')
    ax.axvline(220, color='orange', ls='--', alpha=0.5, label='Operating R')
    ax.set_xlabel('Load Resistance (Ohm)'); ax.set_ylabel('Bandwidth (Hz)')
    ax.set_title('Fig. 2: Bandwidth vs Load', fontweight='bold')
    ax.legend(fontsize=8); ax.grid(True, which='both', alpha=0.3)

    ax = axes[1]
    n_s = 1000
    t_ms2 = td['t'][:n_s] * 1e3
    ax.plot(t_ms2, td['V_amp'][:n_s], 'g-', lw=1.5, label='Amplified')
    ax.axhline(PARAMS['slicer_threshold_v'], color='r', ls='--', lw=1.5, label='Threshold')
    ax.set_xlabel('Time (ms)'); ax.set_ylabel('Voltage (V)')
    ax.set_title('Amplifier Output + Slicer', fontweight='bold')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.semilogy(levels * 100, np.array(bers) + 1e-4, 'bo-', lw=2, ms=6)
    ax.set_xlabel('100 Hz Interference Level (%)')
    ax.set_ylabel('BER')
    ax.set_title('BER vs Interference Level', fontweight='bold')
    ax.grid(True, alpha=0.3); ax.set_ylim([1e-4, 1])

    plt.tight_layout()
    path = os.path.join(output_dir, 'gonzalez_validation_combined.png')
    plt.savefig(path, dpi=200, bbox_inches='tight'); plt.close()
    print(f"    Saved: {path}")


def run_validation(output_dir=None):
    if output_dir is None:
        output_dir = os.path.join('workspace', 'validation_gonzalez2024')
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 65)
    print("  GONZALEZ-URIARTE et al. (2024) - LOW-COST VLC VALIDATION")
    print("  IEEE LATINCOM 2024")
    print("=" * 65)

    R_loads, bws, bw_220 = sim_bandwidth_vs_load()
    bw_err = abs(bw_220 - TARGETS['bw_at_220_hz']) / TARGETS['bw_at_220_hz'] * 100
    print(f"\n  BW @ 220 Ohm: {bw_220/1e3:.1f} kHz (target: 50 kHz), error: {bw_err:.1f}%")

    td = sim_time_domain(n_bits=200)
    print(f"  BER @ 4.8 kBd: {td['ber']:.4f}")

    levels, bers = sim_ber_vs_interference()
    print(f"  BER @ 100% interference: {bers[5]:.4f}")

    print("\n  Generating figures...")
    _plot_all(output_dir, R_loads, bws, td, levels, bers)

    all_pass = bw_err < 20 and td['ber'] < 0.01
    print(f"\n  Overall: {'PASS' if all_pass else 'REVIEW'}")
    print(f"  Output: {output_dir}")
    return all_pass


if __name__ == "__main__":
    run_validation()
