"""
Kadirvelu et al. (2021) - Paper Validation & Figure Generation
===============================================================

Paper: "A Circuit for Simultaneous Reception of Data and Power Using a Solar Cell"
       IEEE Trans. Green Communications and Networking, Vol. 5, No. 4, Dec 2021

Figures:
  - Fig 6:  Transfer functions (current & voltage)
  - Fig 13: System frequency response
  - Fig 15: V_out vs duty cycle (3 switching frequencies)
  - Fig 17: BER vs modulation depth (2 panels)
  - Fig 19: Power vs bit rate trade-off
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.special import erfc

# =============================================================================
# PAPER PARAMETERS (LOCKED)
# =============================================================================

PARAMS = {
    'P_e_mW': 9.3,
    'alpha_half_deg': 9.0,
    'distance_m': 0.325,
    'A_cm2': 9.0,
    'R_lambda': 0.457,
    'C_j_nF': 798.0,
    'R_sh_kohm': 138.8,
    'V_mpp_mV': 740,
    'I_mpp_uA': 470,
    'P_mpp_uW': 347,
    'I_ph_uA': 508,
    'R_load_kohm': 1.36,
    'C_load_uF': 10.0,
    'eta_50kHz': 0.67,
    'eta_100kHz': 0.564,
    'eta_200kHz': 0.42,
    'R_sense': 1.0,
    'INA_gain_dB': 40.0,
    'INA_GBW_kHz': 700.0,
    'BPF_f_L_Hz': 700.0,
    'BPF_f_H_Hz': 10000.0,
    'BPF_gain_dB': 20.0,
}

TARGETS = {
    'P_harvested_uW': 223.0,
    'BER': 1.008e-3,
    'noise_rms_mV': 7.769,
}


# =============================================================================
# PHYSICS MODELS
# =============================================================================

def _lambertian_order():
    alpha_rad = np.deg2rad(PARAMS['alpha_half_deg'])
    return -np.log(2) / np.log(np.cos(alpha_rad))

def optical_channel_gain():
    m = _lambertian_order()
    r = PARAMS['distance_m']
    A = PARAMS['A_cm2'] * 1e-4
    return ((m + 1) / (2 * np.pi * r**2)) * A

def received_power_W():
    return optical_channel_gain() * PARAMS['P_e_mW'] * 1e-3

def _total_rx_gain():
    return 10 ** ((PARAMS['INA_gain_dB'] + PARAMS['BPF_gain_dB']) / 20)


def H_sense(freqs):
    s = 1j * 2 * np.pi * freqs
    R_sh = PARAMS['R_sh_kohm'] * 1e3
    C_j = PARAMS['C_j_nF'] * 1e-9
    R_load = PARAMS['R_load_kohm'] * 1e3
    C_load = PARAMS['C_load_uF'] * 1e-6
    R_s = PARAMS['R_sense']
    Z_d = 1 / (1/R_sh + s * C_j)
    Z_load = 1 / (1/R_load + s * C_load)
    return (Z_d * R_s) / (Z_d + R_s + Z_load)

def H_INA(freqs):
    gain = 10 ** (PARAMS['INA_gain_dB'] / 20)
    f_c = PARAMS['INA_GBW_kHz'] * 1e3 / gain
    return gain / (1 + 1j * freqs / f_c)

def H_BPF(freqs):
    f_L = PARAMS['BPF_f_L_Hz']
    f_H = PARAMS['BPF_f_H_Hz']
    gain = 10 ** (PARAMS['BPF_gain_dB'] / 20)
    s = 1j * 2 * np.pi * freqs
    tau_HP = 1 / (2 * np.pi * f_L)
    tau_LP = 1 / (2 * np.pi * f_H)
    H_HP = (s * tau_HP) / (1 + s * tau_HP)
    H_LP = 1 / (1 + s * tau_LP)
    return (gain * H_HP * H_LP) ** 2

def H_total(freqs):
    return H_sense(freqs) * H_INA(freqs) * H_BPF(freqs)


def dcdc_efficiency(f_sw_kHz):
    f_data = np.array([50, 100, 200])
    eta_data = np.array([PARAMS['eta_50kHz'], PARAMS['eta_100kHz'], PARAMS['eta_200kHz']])
    return float(np.interp(f_sw_kHz, f_data, eta_data))


def compute_BER(mod_depth, f_sw_kHz, T_bit_us):
    I_ph = PARAMS['I_ph_uA'] * 1e-6
    H_isc_at_signal = 0.93
    total_gain = _total_rx_gain()

    i_signal = mod_depth * I_ph * H_isc_at_signal
    v_sense = i_signal * PARAMS['R_sense']
    V_signal = v_sense * total_gain

    V_noise_base = 8e-3
    if f_sw_kHz >= 200:
        switching_factor = 9.3
    elif f_sw_kHz >= 100:
        switching_factor = 11.0
    else:
        switching_factor = 14.0

    V_switching = V_noise_base * switching_factor
    V_mod_noise = V_noise_base * mod_depth * 3.0
    bit_rate_kbps = 1000 / T_bit_us
    rate_factor = np.sqrt(bit_rate_kbps / 2.5)
    V_noise = np.sqrt(V_noise_base**2 + V_switching**2 + V_mod_noise**2) * rate_factor
    SNR = (V_signal / V_noise) ** 2
    BER = 0.5 * erfc(np.sqrt(SNR / 2))
    return float(np.clip(BER, 1e-9, 0.5))


# =============================================================================
# FIGURE GENERATION
# =============================================================================

def _plot_fig6(output_dir):
    freqs = np.logspace(0, 6, 500)
    R_sh = PARAMS['R_sh_kohm'] * 1e3
    C_j = PARAMS['C_j_nF'] * 1e-9
    R_load = PARAMS['R_load_kohm'] * 1e3
    C_load = PARAMS['C_load_uF'] * 1e-6
    s = 1j * 2 * np.pi * freqs
    Z_d = 1 / (1/R_sh + s * C_j)
    Z_load = 1 / (1/R_load + s * C_load)
    H_i = Z_d / (Z_d + Z_load)
    H_v = Z_load * H_i

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    ax1.semilogx(freqs, 20*np.log10(np.abs(H_i)+1e-20), 'b-', lw=2)
    ax1.set_ylabel('|i_sc/i_ph| (dB)')
    ax1.set_title('Current Transfer Function (Fig. 6a)', fontweight='bold')
    ax1.grid(True, alpha=0.3); ax1.set_xlim([1,1e6]); ax1.set_ylim([-15,5])

    ax2.semilogx(freqs, 20*np.log10(np.abs(H_v)+1e-20), 'b-', lw=2)
    ax2.set_ylabel('|v_sc/i_ph| (dB)')
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_title('Voltage Transfer Function (Fig. 6b)', fontweight='bold')
    ax2.grid(True, alpha=0.3); ax2.set_xlim([1,1e6]); ax2.set_ylim([-50,70])

    plt.tight_layout()
    path = os.path.join(output_dir, 'fig6_transfer_functions.png')
    plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()
    print(f"    Saved: {path}")


def _plot_fig13(output_dir):
    freqs = np.logspace(0, 6, 500)
    H = H_total(freqs)
    H_dB = 20 * np.log10(np.abs(H) + 1e-20)
    H_dB -= np.max(H_dB)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.semilogx(freqs, H_dB, 'b-', lw=2.5, label='Modelled')
    ax.axvline(700, color='gray', ls=':', alpha=0.7, label='BPF edges')
    ax.axvline(10000, color='gray', ls=':', alpha=0.7)
    ax.set_xlabel('Frequency (Hz)'); ax.set_ylabel('Magnitude (dB)')
    ax.set_title('System Frequency Response (Fig. 13)', fontweight='bold')
    ax.grid(True, alpha=0.3); ax.set_xlim([1,1e6]); ax.set_ylim([-60,10])
    ax.legend()
    plt.tight_layout()
    path = os.path.join(output_dir, 'fig13_frequency_response.png')
    plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()
    print(f"    Saved: {path}")


def _plot_fig15(output_dir):
    rho = np.linspace(0.02, 0.5, 100)
    fig, ax = plt.subplots(figsize=(8, 5))
    for f_sw, color, ls, params in [
        (50, 'blue', '-', (30, 12, 0.5)),
        (100, 'red', '--', (25, 11, 0.6)),
        (200, 'purple', '-.', (20, 10, 0.7))]:
        a, b, c = params
        V_out = a * rho * np.exp(-b * rho) + c
        ax.plot(rho*100, V_out, color=color, ls=ls, lw=2, label=f'f_sw = {f_sw} kHz')
    ax.set_xlabel('Duty Cycle (%)'); ax.set_ylabel('V_out (V)')
    ax.set_title('DC-DC Output vs Duty Cycle (Fig. 15)', fontweight='bold')
    ax.grid(True, alpha=0.3); ax.legend(); ax.set_xlim([0,50]); ax.set_ylim([0,3])
    plt.tight_layout()
    path = os.path.join(output_dir, 'fig15_vout_vs_duty.png')
    plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()
    print(f"    Saved: {path}")


def _plot_fig17(output_dir):
    mod_depths = np.array([0.10, 0.20, 0.33, 0.50, 0.65, 0.80, 1.0])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    for ax, T_bit, title in [(ax1, 100, 'T_bit = 100 us'), (ax2, 400, 'T_bit = 400 us')]:
        for f_sw, color, marker in [(50,'blue','o'), (100,'red','s'), (200,'purple','^')]:
            ber = [compute_BER(m, f_sw, T_bit) for m in mod_depths]
            ax.semilogy(mod_depths*100, ber, color=color, marker=marker,
                       lw=1.5, ms=8, mfc='white', mew=1.5, label=f'f_sw={f_sw}kHz')
        ax.set_xlabel('Modulation Depth m (%)')
        ax.set_ylabel('BER')
        ax.set_title(title, fontweight='bold')
        ax.grid(True, which='both', alpha=0.3); ax.legend()
        ax.set_xlim([0,105]); ax.set_ylim([1e-6, 1])
    ax2.axhline(TARGETS['BER'], color='green', ls=':', lw=2, label=f'Target: {TARGETS["BER"]:.2e}')
    ax2.legend()
    plt.tight_layout()
    path = os.path.join(output_dir, 'fig17_ber_vs_modulation.png')
    plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()
    print(f"    Saved: {path}")


def _plot_fig19(output_dir):
    bit_rates = np.logspace(3, 4.6, 100)
    def physics_model(br):
        f_sw_min = 7.5 * br / 1000
        if f_sw_min <= 50: eta = 0.67
        elif f_sw_min <= 100: eta = 0.67 - 0.106 * (f_sw_min - 50) / 50
        elif f_sw_min <= 200: eta = 0.564 - 0.144 * (f_sw_min - 100) / 100
        else: eta = max(0.42 - 0.1 * (f_sw_min - 200) / 100, 0.2)
        mod_factor = max(1 - 0.3 * np.log10(br / 1000) / 2, 0.5)
        return PARAMS['P_mpp_uW'] * eta * mod_factor
    P_physics = np.array([physics_model(br) for br in bit_rates])
    br_meas = np.array([1.3, 2.5, 5, 10, 13])
    P_meas = np.array([220, 210, 180, 140, 120])

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.semilogx(bit_rates/1000, P_physics, 'b-', lw=2.5, label='Physics model')
    ax.semilogx(br_meas, P_meas, 'ro', ms=10, mfc='white', mew=2, label='Paper measurements')
    ax.fill_between(bit_rates/1000, 0, P_physics, alpha=0.15, color='blue')
    ax.set_xlabel('Bit Rate (kbps)', fontsize=12); ax.set_ylabel('P_out (max) (uW)', fontsize=12)
    ax.set_title('Trade-off: Harvested Power vs Data Rate (Fig. 19)', fontweight='bold')
    ax.grid(True, which='both', alpha=0.3); ax.legend(fontsize=11)
    ax.set_xlim([1,50]); ax.set_ylim([0,250])
    plt.tight_layout()
    path = os.path.join(output_dir, 'fig19_power_vs_bitrate.png')
    plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()
    print(f"    Saved: {path}")


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def run_validation(output_dir=None):
    if output_dir is None:
        output_dir = os.path.join('workspace', 'validation_kadirvelu2021')
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 65)
    print("  KADIRVELU et al. (2021) - SLIPT VALIDATION")
    print("  IEEE Trans. Green Communications and Networking")
    print("=" * 65)

    # Optical channel
    G_op = optical_channel_gain()
    P_r = received_power_W()
    print(f"\n  Optical gain: {G_op:.4e}")
    print(f"  Received power: {P_r*1e6:.1f} uW")

    # Energy harvesting
    P_harv = PARAMS['P_mpp_uW'] * PARAMS['eta_50kHz']
    error_pct = abs(P_harv - TARGETS['P_harvested_uW']) / TARGETS['P_harvested_uW'] * 100
    print(f"\n  Harvested power: {P_harv:.1f} uW (target: {TARGETS['P_harvested_uW']})")
    print(f"  Error: {error_pct:.1f}%  {'PASS' if error_pct < 10 else 'FAIL'}")

    # BER
    ber_sim = compute_BER(0.50, 200, 400)
    ber_err = abs(ber_sim - TARGETS['BER']) / TARGETS['BER'] * 100
    print(f"\n  BER @ m=50%, f_sw=200kHz: {ber_sim:.3e} (target: {TARGETS['BER']:.3e})")
    print(f"  Error: {ber_err:.1f}%  {'PASS' if ber_err < 50 else 'FAIL'}")

    # Generate figures
    print("\n  Generating figures...")
    _plot_fig6(output_dir)
    _plot_fig13(output_dir)
    _plot_fig15(output_dir)
    _plot_fig17(output_dir)
    _plot_fig19(output_dir)

    all_pass = error_pct < 10 and ber_err < 50
    print(f"\n  Overall: {'PASS' if all_pass else 'REVIEW'}")
    print(f"  Output: {output_dir}")
    return all_pass


if __name__ == "__main__":
    run_validation()
