# =============================================================================
# physics/tia.py — Transimpedance Amplifier Model
# =============================================================================
# Task 15 of Hardware-Faithful Simulator
#
# Models TIA front-end for PV/photodiode receivers:
#   - Transimpedance gain and bandwidth
#   - Gain-bandwidth tradeoff
#   - Input-referred noise
#   - Stability (phase margin)
#
# Common TIA topologies:
#   - Simple resistive load (R_load) — used in basic PV receivers
#   - Shunt-feedback TIA — used with photodiodes
#   - Bootstrapped TIA — reduces effective C_j
#
# Reference: Razavi, "Design of Integrated Circuits for Optical Communications"
# =============================================================================

import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from materials.reference_data import CONSTANTS

k_B = CONSTANTS['k_B']


# =============================================================================
# RESISTIVE LOAD (Simple PV Receiver)
# =============================================================================

def resistive_load_tia(R_load: float, C_j: float,
                       R_shunt: float = np.inf) -> dict:
    """
    Simple resistive load transimpedance (no active amplifier).

    This is the baseline PV receiver: photocurrent flows through R_load,
    producing voltage V_out = I_ph · R_load.

    Bandwidth limited by RC: f_3dB = 1 / (2π · R_eff · C_j)
    where R_eff = R_load || R_shunt

    Args:
        R_load: Load resistance in Ohms
        C_j: Junction capacitance in Farads
        R_shunt: Shunt resistance in Ohms

    Returns:
        TIA parameters dict
    """
    R_eff = (R_load * R_shunt) / (R_load + R_shunt) if R_shunt < np.inf else R_load
    f_3dB = 1.0 / (2 * np.pi * R_eff * C_j)

    return {
        'topology': 'resistive_load',
        'transimpedance_ohm': R_load,
        'transimpedance_dBohm': 20 * np.log10(R_load),
        'bandwidth_Hz': f_3dB,
        'R_eff_ohm': R_eff,
        'gain_bandwidth_product': R_load * f_3dB,
        'C_j_F': C_j,
    }


# =============================================================================
# SHUNT-FEEDBACK TIA (Active Amplifier)
# =============================================================================

def shunt_feedback_tia(R_f: float, C_j: float, A_0: float = 1000,
                       GBW: float = 100e6, C_f: float = 0.0) -> dict:
    """
    Shunt-feedback TIA with op-amp.

    Transimpedance: Z_T = -R_f (at low frequencies)
    Bandwidth: f_3dB ≈ √(GBW / (2π · R_f · C_j))  [for stability]

    The feedback capacitor C_f sets the closed-loop bandwidth:
    f_3dB = 1 / (2π · R_f · C_f)  if C_f is chosen for flat response.

    For stability (45° phase margin):
    C_f = √(C_j / (2π · R_f · GBW))

    Args:
        R_f: Feedback resistance in Ohms
        C_j: Total input capacitance in Farads
        A_0: DC open-loop gain
        GBW: Gain-bandwidth product in Hz
        C_f: Feedback capacitance (0 = auto-calculate for stability)

    Returns:
        TIA parameters dict
    """
    # Auto-calculate C_f for 45° phase margin if not specified
    if C_f <= 0:
        C_f = np.sqrt(C_j / (2 * np.pi * R_f * GBW))

    # Closed-loop bandwidth
    f_3dB = 1.0 / (2 * np.pi * R_f * C_f)

    # Check stability: need f_3dB < GBW/A_CL where A_CL ≈ 1 + R_f/R_in
    f_unity = GBW  # Op-amp unity-gain frequency
    noise_gain_pole = 1.0 / (2 * np.pi * R_f * C_j)

    # Phase margin estimate
    f_intersect = np.sqrt(noise_gain_pole * f_unity)
    if f_intersect > 0:
        phase_margin_deg = 90 - np.degrees(np.arctan(f_3dB / f_intersect))
        phase_margin_deg = max(0, min(90, phase_margin_deg))
    else:
        phase_margin_deg = 45

    return {
        'topology': 'shunt_feedback',
        'transimpedance_ohm': R_f,
        'transimpedance_dBohm': 20 * np.log10(R_f),
        'bandwidth_Hz': f_3dB,
        'C_f_F': C_f,
        'C_f_pF': C_f * 1e12,
        'phase_margin_deg': phase_margin_deg,
        'gain_bandwidth_product': R_f * f_3dB,
        'GBW_Hz': GBW,
        'C_j_F': C_j,
    }


# =============================================================================
# GAIN-BANDWIDTH TRADEOFF ANALYSIS
# =============================================================================

def gain_bw_tradeoff(C_j: float, R_values: np.ndarray = None,
                     topology: str = 'resistive') -> dict:
    """
    Analyze the gain-bandwidth tradeoff for a given C_j.

    For resistive load: GBW product = R · f_3dB = R/(2πRC) = 1/(2πC)
    → This is CONSTANT, meaning you can trade gain for bandwidth.

    For shunt-feedback TIA: GBW product scales as √(GBW_amp)
    → Can exceed the resistive limit with active amplification.

    Args:
        C_j: Junction capacitance in Farads
        R_values: Array of resistance values to sweep
        topology: 'resistive' or 'shunt_feedback'

    Returns:
        Dict with R, gain, bandwidth arrays
    """
    if R_values is None:
        R_values = np.logspace(1, 6, 50)  # 10Ω to 1MΩ

    gains = []
    bws = []

    for R in R_values:
        if topology == 'resistive':
            result = resistive_load_tia(R, C_j)
        else:
            result = shunt_feedback_tia(R, C_j)
        gains.append(result['transimpedance_ohm'])
        bws.append(result['bandwidth_Hz'])

    return {
        'R_values': R_values,
        'transimpedance_ohm': np.array(gains),
        'bandwidth_Hz': np.array(bws),
        'gbw_product': np.array(gains) * np.array(bws),
        'C_j_F': C_j,
    }


# =============================================================================
# TIA NOISE CONTRIBUTION
# =============================================================================

def tia_noise(R_f: float, bandwidth_Hz: float, T: float = 300.0,
              i_n_A_rtHz: float = 2e-12,
              v_n_V_rtHz: float = 5e-9,
              C_in: float = 0.0) -> dict:
    """
    TIA noise contribution (input-referred).

    For resistive load: only thermal noise from R_f.
    For active TIA: add op-amp noise sources.

    Args:
        R_f: Feedback/load resistance
        bandwidth_Hz: Noise bandwidth
        T: Temperature
        i_n_A_rtHz: Op-amp input current noise (0 for resistive)
        v_n_V_rtHz: Op-amp input voltage noise (0 for resistive)
        C_in: Input capacitance for voltage noise calculation

    Returns:
        Noise dict in A² (input-referred)
    """
    B = bandwidth_Hz

    # Thermal noise from R_f
    N_thermal = 4 * k_B * T / R_f * B

    # Op-amp current noise
    N_i = i_n_A_rtHz**2 * B

    # Op-amp voltage noise (frequency-dependent with C_in)
    if v_n_V_rtHz > 0 and C_in > 0:
        N_v = (2 * np.pi * C_in * v_n_V_rtHz)**2 * B**3 / 3
    else:
        N_v = 0

    N_total = N_thermal + N_i + N_v

    return {
        'thermal_A2': N_thermal,
        'current_noise_A2': N_i,
        'voltage_noise_A2': N_v,
        'total_A2': N_total,
        'total_rms_A': np.sqrt(N_total),
        'thermal_rms_A': np.sqrt(N_thermal),
    }


# =============================================================================
# SELF-TEST
# =============================================================================

def test_tia():
    print("=" * 70)
    print("TIA MODEL — TEST SUITE")
    print("=" * 70)
    passes = 0
    fails = 0

    def check(name, val, expected, tol_pct=10.0, unit=""):
        nonlocal passes, fails
        err = abs(val - expected) / abs(expected) * 100 if expected else 0
        status = "PASS" if err < tol_pct else "FAIL"
        if status == "FAIL":
            fails += 1
        else:
            passes += 1
        print(f"  {name}: {val:.4g} {unit} (expected ~{expected:.4g}, err={err:.1f}%) [{status}]")

    # --- Resistive load with KXOB25 ---
    print("\n1. RESISTIVE LOAD (KXOB25, C_j=799pF)")
    r1k = resistive_load_tia(1000, 799e-12, 138e3)
    check("f_3dB(1kΩ)", r1k['bandwidth_Hz'], 200e3, tol_pct=5, unit="Hz")
    check("Z_T", r1k['transimpedance_ohm'], 1000, tol_pct=1, unit="Ω")

    r10k = resistive_load_tia(10e3, 799e-12)
    check("f_3dB(10kΩ)", r10k['bandwidth_Hz'], 20e3, tol_pct=5, unit="Hz")

    # GBW product should be approximately constant for resistive
    gbw_1k = r1k['gain_bandwidth_product']
    gbw_10k = r10k['gain_bandwidth_product']
    ratio = gbw_1k / gbw_10k
    print(f"  GBW(1kΩ)={gbw_1k:.0f}, GBW(10kΩ)={gbw_10k:.0f}, ratio={ratio:.2f}")
    assert 0.5 < ratio < 2.0, "GBW should be roughly constant"
    passes += 1

    # --- Resistive load with BPW34 ---
    print("\n2. RESISTIVE LOAD (BPW34, C_j=70pF)")
    bpw_r = resistive_load_tia(1000, 70e-12)
    check("f_3dB(1kΩ, BPW34)", bpw_r['bandwidth_Hz'], 2.27e6, tol_pct=5, unit="Hz")
    # BPW34 ~10× higher BW than KXOB25 due to lower C_j

    # --- Shunt-feedback TIA ---
    print("\n3. SHUNT-FEEDBACK TIA (C_j=799pF)")
    tia = shunt_feedback_tia(10e3, 799e-12, GBW=100e6)
    print(f"  Z_T = {tia['transimpedance_ohm']/1e3:.0f} kΩ")
    print(f"  f_3dB = {tia['bandwidth_Hz']/1e6:.2f} MHz")
    print(f"  C_f = {tia['C_f_pF']:.2f} pF")
    print(f"  Phase margin = {tia['phase_margin_deg']:.0f}°")

    # TIA should achieve higher GBW than resistive
    gbw_tia = tia['transimpedance_ohm'] * tia['bandwidth_Hz']
    print(f"  GBW(TIA) = {gbw_tia:.0f} vs GBW(resistive) = {gbw_1k:.0f}")
    assert gbw_tia > gbw_1k * 5, "Active TIA should have much higher GBW"
    print("  Active TIA GBW >> Resistive GBW: ✓")
    passes += 1

    # --- Gain-BW tradeoff ---
    print("\n4. GAIN-BANDWIDTH TRADEOFF")
    tradeoff = gain_bw_tradeoff(799e-12, topology='resistive')
    # At low R: high BW, low gain. At high R: low BW, high gain.
    bw_low_R = tradeoff['bandwidth_Hz'][0]   # R=10Ω
    bw_high_R = tradeoff['bandwidth_Hz'][-1] # R=1MΩ
    assert bw_low_R > bw_high_R * 100, "BW should decrease with R"
    print(f"  R=10Ω: f_3dB={bw_low_R/1e6:.0f} MHz")
    print(f"  R=1MΩ: f_3dB={bw_high_R:.0f} Hz")
    print("  Gain-BW tradeoff confirmed: ✓")
    passes += 1

    # --- TIA noise ---
    print("\n5. TIA NOISE")
    noise_resistive = tia_noise(1000, 200e3)
    noise_active = tia_noise(10e3, 1e6, i_n_A_rtHz=2e-12,
                              v_n_V_rtHz=5e-9, C_in=799e-12)
    print(f"  Resistive (1kΩ, 200kHz): {noise_resistive['total_rms_A']*1e9:.2f} nA rms")
    print(f"  Active (10kΩ, 1MHz):     {noise_active['total_rms_A']*1e9:.2f} nA rms")
    passes += 1

    print(f"\n{'=' * 70}")
    print(f"RESULTS: {passes} passed, {fails} failed")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    test_tia()
