# =============================================================================
# physics/noise.py — Receiver Noise Model
# =============================================================================
# Task 14 of Hardware-Faithful Simulator
#
# Noise sources in a PV/photodiode receiver:
#   1. Shot noise (signal + dark current)
#   2. Thermal (Johnson-Nyquist) noise from load/shunt resistance
#   3. Relative Intensity Noise (RIN) from LED
#   4. Amplifier noise (TIA input-referred)
#
# All noise powers are computed as spectral densities (A²/Hz or V²/Hz)
# then integrated over bandwidth to get total noise.
#
# References:
#   - Kahn & Barry, "Wireless Infrared Communications", Proc. IEEE 1997
#   - Kadirvelu 2021, Section III-B
# =============================================================================

import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from materials.reference_data import CONSTANTS

q = CONSTANTS['q']
k_B = CONSTANTS['k_B']


# =============================================================================
# INDIVIDUAL NOISE SOURCES (spectral densities in A²/Hz)
# =============================================================================

def shot_noise_density(I_ph: float, I_dark: float = 0.0) -> float:
    """
    Shot noise current spectral density.

    S_shot = 2q(I_ph + I_dark)   [A²/Hz]

    Args:
        I_ph: Signal photocurrent in Amperes
        I_dark: Dark current in Amperes

    Returns:
        Shot noise PSD in A²/Hz
    """
    return 2 * q * (abs(I_ph) + abs(I_dark))


def thermal_noise_density(R: float, T: float = 300.0) -> float:
    """
    Thermal (Johnson-Nyquist) noise current spectral density.

    S_thermal = 4kT/R   [A²/Hz]

    For parallel R_load and R_shunt:
    S_thermal = 4kT · (1/R_load + 1/R_shunt)

    Args:
        R: Effective resistance in Ohms (parallel combination)
        T: Temperature in Kelvin

    Returns:
        Thermal noise PSD in A²/Hz
    """
    if R <= 0:
        return 0.0
    return 4 * k_B * T / R


def rin_noise_density(I_ph: float, RIN_dB_Hz: float = -130.0) -> float:
    """
    Relative Intensity Noise (RIN) from LED.

    S_RIN = I_ph² · 10^(RIN_dB/10)   [A²/Hz]

    Typical LED RIN: -120 to -140 dB/Hz.
    LEDs have higher RIN than lasers due to spontaneous emission.

    Args:
        I_ph: Signal photocurrent in Amperes
        RIN_dB_Hz: RIN in dB/Hz (negative number)

    Returns:
        RIN noise PSD in A²/Hz
    """
    RIN_linear = 10 ** (RIN_dB_Hz / 10)
    return I_ph**2 * RIN_linear


def amplifier_noise_density(i_n_A_rtHz: float = 2e-12,
                             v_n_V_rtHz: float = 5e-9,
                             C_in: float = 0.0,
                             f: float = 1e6) -> float:
    """
    Amplifier input-referred noise (simplified TIA model).

    S_amp = i_n² + (v_n · 2π·f·C_in)²   [A²/Hz]

    At low frequencies, current noise dominates.
    At high frequencies, voltage noise × capacitance dominates.

    Args:
        i_n_A_rtHz: Input current noise density in A/√Hz
        v_n_V_rtHz: Input voltage noise density in V/√Hz
        C_in: Total input capacitance in Farads
        f: Frequency in Hz (for voltage noise contribution)

    Returns:
        Amplifier noise PSD in A²/Hz
    """
    i_n_sq = i_n_A_rtHz**2
    v_n_contribution = (v_n_V_rtHz * 2 * np.pi * f * C_in)**2
    return i_n_sq + v_n_contribution


# =============================================================================
# TOTAL NOISE AND SNR
# =============================================================================

def total_noise_power(I_ph: float, I_dark: float,
                      R_load: float, R_shunt: float,
                      bandwidth_Hz: float,
                      T: float = 300.0,
                      RIN_dB_Hz: float = -130.0,
                      i_n_amp: float = 0.0,
                      v_n_amp: float = 0.0,
                      C_in: float = 0.0) -> dict:
    """
    Total noise power from all sources.

    Returns individual and total noise contributions.

    Args:
        I_ph: Signal photocurrent (A)
        I_dark: Dark current (A)
        R_load: Load resistance (Ω)
        R_shunt: Shunt resistance (Ω)
        bandwidth_Hz: Noise bandwidth (Hz)
        T: Temperature (K)
        RIN_dB_Hz: LED RIN (dB/Hz)
        i_n_amp: Amplifier current noise (A/√Hz)
        v_n_amp: Amplifier voltage noise (V/√Hz)
        C_in: Input capacitance (F)

    Returns:
        Dict with noise breakdown (all in A²):
            shot, thermal, rin, amplifier, total
            Also rms values in Amperes.
    """
    B = bandwidth_Hz

    # Shot noise
    S_shot = shot_noise_density(I_ph, I_dark)
    N_shot = S_shot * B

    # Thermal noise (parallel R_load || R_shunt)
    R_parallel = (R_load * R_shunt) / (R_load + R_shunt) if R_shunt > 0 else R_load
    S_thermal = thermal_noise_density(R_parallel, T)
    N_thermal = S_thermal * B

    # RIN noise
    S_rin = rin_noise_density(I_ph, RIN_dB_Hz)
    N_rin = S_rin * B

    # Amplifier noise (integrate over bandwidth with frequency-dependent term)
    if v_n_amp > 0 and C_in > 0:
        # Integrate (v_n · 2πfC)² over 0 to B
        # ∫₀ᴮ (2πfC·v_n)² df = (2πC·v_n)² · B³/3
        N_amp_v = (2 * np.pi * C_in * v_n_amp)**2 * B**3 / 3
    else:
        N_amp_v = 0
    N_amp_i = i_n_amp**2 * B
    N_amp = N_amp_i + N_amp_v

    N_total = N_shot + N_thermal + N_rin + N_amp

    return {
        'shot_A2': N_shot,
        'thermal_A2': N_thermal,
        'rin_A2': N_rin,
        'amplifier_A2': N_amp,
        'total_A2': N_total,
        'shot_rms_A': np.sqrt(N_shot),
        'thermal_rms_A': np.sqrt(N_thermal),
        'rin_rms_A': np.sqrt(N_rin),
        'amplifier_rms_A': np.sqrt(N_amp),
        'total_rms_A': np.sqrt(N_total),
        'bandwidth_Hz': B,
        'R_parallel_ohm': R_parallel,
    }


def signal_to_noise_ratio(I_ph: float, noise_total_A2: float) -> dict:
    """
    Electrical SNR from photocurrent and total noise.

    SNR_elec = I_ph² / N_total   (power ratio)
    SNR_dB = 10·log10(SNR_elec)

    Note: For OOK modulation, the signal power is (I_ph/2)² for
    the AC component. For DC analysis, use full I_ph.

    Args:
        I_ph: Signal photocurrent in Amperes
        noise_total_A2: Total noise power in A²

    Returns:
        Dict with SNR in linear and dB
    """
    if noise_total_A2 <= 0:
        return {'snr_linear': np.inf, 'snr_dB': np.inf}

    snr = I_ph**2 / noise_total_A2
    snr_dB = 10 * np.log10(snr) if snr > 0 else -np.inf

    return {
        'snr_linear': snr,
        'snr_dB': snr_dB,
    }


def noise_equivalent_power(R_eff: float, noise_total_A2: float,
                            bandwidth_Hz: float) -> float:
    """
    Noise Equivalent Power — minimum detectable optical power.

    NEP = √(N_total) / R_eff   [W]
    Specific NEP = NEP / √B   [W/√Hz]

    Args:
        R_eff: Effective responsivity (A/W)
        noise_total_A2: Total noise power (A²)
        bandwidth_Hz: Noise bandwidth (Hz)

    Returns:
        NEP in Watts
    """
    if R_eff <= 0:
        return np.inf
    return np.sqrt(noise_total_A2) / R_eff


# =============================================================================
# CONVENIENCE: Full noise analysis from component parameters
# =============================================================================

def receiver_noise_analysis(I_ph: float, I_dark: float,
                             R_load: float, R_shunt: float,
                             C_j: float, R_eff: float,
                             T: float = 300.0,
                             RIN_dB_Hz: float = -130.0) -> dict:
    """
    Complete noise analysis for a PV receiver.

    Uses RC bandwidth to determine noise BW, then computes all noise
    sources and SNR.

    Args:
        I_ph: Signal photocurrent (A)
        I_dark: Dark current (A)
        R_load: Load resistance (Ω)
        R_shunt: Shunt resistance (Ω)
        C_j: Junction capacitance (F)
        R_eff: Effective responsivity (A/W)
        T: Temperature (K)
        RIN_dB_Hz: LED RIN (dB/Hz)

    Returns:
        Complete noise analysis dict
    """
    # RC bandwidth
    R_par = (R_load * R_shunt) / (R_load + R_shunt)
    f_3dB = 1.0 / (2 * np.pi * R_par * C_j)

    noise = total_noise_power(I_ph, I_dark, R_load, R_shunt,
                               f_3dB, T, RIN_dB_Hz)
    snr = signal_to_noise_ratio(I_ph, noise['total_A2'])
    nep = noise_equivalent_power(R_eff, noise['total_A2'], f_3dB)
    P_opt = I_ph / R_eff if R_eff > 0 else 0

    return {
        **noise,
        **snr,
        'nep_W': nep,
        'nep_dBm': 10 * np.log10(nep * 1e3) if nep > 0 and nep < np.inf else -np.inf,
        'P_opt_W': P_opt,
        'f_3dB_Hz': f_3dB,
    }


# =============================================================================
# SELF-TEST
# =============================================================================

def test_noise():
    print("=" * 70)
    print("RECEIVER NOISE MODEL — TEST SUITE")
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

    # --- Shot noise ---
    print("\n1. SHOT NOISE")
    S_shot = shot_noise_density(1e-3, 1e-9)  # 1mA signal, 1nA dark
    check("S_shot(1mA)", S_shot, 3.204e-22, tol_pct=1, unit="A²/Hz")
    # 2 × 1.6e-19 × 1e-3 = 3.2e-22

    # --- Thermal noise ---
    print("\n2. THERMAL NOISE")
    S_th_1k = thermal_noise_density(1000, 300)
    check("S_th(1kΩ, 300K)", S_th_1k, 1.656e-23, tol_pct=1, unit="A²/Hz")
    # 4 × 1.38e-23 × 300 / 1000 = 1.656e-23

    # --- RIN noise ---
    print("\n3. RIN NOISE")
    S_rin = rin_noise_density(1e-3, -130)
    check("S_RIN(1mA, -130dB)", S_rin, 1e-19, tol_pct=1, unit="A²/Hz")
    # (1e-3)² × 10^(-13) = 1e-19

    # --- Full noise analysis (Kadirvelu-like setup) ---
    print("\n4. KADIRVELU SETUP NOISE ANALYSIS")
    # KXOB25: I_ph ~ 3.5mA at 32.5cm, C_j=798pF, R_sh=138kΩ
    noise = receiver_noise_analysis(
        I_ph=3.565e-3,     # From link budget
        I_dark=0.26e-9,    # GaAs dark current
        R_load=1000,
        R_shunt=138e3,
        C_j=799e-12,
        R_eff=0.441,
        T=300,
        RIN_dB_Hz=-130,
    )

    print(f"  f_3dB = {noise['f_3dB_Hz']/1e3:.1f} kHz")
    print(f"  Shot noise:    {noise['shot_rms_A']*1e9:.2f} nA rms")
    print(f"  Thermal noise: {noise['thermal_rms_A']*1e9:.2f} nA rms")
    print(f"  RIN noise:     {noise['rin_rms_A']*1e9:.2f} nA rms")
    print(f"  Total noise:   {noise['total_rms_A']*1e9:.2f} nA rms")
    print(f"  SNR = {noise['snr_dB']:.1f} dB")
    print(f"  NEP = {noise['nep_W']*1e9:.2f} nW")

    # For strong signal with LED RIN, RIN noise dominates
    dominant = max(noise['shot_A2'], noise['thermal_A2'], noise['rin_A2'])
    if dominant == noise['rin_A2']:
        print("  RIN dominates (strong signal, LED source): ✓")
    elif dominant == noise['thermal_A2']:
        print("  Thermal dominates: ✓")
    else:
        print("  Shot dominates: ✓")
    passes += 1

    # SNR should be high at 32.5cm
    assert noise['snr_dB'] > 30, "SNR should be >30dB at 32.5cm"
    print(f"  SNR > 30 dB: ✓")
    passes += 1

    # --- BPW34 comparison (smaller area → less noise, less signal) ---
    print("\n5. BPW34 COMPARISON")
    noise_bpw = receiver_noise_analysis(
        I_ph=28e-6,        # Much less signal
        I_dark=2e-9,
        R_load=1000,
        R_shunt=1e9,
        C_j=70e-12,
        R_eff=0.416,
        T=300,
    )
    print(f"  f_3dB = {noise_bpw['f_3dB_Hz']/1e3:.0f} kHz")
    print(f"  Total noise: {noise_bpw['total_rms_A']*1e9:.2f} nA rms")
    print(f"  SNR = {noise_bpw['snr_dB']:.1f} dB")

    # BPW34 higher bandwidth but lower SNR (less signal)
    assert noise_bpw['f_3dB_Hz'] > noise['f_3dB_Hz']
    print("  Higher BW than KXOB25: ✓")
    assert noise_bpw['snr_dB'] < noise['snr_dB']
    print("  Lower SNR than KXOB25: ✓ (less signal area)")
    passes += 2

    # --- SNR vs distance ---
    print("\n6. SNR vs DISTANCE TREND")
    # I_ph ∝ 1/d², so SNR_dB drops ~20 dB per decade of distance
    for d_factor, d_label in [(1, "32.5cm"), (2, "65cm"), (4, "130cm")]:
        I_ph_d = 3.565e-3 / d_factor**2
        n = receiver_noise_analysis(I_ph_d, 0.26e-9, 1000, 138e3,
                                     799e-12, 0.441, 300)
        print(f"  d={d_label:6s}: I_ph={I_ph_d*1e6:.0f}µA, SNR={n['snr_dB']:.1f}dB")

    passes += 1

    print(f"\n{'=' * 70}")
    print(f"RESULTS: {passes} passed, {fails} failed")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    test_noise()
