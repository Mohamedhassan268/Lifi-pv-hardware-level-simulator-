# =============================================================================
# physics/led_emission.py — LED Emission Physics
# =============================================================================
# Task 5 of Hardware-Faithful Simulator
#
# Derives LED optical output from material bandgap, drive current,
# radiative efficiency, and thermal properties. Generates emission
# spectra and radiation patterns from physics rather than hardcoded values.
#
# References:
#   - Sze & Ng, Ch. 12 "LEDs and Lasers"
#   - Schubert, "Light-Emitting Diodes", 2nd Ed.
#   - OSRAM LR W5SN datasheet (validation target)
# =============================================================================

import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from materials.properties import bandgap
from materials.reference_data import CONSTANTS

q = CONSTANTS['q']
h = CONSTANTS['h']
c = CONSTANTS['c']
k_B = CONSTANTS['k_B']


# =============================================================================
# EMISSION SPECTRUM
# =============================================================================

def peak_wavelength_from_bandgap(material_name: str, T: float = 300.0) -> float:
    """
    Peak emission wavelength from material bandgap.

    λ_peak ≈ hc / E_g = 1240 / E_g(eV)   [nm]

    Note: real LEDs emit slightly below the bandgap energy due to
    thermalization. The peak is at E_g + 0.5*k_B*T approximately.

    Args:
        material_name: Material key
        T: Temperature in K

    Returns:
        λ_peak in nm
    """
    E_g = bandgap(material_name, T)
    # Thermal broadening shifts peak slightly to higher energy
    kT_eV = k_B * T / q
    E_peak = E_g + 0.5 * kT_eV  # ~13 meV shift at 300K
    return 1240.0 / E_peak


def emission_spectrum(peak_wavelength_nm: float,
                      fwhm_nm: float = 20.0,
                      wavelengths_nm: np.ndarray = None) -> dict:
    """
    LED emission spectrum modeled as a Gaussian in wavelength space.

    Real LED spectra are approximately Gaussian with FWHM determined
    by the alloy composition and temperature broadening.

    S(λ) = exp( -4·ln(2)·(λ - λ_peak)² / FWHM² )

    Args:
        peak_wavelength_nm: Center wavelength in nm
        fwhm_nm: Full-width at half-maximum in nm (typ. 15-30nm for AlInGaP)
        wavelengths_nm: Array of wavelengths (default: peak ± 3*FWHM)

    Returns:
        Dict with:
            'wavelength_nm': wavelength array
            'power_normalized': normalized spectral power (peak = 1.0)
            'power_density': spectral power density (integrates to 1.0)
    """
    if wavelengths_nm is None:
        hw = 3 * fwhm_nm  # ±3 FWHM covers >99% of power
        wavelengths_nm = np.arange(
            peak_wavelength_nm - hw,
            peak_wavelength_nm + hw + 1,
            1.0
        )

    sigma = fwhm_nm / (2 * np.sqrt(2 * np.log(2)))
    gaussian = np.exp(-0.5 * ((wavelengths_nm - peak_wavelength_nm) / sigma) ** 2)

    # Normalize so integral = 1
    integral = np.trapezoid(gaussian, wavelengths_nm)
    power_density = gaussian / integral if integral > 0 else gaussian

    return {
        'wavelength_nm': wavelengths_nm,
        'power_normalized': gaussian,
        'power_density': power_density,
        'peak_wavelength_nm': peak_wavelength_nm,
        'fwhm_nm': fwhm_nm,
    }


def spectral_fwhm_from_temperature(E_g_eV: float, T: float = 300.0) -> float:
    """
    Estimate spectral FWHM from temperature broadening.

    FWHM ≈ 1.8 · k_B · T  (in eV, for thermal broadening)

    Convert to nm: Δλ ≈ λ_peak² · ΔE / (h·c)

    Source: Schubert, "Light-Emitting Diodes", Eq. 5.9

    Args:
        E_g_eV: Bandgap energy in eV
        T: Temperature in K

    Returns:
        FWHM in nm
    """
    kT_eV = k_B * T / q
    delta_E = 1.8 * kT_eV  # eV
    lambda_peak = 1240.0 / E_g_eV  # nm

    # Δλ = λ² · ΔE / (h·c) in consistent units
    # λ in nm, ΔE in eV → Δλ = λ² · ΔE / 1240
    fwhm_nm = lambda_peak**2 * delta_E / 1240.0

    return fwhm_nm


# =============================================================================
# OPTICAL POWER
# =============================================================================

def optical_power(I_drive_A: float, V_f_V: float,
                  eta_radiant: float = 0.15) -> float:
    """
    LED optical output power from drive current and efficiency.

    P_opt = η_rad · P_elec = η_rad · V_f · I_drive

    Where η_rad is the wall-plug (radiant) efficiency, which includes:
        η_rad = η_internal · η_extraction · η_electrical

    Args:
        I_drive_A: Drive current in Amperes
        V_f_V: Forward voltage in Volts
        eta_radiant: Radiant (wall-plug) efficiency (0-1)

    Returns:
        P_opt in Watts

    Source: Schubert Ch. 5
    """
    P_elec = V_f_V * I_drive_A
    return eta_radiant * P_elec


def optical_power_vs_current(I_array_A: np.ndarray, V_f_V: float,
                              R_s_ohm: float = 5.0,
                              eta_radiant: float = 0.15) -> np.ndarray:
    """
    Optical power vs. drive current including series resistance droop.

    V_f(I) = V_f0 + I · R_s
    η(I) = η_0 · (1 - droop_factor · I)  [simple efficiency droop model]

    Args:
        I_array_A: Array of drive currents
        V_f_V: Forward voltage at nominal current
        R_s_ohm: Series resistance
        eta_radiant: Nominal radiant efficiency

    Returns:
        Array of optical powers in Watts
    """
    V_f_actual = V_f_V + I_array_A * R_s_ohm
    P_elec = V_f_actual * I_array_A

    # Simple droop model: efficiency drops at high current
    # Typical droop onset ~100mA for power LEDs
    droop = 1.0 / (1.0 + (I_array_A / 0.5)**0.5)  # Soft droop
    eta_actual = eta_radiant * droop

    return eta_actual * P_elec


# =============================================================================
# RADIATION PATTERN (Lambertian)
# =============================================================================

def lambertian_order(half_angle_deg: float) -> float:
    """
    Lambertian order from half-power angle.

    m = -ln(2) / ln(cos(θ_half))

    For bare LED (θ_half ≈ 60°): m ≈ 1.0
    For lens-focused (θ_half ≈ 9°): m ≈ 53

    Args:
        half_angle_deg: Half-power angle in degrees

    Returns:
        Lambertian order m (dimensionless)

    Source: Schubert Eq. 5.39
    """
    theta = np.radians(half_angle_deg)
    cos_theta = np.cos(theta)

    if cos_theta <= 0 or cos_theta >= 1:
        return 1.0  # Fallback to Lambertian

    return -np.log(2) / np.log(cos_theta)


def radiation_pattern(theta_deg: np.ndarray, m_order: float) -> np.ndarray:
    """
    Normalized LED radiation intensity pattern.

    I(θ) = cos^m(θ)

    Args:
        theta_deg: Array of angles in degrees (0 = on-axis)
        m_order: Lambertian order

    Returns:
        Normalized intensity array (on-axis = 1.0)
    """
    theta_rad = np.radians(theta_deg)
    return np.clip(np.cos(theta_rad), 0, 1) ** m_order


def irradiance_at_distance(P_opt_W: float, distance_m: float,
                            m_order: float, theta_deg: float = 0.0) -> float:
    """
    Irradiance (W/m²) at a given distance and angle.

    E(d, θ) = P_opt · (m+1) / (2π·d²) · cos^m(θ)

    Args:
        P_opt_W: Total optical power in Watts
        distance_m: Distance from LED in meters
        m_order: Lambertian order
        theta_deg: Off-axis angle in degrees

    Returns:
        Irradiance in W/m²

    Source: Schubert Eq. 5.41
    """
    theta_rad = np.radians(theta_deg)
    cos_m = np.clip(np.cos(theta_rad), 0, 1) ** m_order

    return P_opt_W * (m_order + 1) / (2 * np.pi * distance_m**2) * cos_m


# =============================================================================
# JUNCTION TEMPERATURE
# =============================================================================

def junction_temperature(I_drive_A: float, V_f_V: float,
                          eta_radiant: float, R_th_K_per_W: float,
                          T_ambient_K: float = 300.0) -> float:
    """
    LED junction temperature from thermal resistance model.

    P_heat = P_elec - P_opt = P_elec · (1 - η_rad)
    T_j = T_ambient + P_heat · R_th

    Args:
        I_drive_A: Drive current in Amperes
        V_f_V: Forward voltage in Volts
        eta_radiant: Radiant efficiency (0-1)
        R_th_K_per_W: Thermal resistance junction-to-ambient (K/W)
        T_ambient_K: Ambient temperature in Kelvin

    Returns:
        Junction temperature in Kelvin

    Source: Standard thermal model
    """
    P_elec = V_f_V * I_drive_A
    P_heat = P_elec * (1.0 - eta_radiant)

    return T_ambient_K + P_heat * R_th_K_per_W


def wavelength_shift_with_temperature(lambda_0_nm: float,
                                       T_j_K: float,
                                       T_ref_K: float = 300.0,
                                       coeff_nm_per_K: float = 0.1) -> float:
    """
    Emission wavelength red-shift with junction temperature.

    λ(T) = λ_0 + dλ/dT · (T_j - T_ref)

    Typical: ~0.1 nm/K for AlInGaP red LEDs
             ~0.03 nm/K for InGaN blue LEDs

    Args:
        lambda_0_nm: Nominal peak wavelength at T_ref
        T_j_K: Junction temperature in K
        T_ref_K: Reference temperature in K
        coeff_nm_per_K: Temperature coefficient

    Returns:
        Shifted wavelength in nm
    """
    return lambda_0_nm + coeff_nm_per_K * (T_j_K - T_ref_K)


# =============================================================================
# SELF-TEST
# =============================================================================

def test_led_emission():
    """Verify LED emission physics against OSRAM LR W5SN targets."""
    print("=" * 70)
    print("LED EMISSION PHYSICS — TEST SUITE")
    print("=" * 70)
    passes = 0
    fails = 0

    def check(name, val, expected, tol_pct=10.0, unit=""):
        nonlocal passes, fails
        err = abs(val - expected) / abs(expected) * 100 if expected != 0 else abs(val)
        status = "PASS" if err < tol_pct else "FAIL"
        if status == "FAIL":
            fails += 1
        else:
            passes += 1
        print(f"  {name}: {val:.4g} {unit} (expected ~{expected:.4g}, err={err:.1f}%) [{status}]")

    # --- Peak wavelength from bandgap ---
    print("\n1. PEAK WAVELENGTH FROM BANDGAP")
    lam_AlInGaP = peak_wavelength_from_bandgap('AlInGaP', 300)
    check("λ_peak(AlInGaP)", lam_AlInGaP, 645, tol_pct=5, unit="nm")
    # AlInGaP E_g≈1.91eV → ~649nm, with thermal shift ~645nm
    # OSRAM LR W5SN: 625nm (alloy tuned below pure bandgap)

    lam_GaN = peak_wavelength_from_bandgap('GaN', 300)
    print(f"  λ_peak(GaN) = {lam_GaN:.0f} nm (UV — correct for pure GaN)")

    # --- Emission spectrum ---
    print("\n2. EMISSION SPECTRUM")
    spec = emission_spectrum(625, fwhm_nm=20)
    wl = spec['wavelength_nm']
    pw = spec['power_normalized']

    # Peak should be at 625nm
    peak_idx = np.argmax(pw)
    check("Spectrum peak", wl[peak_idx], 625, tol_pct=1, unit="nm")

    # FWHM verification: find where power drops to 0.5
    half_max_points = wl[pw >= 0.49]
    measured_fwhm = half_max_points[-1] - half_max_points[0]
    check("Spectrum FWHM", measured_fwhm, 20, tol_pct=10, unit="nm")

    # Power density integrates to 1.0
    integral = np.trapezoid(spec['power_density'], wl)
    check("Spectrum integral", integral, 1.0, tol_pct=1)

    # --- Thermal FWHM estimate ---
    print("\n3. THERMAL FWHM")
    fwhm_thermal = spectral_fwhm_from_temperature(1.91, 300)
    print(f"  FWHM(AlInGaP, 300K) = {fwhm_thermal:.1f} nm (thermal component only)")
    # Real FWHM includes alloy broadening → typically 15-30nm for AlInGaP
    # Thermal alone: ~10-15nm

    # --- Optical power ---
    print("\n4. OPTICAL POWER")
    # OSRAM LR W5SN: V_f=2.1V, η_rad≈15%
    P_20mA = optical_power(20e-3, 2.1, 0.15)
    check("P_opt(20mA)", P_20mA * 1e3, 6.3, tol_pct=1, unit="mW")
    # 0.15 × 2.1V × 20mA = 6.3 mW

    P_350mA = optical_power(350e-3, 2.1, 0.15)
    check("P_opt(350mA)", P_350mA * 1e3, 110.25, tol_pct=1, unit="mW")

    # --- Lambertian order ---
    print("\n5. LAMBERTIAN ORDER")
    m_bare = lambertian_order(60)
    check("m(60° bare)", m_bare, 1.0, tol_pct=5)

    m_lens = lambertian_order(9)
    check("m(9° with lens)", m_lens, 53, tol_pct=10)
    # Kadirvelu uses 9° half-angle with Fraen lens

    # --- Radiation pattern ---
    print("\n6. RADIATION PATTERN")
    angles = np.array([0, 5, 9, 15, 30, 45, 60, 90])
    pattern_bare = radiation_pattern(angles, 1.0)
    pattern_lens = radiation_pattern(angles, m_lens)

    print(f"  Bare LED:  I(0°)={pattern_bare[0]:.2f}, I(30°)={pattern_bare[4]:.2f}, I(60°)={pattern_bare[6]:.2f}")
    print(f"  With lens: I(0°)={pattern_lens[0]:.2f}, I(9°)={pattern_lens[2]:.2f}, I(15°)={pattern_lens[3]:.2f}")

    # At half-angle, intensity should be 0.5
    I_at_half = radiation_pattern(np.array([9.0]), m_lens)[0]
    check("I(θ_half) = 0.5", I_at_half, 0.5, tol_pct=5)

    # --- Irradiance ---
    print("\n7. IRRADIANCE AT DISTANCE")
    P_opt = 110e-3  # 110 mW
    d = 0.325       # 32.5 cm (Kadirvelu setup)
    E_0 = irradiance_at_distance(P_opt, d, m_lens, theta_deg=0)
    print(f"  E(0°, 32.5cm, 110mW) = {E_0:.1f} W/m²")
    # Expected: high concentration due to narrow beam
    assert E_0 > 5, "Irradiance should be significant at 32.5cm"
    print("  [PASS]")
    passes += 1

    # --- Junction temperature ---
    print("\n8. JUNCTION TEMPERATURE")
    T_j = junction_temperature(350e-3, 2.1, 0.15, R_th_K_per_W=250, T_ambient_K=300)
    check("T_j(350mA)", T_j, 456, tol_pct=10, unit="K")
    # P_heat = 2.1×0.35×0.85 = 0.624W, ΔT = 0.624×250 = 156K → T_j ≈ 456K

    # --- Wavelength shift ---
    print("\n9. WAVELENGTH SHIFT WITH TEMPERATURE")
    T_j_actual = junction_temperature(350e-3, 2.1, 0.15, 250, 300)
    lam_shifted = wavelength_shift_with_temperature(625, T_j_actual, 300, 0.1)
    delta = lam_shifted - 625
    print(f"  λ shift at T_j={T_j_actual:.0f}K: +{delta:.1f} nm ", end="")
    assert 10 < delta < 25, "Wavelength shift should be 10-25nm"
    print("[PASS]")
    passes += 1

    # --- Summary ---
    print(f"\n{'=' * 70}")
    print(f"RESULTS: {passes} passed, {fails} failed")
    print(f"{'=' * 70}")
    return fails == 0


if __name__ == '__main__':
    test_led_emission()
