# =============================================================================
# physics/photodetection.py — Photodetection Physics
# =============================================================================
# Task 4 of Hardware-Faithful Simulator
#
# Derives responsivity R(λ) and quantum efficiency QE(λ) from material
# properties (bandgap, absorption coefficient) and device geometry
# (junction depth, surface reflectance, collection efficiency).
#
# References:
#   - Sze & Ng, Ch. 13 "Photodetectors"
#   - Pierret, "Semiconductor Fundamentals"
#   - Kadirvelu et al. 2021, IEEE TGCN (validation target)
# =============================================================================

import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from materials.reference_data import CONSTANTS, GAAS_QE_CURVE, BPW34_SPECTRAL_RESPONSE
from materials.properties import (
    bandgap, bandgap_wavelength, absorption_coefficient,
)

q = CONSTANTS['q']
h = CONSTANTS['h']
c = CONSTANTS['c']


# =============================================================================
# QUANTUM EFFICIENCY
# =============================================================================

def quantum_efficiency_analytical(material_name: str, wavelength_nm: float,
                                  absorption_depth_um: float = 2.0,
                                  surface_reflectance: float = 0.05,
                                  collection_efficiency: float = 0.95,
                                  T: float = 300.0) -> float:
    """
    Quantum efficiency from material absorption and device geometry.

    QE(λ) = (1 - R_surface) · (1 - exp(-α·d)) · η_collection

    Where:
        R_surface = surface reflectance (with AR coating)
        α = absorption coefficient at wavelength λ
        d = effective collection depth = junction + depletion + L_minority
            Typical: GaAs ~2µm (0.5 junction + 1 depletion + 0.5 diffusion)
                     Si   ~10-200µm (much longer diffusion lengths)
        η_collection = fraction of generated carriers that are collected

    Args:
        material_name: Material key ('GaAs', 'Si', etc.)
        wavelength_nm: Wavelength in nm
        absorption_depth_um: Total collection depth in µm
            (junction + depletion + minority carrier diffusion length)
        surface_reflectance: Surface reflectance fraction (0-1)
        collection_efficiency: Carrier collection efficiency (0-1)
        T: Temperature in K

    Returns:
        QE as fraction (0-1)

    Source: Sze Eq. 13.3
    """
    # Check if photon energy is above bandgap
    lambda_cutoff = bandgap_wavelength(material_name, T)
    if wavelength_nm > lambda_cutoff * 1.05:  # 5% margin for band tailing
        return 0.0

    # Absorption coefficient in cm⁻¹
    alpha = absorption_coefficient(material_name, wavelength_nm, T)

    # Convert depth to cm
    d_cm = absorption_depth_um * 1e-4

    # QE = (1 - R) · (1 - exp(-α·d)) · η_c
    absorption_fraction = 1.0 - np.exp(-alpha * d_cm)
    qe = (1.0 - surface_reflectance) * absorption_fraction * collection_efficiency

    return np.clip(qe, 0.0, 1.0)


def quantum_efficiency_from_curve(wavelength_nm: float,
                                  qe_data: dict) -> float:
    """
    Quantum efficiency from measured/datasheet QE curve via interpolation.

    This is the hybrid approach — use real measured data when available.

    Args:
        wavelength_nm: Query wavelength in nm
        qe_data: Dict with 'wavelength_nm' (array) and 'QE_fraction' (array)

    Returns:
        QE as fraction (0-1)
    """
    wl = qe_data['wavelength_nm']
    qe = qe_data['QE_fraction']

    # Interpolate, returning 0 outside the data range
    return float(np.interp(wavelength_nm, wl, qe, left=0.0, right=0.0))


# =============================================================================
# RESPONSIVITY
# =============================================================================

def responsivity(wavelength_nm: float, QE: float) -> float:
    """
    Spectral responsivity from quantum efficiency.

    R(λ) = q·λ·QE / (h·c) = QE · λ[nm] / 1240   [A/W]

    Args:
        wavelength_nm: Wavelength in nm
        QE: Quantum efficiency as fraction (0-1)

    Returns:
        R in A/W

    Source: Sze Eq. 13.1
    """
    return QE * wavelength_nm / 1240.0


def responsivity_at_wavelength(material_name: str, wavelength_nm: float,
                               absorption_depth_um: float = 2.0,
                               surface_reflectance: float = 0.05,
                               collection_efficiency: float = 0.95,
                               T: float = 300.0) -> float:
    """
    Full responsivity calculation from material properties.

    Combines QE derivation + responsivity formula.

    Returns:
        R(λ) in A/W
    """
    qe = quantum_efficiency_analytical(
        material_name, wavelength_nm,
        absorption_depth_um, surface_reflectance, collection_efficiency, T
    )
    return responsivity(wavelength_nm, qe)


def responsivity_from_qe_curve(wavelength_nm: float,
                               qe_data: dict) -> float:
    """
    Responsivity using measured QE curve data (hybrid approach).

    Args:
        wavelength_nm: Wavelength in nm
        qe_data: QE curve dict

    Returns:
        R in A/W
    """
    qe = quantum_efficiency_from_curve(wavelength_nm, qe_data)
    return responsivity(wavelength_nm, qe)


# =============================================================================
# SPECTRAL CURVES
# =============================================================================

def spectral_responsivity_curve(material_name: str,
                                wavelengths_nm: np.ndarray = None,
                                absorption_depth_um: float = 2.0,
                                surface_reflectance: float = 0.05,
                                collection_efficiency: float = 0.95,
                                T: float = 300.0) -> dict:
    """
    Generate full spectral responsivity curve R(λ).

    Args:
        material_name: Material key
        wavelengths_nm: Array of wavelengths (default: 350-1100nm)
        absorption_depth_um: Collection depth in µm
        surface_reflectance: Surface reflectance
        collection_efficiency: Collection efficiency
        T: Temperature in K

    Returns:
        Dict with 'wavelength_nm', 'QE', 'responsivity_A_per_W' arrays
    """
    if wavelengths_nm is None:
        wavelengths_nm = np.arange(350, 1101, 10, dtype=float)

    qe_arr = np.array([
        quantum_efficiency_analytical(
            material_name, wl, absorption_depth_um,
            surface_reflectance, collection_efficiency, T
        ) for wl in wavelengths_nm
    ])

    r_arr = qe_arr * wavelengths_nm / 1240.0

    return {
        'wavelength_nm': wavelengths_nm,
        'QE': qe_arr,
        'responsivity_A_per_W': r_arr,
    }


# =============================================================================
# PHOTOCURRENT
# =============================================================================

def photocurrent(R_A_per_W: float, P_optical_W: float) -> float:
    """
    Photocurrent from responsivity and incident optical power.

    I_ph = R · P_optical

    Args:
        R_A_per_W: Responsivity in A/W
        P_optical_W: Incident optical power in Watts

    Returns:
        I_ph in Amperes
    """
    return R_A_per_W * P_optical_W


def effective_responsivity(led_spectrum: dict, pv_qe_data: dict) -> float:
    """
    Effective responsivity for a specific LED + PV combination.

    Integrates the LED emission spectrum weighted by the PV responsivity:
    R_eff = ∫ R(λ) · S_LED(λ) dλ / ∫ S_LED(λ) dλ

    Args:
        led_spectrum: Dict with 'wavelength_nm' and 'power_normalized' arrays
        pv_qe_data: QE curve dict for the PV receiver

    Returns:
        R_eff in A/W
    """
    wl = led_spectrum['wavelength_nm']
    s_led = led_spectrum['power_normalized']

    # Responsivity at each LED wavelength
    r_at_wl = np.array([responsivity_from_qe_curve(w, pv_qe_data) for w in wl])

    # Weighted average
    numerator = np.trapz(r_at_wl * s_led, wl)
    denominator = np.trapz(s_led, wl)

    if denominator == 0:
        return 0.0

    return numerator / denominator


# =============================================================================
# SELF-TEST
# =============================================================================

def test_photodetection():
    """Verify photodetection physics against KXOB25 and BPW34 targets."""
    print("=" * 70)
    print("PHOTODETECTION PHYSICS — TEST SUITE")
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

    # --- Basic responsivity formula ---
    print("\n1. RESPONSIVITY FORMULA  R = QE·λ/1240")
    R_ideal = responsivity(650, 1.0)
    check("R(650nm, QE=1.0)", R_ideal, 0.5242, tol_pct=1, unit="A/W")

    R_85 = responsivity(650, 0.85)
    check("R(650nm, QE=0.85)", R_85, 0.4456, tol_pct=1, unit="A/W")
    # Target: KXOB25 datasheet = 0.457 A/W

    # --- QE from analytical model (GaAs) ---
    print("\n2. QUANTUM EFFICIENCY — GaAs Analytical Model")
    # GaAs collection depth ~2µm: 0.5µm junction + 1µm depletion + 0.5µm diffusion
    qe_650 = quantum_efficiency_analytical('GaAs', 650, absorption_depth_um=2.0,
                                            surface_reflectance=0.05)
    check("QE(GaAs, 650nm, d=2µm)", qe_650, 0.85, tol_pct=15)

    qe_400 = quantum_efficiency_analytical('GaAs', 400, absorption_depth_um=2.0)
    print(f"  QE(GaAs, 400nm) = {qe_400:.3f} (high α → nearly full absorption)")

    qe_850 = quantum_efficiency_analytical('GaAs', 850, absorption_depth_um=2.0)
    print(f"  QE(GaAs, 850nm) = {qe_850:.3f} (low α near bandgap edge)")

    qe_900 = quantum_efficiency_analytical('GaAs', 900, absorption_depth_um=2.0)
    print(f"  QE(GaAs, 900nm) = {qe_900:.3f} (should be ~0 — below bandgap) ", end="")
    assert qe_900 < 0.05, "GaAs should not detect below bandgap"
    print("[PASS]")
    passes += 1

    # --- QE from measured curve (hybrid) ---
    print("\n3. QUANTUM EFFICIENCY — From Measured Curve (Hybrid)")
    qe_curve_650 = quantum_efficiency_from_curve(650, GAAS_QE_CURVE)
    check("QE_curve(GaAs, 650nm)", qe_curve_650, 0.85, tol_pct=5)

    # --- Full responsivity: GaAs at 650nm ---
    print("\n4. FULL RESPONSIVITY — GaAs (KXOB25 target)")
    R_gaas = responsivity_at_wavelength('GaAs', 650,
                                         absorption_depth_um=2.0,
                                         surface_reflectance=0.05)
    check("R(GaAs, 650nm)", R_gaas, 0.457, tol_pct=12, unit="A/W")

    R_gaas_curve = responsivity_from_qe_curve(650, GAAS_QE_CURVE)
    check("R_curve(GaAs, 650nm)", R_gaas_curve, 0.457, tol_pct=10, unit="A/W")

    # --- Si responsivity (BPW34) ---
    print("\n5. FULL RESPONSIVITY — Si (BPW34)")
    # Si: long diffusion lengths (~100µm) allow deep collection
    R_si_650 = responsivity_at_wavelength('Si', 650,
                                           absorption_depth_um=100.0,
                                           surface_reflectance=0.30,
                                           collection_efficiency=0.90)
    check("R(Si, 650nm)", R_si_650, 0.35, tol_pct=30, unit="A/W")

    R_si_900 = responsivity_at_wavelength('Si', 900,
                                           absorption_depth_um=100.0,
                                           surface_reflectance=0.30,
                                           collection_efficiency=0.90)
    print(f"  R(Si, 900nm) = {R_si_900:.3f} A/W (peak region for BPW34)")

    # Si should respond at 900nm but GaAs should not
    R_gaas_900 = responsivity_at_wavelength('GaAs', 900)
    print(f"  R(GaAs,900nm)={R_gaas_900:.4f}, R(Si,900nm)={R_si_900:.3f} ", end="")
    assert R_si_900 > R_gaas_900, "Si should respond better at 900nm"
    print("[PASS]")
    passes += 1

    # --- Spectral curve ---
    print("\n6. SPECTRAL CURVE GENERATION")
    curve = spectral_responsivity_curve('GaAs', absorption_depth_um=2.0)
    peak_R = np.max(curve['responsivity_A_per_W'])
    peak_wl = curve['wavelength_nm'][np.argmax(curve['responsivity_A_per_W'])]
    print(f"  GaAs peak R = {peak_R:.3f} A/W at {peak_wl:.0f} nm ", end="")
    assert 600 < peak_wl < 870, "GaAs peak should be in visible-NIR"
    print("[PASS]")
    passes += 1

    # --- Photocurrent ---
    print("\n7. PHOTOCURRENT")
    P_opt = 9.3e-3  # 9.3 mW (Kadirvelu paper LED power)
    I_ph = photocurrent(0.457, P_opt)
    I_ph_mA = I_ph * 1e3
    check("I_ph(0.457 A/W, 9.3mW)", I_ph_mA, 4.25, tol_pct=1, unit="mA")

    # --- Summary ---
    print(f"\n{'=' * 70}")
    print(f"RESULTS: {passes} passed, {fails} failed")
    print(f"{'=' * 70}")
    return fails == 0


if __name__ == '__main__':
    test_photodetection()
