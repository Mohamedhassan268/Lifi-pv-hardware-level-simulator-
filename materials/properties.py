# =============================================================================
# materials/properties.py — Temperature-Dependent Semiconductor Property Functions
# =============================================================================
# Task 2 of Hardware-Faithful Simulator
#
# These functions compute physical properties as a function of temperature,
# wavelength, etc. using the reference data from materials/reference_data.py.
# Every equation is traceable to a standard semiconductor physics textbook.
#
# References:
#   - Sze & Ng, "Physics of Semiconductor Devices", 3rd Ed.
#   - Pierret, "Semiconductor Fundamentals"
#   - Ioffe Institute NSM Archive
# =============================================================================

import numpy as np
from .reference_data import MATERIALS, CONSTANTS

# Unpack constants for convenience
q    = CONSTANTS['q']
k_B  = CONSTANTS['k_B']
h    = CONSTANTS['h']
c    = CONSTANTS['c']
eps_0 = CONSTANTS['eps_0']
m_0  = CONSTANTS['m_0']


# =============================================================================
# BANDGAP
# =============================================================================

def bandgap(material_name: str, T: float = 300.0) -> float:
    """
    Temperature-dependent bandgap energy using the Varshni equation.

    E_g(T) = E_g(0) - α·T² / (T + β)

    Args:
        material_name: Key into MATERIALS dict (e.g. 'GaAs', 'Si')
        T: Temperature in Kelvin

    Returns:
        E_g in eV

    Source: Varshni (1967), parameters from Ioffe Institute
    """
    mat = MATERIALS[material_name]
    E_g0 = mat['E_g_0K_eV']
    alpha = mat['varshni_alpha_eV_per_K']
    beta = mat['varshni_beta_K']

    return E_g0 - alpha * T**2 / (T + beta)


def bandgap_wavelength(material_name: str, T: float = 300.0) -> float:
    """
    Bandgap cutoff wavelength: λ_g = hc / E_g

    Args:
        material_name: Material key
        T: Temperature in Kelvin

    Returns:
        Cutoff wavelength in nm
    """
    E_g_eV = bandgap(material_name, T)
    # λ(nm) = 1240 / E_g(eV)
    return 1240.0 / E_g_eV


# =============================================================================
# INTRINSIC CARRIER CONCENTRATION
# =============================================================================

def effective_dos_conduction(material_name: str, T: float = 300.0) -> float:
    """
    Effective density of states in the conduction band.

    N_c(T) = N_c_coeff · T^(3/2)    [cm⁻³]

    For materials without N_c_coeff, scales from 300K value:
    N_c(T) = N_c(300) · (T/300)^(3/2)

    Source: Ioffe Institute
    """
    mat = MATERIALS[material_name]
    if 'N_c_coeff' in mat:
        return mat['N_c_coeff'] * T**1.5
    else:
        return mat['N_c_300K_cm3'] * (T / 300.0)**1.5


def effective_dos_valence(material_name: str, T: float = 300.0) -> float:
    """
    Effective density of states in the valence band.

    N_v(T) = N_v_coeff · T^(3/2)    [cm⁻³]

    Source: Ioffe Institute
    """
    mat = MATERIALS[material_name]
    if 'N_v_coeff' in mat:
        return mat['N_v_coeff'] * T**1.5
    else:
        return mat['N_v_300K_cm3'] * (T / 300.0)**1.5


def intrinsic_carrier_concentration(material_name: str, T: float = 300.0) -> float:
    """
    Intrinsic carrier concentration.

    n_i(T) = sqrt(N_c · N_v) · exp(-E_g / (2·k_B·T))

    Args:
        material_name: Material key
        T: Temperature in Kelvin

    Returns:
        n_i in cm⁻³

    Source: Sze Eq. 1.14
    """
    N_c = effective_dos_conduction(material_name, T)
    N_v = effective_dos_valence(material_name, T)
    E_g = bandgap(material_name, T)

    # k_B·T in eV
    kT_eV = k_B * T / q

    return np.sqrt(N_c * N_v) * np.exp(-E_g / (2 * kT_eV))


# =============================================================================
# CARRIER MOBILITY
# =============================================================================

def electron_mobility(material_name: str, T: float = 300.0,
                      N_d: float = 0.0) -> float:
    """
    Electron mobility with temperature and doping dependence.

    Temperature: μ_n(T) = μ_n(300) · (300/T)^γ
    Doping (GaAs only): μ_n = μ_0 / sqrt(1 + N_d/1e17)  [Hilsum 1974]

    Args:
        material_name: Material key
        T: Temperature in Kelvin
        N_d: Donor doping concentration in cm⁻³ (0 = undoped)

    Returns:
        μ_n in cm²/V·s

    Source: Ioffe Institute
    """
    mat = MATERIALS[material_name]
    mu_300 = mat['mu_n_300K_cm2_Vs']
    gamma = mat.get('mu_n_T_exponent', 1.5)

    mu = mu_300 * (300.0 / T) ** gamma

    # Doping-dependent degradation
    if N_d > 0 and material_name == 'GaAs':
        # Hilsum (1974): μ = μ_OH / sqrt(1 + Nd/1e17)
        mu = mu / np.sqrt(1 + N_d / 1e17)
    elif N_d > 0 and material_name == 'Si':
        # Caughey-Thomas model (simplified)
        # μ = μ_min + (μ_max - μ_min) / (1 + (N/N_ref)^alpha)
        mu_min = 65
        N_ref = 8.5e16
        alpha_ct = 0.72
        mu = mu_min + (mu - mu_min) / (1 + (N_d / N_ref)**alpha_ct)

    return mu


def hole_mobility(material_name: str, T: float = 300.0,
                  N_a: float = 0.0) -> float:
    """
    Hole mobility with temperature dependence.

    μ_p(T) = μ_p(300) · (300/T)^γ

    Args:
        material_name: Material key
        T: Temperature in Kelvin
        N_a: Acceptor doping in cm⁻³

    Returns:
        μ_p in cm²/V·s

    Source: Ioffe Institute
    """
    mat = MATERIALS[material_name]
    mu_300 = mat['mu_p_300K_cm2_Vs']
    gamma = mat.get('mu_p_T_exponent', 2.0)

    mu = mu_300 * (300.0 / T) ** gamma

    if N_a > 0 and material_name == 'Si':
        mu_min = 47
        N_ref = 2.23e17
        alpha_ct = 0.72
        mu = mu_min + (mu - mu_min) / (1 + (N_a / N_ref)**alpha_ct)

    return mu


def diffusion_coefficient(mobility: float, T: float = 300.0) -> float:
    """
    Diffusion coefficient from Einstein relation.

    D = μ · k_B · T / q

    Args:
        mobility: Carrier mobility in cm²/V·s
        T: Temperature in Kelvin

    Returns:
        D in cm²/s

    Source: Sze Eq. 1.49
    """
    return mobility * k_B * T / q


# =============================================================================
# ABSORPTION COEFFICIENT
# =============================================================================

def absorption_coefficient(material_name: str, wavelength_nm: float,
                           T: float = 300.0) -> float:
    """
    Absorption coefficient α(λ) from lookup table with interpolation.

    For direct bandgap (GaAs): above-bandgap absorption follows
        α ∝ sqrt(E - E_g) approximately
    For indirect bandgap (Si): weaker absorption near edge.

    Below the bandgap cutoff wavelength, α drops to ~0.

    Args:
        material_name: Material key
        wavelength_nm: Wavelength in nm
        T: Temperature in Kelvin (shifts bandgap edge)

    Returns:
        α in cm⁻¹

    Source: Lookup tables from Sze, Green & Keevers (Si), Casey (GaAs)
    """
    mat = MATERIALS[material_name]
    data = mat.get('absorption_data_nm_cm1', {})

    if not data:
        # Fallback: analytical model for direct bandgap
        return _absorption_analytical(material_name, wavelength_nm, T)

    # Convert dict to sorted arrays
    wl_arr = np.array(sorted(data.keys()), dtype=float)
    alpha_arr = np.array([data[wl] for wl in sorted(data.keys())], dtype=float)

    # Temperature shift: bandgap decreases with T → absorption edge shifts
    # Shift wavelengths by delta_lambda = lambda_g(T) - lambda_g(300)
    lambda_g_300 = bandgap_wavelength(material_name, 300.0)
    lambda_g_T = bandgap_wavelength(material_name, T)
    delta_lambda = lambda_g_T - lambda_g_300

    # Apply shift to the effective query wavelength (shift the query, not the table)
    effective_wl = wavelength_nm - delta_lambda

    # Interpolate in log space for smoother behavior
    # Clamp alpha to minimum of 1e-2 to avoid log(0)
    log_alpha = np.interp(effective_wl, wl_arr, np.log10(np.maximum(alpha_arr, 1e-2)),
                          left=np.log10(alpha_arr[0]),
                          right=-2)  # Below bandgap → ~0.01 cm⁻¹

    return 10**log_alpha


def _absorption_analytical(material_name: str, wavelength_nm: float,
                           T: float = 300.0) -> float:
    """
    Analytical absorption model for materials without lookup tables.

    Direct bandgap: α(E) = A · sqrt(E - E_g)  for E > E_g
    where A ~ 1e4 cm⁻¹ eV⁻¹/² (typical for III-V)

    Args:
        material_name: Material key
        wavelength_nm: Wavelength in nm
        T: Temperature in K

    Returns:
        α in cm⁻¹
    """
    E_g = bandgap(material_name, T)
    E_photon = 1240.0 / wavelength_nm  # eV

    if E_photon <= E_g:
        return 0.01  # Below bandgap

    mat = MATERIALS[material_name]
    if mat.get('bandgap_type') == 'direct':
        # Direct: α ∝ sqrt(E - E_g)
        A = 1.0e4   # Typical coefficient for III-V, cm⁻¹ eV⁻¹/²
        return A * np.sqrt(E_photon - E_g)
    else:
        # Indirect: much weaker absorption near edge
        A = 1.0e3
        return A * (E_photon - E_g)**2


# =============================================================================
# THERMAL VOLTAGE
# =============================================================================

def thermal_voltage(T: float = 300.0) -> float:
    """
    Thermal voltage V_T = k_B · T / q

    Args:
        T: Temperature in Kelvin

    Returns:
        V_T in Volts (~25.85 mV at 300K)
    """
    return k_B * T / q


# =============================================================================
# PERMITTIVITY
# =============================================================================

def permittivity(material_name: str) -> float:
    """
    Static permittivity ε_s = ε_r · ε_0

    Args:
        material_name: Material key

    Returns:
        ε_s in F/m
    """
    return MATERIALS[material_name]['eps_r'] * eps_0


def relative_permittivity(material_name: str) -> float:
    """Return relative permittivity ε_r (dimensionless)."""
    return MATERIALS[material_name]['eps_r']


# =============================================================================
# SELF-TEST
# =============================================================================

def test_properties():
    """Verify all property functions against known values."""
    print("=" * 70)
    print("MATERIAL PROPERTY FUNCTIONS — TEST SUITE")
    print("=" * 70)
    passes = 0
    fails = 0

    def check(name, val, expected, tol_pct=5.0):
        nonlocal passes, fails
        err = abs(val - expected) / abs(expected) * 100
        status = "PASS" if err < tol_pct else "FAIL"
        if status == "FAIL":
            fails += 1
        else:
            passes += 1
        print(f"  {name}: {val:.6g}  (expected {expected:.6g}, err={err:.1f}%) [{status}]")

    # --- Bandgap tests ---
    print("\n1. BANDGAP E_g(T)")
    check("GaAs E_g(300K)", bandgap('GaAs', 300), 1.424, tol_pct=1)
    check("GaAs E_g(400K)", bandgap('GaAs', 400), 1.38, tol_pct=2)
    check("GaAs E_g(0K)",   bandgap('GaAs', 0.01), 1.519, tol_pct=0.5)
    check("Si E_g(300K)",   bandgap('Si', 300), 1.12, tol_pct=1)
    check("Si E_g(400K)",   bandgap('Si', 400), 1.097, tol_pct=2)

    print("\n2. BANDGAP WAVELENGTH λ_g")
    check("GaAs λ_g(300K)", bandgap_wavelength('GaAs', 300), 871, tol_pct=1)
    check("Si λ_g(300K)",   bandgap_wavelength('Si', 300), 1107, tol_pct=1)

    # --- Intrinsic carrier concentration ---
    print("\n3. INTRINSIC CARRIER CONCENTRATION n_i(T)")
    check("GaAs n_i(300K)", intrinsic_carrier_concentration('GaAs', 300), 2.1e6, tol_pct=30)
    check("Si n_i(300K)",   intrinsic_carrier_concentration('Si', 300), 1.0e10, tol_pct=20)

    # n_i should increase strongly with temperature
    ni_gaas_400 = intrinsic_carrier_concentration('GaAs', 400)
    ni_gaas_300 = intrinsic_carrier_concentration('GaAs', 300)
    ratio = ni_gaas_400 / ni_gaas_300
    print(f"  GaAs n_i(400K)/n_i(300K) = {ratio:.0f}x (should be >> 1) ", end="")
    assert ratio > 10, "n_i temperature dependence too weak"
    print("[PASS]")
    passes += 1

    # --- Mobility ---
    print("\n4. CARRIER MOBILITY μ(T)")
    check("GaAs μ_n(300K, undoped)", electron_mobility('GaAs', 300), 8500, tol_pct=1)
    check("GaAs μ_p(300K, undoped)", hole_mobility('GaAs', 300), 400, tol_pct=1)
    check("Si μ_n(300K, undoped)",   electron_mobility('Si', 300), 1450, tol_pct=1)

    # Mobility should decrease with temperature
    mu_n_400 = electron_mobility('GaAs', 400)
    print(f"  GaAs μ_n(400K) = {mu_n_400:.0f} cm²/Vs (should be < 8500) ", end="")
    assert mu_n_400 < 8500, "Mobility should decrease with T"
    print("[PASS]")
    passes += 1

    # Doping-dependent mobility
    mu_doped = electron_mobility('GaAs', 300, N_d=1e17)
    print(f"  GaAs μ_n(300K, Nd=1e17) = {mu_doped:.0f} cm²/Vs (should be ~6000) ", end="")
    assert 4000 < mu_doped < 8000, "Doped mobility unexpected"
    print("[PASS]")
    passes += 1

    # --- Diffusion coefficient ---
    print("\n5. DIFFUSION COEFFICIENT D = μkT/q")
    D_n = diffusion_coefficient(electron_mobility('GaAs', 300), 300)
    check("GaAs D_n(300K)", D_n, 220, tol_pct=10)

    # --- Absorption coefficient ---
    print("\n6. ABSORPTION COEFFICIENT α(λ)")
    alpha_gaas_650 = absorption_coefficient('GaAs', 650, 300)
    print(f"  GaAs α(650nm) = {alpha_gaas_650:.0f} cm⁻¹ (expected ~10000) ", end="")
    assert 5e3 < alpha_gaas_650 < 2e4, "GaAs absorption at 650nm unexpected"
    print("[PASS]")
    passes += 1

    alpha_gaas_900 = absorption_coefficient('GaAs', 900, 300)
    print(f"  GaAs α(900nm) = {alpha_gaas_900:.2f} cm⁻¹ (expected ~1, below bandgap) ", end="")
    assert alpha_gaas_900 < 100, "GaAs should not absorb well below bandgap"
    print("[PASS]")
    passes += 1

    alpha_si_650 = absorption_coefficient('Si', 650, 300)
    print(f"  Si   α(650nm) = {alpha_si_650:.0f} cm⁻¹ (expected ~2900) ", end="")
    assert 1e3 < alpha_si_650 < 5e3, "Si absorption at 650nm unexpected"
    print("[PASS]")
    passes += 1

    # GaAs should absorb MORE than Si at 650nm (direct vs indirect)
    print(f"  GaAs α > Si α at 650nm? {alpha_gaas_650:.0f} > {alpha_si_650:.0f} ", end="")
    assert alpha_gaas_650 > alpha_si_650, "GaAs should absorb more (direct gap)"
    print("[PASS]")
    passes += 1

    # --- Thermal voltage ---
    print("\n7. THERMAL VOLTAGE V_T")
    check("V_T(300K)", thermal_voltage(300), 0.02585, tol_pct=0.5)

    # --- Permittivity ---
    print("\n8. PERMITTIVITY")
    eps_gaas = permittivity('GaAs')
    check("GaAs ε_s", eps_gaas, 12.9 * 8.854e-12, tol_pct=0.1)

    # --- Summary ---
    print(f"\n{'=' * 70}")
    print(f"RESULTS: {passes} passed, {fails} failed")
    print(f"{'=' * 70}")
    return fails == 0


if __name__ == '__main__':
    test_properties()
