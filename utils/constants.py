# utils/constants.py
"""
Physical Constants for Hardware-Faithful LiFi-PV Simulator

This module provides fundamental physical constants used throughout the simulator.
All values are in SI units unless otherwise noted.

References:
    - CODATA 2018 recommended values
    - Semiconductor physics textbooks (Sze, Neamen)
"""

import numpy as np

# =============================================================================
# FUNDAMENTAL PHYSICAL CONSTANTS
# =============================================================================

# Electron charge (C)
Q = 1.602176634e-19
Q_ELECTRON = Q  # Alias for compatibility

# Boltzmann constant (J/K)
K_B = 1.380649e-23
K_BOLTZMANN = K_B  # Alias

# Planck constant (J·s)
H = 6.62607015e-34
H_PLANCK = H  # Alias

# Reduced Planck constant (J·s)
HBAR = H / (2 * np.pi)

# Speed of light in vacuum (m/s)
C = 299792458.0
C_LIGHT = C  # Alias

# Permittivity of free space (F/m)
EPSILON_0 = 8.8541878128e-12

# Electron mass (kg)
M_E = 9.1093837015e-31
M_ELECTRON = M_E  # Alias

# Thermal voltage at 300K (V)
V_T_300K = K_B * 300 / Q  # ≈ 25.85 mV

# =============================================================================
# DERIVED CONSTANTS
# =============================================================================

# h·c product for wavelength-energy conversion (eV·nm)
HC_EV_NM = (H * C) / Q * 1e9  # ≈ 1239.84 eV·nm

# Responsivity conversion factor: R(λ) = QE × λ[nm] / 1240
RESPONSIVITY_FACTOR = Q / (H * C) * 1e-9  # A/W per nm per QE

# =============================================================================
# ROOM CONDITIONS
# =============================================================================

ROOM_TEMPERATURE_K = 300.0
STANDARD_PRESSURE_PA = 101325.0

# =============================================================================
# UNIT CONVERSION HELPERS
# =============================================================================

def eV_to_J(eV):
    """Convert electron-volts to Joules."""
    return eV * Q

def J_to_eV(J):
    """Convert Joules to electron-volts."""
    return J / Q

def nm_to_eV(wavelength_nm):
    """Convert wavelength (nm) to photon energy (eV)."""
    return HC_EV_NM / wavelength_nm

def eV_to_nm(energy_eV):
    """Convert photon energy (eV) to wavelength (nm)."""
    return HC_EV_NM / energy_eV

def thermal_voltage(temperature_K):
    """Calculate thermal voltage V_T = kT/q at given temperature."""
    return K_B * temperature_K / Q

# =============================================================================
# WAVELENGTH-RESPONSIVITY RELATION
# =============================================================================

def ideal_responsivity(wavelength_nm, quantum_efficiency=1.0):
    """
    Calculate ideal responsivity for a photodetector.
    
    R(λ) = (q·λ·QE) / (h·c) = QE × λ[nm] / 1240  [A/W]
    
    Args:
        wavelength_nm: Wavelength in nanometers
        quantum_efficiency: External quantum efficiency (0-1)
    
    Returns:
        Responsivity in A/W
    """
    return quantum_efficiency * wavelength_nm / HC_EV_NM


# =============================================================================
# SELF-TEST
# =============================================================================

if __name__ == "__main__":
    print("Physical Constants Module")
    print("=" * 50)
    print(f"Electron charge q:     {Q:.6e} C")
    print(f"Boltzmann constant k:  {K_B:.6e} J/K")
    print(f"Planck constant h:     {H:.6e} J·s")
    print(f"Speed of light c:      {C:.0f} m/s")
    print(f"Thermal voltage @300K: {V_T_300K*1e3:.2f} mV")
    print(f"h·c factor:            {HC_EV_NM:.2f} eV·nm")
    print()
    print("Conversions:")
    print(f"  1.42 eV → {eV_to_nm(1.42):.1f} nm (GaAs bandgap)")
    print(f"  850 nm  → {nm_to_eV(850):.3f} eV")
    print(f"  Ideal R @ 850nm, QE=1: {ideal_responsivity(850, 1.0):.3f} A/W")
    print(f"  Ideal R @ 850nm, QE=0.9: {ideal_responsivity(850, 0.9):.3f} A/W")
