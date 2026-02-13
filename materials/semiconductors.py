# materials/semiconductors.py
"""
Semiconductor Materials Database for Hardware-Faithful LiFi-PV Simulator

This module provides temperature-dependent semiconductor properties using
validated physics models (Varshni equation, empirical mobility models, etc.)

Supported Materials:
    - Silicon (Si) - Standard photodiodes, solar cells
    - Gallium Arsenide (GaAs) - High-efficiency PV (KXOB25)
    - Gallium Nitride (GaN) - Blue/UV LEDs
    - Indium Gallium Nitride (InGaN) - White/Blue LEDs
    - Aluminum Indium Gallium Phosphide (AlInGaP) - Red/Orange LEDs

References:
    - Vurgaftman et al., J. Appl. Phys. 89, 5815 (2001) - III-V parameters
    - Sze & Ng, "Physics of Semiconductor Devices" 3rd ed.
    - Piprek, "Semiconductor Optoelectronic Devices"
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Callable
import sys
import os

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.constants import K_B, Q, EPSILON_0, HC_EV_NM


# =============================================================================
# MATERIAL PROPERTY DATA CLASS
# =============================================================================

@dataclass
class SemiconductorMaterial:
    """
    Complete semiconductor material specification.
    
    All properties can be temperature-dependent via callable functions,
    or fixed values for simplified models.
    """
    
    # Identity
    name: str
    formula: str
    crystal_structure: str = "zinc_blende"
    
    # Bandgap (Varshni parameters)
    Eg_0: float = 1.12          # Bandgap at 0K (eV)
    Eg_alpha: float = 4.73e-4   # Varshni α (eV/K)
    Eg_beta: float = 636.0      # Varshni β (K)
    bandgap_type: str = "indirect"  # "direct" or "indirect"
    
    # Effective masses (relative to m_e)
    m_e_eff: float = 1.0        # Electron effective mass
    m_h_eff: float = 1.0        # Hole effective mass
    
    # Dielectric constant
    epsilon_r: float = 11.7     # Relative permittivity
    epsilon_inf: float = 11.7   # High-frequency permittivity
    
    # Mobility at 300K (cm²/Vs)
    mu_e_300: float = 1400.0    # Electron mobility
    mu_h_300: float = 450.0     # Hole mobility
    
    # Mobility temperature exponent: μ(T) = μ_300 × (300/T)^α
    mu_e_alpha: float = 2.4     # Electron mobility exponent
    mu_h_alpha: float = 2.2     # Hole mobility exponent
    
    # Intrinsic carrier concentration at 300K (cm⁻³)
    ni_300: float = 1.0e10
    
    # Density of states parameters (cm⁻³)
    Nc_300: float = 2.8e19      # Conduction band DOS at 300K
    Nv_300: float = 1.04e19     # Valence band DOS at 300K
    
    # Absorption coefficient parameters
    # α(E) = A × sqrt(E - Eg) for direct bandgap
    # α(E) = A × (E - Eg)² for indirect bandgap
    absorption_A: float = 1e4   # Absorption coefficient prefactor (cm⁻¹)
    
    # Recombination parameters
    tau_n: float = 1e-6         # Electron lifetime (s)
    tau_p: float = 1e-6         # Hole lifetime (s)
    B_rad: float = 1e-10        # Radiative recombination coeff (cm³/s)
    C_auger: float = 1e-30      # Auger recombination coeff (cm⁶/s)
    
    # Thermal properties
    thermal_conductivity: float = 1.5  # W/(cm·K)
    
    # LED-specific (for emitters)
    peak_wavelength_nm: Optional[float] = None
    spectral_width_nm: Optional[float] = None
    internal_quantum_efficiency: float = 0.5
    
    # =========================================================================
    # TEMPERATURE-DEPENDENT PROPERTY METHODS
    # =========================================================================
    
    def bandgap(self, T: float = 300.0) -> float:
        """
        Calculate bandgap at temperature T using Varshni equation.
        
        Eg(T) = Eg(0) - α×T² / (T + β)
        
        Args:
            T: Temperature in Kelvin
            
        Returns:
            Bandgap energy in eV
        """
        return self.Eg_0 - (self.Eg_alpha * T**2) / (T + self.Eg_beta)
    
    def peak_wavelength(self, T: float = 300.0) -> float:
        """
        Calculate peak emission/absorption wavelength from bandgap.
        
        λ_peak = hc / Eg
        
        Args:
            T: Temperature in Kelvin
            
        Returns:
            Wavelength in nm
        """
        if self.peak_wavelength_nm is not None:
            # Use specified peak (for LEDs with known spectrum)
            return self.peak_wavelength_nm
        Eg = self.bandgap(T)
        return HC_EV_NM / Eg
    
    def electron_mobility(self, T: float = 300.0) -> float:
        """
        Calculate electron mobility at temperature T.
        
        μ_e(T) = μ_e(300K) × (300/T)^α
        
        Args:
            T: Temperature in Kelvin
            
        Returns:
            Mobility in cm²/(V·s)
        """
        return self.mu_e_300 * (300.0 / T) ** self.mu_e_alpha
    
    def hole_mobility(self, T: float = 300.0) -> float:
        """
        Calculate hole mobility at temperature T.
        
        μ_h(T) = μ_h(300K) × (300/T)^α
        
        Args:
            T: Temperature in Kelvin
            
        Returns:
            Mobility in cm²/(V·s)
        """
        return self.mu_h_300 * (300.0 / T) ** self.mu_h_alpha
    
    def intrinsic_concentration(self, T: float = 300.0) -> float:
        """
        Calculate intrinsic carrier concentration.
        
        n_i(T) = √(Nc × Nv) × exp(-Eg/(2kT))
        
        Args:
            T: Temperature in Kelvin
            
        Returns:
            Intrinsic carrier concentration in cm⁻³
        """
        Eg = self.bandgap(T)
        kT = K_B * T / Q  # Convert to eV
        
        # Temperature-dependent DOS
        Nc = self.Nc_300 * (T / 300.0) ** 1.5
        Nv = self.Nv_300 * (T / 300.0) ** 1.5
        
        return np.sqrt(Nc * Nv) * np.exp(-Eg / (2 * kT))
    
    def permittivity(self) -> float:
        """Return absolute permittivity in F/m."""
        return self.epsilon_r * EPSILON_0
    
    def absorption_coefficient(self, wavelength_nm: float, T: float = 300.0) -> float:
        """
        Calculate absorption coefficient at given wavelength.
        
        For direct bandgap:  α = A × √(E - Eg)
        For indirect bandgap: α = A × (E - Eg)²
        
        Args:
            wavelength_nm: Wavelength in nm
            T: Temperature in Kelvin
            
        Returns:
            Absorption coefficient in cm⁻¹
        """
        E_photon = HC_EV_NM / wavelength_nm  # eV
        Eg = self.bandgap(T)
        
        if E_photon < Eg:
            return 0.0  # Below bandgap, no absorption
        
        delta_E = E_photon - Eg
        
        if self.bandgap_type == "direct":
            return self.absorption_A * np.sqrt(delta_E)
        else:  # indirect
            return self.absorption_A * delta_E**2
    
    def diffusion_length(self, carrier: str = 'electron', T: float = 300.0) -> float:
        """
        Calculate carrier diffusion length.
        
        L = √(D × τ) where D = μ × kT/q
        
        Args:
            carrier: 'electron' or 'hole'
            T: Temperature in Kelvin
            
        Returns:
            Diffusion length in cm
        """
        kT_q = K_B * T / Q  # Thermal voltage in V
        
        if carrier == 'electron':
            mu = self.electron_mobility(T)
            tau = self.tau_n
        else:
            mu = self.hole_mobility(T)
            tau = self.tau_p
        
        D = mu * kT_q  # Diffusion coefficient in cm²/s
        return np.sqrt(D * tau)
    
    def depletion_width(self, V_bi: float, V_applied: float, 
                        N_a: float, N_d: float) -> float:
        """
        Calculate depletion region width for p-n junction.
        
        W = √(2ε(V_bi - V) / q × (1/N_a + 1/N_d))
        
        Args:
            V_bi: Built-in voltage (V)
            V_applied: Applied voltage (V), negative for reverse bias
            N_a: Acceptor concentration (cm⁻³)
            N_d: Donor concentration (cm⁻³)
            
        Returns:
            Depletion width in cm
        """
        # Use CGS units for simplicity
        # ε in F/cm = ε_r × ε_0 × 100 (since ε_0 is in F/m)
        eps_cgs = self.epsilon_r * 8.854e-14  # F/cm
        
        V_total = V_bi - V_applied
        
        if V_total < 0:
            V_total = 0.01  # Forward bias, minimal depletion
        
        # W in cm, N_a and N_d in cm⁻³
        W = np.sqrt(2 * eps_cgs * V_total / Q * (1/N_a + 1/N_d))
        return W
    
    def built_in_voltage(self, N_a: float, N_d: float, T: float = 300.0) -> float:
        """
        Calculate built-in voltage for p-n junction.
        
        V_bi = (kT/q) × ln(N_a × N_d / n_i²)
        
        Args:
            N_a: Acceptor concentration (cm⁻³)
            N_d: Donor concentration (cm⁻³)
            T: Temperature in Kelvin
            
        Returns:
            Built-in voltage in V
        """
        ni = self.intrinsic_concentration(T)
        V_T = K_B * T / Q
        return V_T * np.log(N_a * N_d / ni**2)
    
    def __repr__(self):
        return f"SemiconductorMaterial('{self.name}', Eg={self.bandgap(300):.3f}eV @ 300K)"


# =============================================================================
# MATERIAL LIBRARY
# =============================================================================

# --- Silicon ---
SILICON = SemiconductorMaterial(
    name="Silicon",
    formula="Si",
    crystal_structure="diamond",
    
    # Bandgap (indirect)
    Eg_0=1.166,
    Eg_alpha=4.73e-4,
    Eg_beta=636.0,
    bandgap_type="indirect",
    
    # Effective masses
    m_e_eff=0.26,   # Conductivity effective mass
    m_h_eff=0.39,
    
    # Dielectric
    epsilon_r=11.7,
    epsilon_inf=11.7,
    
    # Mobility
    mu_e_300=1400.0,
    mu_h_300=450.0,
    mu_e_alpha=2.4,
    mu_h_alpha=2.2,
    
    # Intrinsic concentration
    ni_300=1.0e10,
    Nc_300=2.8e19,
    Nv_300=1.04e19,
    
    # Absorption (indirect, weaker)
    absorption_A=1e3,
    
    # Lifetimes (typical for device-grade Si)
    tau_n=10e-6,
    tau_p=10e-6,
    B_rad=4.73e-15,
    C_auger=3.8e-31,
)

# --- Gallium Arsenide ---
GAAS = SemiconductorMaterial(
    name="Gallium Arsenide",
    formula="GaAs",
    crystal_structure="zinc_blende",
    
    # Bandgap (direct)
    Eg_0=1.519,
    Eg_alpha=5.405e-4,
    Eg_beta=204.0,
    bandgap_type="direct",
    
    # Effective masses
    m_e_eff=0.067,
    m_h_eff=0.45,
    
    # Dielectric
    epsilon_r=12.9,
    epsilon_inf=10.89,
    
    # Mobility (very high electron mobility)
    mu_e_300=8500.0,
    mu_h_300=400.0,
    mu_e_alpha=2.3,
    mu_h_alpha=2.1,
    
    # Intrinsic concentration (much lower than Si due to larger bandgap)
    ni_300=2.1e6,
    Nc_300=4.7e17,
    Nv_300=7.0e18,
    
    # Absorption (direct, strong)
    absorption_A=1e4,
    
    # Lifetimes
    tau_n=1e-8,
    tau_p=1e-8,
    B_rad=7.2e-10,
    C_auger=1e-30,
)

# --- Gallium Nitride ---
GAN = SemiconductorMaterial(
    name="Gallium Nitride",
    formula="GaN",
    crystal_structure="wurtzite",
    
    # Bandgap (direct, wide)
    Eg_0=3.507,
    Eg_alpha=9.09e-4,
    Eg_beta=830.0,
    bandgap_type="direct",
    
    # Effective masses
    m_e_eff=0.20,
    m_h_eff=1.0,
    
    # Dielectric
    epsilon_r=8.9,
    epsilon_inf=5.35,
    
    # Mobility (lower due to wide bandgap)
    mu_e_300=1000.0,
    mu_h_300=30.0,
    mu_e_alpha=1.5,
    mu_h_alpha=1.5,
    
    # Intrinsic concentration (extremely low)
    ni_300=1.9e-10,
    Nc_300=2.3e18,
    Nv_300=4.6e19,
    
    # Absorption
    absorption_A=1e5,
    
    # LED properties (UV)
    peak_wavelength_nm=365,
    spectral_width_nm=15,
    internal_quantum_efficiency=0.3,
)

# --- Indium Gallium Nitride (In_0.2 Ga_0.8 N for blue) ---
INGAN_BLUE = SemiconductorMaterial(
    name="Indium Gallium Nitride (Blue)",
    formula="In0.2Ga0.8N",
    crystal_structure="wurtzite",
    
    # Bandgap (tunable by In content, ~2.7eV for blue)
    Eg_0=2.90,
    Eg_alpha=7.0e-4,
    Eg_beta=600.0,
    bandgap_type="direct",
    
    # Effective masses (interpolated)
    m_e_eff=0.18,
    m_h_eff=0.9,
    
    # Dielectric
    epsilon_r=9.5,
    epsilon_inf=5.8,
    
    # Mobility
    mu_e_300=300.0,
    mu_h_300=10.0,
    mu_e_alpha=1.5,
    mu_h_alpha=1.5,
    
    # Intrinsic concentration
    ni_300=1e-5,
    Nc_300=1.5e18,
    Nv_300=3.0e19,
    
    # Absorption
    absorption_A=1e5,
    
    # LED properties (Blue 450nm)
    peak_wavelength_nm=450,
    spectral_width_nm=20,
    internal_quantum_efficiency=0.7,
)

# --- Aluminum Indium Gallium Phosphide (Red LED) ---
ALINGAP_RED = SemiconductorMaterial(
    name="Aluminum Indium Gallium Phosphide (Red)",
    formula="AlInGaP",
    crystal_structure="zinc_blende",
    
    # Bandgap (~2.0 eV for red emission)
    Eg_0=2.10,
    Eg_alpha=4.5e-4,
    Eg_beta=200.0,
    bandgap_type="direct",
    
    # Effective masses
    m_e_eff=0.12,
    m_h_eff=0.60,
    
    # Dielectric
    epsilon_r=11.1,
    epsilon_inf=9.1,
    
    # Mobility
    mu_e_300=1000.0,
    mu_h_300=100.0,
    mu_e_alpha=2.0,
    mu_h_alpha=2.0,
    
    # Intrinsic concentration
    ni_300=1e2,
    Nc_300=8e17,
    Nv_300=1.5e19,
    
    # Absorption
    absorption_A=5e4,
    
    # LED properties (Red 625nm)
    peak_wavelength_nm=625,
    spectral_width_nm=20,
    internal_quantum_efficiency=0.5,
)


# =============================================================================
# MATERIAL LOOKUP INTERFACE
# =============================================================================

MATERIALS_DATABASE = {
    'Si': SILICON,
    'Silicon': SILICON,
    'GaAs': GAAS,
    'Gallium Arsenide': GAAS,
    'GaN': GAN,
    'Gallium Nitride': GAN,
    'InGaN': INGAN_BLUE,
    'InGaN_Blue': INGAN_BLUE,
    'AlInGaP': ALINGAP_RED,
    'AlInGaP_Red': ALINGAP_RED,
}


def get_material(name: str) -> SemiconductorMaterial:
    """
    Get semiconductor material by name.
    
    Args:
        name: Material name (e.g., 'Si', 'GaAs', 'InGaN')
        
    Returns:
        SemiconductorMaterial instance
        
    Raises:
        KeyError: If material not found
    """
    if name not in MATERIALS_DATABASE:
        available = list(MATERIALS_DATABASE.keys())
        raise KeyError(f"Material '{name}' not found. Available: {available}")
    return MATERIALS_DATABASE[name]


def list_materials() -> list:
    """Return list of available material names."""
    return list(set(m.name for m in MATERIALS_DATABASE.values()))


# =============================================================================
# InGaN COMPOSITION INTERPOLATION
# =============================================================================

def InGaN(x_In: float) -> SemiconductorMaterial:
    """
    Create InGaN material with specified Indium composition.
    
    Uses Vegard's law with bowing parameter for bandgap interpolation:
    Eg(x) = x·Eg(InN) + (1-x)·Eg(GaN) - b·x·(1-x)
    
    Args:
        x_In: Indium mole fraction (0 = GaN, 1 = InN)
        
    Returns:
        SemiconductorMaterial with interpolated properties
    """
    # End-member bandgaps at 0K
    Eg_GaN = 3.507
    Eg_InN = 0.69
    b_InGaN = 1.4  # Bowing parameter (eV)
    
    # Interpolated bandgap
    Eg_0 = x_In * Eg_InN + (1 - x_In) * Eg_GaN - b_InGaN * x_In * (1 - x_In)
    
    # Peak wavelength from bandgap at 300K
    Eg_300 = Eg_0 - (7e-4 * 300**2) / (300 + 600)  # Approximate Varshni
    peak_nm = HC_EV_NM / Eg_300
    
    return SemiconductorMaterial(
        name=f"In{x_In:.2f}Ga{1-x_In:.2f}N",
        formula=f"In{x_In:.2f}Ga{1-x_In:.2f}N",
        crystal_structure="wurtzite",
        
        Eg_0=Eg_0,
        Eg_alpha=7e-4,
        Eg_beta=600.0,
        bandgap_type="direct",
        
        m_e_eff=0.07 * x_In + 0.20 * (1 - x_In),
        m_h_eff=0.6 * x_In + 1.0 * (1 - x_In),
        
        epsilon_r=15.3 * x_In + 8.9 * (1 - x_In),
        
        mu_e_300=1000 * (1 - x_In),  # Decreases with In
        mu_h_300=30,
        
        ni_300=1e-5,  # Still very low
        Nc_300=1.5e18,
        Nv_300=3.0e19,
        
        absorption_A=1e5,
        
        peak_wavelength_nm=peak_nm,
        spectral_width_nm=20,
        internal_quantum_efficiency=0.6 if x_In < 0.3 else 0.3,  # Green gap
    )


# =============================================================================
# SELF-TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("SEMICONDUCTOR MATERIALS DATABASE - SELF TEST")
    print("=" * 70)
    
    for name in ['Si', 'GaAs', 'GaN', 'InGaN_Blue', 'AlInGaP_Red']:
        mat = get_material(name)
        print(f"\n{mat.name} ({mat.formula})")
        print("-" * 50)
        print(f"  Bandgap @ 300K:  {mat.bandgap(300):.3f} eV")
        print(f"  Peak wavelength: {mat.peak_wavelength(300):.0f} nm")
        print(f"  n_i @ 300K:      {mat.intrinsic_concentration(300):.2e} cm⁻³")
        print(f"  μ_e @ 300K:      {mat.electron_mobility(300):.0f} cm²/(V·s)")
        print(f"  μ_h @ 300K:      {mat.hole_mobility(300):.0f} cm²/(V·s)")
        print(f"  ε_r:             {mat.epsilon_r}")
        
        if mat.peak_wavelength_nm:
            print(f"  LED peak:        {mat.peak_wavelength_nm} nm")
    
    # Test temperature dependence
    print("\n" + "=" * 70)
    print("TEMPERATURE DEPENDENCE - GaAs")
    print("=" * 70)
    gaas = get_material('GaAs')
    for T in [250, 300, 350, 400]:
        print(f"  T={T}K: Eg={gaas.bandgap(T):.4f}eV, "
              f"n_i={gaas.intrinsic_concentration(T):.2e}cm⁻³, "
              f"μ_e={gaas.electron_mobility(T):.0f}cm²/Vs")
    
    # Test InGaN composition
    print("\n" + "=" * 70)
    print("InGaN COMPOSITION TUNING")
    print("=" * 70)
    for x in [0.0, 0.15, 0.25, 0.35, 0.50]:
        mat = InGaN(x)
        print(f"  x={x:.2f}: Eg={mat.bandgap(300):.3f}eV, λ={mat.peak_wavelength(300):.0f}nm")
    
    print("\n[OK] All materials tests passed!")
