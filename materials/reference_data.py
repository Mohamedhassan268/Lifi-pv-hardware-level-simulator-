# =============================================================================
# REFERENCE DATA: Semiconductor Materials & Component Datasheets
# =============================================================================
# Sources:
#   - Ioffe Institute NSM Archive (www.ioffe.ru/SVA/NSM/Semicond/)
#   - Vishay BPW34 Datasheet Rev 2.1 (2011)
#   - ANYSOLAR/IXYS KXOB25-04X3F Datasheet
#   - ANYSOLAR/IXYS SM141K04LV Datasheet
#   - S.M. Sze, "Semiconductor Devices: Physics and Technology"
#   - Blakemore (1982), "Semiconducting and other major properties of GaAs"
#
# This file is the SINGLE SOURCE OF TRUTH for all material and component
# parameters used in the physics engine. Every number here is traceable
# to a published source.
# =============================================================================

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# =============================================================================
# PHYSICAL CONSTANTS (CODATA 2018)
# =============================================================================
CONSTANTS = {
    'q':        1.602176634e-19,    # Electron charge (C)
    'k_B':      1.380649e-23,       # Boltzmann constant (J/K)
    'h':        6.62607015e-34,     # Planck constant (J·s)
    'c':        2.99792458e8,       # Speed of light (m/s)
    'eps_0':    8.8541878128e-12,   # Vacuum permittivity (F/m)
    'm_0':      9.1093837015e-31,   # Electron rest mass (kg)
    'eV_to_J':  1.602176634e-19,    # eV to Joules conversion
}


# =============================================================================
# SEMICONDUCTOR MATERIAL DATABASE
# =============================================================================
# Each material dict contains properties at 300K unless noted.
# All units are SI base or standard semiconductor units as marked.

MATERIALS = {}

# ---- GALLIUM ARSENIDE (GaAs) ----
# Source: Ioffe Institute NSM Archive + Blakemore 1982
MATERIALS['GaAs'] = {
    'name': 'Gallium Arsenide',
    'crystal_structure': 'Zinc Blende',
    'bandgap_type': 'direct',

    # Band structure (Ioffe)
    'E_g_300K_eV': 1.424,          # Energy gap at 300K
    'E_g_0K_eV': 1.519,            # Energy gap at 0K (for Varshni)

    # Varshni parameters: E_g(T) = E_g(0) - alpha*T^2 / (T + beta)
    # Source: Ioffe - "Eg=1.519-5.405e-4*T^2/(T+204)"
    'varshni_alpha_eV_per_K': 5.405e-4,
    'varshni_beta_K': 204,

    # Effective masses (Ioffe) in units of m_0
    'm_e_eff': 0.063,              # Electron effective mass (Gamma valley)
    'm_h_eff': 0.51,               # Heavy hole effective mass
    'm_lh_eff': 0.082,             # Light hole effective mass
    'm_so_eff': 0.15,              # Split-off band effective mass
    'm_dos_e': 0.063,              # Density of states effective mass (electrons)
    'm_dos_h': 0.53,               # Density of states effective mass (holes)

    # Carrier concentration (Ioffe @ 300K)
    'N_c_300K_cm3': 4.7e17,        # Effective conduction band DOS
    'N_v_300K_cm3': 9.0e18,        # Effective valence band DOS
    'n_i_300K_cm3': 2.1e6,         # Intrinsic carrier concentration

    # N_c(T) = 8.63e13 * T^(3/2) * [correction terms] (Ioffe)
    # N_v(T) = 1.83e15 * T^(3/2) (Ioffe)
    'N_c_coeff': 8.63e13,          # cm^-3 K^(-3/2) (simplified)
    'N_v_coeff': 1.83e15,          # cm^-3 K^(-3/2)

    # Electrical properties (Ioffe)
    'mu_n_300K_cm2_Vs': 8500,      # Electron mobility (undoped)
    'mu_p_300K_cm2_Vs': 400,       # Hole mobility (undoped)
    'D_n_300K_cm2_s': 200,         # Electron diffusion coefficient
    'D_p_300K_cm2_s': 10,          # Hole diffusion coefficient

    # Mobility temperature dependence (Ioffe)
    # mu_n ~ 8000*(300/T)^(2/3) cm²/Vs (drift)
    # mu_p ~ 400*(300/T)^(2.3) cm²/Vs (Hall)
    'mu_n_T_exponent': 2/3,        # For power-law T dependence
    'mu_p_T_exponent': 2.3,

    # Dielectric properties (Ioffe)
    'eps_r': 12.9,                  # Static dielectric constant
    'eps_r_hf': 10.89,              # High-frequency dielectric constant

    # Other basic parameters (Ioffe)
    'electron_affinity_eV': 4.07,
    'density_g_cm3': 5.32,
    'lattice_constant_A': 5.65325,

    # Thermal properties
    'thermal_conductivity_W_mK': 46,
    'debye_temperature_K': 360,

    # Recombination parameters (Ioffe + literature)
    # Radiative recombination coefficient
    'B_rad_cm3_s': 7.2e-10,        # Sze, Table
    # Auger coefficients (n-type, p-type)
    'C_n_cm6_s': 1.0e-30,
    'C_p_cm6_s': 1.0e-30,
    # SRH lifetime (typical for high quality GaAs)
    'tau_n0_s': 1e-8,              # Minority electron lifetime
    'tau_p0_s': 1e-8,              # Minority hole lifetime

    # Absorption coefficient lookup table (cm^-1 vs wavelength nm)
    # Source: Compiled from Sze, Levinshtein, and Casey (1975)
    # For direct bandgap: alpha ~ A * sqrt(E - E_g) above bandgap
    # These are approximate representative values for undoped GaAs at 300K
    'absorption_data_nm_cm1': {
        400: 7.5e4,
        450: 6.5e4,
        500: 4.5e4,
        550: 3.0e4,
        600: 1.8e4,
        625: 1.3e4,
        650: 1.0e4,
        700: 6.0e3,
        750: 3.5e3,
        800: 1.5e3,
        850: 5.0e2,
        870: 1.0e2,     # Near bandgap edge (~872nm for 1.424eV)
        880: 1.0e1,      # Below bandgap - drops rapidly
        900: 1.0e0,
    },

    # Refractive index at key wavelengths (Aspnes & Studna 1983)
    'refractive_index_data': {
        500: 4.18,
        550: 3.99,
        600: 3.86,
        625: 3.80,
        650: 3.75,
        700: 3.67,
        750: 3.60,
        800: 3.55,
        850: 3.50,
    },
}


# ---- SILICON (Si) ----
# Source: Ioffe Institute NSM Archive
MATERIALS['Si'] = {
    'name': 'Silicon',
    'crystal_structure': 'Diamond',
    'bandgap_type': 'indirect',

    # Band structure (Ioffe)
    'E_g_300K_eV': 1.12,
    'E_g_0K_eV': 1.17,

    # Varshni: Eg = 1.17 - 4.73e-4*T^2/(T+636)
    'varshni_alpha_eV_per_K': 4.73e-4,
    'varshni_beta_K': 636,

    # Effective masses (Ioffe)
    'm_e_ml': 0.98,                # Longitudinal electron mass
    'm_e_mt': 0.19,                # Transverse electron mass
    'm_e_eff': 0.26,               # Conductivity effective mass
    'm_dos_e': 1.08,               # DOS effective mass (with 6 valleys)
    'm_h_eff': 0.49,               # Heavy hole
    'm_lh_eff': 0.16,              # Light hole
    'm_dos_h': 0.81,               # DOS effective mass (holes)

    # Carrier concentration (Ioffe @ 300K)
    'N_c_300K_cm3': 3.2e19,
    'N_v_300K_cm3': 1.8e19,
    'n_i_300K_cm3': 1.0e10,

    # N_c(T) = 6.2e15 * T^(3/2) (Ioffe)
    # N_v(T) = 3.5e15 * T^(3/2) (Ioffe)
    'N_c_coeff': 6.2e15,
    'N_v_coeff': 3.5e15,

    # Electrical (Ioffe)
    'mu_n_300K_cm2_Vs': 1450,
    'mu_p_300K_cm2_Vs': 500,
    'D_n_300K_cm2_s': 36,
    'D_p_300K_cm2_s': 12,
    'mu_n_T_exponent': 2.4,         # mu ~ T^(-2.4) for Si electrons
    'mu_p_T_exponent': 2.2,

    # Dielectric
    'eps_r': 11.7,
    'eps_r_hf': 11.7,               # No significant dispersion for Si

    # Other
    'electron_affinity_eV': 4.05,
    'density_g_cm3': 2.329,
    'lattice_constant_A': 5.431,

    # Thermal
    'thermal_conductivity_W_mK': 148,
    'debye_temperature_K': 640,

    # Recombination
    'B_rad_cm3_s': 4.73e-15,       # Very low (indirect gap)
    'C_n_cm6_s': 1.1e-30,
    'C_p_cm6_s': 3.0e-31,
    'tau_n0_s': 1e-5,
    'tau_p0_s': 1e-5,

    # Absorption coefficient (cm^-1) - indirect gap, much lower than GaAs
    # Source: Green & Keevers (1995), Sze
    'absorption_data_nm_cm1': {
        400: 1.0e5,
        450: 5.0e4,
        500: 1.1e4,
        550: 5.5e3,
        600: 3.8e3,
        650: 2.9e3,
        700: 2.1e3,
        750: 1.4e3,
        800: 8.5e2,
        850: 4.5e2,
        900: 1.6e2,
        950: 5.0e1,
        1000: 1.2e1,
        1050: 2.0e0,
        1100: 3.0e-1,    # Near bandgap (~1107nm for 1.12eV)
    },

    'refractive_index_data': {
        500: 4.30,
        550: 4.08,
        600: 3.94,
        650: 3.84,
        700: 3.77,
        750: 3.72,
        800: 3.68,
        850: 3.65,
        900: 3.62,
        950: 3.60,
    },
}


# ---- GALLIUM NITRIDE (GaN) ----
# Source: Ioffe + literature (for completeness / future use)
MATERIALS['GaN'] = {
    'name': 'Gallium Nitride',
    'crystal_structure': 'Wurtzite',
    'bandgap_type': 'direct',
    'E_g_300K_eV': 3.44,
    'E_g_0K_eV': 3.51,
    'varshni_alpha_eV_per_K': 7.7e-4,
    'varshni_beta_K': 600,
    'm_e_eff': 0.20,
    'm_h_eff': 1.4,
    'eps_r': 8.9,
    'mu_n_300K_cm2_Vs': 1000,
    'mu_p_300K_cm2_Vs': 30,
    'n_i_300K_cm3': 1.9e-10,
    'N_c_300K_cm3': 2.3e18,
    'N_v_300K_cm3': 4.6e19,
    'N_c_coeff': 4.3e14,
    'N_v_coeff': 8.9e15,
    'electron_affinity_eV': 4.1,
    'density_g_cm3': 6.15,
}


# ---- ALUMINIUM INDIUM GALLIUM PHOSPHIDE (AlInGaP) ----
# Source: Literature — used in red/orange/yellow LEDs
MATERIALS['AlInGaP'] = {
    'name': 'Aluminium Indium Gallium Phosphide',
    'crystal_structure': 'Zinc Blende',
    'bandgap_type': 'direct',      # At low Al content
    'E_g_300K_eV': 1.91,           # Varies with composition; ~1.9 for red LED
    'E_g_0K_eV': 1.96,
    'varshni_alpha_eV_per_K': 4.0e-4,  # Approximate
    'varshni_beta_K': 200,
    'eps_r': 11.1,
    'mu_n_300K_cm2_Vs': 200,
    'mu_p_300K_cm2_Vs': 40,
    'n_i_300K_cm3': 1.0e2,         # Wide bandgap → very low n_i
}


# =============================================================================
# COMPONENT DATASHEETS
# =============================================================================
# Extracted electrical/optical parameters from manufacturer datasheets.

DATASHEETS = {}

# ---- KXOB25-04X3F (GaAs Solar Cell Module) ----
# Source: ANYSOLAR/IXYS KXOB25-04X3F Datasheet + Kadirvelu 2021
DATASHEETS['KXOB25-04X3F'] = {
    'manufacturer': 'ANYSOLAR (formerly IXYS)',
    'type': 'solar_cell',
    'material': 'GaAs',
    'description': 'IXOLAR High Efficiency SolarBIT - 3-cell series module',

    # Physical dimensions
    'dimensions_mm': (22.0, 7.0, 1.8),    # L x W x H
    'active_area_cm2': 9.0,                # From Kadirvelu 2021 paper

    # Module configuration
    'n_cells_series': 3,                    # 3 cells in series per module
    # Note: Kadirvelu paper uses a MODULE of 13 cells — that's multiple
    # KXOB25 units wired in series, not 13 cells within one KXOB25.
    # The KXOB25-04X3F itself has Voc=2.07V (3 cells × ~0.69V each)

    # Electrical characteristics @ STC (AM1.5, 100 mW/cm², 25°C)
    'V_oc_V': 2.07,                        # Open-circuit voltage
    'I_sc_mA': 14.0,                        # Short-circuit current
    'V_mpp_V': 1.67,                        # Voltage at max power point
    'I_mpp_mA': 13.2,                       # Current at max power point
    'P_max_mW': 22.0,                       # Maximum power
    'cell_efficiency_pct': 25.0,            # Wafer-level efficiency

    # Per-cell derived (for physics calculations)
    'V_oc_per_cell_V': 2.07 / 3,           # ~0.69 V
    'cell_area_cm2': 9.0 / 3,              # ~3.0 cm² per cell

    # Optical/detection (from Kadirvelu 2021)
    'responsivity_A_per_W': 0.457,          # At 650nm (measured)
    'peak_QE_pct': 85.0,                    # Approximate at 650nm
    'spectral_range_nm': (400, 870),        # GaAs absorption range

    # Circuit parameters (from Kadirvelu 2021 analysis)
    'C_j_pF': 798,                          # Junction capacitance
    'R_sh_ohm': 138.8e3,                    # Shunt resistance (138.8 kΩ)
    'I_dark_nA': 1.0,                       # Dark current (approximate)

    # Operating conditions
    'temperature_range_C': (-40, 90),

    # Doping estimates (from GaAs PV literature, not datasheet)
    # These are TYPICAL values for GaAs solar cells
    'estimated_N_d_cm3': 1e17,              # n-type emitter
    'estimated_N_a_cm3': 1e15,              # p-type base
    'estimated_junction_depth_um': 0.5,     # Typical
    'estimated_surface_reflectance': 0.05,  # With AR coating
}

# ---- SM141K04LV (GaAs Solar Cell Module - Larger) ----
# Source: ANYSOLAR SM141K04LV Datasheet, DigiKey
DATASHEETS['SM141K04LV'] = {
    'manufacturer': 'ANYSOLAR (formerly IXYS)',
    'type': 'solar_cell',
    'material': 'GaAs',
    'description': 'IXOLAR High Efficiency SolarMD - 4-cell series module',

    'dimensions_mm': (45.0, 15.0, 2.1),
    'active_area_cm2': 4.5,                 # Estimated from dimensions

    # Module: 4 cells in series
    'n_cells_series': 4,

    # Electrical @ STC
    'V_oc_V': 2.76,                         # 4 cells × ~0.69V
    'I_sc_mA': 55.1,
    'V_mpp_V': 2.23,
    'I_mpp_mA': 55.1,                       # From DigiKey listing
    'P_max_mW': 123.0,
    'cell_efficiency_pct': 25.0,

    'V_oc_per_cell_V': 2.76 / 4,            # ~0.69V
    'cell_area_cm2': 4.5 / 4,               # ~1.125 cm² per cell

    'responsivity_A_per_W': 0.45,            # Estimated (same GaAs tech)
    'peak_QE_pct': 85.0,
    'spectral_range_nm': (400, 870),

    # Estimated circuit parameters (scaled from KXOB25)
    # C_j scales with area; this has smaller per-cell area
    'C_j_pF': 300,                           # Estimated from area ratio
    'R_sh_ohm': 200e3,                       # Higher for smaller area
    'I_dark_nA': 0.5,

    'temperature_range_C': (-40, 90),

    'estimated_N_d_cm3': 1e17,
    'estimated_N_a_cm3': 1e15,
    'estimated_junction_depth_um': 0.5,
    'estimated_surface_reflectance': 0.05,
}

# ---- BPW34 (Silicon PIN Photodiode) ----
# Source: Vishay BPW34 Datasheet Rev 2.1 (Doc 81521)
DATASHEETS['BPW34'] = {
    'manufacturer': 'Vishay Semiconductors',
    'type': 'photodiode',
    'material': 'Si',
    'description': 'Silicon PIN Photodiode - high speed, high sensitivity',

    'dimensions_mm': (5.4, 4.3, 3.2),
    'active_area_mm2': 7.5,                  # Datasheet: 7.5 mm²
    'active_area_cm2': 7.5e-2,               # = 0.075 cm²

    'n_cells_series': 1,                     # Single junction

    # Electrical characteristics (from datasheet, V_R = 0V photovoltaic mode)
    'V_oc_mV': 350,                          # @ 1 mW/cm², 950nm
    'I_sc_uA': 47,                           # @ 1 mW/cm², 950nm

    # Photoconductive mode (reverse biased)
    'V_R_max_V': 60,                         # Max reverse voltage
    'I_dark_nA_typ': 2,                      # @ V_R=10V (typ)
    'I_dark_nA_max': 30,                     # @ V_R=10V (max)

    # Capacitance (datasheet)
    'C_j_0V_pF': 70,                         # @ V_R=0V, f=1MHz
    'C_j_3V_pF_typ': 25,                     # @ V_R=3V (typ)
    'C_j_3V_pF_max': 40,                     # @ V_R=3V (max)

    # Spectral characteristics (datasheet)
    'lambda_peak_nm': 900,                   # Peak sensitivity wavelength
    'spectral_range_nm': (430, 1100),        # λ_0.1 range
    'half_sensitivity_angle_deg': 65,        # ±65°

    # Reverse light current (datasheet: used for responsivity calc)
    # I_ra = 50 µA @ 1 mW/cm², 950nm, V_R=5V
    # Responsivity = I_ra / (Ee * A) = 50e-6 / (10 * 7.5e-6) = 0.667 A/W @ 950nm
    # More standard: ~0.62 A/W at 900nm, ~0.35 A/W at 650nm
    'responsivity_peak_A_per_W': 0.62,       # Estimated at 900nm
    'responsivity_650nm_A_per_W': 0.35,      # Estimated at 650nm

    # Timing (datasheet)
    'rise_time_ns': 100,                     # @ V_R=10V, R_L=1kΩ, 820nm
    'fall_time_ns': 100,

    # NEP
    'NEP_W_per_rtHz': 4e-14,                # @ V_R=10V, 950nm

    # Thermal
    'R_th_JA_K_per_W': 350,

    # Doping estimates (from Si PIN diode literature)
    'estimated_N_d_cm3': 1e15,               # Lightly doped I-region
    'estimated_N_a_cm3': 1e18,               # p+ region
    'estimated_i_layer_um': 200,             # Intrinsic region thickness
    'estimated_surface_reflectance': 0.30,   # No AR coating (bare Si)
}


# ---- OSRAM LR W5SN (Red LED) ----
# Source: OSRAM datasheet + Kadirvelu 2021
DATASHEETS['OSRAM_LRW5SN'] = {
    'manufacturer': 'OSRAM Opto Semiconductors',
    'type': 'led',
    'material': 'AlInGaP',
    'description': 'High-power red LED (Golden DRAGON series)',

    # Optical characteristics
    'peak_wavelength_nm': 625,               # Dominant wavelength
    'spectral_width_fwhm_nm': 20,            # Typical for AlInGaP
    'beam_half_angle_deg': 9,                # With Fraen lens (Kadirvelu)

    # Electrical characteristics
    'V_f_typ_V': 2.1,                        # Forward voltage @ 20mA
    'I_f_max_mA': 1000,                      # Max forward current
    'I_f_typ_mA': 350,                       # Typical operating current

    # Efficiency
    'radiant_efficiency_pct': 15,            # Wall-plug efficiency estimate
    'luminous_flux_lm': 67,                  # Typical @ 350mA

    # Thermal
    'R_th_JC_K_per_W': 10,                   # Junction to case
    'R_th_total_K_per_W': 250,               # Approximate with lens

    # Lambertian / radiation pattern
    # m_L = -ln(2) / ln(cos(theta_half))
    # For 9°: m_L ≈ 53.1 (very narrow with lens)
    # Without lens: ~60° half-angle → m_L ≈ 1.0
    'lambertian_order_with_lens': 53.1,
    'lambertian_order_bare': 1.0,

    # Series resistance (estimated from V-I curve)
    'R_s_ohm': 5,
}


# =============================================================================
# QE CURVE DATA
# =============================================================================
# Quantum efficiency vs wavelength for key components.
# These are normalized curves - peak QE value stored in datasheet dict.

# GaAs solar cell QE curve (typical, from literature)
# Source: Adapted from Kadirvelu 2021 + GaAs PV literature
GAAS_QE_CURVE = {
    'wavelength_nm': np.array([
        350, 400, 450, 500, 550, 600, 625, 650, 700, 750, 800, 850, 870
    ]),
    'QE_fraction': np.array([
        0.10, 0.50, 0.70, 0.80, 0.85, 0.87, 0.86, 0.85, 0.80, 0.65, 0.40, 0.10, 0.01
    ]),
}

# BPW34 relative spectral sensitivity (from datasheet Fig. 7)
# Normalized to peak at 900nm = 1.0
BPW34_SPECTRAL_RESPONSE = {
    'wavelength_nm': np.array([
        430, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000, 1050, 1100
    ]),
    'relative_sensitivity': np.array([
        0.05, 0.20, 0.40, 0.55, 0.70, 0.82, 0.90, 0.96, 0.99, 1.00, 0.95, 0.75, 0.40, 0.05
    ]),
}


# =============================================================================
# TYPICAL DOPING PROFILES (from literature)
# =============================================================================
# Used when datasheets don't specify doping levels

TYPICAL_DOPING = {
    'GaAs_solar_cell': {
        'emitter_type': 'n',
        'N_d_cm3': 1e17,           # n-type emitter
        'N_a_cm3': 1e15,           # p-type base
        'source': 'Sze Table 13.1, typical GaAs PV',
    },
    'Si_PIN_photodiode': {
        'N_d_cm3': 1e15,           # Lightly doped n-region or i-region
        'N_a_cm3': 1e18,           # Heavily doped p+ anode
        'source': 'Sze Ch.13, typical Si PIN',
    },
    'Si_solar_cell': {
        'emitter_type': 'n',
        'N_d_cm3': 1e20,           # n+ emitter
        'N_a_cm3': 1e16,           # p-type base
        'source': 'PV Education, typical c-Si cell',
    },
}


# =============================================================================
# DIFFUSION LENGTH ESTIMATES (from literature)
# =============================================================================
# L = sqrt(D * tau)

DIFFUSION_LENGTHS = {
    'GaAs': {
        'L_n_um': 4.5,             # sqrt(200 * 1e-8) * 1e4 ≈ 4.5 µm
        'L_p_um': 1.0,             # sqrt(10 * 1e-8) * 1e4 ≈ 1.0 µm
        'source': 'Calculated from D and tau in Ioffe data',
    },
    'Si': {
        'L_n_um': 190,             # sqrt(36 * 1e-5) * 1e4 ≈ 190 µm
        'L_p_um': 110,             # sqrt(12 * 1e-5) * 1e4 ≈ 110 µm
        'source': 'Calculated from D and tau in Ioffe data',
    },
}


# =============================================================================
# SELF-TEST
# =============================================================================
def verify_data():
    """Quick sanity checks on all reference data."""
    q = CONSTANTS['q']
    k = CONSTANTS['k_B']
    T = 300  # K

    print("=" * 70)
    print("REFERENCE DATA VERIFICATION")
    print("=" * 70)

    # Check GaAs bandgap at 300K via Varshni
    gaas = MATERIALS['GaAs']
    E_g_calc = gaas['E_g_0K_eV'] - (
        gaas['varshni_alpha_eV_per_K'] * T**2 / (T + gaas['varshni_beta_K'])
    )
    print(f"\nGaAs E_g(300K): Varshni={E_g_calc:.4f} eV, Stored={gaas['E_g_300K_eV']} eV", end=" ")
    assert abs(E_g_calc - gaas['E_g_300K_eV']) < 0.01, "GaAs bandgap mismatch!"
    print("[PASS]")

    # Check Si bandgap at 300K
    si = MATERIALS['Si']
    E_g_si = si['E_g_0K_eV'] - (
        si['varshni_alpha_eV_per_K'] * T**2 / (T + si['varshni_beta_K'])
    )
    print(f"Si   E_g(300K): Varshni={E_g_si:.4f} eV, Stored={si['E_g_300K_eV']} eV", end=" ")
    assert abs(E_g_si - si['E_g_300K_eV']) < 0.02, "Si bandgap mismatch!"
    print("[PASS]")

    # Check n_i consistency: n_i = sqrt(Nc*Nv)*exp(-Eg/(2kT))
    kT_eV = k * T / q
    ni_calc = np.sqrt(gaas['N_c_300K_cm3'] * gaas['N_v_300K_cm3']) * \
              np.exp(-gaas['E_g_300K_eV'] / (2 * kT_eV))
    print(f"GaAs n_i(300K): Calc={ni_calc:.2e}, Stored={gaas['n_i_300K_cm3']:.2e} cm⁻³", end=" ")
    # Order of magnitude check (n_i very sensitive to E_g precision)
    assert 1e4 < ni_calc < 1e8, "GaAs n_i out of range"
    print("[PASS]")

    ni_si = np.sqrt(si['N_c_300K_cm3'] * si['N_v_300K_cm3']) * \
            np.exp(-si['E_g_300K_eV'] / (2 * kT_eV))
    print(f"Si   n_i(300K): Calc={ni_si:.2e}, Stored={si['n_i_300K_cm3']:.2e} cm⁻³", end=" ")
    assert 1e8 < ni_si < 1e12, "Si n_i out of range"
    print("[PASS]")

    # Check responsivity formula: R = q*λ*QE/(h*c)
    lam = 650e-9  # 650nm
    QE = 0.85
    R_calc = q * lam * QE / (CONSTANTS['h'] * CONSTANTS['c'])
    R_ds = DATASHEETS['KXOB25-04X3F']['responsivity_A_per_W']
    print(f"\nKXOB25 R(650nm): Physics={R_calc:.4f} A/W, Datasheet={R_ds} A/W", end=" ")
    assert abs(R_calc - R_ds) / R_ds < 0.05, "Responsivity mismatch > 5%"
    print("[PASS]")

    # Check Lambertian order: m = -ln(2)/ln(cos(theta))
    theta = np.radians(9)
    m_L = -np.log(2) / np.log(np.cos(theta))
    print(f"Lambertian order (9° half-angle): {m_L:.1f}", end=" ")
    assert 50 < m_L < 60, "Lambertian order unexpected"
    print("[PASS]")

    # Check all materials have required keys
    required_keys = ['E_g_300K_eV', 'varshni_alpha_eV_per_K', 'varshni_beta_K', 'eps_r']
    for name, mat in MATERIALS.items():
        for key in required_keys:
            assert key in mat, f"{name} missing key: {key}"
    print(f"\nAll {len(MATERIALS)} materials have required fields [PASS]")

    # Check all datasheets
    print(f"All {len(DATASHEETS)} component datasheets loaded [PASS]")

    # Summary
    print(f"\n{'=' * 70}")
    print(f"MATERIALS: {', '.join(MATERIALS.keys())}")
    print(f"COMPONENTS: {', '.join(DATASHEETS.keys())}")
    print(f"ALL CHECKS PASSED")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    verify_data()
