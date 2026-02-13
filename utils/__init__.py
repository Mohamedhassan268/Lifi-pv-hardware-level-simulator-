# utils/__init__.py
"""
Utility modules for Hardware-Faithful LiFi-PV Simulator.
"""

from .constants import (
    Q, Q_ELECTRON,
    K_B, K_BOLTZMANN,
    H, H_PLANCK, HBAR,
    C, C_LIGHT,
    EPSILON_0,
    M_E, M_ELECTRON,
    V_T_300K,
    HC_EV_NM,
    ROOM_TEMPERATURE_K,
    eV_to_J, J_to_eV,
    nm_to_eV, eV_to_nm,
    thermal_voltage,
    ideal_responsivity,
)

__all__ = [
    'Q', 'Q_ELECTRON',
    'K_B', 'K_BOLTZMANN', 
    'H', 'H_PLANCK', 'HBAR',
    'C', 'C_LIGHT',
    'EPSILON_0',
    'M_E', 'M_ELECTRON',
    'V_T_300K',
    'HC_EV_NM',
    'ROOM_TEMPERATURE_K',
    'eV_to_J', 'J_to_eV',
    'nm_to_eV', 'eV_to_nm',
    'thermal_voltage',
    'ideal_responsivity',
]
