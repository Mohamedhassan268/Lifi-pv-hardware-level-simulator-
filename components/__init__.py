# components/__init__.py
"""
Hardware Component Library for Hardware-Faithful LiFi-PV Simulator

Provides pre-configured hardware components where electrical parameters
EMERGE from component selection rather than manual configuration.

Solar Cells:
    - KXOB25_04X3F: IXYS GaAs (primary target)
    - SM141K: IXYS Silicon
    - GenericGaAsPV: Parametric GaAs model

Photodiodes:
    - BPW34: Vishay general-purpose PIN
    - SFH206K: OSRAM high-speed PIN
    - VEMD5510: Vishay VLC-optimized

LEDs:
    - (Coming soon)

Amplifiers:
    - (Coming soon)

Usage:
    from components import KXOB25_04X3F, BPW34
    
    # Select component - parameters EMERGE
    rx = KXOB25_04X3F()
    print(rx.responsivity)  # 0.457 A/W - from physics
    print(rx.capacitance)   # 798 pF - from datasheet
    print(rx.bandwidth(220)) # ~14 kHz - derived
    
    # Get all parameters for simulation
    params = rx.get_parameters()
"""

# Base classes
from .base import (
    PhotodetectorBase,
    LEDBase,
    AmplifierBase,
)

# Solar cells
from .solar_cells import (
    KXOB25_04X3F,
    SM141K,
    GenericGaAsPV,
)

# Photodiodes
from .photodiodes import (
    BPW34,
    SFH206K,
    VEMD5510,
    PhotodiodeFromSPICE,
)

__all__ = [
    # Base classes
    'PhotodetectorBase',
    'LEDBase',
    'AmplifierBase',
    
    # Solar cells
    'KXOB25_04X3F',
    'SM141K',
    'GenericGaAsPV',
    
    # Photodiodes
    'BPW34',
    'SFH206K',
    'VEMD5510',
    'PhotodiodeFromSPICE',
]


# =============================================================================
# COMPONENT REGISTRY
# =============================================================================

COMPONENT_REGISTRY = {
    # Solar cells
    'KXOB25-04X3F': KXOB25_04X3F,
    'KXOB25_04X3F': KXOB25_04X3F,
    'SM141K': SM141K,
    
    # Photodiodes
    'BPW34': BPW34,
    'SFH206K': SFH206K,
    'VEMD5510': VEMD5510,
}


def get_component(name: str, **kwargs):
    """
    Get component by part number.
    
    Args:
        name: Part number (e.g., 'KXOB25-04X3F', 'BPW34')
        **kwargs: Additional arguments for component constructor
        
    Returns:
        Component instance
        
    Raises:
        KeyError: If component not found
    """
    name_clean = name.upper().replace('-', '_')
    
    for key, cls in COMPONENT_REGISTRY.items():
        if key.upper().replace('-', '_') == name_clean:
            return cls(**kwargs)
    
    available = list(COMPONENT_REGISTRY.keys())
    raise KeyError(f"Component '{name}' not found. Available: {available}")


def list_components() -> list:
    """Return list of available component names."""
    return list(COMPONENT_REGISTRY.keys())
