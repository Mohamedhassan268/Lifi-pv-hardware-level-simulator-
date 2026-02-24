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
    - LXM5_PD01: Philips Lumileds white LED (with Fraen lens)

Amplifiers:
    - INA322: TI instrumentation amplifier
    - TLV2379: TI quad RRIO op-amp
    - ADA4891: ADI high-speed op-amp

Comparators:
    - TLV7011: TI nanopower comparator

MOSFETs:
    - BSD235N: Infineon (LED driver)
    - NTS4409: ON Semi (DC-DC switch)

Usage:
    from components import KXOB25_04X3F, BPW34, LXM5_PD01, INA322

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

# LEDs
from .leds import (
    LXM5_PD01,
)

# Amplifiers
from .amplifiers import (
    INA322,
    TLV2379,
    ADA4891,
)

# Comparators
from .comparators import (
    ComparatorBase,
    TLV7011,
)

# MOSFETs
from .mosfets import (
    MOSFETBase,
    BSD235N,
    NTS4409,
)

__all__ = [
    # Base classes
    'PhotodetectorBase',
    'LEDBase',
    'AmplifierBase',
    'ComparatorBase',
    'MOSFETBase',

    # Solar cells
    'KXOB25_04X3F',
    'SM141K',
    'GenericGaAsPV',

    # Photodiodes
    'BPW34',
    'SFH206K',
    'VEMD5510',
    'PhotodiodeFromSPICE',

    # LEDs
    'LXM5_PD01',

    # Amplifiers
    'INA322',
    'TLV2379',
    'ADA4891',

    # Comparators
    'TLV7011',

    # MOSFETs
    'BSD235N',
    'NTS4409',
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

    # LEDs
    'LXM5-PD01': LXM5_PD01,
    'LXM5_PD01': LXM5_PD01,

    # Amplifiers
    'INA322': INA322,
    'TLV2379': TLV2379,
    'ADA4891': ADA4891,

    # Comparators
    'TLV7011': TLV7011,

    # MOSFETs
    'BSD235N': BSD235N,
    'NTS4409': NTS4409,
}


def get_component(name: str, **kwargs):
    """
    Get component by part number.

    Args:
        name: Part number (e.g., 'KXOB25-04X3F', 'BPW34', 'INA322')
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
