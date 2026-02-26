# systems/__init__.py
"""
Paper Validation Systems for Hardware-Faithful LiFi-PV Simulator

Each module implements a complete system from a published paper
with full SPICE-level simulation capability.

Available Systems:
    - kadirvelu2021: "A Circuit for Simultaneous Reception of Data and Power Using a Solar Cell"
                     IEEE TGCN 2021

System Registry:
    SYSTEM_REGISTRY maps paper keys to system classes.
    Use get_system(name) to retrieve a system instance.

Modules:
    - base:                    BaseSystem abstract class
    - kadirvelu2021:           Core parameters, netlist gen, simulation runner
    - kadirvelu2021_netlist:   Unified SPICE netlist generator (full system)
    - kadirvelu2021_schematic: Publication-quality visual schematics (schemdraw)
    - kadirvelu2021_channel:   Optical channel model (Lambertian, AWGN)

Usage:
    from systems import get_system, list_systems
    from systems.kadirvelu2021 import KadirveluSimulation, KadirveluParams

    # Registry-based access
    sys = get_system('kadirvelu2021')
    params = sys.get_params()

    # Direct import
    sim = KadirveluSimulation()
    sim.run_transient(fsw=50e3, modulation_depth=0.5)
"""

from .base import BaseSystem

from .kadirvelu2021 import (
    KadirveluParams,
    KadirveluNetlist,
    KadirveluSimulation,
    plot_frequency_response,
    plot_dcdc_efficiency,
)

from .kadirvelu2021_netlist import FullSystemNetlist
from .kadirvelu2021_channel import OpticalChannel


# =============================================================================
# System Registry
# =============================================================================

class _KadirveluSystem(BaseSystem):
    """Kadirvelu 2021 system wrapper conforming to BaseSystem interface."""

    name = 'kadirvelu2021'
    paper_reference = (
        'Kadirvelu et al., "A Circuit for Simultaneous Reception of Data '
        'and Power Using a Solar Cell", IEEE TGCN 2021'
    )
    simulation_engine = 'spice'

    def __init__(self):
        self._params = KadirveluParams()
        self._netlist_gen = KadirveluNetlist(self._params)

    def get_params(self):
        return {k: v for k, v in vars(type(self._params)).items()
                if not k.startswith('_') and not callable(v)}

    def get_validation_targets(self):
        return {
            'target_harvested_power_uW': self._params.TARGET_HARVESTED_POWER_uW,
            'target_ber': self._params.TARGET_BER,
            'target_noise_rms_mV': self._params.TARGET_NOISE_RMS_mV,
        }

    def calculate_theoretical_values(self):
        sim = KadirveluSimulation()
        return sim.calculate_theoretical_values()

    def generate_netlist(self, **kwargs):
        return self._netlist_gen.generate_full_system(**kwargs)

    def generate_schematics(self, output_dir, fmt='svg'):
        try:
            from .kadirvelu2021_schematic import draw_all_schematics
            return draw_all_schematics(output_dir, fmt=fmt)
        except ImportError:
            return []

    def supported_modulations(self):
        return ['OOK', 'OOK_Manchester']


# Registry: maps paper key -> system class
SYSTEM_REGISTRY = {
    'kadirvelu2021': _KadirveluSystem,
}


def get_system(name: str) -> BaseSystem:
    """
    Get a system instance by paper key.

    Args:
        name: Paper key (e.g., 'kadirvelu2021')

    Returns:
        BaseSystem instance

    Raises:
        KeyError: If system not found
    """
    if name not in SYSTEM_REGISTRY:
        available = ', '.join(SYSTEM_REGISTRY.keys())
        raise KeyError(f"System '{name}' not found. Available: {available}")
    return SYSTEM_REGISTRY[name]()


def list_systems():
    """List all registered system keys."""
    return list(SYSTEM_REGISTRY.keys())


__all__ = [
    # Base
    'BaseSystem',
    # Registry
    'SYSTEM_REGISTRY',
    'get_system',
    'list_systems',
    # Kadirvelu 2021
    'KadirveluParams',
    'KadirveluNetlist',
    'KadirveluSimulation',
    'FullSystemNetlist',
    'OpticalChannel',
    'plot_frequency_response',
    'plot_dcdc_efficiency',
]
