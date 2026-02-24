# systems/__init__.py
"""
Paper Validation Systems for Hardware-Faithful LiFi-PV Simulator

Each module implements a complete system from a published paper
with full SPICE-level simulation capability.

Available Systems:
    - kadirvelu2021: "A Circuit for Simultaneous Reception of Data and Power Using a Solar Cell"
                     IEEE TGCN 2021

Modules:
    - kadirvelu2021:           Core parameters, netlist gen, simulation runner
    - kadirvelu2021_netlist:   Unified SPICE netlist generator (full system)
    - kadirvelu2021_schematic: Publication-quality visual schematics (schemdraw)
    - kadirvelu2021_channel:   Optical channel model (Lambertian, AWGN)

Usage:
    from systems.kadirvelu2021 import KadirveluSimulation, KadirveluParams
    from systems.kadirvelu2021_netlist import FullSystemNetlist
    from systems.kadirvelu2021_channel import OpticalChannel

    sim = KadirveluSimulation()
    sim.run_transient(fsw=50e3, modulation_depth=0.5)
"""

from .kadirvelu2021 import (
    KadirveluParams,
    KadirveluNetlist,
    KadirveluSimulation,
    plot_frequency_response,
    plot_dcdc_efficiency,
)

from .kadirvelu2021_netlist import FullSystemNetlist
from .kadirvelu2021_channel import OpticalChannel

__all__ = [
    'KadirveluParams',
    'KadirveluNetlist',
    'KadirveluSimulation',
    'FullSystemNetlist',
    'OpticalChannel',
    'plot_frequency_response',
    'plot_dcdc_efficiency',
]
