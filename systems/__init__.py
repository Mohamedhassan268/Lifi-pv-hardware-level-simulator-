# systems/__init__.py
"""
Paper Validation Systems for Hardware-Faithful LiFi-PV Simulator

Each module implements a complete system from a published paper
with full SPICE-level simulation capability.

Available Systems:
    - kadirvelu2021: "A Circuit for Simultaneous Reception of Data and Power Using a Solar Cell"
                     IEEE TGCN 2021

Usage:
    from systems.kadirvelu2021 import KadirveluSimulation, KadirveluParams
    
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

__all__ = [
    'KadirveluParams',
    'KadirveluNetlist', 
    'KadirveluSimulation',
    'plot_frequency_response',
    'plot_dcdc_efficiency',
]
