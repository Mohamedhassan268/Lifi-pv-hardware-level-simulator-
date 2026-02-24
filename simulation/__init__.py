# simulation/__init__.py
"""
Simulation Module for Hardware-Faithful LiFi-PV Simulator

Provides tools for running and analyzing SPICE simulations:
    - NgSpiceRunner: Interface to ngspice batch simulation
    - PRBSGenerator: PRBS/OOK signal generation
    - Analysis: BER, eye diagrams, frequency response post-processing
"""

from .ngspice_runner import NgSpiceRunner
from .prbs_generator import generate_prbs, generate_ook_waveform, write_pwl_file
from .analysis import (
    calculate_ber_from_transient,
    theoretical_ber_ook,
    eye_diagram_data,
)

__all__ = [
    'NgSpiceRunner',
    'generate_prbs',
    'generate_ook_waveform',
    'write_pwl_file',
    'calculate_ber_from_transient',
    'theoretical_ber_ook',
    'eye_diagram_data',
]
