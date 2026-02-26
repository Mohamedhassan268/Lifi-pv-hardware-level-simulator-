# cosim/__init__.py
"""
Co-Simulation Infrastructure for Hardware-Faithful LiFi-PV Simulator

Provides paper-agnostic simulation pipeline:
    TX (Python/SPICE) -> Channel (Python) -> RX (LTspice/ngspice)

Modules:
    system_config  - SystemConfig dataclass + JSON serialization
    ltspice_runner - LTspice auto-detect + batch runner
    raw_parser     - Parse LTspice .raw binary files
    pwl_writer     - Write photocurrent PWL bridge files
    session        - Session directory management
    pipeline       - 3-step TX->CH->RX orchestrator
"""

from .system_config import SystemConfig
from .ltspice_runner import LTSpiceRunner
from .raw_parser import LTSpiceRawParser
from .pwl_writer import write_photocurrent_pwl
from .session import SessionManager
from .pipeline import SimulationPipeline
from .spice_finder import find_ngspice, find_ltspice, spice_available

__all__ = [
    'SystemConfig',
    'LTSpiceRunner',
    'LTSpiceRawParser',
    'write_photocurrent_pwl',
    'SessionManager',
    'SimulationPipeline',
    'find_ngspice',
    'find_ltspice',
    'spice_available',
]
