# cosim/__init__.py
"""
Co-Simulation Infrastructure for Hardware-Faithful LiFi-PV Simulator

Provides paper-agnostic simulation pipeline:
    TX (Python/SPICE) -> Channel (Python) -> RX (LTspice/ngspice)

Modules:
    system_config  - SystemConfig dataclass + JSON serialization
    channel        - Optical channel (Lambertian, Beer-Lambert, MIMO)
    noise          - 6-source physical noise model
    modulation     - 5 modulation schemes + BER prediction
    python_engine  - Pure-Python RX simulation engine
    ltspice_runner - LTspice auto-detect + batch runner
    raw_parser     - Parse LTspice .raw binary files
    pwl_writer     - Write photocurrent PWL bridge files
    session        - Session directory management
    pipeline       - 3-step TX->CH->RX orchestrator
"""

from .system_config import SystemConfig
from .channel import OpticalChannel
from .noise import NoiseModel, NoiseBreakdown
from .modulation import modulate, demodulate, predict_ber, calculate_ber
from .pv_model import PVCellModel
from .rx_chain import ReceiverChain
from .dcdc_model import BoostConverter
from .tx_model import LEDTransmitter
from .ltspice_runner import LTSpiceRunner
from .ngspice_runner import NgSpiceRunner
from .raw_parser import LTSpiceRawParser
from .pwl_writer import write_photocurrent_pwl, write_noise_pwl
from .spice_extract import extract_spice_waveforms, compute_ber_from_spice
from .sim_result import SimulationResult
from .session import SessionManager
from .pipeline import SimulationPipeline
from .spice_finder import find_ngspice, find_ltspice, spice_available

__all__ = [
    'SystemConfig',
    'OpticalChannel',
    'NoiseModel',
    'NoiseBreakdown',
    'modulate',
    'demodulate',
    'predict_ber',
    'calculate_ber',
    'PVCellModel',
    'ReceiverChain',
    'BoostConverter',
    'LEDTransmitter',
    'LTSpiceRunner',
    'NgSpiceRunner',
    'LTSpiceRawParser',
    'write_photocurrent_pwl',
    'write_noise_pwl',
    'extract_spice_waveforms',
    'compute_ber_from_spice',
    'SimulationResult',
    'SessionManager',
    'SimulationPipeline',
    'find_ngspice',
    'find_ltspice',
    'spice_available',
]
