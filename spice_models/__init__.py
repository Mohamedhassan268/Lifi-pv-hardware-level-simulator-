# spice_models/__init__.py
"""
Behavioral SPICE Model Library

Contains .lib files for all components in the Kadirvelu 2021 system:
    - lxm5_pd01.lib  : LED (InGaN diode + optical output)
    - ina322.lib      : Instrumentation amplifier (behavioral)
    - tlv2379.lib     : RRIO op-amp (behavioral)
    - tlv7011.lib     : Nanopower comparator (behavioral)
    - ada4891.lib     : High-speed op-amp (behavioral)
    - mosfets.lib     : BSD235N, NTS4409, Schottky diode models

Usage:
    Models can be included in SPICE netlists via:
        .include path/to/spice_models/ina322.lib
"""

import os

SPICE_MODELS_DIR = os.path.dirname(os.path.abspath(__file__))

def get_model_path(model_name: str) -> str:
    """Get absolute path to a .lib file."""
    path = os.path.join(SPICE_MODELS_DIR, f"{model_name}.lib")
    if not os.path.exists(path):
        available = [f for f in os.listdir(SPICE_MODELS_DIR) if f.endswith('.lib')]
        raise FileNotFoundError(
            f"Model '{model_name}.lib' not found. Available: {available}")
    return path

def list_models() -> list:
    """List available SPICE model files."""
    return [f.replace('.lib', '') for f in os.listdir(SPICE_MODELS_DIR)
            if f.endswith('.lib')]
