# materials/__init__.py
"""Materials database and semiconductor property functions."""
from .reference_data import MATERIALS, DATASHEETS, CONSTANTS, GAAS_QE_CURVE, BPW34_SPECTRAL_RESPONSE
from .semiconductors import SemiconductorMaterial, GAAS, SILICON, get_material
