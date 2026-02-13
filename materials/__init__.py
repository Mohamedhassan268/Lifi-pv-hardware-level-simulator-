# materials/__init__.py
"""
Semiconductor Materials Database for Hardware-Faithful LiFi-PV Simulator.

Provides temperature-dependent material properties using validated physics models.

Usage:
    from materials import get_material, SILICON, GAAS
    
    # Get by name
    si = get_material('Si')
    gaas = get_material('GaAs')
    
    # Access properties
    print(gaas.bandgap(300))  # 1.42 eV at 300K
    print(gaas.electron_mobility(350))  # Temperature-dependent
    
    # Create custom InGaN composition
    from materials import InGaN
    blue_led = InGaN(x_In=0.20)  # 20% Indium for blue
"""

from .semiconductors import (
    SemiconductorMaterial,
    get_material,
    list_materials,
    InGaN,
    SILICON,
    GAAS,
    GAN,
    INGAN_BLUE,
    ALINGAP_RED,
    MATERIALS_DATABASE,
)

__all__ = [
    'SemiconductorMaterial',
    'get_material',
    'list_materials',
    'InGaN',
    'SILICON',
    'GAAS',
    'GAN',
    'INGAN_BLUE',
    'ALINGAP_RED',
    'MATERIALS_DATABASE',
]
