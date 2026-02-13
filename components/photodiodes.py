# components/photodiodes.py
"""
Photodiode Components for Hardware-Faithful LiFi-PV Simulator

Provides models for silicon PIN photodiodes:
    - BPW34: Vishay general-purpose PIN photodiode
    - SFH206K: OSRAM high-speed PIN photodiode
    - VEMD5510: Vishay VLC-optimized photodiode
    - PhotodiodeFromSPICE: Create from SPICE model

These components can be loaded from SPICE libraries or created
with datasheet parameters.
"""

import numpy as np
from typing import Dict, Any, Optional
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from components.base import PhotodetectorBase
from materials import SILICON
from utils.constants import Q, K_B, HC_EV_NM


# =============================================================================
# BPW34 - CLASSIC PIN PHOTODIODE
# =============================================================================

class BPW34(PhotodetectorBase):
    """
    Vishay BPW34 Silicon PIN Photodiode
    
    Classic general-purpose photodiode, widely used and well-characterized.
    
    Datasheet Parameters:
        - Active area: 7.5 mm²
        - Spectral range: 430-1100 nm
        - Peak wavelength: 850 nm
        - Responsivity: 0.62 A/W @ 850nm
        - Capacitance: 72 pF @ 0V, 26 pF @ 10V
        - Rise time: 20 ns (typ)
        - Dark current: 2 nA (typ)
    
    SPICE Parameters (from DIODE2.lib):
        - CJO = 72 pF
        - TT = 50 ns
        - VJ = 0.35 V
        - M = 0.28
    """
    
    def __init__(self, 
                 temperature_K: float = 300.0,
                 reverse_bias_V: float = 0.0):
        super().__init__(temperature_K, reverse_bias_V)
        self._material = SILICON
        
        # -----------------------------------------------------------------
        # DATASHEET PARAMETERS
        # -----------------------------------------------------------------
        self._active_area_mm2 = 7.5
        self._responsivity_850nm = 0.62  # A/W @ 850nm
        self._peak_wavelength_nm = 850
        
        # From SPICE model
        self._cjo = 72e-12  # Zero-bias capacitance (F)
        self._vj = 0.35      # Junction voltage (V)
        self._m = 0.28       # Grading coefficient
        self._tt = 50e-9     # Transit time (s)
        
        # Dark current
        self._dark_current_nA = 2.0
        
        # Resistances
        self._series_resistance = 3.0  # Ω (from SPICE RS)
        self._shunt_resistance = 10e9  # Ω (very high for good photodiode)
    
    @property
    def name(self) -> str:
        return "BPW34"
    
    @property
    def active_area_m2(self) -> float:
        return self._active_area_mm2 * 1e-6
    
    @property
    def responsivity(self) -> float:
        """Responsivity at peak wavelength (850nm)."""
        return self._responsivity_850nm
    
    @property
    def capacitance(self) -> float:
        """Junction capacitance at operating bias."""
        if self.reverse_bias_V <= 0:
            return self._cjo
        # C(V) = CJO / (1 + V/VJ)^M
        return self._cjo / (1 + self.reverse_bias_V / self._vj) ** self._m
    
    @property
    def dark_current(self) -> float:
        """Dark current (temperature dependent)."""
        # Doubles approximately every 10°C
        T = self.temperature_K
        T_ref = 300.0
        factor = 2 ** ((T - T_ref) / 10)
        return self._dark_current_nA * 1e-9 * factor
    
    @property
    def shunt_resistance(self) -> float:
        return self._shunt_resistance
    
    @property
    def series_resistance(self) -> float:
        return self._series_resistance
    
    @property
    def transit_time(self) -> float:
        """Transit time in seconds."""
        return self._tt
    
    def transit_bandwidth(self) -> float:
        """Transit-time limited bandwidth."""
        return 0.44 / self._tt
    
    def total_bandwidth(self, R_load: float) -> float:
        """
        Total bandwidth considering both RC and transit time.
        
        1/f_total² = 1/f_RC² + 1/f_tr²
        """
        f_rc = self.bandwidth(R_load)
        f_tr = self.transit_bandwidth()
        
        return 1 / np.sqrt(1/f_rc**2 + 1/f_tr**2)
    
    def responsivity_at_wavelength(self, wavelength_nm: float) -> float:
        """
        Calculate responsivity at specific wavelength.
        
        BPW34 has response from 430-1100nm with peak at 850nm.
        """
        if wavelength_nm < 430 or wavelength_nm > 1100:
            return 0.0
        
        # Gaussian-like response centered at 850nm
        sigma = 200
        relative = np.exp(-((wavelength_nm - self._peak_wavelength_nm)**2) / (2 * sigma**2))
        
        return self._responsivity_850nm * relative
    
    def get_parameters(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'type': 'photodiode',
            'material': 'Si',
            'active_area_m2': self.active_area_m2,
            'active_area_mm2': self._active_area_mm2,
            'responsivity': self.responsivity,
            'responsivity_peak_nm': self._peak_wavelength_nm,
            'capacitance': self.capacitance,
            'capacitance_0V': self._cjo,
            'dark_current': self.dark_current,
            'shunt_resistance': self.shunt_resistance,
            'series_resistance': self.series_resistance,
            'transit_time': self._tt,
            'bandwidth_rc_1kohm': self.bandwidth(1000),
            'bandwidth_transit': self.transit_bandwidth(),
            'bandwidth_total_1kohm': self.total_bandwidth(1000),
            'temperature_K': self.temperature_K,
            'reverse_bias_V': self.reverse_bias_V,
        }
    
    def __repr__(self):
        return f"BPW34(R={self.responsivity:.2f}A/W, C={self.capacitance*1e12:.0f}pF)"


# =============================================================================
# SFH206K - HIGH-SPEED PHOTODIODE
# =============================================================================

class SFH206K(PhotodetectorBase):
    """
    OSRAM SFH206K High-Speed Silicon PIN Photodiode
    
    Faster than BPW34, smaller area, lower capacitance.
    
    Datasheet Parameters:
        - Active area: 1 mm²
        - Spectral range: 400-1100 nm
        - Peak wavelength: 850 nm
        - Responsivity: 0.62 A/W @ 850nm
        - Capacitance: 11 pF @ 0V
        - Rise time: 5 ns (typ)
    """
    
    def __init__(self,
                 temperature_K: float = 300.0,
                 reverse_bias_V: float = 0.0):
        super().__init__(temperature_K, reverse_bias_V)
        self._material = SILICON
        
        self._active_area_mm2 = 1.0
        self._responsivity_850nm = 0.62
        self._peak_wavelength_nm = 850
        
        self._cjo = 11e-12
        self._vj = 0.6
        self._m = 0.33
        self._tt = 10e-9
        
        self._dark_current_nA = 1.0
        self._series_resistance = 5.0
        self._shunt_resistance = 50e9
    
    @property
    def name(self) -> str:
        return "SFH206K"
    
    @property
    def active_area_m2(self) -> float:
        return self._active_area_mm2 * 1e-6
    
    @property
    def responsivity(self) -> float:
        return self._responsivity_850nm
    
    @property
    def capacitance(self) -> float:
        if self.reverse_bias_V <= 0:
            return self._cjo
        return self._cjo / (1 + self.reverse_bias_V / self._vj) ** self._m
    
    @property
    def dark_current(self) -> float:
        T = self.temperature_K
        factor = 2 ** ((T - 300) / 10)
        return self._dark_current_nA * 1e-9 * factor
    
    @property
    def shunt_resistance(self) -> float:
        return self._shunt_resistance
    
    @property
    def series_resistance(self) -> float:
        return self._series_resistance
    
    def get_parameters(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'type': 'photodiode',
            'material': 'Si',
            'active_area_m2': self.active_area_m2,
            'responsivity': self.responsivity,
            'capacitance': self.capacitance,
            'dark_current': self.dark_current,
            'shunt_resistance': self.shunt_resistance,
            'series_resistance': self.series_resistance,
            'bandwidth_1kohm': self.bandwidth(1000),
            'temperature_K': self.temperature_K,
        }
    
    def __repr__(self):
        return f"SFH206K(R={self.responsivity:.2f}A/W, C={self.capacitance*1e12:.0f}pF)"


# =============================================================================
# VEMD5510 - VLC-OPTIMIZED PHOTODIODE
# =============================================================================

class VEMD5510(PhotodetectorBase):
    """
    Vishay VEMD5510 VLC-Optimized PIN Photodiode
    
    Specifically designed for visible light communication.
    
    Datasheet Parameters:
        - Active area: 0.23 mm²
        - Spectral range: 440-700 nm
        - Peak wavelength: 570 nm
        - Responsivity: 0.28 A/W @ 570nm
        - Capacitance: 4.4 pF @ 0V
        - Rise time: 1.5 ns
    """
    
    def __init__(self,
                 temperature_K: float = 300.0,
                 reverse_bias_V: float = 0.0):
        super().__init__(temperature_K, reverse_bias_V)
        self._material = SILICON
        
        self._active_area_mm2 = 0.23
        self._responsivity_570nm = 0.28
        self._peak_wavelength_nm = 570
        
        self._cjo = 4.4e-12
        self._vj = 0.8
        self._m = 0.33
        self._tt = 5e-9
        
        self._dark_current_nA = 0.5
        self._series_resistance = 10.0
        self._shunt_resistance = 100e9
    
    @property
    def name(self) -> str:
        return "VEMD5510"
    
    @property
    def active_area_m2(self) -> float:
        return self._active_area_mm2 * 1e-6
    
    @property
    def responsivity(self) -> float:
        return self._responsivity_570nm
    
    @property
    def capacitance(self) -> float:
        if self.reverse_bias_V <= 0:
            return self._cjo
        return self._cjo / (1 + self.reverse_bias_V / self._vj) ** self._m
    
    @property
    def dark_current(self) -> float:
        T = self.temperature_K
        factor = 2 ** ((T - 300) / 10)
        return self._dark_current_nA * 1e-9 * factor
    
    @property
    def shunt_resistance(self) -> float:
        return self._shunt_resistance
    
    @property
    def series_resistance(self) -> float:
        return self._series_resistance
    
    def get_parameters(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'type': 'photodiode',
            'material': 'Si',
            'active_area_m2': self.active_area_m2,
            'responsivity': self.responsivity,
            'responsivity_peak_nm': self._peak_wavelength_nm,
            'capacitance': self.capacitance,
            'dark_current': self.dark_current,
            'shunt_resistance': self.shunt_resistance,
            'series_resistance': self.series_resistance,
            'bandwidth_1kohm': self.bandwidth(1000),
            'bandwidth_50ohm': self.bandwidth(50),
            'temperature_K': self.temperature_K,
        }
    
    def __repr__(self):
        return f"VEMD5510(R={self.responsivity:.2f}A/W, C={self.capacitance*1e12:.1f}pF)"


# =============================================================================
# FACTORY FROM SPICE MODEL
# =============================================================================

def PhotodiodeFromSPICE(spice_model, 
                        active_area_mm2: float = 1.0,
                        responsivity: float = 0.5) -> PhotodetectorBase:
    """
    Create photodiode component from SPICE DiodeModel.
    
    Args:
        spice_model: DiodeModel from spice_parser
        active_area_mm2: Active area in mm² (not in SPICE model)
        responsivity: Peak responsivity in A/W (not in SPICE model)
        
    Returns:
        PhotodetectorBase instance
    """
    
    class _SPICEPhotodiode(PhotodetectorBase):
        def __init__(self):
            super().__init__()
            self._spice = spice_model
            self._area = active_area_mm2
            self._R = responsivity
        
        @property
        def name(self) -> str:
            return self._spice.name
        
        @property
        def active_area_m2(self) -> float:
            return self._area * 1e-6
        
        @property
        def responsivity(self) -> float:
            return self._R
        
        @property
        def capacitance(self) -> float:
            return self._spice.capacitance(self.reverse_bias_V)
        
        @property
        def dark_current(self) -> float:
            return self._spice.dark_current(self.temperature_K)
        
        @property
        def shunt_resistance(self) -> float:
            return 1e9  # Assume high for photodiode
        
        @property
        def series_resistance(self) -> float:
            return self._spice.RS
        
        def get_parameters(self) -> Dict[str, Any]:
            return {
                'name': self.name,
                'type': 'photodiode',
                'source': 'SPICE',
                'active_area_m2': self.active_area_m2,
                'responsivity': self.responsivity,
                'capacitance': self.capacitance,
                'dark_current': self.dark_current,
                'series_resistance': self.series_resistance,
                'bandwidth_1kohm': self.bandwidth(1000),
            }
        
        def __repr__(self):
            return f"PhotodiodeFromSPICE('{self.name}', C={self.capacitance*1e12:.0f}pF)"
    
    return _SPICEPhotodiode()


# =============================================================================
# SELF-TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("PHOTODIODE COMPONENTS - SELF TEST")
    print("=" * 60)
    
    photodiodes = [BPW34(), SFH206K(), VEMD5510()]
    
    for pd in photodiodes:
        print(f"\n--- {pd.name} ---")
        params = pd.get_parameters()
        print(f"  Area:        {params['active_area_m2']*1e6:.2f} mm²")
        print(f"  R:           {params['responsivity']:.2f} A/W")
        print(f"  C @ 0V:      {params['capacitance']*1e12:.1f} pF")
        print(f"  I_dark:      {params['dark_current']*1e9:.1f} nA")
        print(f"  BW @ 1kΩ:    {params['bandwidth_1kohm']/1e6:.2f} MHz")
    
    # Test bias-dependent capacitance
    print("\n--- Bias-Dependent Capacitance ---")
    for V in [0, 5, 10, 20]:
        bpw = BPW34(reverse_bias_V=V)
        print(f"  BPW34 @ {V}V: C = {bpw.capacitance*1e12:.1f} pF")
    
    print("\n[OK] All photodiode tests passed!")
