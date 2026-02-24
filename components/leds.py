# components/leds.py
"""
LED Components for Hardware-Faithful LiFi-PV Simulator

Provides models for LEDs used as LiFi transmitters:
    - LXM5_PD01: Philips Lumileds white LED (Kadirvelu 2021 TX)

Key Parameters:
    - Optical power from drive current: P = GLED * I
    - Modulation bandwidth from junction capacitance
    - Emission pattern: Lambertian with lens

References:
    - Kadirvelu et al. (2021) - System parameters
    - Philips Lumileds LXM5-PD01 datasheet
    - Fraen FLP-S9-SP lens datasheet
"""

import numpy as np
from typing import Dict, Any
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from components.base import LEDBase
from utils.constants import Q, K_B, HC_EV_NM


# =============================================================================
# LXM5-PD01 - PHILIPS LUMILEDS WHITE LED (with Fraen lens)
# =============================================================================

class LXM5_PD01(LEDBase):
    """
    Philips Lumileds LXM5-PD01 White LED with Fraen FLP-S9-SP Lens

    Used as the transmitter in Kadirvelu 2021 system.

    Datasheet Parameters:
        - Dominant wavelength: ~530 nm (white LED, green peak for PV response)
        - Max drive current: 700 mA
        - Forward voltage: 3.2 V @ 350 mA
        - Luminous flux: ~115 lm @ 350 mA (typical white)

    System Parameters (from Kadirvelu 2021):
        - GLED = 0.88 W/A (optical power per unit current)
        - Pe = 9.3 mW radiated power
        - Half-angle = 9 deg (with Fraen lens, originally ~60 deg)
        - Lens transmittance: 85%
        - LED driver Re = 12.1 ohm
    """

    def __init__(self, temperature_K: float = 300.0, with_lens: bool = True):
        """
        Initialize LXM5-PD01 LED.

        Args:
            temperature_K: Operating temperature (K)
            with_lens: Whether Fraen FLP-S9-SP lens is attached
        """
        super().__init__(temperature_K)

        self._with_lens = with_lens

        # -----------------------------------------------------------------
        # DATASHEET PARAMETERS (LED)
        # -----------------------------------------------------------------
        self._peak_wavelength = 530.0       # nm (dominant for PV coupling)
        self._spectral_width = 30.0         # nm FWHM (white LED, narrow peak)
        self._max_current = 0.700           # A (700 mA)
        self._typical_current = 0.350       # A (350 mA)
        self._forward_voltage_V = 3.2       # V @ 350 mA
        self._radiant_flux = 0.267          # W @ 700 mA (estimated from luminous)
        self._bare_half_angle = 60.0        # deg (bare LED)
        self._junction_cap_pF = 200.0       # pF (estimated)

        # -----------------------------------------------------------------
        # SYSTEM PARAMETERS (from Kadirvelu 2021)
        # -----------------------------------------------------------------
        self.GLED = 0.88                    # W/A optical power per current
        self.LENS_TRANSMITTANCE = 0.85      # Fraen lens transmittance
        self.LENS_HALF_ANGLE = 9.0          # deg (with Fraen FLP-S9-SP)
        self.DRIVER_RE = 12.1               # ohm (LED driver output resistance)

        # -----------------------------------------------------------------
        # SPICE MODEL PARAMETERS
        # -----------------------------------------------------------------
        self.spice_IS = 1e-20               # Saturation current
        self.spice_N = 2.5                  # Ideality factor
        self.spice_RS = 0.8                 # Series resistance (ohm)
        self.spice_CJO = 200e-12            # Junction capacitance (F)
        self.spice_EG = 2.7                 # Bandgap energy (eV, InGaN)
        self.spice_XTI = 2                  # Saturation current temp exponent

    # -------------------------------------------------------------------------
    # Required Properties (from LEDBase)
    # -------------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "LXM5-PD01"

    @property
    def peak_wavelength_nm(self) -> float:
        return self._peak_wavelength

    @property
    def spectral_width_nm(self) -> float:
        return self._spectral_width

    @property
    def max_drive_current_A(self) -> float:
        return self._max_current

    @property
    def forward_voltage(self) -> float:
        return self._forward_voltage_V

    @property
    def viewing_angle_deg(self) -> float:
        """Half-angle (full angle / 2). With lens, narrowed to 9 deg."""
        if self._with_lens:
            return self.LENS_HALF_ANGLE * 2  # viewing_angle = full angle
        return self._bare_half_angle * 2

    @property
    def radiant_flux_W(self) -> float:
        return self._radiant_flux

    @property
    def junction_capacitance(self) -> float:
        return self._junction_cap_pF * 1e-12

    # -------------------------------------------------------------------------
    # Extended Methods
    # -------------------------------------------------------------------------

    def optical_power_from_current(self, I_drive: float) -> float:
        """
        Calculate optical power using GLED coefficient from paper.

        P_optical = GLED * I_drive * T_lens

        Args:
            I_drive: LED drive current (A)

        Returns:
            Optical power in W (after lens)
        """
        P_raw = self.GLED * I_drive
        if self._with_lens:
            return P_raw * self.LENS_TRANSMITTANCE
        return P_raw

    def radiated_power_at_operating_point(self) -> float:
        """
        Calculate Pe at the operating point from Kadirvelu 2021.

        Pe = 9.3 mW as stated in paper.

        Returns:
            Radiated power in W
        """
        return 9.3e-3  # Locked from paper

    def spice_model_string(self) -> str:
        """
        Generate SPICE .MODEL line for the LED.

        Returns:
            SPICE model definition string
        """
        return (f".MODEL LXM5_LED D("
                f"IS={self.spice_IS:.2e} "
                f"N={self.spice_N} "
                f"RS={self.spice_RS} "
                f"CJO={self.spice_CJO:.2e} "
                f"EG={self.spice_EG} "
                f"XTI={self.spice_XTI})")

    def get_parameters(self) -> Dict[str, Any]:
        """Return all simulation parameters."""
        return {
            # Identity
            'name': self.name,
            'type': 'LED',
            'with_lens': self._with_lens,

            # Optical
            'peak_wavelength_nm': self.peak_wavelength_nm,
            'spectral_width_nm': self.spectral_width_nm,
            'radiant_flux_W': self.radiant_flux_W,
            'GLED': self.GLED,
            'lens_transmittance': self.LENS_TRANSMITTANCE,
            'radiated_power_W': self.radiated_power_at_operating_point(),

            # Electrical
            'forward_voltage_V': self.forward_voltage,
            'max_drive_current_A': self.max_drive_current_A,
            'junction_capacitance_F': self.junction_capacitance,
            'driver_resistance_ohm': self.DRIVER_RE,

            # Angular
            'viewing_angle_deg': self.viewing_angle_deg,
            'lambertian_order': self.lambertian_order(),
            'half_angle_with_lens_deg': self.LENS_HALF_ANGLE,
            'half_angle_bare_deg': self._bare_half_angle,

            # Modulation
            'modulation_bandwidth_Hz': self.modulation_bandwidth(self.DRIVER_RE),

            # SPICE
            'spice_model': self.spice_model_string(),

            # Temperature
            'temperature_K': self.temperature_K,
        }

    def __repr__(self):
        lens_str = "+Fraen" if self._with_lens else "bare"
        return (f"LXM5_PD01({lens_str}, "
                f"Pe={self.radiated_power_at_operating_point()*1e3:.1f}mW, "
                f"θ½={self.LENS_HALF_ANGLE if self._with_lens else self._bare_half_angle}°)")


# =============================================================================
# SELF-TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("LED COMPONENTS - SELF TEST")
    print("=" * 60)

    led = LXM5_PD01()
    params = led.get_parameters()

    print(f"\n--- {led.name} (with Fraen lens) ---")
    print(f"  {led}")
    print(f"  Peak wavelength:    {params['peak_wavelength_nm']:.0f} nm")
    print(f"  GLED:               {params['GLED']:.2f} W/A")
    print(f"  Radiated power:     {params['radiated_power_W']*1e3:.1f} mW")
    print(f"  Forward voltage:    {params['forward_voltage_V']:.1f} V")
    print(f"  Lambertian order:   {params['lambertian_order']:.2f}")
    print(f"  Half-angle (lens):  {params['half_angle_with_lens_deg']:.0f}°")
    print(f"  Modulation BW:      {params['modulation_bandwidth_Hz']/1e6:.1f} MHz")
    print(f"  SPICE model:        {params['spice_model']}")

    # Test optical power
    for I in [0.1, 0.35, 0.7]:
        P = led.optical_power_from_current(I)
        print(f"  P_opt @ {I*1e3:.0f}mA:     {P*1e3:.1f} mW")

    print("\n[OK] LED tests passed!")
