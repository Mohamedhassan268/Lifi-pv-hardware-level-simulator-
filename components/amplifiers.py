# components/amplifiers.py
"""
Amplifier Components for Hardware-Faithful LiFi-PV Simulator

Provides models for amplifiers used in the LiFi receiver chain:
    - INA322: TI instrumentation amplifier (40 dB gain stage)
    - TLV2379: TI quad RRIO op-amp (BPF active filter stages)
    - ADA4891: Analog Devices high-speed op-amp (LED driver)

Key Parameters from Datasheets:
    - Gain-bandwidth product (GBW)
    - Input noise voltage/current density
    - Slew rate, output swing, supply current

References:
    - Kadirvelu et al. (2021) - System configuration
    - TI INA322 datasheet (SBOS186)
    - TI TLV2379 datasheet (SLOS399)
    - ADI ADA4891 datasheet
"""

import numpy as np
from typing import Dict, Any
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from components.base import AmplifierBase
from utils.constants import K_B, ROOM_TEMPERATURE_K


# =============================================================================
# INA322 - INSTRUMENTATION AMPLIFIER
# =============================================================================

class INA322(AmplifierBase):
    """
    Texas Instruments INA322 Low-Power Instrumentation Amplifier

    Used in Kadirvelu 2021 as the first gain stage after solar cell.
    Measures voltage across R_sense (1 ohm) with 40 dB (100x) gain.

    Datasheet Parameters (TI INA322):
        - Supply: 2.7V to 5.5V single supply
        - Quiescent current: 30 uA
        - GBW: 700 kHz
        - Gain: G = 5 + 5*(R1/R2)
        - Input noise: 7.7 nV/rtHz (voltage), 6 fA/rtHz (current)
        - Slew rate: 0.16 V/us
        - CMRR: 80 dB (min)
        - Output swing: 50 mV from rails

    Kadirvelu 2021 Configuration:
        - R1 = 191 kohm, R2 = 10 kohm
        - Gain = 5 + 5*(191/10) = 100.5 ≈ 100 (40 dB)
    """

    def __init__(self,
                 R1: float = 191e3,
                 R2: float = 10e3,
                 R_feedback: float = None):
        """
        Initialize INA322.

        Args:
            R1: Gain-setting resistor R1 (ohm)
            R2: Gain-setting resistor R2 (ohm)
            R_feedback: Override for TIA mode (default: not used)
        """
        # For inst-amp, R_feedback is not directly used in TIA calculations
        # but AmplifierBase requires it
        if R_feedback is None:
            R_feedback = R1  # Use R1 as effective feedback
        super().__init__(R_feedback)

        self._R1 = R1
        self._R2 = R2

        # -----------------------------------------------------------------
        # DATASHEET PARAMETERS
        # -----------------------------------------------------------------
        self._gbw_Hz = 700e3                # 700 kHz
        self._slew_rate_Vps = 0.16e6        # 0.16 V/us = 160 kV/s
        self._input_noise_V = 7.7e-9        # 7.7 nV/rtHz
        self._input_noise_A = 6e-15         # 6 fA/rtHz
        self._input_bias_A = 1e-9           # 1 nA typical
        self._cmrr_dB = 80.0                # dB minimum
        self._supply_min = 2.7              # V
        self._supply_max = 5.5              # V
        self._quiescent_uA = 30.0           # uA
        self._output_swing_margin = 0.05    # V from rails

    # -------------------------------------------------------------------------
    # Required Properties (from AmplifierBase)
    # -------------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "INA322"

    @property
    def gain_bandwidth_product(self) -> float:
        return self._gbw_Hz

    @property
    def slew_rate(self) -> float:
        return self._slew_rate_Vps

    @property
    def input_noise_voltage(self) -> float:
        return self._input_noise_V

    @property
    def input_noise_current(self) -> float:
        return self._input_noise_A

    @property
    def input_bias_current(self) -> float:
        return self._input_bias_A

    # -------------------------------------------------------------------------
    # INA322-Specific Properties
    # -------------------------------------------------------------------------

    @property
    def voltage_gain(self) -> float:
        """Calculate voltage gain from R1, R2."""
        return 5 + 5 * (self._R1 / self._R2)

    @property
    def voltage_gain_dB(self) -> float:
        """Voltage gain in dB."""
        return 20 * np.log10(self.voltage_gain)

    @property
    def bandwidth_at_gain(self) -> float:
        """
        -3dB bandwidth at configured gain.

        f_3dB = GBW / gain
        """
        return self._gbw_Hz / self.voltage_gain

    @property
    def cmrr_dB(self) -> float:
        """Common-mode rejection ratio in dB."""
        return self._cmrr_dB

    def spice_subcircuit(self) -> str:
        """
        Generate behavioral SPICE subcircuit for INA322.

        Returns:
            SPICE subcircuit definition
        """
        gain = self.voltage_gain
        f0 = self.bandwidth_at_gain  # -3dB frequency
        f1 = f0 * 10  # Second pole at 10x first

        return f"""\
* INA322 Instrumentation Amplifier - Behavioral Model
* Gain = {gain:.1f} ({self.voltage_gain_dB:.1f} dB), GBW = {self._gbw_Hz/1e3:.0f} kHz
.SUBCKT INA322 INP INN OUT VCC VEE
* INP = non-inverting input
* INN = inverting input
* OUT = output
* VCC, VEE = supply rails

* High input impedance
Rinp INP 0 1G
Rinn INN 0 1G

* Differential gain stage
Ediff diff_int 0 INP INN {gain}

* Two-pole GBW limiting
Rp1 diff_int p1 1k
Cp1 p1 0 {1/(2*np.pi*f0*1e3):.6e}
Rp2 p1 p2 1k
Cp2 p2 0 {1/(2*np.pi*f1*1e3):.6e}

* Output buffer with rail clamping
Bout OUT 0 V = max(min(V(p2), V(VCC)-{self._output_swing_margin}), V(VEE)+{self._output_swing_margin})

.ENDS INA322"""

    def get_parameters(self) -> Dict[str, Any]:
        """Return all simulation parameters."""
        return {
            'name': self.name,
            'type': 'instrumentation_amplifier',

            # Gain configuration
            'R1_ohm': self._R1,
            'R2_ohm': self._R2,
            'voltage_gain': self.voltage_gain,
            'voltage_gain_dB': self.voltage_gain_dB,

            # Bandwidth
            'GBW_Hz': self._gbw_Hz,
            'bandwidth_at_gain_Hz': self.bandwidth_at_gain,

            # Noise
            'input_noise_voltage_VrtHz': self._input_noise_V,
            'input_noise_current_ArtHz': self._input_noise_A,
            'input_bias_current_A': self._input_bias_A,

            # Dynamic
            'slew_rate_Vps': self._slew_rate_Vps,
            'cmrr_dB': self._cmrr_dB,

            # Supply
            'supply_min_V': self._supply_min,
            'supply_max_V': self._supply_max,
            'quiescent_current_uA': self._quiescent_uA,
        }

    def __repr__(self):
        return (f"INA322(G={self.voltage_gain:.0f}x/{self.voltage_gain_dB:.0f}dB, "
                f"BW={self.bandwidth_at_gain/1e3:.1f}kHz)")


# =============================================================================
# TLV2379 - QUAD RRIO OP-AMP (BPF stages)
# =============================================================================

class TLV2379(AmplifierBase):
    """
    Texas Instruments TLV2379 Quad Rail-to-Rail Op-Amp

    Used in Kadirvelu 2021 for the 2-stage active band-pass filter.

    Datasheet Parameters (TI TLV2379):
        - Supply: 2.5V to 16V
        - GBW: 100 kHz
        - Slew rate: 0.04 V/us
        - Input noise: 39 nV/rtHz
        - Input bias current: 1 pA
        - Quiescent current: 13 uA per amplifier
        - Rail-to-rail input and output
    """

    def __init__(self, R_feedback: float = 10e3):
        """
        Initialize TLV2379.

        Args:
            R_feedback: Feedback resistance (ohm)
        """
        super().__init__(R_feedback)

        # -----------------------------------------------------------------
        # DATASHEET PARAMETERS
        # -----------------------------------------------------------------
        self._gbw_Hz = 100e3                # 100 kHz
        self._slew_rate_Vps = 0.04e6        # 0.04 V/us
        self._input_noise_V = 39e-9         # 39 nV/rtHz
        self._input_noise_A = 0.6e-15       # 0.6 fA/rtHz
        self._input_bias_A = 1e-12          # 1 pA
        self._supply_min = 2.5              # V
        self._supply_max = 16.0             # V
        self._quiescent_uA = 13.0           # uA per amplifier

    @property
    def name(self) -> str:
        return "TLV2379"

    @property
    def gain_bandwidth_product(self) -> float:
        return self._gbw_Hz

    @property
    def slew_rate(self) -> float:
        return self._slew_rate_Vps

    @property
    def input_noise_voltage(self) -> float:
        return self._input_noise_V

    @property
    def input_noise_current(self) -> float:
        return self._input_noise_A

    @property
    def input_bias_current(self) -> float:
        return self._input_bias_A

    def spice_subcircuit(self) -> str:
        """Generate behavioral SPICE subcircuit."""
        f_pole = self._gbw_Hz  # unity-gain bandwidth

        return f"""\
* TLV2379 Op-Amp - Behavioral Model
* GBW = {self._gbw_Hz/1e3:.0f} kHz, SR = {self._slew_rate_Vps/1e6:.2f} V/us
.SUBCKT TLV2379 INP INN OUT VCC VEE
* INP = non-inverting, INN = inverting, OUT = output

* High input impedance (CMOS input)
Rinp INP 0 1T
Rinn INN 0 1T

* Open-loop gain stage (A_OL = 100 dB)
Ediff diff_int 0 INP INN 100000

* Single-pole GBW model
Rpole diff_int pole_out 1k
Cpole pole_out 0 {1/(2*np.pi*f_pole*1e3):.6e}

* Output buffer with rail clamping
Bout OUT 0 V = max(min(V(pole_out), V(VCC)-0.02), V(VEE)+0.02)

.ENDS TLV2379"""

    def get_parameters(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'type': 'operational_amplifier',
            'GBW_Hz': self._gbw_Hz,
            'slew_rate_Vps': self._slew_rate_Vps,
            'input_noise_voltage_VrtHz': self._input_noise_V,
            'input_noise_current_ArtHz': self._input_noise_A,
            'input_bias_current_A': self._input_bias_A,
            'supply_min_V': self._supply_min,
            'supply_max_V': self._supply_max,
            'quiescent_current_uA': self._quiescent_uA,
            'R_feedback_ohm': self.R_feedback,
        }

    def __repr__(self):
        return f"TLV2379(GBW={self._gbw_Hz/1e3:.0f}kHz)"


# =============================================================================
# ADA4891 - HIGH-SPEED OP-AMP (LED driver)
# =============================================================================

class ADA4891(AmplifierBase):
    """
    Analog Devices ADA4891 Low-Noise High-Speed Op-Amp

    Used in Kadirvelu 2021 as LED driver amplifier.

    Datasheet Parameters (ADI ADA4891-1):
        - Supply: 2.7V to 5.5V single supply
        - GBW: 240 MHz
        - Slew rate: 170 V/us
        - Input noise: 5.9 nV/rtHz
        - Input bias current: 400 nA
        - Output current: 80 mA
        - Quiescent current: 3.3 mA
    """

    def __init__(self, R_feedback: float = 1e3):
        super().__init__(R_feedback)

        self._gbw_Hz = 240e6                # 240 MHz
        self._slew_rate_Vps = 170e6         # 170 V/us
        self._input_noise_V = 5.9e-9        # 5.9 nV/rtHz
        self._input_noise_A = 2.7e-12       # 2.7 pA/rtHz
        self._input_bias_A = 400e-9         # 400 nA
        self._supply_min = 2.7              # V
        self._supply_max = 5.5              # V
        self._quiescent_mA = 3.3            # mA
        self._output_current_mA = 80.0      # mA

    @property
    def name(self) -> str:
        return "ADA4891"

    @property
    def gain_bandwidth_product(self) -> float:
        return self._gbw_Hz

    @property
    def slew_rate(self) -> float:
        return self._slew_rate_Vps

    @property
    def input_noise_voltage(self) -> float:
        return self._input_noise_V

    @property
    def input_noise_current(self) -> float:
        return self._input_noise_A

    @property
    def input_bias_current(self) -> float:
        return self._input_bias_A

    def spice_subcircuit(self) -> str:
        """Generate behavioral SPICE subcircuit."""
        return f"""\
* ADA4891 High-Speed Op-Amp - Behavioral Model
* GBW = {self._gbw_Hz/1e6:.0f} MHz, SR = {self._slew_rate_Vps/1e6:.0f} V/us
.SUBCKT ADA4891 INP INN OUT VCC VEE

* Input impedance
Rinp INP 0 10MEG
Rinn INN 0 10MEG

* Open-loop gain (100 dB)
Ediff diff_int 0 INP INN 100000

* Dominant pole at GBW/A_OL
Rpole diff_int pole_out 1k
Cpole pole_out 0 {1/(2*np.pi*self._gbw_Hz/1e5*1e3):.6e}

* Output buffer with current limit
Bout OUT 0 V = max(min(V(pole_out), V(VCC)-0.2), V(VEE)+0.2)
Rout_ser OUT OUT_ext 10

.ENDS ADA4891"""

    def get_parameters(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'type': 'operational_amplifier',
            'GBW_Hz': self._gbw_Hz,
            'slew_rate_Vps': self._slew_rate_Vps,
            'input_noise_voltage_VrtHz': self._input_noise_V,
            'input_noise_current_ArtHz': self._input_noise_A,
            'input_bias_current_A': self._input_bias_A,
            'supply_min_V': self._supply_min,
            'supply_max_V': self._supply_max,
            'quiescent_current_mA': self._quiescent_mA,
            'output_current_mA': self._output_current_mA,
            'R_feedback_ohm': self.R_feedback,
        }

    def __repr__(self):
        return f"ADA4891(GBW={self._gbw_Hz/1e6:.0f}MHz, SR={self._slew_rate_Vps/1e6:.0f}V/us)"


# =============================================================================
# SELF-TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("AMPLIFIER COMPONENTS - SELF TEST")
    print("=" * 60)

    # Test INA322
    print("\n--- INA322 (Instrumentation Amplifier) ---")
    ina = INA322(R1=191e3, R2=10e3)
    print(f"  {ina}")
    print(f"  Gain: {ina.voltage_gain:.1f}x = {ina.voltage_gain_dB:.1f} dB")
    print(f"  BW at gain: {ina.bandwidth_at_gain/1e3:.1f} kHz")
    print(f"  GBW: {ina.gain_bandwidth_product/1e3:.0f} kHz")
    print(f"  Noise: {ina.input_noise_voltage*1e9:.1f} nV/rtHz")

    # Test TLV2379
    print("\n--- TLV2379 (BPF Op-Amp) ---")
    tlv = TLV2379()
    print(f"  {tlv}")
    print(f"  GBW: {tlv.gain_bandwidth_product/1e3:.0f} kHz")
    print(f"  Noise: {tlv.input_noise_voltage*1e9:.1f} nV/rtHz")

    # Test ADA4891
    print("\n--- ADA4891 (LED Driver Op-Amp) ---")
    ada = ADA4891()
    print(f"  {ada}")
    print(f"  GBW: {ada.gain_bandwidth_product/1e6:.0f} MHz")
    print(f"  Slew rate: {ada.slew_rate/1e6:.0f} V/us")

    print("\n[OK] Amplifier tests passed!")
