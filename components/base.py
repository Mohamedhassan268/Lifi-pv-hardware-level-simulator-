# components/base.py
"""
Base Component Classes for Hardware-Faithful LiFi-PV Simulator

Defines abstract base classes for all hardware components:
    - PhotodetectorBase: Photodiodes, solar cells
    - LEDBase: Light-emitting diodes
    - AmplifierBase: Op-amps, TIAs

All derived components must implement get_parameters() which returns
a dictionary of simulation-ready parameters.

Design Principle:
    Components encapsulate physics. Users select parts by name,
    and electrical parameters EMERGE from the component model.
"""

import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple, List
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.constants import Q, K_B, HC_EV_NM, ROOM_TEMPERATURE_K
from components.pins import Port, PortRole, OPAMP_PORTS


# =============================================================================
# PHOTODETECTOR BASE CLASS
# =============================================================================

class PhotodetectorBase(ABC):
    """
    Abstract base class for photodetectors (photodiodes, solar cells).
    
    Subclasses must implement:
        - get_parameters(): Return dict of simulation parameters
        
    Provides common calculations:
        - RC bandwidth
        - Responsivity from quantum efficiency
        - Noise calculations
    """
    
    def __init__(self, temperature_K: float = 300.0, reverse_bias_V: float = 0.0):
        """
        Initialize photodetector.
        
        Args:
            temperature_K: Operating temperature (K)
            reverse_bias_V: Applied reverse bias (V), positive value
        """
        self.temperature_K = temperature_K
        self.reverse_bias_V = reverse_bias_V
    
    # -------------------------------------------------------------------------
    # Abstract Methods (must be implemented by subclasses)
    # -------------------------------------------------------------------------
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Component part number/name."""
        pass
    
    @property
    @abstractmethod
    def active_area_m2(self) -> float:
        """Active area in m²."""
        pass
    
    @property
    @abstractmethod
    def responsivity(self) -> float:
        """Responsivity in A/W at peak wavelength."""
        pass
    
    @property
    @abstractmethod
    def capacitance(self) -> float:
        """Junction capacitance in Farads."""
        pass
    
    @property
    @abstractmethod
    def dark_current(self) -> float:
        """Dark current in Amperes."""
        pass
    
    @property
    @abstractmethod
    def shunt_resistance(self) -> float:
        """Shunt resistance in Ohms."""
        pass
    
    @property
    @abstractmethod
    def series_resistance(self) -> float:
        """Series resistance in Ohms."""
        pass
    
    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        """
        Return all parameters needed for simulation.
        
        Returns:
            Dict with keys:
                - 'responsivity': A/W
                - 'capacitance': F
                - 'dark_current': A
                - 'shunt_resistance': Ω
                - 'series_resistance': Ω
                - 'active_area': m²
                - 'bandwidth_3dB': Hz (at default load)
                - etc.
        """
        pass
    
    # -------------------------------------------------------------------------
    # Common Calculations
    # -------------------------------------------------------------------------
    
    def bandwidth(self, R_load: float) -> float:
        """
        Calculate RC-limited bandwidth.
        
        f_3dB = 1 / (2π × R_total × C)
        
        Args:
            R_load: Load resistance in Ohms
            
        Returns:
            -3dB bandwidth in Hz
        """
        R_total = R_load + self.series_resistance
        C = self.capacitance
        
        if R_total <= 0 or C <= 0:
            return float('inf')
        
        return 1.0 / (2 * np.pi * R_total * C)
    
    def shot_noise_current(self, I_signal: float, bandwidth_Hz: float) -> float:
        """
        Calculate shot noise current.
        
        i_shot = √(2 × q × I × B)
        
        Args:
            I_signal: Signal current (A)
            bandwidth_Hz: Noise bandwidth (Hz)
            
        Returns:
            RMS shot noise current in A
        """
        I_total = abs(I_signal) + self.dark_current
        return np.sqrt(2 * Q * I_total * bandwidth_Hz)
    
    def thermal_noise_current(self, R_load: float, bandwidth_Hz: float) -> float:
        """
        Calculate thermal (Johnson) noise current.
        
        i_thermal = √(4 × k × T × B / R)
        
        Args:
            R_load: Load resistance (Ω)
            bandwidth_Hz: Noise bandwidth (Hz)
            
        Returns:
            RMS thermal noise current in A
        """
        R_parallel = 1 / (1/R_load + 1/self.shunt_resistance)
        return np.sqrt(4 * K_B * self.temperature_K * bandwidth_Hz / R_parallel)
    
    def snr(self, P_optical: float, R_load: float, bandwidth_Hz: float) -> float:
        """
        Calculate signal-to-noise ratio.
        
        Args:
            P_optical: Received optical power (W)
            R_load: Load resistance (Ω)
            bandwidth_Hz: Noise bandwidth (Hz)
            
        Returns:
            SNR in dB
        """
        I_signal = P_optical * self.responsivity
        
        i_shot = self.shot_noise_current(I_signal, bandwidth_Hz)
        i_thermal = self.thermal_noise_current(R_load, bandwidth_Hz)
        
        noise_total = np.sqrt(i_shot**2 + i_thermal**2)
        
        if noise_total <= 0:
            return float('inf')
        
        return 20 * np.log10(I_signal / noise_total)
    
    def photocurrent(self, P_optical: float) -> float:
        """
        Calculate photocurrent for given optical power.

        I_ph = R × P_opt

        Args:
            P_optical: Optical power in Watts

        Returns:
            Photocurrent in Amperes
        """
        return self.responsivity * P_optical

    def spice_ports(self) -> List[Port]:
        """SPICE port list: anode, cathode, and a behavioral optical input
        (a voltage node whose value, in W, drives the photocurrent source)."""
        return [
            Port("anode", PortRole.SIGNAL_OUT),
            Port("cathode", PortRole.SIGNAL),
            Port("photo_in", PortRole.OPTICAL_IN),
        ]

    def spice_subcircuit(self) -> str:
        """Behavioral photodetector subcircuit (Iph || Cj || Rsh || diode, Rs
        in series), parameterised from this part's datasheet values. Node order
        matches spice_ports(): anode cathode photo_in.

        Gph converts optical power on `photo_in` (V == W) into photocurrent via
        the responsivity; the diode/Cj/Rsh model the junction. This is the same
        topology the pipeline uses for the solar cell, generalised to any
        photodetector part.
        """
        name = self.name.upper().replace('-', '_')
        R_lambda = self.responsivity
        Cj = self.capacitance
        Rsh = self.shunt_resistance
        Rs = self.series_resistance
        return f"""\
* {self.name} photodetector - behavioral model
.SUBCKT {name} anode cathode photo_in
Gph cathode anode_int VALUE = {{V(photo_in) * {R_lambda}}}
Rs anode_int anode {Rs}
Cj anode_int cathode {Cj:.6e}
Rsh anode_int cathode {Rsh:.1f}
D1 anode_int cathode {name}_D
.MODEL {name}_D D(IS=1e-10 N=1.5 RS=0.01)
.ENDS {name}"""


# =============================================================================
# LED BASE CLASS
# =============================================================================

class LEDBase(ABC):
    """
    Abstract base class for LEDs.
    
    Provides common calculations:
        - Optical power from drive current
        - Modulation bandwidth
        - Lambertian emission pattern
    """
    
    def __init__(self, temperature_K: float = 300.0):
        """
        Initialize LED.
        
        Args:
            temperature_K: Operating temperature (K)
        """
        self.temperature_K = temperature_K
    
    # -------------------------------------------------------------------------
    # Abstract Methods
    # -------------------------------------------------------------------------
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Component part number/name."""
        pass
    
    @property
    @abstractmethod
    def peak_wavelength_nm(self) -> float:
        """Peak emission wavelength in nm."""
        pass
    
    @property
    @abstractmethod
    def spectral_width_nm(self) -> float:
        """FWHM spectral width in nm."""
        pass
    
    @property
    @abstractmethod
    def max_drive_current_A(self) -> float:
        """Maximum continuous drive current in A."""
        pass
    
    @property
    @abstractmethod
    def forward_voltage(self) -> float:
        """Forward voltage at rated current in V."""
        pass
    
    @property
    @abstractmethod
    def viewing_angle_deg(self) -> float:
        """Half-angle viewing angle in degrees."""
        pass
    
    @property
    @abstractmethod
    def radiant_flux_W(self) -> float:
        """Radiant flux at rated current in W."""
        pass
    
    @property
    @abstractmethod
    def junction_capacitance(self) -> float:
        """Junction capacitance in F."""
        pass
    
    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        """Return all simulation parameters."""
        pass
    
    # -------------------------------------------------------------------------
    # Common Calculations
    # -------------------------------------------------------------------------
    
    def lambertian_order(self) -> float:
        """
        Calculate Lambertian order from viewing angle.
        
        m = -ln(2) / ln(cos(θ_1/2))
        
        Returns:
            Lambertian order
        """
        theta_half = np.radians(self.viewing_angle_deg / 2)
        if np.cos(theta_half) <= 0:
            return 1.0
        return -np.log(2) / np.log(np.cos(theta_half))
    
    def optical_power(self, I_drive: float) -> float:
        """
        Calculate optical power output for given drive current.
        
        Assumes linear relationship (valid for small signals).
        
        Args:
            I_drive: Drive current in A
            
        Returns:
            Optical power in W
        """
        # Linear approximation from rated power
        I_rated = self.max_drive_current_A
        P_rated = self.radiant_flux_W
        
        return P_rated * (I_drive / I_rated)
    
    def modulation_bandwidth(self, R_drive: float = 0.0) -> float:
        """
        Estimate modulation bandwidth.
        
        For standard LEDs: f_3dB ≈ 1/(2π×τ_carrier) ≈ few MHz
        For RC-LEDs: up to 100s of MHz
        
        Args:
            R_drive: Driver output resistance (Ω)
            
        Returns:
            -3dB modulation bandwidth in Hz
        """
        C = self.junction_capacitance
        
        if C <= 0:
            return 10e6  # Default 10 MHz
        
        # RC limit from junction capacitance
        if R_drive > 0:
            f_rc = 1 / (2 * np.pi * R_drive * C)
        else:
            f_rc = float('inf')
        
        # Carrier recombination limit (typical ~10ns for InGaN)
        tau_carrier = 10e-9  # 10 ns typical
        f_carrier = 1 / (2 * np.pi * tau_carrier)
        
        # Return lower of the two limits
        return min(f_rc, f_carrier)
    
    def efficiency(self, I_drive: float) -> float:
        """
        Calculate wall-plug efficiency.
        
        η = P_optical / P_electrical
        
        Args:
            I_drive: Drive current in A
            
        Returns:
            Efficiency (0 to 1)
        """
        P_opt = self.optical_power(I_drive)
        P_elec = I_drive * self.forward_voltage

        if P_elec <= 0:
            return 0.0

        return P_opt / P_elec

    def spice_ports(self) -> List[Port]:
        """SPICE port list for an LED used as a 2-terminal diode."""
        return [
            Port("anode", PortRole.SIGNAL_IN),
            Port("cathode", PortRole.SIGNAL),
        ]

    def spice_subcircuit(self) -> str:
        """Wrap the LED's diode .MODEL in a 2-pin subcircuit (anode cathode),
        so it can be placed and wired like any other part. Uses the part's
        ``spice_model_string()`` when available; otherwise a generic diode.
        """
        name = self.name.upper().replace('-', '_')
        if hasattr(self, 'spice_model_string'):
            model = self.spice_model_string()
            # Re-key the .MODEL name to <NAME>_D for a stable internal ref.
            model_name = f"{name}_D"
            # spice_model_string emits ".MODEL <something> D(...)"; swap the name.
            after = model.split('D(', 1)
            model_line = f".MODEL {model_name} D({after[1]}" if len(after) == 2 else \
                f".MODEL {model_name} D(IS=1e-18 N=2.5 RS=1)"
        else:
            model_name = f"{name}_D"
            model_line = f".MODEL {model_name} D(IS=1e-18 N=2.5 RS=1)"
        return f"""\
* {self.name} LED - diode model wrapped as 2-pin subcircuit
.SUBCKT {name} anode cathode
D1 anode cathode {model_name}
{model_line}
.ENDS {name}"""


# =============================================================================
# AMPLIFIER BASE CLASS
# =============================================================================

class AmplifierBase(ABC):
    """
    Abstract base class for amplifiers (op-amps, TIAs).
    
    Provides common calculations:
        - Bandwidth with photodiode
        - Noise analysis
        - Gain calculations
    """
    
    def __init__(self, R_feedback: float = 10e3):
        """
        Initialize amplifier.
        
        Args:
            R_feedback: Feedback resistance for TIA configuration (Ω)
        """
        self.R_feedback = R_feedback
    
    # -------------------------------------------------------------------------
    # Abstract Methods
    # -------------------------------------------------------------------------
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Component part number/name."""
        pass
    
    @property
    @abstractmethod
    def gain_bandwidth_product(self) -> float:
        """GBW in Hz."""
        pass
    
    @property
    @abstractmethod
    def slew_rate(self) -> float:
        """Slew rate in V/s."""
        pass
    
    @property
    @abstractmethod
    def input_noise_voltage(self) -> float:
        """Input voltage noise density in V/√Hz."""
        pass
    
    @property
    @abstractmethod
    def input_noise_current(self) -> float:
        """Input current noise density in A/√Hz."""
        pass
    
    @property
    @abstractmethod
    def input_bias_current(self) -> float:
        """Input bias current in A."""
        pass
    
    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        """Return all simulation parameters."""
        pass

    def spice_ports(self) -> List[Port]:
        """Structured SPICE port list (name + role), in subcircuit order.

        Default is a single-supply op-amp: INP INN OUT VCC VEE. Parts whose
        ``spice_subcircuit()`` exposes extra ports (e.g. an instrumentation
        amplifier's REF) override this.
        """
        return list(OPAMP_PORTS)

    # -------------------------------------------------------------------------
    # Common Calculations
    # -------------------------------------------------------------------------

    def tia_bandwidth(self, C_photodiode: float) -> float:
        """
        Calculate TIA bandwidth with photodiode.
        
        For a TIA: f_3dB ≈ √(GBW / (2π × R_f × C_pd))
        
        Args:
            C_photodiode: Photodiode capacitance (F)
            
        Returns:
            -3dB bandwidth in Hz
        """
        GBW = self.gain_bandwidth_product
        R_f = self.R_feedback
        
        if GBW <= 0 or R_f <= 0 or C_photodiode <= 0:
            return float('inf')
        
        return np.sqrt(GBW / (2 * np.pi * R_f * C_photodiode))
    
    def transimpedance_gain(self) -> float:
        """
        Return transimpedance gain.
        
        For TIA: Z_T = R_f
        
        Returns:
            Transimpedance in V/A (= Ω)
        """
        return self.R_feedback
    
    def output_noise_voltage(self, C_photodiode: float, bandwidth_Hz: float) -> float:
        """
        Calculate total output noise voltage.
        
        Includes:
            - Op-amp voltage noise × noise gain
            - Op-amp current noise × R_f
            - Thermal noise of R_f
        
        Args:
            C_photodiode: Photodiode capacitance (F)
            bandwidth_Hz: Noise bandwidth (Hz)
            
        Returns:
            RMS output noise voltage in V
        """
        R_f = self.R_feedback
        e_n = self.input_noise_voltage
        i_n = self.input_noise_current
        T = ROOM_TEMPERATURE_K
        
        # Noise gain at high frequency ≈ 1 + C_pd/C_f
        # Simplified: assume noise gain ≈ 2 for typical design
        noise_gain = 2.0
        
        # Voltage noise contribution
        v_n_opamp = e_n * noise_gain * np.sqrt(bandwidth_Hz)
        
        # Current noise contribution
        v_n_current = i_n * R_f * np.sqrt(bandwidth_Hz)
        
        # Thermal noise of feedback resistor
        v_n_thermal = np.sqrt(4 * K_B * T * R_f * bandwidth_Hz)
        
        # Total (RSS)
        return np.sqrt(v_n_opamp**2 + v_n_current**2 + v_n_thermal**2)
    
    def snr_improvement(self, C_photodiode: float, R_load_passive: float) -> float:
        """
        Calculate SNR improvement vs passive load.
        
        TIA typically improves SNR by increasing bandwidth
        while maintaining similar noise floor.
        
        Args:
            C_photodiode: Photodiode capacitance (F)
            R_load_passive: Passive load resistance (Ω)
            
        Returns:
            SNR improvement in dB
        """
        # Passive bandwidth
        f_passive = 1 / (2 * np.pi * R_load_passive * C_photodiode)
        
        # TIA bandwidth
        f_tia = self.tia_bandwidth(C_photodiode)
        
        # Bandwidth improvement
        bw_ratio = f_tia / f_passive if f_passive > 0 else 1.0
        
        # SNR improvement ≈ √(BW_ratio) for shot-noise limited
        return 10 * np.log10(bw_ratio)


# =============================================================================
# SELF-TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("BASE COMPONENT CLASSES - SELF TEST")
    print("=" * 60)
    print("\nBase classes are abstract - see concrete implementations in:")
    print("  - components/photodiodes.py")
    print("  - components/solar_cells.py")
    print("  - components/leds.py")
    print("  - components/amplifiers.py")
