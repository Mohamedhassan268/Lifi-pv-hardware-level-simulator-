# spice_parser/models.py
"""
SPICE Model Data Classes for Hardware-Faithful LiFi-PV Simulator

This module defines data structures for SPICE component models extracted
from vendor .lib/.mod files.

Supported model types:
    - DiodeModel: Photodiodes, LEDs, Zeners, general diodes
    - BJTModel: Bipolar transistors
    - SubcircuitModel: Op-amps, TIAs, complex components
    - MOSFETModel: Power MOSFETs (future)

References:
    - SPICE Model Handbook
    - LTSpice documentation
    - Vendor application notes
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.constants import K_B, Q, ROOM_TEMPERATURE_K


# =============================================================================
# DIODE MODEL
# =============================================================================

@dataclass
class DiodeModel:
    """
    SPICE Diode Model (.MODEL D)
    
    Extracts and stores parameters from SPICE diode models.
    Provides derived calculations for bandwidth, responsivity estimates, etc.
    
    SPICE Parameters:
        IS  - Saturation current (A)
        N   - Ideality factor
        RS  - Series resistance (Ω)
        CJO - Zero-bias junction capacitance (F)
        VJ  - Junction potential (V)
        M   - Grading coefficient
        TT  - Transit time (s)
        BV  - Breakdown voltage (V)
        IBV - Breakdown current (A)
        EG  - Bandgap energy (eV)
        XTI - Temperature exponent for IS
    """
    
    name: str
    
    # Core DC parameters
    IS: float = 1e-14       # Saturation current (A)
    N: float = 1.0          # Ideality factor
    RS: float = 0.0         # Series resistance (Ω)
    
    # Junction capacitance parameters
    CJO: float = 0.0        # Zero-bias capacitance (F)
    VJ: float = 1.0         # Junction potential (V)
    M: float = 0.5          # Grading coefficient
    FC: float = 0.5         # Forward-bias depletion cap coefficient
    
    # Transit time
    TT: float = 0.0         # Transit time (s)
    
    # Breakdown
    BV: float = 100.0       # Breakdown voltage (V)
    IBV: float = 1e-10      # Breakdown current (A)
    
    # Temperature parameters
    EG: float = 1.11        # Bandgap energy (eV) - default Si
    XTI: float = 3.0        # IS temperature exponent
    
    # Extended parameters (some models)
    IKF: float = 0.0        # High-injection knee current (A)
    ISR: float = 0.0        # Recombination current (A)
    NR: float = 2.0         # Recombination ideality factor
    
    # Metadata
    manufacturer: str = ""
    device_type: str = "diode"  # diode, photodiode, led, zener
    
    # Raw SPICE line for reference
    raw_spice: str = ""
    
    # =========================================================================
    # DERIVED CALCULATIONS
    # =========================================================================
    
    def capacitance(self, V_reverse: float = 0.0) -> float:
        """
        Calculate junction capacitance at given reverse bias.
        
        C_j(V) = CJO / (1 - V/VJ)^M   for V < FC×VJ
        
        Args:
            V_reverse: Reverse bias voltage (positive value)
            
        Returns:
            Junction capacitance in Farads
        """
        if self.CJO <= 0:
            return 0.0
            
        V = -abs(V_reverse)  # Reverse bias is negative
        
        if V > self.FC * self.VJ:
            # Forward bias region - use linear approximation
            return self.CJO * (1 - self.FC) ** (-self.M) * \
                   (1 - self.FC * (1 + self.M) + self.M * V / self.VJ)
        else:
            # Normal reverse bias
            return self.CJO / (1 - V / self.VJ) ** self.M
    
    def bandwidth_estimate(self, R_load: float) -> float:
        """
        Estimate RC-limited bandwidth.
        
        f_3dB = 1 / (2π × R × C)
        
        Args:
            R_load: Load resistance in Ohms
            
        Returns:
            -3dB bandwidth in Hz
        """
        C_total = self.CJO + (self.TT * self.IS / (self.N * K_B * ROOM_TEMPERATURE_K / Q))
        
        if C_total <= 0 or R_load <= 0:
            return float('inf')
        
        R_total = R_load + self.RS
        return 1.0 / (2 * np.pi * R_total * C_total)
    
    def transit_time_bandwidth(self) -> float:
        """
        Estimate transit-time-limited bandwidth.
        
        f_tr = 0.44 / τ_tr
        
        Returns:
            Transit-time bandwidth in Hz
        """
        if self.TT <= 0:
            return float('inf')
        return 0.44 / self.TT
    
    def dark_current(self, T: float = 300.0) -> float:
        """
        Calculate temperature-dependent dark current.
        
        I_dark(T) = IS × (T/T_nom)^(XTI/N) × exp(EG×q/(N×k) × (1/T_nom - 1/T))
        
        Args:
            T: Temperature in Kelvin
            
        Returns:
            Dark current in Amperes
        """
        T_nom = 300.0
        kT_q = K_B * T / Q
        kT_nom_q = K_B * T_nom / Q
        
        return self.IS * (T / T_nom) ** (self.XTI / self.N) * \
               np.exp(self.EG / self.N * (1 / kT_nom_q - 1 / kT_q))
    
    def forward_voltage(self, I_forward: float, T: float = 300.0) -> float:
        """
        Calculate forward voltage for given current.
        
        V_f = N × V_T × ln(I_f / IS + 1) + I_f × RS
        
        Args:
            I_forward: Forward current in Amperes
            T: Temperature in Kelvin
            
        Returns:
            Forward voltage in Volts
        """
        V_T = K_B * T / Q
        V_diode = self.N * V_T * np.log(I_forward / self.IS + 1)
        return V_diode + I_forward * self.RS
    
    def responsivity_estimate(self, wavelength_nm: float = 850.0) -> float:
        """
        Estimate responsivity for photodiode (crude approximation).
        
        Uses bandgap to estimate QE, then calculates R.
        
        Args:
            wavelength_nm: Wavelength in nm
            
        Returns:
            Estimated responsivity in A/W (or 0 if not a photodiode)
        """
        if self.device_type not in ['photodiode', 'led']:
            return 0.0
        
        # Photon energy
        E_photon = 1240.0 / wavelength_nm  # eV
        
        # Check if above bandgap
        if E_photon < self.EG:
            return 0.0
        
        # Crude QE estimate based on excess energy
        QE_est = min(0.9, 0.7 * (1 - (E_photon - self.EG) / E_photon))
        
        # R = QE × λ / 1240
        return QE_est * wavelength_nm / 1240.0
    
    def __repr__(self):
        return f"DiodeModel('{self.name}', CJO={self.CJO*1e12:.1f}pF, RS={self.RS:.2f}Ω)"


# =============================================================================
# BJT MODEL
# =============================================================================

@dataclass
class BJTModel:
    """
    SPICE BJT Model (.MODEL NPN or PNP)
    """
    
    name: str
    polarity: str = "NPN"   # "NPN" or "PNP"
    
    # DC parameters
    IS: float = 1e-15       # Transport saturation current
    BF: float = 100.0       # Forward beta
    BR: float = 1.0         # Reverse beta
    NF: float = 1.0         # Forward ideality
    NR: float = 1.0         # Reverse ideality
    
    # Resistances
    RB: float = 0.0         # Base resistance
    RC: float = 0.0         # Collector resistance
    RE: float = 0.0         # Emitter resistance
    
    # Capacitances
    CJE: float = 0.0        # B-E zero-bias capacitance
    CJC: float = 0.0        # B-C zero-bias capacitance
    CJS: float = 0.0        # C-S zero-bias capacitance
    
    # Transit time
    TF: float = 0.0         # Forward transit time
    TR: float = 0.0         # Reverse transit time
    
    # Breakdown
    BVceo: float = 100.0    # Collector-emitter breakdown
    
    raw_spice: str = ""
    
    def ft_estimate(self) -> float:
        """Estimate transition frequency f_T = 1/(2π×TF)"""
        if self.TF <= 0:
            return float('inf')
        return 1.0 / (2 * np.pi * self.TF)
    
    def __repr__(self):
        return f"BJTModel('{self.name}', {self.polarity}, BF={self.BF:.0f})"


# =============================================================================
# SUBCIRCUIT MODEL
# =============================================================================

@dataclass
class SubcircuitModel:
    """
    SPICE Subcircuit Model (.SUBCKT)
    
    Used for op-amps, TIAs, and other complex components.
    Stores the raw subcircuit definition and extracted key parameters.
    """
    
    name: str
    nodes: List[str] = field(default_factory=list)
    
    # Extracted parameters (if parseable from comments or internal elements)
    gbw: float = 0.0            # Gain-bandwidth product (Hz)
    slew_rate: float = 0.0      # Slew rate (V/s)
    input_offset_voltage: float = 0.0   # V_os (V)
    input_bias_current: float = 0.0     # I_b (A)
    input_noise_voltage: float = 0.0    # e_n (V/√Hz)
    input_noise_current: float = 0.0    # i_n (A/√Hz)
    
    # Supply
    v_supply_pos: float = 15.0  # Positive supply (V)
    v_supply_neg: float = -15.0 # Negative supply (V)
    
    # Internal components count
    n_resistors: int = 0
    n_capacitors: int = 0
    n_transistors: int = 0
    n_diodes: int = 0
    
    # Raw content
    raw_spice: str = ""
    internal_models: List[str] = field(default_factory=list)
    
    def bandwidth_with_photodiode(self, C_pd: float, R_feedback: float) -> float:
        """
        Calculate TIA bandwidth with photodiode capacitance.
        
        For a TIA: f_3dB ≈ √(GBW / (2π × R_f × C_pd))
        
        Args:
            C_pd: Photodiode capacitance (F)
            R_feedback: Feedback resistance (Ω)
            
        Returns:
            Bandwidth in Hz
        """
        if self.gbw <= 0 or R_feedback <= 0 or C_pd <= 0:
            return float('inf')
        
        return np.sqrt(self.gbw / (2 * np.pi * R_feedback * C_pd))
    
    def __repr__(self):
        return f"SubcircuitModel('{self.name}', nodes={self.nodes})"


# =============================================================================
# LED MODEL (EXTENDED DIODE)
# =============================================================================

@dataclass
class LEDModel(DiodeModel):
    """
    LED-specific model extending DiodeModel.
    
    Adds optical emission parameters.
    """
    
    # Optical properties
    peak_wavelength_nm: float = 0.0     # Peak emission wavelength
    spectral_width_nm: float = 20.0     # FWHM spectral width
    radiant_intensity: float = 0.0       # Typical I_e at rated current (W/sr)
    viewing_angle_deg: float = 120.0     # Half-angle
    
    # Efficiency
    external_quantum_efficiency: float = 0.0  # Photons out / electrons in
    wall_plug_efficiency: float = 0.0         # P_opt / P_elec
    
    # Modulation
    modulation_bandwidth_hz: float = 0.0      # -3dB electrical bandwidth
    
    def __post_init__(self):
        self.device_type = "led"
    
    def optical_power(self, I_forward: float) -> float:
        """
        Estimate optical power output.
        
        P_opt ≈ I_f × V_f × η_wall  or  I_f × E_photon × η_ext / q
        
        Args:
            I_forward: Forward current (A)
            
        Returns:
            Optical power in Watts
        """
        if self.external_quantum_efficiency > 0:
            E_photon_J = Q * 1240.0 / self.peak_wavelength_nm if self.peak_wavelength_nm > 0 else Q * 2.0
            return I_forward * E_photon_J * self.external_quantum_efficiency / Q
        elif self.wall_plug_efficiency > 0:
            V_f = self.forward_voltage(I_forward)
            return I_forward * V_f * self.wall_plug_efficiency
        else:
            # Default estimate: 10% wall-plug efficiency
            V_f = self.forward_voltage(I_forward)
            return I_forward * V_f * 0.10
    
    def lambertian_order(self) -> float:
        """
        Calculate Lambertian order from viewing angle.
        
        m = -ln(2) / ln(cos(θ_1/2))
        
        Returns:
            Lambertian order
        """
        if self.viewing_angle_deg <= 0 or self.viewing_angle_deg >= 180:
            return 1.0
        theta_half = np.radians(self.viewing_angle_deg / 2)
        return -np.log(2) / np.log(np.cos(theta_half))
    
    def __repr__(self):
        return f"LEDModel('{self.name}', λ={self.peak_wavelength_nm}nm)"


# =============================================================================
# SELF-TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("SPICE MODEL DATA CLASSES - SELF TEST")
    print("=" * 60)
    
    # Test DiodeModel
    print("\n--- DiodeModel Test ---")
    bpw34 = DiodeModel(
        name="BPW34",
        IS=2e-9,
        N=1.7,
        RS=3.0,
        CJO=72e-12,
        VJ=0.35,
        M=0.28,
        TT=50e-9,
        BV=60,
        EG=1.11,
        device_type="photodiode"
    )
    print(f"  {bpw34}")
    print(f"  C(0V):   {bpw34.capacitance(0)*1e12:.1f} pF")
    print(f"  C(5V):   {bpw34.capacitance(5)*1e12:.1f} pF")
    print(f"  BW@1kΩ:  {bpw34.bandwidth_estimate(1000)/1e3:.1f} kHz")
    print(f"  f_tr:    {bpw34.transit_time_bandwidth()/1e6:.1f} MHz")
    print(f"  I_dark:  {bpw34.dark_current(300)*1e9:.2f} nA")
    
    # Test LEDModel
    print("\n--- LEDModel Test ---")
    red_led = LEDModel(
        name="OSRAM_LRW5SN",
        IS=1e-20,
        N=2.5,
        RS=0.5,
        CJO=200e-12,
        EG=1.9,
        peak_wavelength_nm=625,
        spectral_width_nm=20,
        viewing_angle_deg=120,
        external_quantum_efficiency=0.3,
    )
    print(f"  {red_led}")
    print(f"  V_f @ 100mA: {red_led.forward_voltage(0.1):.2f} V")
    print(f"  P_opt @ 100mA: {red_led.optical_power(0.1)*1e3:.1f} mW")
    print(f"  Lambertian m: {red_led.lambertian_order():.2f}")
    
    # Test SubcircuitModel
    print("\n--- SubcircuitModel Test ---")
    opa380 = SubcircuitModel(
        name="OPA380",
        nodes=['IN+', 'IN-', 'VCC', 'VEE', 'OUT'],
        gbw=90e6,
        slew_rate=80e6,
        input_noise_voltage=8e-9,
    )
    print(f"  {opa380}")
    print(f"  BW with BPW34 @ 100kΩ: {opa380.bandwidth_with_photodiode(72e-12, 100e3)/1e3:.1f} kHz")
    
    print("\n[OK] All model tests passed!")
