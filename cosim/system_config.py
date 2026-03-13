# cosim/system_config.py
"""
SystemConfig - Paper-agnostic system configuration dataclass.

Holds all parameters needed to define a TX->Channel->RX simulation:
component selections, channel geometry, filter settings, and simulation controls.

Usage:
    from cosim.system_config import SystemConfig

    cfg = SystemConfig()                          # defaults
    cfg = SystemConfig.from_preset('kadirvelu2021')  # load preset
    cfg.save('my_config.json')
    cfg = SystemConfig.load('my_config.json')
"""

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


PRESETS_DIR = Path(__file__).parent.parent / 'presets'


@dataclass
class SystemConfig:
    """Complete system configuration for a LiFi-PV simulation."""

    # -- Transmitter -----------------------------------------------------------
    led_part: str = 'LXM5-PD01'
    driver_part: str = 'ADA4891'
    bias_current_A: float = 0.350
    modulation_depth: float = 0.33
    led_radiated_power_mW: float = 9.3
    led_half_angle_deg: float = 9.0
    led_driver_re: float = 12.1
    led_gled: float = 0.88
    lens_transmittance: float = 0.85

    # -- Channel ---------------------------------------------------------------
    distance_m: float = 0.325
    tx_angle_deg: float = 0.0
    rx_tilt_deg: float = 0.0
    fov_half_angle_deg: float = 90.0      # Receiver FOV half-angle (90 = hemispherical)
    beer_lambert_enabled: bool = False     # Enable atmospheric attenuation
    n_reflections: int = 0                 # Number of wall-bounce reflections (0 = LOS only)
    room_length_m: float = 5.0            # Room dimensions for multipath
    room_width_m: float = 5.0
    room_height_m: float = 3.0
    wall_reflectivity: float = 0.7        # Diffuse wall reflectance (0-1)

    # -- Receiver --------------------------------------------------------------
    pv_part: str = 'KXOB25-04X3F'
    sc_area_cm2: float = 9.0
    sc_responsivity: float = 0.457
    sc_cj_nF: float = 798.0
    sc_rsh_kOhm: float = 138.8
    sc_iph_uA: float = 508.0
    sc_vmpp_mV: float = 740.0
    sc_impp_uA: float = 470.0
    sc_pmpp_uW: float = 347.0

    r_sense_ohm: float = 1.0
    ina_part: str = 'INA322'
    ina_gain_dB: float = 40.0
    ina_gbw_kHz: float = 700.0

    comparator_part: str = 'TLV7011'

    bpf_stages: int = 2
    bpf_f_low_Hz: float = 700.0
    bpf_f_high_Hz: float = 10000.0
    bpf_rhp: float = 100e3
    bpf_chp_pF: float = 470000.0     # 470 nF (in pF)
    bpf_rlp: float = 10e3
    bpf_clf_nF: float = 1.5           # 1.5 nF

    # -- DC-DC Converter -------------------------------------------------------
    dcdc_fsw_kHz: float = 50.0
    dcdc_l_uH: float = 22.0
    dcdc_cp_uF: float = 10.0
    dcdc_cl_uF: float = 47.0
    r_load_ohm: float = 180000.0
    vcc_volts: float = 3.3

    # -- Simulation Controls ---------------------------------------------------
    t_stop_s: float = 1e-3
    data_rate_bps: float = 5000.0
    modulation: str = 'OOK'             # OOK, OOK_Manchester, OFDM, BFSK, PWM_ASK
    prbs_order: int = 7
    n_bits: int = 100
    simulation_engine: str = 'spice'    # 'spice' or 'python'
    random_seed: Optional[int] = None    # RNG seed for reproducibility (None = random)

    # -- Noise Configuration ---------------------------------------------------
    noise_enable: bool = False
    noise_shot_enable: bool = True            # Source 1: shot noise
    noise_thermal_enable: bool = True         # Source 2: thermal noise
    noise_ambient_enable: bool = True         # Source 3: ambient light noise
    noise_amplifier_enable: bool = True       # Source 4: amplifier noise
    noise_adc_enable: bool = False            # Source 5: ADC quantization noise
    noise_processing_enable: bool = False     # Source 6: processing/threshold noise
    ina_noise_nV_rtHz: float = 45.0           # INA322 input-referred voltage noise density
    ina_noise_current_pA_rtHz: float = 0.1    # INA322 input-referred current noise density
    ambient_illuminance_lux: float = 0.0      # Background ambient light level
    comparator_offset_mV: float = 1.0         # TLV7011 input offset voltage
    comparator_jitter_ns: float = 5.0         # TLV7011 propagation delay jitter (1σ)
    adc_bits: int = 12                        # ADC resolution (0 = no ADC in chain)
    adc_vref: float = 3.3                     # ADC reference voltage

    # Legacy aliases (backward compat with old presets)
    shot_noise_enable: bool = True
    thermal_noise_enable: bool = True

    # -- Multi-Paper Extensions ------------------------------------------------
    # OFDM parameters (Sarwar 2017, Oliveira 2024)
    ofdm_nfft: int = 256
    ofdm_qam_order: int = 16
    ofdm_n_subcarriers: int = 80
    ofdm_cp_len: int = 32
    ofdm_sample_rate_hz: float = 15e6

    # BFSK parameters (Xu 2024)
    bfsk_f0_hz: float = 1600.0
    bfsk_f1_hz: float = 2000.0

    # PWM-ASK parameters (Correa 2025)
    pwm_freq_hz: float = 10.0
    carrier_freq_hz: float = 10000.0

    # Environment (Correa 2025 greenhouse)
    humidity_rh: Optional[float] = None   # relative humidity 0-1 for Beer-Lambert
    temperature_K: float = 300.0

    # Multi-cell / reconfigurable (Xu 2024)
    n_cells_series: int = 1
    n_cells_parallel: int = 1

    # Receiver chain extensions (González 2024)
    amp_gain_linear: float = 1.0          # voltage amplifier gain after INA
    notch_freq_hz: Optional[float] = None # notch filter for mains rejection
    notch_Q: float = 30.0

    # -- Phase 4: Generalized Architecture ---------------------------------------
    # RX topology: 'ina_bpf_comp' | 'amp_slicer' | 'direct' | 'auto'
    # 'auto' means auto-detect from INA/BPF/amp fields in __post_init__
    rx_topology: str = 'auto'
    dcdc_enable: Optional[bool] = None    # None = auto-detect from dcdc_fsw_kHz > 0

    # -- Phase 2: Enhanced Python Engine ----------------------------------------
    # PV Cell ODE model
    pv_ode_enable: bool = False           # Enable transient ODE solver (False = simple I=R*P)
    pv_dark_current_A: float = 1e-10      # Diode saturation current I_s
    pv_ideality_factor: float = 1.5       # Diode ideality factor n
    pv_vbi_V: float = 1.1                 # Built-in voltage for C_j(V) model
    pv_series_resistance_ohm: float = 2.5 # Series resistance Rs

    # LED TX model
    led_bandwidth_limit_enable: bool = False  # Apply LED frequency response to P_tx

    # DC-DC converter (Python model params)
    dcdc_rds_on_mohm: float = 52.0        # MOSFET on-resistance (NTS4409)
    dcdc_diode_vf_V: float = 0.3          # Schottky forward voltage
    dcdc_inductor_dcr_ohm: float = 0.5    # Inductor DC resistance

    # Receiver chain model
    rx_chain_enable: bool = False          # Enable detailed RX chain (False = simple TIA+demod)
    comparator_prop_delay_ns: float = 260.0  # TLV7011 propagation delay

    # -- Validation Targets (optional) -----------------------------------------
    target_harvested_power_uW: Optional[float] = None
    target_ber: Optional[float] = None
    target_noise_rms_mV: Optional[float] = None
    target_data_rate_mbps: Optional[float] = None
    target_fec_threshold: Optional[float] = None

    # -- Metadata --------------------------------------------------------------
    preset_name: str = ''
    paper_reference: str = ''

    # -------------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------------

    _VALID_MODULATIONS = frozenset({
        'OOK', 'OOK_Manchester', 'OFDM', 'BFSK', 'PWM_ASK',
    })
    _VALID_TOPOLOGIES = frozenset({
        'ina_bpf_comp', 'amp_slicer', 'direct', 'auto',
    })

    def __post_init__(self):
        """Validate parameter ranges and auto-detect derived fields."""
        # --- Auto-detect rx_topology ---
        if self.rx_topology == 'auto':
            has_ina = self.ina_gain_dB > 0 and self.ina_part != 'N/A'
            has_bpf = self.bpf_stages > 0
            has_amp = self.amp_gain_linear > 1 or (has_ina and not has_bpf)
            if has_ina and has_bpf:
                object.__setattr__(self, 'rx_topology', 'ina_bpf_comp')
            elif has_amp or (has_ina and not has_bpf):
                object.__setattr__(self, 'rx_topology', 'amp_slicer')
            else:
                object.__setattr__(self, 'rx_topology', 'direct')

        # --- Auto-detect dcdc_enable ---
        if self.dcdc_enable is None:
            object.__setattr__(self, 'dcdc_enable', self.dcdc_fsw_kHz > 0)

        errors = []

        if self.distance_m <= 0:
            errors.append(f"distance_m must be > 0, got {self.distance_m}")
        if self.data_rate_bps <= 0:
            errors.append(f"data_rate_bps must be > 0, got {self.data_rate_bps}")
        if self.modulation not in self._VALID_MODULATIONS:
            errors.append(
                f"modulation must be one of {sorted(self._VALID_MODULATIONS)}, "
                f"got '{self.modulation}'"
            )
        if self.rx_topology not in self._VALID_TOPOLOGIES:
            errors.append(
                f"rx_topology must be one of {sorted(self._VALID_TOPOLOGIES)}, "
                f"got '{self.rx_topology}'"
            )
        if self.bpf_f_low_Hz >= self.bpf_f_high_Hz and self.bpf_f_low_Hz > 0:
            errors.append(
                f"bpf_f_low_Hz ({self.bpf_f_low_Hz}) must be < "
                f"bpf_f_high_Hz ({self.bpf_f_high_Hz})"
            )

        # Channel validation
        if self.fov_half_angle_deg <= 0 or self.fov_half_angle_deg > 90:
            errors.append(
                f"fov_half_angle_deg must be in (0, 90], got {self.fov_half_angle_deg}"
            )
        if self.n_reflections < 0:
            errors.append(f"n_reflections must be >= 0, got {self.n_reflections}")
        if self.wall_reflectivity < 0 or self.wall_reflectivity > 1:
            errors.append(
                f"wall_reflectivity must be in [0, 1], got {self.wall_reflectivity}"
            )

        # Noise validation
        if self.ambient_illuminance_lux < 0:
            errors.append(
                f"ambient_illuminance_lux must be >= 0, got {self.ambient_illuminance_lux}"
            )

        # Physical quantities must be non-negative
        for field_name in ('sc_area_cm2', 'sc_responsivity', 'sc_cj_nF',
                           'led_radiated_power_mW', 'r_sense_ohm',
                           'bias_current_A', 'vcc_volts'):
            val = getattr(self, field_name)
            if val < 0:
                errors.append(f"{field_name} must be >= 0, got {val}")

        if errors:
            raise ValueError(
                "Invalid SystemConfig:\n  " + "\n  ".join(errors)
            )

    # -------------------------------------------------------------------------
    # Serialization
    # -------------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Convert to plain dict."""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def save(self, path) -> None:
        """Save configuration to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(), encoding='utf-8')

    @classmethod
    def load(cls, path) -> 'SystemConfig':
        """Load configuration from JSON file."""
        path = Path(path)
        data = json.loads(path.read_text(encoding='utf-8'))
        # Filter out unknown keys so old configs still load
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)

    @classmethod
    def from_preset(cls, name: str) -> 'SystemConfig':
        """
        Load a named preset from the presets/ directory.

        Args:
            name: Preset name (without .json extension)

        Returns:
            SystemConfig with preset values
        """
        path = PRESETS_DIR / f'{name}.json'
        if not path.exists():
            available = [p.stem for p in PRESETS_DIR.glob('*.json')]
            raise FileNotFoundError(
                f"Preset '{name}' not found. Available: {available}")
        return cls.load(path)

    @classmethod
    def list_presets(cls) -> list:
        """List available preset names."""
        if not PRESETS_DIR.exists():
            return []
        return sorted(p.stem for p in PRESETS_DIR.glob('*.json'))

    # -------------------------------------------------------------------------
    # Derived quantities (convenience)
    # -------------------------------------------------------------------------

    def lambertian_order(self) -> float:
        """Lambertian emission order m = -ln2 / ln(cos(alpha_half))."""
        import numpy as np
        alpha = np.radians(self.led_half_angle_deg)
        return -np.log(2) / np.log(np.cos(alpha))

    def optical_channel_gain(self) -> float:
        """DC optical channel gain H(0)."""
        import numpy as np
        m = self.lambertian_order()
        r = self.distance_m
        A = self.sc_area_cm2 * 1e-4
        theta = np.radians(self.tx_angle_deg)
        beta = np.radians(self.rx_tilt_deg)
        return (m + 1) / (2 * np.pi * r**2) * np.cos(theta)**m * np.cos(beta) * A

    def received_power_W(self) -> float:
        """Received optical power in watts."""
        P_tx = self.led_radiated_power_mW * 1e-3
        return P_tx * self.optical_channel_gain()

    def photocurrent_A(self) -> float:
        """Photocurrent at receiver in amps."""
        return self.sc_responsivity * self.received_power_W()

    def snr_estimate_dB(self) -> float:
        """Quick SNR estimate from link budget."""
        import numpy as np
        I_ph = self.photocurrent_A()
        I_signal = I_ph * self.modulation_depth
        # Shot noise + thermal noise estimate
        q = 1.602e-19
        kT = 1.38e-23 * 300
        BW = self.data_rate_bps / 2
        R_sense = self.r_sense_ohm
        noise_shot = np.sqrt(2 * q * I_ph * BW)
        noise_thermal = np.sqrt(4 * kT * BW / max(R_sense, 1e-6))
        noise_total = np.sqrt(noise_shot**2 + noise_thermal**2)
        if noise_total > 0:
            return float(20 * np.log10(I_signal / noise_total))
        return 200.0  # Cap at 200 dB (effectively noiseless)

    def __str__(self) -> str:
        name = self.preset_name or 'Custom'
        return (f"SystemConfig({name}: "
                f"LED={self.led_part}, PV={self.pv_part}, "
                f"d={self.distance_m*100:.0f}cm, "
                f"rate={self.data_rate_bps/1e3:.0f}kbps)")
