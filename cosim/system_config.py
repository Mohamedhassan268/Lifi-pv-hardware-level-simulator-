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

    # -- Simulation Controls ---------------------------------------------------
    t_stop_s: float = 1e-3
    data_rate_bps: float = 5000.0
    modulation: str = 'OOK'
    prbs_order: int = 7
    n_bits: int = 100

    # -- Noise Configuration ---------------------------------------------------
    noise_enable: bool = False
    ina_noise_nV_rtHz: float = 45.0       # INA322 input-referred voltage noise
    shot_noise_enable: bool = True         # Shot noise on photocurrent
    thermal_noise_enable: bool = True      # Thermal noise on Rsense

    # -- Validation Targets (optional) -----------------------------------------
    target_harvested_power_uW: Optional[float] = None
    target_ber: Optional[float] = None
    target_noise_rms_mV: Optional[float] = None

    # -- Metadata --------------------------------------------------------------
    preset_name: str = ''
    paper_reference: str = ''

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
        noise_thermal = np.sqrt(4 * kT * BW / (self.sc_rsh_kOhm * 1e3))
        noise_total = np.sqrt(noise_shot**2 + noise_thermal**2)
        if noise_total > 0:
            return 20 * np.log10(I_signal / noise_total)
        return float('inf')

    def __str__(self) -> str:
        name = self.preset_name or 'Custom'
        return (f"SystemConfig({name}: "
                f"LED={self.led_part}, PV={self.pv_part}, "
                f"d={self.distance_m*100:.0f}cm, "
                f"rate={self.data_rate_bps/1e3:.0f}kbps)")
