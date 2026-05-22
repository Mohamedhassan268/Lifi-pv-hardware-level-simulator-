# cosim/system_config.py
"""
SystemConfig - Paper-agnostic system configuration dataclass.

Holds all parameters needed to define a TX->Channel->RX simulation:
component selections, channel geometry, filter settings, and simulation controls.

Organized into domain sub-configs for clarity:
    TXConfig       — LED, driver, bias, modulation depth, lens
    ChannelConfig  — Distance, geometry, FOV, Beer-Lambert, multipath
    RXConfig       — PV cell, amplifier, BPF, comparator, DC-DC
    NoiseConfig    — 6-source noise enable/disable + parameters
    SimConfig      — Timing, modulation, engine, RNG
    ModulationConfig — Scheme-specific params (OFDM, BFSK, PWM-ASK)
    ValidationConfig — Paper target values + metadata

SystemConfig composes all sub-configs and exposes every field as a flat
attribute (cfg.distance_m, cfg.sc_responsivity, etc.) for full backward
compatibility.  JSON serialization is unchanged — flat dict, same keys.

Usage:
    from cosim.system_config import SystemConfig

    cfg = SystemConfig()                          # defaults
    cfg = SystemConfig.from_preset('kadirvelu2021')  # load preset
    cfg.save('my_config.json')
    cfg = SystemConfig.load('my_config.json')
"""

import json
from dataclasses import dataclass, field, fields, asdict
from pathlib import Path
from typing import Optional


PRESETS_DIR = Path(__file__).parent.parent / 'presets'


# =============================================================================
# SUB-CONFIG DATACLASSES
# =============================================================================

@dataclass
class TXConfig:
    """Transmitter parameters: LED, driver, bias, lens."""
    led_part: str = 'LXM5-PD01'
    driver_part: str = 'ADA4891'
    bias_current_A: float = 0.350
    modulation_depth: float = 0.33
    led_radiated_power_mW: float = 9.3
    led_half_angle_deg: float = 9.0
    led_driver_re: float = 12.1
    led_gled: float = 0.88
    lens_transmittance: float = 0.85

    # Phase 2: LED TX model
    led_bandwidth_limit_enable: bool = False


@dataclass
class ChannelConfig:
    """Optical channel parameters: geometry, FOV, Beer-Lambert, multipath."""
    distance_m: float = 0.325
    tx_angle_deg: float = 0.0
    rx_tilt_deg: float = 0.0
    fov_half_angle_deg: float = 90.0
    beer_lambert_enabled: bool = False
    n_reflections: int = 0
    room_length_m: float = 5.0
    room_width_m: float = 5.0
    room_height_m: float = 3.0
    wall_reflectivity: float = 0.7

    # Environment
    humidity_rh: Optional[float] = None
    temperature_K: float = 300.0


@dataclass
class RXConfig:
    """Receiver parameters: PV cell, amplifiers, BPF, comparator, DC-DC."""
    # PV cell
    pv_part: str = 'KXOB25-04X3F'
    sc_area_cm2: float = 9.0
    sc_responsivity: float = 0.457
    sc_cj_nF: float = 798.0
    sc_rsh_kOhm: float = 138.8
    sc_iph_uA: float = 508.0
    sc_vmpp_mV: float = 740.0
    sc_impp_uA: float = 470.0
    sc_pmpp_uW: float = 347.0

    # Current sense
    r_sense_ohm: float = 1.0

    # Instrumentation amplifier
    ina_part: str = 'INA322'
    ina_gain_dB: float = 40.0
    ina_gbw_kHz: float = 700.0

    # Comparator
    comparator_part: str = 'TLV7011'

    # Band-pass filter
    bpf_stages: int = 2
    bpf_f_low_Hz: float = 700.0
    bpf_f_high_Hz: float = 10000.0
    bpf_rhp: float = 100e3
    bpf_chp_pF: float = 470000.0
    bpf_rlp: float = 10e3
    bpf_clf_nF: float = 1.5

    # DC-DC converter
    dcdc_fsw_kHz: float = 50.0
    dcdc_l_uH: float = 22.0
    dcdc_cp_uF: float = 10.0
    dcdc_cl_uF: float = 47.0
    r_load_ohm: float = 180000.0
    vcc_volts: float = 3.3

    # Multi-cell / reconfigurable (Xu 2024)
    n_cells_series: int = 1
    n_cells_parallel: int = 1

    # Receiver chain extensions (González 2024)
    amp_gain_linear: float = 1.0
    notch_freq_hz: Optional[float] = None
    notch_Q: float = 30.0

    # Architecture (Phase 4)
    rx_topology: str = 'auto'      # 'ina_bpf_comp' | 'amp_slicer' | 'direct' | 'auto'
    dcdc_enable: Optional[bool] = None  # None = auto-detect from dcdc_fsw_kHz > 0

    # Phase 2: Enhanced Python Engine
    pv_ode_enable: bool = False
    pv_dark_current_A: float = 1e-10
    pv_dark_current_ref_T_K: float = 300.0
    dark_current_doubling_dT_K: float = 10.0  # Si=10K, GaAs~7K
    pv_ideality_factor: float = 1.5
    pv_vbi_V: float = 1.1
    pv_series_resistance_ohm: float = 2.5
    sc_responsivity_ref_T_K: float = 300.0
    sc_responsivity_tempco_per_K: float = 4e-4  # Si=4e-4, GaAs=6e-4

    dcdc_rds_on_mohm: float = 52.0
    dcdc_diode_vf_V: float = 0.3
    dcdc_inductor_dcr_ohm: float = 0.5

    rx_chain_enable: bool = False
    comparator_prop_delay_ns: float = 260.0


@dataclass
class NoiseConfig:
    """Noise model configuration: 6-source enable/disable + parameters."""
    noise_enable: bool = False
    noise_shot_enable: bool = True
    noise_thermal_enable: bool = True
    noise_ambient_enable: bool = True
    noise_amplifier_enable: bool = True
    noise_adc_enable: bool = False
    noise_processing_enable: bool = False

    ina_noise_nV_rtHz: float = 45.0
    ina_noise_current_pA_rtHz: float = 0.1
    ambient_illuminance_lux: float = 0.0
    comparator_offset_mV: float = 1.0
    comparator_jitter_ns: float = 5.0
    adc_bits: int = 12
    adc_vref: float = 3.3

    # Real-world non-AWGN sources (deterministic mains flicker + amplifier 1/f).
    # These are NOT in the original 6-source AWGN model — they sit on top.
    enable_mains_flicker: bool = True
    mains_flicker_freq_hz: float = 100.0   # 100 for 50-Hz mains (EU/MEA), 120 for US/JP
    mains_flicker_depth: float = 0.05      # fraction of I_ambient at the fundamental
    enable_amp_flicker: bool = True
    amp_flicker_corner_hz: float = 100.0   # TL072 ~100, INA322 ~10, OPA827 ~3

    # Legacy aliases (backward compat with old presets)
    shot_noise_enable: bool = True
    thermal_noise_enable: bool = True


@dataclass
class SimConfig:
    """Simulation control parameters: timing, modulation, engine."""
    t_stop_s: float = 1e-3
    data_rate_bps: float = 5000.0
    modulation: str = 'OOK'
    prbs_order: int = 7
    # Bumped from 100 → 10000 so the measurable BER floor is ~1e-4 instead
    # of ~1e-2 without making typical runs unbearably slow. The paper
    # validation harnesses (papers/*.py) override this where they need to
    # reproduce the original publication's bit count.
    n_bits: int = 10000
    simulation_engine: str = 'spice'
    # When True, run_step_rx executes BOTH the existing LTspice/ngspice
    # subprocess path AND the in-process PySpice+scipy path (Phase A
    # migration). Outputs from both are saved to the session so an
    # equivalence diff can be computed offline. Off by default; flip
    # per-run via the GUI or a preset for verification work.
    engine_compare: bool = False
    random_seed: Optional[int] = None

    # Glass-box observability (Phase 8). capture_probes accepts either the
    # literal string "all", a list of probe IDs (see cosim.probes), or None
    # to skip capture entirely. execution_mode controls the WebSocket runner:
    # 'continuous' (run all blocks straight through, today's behaviour) or
    # 'stepwise' (pause after each block, wait for {"type":"step"} from the
    # client before continuing).
    capture_probes: Optional[object] = None        # None | "all" | list[str]
    execution_mode: str = 'continuous'             # 'continuous' | 'stepwise'

    # Scientific-grade SNR measurement (two-run technique — see cosim/snr.py).
    # When True the pipeline runs a noise-disabled reference pass after the
    # main run and reports measured SNR at the demodulator input alongside
    # the closed-form link-budget number. Doubles per-run cost; opt-in.
    measure_snr_enable: bool = False


@dataclass
class ModulationConfig:
    """Scheme-specific modulation parameters (OFDM, BFSK, PWM-ASK)."""
    # OFDM (Sarwar 2017, Oliveira 2024)
    ofdm_nfft: int = 256
    ofdm_qam_order: int = 16
    ofdm_n_subcarriers: int = 80
    ofdm_cp_len: int = 32
    ofdm_sample_rate_hz: float = 15e6

    # BFSK (Xu 2024)
    bfsk_f0_hz: float = 1600.0
    bfsk_f1_hz: float = 2000.0

    # PWM-ASK (Correa 2025)
    pwm_freq_hz: float = 10.0
    carrier_freq_hz: float = 10000.0


@dataclass
class ValidationConfig:
    """Paper validation targets + metadata."""
    target_harvested_power_uW: Optional[float] = None
    target_ber: Optional[float] = None
    target_noise_rms_mV: Optional[float] = None
    target_data_rate_mbps: Optional[float] = None
    target_fec_threshold: Optional[float] = None

    preset_name: str = ''
    paper_reference: str = ''


# Registry of sub-config classes in composition order
_SUB_CONFIGS = (
    TXConfig, ChannelConfig, RXConfig, NoiseConfig,
    SimConfig, ModulationConfig, ValidationConfig,
)


# =============================================================================
# SYSTEM CONFIG (FACADE)
# =============================================================================

# Build the set of all valid field names from all sub-configs
_ALL_SUB_FIELDS: dict = {}  # field_name -> sub_config_class
for _cls in _SUB_CONFIGS:
    for _f in fields(_cls):
        _ALL_SUB_FIELDS[_f.name] = _cls

# Pre-built mapping: field_name -> sub-config attribute name on SystemConfig
_CLS_TO_ATTR = {
    TXConfig: 'tx',
    ChannelConfig: 'channel',
    RXConfig: 'rx',
    NoiseConfig: 'noise',
    SimConfig: 'sim',
    ModulationConfig: 'modulation_config',
    ValidationConfig: 'validation',
}
_SUB_ATTR_MAP: dict = {}  # field_name -> 'tx'|'channel'|'rx'|...
for _fname, _cls in _ALL_SUB_FIELDS.items():
    _SUB_ATTR_MAP[_fname] = _CLS_TO_ATTR[_cls]


class SystemConfig:
    """
    Complete system configuration for a LiFi-PV simulation.

    Composes domain sub-configs (TXConfig, ChannelConfig, RXConfig, etc.)
    while exposing every field as a flat attribute for backward compatibility.

    Construction supports both flat kwargs and sub-config instances:
        cfg = SystemConfig(distance_m=0.5, modulation='OFDM')   # flat kwargs
        cfg = SystemConfig(tx=TXConfig(...), rx=RXConfig(...))   # sub-configs
        cfg = SystemConfig()                                      # all defaults

    All existing code continues to work unchanged:
        cfg.distance_m          # delegated to cfg.channel.distance_m
        cfg.sc_responsivity     # delegated to cfg.rx.sc_responsivity
    """

    _VALID_MODULATIONS = frozenset({
        'OOK', 'OOK_Manchester', 'OFDM', 'BFSK', 'PWM_ASK',
    })
    _VALID_TOPOLOGIES = frozenset({
        'ina_bpf_comp', 'amp_slicer', 'direct', 'auto',
    })

    # Names of the sub-config attributes (used by __getattr__, __setattr__)
    _SUB_ATTR_NAMES = frozenset({
        'tx', 'channel', 'rx', 'noise', 'sim', 'modulation_config', 'validation',
    })

    def __init__(self, *, tx=None, channel=None, rx=None, noise=None,
                 sim=None, modulation_config=None, validation=None, **kwargs):
        """
        Initialize SystemConfig.

        Accepts sub-config instances (tx=TXConfig(...)) and/or flat kwargs
        (distance_m=0.5). Flat kwargs are routed to the appropriate sub-config.
        """
        # Route flat kwargs to sub-config buckets
        sub_kwargs = {
            'tx': {}, 'channel': {}, 'rx': {}, 'noise': {},
            'sim': {}, 'modulation_config': {}, 'validation': {},
        }
        for key, value in kwargs.items():
            if key in _ALL_SUB_FIELDS:
                sub_name = _SUB_ATTR_MAP[key]
                sub_kwargs[sub_name][key] = value
            # Unknown keys silently ignored (forward compat)

        # Construct sub-configs: explicit instance wins, then merge flat kwargs
        object.__setattr__(self, 'tx', tx or TXConfig(**sub_kwargs['tx']))
        object.__setattr__(self, 'channel', channel or ChannelConfig(**sub_kwargs['channel']))
        object.__setattr__(self, 'rx', rx or RXConfig(**sub_kwargs['rx']))
        object.__setattr__(self, 'noise', noise or NoiseConfig(**sub_kwargs['noise']))
        object.__setattr__(self, 'sim', sim or SimConfig(**sub_kwargs['sim']))
        object.__setattr__(self, 'modulation_config', modulation_config or ModulationConfig(**sub_kwargs['modulation_config']))
        object.__setattr__(self, 'validation', validation or ValidationConfig(**sub_kwargs['validation']))

        # Build routing table for __getattr__ / __setattr__
        object.__setattr__(self, '_FIELD_TO_SUB', dict(_SUB_ATTR_MAP))

        # Auto-detect and validate
        self._post_init()

    def _post_init(self):
        """Auto-detect derived fields and validate."""
        # --- Auto-detect rx_topology ---
        if self.rx.rx_topology == 'auto':
            has_ina = self.rx.ina_gain_dB > 0 and self.rx.ina_part != 'N/A'
            has_bpf = self.rx.bpf_stages > 0
            has_amp = self.rx.amp_gain_linear > 1 or (has_ina and not has_bpf)
            if has_ina and has_bpf:
                self.rx.rx_topology = 'ina_bpf_comp'
            elif has_amp or (has_ina and not has_bpf):
                self.rx.rx_topology = 'amp_slicer'
            else:
                self.rx.rx_topology = 'direct'

        # --- Auto-detect dcdc_enable ---
        if self.rx.dcdc_enable is None:
            self.rx.dcdc_enable = self.rx.dcdc_fsw_kHz > 0

        # --- Validation ---
        errors = []

        if self.channel.distance_m <= 0:
            errors.append(f"distance_m must be > 0, got {self.channel.distance_m}")
        if self.sim.data_rate_bps <= 0:
            errors.append(f"data_rate_bps must be > 0, got {self.sim.data_rate_bps}")
        if self.sim.modulation not in self._VALID_MODULATIONS:
            errors.append(
                f"modulation must be one of {sorted(self._VALID_MODULATIONS)}, "
                f"got '{self.sim.modulation}'"
            )
        if self.rx.rx_topology not in self._VALID_TOPOLOGIES:
            errors.append(
                f"rx_topology must be one of {sorted(self._VALID_TOPOLOGIES)}, "
                f"got '{self.rx.rx_topology}'"
            )
        if self.rx.bpf_f_low_Hz >= self.rx.bpf_f_high_Hz and self.rx.bpf_f_low_Hz > 0:
            errors.append(
                f"bpf_f_low_Hz ({self.rx.bpf_f_low_Hz}) must be < "
                f"bpf_f_high_Hz ({self.rx.bpf_f_high_Hz})"
            )

        # Channel validation
        if self.channel.fov_half_angle_deg <= 0 or self.channel.fov_half_angle_deg > 90:
            errors.append(
                f"fov_half_angle_deg must be in (0, 90], got {self.channel.fov_half_angle_deg}"
            )
        if self.channel.n_reflections < 0:
            errors.append(f"n_reflections must be >= 0, got {self.channel.n_reflections}")
        if self.channel.wall_reflectivity < 0 or self.channel.wall_reflectivity > 1:
            errors.append(
                f"wall_reflectivity must be in [0, 1], got {self.channel.wall_reflectivity}"
            )

        # Noise validation
        if self.noise.ambient_illuminance_lux < 0:
            errors.append(
                f"ambient_illuminance_lux must be >= 0, got {self.noise.ambient_illuminance_lux}"
            )

        # Statistical-resolution soft guard. Don't error out — papers
        # legitimately use small n_bits for fast smoke runs — but warn so
        # users don't mistake a too-small sample for a clean signal.
        target_ber = self.validation.target_ber or 0.0
        if target_ber > 0:
            import math as _math
            min_bits = int(_math.ceil(10.0 / target_ber))
            if self.sim.n_bits < min_bits:
                import logging as _logging
                _logging.getLogger(__name__).warning(
                    "n_bits=%d is too small to resolve target_ber=%.0e — "
                    "need ≥ %d bits (~10 errors expected). BER may read 0 by chance.",
                    self.sim.n_bits, target_ber, min_bits,
                )

        # Physical quantities must be non-negative
        for field_name in ('sc_area_cm2', 'sc_responsivity', 'sc_cj_nF',
                           'r_sense_ohm', 'vcc_volts'):
            val = getattr(self.rx, field_name)
            if val < 0:
                errors.append(f"{field_name} must be >= 0, got {val}")
        if self.tx.led_radiated_power_mW < 0:
            errors.append(f"led_radiated_power_mW must be >= 0, got {self.tx.led_radiated_power_mW}")
        if self.tx.bias_current_A < 0:
            errors.append(f"bias_current_A must be >= 0, got {self.tx.bias_current_A}")

        if errors:
            raise ValueError(
                "Invalid SystemConfig:\n  " + "\n  ".join(errors)
            )

    # -------------------------------------------------------------------------
    # Flat attribute access (backward compatibility)
    # -------------------------------------------------------------------------

    def __getattr__(self, name: str):
        """Delegate flat field access to the appropriate sub-config."""
        # Avoid infinite recursion during __init__ / unpickling
        if name.startswith('_') or name in ('tx', 'channel', 'rx', 'noise',
                                             'sim', 'modulation_config', 'validation'):
            raise AttributeError(name)

        # Check routing table
        routing = object.__getattribute__(self, '_FIELD_TO_SUB')
        if name in routing:
            sub_name = routing[name]
            sub = object.__getattribute__(self, sub_name)
            return getattr(sub, name)

        raise AttributeError(
            f"'SystemConfig' object has no attribute '{name}'"
        )

    def __setattr__(self, name: str, value):
        """Delegate flat field writes to the appropriate sub-config."""
        # Allow setting sub-config instances and private attrs directly
        if name in ('tx', 'channel', 'rx', 'noise', 'sim',
                    'modulation_config', 'validation', '_FIELD_TO_SUB'):
            object.__setattr__(self, name, value)
            return

        # Route to sub-config if it's a known field
        try:
            routing = object.__getattribute__(self, '_FIELD_TO_SUB')
        except AttributeError:
            # During __init__ before _FIELD_TO_SUB exists
            object.__setattr__(self, name, value)
            return

        if name in routing:
            sub_name = routing[name]
            sub = object.__getattribute__(self, sub_name)
            setattr(sub, name, value)
            return

        object.__setattr__(self, name, value)

    # -------------------------------------------------------------------------
    # Copy / replace (backward compat for dataclasses.replace())
    # -------------------------------------------------------------------------

    def replace(self, **kwargs) -> 'SystemConfig':
        """Create a copy with flat kwargs overridden.  Replaces dataclasses.replace()."""
        d = self.to_dict()
        d.update(kwargs)
        return SystemConfig._from_flat_dict(d)

    def __replace__(self, **kwargs) -> 'SystemConfig':
        """Support dataclasses.replace() protocol (Python 3.13+)."""
        return self.replace(**kwargs)

    def __repr__(self) -> str:
        name = self.validation.preset_name or 'Custom'
        return (f"SystemConfig({name}: "
                f"LED={self.tx.led_part}, PV={self.rx.pv_part}, "
                f"d={self.channel.distance_m*100:.0f}cm, "
                f"rate={self.sim.data_rate_bps/1e3:.0f}kbps)")

    def __eq__(self, other) -> bool:
        if not isinstance(other, SystemConfig):
            return NotImplemented
        return self.to_dict() == other.to_dict()

    # -------------------------------------------------------------------------
    # Serialization (flat dict — backward compatible)
    # -------------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Convert to flat dict (same format as before decomposition)."""
        result = {}
        for sub_attr in ('tx', 'channel', 'rx', 'noise', 'sim',
                         'modulation_config', 'validation'):
            sub = getattr(self, sub_attr)
            result.update(asdict(sub))
        return result

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def save(self, path) -> None:
        """Save configuration to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(), encoding='utf-8')

    @classmethod
    def _from_flat_dict(cls, data: dict) -> 'SystemConfig':
        """
        Create SystemConfig from a flat dict (JSON preset format).

        Routes each key to the appropriate sub-config based on field membership.
        Unknown keys are silently ignored for forward compatibility.
        """
        sub_kwargs = {
            'tx': {}, 'channel': {}, 'rx': {}, 'noise': {},
            'sim': {}, 'modulation_config': {}, 'validation': {},
        }
        sub_attr_names = {
            TXConfig: 'tx',
            ChannelConfig: 'channel',
            RXConfig: 'rx',
            NoiseConfig: 'noise',
            SimConfig: 'sim',
            ModulationConfig: 'modulation_config',
            ValidationConfig: 'validation',
        }

        for key, value in data.items():
            if key in _ALL_SUB_FIELDS:
                sub_name = sub_attr_names[_ALL_SUB_FIELDS[key]]
                sub_kwargs[sub_name][key] = value
            # Unknown keys silently ignored (forward compat)

        return cls(
            tx=TXConfig(**sub_kwargs['tx']),
            channel=ChannelConfig(**sub_kwargs['channel']),
            rx=RXConfig(**sub_kwargs['rx']),
            noise=NoiseConfig(**sub_kwargs['noise']),
            sim=SimConfig(**sub_kwargs['sim']),
            modulation_config=ModulationConfig(**sub_kwargs['modulation_config']),
            validation=ValidationConfig(**sub_kwargs['validation']),
        )

    @classmethod
    def load(cls, path) -> 'SystemConfig':
        """Load configuration from JSON file."""
        path = Path(path)
        data = json.loads(path.read_text(encoding='utf-8'))
        return cls._from_flat_dict(data)

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
        return 200.0

    def __str__(self) -> str:
        name = self.preset_name or 'Custom'
        return (f"SystemConfig({name}: "
                f"LED={self.led_part}, PV={self.pv_part}, "
                f"d={self.distance_m*100:.0f}cm, "
                f"rate={self.data_rate_bps/1e3:.0f}kbps)")
