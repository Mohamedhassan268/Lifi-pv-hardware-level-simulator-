# cosim/noise.py
"""
Full 6-Source Physical Noise Model.

Computes receiver noise from first principles with per-source breakdown.
Used by both the Python engine and SPICE pipeline (noise injection).

Sources:
    1. Shot noise:           2q·(I_ph + I_dark)·Bn
    2. Thermal noise:        4kT·Bn / R_load
    3. Ambient light noise:  2q·I_ambient·Bn
    4. Amplifier noise:      (e_n² + (i_n·Z_in)²)·Bn
    5. ADC quantization:     V_LSB²/12  (input-referred)
    6. Processing/threshold: σ_offset² + σ_jitter²

References:
    - Kahn & Barry, "Wireless Infrared Communications", Proc. IEEE 1997
    - INA322 datasheet (SBOS163), TLV7011 datasheet (SBOS819)
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional

# Physical constants
Q_ELECTRON = 1.602176634e-19     # Elementary charge (C)
K_BOLTZMANN = 1.380649e-23       # Boltzmann constant (J/K)


# =============================================================================
# PINK NOISE GENERATOR — Paul Kellet "economy" 1/f filter
# =============================================================================

def _pink_voss(n: int, rng: np.random.Generator) -> np.ndarray:
    """Generate `n` samples of approximately-1/f (pink) noise.

    6-pole IIR over white gaussian input, after Paul Kellet's "economy"
    method. Output is approximately 1/f over ~4 decades — adequate for
    audio-band amplifier flicker simulation. Amplitude is not normalized
    here; callers should renormalize to their target sigma.
    """
    white = rng.standard_normal(n)
    b0 = b1 = b2 = b3 = b4 = b5 = b6 = 0.0
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        x = float(white[i])
        b0 = 0.99886 * b0 + x * 0.0555179
        b1 = 0.99332 * b1 + x * 0.0750759
        b2 = 0.96900 * b2 + x * 0.1538520
        b3 = 0.86650 * b3 + x * 0.3104856
        b4 = 0.55000 * b4 + x * 0.5329522
        b5 = -0.7616 * b5 - x * 0.0168980
        out[i] = b0 + b1 + b2 + b3 + b4 + b5 + b6 + x * 0.5362
        b6 = x * 0.115926
    return out


# =============================================================================
# NOISE BREAKDOWN RESULT
# =============================================================================

@dataclass
class NoiseBreakdown:
    """Per-source noise variance breakdown (all in A²)."""
    shot: float = 0.0             # Source 1: shot noise variance
    thermal: float = 0.0          # Source 2: thermal noise variance
    ambient: float = 0.0          # Source 3: ambient light noise variance
    amplifier: float = 0.0        # Source 4: amplifier noise variance
    adc: float = 0.0              # Source 5: ADC quantization noise variance
    processing: float = 0.0       # Source 6: processing/threshold noise variance
    total: float = 0.0            # Sum of all enabled sources

    @property
    def total_std(self) -> float:
        """Total noise standard deviation (A)."""
        return np.sqrt(max(self.total, 0.0))

    @property
    def total_rms_mV(self) -> float:
        """Total noise as RMS voltage across 1 ohm (mV), for quick reference."""
        return self.total_std * 1e3

    def as_dict(self) -> dict:
        """Return as dictionary for serialization."""
        return {
            'shot_A2': self.shot,
            'thermal_A2': self.thermal,
            'ambient_A2': self.ambient,
            'amplifier_A2': self.amplifier,
            'adc_A2': self.adc,
            'processing_A2': self.processing,
            'total_A2': self.total,
            'total_std_A': self.total_std,
        }


# =============================================================================
# NOISE MODEL
# =============================================================================

class NoiseModel:
    """
    Physical 6-source noise model for PV-based optical receivers.

    Usage:
        from cosim.noise import NoiseModel

        nm = NoiseModel.from_config(system_config)
        breakdown = nm.compute_noise(I_ph=328e-6, bandwidth=2500)
        noise_samples = nm.generate_time_domain(n_samples, I_ph, bandwidth)
    """

    def __init__(self, temperature_K: float = 300.0,
                 R_load: float = 1.0,
                 ina_noise_nV_rtHz: float = 45.0,
                 ina_noise_current_pA_rtHz: float = 0.1,
                 ambient_illuminance_lux: float = 0.0,
                 rx_area_cm2: float = 9.0,
                 responsivity: float = 0.457,
                 adc_bits: int = 12,
                 adc_vref: float = 3.3,
                 amp_gain: float = 100.0,
                 comparator_offset_mV: float = 1.0,
                 comparator_jitter_ns: float = 5.0,
                 data_rate_bps: float = 5000.0,
                 # Enable flags for each source
                 enable_shot: bool = True,
                 enable_thermal: bool = True,
                 enable_ambient: bool = True,
                 enable_amplifier: bool = True,
                 enable_adc: bool = False,
                 enable_processing: bool = False,
                 # Real-world non-AWGN sources (mains flicker + amp 1/f)
                 enable_mains_flicker: bool = False,
                 mains_flicker_freq_hz: float = 100.0,
                 mains_flicker_depth: float = 0.05,
                 enable_amp_flicker: bool = False,
                 amp_flicker_corner_hz: float = 100.0):
        """
        Args:
            temperature_K: Ambient temperature (Kelvin)
            R_load: Sense/load resistance (ohms)
            ina_noise_nV_rtHz: INA voltage noise density (nV/√Hz)
            ina_noise_current_pA_rtHz: INA current noise density (pA/√Hz)
            ambient_illuminance_lux: Background light level (lux)
            rx_area_cm2: Receiver active area (cm²)
            responsivity: PV cell responsivity (A/W)
            adc_bits: ADC resolution (0 = no ADC)
            adc_vref: ADC reference voltage (V)
            amp_gain: Total amplifier gain (linear, e.g. INA322 = 100)
            comparator_offset_mV: Comparator input offset voltage (mV)
            comparator_jitter_ns: Comparator propagation delay jitter 1σ (ns)
            data_rate_bps: Data rate for jitter-to-noise conversion
            enable_*: Per-source enable flags
        """
        self.T = temperature_K
        self.R_load = max(R_load, 1e-6)
        self.e_n = ina_noise_nV_rtHz * 1e-9       # V/√Hz
        self.i_n = ina_noise_current_pA_rtHz * 1e-12  # A/√Hz
        self.ambient_lux = ambient_illuminance_lux
        self.rx_area_cm2 = rx_area_cm2
        self.responsivity = responsivity
        self.adc_bits = adc_bits
        self.adc_vref = adc_vref
        self.amp_gain = max(amp_gain, 1.0)
        self.comp_offset_V = comparator_offset_mV * 1e-3
        self.comp_jitter_s = comparator_jitter_ns * 1e-9
        self.data_rate = data_rate_bps

        self.enable_shot = enable_shot
        self.enable_thermal = enable_thermal
        self.enable_ambient = enable_ambient
        self.enable_amplifier = enable_amplifier
        self.enable_adc = enable_adc
        self.enable_processing = enable_processing

        self.enable_mains_flicker = enable_mains_flicker
        self.mains_flicker_freq_hz = mains_flicker_freq_hz
        self.mains_flicker_depth = mains_flicker_depth
        self.enable_amp_flicker = enable_amp_flicker
        self.amp_flicker_corner_hz = amp_flicker_corner_hz

    @classmethod
    def from_config(cls, config) -> 'NoiseModel':
        """Create NoiseModel from a SystemConfig instance."""
        ina_gain_linear = 10 ** (config.ina_gain_dB / 20)
        return cls(
            temperature_K=config.temperature_K,
            R_load=config.r_sense_ohm,
            ina_noise_nV_rtHz=config.ina_noise_nV_rtHz,
            ina_noise_current_pA_rtHz=config.ina_noise_current_pA_rtHz,
            ambient_illuminance_lux=config.ambient_illuminance_lux,
            rx_area_cm2=config.sc_area_cm2,
            responsivity=config.sc_responsivity,
            adc_bits=config.adc_bits,
            adc_vref=config.adc_vref,
            amp_gain=ina_gain_linear,
            comparator_offset_mV=config.comparator_offset_mV,
            comparator_jitter_ns=config.comparator_jitter_ns,
            data_rate_bps=config.data_rate_bps,
            enable_shot=config.noise_shot_enable,
            enable_thermal=config.noise_thermal_enable,
            enable_ambient=config.noise_ambient_enable,
            enable_amplifier=config.noise_amplifier_enable,
            enable_adc=config.noise_adc_enable,
            enable_processing=config.noise_processing_enable,
            enable_mains_flicker=getattr(config, 'enable_mains_flicker', False),
            mains_flicker_freq_hz=getattr(config, 'mains_flicker_freq_hz', 100.0),
            mains_flicker_depth=getattr(config, 'mains_flicker_depth', 0.05),
            enable_amp_flicker=getattr(config, 'enable_amp_flicker', False),
            amp_flicker_corner_hz=getattr(config, 'amp_flicker_corner_hz', 100.0),
        )

    # -------------------------------------------------------------------------
    # Individual noise sources (all return variance in A²)
    # -------------------------------------------------------------------------

    def shot_noise_variance(self, I_ph: float, bandwidth: float,
                            I_dark: float = 0.0) -> float:
        """
        Source 1: Shot noise.
        σ² = 2q·(I_ph + I_dark)·Bn
        """
        if not self.enable_shot:
            return 0.0
        return 2 * Q_ELECTRON * (abs(I_ph) + abs(I_dark)) * bandwidth

    def thermal_noise_variance(self, bandwidth: float) -> float:
        """
        Source 2: Thermal (Johnson-Nyquist) noise.
        σ² = 4kT·Bn / R_load
        """
        if not self.enable_thermal:
            return 0.0
        return 4 * K_BOLTZMANN * self.T * bandwidth / self.R_load

    def ambient_noise_variance(self, bandwidth: float) -> float:
        """
        Source 3: Ambient light noise.
        σ² = 2q·I_ambient·Bn

        I_ambient is derived from illuminance:
            P_ambient = lux × 1.46e-6 W/lux × area_cm²
            I_ambient = responsivity × P_ambient
        """
        if not self.enable_ambient or self.ambient_lux <= 0:
            return 0.0
        # 1 lux ≈ 1.46 µW/cm² for broadband visible light (standard luminous efficacy)
        P_ambient_W = self.ambient_lux * 1.46e-6 * self.rx_area_cm2
        I_ambient = self.responsivity * P_ambient_W
        return 2 * Q_ELECTRON * I_ambient * bandwidth

    def amplifier_noise_variance(self, bandwidth: float) -> float:
        """
        Source 4: Amplifier input-referred noise.
        σ² = (e_n² / R_load² + i_n²) · Bn

        Input-referred to the photocurrent domain (A²).
        e_n is voltage noise → divide by Z_in (≈R_load) to get current.
        i_n is current noise already in A/√Hz.
        """
        if not self.enable_amplifier:
            return 0.0
        # Voltage noise → current noise via Z_in ≈ R_load
        V_noise_current = (self.e_n / self.R_load) ** 2
        I_noise_current = self.i_n ** 2
        return (V_noise_current + I_noise_current) * bandwidth

    def adc_quantization_variance(self) -> float:
        """
        Source 5: ADC quantization noise.
        σ²_V = V_LSB² / 12
        Input-referred to current: σ²_I = σ²_V / (R_load · amp_gain)²
        """
        if not self.enable_adc or self.adc_bits <= 0:
            return 0.0
        V_lsb = self.adc_vref / (2 ** self.adc_bits)
        sigma_v_sq = V_lsb ** 2 / 12.0
        transimpedance = self.R_load * self.amp_gain
        return sigma_v_sq / max(transimpedance ** 2, 1e-30)

    def processing_noise_variance(self) -> float:
        """
        Source 6: Processing/threshold noise from comparator.
        σ² = σ_offset² + σ_jitter²

        - σ_offset: comparator input offset voltage → current via Z_in
        - σ_jitter: timing jitter → amplitude noise at the data rate

        Input-referred to current domain.
        """
        if not self.enable_processing:
            return 0.0
        # Offset voltage → current
        transimpedance = self.R_load * self.amp_gain
        sigma_offset_I = self.comp_offset_V / max(transimpedance, 1e-30)

        # Jitter → amplitude noise: σ_A ≈ slew_rate × σ_t
        # Approximate slew rate as V_swing / T_bit
        T_bit = 1.0 / max(self.data_rate, 1.0)
        V_swing = self.adc_vref  # Approximate signal swing
        slew_rate = V_swing / T_bit
        sigma_jitter_V = slew_rate * self.comp_jitter_s
        sigma_jitter_I = sigma_jitter_V / max(transimpedance, 1e-30)

        return sigma_offset_I ** 2 + sigma_jitter_I ** 2

    # -------------------------------------------------------------------------
    # Aggregate computation
    # -------------------------------------------------------------------------

    def compute_noise(self, I_ph: float, bandwidth: float,
                      I_dark: float = 0.0) -> NoiseBreakdown:
        """
        Compute full noise breakdown.

        Args:
            I_ph: Average photocurrent (A)
            bandwidth: Noise bandwidth (Hz), typically data_rate / 2
            I_dark: Dark current (A)

        Returns:
            NoiseBreakdown with per-source and total noise variances
        """
        s_shot = self.shot_noise_variance(I_ph, bandwidth, I_dark)
        s_thermal = self.thermal_noise_variance(bandwidth)
        s_ambient = self.ambient_noise_variance(bandwidth)
        s_amp = self.amplifier_noise_variance(bandwidth)
        s_adc = self.adc_quantization_variance()
        s_proc = self.processing_noise_variance()

        total = s_shot + s_thermal + s_ambient + s_amp + s_adc + s_proc

        return NoiseBreakdown(
            shot=s_shot,
            thermal=s_thermal,
            ambient=s_ambient,
            amplifier=s_amp,
            adc=s_adc,
            processing=s_proc,
            total=total,
        )

    def total_noise_std(self, I_ph, bandwidth: float) -> float:
        """Total noise standard deviation (A). Convenience method."""
        I_avg = float(np.mean(np.abs(I_ph))) if hasattr(I_ph, '__len__') else abs(I_ph)
        return self.compute_noise(I_avg, bandwidth).total_std

    # -------------------------------------------------------------------------
    # Time-domain noise generation
    # -------------------------------------------------------------------------

    def generate_time_domain(self, n_samples: int, I_ph, bandwidth: float,
                             rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """
        Generate AWGN noise samples in the current domain.

        Args:
            n_samples: Number of samples to generate
            I_ph: Photocurrent (scalar or array, for shot noise level)
            bandwidth: Noise bandwidth (Hz)
            rng: Optional numpy random Generator for reproducibility

        Returns:
            Noise samples array of shape (n_samples,) in Amperes
        """
        sigma = self.total_noise_std(I_ph, bandwidth)
        if sigma <= 0:
            return np.zeros(n_samples)
        if rng is None:
            return np.random.normal(0, sigma, n_samples)
        return rng.normal(0, sigma, n_samples)

    def generate_per_source_time_domain(
        self, n_samples: int, I_ph, bandwidth: float,
        rng: Optional[np.random.Generator] = None,
    ) -> dict:
        """
        Generate independent AWGN samples for each enabled noise source.

        Each source draws from its own variance (computed via compute_noise)
        with statistically independent samples. Used by glass-box observability
        so the UI can display per-source contributions; the summed `total`
        array is what gets added to I_ph for the actual simulation.

        Returns a dict with keys 'shot', 'thermal', 'ambient', 'amplifier',
        'adc', 'processing', 'total' — each a length-n_samples np.ndarray
        in Amperes. Disabled sources return all-zeros.
        """
        if rng is None:
            rng = np.random.default_rng()

        I_avg = (
            float(np.mean(np.abs(I_ph))) if hasattr(I_ph, "__len__") else float(abs(I_ph))
        )
        breakdown = self.compute_noise(I_avg, bandwidth)

        def _draw(variance: float) -> np.ndarray:
            if variance <= 0:
                return np.zeros(n_samples)
            return rng.normal(0.0, float(np.sqrt(variance)), n_samples)

        contributions = {
            "shot": _draw(breakdown.shot),
            "thermal": _draw(breakdown.thermal),
            "ambient": _draw(breakdown.ambient),
            "amplifier": _draw(breakdown.amplifier),
            "adc": _draw(breakdown.adc),
            "processing": _draw(breakdown.processing),
        }
        contributions["total"] = sum(contributions.values())
        return contributions

    # -------------------------------------------------------------------------
    # Non-AWGN real-world sources: mains flicker + amplifier 1/f
    # -------------------------------------------------------------------------

    def mains_flicker_current(
        self, t: np.ndarray,
        I_ambient: Optional[float] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """
        Deterministic AC-mains-flicker interferer (in Amperes).

        Indoor lighting driven by AC mains rectification flickers at 2x the
        mains frequency (100 Hz for 50 Hz mains, 120 Hz for 60 Hz). Modeled
        as a fraction of the DC ambient photocurrent, with two decreasing
        harmonics that approximate fluorescent / LED ballast spectra:

            i_fl(t) = depth * I_ambient * Σ_k a_k * cos(2π * k * f * t + φ_k)
            a_1 = 1.0,  a_2 = 0.30,  a_3 = 0.10

        Phases are random per run; pass a seeded rng for repeatability.
        Returns zeros if the source or I_ambient is off.
        """
        if not self.enable_mains_flicker:
            return np.zeros_like(np.asarray(t), dtype=np.float64)
        if rng is None:
            rng = np.random.default_rng()

        if I_ambient is None:
            P_amb = self.ambient_lux * 1.46e-6 * self.rx_area_cm2
            I_ambient = self.responsivity * P_amb

        f = float(self.mains_flicker_freq_hz)
        depth = float(self.mains_flicker_depth)
        t_arr = np.asarray(t, dtype=np.float64)
        if I_ambient <= 0 or depth <= 0 or f <= 0:
            return np.zeros_like(t_arr, dtype=np.float64)

        phi_1 = 2 * np.pi * rng.random()
        phi_2 = 2 * np.pi * rng.random()
        phi_3 = 2 * np.pi * rng.random()
        omega = 2 * np.pi * f * t_arr
        sig = (
            1.00 * np.cos(omega + phi_1) +
            0.30 * np.cos(2 * omega + phi_2) +
            0.10 * np.cos(3 * omega + phi_3)
        )
        return (depth * float(I_ambient)) * sig

    def mains_flicker_variance(self) -> float:
        """
        Analytical variance of the mains-flicker current (A²).

        i_fl(t) = depth · I_amb · Σ_k a_k · cos(2π k f t + φ_k)
        With a_1 = 1.0, a_2 = 0.30, a_3 = 0.10 (decorrelated harmonics by
        random phases), the mean-square value is

            Var[i_fl] = (depth · I_amb)² · Σ_k (a_k² / 2)
                      = (depth · I_amb)² · (1² + 0.30² + 0.10²) / 2
                      = (depth · I_amb)² · 0.55

        Returns 0 if the source is disabled, the ambient illuminance is zero,
        or the depth is zero.
        """
        if not self.enable_mains_flicker:
            return 0.0
        if self.ambient_lux <= 0 or self.mains_flicker_depth <= 0:
            return 0.0
        P_amb = self.ambient_lux * 1.46e-6 * self.rx_area_cm2
        I_amb = self.responsivity * P_amb
        if I_amb <= 0:
            return 0.0
        # 1.0² + 0.30² + 0.10² = 1.10, then /2 → 0.55
        return float((self.mains_flicker_depth * I_amb) ** 2 * 0.55)

    def amp_flicker_variance(self, t_total_s: float, bandwidth_hz: float) -> float:
        """
        Analytical variance of the amplifier 1/f noise (A²), input-referred.

        From the wideband-PSD model e_n(f) = e_n_white · √(1 + f_c/f),
        the excess variance from the 1/f shape integrated from f_low = 1/T
        to the noise bandwidth B_n is

            σ²_pink_V = e_n_white² · f_c · ln(B_n / f_low)

        Divided by R_load² to refer it back to the current domain.
        """
        if not self.enable_amp_flicker:
            return 0.0
        f_low = max(1.0 / t_total_s, 0.1) if t_total_s > 0 else 0.1
        f_c = float(self.amp_flicker_corner_hz)
        if bandwidth_hz <= f_low or f_c <= 0:
            return 0.0
        sigma_v_sq = (self.e_n ** 2) * f_c * np.log(bandwidth_hz / f_low)
        return float(sigma_v_sq / max(self.R_load, 1e-6) ** 2)

    def amp_flicker_current(
        self, n_samples: int, dt: float,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """
        Amplifier 1/f (pink) noise referred to the photocurrent domain (A).

        Voltage-noise density rises below the flicker corner f_c as
            e_n(f) = e_n_white * sqrt(1 + f_c/f)
        which when integrated from f_low (set by run length 1/T_total) up to
        the Nyquist bandwidth gives an excess variance:
            σ²_pink_V = e_n_white² * f_c * ln(B_n / f_low)
        Dividing by R_load² yields the equivalent input-referred current
        variance the time-domain samples are normalized to.

        Time-domain shaping uses Paul Kellet's economy 6-pole IIR over white
        noise (see `_pink_voss`).
        """
        if not self.enable_amp_flicker or n_samples <= 0:
            return np.zeros(max(n_samples, 0), dtype=np.float64)
        if rng is None:
            rng = np.random.default_rng()

        T_total = n_samples * dt
        f_low = max(1.0 / T_total, 0.1) if T_total > 0 else 0.1
        bandwidth = 0.5 / dt if dt > 0 else 1.0
        f_c = float(self.amp_flicker_corner_hz)
        if bandwidth <= f_low or f_c <= 0:
            return np.zeros(n_samples, dtype=np.float64)

        sigma_v = self.e_n * np.sqrt(f_c * np.log(bandwidth / f_low))
        sigma_i = sigma_v / max(self.R_load, 1e-6)

        pink = _pink_voss(n_samples, rng)
        std = float(np.std(pink))
        if std > 0:
            pink = pink * (sigma_i / std)
        return pink.astype(np.float64, copy=False)

    # -------------------------------------------------------------------------
    # SPICE noise source generation
    # -------------------------------------------------------------------------

    def generate_spice_source(self, t: np.ndarray, I_ph: float,
                              bandwidth: float,
                              rng: Optional[np.random.Generator] = None) -> str:
        """
        Generate a PWL noise current source string for SPICE injection.

        Returns a string like:
            Inoise node_p node_n PWL(0 1.2e-7 1e-6 -3.4e-8 ...)

        Args:
            t: Time array (s)
            I_ph: Average photocurrent for noise level calculation
            bandwidth: Noise bandwidth (Hz)
            rng: Optional RNG for reproducibility

        Returns:
            SPICE PWL source definition string
        """
        noise = self.generate_time_domain(len(t), I_ph, bandwidth, rng)
        pairs = []
        # Downsample for SPICE efficiency (max 10000 points)
        step = max(1, len(t) // 10000)
        for i in range(0, len(t), step):
            pairs.append(f"{t[i]:.6e} {noise[i]:.6e}")
        pwl_data = " ".join(pairs)
        return f"PWL({pwl_data})"

    # -------------------------------------------------------------------------
    # String representation
    # -------------------------------------------------------------------------

    def __repr__(self) -> str:
        sources = []
        if self.enable_shot: sources.append("shot")
        if self.enable_thermal: sources.append("thermal")
        if self.enable_ambient: sources.append("ambient")
        if self.enable_amplifier: sources.append("amp")
        if self.enable_adc: sources.append("adc")
        if self.enable_processing: sources.append("proc")
        return f"NoiseModel(T={self.T}K, R={self.R_load}ohm, sources=[{', '.join(sources)}])"
