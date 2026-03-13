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
                 enable_processing: bool = False):
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
