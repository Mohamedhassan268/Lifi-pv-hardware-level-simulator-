# cosim/python_engine.py
"""
Python System-Level Simulation Engine.

Provides a pure-Python simulation pipeline for papers that cannot use SPICE
(OFDM, BFSK, PWM-ASK, MIMO architectures). Adapted from the validated
system-level simulator (lifi_pv_simulator).

Supports 6 modulation schemes: OOK, OOK_Manchester, OFDM, BFSK, PWM_ASK.
Implements: Transmitter, OpticalChannel, PVReceiver, NoiseModel, Demodulator.

Usage:
    from cosim.python_engine import run_python_simulation
    result = run_python_simulation(config)
"""

import numpy as np
from scipy import signal as sp_signal
from scipy.special import erfc
from dataclasses import dataclass
from typing import Dict, Optional

# Physical constants
Q_ELECTRON = 1.602e-19
K_BOLTZMANN = 1.38e-23


# =============================================================================
# BER PREDICTION FUNCTIONS
# =============================================================================

def predict_ber_ook(snr_linear):
    """BER for OOK: 0.5 * erfc(sqrt(SNR/2))."""
    return 0.5 * erfc(np.sqrt(np.maximum(snr_linear, 0) / 2))

def predict_ber_ook_db(snr_db):
    """BER for OOK given SNR in dB."""
    snr_linear = 10 ** (np.asarray(snr_db, dtype=float) / 10)
    return predict_ber_ook(snr_linear)

def predict_ber_bpsk(snr_linear):
    """BER for BPSK: 0.5 * erfc(sqrt(SNR))."""
    return 0.5 * erfc(np.sqrt(np.maximum(snr_linear, 0)))

def predict_ber_bfsk(snr_linear):
    """BER for non-coherent BFSK: 0.5 * exp(-SNR/2)."""
    return 0.5 * np.exp(-np.maximum(snr_linear, 0) / 2)

def predict_ber_mqam(snr_linear, M):
    """Approximate BER for M-QAM."""
    if M <= 1:
        return 0.0
    k = np.log2(M)
    factor = (4 / k) * (1 - 1 / np.sqrt(M))
    arg = np.sqrt(3 * np.maximum(snr_linear, 0) / (M - 1))
    return factor * 0.5 * erfc(arg / np.sqrt(2))


# =============================================================================
# MANCHESTER CODEC
# =============================================================================

def manchester_encode(bits):
    """Manchester-encode: bit 1 -> [1,0], bit 0 -> [0,1]."""
    bits = np.asarray(bits, dtype=int)
    symbols = np.empty(2 * len(bits), dtype=int)
    symbols[0::2] = bits
    symbols[1::2] = 1 - bits
    return symbols

def manchester_decode(symbols):
    """Decode Manchester symbols: compare first/second half of each bit."""
    symbols = np.asarray(symbols, dtype=float)
    n_bits = len(symbols) // 2
    first_half = symbols[0::2][:n_bits]
    second_half = symbols[1::2][:n_bits]
    return (first_half > second_half).astype(int)


# =============================================================================
# TRANSMITTER
# =============================================================================

class Transmitter:
    """Optical transmitter with multiple modulation schemes."""

    def __init__(self, dc_bias_mA=12, mod_depth=0.5, led_efficiency=0.08):
        self.I_dc_ma = dc_bias_mA
        self.m = mod_depth
        self.eta_led = led_efficiency

    def modulate_ook(self, bits, t):
        """OOK modulation: P_tx = eta * (I_dc + m*I_dc*d(t))."""
        sps = len(t) // len(bits)
        d = np.repeat(bits, sps)[:len(t)]
        I_tx_ma = self.I_dc_ma + self.m * self.I_dc_ma * d
        return self.eta_led * (I_tx_ma * 1e-3)

    def modulate_manchester(self, bits, t):
        """Manchester-encoded OOK."""
        sps = len(t) // len(bits)
        symbols = manchester_encode(bits)
        sps_sym = sps // 2
        d = np.repeat(symbols.astype(float), sps_sym)[:len(t)]
        I_tx_ma = self.I_dc_ma + self.m * self.I_dc_ma * d
        return self.eta_led * (I_tx_ma * 1e-3)

    def modulate_ofdm(self, bits, t, qam_order=16, n_fft=64, cp_len=16):
        """DCO-OFDM modulation."""
        n_data_carriers = n_fft // 2 - 1
        bps = int(np.log2(qam_order))
        bits_per_symbol = n_data_carriers * bps
        n_symbols = max(1, len(bits) // bits_per_symbol)
        # Pad bits if needed for at least 1 full symbol
        if len(bits) < bits_per_symbol:
            bits = np.concatenate([bits, np.zeros(bits_per_symbol - len(bits), dtype=int)])
        bits_used = bits[:n_symbols * bits_per_symbol]

        ofdm_signal = []
        for s in range(n_symbols):
            frame_bits = bits_used[s * bits_per_symbol:(s + 1) * bits_per_symbol]
            qam_syms = self._bits_to_qam(frame_bits, qam_order, n_data_carriers)

            # Hermitian symmetry
            freq = np.zeros(n_fft, dtype=complex)
            freq[1:n_fft // 2] = qam_syms
            freq[n_fft // 2 + 1:] = np.conj(qam_syms[::-1])

            sig = np.fft.ifft(freq).real
            sig_cp = np.concatenate([sig[-cp_len:], sig])
            ofdm_signal.extend(sig_cp)

        ofdm_signal = np.array(ofdm_signal)
        if np.std(ofdm_signal) > 0:
            ofdm_signal /= np.std(ofdm_signal)

        I_tx = self.I_dc_ma + ofdm_signal * self.m * self.I_dc_ma
        I_tx = np.maximum(I_tx, 0)

        # Interpolate to physics time
        indices = np.linspace(0, len(I_tx) - 1, len(t))
        I_tx_interp = np.interp(indices, np.arange(len(I_tx)), I_tx)
        return self.eta_led * (I_tx_interp * 1e-3)

    def modulate_bfsk(self, bits, t, f0=1600, f1=2000,
                      rise_time=1.34e-3, fall_time=0.15e-3):
        """Passive BFSK via LC shutter modulation."""
        dt = t[1] - t[0]
        sps = len(t) // len(bits)

        freq = np.repeat(bits.astype(float), sps)[:len(t)]
        freq = np.where(freq == 0, f0, f1)

        phase = np.cumsum(freq) * dt * 2 * np.pi
        ideal_sq = 0.5 * (1 + np.sign(np.sin(phase)))

        # Smooth with LC shutter dynamics
        alpha_r = min(dt / rise_time, 1.0) if rise_time > 0 else 1.0
        alpha_f = min(dt / fall_time, 1.0) if fall_time > 0 else 1.0

        lc_state = np.zeros_like(ideal_sq)
        val = 0.0
        for i in range(len(t)):
            target = ideal_sq[i]
            if target > val:
                val += alpha_r * (target - val)
            else:
                val += alpha_f * (target - val)
            lc_state[i] = val

        P_tx_dc = self.eta_led * (self.I_dc_ma * 1e-3)
        return lc_state * P_tx_dc

    def modulate_pwm_ask(self, t, pwm_freq=10, carrier_freq=10000, duty=0.5):
        """PWM-ASK modulation (Correa 2025)."""
        phase_pwm = (t * pwm_freq) % 1.0
        pwm = (phase_pwm < duty).astype(float)

        phase_carrier = (t * carrier_freq) % 1.0
        carrier = (phase_carrier < 0.5).astype(float)

        signal_d = pwm * carrier
        level_hi = self.I_dc_ma * (1 + self.m)
        level_lo = self.I_dc_ma * (1 - self.m)
        I_tx = level_lo + (level_hi - level_lo) * signal_d
        return self.eta_led * (I_tx * 1e-3)

    def _bits_to_qam(self, bits, qam_order, n_carriers):
        """Map bits to QAM constellation symbols."""
        bps = int(np.log2(qam_order))
        symbols = np.zeros(n_carriers, dtype=complex)
        for i in range(n_carriers):
            sub = bits[i * bps:(i + 1) * bps]
            if len(sub) < bps:
                break
            if qam_order == 4:
                I = 2 * sub[0] - 1
                Q = 2 * sub[1] - 1
                symbols[i] = (I + 1j * Q) / np.sqrt(2)
            elif qam_order in (16, 64):
                half = bps // 2
                I_bits = sub[:half]
                Q_bits = sub[half:]
                I_val = sum(b * 2**k for k, b in enumerate(reversed(I_bits)))
                Q_val = sum(b * 2**k for k, b in enumerate(reversed(Q_bits)))
                M = 2**half
                I_pam = 2 * I_val - (M - 1)
                Q_pam = 2 * Q_val - (M - 1)
                norm = np.sqrt(2 * (M**2 - 1) / 3)
                symbols[i] = (I_pam + 1j * Q_pam) / norm
            else:
                symbols[i] = 2 * sub[0] - 1  # BPSK fallback
        return symbols


# =============================================================================
# OPTICAL CHANNEL
# =============================================================================

class OpticalChannel:
    """Free-space optical channel with Lambertian path loss and Beer-Lambert."""

    def __init__(self, distance_m, beam_half_angle_deg, rx_area_cm2,
                 humidity=None, temperature_K=300):
        self.distance = distance_m
        self.rx_area_m2 = rx_area_cm2 * 1e-4
        self.temp_k = temperature_K

        # Lambertian order
        alpha = np.radians(beam_half_angle_deg)
        self.m_L = -np.log(2) / np.log(np.cos(alpha))

        # Beer-Lambert attenuation
        self.humidity = humidity
        self.alpha_atten = self._compute_alpha(humidity)

    def _compute_alpha(self, humidity):
        """Beer-Lambert coefficient from humidity (Correa 2025 model)."""
        if humidity is None:
            return 0.0
        humidity = np.clip(humidity, 0.0, 1.0)
        alpha_base = 0.3
        if humidity >= 0.3:
            alpha_humidity = 4.0 * (humidity - 0.3) ** 1.5
        else:
            alpha_humidity = 0.0
        return alpha_base + alpha_humidity

    def channel_gain(self):
        """Compute DC channel gain H(0)."""
        gain = ((self.m_L + 1) * self.rx_area_m2) / (2 * np.pi * self.distance**2)
        beer_lambert = np.exp(-self.alpha_atten * self.distance)
        return gain * beer_lambert

    def propagate(self, P_tx):
        """Apply channel gain to transmitted optical power."""
        return self.channel_gain() * P_tx


# =============================================================================
# NOISE MODEL (6-source, from validated simulator)
# =============================================================================

class NoiseModel:
    """Physical noise model: shot + thermal + ambient + amp + ADC + processing."""

    def __init__(self, temperature_K=300, R_load=50, ina_noise_nV=10,
                 ambient_lux=0, rx_area_cm2=9.0, adc_bits=12,
                 adc_vref=3.3, amp_gain=11.0):
        self.T = temperature_K
        self.R_load = R_load
        self.V_n_ina = ina_noise_nV * 1e-9
        self.ambient_lux = ambient_lux
        self.rx_area_cm2 = rx_area_cm2
        self.adc_bits = adc_bits
        self.adc_vref = adc_vref
        self.amp_gain = amp_gain

    def total_noise_std(self, I_ph, bandwidth):
        """Total noise standard deviation in Amperes."""
        I_avg = np.mean(np.abs(I_ph)) if hasattr(I_ph, '__len__') else abs(I_ph)

        # Shot noise
        s_shot = 2 * Q_ELECTRON * I_avg * bandwidth

        # Thermal noise
        s_thermal = 4 * K_BOLTZMANN * self.T * bandwidth / max(self.R_load, 1)

        # Ambient noise
        if self.ambient_lux > 0:
            P_amb = self.ambient_lux * 1.46e-6 * self.rx_area_cm2
            I_amb = 0.4 * P_amb
            s_ambient = 2 * Q_ELECTRON * I_amb * bandwidth
        else:
            s_ambient = 0.0

        # Amplifier noise
        I_n_from_V = self.V_n_ina / max(self.R_load, 1)
        s_amp = I_n_from_V**2 * bandwidth

        # ADC quantization noise
        if self.adc_bits > 0:
            V_lsb = self.adc_vref / (2 ** self.adc_bits)
            sigma_v = V_lsb / np.sqrt(12)
            ti = self.R_load * self.amp_gain
            s_adc = (sigma_v / max(ti, 1))**2
        else:
            s_adc = 0.0

        return np.sqrt(s_shot + s_thermal + s_ambient + s_amp + s_adc)

    def generate_noise(self, length, I_ph, bandwidth):
        """Generate AWGN noise samples."""
        sigma = self.total_noise_std(I_ph, bandwidth)
        return np.random.normal(0, sigma, length)


# =============================================================================
# PV RECEIVER
# =============================================================================

class PVReceiver:
    """PV cell receiver with TIA and signal chain processing."""

    def __init__(self, responsivity=0.457, capacitance_nF=798,
                 shunt_resistance_kOhm=138.8, n_cells=1, temperature_K=300):
        self.R = responsivity
        self.C_j = capacitance_nF * 1e-9
        self.R_sh = shunt_resistance_kOhm * 1e3
        self.n_cells = n_cells
        self.T = temperature_K
        self.V_T = K_BOLTZMANN * temperature_K / Q_ELECTRON

    def optical_to_current(self, P_rx):
        """I_ph = R * P_rx."""
        return self.R * P_rx

    def apply_tia(self, I_ph, t, R_tia=50e3, f_3db=3e6):
        """TIA: V = R_tia * I_ph with bandwidth limit."""
        V_ideal = R_tia * I_ph
        dt = np.mean(np.diff(t))
        fs = 1.0 / dt
        wn = f_3db / (fs / 2)
        if wn >= 1.0:
            return V_ideal
        b, a = sp_signal.butter(1, wn, btype='low')
        return sp_signal.filtfilt(b, a, V_ideal)

    def apply_bandpass(self, signal_in, t, f_low=700, f_high=10000, order=2):
        """Butterworth bandpass filter."""
        dt = np.mean(np.diff(t))
        fs = 1.0 / dt
        nyq = fs / 2
        low_n = f_low / nyq
        high_n = f_high / nyq
        if low_n >= 1.0 or high_n >= 1.0 or low_n <= 0:
            return signal_in
        b, a = sp_signal.butter(order, [low_n, high_n], btype='band')
        try:
            return sp_signal.filtfilt(b, a, signal_in)
        except ValueError:
            return signal_in

    def apply_notch(self, signal_in, t, f_notch=100, Q=30):
        """IIR notch filter for mains rejection (González 2024)."""
        dt = np.mean(np.diff(t))
        fs = 1.0 / dt
        if f_notch >= fs / 2:
            return signal_in
        b, a = sp_signal.iirnotch(f_notch, Q, fs)
        return sp_signal.filtfilt(b, a, signal_in)


# =============================================================================
# DEMODULATOR
# =============================================================================

class Demodulator:
    """Signal processing and bit recovery."""

    def __init__(self, sample_rate=1e6, hpf_cutoff=1e4, lpf_cutoff=4e5):
        self.fs = sample_rate
        self.hpf_cutoff = hpf_cutoff
        self.lpf_cutoff = lpf_cutoff

    def apply_hpf(self, signal_in):
        """High-pass filter to remove DC."""
        nyq = self.fs / 2
        wn = self.hpf_cutoff / nyq
        if wn <= 0 or wn >= 1:
            return signal_in
        b, a = sp_signal.butter(4, wn, btype='high')
        return sp_signal.filtfilt(b, a, signal_in)

    def apply_lpf(self, signal_in):
        """Low-pass filter for noise reduction."""
        nyq = self.fs / 2
        wn = self.lpf_cutoff / nyq
        if wn <= 0 or wn >= 1:
            return signal_in
        b, a = sp_signal.butter(4, wn, btype='low')
        return sp_signal.filtfilt(b, a, signal_in)

    def demodulate_ook(self, signal_in, n_bits, sps):
        """OOK demodulation: HPF -> LPF -> sample -> decide."""
        V_hpf = self.apply_hpf(signal_in)
        V_lpf = self.apply_lpf(V_hpf)
        indices = np.arange(n_bits) * sps + sps // 2
        indices = np.clip(indices.astype(int), 0, len(V_lpf) - 1)
        samples = V_lpf[indices]
        threshold = (np.max(samples) + np.min(samples)) / 2
        return (samples > threshold).astype(int)

    def demodulate_manchester(self, signal_in, n_bits, sps):
        """Manchester demodulation: compare half-bit energies."""
        V_hpf = self.apply_hpf(signal_in)
        V_lpf = self.apply_lpf(V_hpf)
        bits_rx = np.zeros(n_bits, dtype=int)
        for i in range(n_bits):
            idx1 = min(i * sps + sps // 4, len(V_lpf) - 1)
            idx2 = min(i * sps + 3 * sps // 4, len(V_lpf) - 1)
            bits_rx[i] = 1 if V_lpf[idx1] > V_lpf[idx2] else 0
        return bits_rx

    def demodulate_ofdm(self, signal_rx, bits_tx, qam_order, n_fft, cp_len,
                        n_subcarriers):
        """OFDM demodulation: remove CP, FFT, QAM demap."""
        bps = int(np.log2(qam_order))
        bits_per_symbol = n_subcarriers * bps
        sym_len = n_fft + cp_len
        n_symbols = len(signal_rx) // sym_len

        bits_rx = []
        for s in range(n_symbols):
            sym = signal_rx[s * sym_len:(s + 1) * sym_len]
            # Remove CP
            sig = sym[cp_len:]
            # FFT
            freq = np.fft.fft(sig)
            # Extract data carriers
            data_carriers = freq[1:n_fft // 2][:n_subcarriers]
            # Demap QAM to bits
            for qam_sym in data_carriers:
                bits_rx.extend(self._qam_demap(qam_sym, qam_order))

        return np.array(bits_rx[:len(bits_tx)], dtype=int)

    def demodulate_bfsk(self, signal_in, n_bits, sps, f0, f1, fs):
        """Non-coherent BFSK demodulation using Goertzel energy detection."""
        bits_rx = np.zeros(n_bits, dtype=int)
        for i in range(n_bits):
            start = i * sps
            end = min(start + sps, len(signal_in))
            segment = signal_in[start:end]
            if len(segment) == 0:
                continue
            # Energy at f0 and f1
            e0 = self._goertzel_energy(segment, f0, fs)
            e1 = self._goertzel_energy(segment, f1, fs)
            bits_rx[i] = 1 if e1 > e0 else 0
        return bits_rx

    def _goertzel_energy(self, x, target_freq, fs):
        """Goertzel algorithm for single-frequency energy detection."""
        N = len(x)
        k = int(round(N * target_freq / fs))
        w = 2 * np.pi * k / N
        coeff = 2 * np.cos(w)
        s0 = s1 = s2 = 0.0
        for sample in x:
            s0 = sample + coeff * s1 - s2
            s2 = s1
            s1 = s0
        return s1**2 + s2**2 - coeff * s1 * s2

    def _qam_demap(self, symbol, qam_order):
        """Demap a single QAM symbol to bits."""
        if qam_order == 4:
            I = 1 if symbol.real > 0 else 0
            Q = 1 if symbol.imag > 0 else 0
            return [I, Q]
        elif qam_order == 16:
            half = 2
            M = 4
            norm = np.sqrt(2 * (M**2 - 1) / 3)
            I_pam = symbol.real * norm
            Q_pam = symbol.imag * norm
            I_val = int(np.clip(round((I_pam + M - 1) / 2), 0, M - 1))
            Q_val = int(np.clip(round((Q_pam + M - 1) / 2), 0, M - 1))
            I_bits = [(I_val >> (half - 1 - k)) & 1 for k in range(half)]
            Q_bits = [(Q_val >> (half - 1 - k)) & 1 for k in range(half)]
            return I_bits + Q_bits
        elif qam_order == 64:
            half = 3
            M = 8
            norm = np.sqrt(2 * (M**2 - 1) / 3)
            I_pam = symbol.real * norm
            Q_pam = symbol.imag * norm
            I_val = int(np.clip(round((I_pam + M - 1) / 2), 0, M - 1))
            Q_val = int(np.clip(round((Q_pam + M - 1) / 2), 0, M - 1))
            I_bits = [(I_val >> (half - 1 - k)) & 1 for k in range(half)]
            Q_bits = [(Q_val >> (half - 1 - k)) & 1 for k in range(half)]
            return I_bits + Q_bits
        else:
            return [1 if symbol.real > 0 else 0]

    @staticmethod
    def calculate_ber(bits_tx, bits_rx):
        """Calculate Bit Error Rate."""
        n = min(len(bits_tx), len(bits_rx))
        errors = int(np.sum(bits_tx[:n] != bits_rx[:n]))
        ber = errors / n if n > 0 else 0.0
        return {'ber': ber, 'n_errors': errors, 'n_bits_tested': n}


# =============================================================================
# MAIN SIMULATION RUNNER
# =============================================================================

def run_python_simulation(config) -> Dict:
    """
    Run a full system-level Python simulation using SystemConfig parameters.

    Maps SystemConfig fields to the simulator classes and runs the appropriate
    modulation pipeline.

    Args:
        config: SystemConfig instance

    Returns:
        Dict with keys: 'ber', 'n_errors', 'n_bits_tested', 'snr_est_dB',
                        'time', 'P_tx', 'P_rx', 'I_ph', 'V_rx', 'bits_tx',
                        'bits_rx', 'engine', 'modulation'
    """
    cfg = config
    modulation = cfg.modulation.upper().replace('-', '_')

    # Compute timing
    bit_period = 1.0 / cfg.data_rate_bps
    samples_per_bit = 500
    n_bits = cfg.n_bits
    n_samples = n_bits * samples_per_bit
    dt = bit_period / samples_per_bit
    fs = 1.0 / dt
    t = np.arange(n_samples) * dt

    # Generate TX bits
    np.random.seed(42)
    bits_tx = np.random.randint(0, 2, n_bits)

    # Create transmitter
    tx = Transmitter(
        dc_bias_mA=max(cfg.bias_current_A * 1000, 10),
        mod_depth=cfg.modulation_depth,
        led_efficiency=cfg.led_gled if cfg.led_gled > 0 else 0.08,
    )

    # Create channel
    channel = OpticalChannel(
        distance_m=cfg.distance_m,
        beam_half_angle_deg=cfg.led_half_angle_deg,
        rx_area_cm2=cfg.sc_area_cm2,
        humidity=cfg.humidity_rh,
        temperature_K=cfg.temperature_K,
    )

    # Create receiver
    rx = PVReceiver(
        responsivity=cfg.sc_responsivity,
        capacitance_nF=cfg.sc_cj_nF,
        shunt_resistance_kOhm=cfg.sc_rsh_kOhm,
        n_cells=cfg.n_cells_series,
        temperature_K=cfg.temperature_K,
    )

    # ========== MODULATION-SPECIFIC PIPELINE ==========

    if modulation in ('OOK',):
        P_tx = tx.modulate_ook(bits_tx, t)
    elif modulation in ('OOK_MANCHESTER',):
        P_tx = tx.modulate_manchester(bits_tx, t)
    elif modulation == 'OFDM':
        P_tx = tx.modulate_ofdm(
            bits_tx, t,
            qam_order=cfg.ofdm_qam_order,
            n_fft=cfg.ofdm_nfft,
            cp_len=cfg.ofdm_cp_len,
        )
    elif modulation == 'BFSK':
        P_tx = tx.modulate_bfsk(
            bits_tx, t,
            f0=cfg.bfsk_f0_hz,
            f1=cfg.bfsk_f1_hz,
        )
    elif modulation == 'PWM_ASK':
        P_tx = tx.modulate_pwm_ask(
            t,
            pwm_freq=cfg.pwm_freq_hz,
            carrier_freq=cfg.carrier_freq_hz,
        )
    else:
        raise ValueError(f"Unsupported modulation: {cfg.modulation}")

    # Channel propagation
    P_rx = channel.propagate(P_tx)

    # Photodetection
    I_ph = rx.optical_to_current(P_rx)

    # Add noise
    bandwidth = cfg.data_rate_bps / 2
    noise_model = NoiseModel(
        temperature_K=cfg.temperature_K,
        R_load=cfg.r_sense_ohm,
        ina_noise_nV=cfg.ina_noise_nV_rtHz if cfg.noise_enable else 0,
        rx_area_cm2=cfg.sc_area_cm2,
    )

    if cfg.noise_enable:
        noise = noise_model.generate_noise(len(I_ph), I_ph, bandwidth)
        I_ph_noisy = I_ph + noise
    else:
        I_ph_noisy = I_ph

    # TIA
    V_tia = rx.apply_tia(I_ph_noisy, t, R_tia=50e3, f_3db=min(bandwidth * 5, fs / 3))

    # Apply notch filter if configured
    if cfg.notch_freq_hz is not None:
        V_tia = rx.apply_notch(V_tia, t, f_notch=cfg.notch_freq_hz, Q=cfg.notch_Q)

    # Apply amplifier gain
    if cfg.amp_gain_linear > 1:
        V_tia = V_tia * cfg.amp_gain_linear

    # ========== DEMODULATION ==========
    demod = Demodulator(
        sample_rate=fs,
        hpf_cutoff=max(cfg.data_rate_bps * 0.01, 10),
        lpf_cutoff=min(cfg.data_rate_bps * 2, fs * 0.45),
    )

    if modulation in ('OOK',):
        bits_rx = demod.demodulate_ook(V_tia, n_bits, samples_per_bit)
    elif modulation == 'OOK_MANCHESTER':
        bits_rx = demod.demodulate_manchester(V_tia, n_bits, samples_per_bit)
    elif modulation == 'OFDM':
        # For OFDM, demod from the digital domain (pre-channel equalized)
        # Simplified: use the photocurrent signal directly
        sym_len = cfg.ofdm_nfft + cfg.ofdm_cp_len
        n_data = cfg.ofdm_nfft // 2 - 1
        n_sc = min(cfg.ofdm_n_subcarriers, n_data)
        bps = int(np.log2(cfg.ofdm_qam_order))

        # Re-do TX in digital domain for coherent demod
        ofdm_tx_signal = _generate_ofdm_digital(bits_tx, cfg.ofdm_qam_order,
                                                  cfg.ofdm_nfft, cfg.ofdm_cp_len,
                                                  n_sc)
        # Apply channel effect (scalar gain + noise)
        G = channel.channel_gain() * rx.R
        ofdm_rx_signal = ofdm_tx_signal * G
        if cfg.noise_enable:
            sigma = noise_model.total_noise_std(np.mean(I_ph), bandwidth)
            ofdm_rx_signal += np.random.normal(0, sigma * 50e3, len(ofdm_rx_signal))

        # Equalize (ZF: divide by known channel)
        if G > 0:
            ofdm_eq = ofdm_rx_signal / G
        else:
            ofdm_eq = ofdm_rx_signal

        bits_rx = demod.demodulate_ofdm(ofdm_eq, bits_tx, cfg.ofdm_qam_order,
                                         cfg.ofdm_nfft, cfg.ofdm_cp_len, n_sc)
    elif modulation == 'BFSK':
        bits_rx = demod.demodulate_bfsk(V_tia, n_bits, samples_per_bit,
                                         cfg.bfsk_f0_hz, cfg.bfsk_f1_hz, fs)
    elif modulation == 'PWM_ASK':
        # PWM-ASK: detect carrier envelope
        bits_rx = demod.demodulate_ook(V_tia, n_bits, samples_per_bit)
    else:
        bits_rx = np.zeros(n_bits, dtype=int)

    # BER calculation
    ber_result = Demodulator.calculate_ber(bits_tx, bits_rx)

    # SNR estimate
    signal_power = np.var(I_ph) if np.var(I_ph) > 0 else 1e-30
    noise_power = noise_model.total_noise_std(I_ph, bandwidth)**2
    snr_db = 10 * np.log10(signal_power / max(noise_power, 1e-30))

    return {
        'ber': ber_result['ber'],
        'n_errors': ber_result['n_errors'],
        'n_bits_tested': ber_result['n_bits_tested'],
        'snr_est_dB': snr_db,
        'time': t,
        'P_tx': P_tx,
        'P_rx': P_rx,
        'I_ph': I_ph,
        'V_rx': V_tia,
        'bits_tx': bits_tx,
        'bits_rx': bits_rx,
        'engine': 'python',
        'modulation': cfg.modulation,
        'channel_gain': channel.channel_gain(),
        'P_rx_avg_uW': np.mean(P_rx) * 1e6,
        'I_ph_avg_uA': np.mean(I_ph) * 1e6,
    }


def _generate_ofdm_digital(bits, qam_order, n_fft, cp_len, n_subcarriers):
    """Generate OFDM signal in digital domain for coherent demodulation."""
    tx = Transmitter(dc_bias_mA=100, mod_depth=0.5, led_efficiency=1.0)
    bps = int(np.log2(qam_order))
    bits_per_sym = n_subcarriers * bps
    n_symbols = len(bits) // bits_per_sym

    signal = []
    for s in range(n_symbols):
        frame_bits = bits[s * bits_per_sym:(s + 1) * bits_per_sym]
        qam_syms = tx._bits_to_qam(frame_bits, qam_order, n_subcarriers)
        freq = np.zeros(n_fft, dtype=complex)
        freq[1:1 + n_subcarriers] = qam_syms
        freq[n_fft - n_subcarriers:] = np.conj(qam_syms[::-1])
        sig = np.fft.ifft(freq).real
        sig_cp = np.concatenate([sig[-cp_len:], sig])
        signal.extend(sig_cp)

    return np.array(signal)
