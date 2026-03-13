# cosim/modulation.py
"""
Modulation and Demodulation Dispatch.

Unified interface for all 5 supported modulation schemes:
    OOK, OOK_Manchester, OFDM (DCO-OFDM), BFSK, PWM_ASK

Also provides analytical BER prediction formulas per scheme.

Usage:
    from cosim.modulation import modulate, demodulate, predict_ber

    P_tx = modulate('OOK', bits, t, config)
    bits_rx = demodulate('OOK', signal, t, n_bits, config)
    ber = predict_ber('OOK', snr_linear)
"""

import numpy as np
from scipy.special import erfc
from typing import Optional


# =============================================================================
# PHYSICAL CONSTANTS
# =============================================================================

SAMPLES_PER_BIT = 500  # Default oversampling ratio


# =============================================================================
# MANCHESTER CODEC
# =============================================================================

def manchester_encode(bits: np.ndarray) -> np.ndarray:
    """Manchester-encode: bit 1 -> [1,0], bit 0 -> [0,1]."""
    bits = np.asarray(bits, dtype=int)
    symbols = np.empty(2 * len(bits), dtype=int)
    symbols[0::2] = bits
    symbols[1::2] = 1 - bits
    return symbols


def manchester_decode(symbols: np.ndarray) -> np.ndarray:
    """Decode Manchester symbols: compare first/second half of each bit."""
    symbols = np.asarray(symbols, dtype=float)
    n_bits = len(symbols) // 2
    first_half = symbols[0::2][:n_bits]
    second_half = symbols[1::2][:n_bits]
    return (first_half > second_half).astype(int)


# =============================================================================
# BER PREDICTION (ANALYTICAL)
# =============================================================================

def predict_ber(scheme: str, snr_linear) -> np.ndarray:
    """
    Analytical BER prediction for a given modulation scheme.

    Args:
        scheme: Modulation scheme name (OOK, OOK_Manchester, OFDM, BFSK, PWM_ASK)
        snr_linear: SNR in linear scale (scalar or array)

    Returns:
        BER value(s)
    """
    snr = np.maximum(np.asarray(snr_linear, dtype=float), 0)
    scheme = scheme.upper().replace('-', '_')

    if scheme in ('OOK', 'OOK_MANCHESTER', 'PWM_ASK'):
        # OOK: BER = 0.5 * erfc(sqrt(SNR/2))
        return 0.5 * erfc(np.sqrt(snr / 2))
    elif scheme == 'BFSK':
        # Non-coherent BFSK: BER = 0.5 * exp(-SNR/2)
        return 0.5 * np.exp(-snr / 2)
    elif scheme == 'OFDM':
        # Default to 16-QAM for OFDM
        return predict_ber_mqam(snr, M=16)
    else:
        # Fallback to OOK
        return 0.5 * erfc(np.sqrt(snr / 2))


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


def predict_ber_mqam(snr_linear, M: int = 16):
    """
    Approximate BER for M-QAM.

    BER ≈ (4/k)(1 - 1/√M) · 0.5 · erfc(√(3·SNR/(M-1)) / √2)
    """
    if M <= 1:
        return np.zeros_like(np.asarray(snr_linear, dtype=float))
    k = np.log2(M)
    factor = (4 / k) * (1 - 1 / np.sqrt(M))
    snr = np.maximum(np.asarray(snr_linear, dtype=float), 0)
    arg = np.sqrt(3 * snr / (M - 1))
    return factor * 0.5 * erfc(arg / np.sqrt(2))


# =============================================================================
# MODULATORS
# =============================================================================

def modulate(scheme: str, bits: np.ndarray, t: np.ndarray,
             config=None, **kwargs) -> np.ndarray:
    """
    Unified modulation dispatch.

    Args:
        scheme: Modulation scheme (OOK, OOK_Manchester, OFDM, BFSK, PWM_ASK)
        bits: Bit array to modulate
        t: Time array (seconds)
        config: Optional SystemConfig for parameter extraction
        **kwargs: Override parameters

    Returns:
        P_tx: Transmitted optical power array (Watts)
    """
    scheme = scheme.upper().replace('-', '_')

    # Extract params from config or kwargs
    dc_bias_mA = kwargs.get('dc_bias_mA', _cfg_val(config, 'bias_current_A', 0.35) * 1000)
    mod_depth = kwargs.get('mod_depth', _cfg_val(config, 'modulation_depth', 0.33))
    led_eff = kwargs.get('led_efficiency', _cfg_val(config, 'led_gled', 0.88))
    I_dc = dc_bias_mA  # mA

    if scheme == 'OOK':
        return _modulate_ook(bits, t, I_dc, mod_depth, led_eff)
    elif scheme == 'OOK_MANCHESTER':
        return _modulate_manchester(bits, t, I_dc, mod_depth, led_eff)
    elif scheme == 'OFDM':
        qam_order = kwargs.get('qam_order', _cfg_val(config, 'ofdm_qam_order', 16))
        n_fft = kwargs.get('n_fft', _cfg_val(config, 'ofdm_nfft', 256))
        cp_len = kwargs.get('cp_len', _cfg_val(config, 'ofdm_cp_len', 32))
        return _modulate_ofdm(bits, t, I_dc, mod_depth, led_eff,
                              qam_order, n_fft, cp_len)
    elif scheme == 'BFSK':
        f0 = kwargs.get('f0', _cfg_val(config, 'bfsk_f0_hz', 1600.0))
        f1 = kwargs.get('f1', _cfg_val(config, 'bfsk_f1_hz', 2000.0))
        return _modulate_bfsk(bits, t, I_dc, mod_depth, led_eff, f0, f1)
    elif scheme == 'PWM_ASK':
        pwm_freq = kwargs.get('pwm_freq', _cfg_val(config, 'pwm_freq_hz', 10.0))
        carrier_freq = kwargs.get('carrier_freq', _cfg_val(config, 'carrier_freq_hz', 10000.0))
        return _modulate_pwm_ask(t, I_dc, mod_depth, led_eff, pwm_freq, carrier_freq)
    else:
        raise ValueError(f"Unsupported modulation scheme: {scheme}")


def _cfg_val(config, attr: str, default):
    """Safely get attribute from config, with fallback."""
    if config is None:
        return default
    return getattr(config, attr, default)


# =============================================================================
# MODULATOR IMPLEMENTATIONS
# =============================================================================

def _modulate_ook(bits, t, I_dc_mA, mod_depth, led_eff):
    """OOK modulation: P_tx = η · (I_dc + m·I_dc·d(t))."""
    sps = len(t) // len(bits)
    d = np.repeat(bits.astype(float), sps)[:len(t)]
    I_tx_mA = I_dc_mA + mod_depth * I_dc_mA * d
    return led_eff * (I_tx_mA * 1e-3)


def _modulate_manchester(bits, t, I_dc_mA, mod_depth, led_eff):
    """Manchester-encoded OOK."""
    sps = len(t) // len(bits)
    symbols = manchester_encode(bits)
    sps_sym = sps // 2
    d = np.repeat(symbols.astype(float), sps_sym)[:len(t)]
    I_tx_mA = I_dc_mA + mod_depth * I_dc_mA * d
    return led_eff * (I_tx_mA * 1e-3)


def _modulate_ofdm(bits, t, I_dc_mA, mod_depth, led_eff,
                   qam_order, n_fft, cp_len):
    """DCO-OFDM with Hermitian symmetry."""
    n_data_carriers = n_fft // 2 - 1
    bps = int(np.log2(qam_order))
    bits_per_symbol = n_data_carriers * bps
    n_symbols = max(1, len(bits) // bits_per_symbol)

    # Pad bits if needed
    bits_arr = np.asarray(bits, dtype=int)
    if len(bits_arr) < bits_per_symbol:
        bits_arr = np.concatenate([bits_arr, np.zeros(bits_per_symbol - len(bits_arr), dtype=int)])
    bits_used = bits_arr[:n_symbols * bits_per_symbol]

    ofdm_signal = []
    for s in range(n_symbols):
        frame_bits = bits_used[s * bits_per_symbol:(s + 1) * bits_per_symbol]
        qam_syms = _bits_to_qam(frame_bits, qam_order, n_data_carriers)

        # Hermitian symmetry for real-valued output
        freq = np.zeros(n_fft, dtype=complex)
        freq[1:n_fft // 2] = qam_syms
        freq[n_fft // 2 + 1:] = np.conj(qam_syms[::-1])

        sig = np.fft.ifft(freq).real
        sig_cp = np.concatenate([sig[-cp_len:], sig])
        ofdm_signal.extend(sig_cp)

    ofdm_signal = np.array(ofdm_signal)
    if np.std(ofdm_signal) > 0:
        ofdm_signal /= np.std(ofdm_signal)

    I_tx = I_dc_mA + ofdm_signal * mod_depth * I_dc_mA
    I_tx = np.maximum(I_tx, 0)

    # Interpolate to physics time
    indices = np.linspace(0, len(I_tx) - 1, len(t))
    I_tx_interp = np.interp(indices, np.arange(len(I_tx)), I_tx)
    return led_eff * (I_tx_interp * 1e-3)


def _modulate_bfsk(bits, t, I_dc_mA, mod_depth, led_eff, f0, f1,
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

    P_tx_dc = led_eff * (I_dc_mA * 1e-3)
    return lc_state * P_tx_dc


def _modulate_pwm_ask(t, I_dc_mA, mod_depth, led_eff,
                      pwm_freq, carrier_freq, duty=0.5):
    """PWM-ASK modulation (Correa 2025)."""
    phase_pwm = (t * pwm_freq) % 1.0
    pwm = (phase_pwm < duty).astype(float)

    phase_carrier = (t * carrier_freq) % 1.0
    carrier = (phase_carrier < 0.5).astype(float)

    signal_d = pwm * carrier
    level_hi = I_dc_mA * (1 + mod_depth)
    level_lo = I_dc_mA * (1 - mod_depth)
    I_tx = level_lo + (level_hi - level_lo) * signal_d
    return led_eff * (I_tx * 1e-3)


# =============================================================================
# QAM CONSTELLATION MAPPING
# =============================================================================

def _bits_to_qam(bits, qam_order, n_carriers):
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


def _qam_demap(symbol, qam_order):
    """Demap a single QAM symbol to bits."""
    if qam_order == 4:
        I = 1 if symbol.real > 0 else 0
        Q = 1 if symbol.imag > 0 else 0
        return [I, Q]
    elif qam_order in (16, 64):
        bps = int(np.log2(qam_order))
        half = bps // 2
        M = 2**half
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


# =============================================================================
# DEMODULATORS
# =============================================================================

def demodulate(scheme: str, signal_in: np.ndarray, t: np.ndarray,
               n_bits: int, config=None, **kwargs) -> np.ndarray:
    """
    Unified demodulation dispatch.

    Args:
        scheme: Modulation scheme
        signal_in: Received signal (voltage domain after TIA/amp)
        t: Time array
        n_bits: Expected number of bits
        config: Optional SystemConfig
        **kwargs: Override parameters

    Returns:
        bits_rx: Recovered bit array
    """
    scheme = scheme.upper().replace('-', '_')
    dt = t[1] - t[0]
    fs = 1.0 / dt
    sps = len(t) // n_bits
    data_rate = _cfg_val(config, 'data_rate_bps', n_bits / t[-1])

    if scheme in ('OOK', 'PWM_ASK'):
        return _demodulate_ook(signal_in, n_bits, sps, fs, data_rate)
    elif scheme == 'OOK_MANCHESTER':
        return _demodulate_manchester(signal_in, n_bits, sps, fs, data_rate)
    elif scheme == 'OFDM':
        bits_tx = kwargs.get('bits_tx', np.zeros(n_bits, dtype=int))
        qam_order = kwargs.get('qam_order', _cfg_val(config, 'ofdm_qam_order', 16))
        n_fft = kwargs.get('n_fft', _cfg_val(config, 'ofdm_nfft', 256))
        cp_len = kwargs.get('cp_len', _cfg_val(config, 'ofdm_cp_len', 32))
        n_sc = kwargs.get('n_subcarriers', _cfg_val(config, 'ofdm_n_subcarriers', 80))
        n_sc = min(n_sc, n_fft // 2 - 1)
        return _demodulate_ofdm(signal_in, bits_tx, qam_order, n_fft, cp_len, n_sc)
    elif scheme == 'BFSK':
        f0 = kwargs.get('f0', _cfg_val(config, 'bfsk_f0_hz', 1600.0))
        f1 = kwargs.get('f1', _cfg_val(config, 'bfsk_f1_hz', 2000.0))
        return _demodulate_bfsk(signal_in, n_bits, sps, f0, f1, fs)
    else:
        raise ValueError(f"Unsupported demodulation scheme: {scheme}")


def _demodulate_ook(signal_in, n_bits, sps, fs, data_rate):
    """OOK demodulation: HPF -> LPF -> sample -> threshold."""
    from scipy import signal as sp_signal

    # HPF to remove DC
    hpf_cutoff = max(data_rate * 0.01, 10)
    wn_hp = hpf_cutoff / (fs / 2)
    if 0 < wn_hp < 1:
        b, a = sp_signal.butter(4, wn_hp, btype='high')
        V_hpf = sp_signal.filtfilt(b, a, signal_in)
    else:
        V_hpf = signal_in

    # LPF for noise reduction
    lpf_cutoff = min(data_rate * 2, fs * 0.45)
    wn_lp = lpf_cutoff / (fs / 2)
    if 0 < wn_lp < 1:
        b, a = sp_signal.butter(4, wn_lp, btype='low')
        V_lpf = sp_signal.filtfilt(b, a, V_hpf)
    else:
        V_lpf = V_hpf

    # Sample at bit centers
    indices = np.arange(n_bits) * sps + sps // 2
    indices = np.clip(indices.astype(int), 0, len(V_lpf) - 1)
    samples = V_lpf[indices]
    threshold = (np.max(samples) + np.min(samples)) / 2
    return (samples > threshold).astype(int)


def _demodulate_manchester(signal_in, n_bits, sps, fs, data_rate):
    """Manchester demodulation: compare half-bit energies."""
    from scipy import signal as sp_signal

    # HPF + LPF
    hpf_cutoff = max(data_rate * 0.01, 10)
    wn_hp = hpf_cutoff / (fs / 2)
    if 0 < wn_hp < 1:
        b, a = sp_signal.butter(4, wn_hp, btype='high')
        V = sp_signal.filtfilt(b, a, signal_in)
    else:
        V = signal_in

    lpf_cutoff = min(data_rate * 2, fs * 0.45)
    wn_lp = lpf_cutoff / (fs / 2)
    if 0 < wn_lp < 1:
        b, a = sp_signal.butter(4, wn_lp, btype='low')
        V = sp_signal.filtfilt(b, a, V)

    bits_rx = np.zeros(n_bits, dtype=int)
    for i in range(n_bits):
        idx1 = min(i * sps + sps // 4, len(V) - 1)
        idx2 = min(i * sps + 3 * sps // 4, len(V) - 1)
        bits_rx[i] = 1 if V[idx1] > V[idx2] else 0
    return bits_rx


def _demodulate_ofdm(signal_rx, bits_tx, qam_order, n_fft, cp_len, n_subcarriers):
    """OFDM demodulation: remove CP, FFT, QAM demap."""
    sym_len = n_fft + cp_len
    n_symbols = len(signal_rx) // sym_len

    bits_rx = []
    for s in range(n_symbols):
        sym = signal_rx[s * sym_len:(s + 1) * sym_len]
        sig = sym[cp_len:]
        freq = np.fft.fft(sig)
        data_carriers = freq[1:n_fft // 2][:n_subcarriers]
        for qam_sym in data_carriers:
            bits_rx.extend(_qam_demap(qam_sym, qam_order))

    return np.array(bits_rx[:len(bits_tx)], dtype=int)


def _demodulate_bfsk(signal_in, n_bits, sps, f0, f1, fs):
    """Non-coherent BFSK demodulation using Goertzel energy detection."""
    bits_rx = np.zeros(n_bits, dtype=int)
    for i in range(n_bits):
        start = i * sps
        end = min(start + sps, len(signal_in))
        segment = signal_in[start:end]
        if len(segment) == 0:
            continue
        e0 = _goertzel_energy(segment, f0, fs)
        e1 = _goertzel_energy(segment, f1, fs)
        bits_rx[i] = 1 if e1 > e0 else 0
    return bits_rx


def _goertzel_energy(x, target_freq, fs):
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


# =============================================================================
# OFDM DIGITAL DOMAIN HELPER
# =============================================================================

def generate_ofdm_digital(bits, qam_order, n_fft, cp_len, n_subcarriers):
    """Generate OFDM signal in digital domain for coherent demodulation."""
    bps = int(np.log2(qam_order))
    bits_per_sym = n_subcarriers * bps
    n_symbols = len(bits) // bits_per_sym

    signal = []
    for s in range(n_symbols):
        frame_bits = bits[s * bits_per_sym:(s + 1) * bits_per_sym]
        qam_syms = _bits_to_qam(frame_bits, qam_order, n_subcarriers)
        freq = np.zeros(n_fft, dtype=complex)
        freq[1:1 + n_subcarriers] = qam_syms
        freq[n_fft - n_subcarriers:] = np.conj(qam_syms[::-1])
        sig = np.fft.ifft(freq).real
        sig_cp = np.concatenate([sig[-cp_len:], sig])
        signal.extend(sig_cp)

    return np.array(signal)


# =============================================================================
# BER CALCULATION
# =============================================================================

def calculate_ber(bits_tx: np.ndarray, bits_rx: np.ndarray) -> dict:
    """
    Calculate Bit Error Rate.

    Returns:
        dict with 'ber', 'n_errors', 'n_bits_tested'
    """
    n = min(len(bits_tx), len(bits_rx))
    if n == 0:
        return {'ber': 0.0, 'n_errors': 0, 'n_bits_tested': 0}
    errors = int(np.sum(bits_tx[:n] != bits_rx[:n]))
    return {'ber': errors / n, 'n_errors': errors, 'n_bits_tested': n}
