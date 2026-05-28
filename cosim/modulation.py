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

import math

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

    # Determine optical power base: prefer led_radiated_power_mW (direct optical
    # spec from datasheet/paper), fall back to I_dc * led_gled (derived optical
    # power from drive current + LED W/A coupling).
    mod_depth = kwargs.get('mod_depth', _cfg_val(config, 'modulation_depth', 0.33))
    P_led_mW = kwargs.get('P_led_mW', _cfg_val(config, 'led_radiated_power_mW', 0.0))

    if P_led_mW and P_led_mW > 0:
        # Route 1: radiated-power-first. P_led is the DC optical power at bias.
        # _modulate_* expect a dc_current (mA) and W/A efficiency whose product
        # equals P_led. Feed P_led_mW as the "current" and 1.0 as "efficiency".
        I_dc = P_led_mW
        led_eff = 1.0
    else:
        # Route 2: derive from bias current and LED coupling efficiency.
        dc_bias_mA = kwargs.get('dc_bias_mA',
                                _cfg_val(config, 'bias_current_A', 0.35) * 1000)
        I_dc = dc_bias_mA
        led_eff = kwargs.get('led_efficiency', _cfg_val(config, 'led_gled', 0.88))

    if scheme == 'OOK':
        return _modulate_ook(bits, t, I_dc, mod_depth, led_eff)
    elif scheme == 'OOK_MANCHESTER':
        return _modulate_manchester(bits, t, I_dc, mod_depth, led_eff)
    elif scheme == 'OFDM':
        qam_order = kwargs.get('qam_order', _cfg_val(config, 'ofdm_qam_order', 16))
        n_fft = kwargs.get('n_fft', _cfg_val(config, 'ofdm_nfft', 256))
        cp_len = kwargs.get('cp_len', _cfg_val(config, 'ofdm_cp_len', 32))
        n_sc = kwargs.get('n_subcarriers',
                          _cfg_val(config, 'ofdm_n_subcarriers', n_fft // 2 - 1))
        n_sc = min(n_sc, n_fft // 2 - 1)
        return _modulate_ofdm(bits, t, I_dc, mod_depth, led_eff,
                              qam_order, n_fft, cp_len, n_sc)
    elif scheme == 'BFSK':
        f0 = kwargs.get('f0', _cfg_val(config, 'bfsk_f0_hz', 1600.0))
        f1 = kwargs.get('f1', _cfg_val(config, 'bfsk_f1_hz', 2000.0))
        return _modulate_bfsk(bits, t, I_dc, mod_depth, led_eff, f0, f1)
    elif scheme == 'PWM_ASK':
        pwm_freq = kwargs.get('pwm_freq', _cfg_val(config, 'pwm_freq_hz', 10.0))
        carrier_freq = kwargs.get('carrier_freq', _cfg_val(config, 'carrier_freq_hz', 10000.0))
        return _modulate_pwm_ask(bits, t, I_dc, mod_depth, led_eff, pwm_freq, carrier_freq)
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
    """OOK modulation: P_tx = η · I_dc · (1 + m·(d(t) − 0.5)).

    The modulation is centred so the average optical power equals I_dc
    (the radiated power), with peak-to-peak AC swing m·I_dc.
    """
    sps = len(t) // len(bits)
    d = np.repeat(bits.astype(float), sps)[:len(t)]
    I_tx_mA = I_dc_mA * (1.0 + mod_depth * (d - 0.5))
    return led_eff * (I_tx_mA * 1e-3)


def _modulate_manchester(bits, t, I_dc_mA, mod_depth, led_eff):
    """Manchester-encoded OOK (centred; average optical power = I_dc)."""
    sps = len(t) // len(bits)
    symbols = manchester_encode(bits)
    sps_sym = sps // 2
    d = np.repeat(symbols.astype(float), sps_sym)[:len(t)]
    I_tx_mA = I_dc_mA * (1.0 + mod_depth * (d - 0.5))
    return led_eff * (I_tx_mA * 1e-3)


def _modulate_ofdm(bits, t, I_dc_mA, mod_depth, led_eff,
                   qam_order, n_fft, cp_len, n_subcarriers=None):
    """DCO-OFDM with Hermitian symmetry.

    n_subcarriers: number of active data carriers (default: n_fft//2 - 1).
    Must match the demodulator's n_subcarriers so that bits_per_symbol and
    symbol count are consistent on both ends.
    """
    n_data_carriers = min(n_subcarriers, n_fft // 2 - 1) if n_subcarriers else n_fft // 2 - 1
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

        # Hermitian symmetry for real-valued output; place data on carriers
        # 1..n_data_carriers and mirror on the negative-frequency side.
        freq = np.zeros(n_fft, dtype=complex)
        freq[1:n_data_carriers + 1] = qam_syms
        freq[n_fft - n_data_carriers:] = np.conj(qam_syms[::-1])

        sig = np.fft.ifft(freq).real
        sig_cp = np.concatenate([sig[-cp_len:], sig])
        ofdm_signal.extend(sig_cp)

    ofdm_signal = np.array(ofdm_signal)
    if np.std(ofdm_signal) > 0:
        ofdm_signal /= np.std(ofdm_signal)

    I_tx = I_dc_mA + ofdm_signal * mod_depth * I_dc_mA
    I_tx = np.maximum(I_tx, 0)

    # Upsample to physics timeline using ideal (FFT-based) resampling so that
    # subcarrier amplitudes are preserved across the full bandwidth.  Linear
    # interpolation (np.interp) acts as a triangular low-pass filter that
    # attenuates high subcarriers by up to ~15%, preventing clean equalization
    # on the receive side.
    from scipy.signal import resample as sp_resample
    I_tx_interp = sp_resample(I_tx, len(t))
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


def _modulate_pwm_ask(bits, t, I_dc_mA, mod_depth, led_eff,
                      pwm_freq, carrier_freq):
    """PWM-ASK modulation (Correa 2025).

    Correa's greenhouse scheme: the LED brightness is PWM'd at pwm_freq for
    agricultural dimming, and data bits are ASK-modulated on a high-freq
    carrier within the PWM-ON windows. The simulator focuses on the data
    channel: each bit gates the carrier (bit=1 -> carrier present,
    bit=0 -> suppressed). The slow PWM envelope is effectively "ON" during
    transmission; its effect on BER is captured by pwm duty in the
    theoretical BER, not by skipping bits here.
    """
    n_bits = len(bits)
    T = t[-1] - t[0] if len(t) > 1 else 1.0
    bit_duration = T / n_bits
    bit_idx = np.clip((t / bit_duration).astype(int), 0, n_bits - 1)
    data_envelope = bits[bit_idx].astype(float)

    phase_carrier = (t * carrier_freq) % 1.0
    carrier = (phase_carrier < 0.5).astype(float)

    signal_d = data_envelope * carrier
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


# Per-QAM-order constellation cache: (points[N], labels[N, bps]).
# `points` are the complex symbols, `labels` the bit pattern each point
# encodes — derived by inverting _bits_to_qam so the soft demap matches
# the hard demap bit-for-bit.
_QAM_CACHE: dict[int, tuple[np.ndarray, np.ndarray]] = {}


def _constellation_for(qam_order: int) -> tuple[np.ndarray, np.ndarray]:
    cached = _QAM_CACHE.get(qam_order)
    if cached is not None:
        return cached

    bps = int(np.log2(qam_order))
    n_points = qam_order
    points = np.empty(n_points, dtype=complex)
    labels = np.empty((n_points, bps), dtype=np.int8)
    for v in range(n_points):
        # Enumerate bit pattern MSB-first to match _bits_to_qam input order.
        bits = np.array([(v >> (bps - 1 - k)) & 1 for k in range(bps)],
                        dtype=np.int8)
        # _bits_to_qam yields exactly one symbol for n_carriers=1.
        sym = _bits_to_qam(bits, qam_order, 1)[0]
        points[v] = sym
        labels[v] = bits

    _QAM_CACHE[qam_order] = (points, labels)
    return points, labels


def _qam_demap_soft(symbol: complex, qam_order: int,
                    n0: float = 1.0) -> list[float]:
    """Per-bit log-likelihood ratios via max-log-MAP.

    Returns one LLR per bit in the order produced by `_qam_demap`.
    Sign convention matches pyldpc: positive LLR -> bit=0 is more likely.

    The closed-form QPSK branch is `4*Re/N0/sqrt(2)`-style on the I and Q
    decisions; larger orders fall through to the min-distance loop over
    the cached constellation grid.
    """
    if qam_order == 4:
        # Bit ordering: [I, Q] (see _qam_demap). Hard demap reads bit=1 when
        # Re/Im > 0 (the +1 PAM point). max-log-MAP across the 4-point grid
        # collapses to L = (min_{bit=1} d² − min_{bit=0} d²)/N0
        #              = (re − 1/√2)² − (re + 1/√2)²   (Q drops out)
        #              = −2√2 · Re(y).
        # Positive LLR => bit 0 more likely (matches pyldpc convention).
        s2 = np.sqrt(2.0)
        return [-(2.0 * s2 * symbol.real) / n0,
                -(2.0 * s2 * symbol.imag) / n0]

    if qam_order in (16, 64):
        points, labels = _constellation_for(qam_order)
        # Distance² from y to every constellation point.
        d2 = np.abs(symbol - points) ** 2
        bps = labels.shape[1]
        llrs: list[float] = []
        for k in range(bps):
            mask0 = labels[:, k] == 0
            min_d2_0 = d2[mask0].min()
            min_d2_1 = d2[~mask0].min()
            # max-log-MAP: L = (min_{bit=1} d² - min_{bit=0} d²) / N0
            llrs.append((min_d2_1 - min_d2_0) / n0)
        return llrs

    # BPSK fallback.
    return [(2.0 * symbol.real) / n0]


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

    if scheme == 'OOK':
        return _demodulate_ook(signal_in, n_bits, sps, fs, data_rate)
    elif scheme == 'PWM_ASK':
        return _demodulate_pwm_ask(signal_in, n_bits, sps)
    elif scheme == 'OOK_MANCHESTER':
        return _demodulate_manchester(signal_in, n_bits, sps, fs, data_rate)
    elif scheme == 'OFDM':
        bits_tx = kwargs.get('bits_tx', np.zeros(n_bits, dtype=int))
        qam_order = kwargs.get('qam_order', _cfg_val(config, 'ofdm_qam_order', 16))
        n_fft = kwargs.get('n_fft', _cfg_val(config, 'ofdm_nfft', 256))
        cp_len = kwargs.get('cp_len', _cfg_val(config, 'ofdm_cp_len', 32))
        n_sc = kwargs.get('n_subcarriers', _cfg_val(config, 'ofdm_n_subcarriers', 80))
        n_sc = min(n_sc, n_fft // 2 - 1)
        return _demodulate_ofdm(signal_in, bits_tx, qam_order, n_fft, cp_len, n_sc,
                                symbols_out=kwargs.get('symbols_out'),
                                llr_out=kwargs.get('llr_out'),
                                equalizer=kwargs.get('equalizer'))
    elif scheme == 'BFSK':
        f0 = kwargs.get('f0', _cfg_val(config, 'bfsk_f0_hz', 1600.0))
        f1 = kwargs.get('f1', _cfg_val(config, 'bfsk_f1_hz', 2000.0))
        return _demodulate_bfsk(signal_in, n_bits, sps, f0, f1, fs,
                                energies_out=kwargs.get('energies_out'))
    else:
        raise ValueError(f"Unsupported demodulation scheme: {scheme}")


def _demodulate_ook(signal_in, n_bits, sps, fs, data_rate):
    """OOK demodulation: HPF -> LPF -> sample -> threshold.

    Skips the HPF/LPF stages when the input is already a binary comparator
    output (only two unique levels), since filtfilt introduces edge
    artifacts that corrupt bits near the stream boundaries.
    """
    from scipy import signal as sp_signal

    # Detect binary comparator output: already-digitized, no need to filter.
    unique_vals = np.unique(signal_in)
    already_binary = len(unique_vals) <= 2

    if already_binary:
        V_lpf = signal_in
    else:
        # HPF to remove DC. Use SOS form — (b, a) polynomial form is
        # numerically unstable for the very low normalized frequencies that
        # come from slow data rates against MHz-class sample rates
        # (e.g. wn ~ 4e-5 for 2.5 kbps @ 1.25 MHz fs). SOS keeps the
        # filter stable; (b, a) silently returns coefficients that diverge
        # under filtfilt.
        hpf_cutoff = max(data_rate * 0.01, 10)
        wn_hp = hpf_cutoff / (fs / 2)
        if 0 < wn_hp < 1:
            sos = sp_signal.butter(4, wn_hp, btype='high', output='sos')
            V_hpf = sp_signal.sosfiltfilt(sos, signal_in)
        else:
            V_hpf = signal_in

        # LPF for noise reduction
        lpf_cutoff = min(data_rate * 2, fs * 0.45)
        wn_lp = lpf_cutoff / (fs / 2)
        if 0 < wn_lp < 1:
            sos = sp_signal.butter(4, wn_lp, btype='low', output='sos')
            V_lpf = sp_signal.sosfiltfilt(sos, V_hpf)
        else:
            V_lpf = V_hpf

    # Sample at bit centers
    indices = np.arange(n_bits) * sps + sps // 2
    indices = np.clip(indices.astype(int), 0, len(V_lpf) - 1)
    samples = V_lpf[indices]
    threshold = (np.max(samples) + np.min(samples)) / 2
    return (samples > threshold).astype(int)


def _demodulate_pwm_ask(signal_in, n_bits, sps):
    """PWM-ASK demodulation via envelope (per-bit mean) + threshold.

    The PWM+carrier product makes point-sampling unreliable, so we average
    the signal over each bit interval (simple envelope) then threshold.
    """
    indices = np.arange(n_bits + 1) * sps
    indices = np.clip(indices.astype(int), 0, len(signal_in))
    envelopes = np.array([
        np.mean(signal_in[indices[i]:indices[i+1]])
        if indices[i+1] > indices[i] else 0.0
        for i in range(n_bits)
    ])
    threshold = (envelopes.max() + envelopes.min()) / 2
    return (envelopes > threshold).astype(int)


def _demodulate_manchester(signal_in, n_bits, sps, fs, data_rate):
    """Manchester demodulation: compare half-bit energies."""
    from scipy import signal as sp_signal

    # HPF + LPF. SOS form — the (b, a) polynomial form silently produces
    # divergent output under filtfilt for the extreme low normalized
    # frequencies that come from slow data rates against MHz-class fs
    # (e.g. wn ~ 4e-5 for 2.5 kbps @ 1.25 MHz). See _demodulate_ook.
    hpf_cutoff = max(data_rate * 0.01, 10)
    wn_hp = hpf_cutoff / (fs / 2)
    if 0 < wn_hp < 1:
        sos = sp_signal.butter(4, wn_hp, btype='high', output='sos')
        V = sp_signal.sosfiltfilt(sos, signal_in)
    else:
        V = signal_in

    lpf_cutoff = min(data_rate * 2, fs * 0.45)
    wn_lp = lpf_cutoff / (fs / 2)
    if 0 < wn_lp < 1:
        sos = sp_signal.butter(4, wn_lp, btype='low', output='sos')
        V = sp_signal.sosfiltfilt(sos, V)

    bits_rx = np.zeros(n_bits, dtype=int)
    for i in range(n_bits):
        idx1 = min(i * sps + sps // 4, len(V) - 1)
        idx2 = min(i * sps + 3 * sps // 4, len(V) - 1)
        bits_rx[i] = 1 if V[idx1] > V[idx2] else 0
    return bits_rx


def _demodulate_ofdm(signal_rx, bits_tx, qam_order, n_fft, cp_len, n_subcarriers,
                     symbols_out=None, llr_out=None, equalizer=None):
    """OFDM demodulation: remove CP, FFT, equalize, QAM demap.

    Args:
        signal_rx: time-domain received signal (real or complex). When fed
            from the analog RX chain it should be DC-centred (mean removed)
            before being passed in.
        equalizer: optional complex array of length ``n_subcarriers`` —
            ``chain_response_at(cfg, subcarrier_frequencies(...))``. Each
            FFT bin is divided by the corresponding ``equalizer[k]`` before
            symbol collection, demap, and N0 estimation, so the channel's
            frequency-domain shape (LED roll-off, PV RC pole, INA pole,
            BPF cascade) is undone.

    After equalization a per-run AGC scales all symbols so the average
    received power matches the constellation's unit grid power. This
    keeps the I/Q scatter visually anchored to the textbook QAM grid
    regardless of upstream amplitude (V_rx in volts vs the unit-normalised
    digital bypass path).

    If `symbols_out` is provided, the equalized + AGC'd complex symbols
    are appended to it (the natural I/Q constellation for OFDM/QAM).

    If `llr_out` is provided, per-bit LLRs (max-log-MAP, sign convention
    matching pyldpc: positive => bit 0) are appended. N0 is estimated
    empirically from the deviation of each received symbol from its
    nearest constellation point — robust to upstream noise-model changes.
    """
    sym_len = n_fft + cp_len
    n_symbols = len(signal_rx) // sym_len

    bits_rx: list[int] = []
    all_symbols: list[complex] = []

    if equalizer is not None:
        eq = np.asarray(equalizer, dtype=complex)
    else:
        eq = None

    # First pass: collect equalized symbols (don't demap yet — we need
    # to compute the AGC scale before final symbol values are decided).
    raw_symbols: list[np.ndarray] = []
    for s in range(n_symbols):
        sym = signal_rx[s * sym_len:(s + 1) * sym_len]
        sig = sym[cp_len:]
        freq = np.fft.fft(sig)
        data_carriers = freq[1:n_fft // 2][:n_subcarriers]
        if eq is not None:
            # Broadcast: eq has length n_subcarriers; data_carriers same.
            data_carriers = data_carriers / eq[:len(data_carriers)]
        raw_symbols.append(data_carriers)

    if not raw_symbols:
        return np.array([], dtype=int)

    all_complex = np.concatenate(raw_symbols)

    # Blind AGC: scale so average power matches the unit constellation.
    # Constellation has E[|s|²]=1 by construction (see _bits_to_qam).
    mean_power = float(np.mean(np.abs(all_complex) ** 2))
    if mean_power > 0:
        scale = 1.0 / math.sqrt(mean_power)
    else:
        scale = 1.0
    all_complex = all_complex * scale

    # Second pass: demap + LLR + collection.
    for qam_sym in all_complex:
        bits_rx.extend(_qam_demap(complex(qam_sym), qam_order))
        if symbols_out is not None:
            symbols_out.append(complex(qam_sym))
        if llr_out is not None:
            all_symbols.append(complex(qam_sym))

    if llr_out is not None and all_symbols:
        n0 = _estimate_n0(all_symbols, qam_order)
        for qam_sym in all_symbols:
            llr_out.extend(_qam_demap_soft(qam_sym, qam_order, n0))

    return np.array(bits_rx[:len(bits_tx)], dtype=int)


def _estimate_n0(symbols, qam_order: int) -> float:
    """Empirical per-real-dimension noise variance from the symbol cloud.

    For each received symbol y, find the nearest constellation point s
    and accumulate |y-s|^2. Mean over both real dimensions gives an
    estimate of 2*sigma^2 = N0 (with our normalization E[|s|^2]=1).
    Clamped to a small positive floor so the LLR division stays finite
    in the noiseless edge case.
    """
    points, _ = _constellation_for(qam_order)
    err_sq = 0.0
    for y in symbols:
        d2 = np.abs(y - points) ** 2
        err_sq += float(d2.min())
    mean_err_sq = err_sq / max(len(symbols), 1)
    return max(mean_err_sq, 1e-12)


def _demodulate_bfsk(signal_in, n_bits, sps, f0, f1, fs, energies_out=None):
    """Non-coherent BFSK demodulation using Goertzel energy detection.

    If `energies_out` is provided, per-bit (E0, E1) pairs are appended.
    BFSK has no true I/Q constellation (non-coherent), but the (E0, E1)
    plane is the equivalent decision diagram: bit=0 lands above the
    diagonal, bit=1 below.
    """
    bits_rx = np.zeros(n_bits, dtype=int)
    for i in range(n_bits):
        start = i * sps
        end = min(start + sps, len(signal_in))
        segment = signal_in[start:end]
        if len(segment) == 0:
            continue
        e0 = _goertzel_energy(segment, f0, fs)
        e1 = _goertzel_energy(segment, f1, fs)
        if energies_out is not None:
            energies_out.append((float(e0), float(e1)))
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
