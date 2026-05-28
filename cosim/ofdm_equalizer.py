# cosim/ofdm_equalizer.py
"""
Frequency-domain channel response for OFDM equalization.

Composes the chain's complex transfer function H(f) so that the OFDM
demodulator can divide each subcarrier FFT bin by H(f_k) and recover
symbols that land on the QAM grid regardless of LED roll-off, PV
capacitance, or amplifier bandwidth limits.

For the OFDM presets today (rx_topology='direct') the chain reduces to

    H(f) = H_LED(f) · H_PV(f)

where:
    H_LED(f) = 1 / (1 + j f/f_LED)         single-pole at the LED 3 dB BW
                                            (when led_bandwidth_limit_enable)
    H_PV(f) = 1 / (1 + j ω R_sense C_j)    PV-cap × sense-resistor RC pole

For non-direct topologies (currently unused with OFDM) the INA pole and
BPF cascade are folded in too — same per-stage shapes already used by
`cosim/ac_analysis.py`.

The DC gain is normalised out by the demodulator's AGC; this module
returns frequency-dependent shape only.
"""

from __future__ import annotations

import math

import numpy as np


def chain_response_at(cfg, freqs: np.ndarray) -> np.ndarray:
    """Complex H(f) of the TX → channel → RX chain, evaluated at ``freqs``.

    Frequency-dependent shape only — DC gain is absorbed by the demod AGC.
    Magnitudes near zero are clamped to a tiny epsilon so the caller's
    division stays finite (a "dead" subcarrier ends up boosted into pure
    noise, which is the physically correct outcome).
    """
    freqs = np.asarray(freqs, dtype=float)
    omega = 2.0 * np.pi * freqs

    H = np.ones_like(freqs, dtype=complex)

    # LED bandwidth (TX side). Only when bandwidth-limit is enabled, otherwise
    # the engine treats the LED as ideal and the equalizer should match.
    if getattr(cfg, "led_bandwidth_limit_enable", False):
        f_led = _led_bandwidth_hz(cfg)
        if f_led > 0:
            H = H * (1.0 / (1.0 + 1j * freqs / f_led))

    # PV cell capacitance + load (R_sense) → RC low-pass.
    R = float(getattr(cfg, "r_sense_ohm", 0.0))
    C = float(getattr(cfg, "sc_cj_nF", 0.0)) * 1e-9
    if R > 0 and C > 0:
        H = H * (1.0 / (1.0 + 1j * omega * R * C))

    # Non-direct topologies pull in the INA pole + BPF cascade. Same shapes
    # used by cosim/ac_analysis.py — reuse the helpers to avoid drift.
    topology = getattr(cfg, "rx_topology", "direct")
    if topology != "direct":
        from cosim.ac_analysis import _ina_tf, _hp_first_order, _lp_first_order

        ina_gain_db = getattr(cfg, "ina_gain_dB", 0.0)
        if ina_gain_db > 0:
            ina_gain_lin = 10.0 ** (ina_gain_db / 20.0)
            ina_gbw_hz = float(getattr(cfg, "ina_gbw_kHz", 700.0)) * 1e3
            H = H * _ina_tf(omega, ina_gain_lin, ina_gbw_hz)
        bpf_stages = int(getattr(cfg, "bpf_stages", 0))
        if bpf_stages > 0:
            f_low = float(getattr(cfg, "bpf_f_low_Hz", 1.0))
            f_high = float(getattr(cfg, "bpf_f_high_Hz", 1e6))
            for _ in range(bpf_stages):
                H = H * _hp_first_order(omega, f_low)
                H = H * _lp_first_order(omega, f_high)

    eps = 1e-30
    mag = np.abs(H)
    too_small = mag < eps
    if np.any(too_small):
        H = np.where(too_small, eps * np.exp(1j * np.angle(H)), H)
    return H


def subcarrier_frequencies(fs: float, n_fft: int, n_subcarriers: int) -> np.ndarray:
    """Subcarrier centre frequencies for an OFDM symbol.

    Matches ``_demodulate_ofdm``'s ``freq[1:n_fft//2][:n_subcarriers]``
    slicing: the k-th data carrier sits at f_k = k · fs / n_fft for
    k = 1, 2, …, n_subcarriers.
    """
    k = np.arange(1, n_subcarriers + 1, dtype=float)
    return k * fs / float(n_fft)


def _led_bandwidth_hz(cfg) -> float:
    """LED modulation 3 dB bandwidth (Hz), 0 if it can't be resolved."""
    try:
        from components import get_component

        led = get_component(getattr(cfg, "led_part", ""))
    except Exception:
        return 0.0
    if not hasattr(led, "modulation_bandwidth"):
        return 0.0
    try:
        bw = float(led.modulation_bandwidth())
    except Exception:
        return 0.0
    return bw if math.isfinite(bw) and bw > 0 else 0.0
