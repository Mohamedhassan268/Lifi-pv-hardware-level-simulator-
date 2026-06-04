"""cosim/features.py — derived-feature extraction for ML / data engineering.

One authoritative function, `extract_features(result, cfg)`, turns a single
simulation result dict (from `cosim.python_engine.run_python_simulation` or the
pipeline) plus its `SystemConfig` into a flat, documented record of physical
and link-layer features. The same record feeds three surfaces:

  - the in-app "Link Analytics" panel,
  - the `POST /api/features` endpoint, and
  - the sweep dataset export (`cosim.feature_sweep`).

Every feature is registered in `FEATURES` with a human label, unit, group, and
one-line description so the UI and exported CSV/JSONL stay self-documenting.

Design notes / conventions:
  - All values are SI-base in the record (seconds, watts, Hz, bits/s); the
    FEATURES metadata names the display unit. Missing inputs yield NaN, never a
    crash — sweeps must not abort on one degenerate point.
  - Three latencies are reported, since "latency" is ambiguous for a PHY link:
    propagation (time-of-flight), DSP/processing (demod buffering + FEC block),
    and frame/end-to-end (delivering the full payload at the net rate).
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

try:  # erfcinv is only needed for the BER→Q mapping; degrade gracefully.
    from scipy.special import erfcinv as _erfcinv
except Exception:  # noqa: BLE001
    _erfcinv = None

_C = 299_792_458.0  # speed of light, m/s


# -----------------------------------------------------------------------------
# Feature registry — label / unit / group / description for every key emitted.
# Groups order the analytics panel and document the CSV columns.
# -----------------------------------------------------------------------------

FEATURES: dict[str, dict[str, str]] = {
    # --- Link quality ---
    "ber":                 {"label": "BER", "unit": "", "group": "Link quality", "desc": "Post-FEC bit error rate (or raw BER when FEC off)."},
    "ber_uncoded":         {"label": "BER (pre-FEC)", "unit": "", "group": "Link quality", "desc": "Channel BER before FEC decoding; equals BER when FEC off."},
    "n_errors":            {"label": "Bit errors", "unit": "bits", "group": "Link quality", "desc": "Number of erroneous bits measured."},
    "n_bits_tested":       {"label": "Bits tested", "unit": "bits", "group": "Link quality", "desc": "Number of bits scored for BER."},
    "ber_ci_low":          {"label": "BER 95% CI low", "unit": "", "group": "Link quality", "desc": "Wilson-score lower bound on BER at 95% confidence."},
    "ber_ci_high":         {"label": "BER 95% CI high", "unit": "", "group": "Link quality", "desc": "Wilson-score upper bound on BER at 95% confidence."},
    "snr_link_budget_dB":  {"label": "SNR (link budget)", "unit": "dB", "group": "Link quality", "desc": "Closed-form link-budget SNR at the demodulator input."},
    "snr_measured_dB":     {"label": "SNR (measured)", "unit": "dB", "group": "Link quality", "desc": "Two-run measured SNR (NaN unless measure_snr_enable)."},
    "eb_n0_dB":            {"label": "Eb/N0", "unit": "dB", "group": "Link quality", "desc": "Energy-per-bit to AWGN-floor noise density."},
    "q_factor":            {"label": "Q-factor", "unit": "", "group": "Link quality", "desc": "Quality factor inferred from BER (Q = √2·erfcinv(2·BER))."},
    "evm_percent":         {"label": "EVM", "unit": "%", "group": "Link quality", "desc": "RMS error-vector magnitude of OFDM symbols (NaN for non-OFDM)."},

    # --- Throughput ---
    "data_rate_bps":       {"label": "Gross data rate", "unit": "bps", "group": "Throughput", "desc": "Channel bit rate before FEC overhead."},
    "code_rate":           {"label": "FEC code rate", "unit": "", "group": "Throughput", "desc": "k/n payload fraction (1.0 when FEC off)."},
    "payload_rate_bps":    {"label": "Payload rate", "unit": "bps", "group": "Throughput", "desc": "Net message rate = gross rate × code rate."},
    "goodput_bps":         {"label": "Goodput", "unit": "bps", "group": "Throughput", "desc": "Error-free payload throughput = payload rate × (1 − BER)."},
    "occupied_bandwidth_hz": {"label": "Occupied BW", "unit": "Hz", "group": "Throughput", "desc": "Scheme occupied bandwidth (symbol/sample rate)."},
    "spectral_efficiency": {"label": "Spectral efficiency", "unit": "bps/Hz", "group": "Throughput", "desc": "Gross data rate per Hz of occupied bandwidth."},
    "noise_bandwidth_hz":  {"label": "Noise BW", "unit": "Hz", "group": "Throughput", "desc": "Matched-filter noise bandwidth used by the sim (Rb/2)."},

    # --- Latency (all three senses) ---
    "latency_propagation_s": {"label": "Propagation delay", "unit": "s", "group": "Latency", "desc": "Optical time-of-flight = distance / c."},
    "latency_dsp_s":         {"label": "DSP/processing latency", "unit": "s", "group": "Latency", "desc": "Demod buffering (symbol/OFDM-symbol + preamble) plus FEC codeword block."},
    "latency_frame_s":       {"label": "Frame latency", "unit": "s", "group": "Latency", "desc": "End-to-end delivery of the payload: payload/net rate + propagation + DSP."},

    # --- Power & energy (LiFi + PV harvester) ---
    "p_tx_avg_W":          {"label": "TX optical power", "unit": "W", "group": "Power & energy", "desc": "Mean transmitted optical power."},
    "p_rx_avg_W":          {"label": "RX optical power", "unit": "W", "group": "Power & energy", "desc": "Mean received optical power."},
    "channel_gain":        {"label": "Channel gain", "unit": "", "group": "Power & energy", "desc": "Optical channel DC gain (P_rx/P_tx)."},
    "path_loss_dB":        {"label": "Path loss", "unit": "dB", "group": "Power & energy", "desc": "Optical path loss = −10·log10(channel gain)."},
    "i_ph_avg_uA":         {"label": "Photocurrent", "unit": "µA", "group": "Power & energy", "desc": "Mean photodiode/PV photocurrent."},
    "harvested_power_W":   {"label": "Harvested power", "unit": "W", "group": "Power & energy", "desc": "Raw mean PV cell electrical output power (NaN if no PV model)."},
    "dcdc_output_power_W": {"label": "DC-DC output power", "unit": "W", "group": "Power & energy", "desc": "Load-regulated DC-DC converter output power (NaN if disabled). Lower than raw harvest; stays flat under regulation while raw harvest rises."},
    "dcdc_efficiency":     {"label": "DC-DC efficiency", "unit": "", "group": "Power & energy", "desc": "Harvester DC-DC converter efficiency (NaN if disabled)."},
    "energy_per_bit_J":    {"label": "Energy per bit", "unit": "J", "group": "Power & energy", "desc": "Received optical energy per bit = P_rx / data rate."},

    # --- Signal statistics ---
    "papr_dB":             {"label": "PAPR", "unit": "dB", "group": "Signal stats", "desc": "Peak-to-average power ratio of the TX optical waveform."},
    "modulation_depth":    {"label": "Modulation depth", "unit": "", "group": "Signal stats", "desc": "Optical modulation index used by the TX."},
    "v_rx_swing":          {"label": "RX swing", "unit": "V", "group": "Signal stats", "desc": "Peak-to-peak voltage swing at the demodulator input."},

    # --- Noise (total; per-source available via noise_breakdown) ---
    "noise_sigma_total":   {"label": "Noise σ (total)", "unit": "A", "group": "Noise", "desc": "Total RMS noise current at the receiver."},
}

# Categorical / identifier columns carried alongside the numeric features.
TAG_KEYS = ("modulation", "engine")


def _f(x) -> float:
    """Coerce to float, mapping None/failures to NaN."""
    try:
        if x is None:
            return float("nan")
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def _wilson_ci(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a proportion (BER). Robust at p≈0."""
    if n <= 0 or math.isnan(p):
        return float("nan"), float("nan")
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return max(0.0, centre - margin), min(1.0, centre + margin)


def _q_from_ber(ber: float) -> float:
    """Q = √2 · erfcinv(2·BER). BER=0 → ∞ (reported as NaN-safe large)."""
    if _erfcinv is None or math.isnan(ber):
        return float("nan")
    if ber <= 0:
        return float("inf")
    if ber >= 0.5:
        return 0.0
    return float(math.sqrt(2.0) * _erfcinv(2.0 * ber))


def _symbol_and_bandwidth(cfg) -> tuple[float, float]:
    """(symbol_rate, occupied_bandwidth) in Hz, scheme-aware.

    Line codes: occupied BW ≈ symbol rate (OOK=Rb, Manchester=2·Rb).
    OFDM: occupied BW ≈ the real DCO sample rate = Rb·(nfft+cp)/(n_sc·bps).
    """
    rb = _f(getattr(cfg, "data_rate_bps", float("nan")))
    scheme = str(getattr(cfg, "modulation", "OOK")).upper()
    if "MANCHESTER" in scheme:
        return 2 * rb, 2 * rb
    if scheme == "OFDM":
        nfft = _f(getattr(cfg, "ofdm_nfft", 256))
        cp = _f(getattr(cfg, "ofdm_cp_len", 0))
        qam = max(2.0, _f(getattr(cfg, "ofdm_qam_order", 16)))
        n_sc = max(1.0, _f(getattr(cfg, "ofdm_n_subcarriers", 80)))
        bps = max(1.0, math.log2(qam))
        fs = rb * (nfft + cp) / (n_sc * bps)
        return rb, fs
    return rb, rb  # OOK / BFSK / PWM_ASK ≈ Rb


def _evm_percent(result, cfg) -> float:
    """RMS EVM of equalized OFDM data symbols vs the nearest constellation point."""
    syms = result.get("ofdm_symbols")
    if not syms:
        return float("nan")
    try:
        from cosim.modulation import _constellation_for
        points, _ = _constellation_for(int(getattr(cfg, "ofdm_qam_order", 16)))
        s = np.asarray(syms, dtype=complex)
        ref = np.array([points[np.argmin(np.abs(p - points))] for p in s])
        err = np.sqrt(np.mean(np.abs(s - ref) ** 2))
        sig = np.sqrt(np.mean(np.abs(ref) ** 2))
        return float(100.0 * err / sig) if sig > 0 else float("nan")
    except Exception:  # noqa: BLE001
        return float("nan")


def _papr_dB(result) -> float:
    p = result.get("P_tx")
    if p is None:
        return float("nan")
    p = np.asarray(p, dtype=float)
    mean = p.mean()
    if mean <= 0:
        return float("nan")
    return float(10.0 * np.log10(p.max() / mean))


def extract_features(result: dict, cfg) -> dict:
    """Compute the full feature record from one run result + its SystemConfig.

    Returns a flat dict: every key in FEATURES (numeric, SI-base) plus the
    TAG_KEYS identifiers. Never raises on a degenerate run — missing pieces
    become NaN so sweeps stay row-complete.
    """
    rb = _f(getattr(cfg, "data_rate_bps", float("nan")))
    distance = _f(getattr(cfg, "distance_m", float("nan")))
    ber = _f(result.get("ber"))
    n_tested = int(result.get("n_bits_tested") or 0)

    symbol_rate, occ_bw = _symbol_and_bandwidth(cfg)

    # FEC code rate + net rates.
    fec_on = bool(getattr(cfg, "fec_enable", False))
    num = _f(getattr(cfg, "fec_rate_num", 1))
    den = _f(getattr(cfg, "fec_rate_den", 1))
    code_rate = (num / den) if (fec_on and den) else 1.0
    payload_rate = rb * code_rate
    goodput = payload_rate * (1.0 - ber) if not math.isnan(ber) else float("nan")

    # Latencies.
    lat_prop = distance / _C if not math.isnan(distance) else float("nan")
    # DSP/processing: buffer one demod unit, plus a FEC codeword block if on.
    scheme = str(getattr(cfg, "modulation", "OOK")).upper()
    if scheme == "OFDM":
        qam = max(2.0, _f(getattr(cfg, "ofdm_qam_order", 16)))
        n_sc = max(1.0, _f(getattr(cfg, "ofdm_n_subcarriers", 80)))
        bits_per_ofdm = n_sc * math.log2(qam)
        # one OFDM symbol + a ~1-symbol pilot preamble.
        lat_dsp = (2.0 * bits_per_ofdm / rb) if rb > 0 else float("nan")
    else:
        lat_dsp = (1.0 / symbol_rate) if symbol_rate > 0 else float("nan")
    if fec_on:
        cw = _f(getattr(cfg, "fec_codeword_n", 0))
        if rb > 0 and not math.isnan(cw):
            lat_dsp = (0.0 if math.isnan(lat_dsp) else lat_dsp) + cw / rb
    n_payload = int(getattr(cfg, "n_bits", n_tested) or n_tested)
    lat_frame = float("nan")
    if payload_rate > 0:
        lat_frame = n_payload / payload_rate
        lat_frame += (0.0 if math.isnan(lat_prop) else lat_prop)
        lat_frame += (0.0 if math.isnan(lat_dsp) else lat_dsp)

    # Power & energy.
    p_tx = result.get("P_tx")
    p_rx = result.get("P_rx")
    p_tx_avg = _f(np.mean(p_tx)) if p_tx is not None else float("nan")
    p_rx_avg = _f(np.mean(p_rx)) if p_rx is not None else _f(result.get("P_rx_avg_uW")) * 1e-6
    gain = _f(result.get("channel_gain"))
    path_loss = -10.0 * math.log10(gain) if gain and gain > 0 else float("nan")
    harvested = float("nan")
    v_cell, i_cell = result.get("V_cell"), result.get("I_cell")
    if v_cell is not None and i_cell is not None:
        harvested = _f(np.mean(np.asarray(v_cell) * np.asarray(i_cell)))
    energy_per_bit = (p_rx_avg / rb) if (rb > 0 and not math.isnan(p_rx_avg)) else float("nan")

    # Signal stats.
    v = result.get("V_rx")
    v_swing = _f(np.max(v) - np.min(v)) if v is not None else float("nan")

    # Noise total σ. Prefer an attached breakdown; else compute from the model.
    sigma_total = float("nan")
    nb = result.get("noise_breakdown")
    if isinstance(nb, dict):
        sigma_total = _f(nb.get("total_noise_std") or nb.get("sigma_total"))
    if math.isnan(sigma_total) and result.get("I_ph") is not None and rb > 0:
        try:
            from cosim.noise import NoiseModel
            sigma_total = _f(
                NoiseModel.from_config(cfg).total_noise_std(
                    np.asarray(result["I_ph"], dtype=float), rb
                )
            )
        except Exception:  # noqa: BLE001
            pass

    ci_low, ci_high = _wilson_ci(ber, n_tested)

    rec = {
        # Link quality
        "ber": ber,
        "ber_uncoded": _f(result.get("ber_uncoded", result.get("ber"))),
        "n_errors": _f(result.get("n_errors")),
        "n_bits_tested": float(n_tested),
        "ber_ci_low": ci_low,
        "ber_ci_high": ci_high,
        "snr_link_budget_dB": _f(result.get("snr_link_budget_dB", result.get("snr_est_dB"))),
        "snr_measured_dB": _f(result.get("snr_measured_dB")),
        "eb_n0_dB": _f(result.get("eb_n0_dB_awgn_floor")),
        "q_factor": _q_from_ber(ber),
        "evm_percent": _evm_percent(result, cfg),
        # Throughput
        "data_rate_bps": rb,
        "code_rate": code_rate,
        "payload_rate_bps": payload_rate,
        "goodput_bps": goodput,
        "occupied_bandwidth_hz": occ_bw,
        "spectral_efficiency": (rb / occ_bw) if occ_bw > 0 else float("nan"),
        "noise_bandwidth_hz": rb / 2.0 if rb > 0 else float("nan"),
        # Latency
        "latency_propagation_s": lat_prop,
        "latency_dsp_s": lat_dsp,
        "latency_frame_s": lat_frame,
        # Power & energy
        "p_tx_avg_W": p_tx_avg,
        "p_rx_avg_W": p_rx_avg,
        "channel_gain": gain,
        "path_loss_dB": path_loss,
        "i_ph_avg_uA": _f(result.get("I_ph_avg_uA")),
        "harvested_power_W": harvested,
        "dcdc_output_power_W": _f(result.get("dcdc_P_out_uW")) * 1e-6,
        "dcdc_efficiency": _f(result.get("dcdc_efficiency")),
        "energy_per_bit_J": energy_per_bit,
        # Signal stats
        "papr_dB": _papr_dB(result),
        "modulation_depth": _f(getattr(cfg, "modulation_depth", float("nan"))),
        "v_rx_swing": v_swing,
        # Noise
        "noise_sigma_total": sigma_total,
    }
    # Identifier tags.
    for k in TAG_KEYS:
        rec[k] = result.get(k, getattr(cfg, k, None))
    return rec
