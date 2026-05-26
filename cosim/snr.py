# cosim/snr.py
"""
Signal-to-noise-ratio quantities — the scientific way to report them.

Three definitions live here. The simulator should report all three so the
gap between them is itself diagnostic.

  1. Link-budget SNR
       SNR_lb = var(I_ph_clean) / Σ σ_k²
     Closed-form, fast, **does not see** unmodelled sources or the chain's
     filtering. It is an upper bound, not a measurement.

  2. Measured SNR (the one that drives BER)
       SNR_meas|_node = var(V_node^signal_only) / var(V_node^noise_only)
     Computed by the canonical two-run technique:
       - reference run with all noise sources disabled (same seed)
       - full run with noise on (same seed)
       noise_only = full − reference
     This automatically captures every noise source the model emits *and*
     every filter the chain applies (notch, BPF, comparator, ...).
     Same technique a vector signal analyser uses in the lab.

  3. Eb/N0 (only valid for the AWGN floor)
       Eb / N0 = (P_signal × T_bit) / (σ_awgn² / B_n)
     Bandwidth-independent. Comparable to textbook BER curves. Reported
     **only against the AWGN portion** of the noise budget — for 1/f and
     mains flicker N₀ is not flat, so the quantity isn't well-defined for
     those sources and we don't try to fold them in.

The simulator's existing `snr_est_dB` field is renamed `snr_link_budget_dB`
in the result dict; `snr_measured_dB` and `eb_n0_dB_awgn_floor` are added
alongside it when `cfg.measure_snr_enable` is True.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Dict, Iterable

import numpy as np

from cosim.noise import NoiseModel


# =============================================================================
# LINK-BUDGET SNR — fast closed-form
# =============================================================================

def link_budget_snr_db(I_ph: np.ndarray, bandwidth_hz: float,
                       noise_model: NoiseModel,
                       t_total_s: float = 1.0) -> float:
    """Closed-form SNR including every modelled noise source (AWGN + non-AWGN).

    Sums variances analytically:
        σ²_total = σ²_AWGN_sum + σ²_mains_flicker + σ²_pink

    The two non-AWGN contributions are computed from
    `NoiseModel.mains_flicker_variance()` and `amp_flicker_variance(...)` —
    these have closed-form expressions (sum-of-cosines mean-square for
    flicker, 1/f integral for pink) so the link-budget SNR now reflects
    the *full* noise budget, not just the six AWGN sources.

    Returns SNR in dB. Capped at ±200 dB for numerical safety; NaN if the
    signal is silent (variance ≤ 0).
    """
    sig_var = float(np.var(I_ph)) if hasattr(I_ph, "__len__") else 0.0
    if sig_var <= 0:
        return float("nan")

    noise_var_awgn = float(noise_model.total_noise_std(I_ph, bandwidth_hz) ** 2)
    noise_var_flicker = noise_model.mains_flicker_variance()
    noise_var_pink = noise_model.amp_flicker_variance(t_total_s, bandwidth_hz)
    total_noise = noise_var_awgn + noise_var_flicker + noise_var_pink

    if total_noise <= 0:
        return 200.0
    return float(10.0 * np.log10(sig_var / total_noise))


# =============================================================================
# MEASURED SNR — two-run technique
# =============================================================================

def measure_snr_db(
    cfg, *,
    nodes: Iterable[str] = ("rx.I_ph_noisy", "rx.V_rx"),
    bits_override=None,
) -> Dict[str, float]:
    """
    Measured SNR at one or more pipeline nodes.

    Runs `run_python_simulation` twice with the SAME seed:
      A. reference — every noise source disabled
      B. full     — the user's config unchanged

    Then for each requested node:
        noise   = full[node] − reference[node]
        SNR_dB  = 10·log10( var(reference[node] − mean) / var(noise − mean) )

    Notes
    -----
    * Both runs must share `cfg.random_seed` or the deterministic mains
      flicker phases and the gaussian draws won't align, making the
      subtraction meaningless. We force a seed if the user didn't set one.
    * Nodes are referenced by probe ID (`rx.I_ph_noisy`, `rx.V_rx`, ...).
      For `rx.V_rx` we use whatever node the engine exposes via the
      `V_rx` key in the result dict — that is, the demodulator input,
      which is the SNR that *physically* drives BER.
    * If a node is missing from one of the runs, its entry is NaN.
    """
    # Local import keeps the SNR module free of pipeline coupling at import time.
    from cosim.probes import ProbeCapture
    from cosim.python_engine import run_python_simulation

    # Force a deterministic seed so the two runs are aligned sample-for-sample.
    if cfg.random_seed is None:
        cfg.random_seed = 0xC0FFEE

    # --- Reference run: all noise sources off ---
    cfg_ref = deepcopy(cfg)
    cfg_ref.noise_enable = False
    cfg_ref.enable_mains_flicker = False
    cfg_ref.enable_amp_flicker = False
    cfg_ref.measure_snr_enable = False   # guard against re-entry

    # --- Full run: explicit copy with the SNR measurement disabled inside
    # so the recursive call from run_python_simulation doesn't fire again.
    cfg_full = deepcopy(cfg)
    cfg_full.measure_snr_enable = False

    cap_ref = ProbeCapture()
    res_ref = run_python_simulation(cfg_ref, bits_override=bits_override, probes=cap_ref)

    cap_full = ProbeCapture()
    res_full = run_python_simulation(cfg_full, bits_override=bits_override, probes=cap_full)

    out: Dict[str, float] = {}
    for node in nodes:
        ref = cap_ref.get(node)
        full = cap_full.get(node)

        # Fall back to the result-dict V_rx when the user asked for it
        # — it's always present even outside the topology branches that
        # capture a V_node probe.
        if node == "rx.V_rx":
            ref = res_ref.get("V_rx") if ref is None else ref
            full = res_full.get("V_rx") if full is None else full

        if ref is None or full is None:
            out[node] = float("nan")
            continue

        ref_arr = np.asarray(ref, dtype=np.float64)
        full_arr = np.asarray(full, dtype=np.float64)
        if ref_arr.shape != full_arr.shape:
            out[node] = float("nan")
            continue

        signal = ref_arr - float(np.mean(ref_arr))
        noise = (full_arr - ref_arr)
        noise = noise - float(np.mean(noise))

        sig_var = float(np.var(signal))
        noise_var = float(np.var(noise))

        if sig_var <= 0:
            out[node] = float("nan")
        elif noise_var <= 0:
            out[node] = 200.0
        else:
            out[node] = float(10.0 * np.log10(sig_var / noise_var))

    return out


# =============================================================================
# Eb/N0 — AWGN floor only
# =============================================================================

def eb_n0_db_awgn(
    I_ph: np.ndarray, bandwidth_hz: float,
    bit_period_s: float, noise_model: NoiseModel,
) -> float:
    """
    Eb/N0 against the AWGN-only noise floor.

    Eb = P_signal × T_bit = var(I_ph) × T_bit               (Joules, per Ω)
    N0 = σ_awgn² / B_n                                       (A²/Hz)
    Eb/N0 = (var(I_ph) × T_bit × B_n) / σ_awgn²              (dimensionless)

    Only the six AWGN sources contribute to σ_awgn²; mains flicker and
    1/f pink are excluded because N₀ isn't flat for them — folding them
    in would silently violate the definition.

    Returns Eb/N0 in dB, or NaN if either quantity is zero.
    """
    sig_var = float(np.var(I_ph)) if hasattr(I_ph, "__len__") else 0.0
    if sig_var <= 0 or bandwidth_hz <= 0 or bit_period_s <= 0:
        return float("nan")

    # Reproduce only the AWGN-domain noise variance by computing the
    # NoiseBreakdown directly — it already excludes mains/pink which are
    # separate methods on the NoiseModel.
    I_avg = float(np.mean(np.abs(I_ph))) if hasattr(I_ph, "__len__") else float(I_ph)
    breakdown = noise_model.compute_noise(I_avg, bandwidth_hz)
    sigma_sq_awgn = float(breakdown.total)
    if sigma_sq_awgn <= 0:
        return 200.0

    eb_n0 = (sig_var * bit_period_s * bandwidth_hz) / sigma_sq_awgn
    return float(10.0 * np.log10(eb_n0))
