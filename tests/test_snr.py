"""Tests for cosim/snr.py — the three scientific SNR definitions."""

from __future__ import annotations

import math

import numpy as np
import pytest

from cosim.noise import NoiseModel
from cosim.python_engine import run_python_simulation
from cosim.snr import eb_n0_db_awgn, link_budget_snr_db, measure_snr_db
from cosim.system_config import SystemConfig


# ---------------------------------------------------------------------------
# Closed-form helpers
# ---------------------------------------------------------------------------

def test_link_budget_snr_helper_matches_engine_when_given_same_T_total():
    """link_budget_snr_db should reproduce the engine's snr_link_budget_dB
    exactly when called with the same T_total (otherwise the pink-noise
    integration bounds drift and the numbers diverge by a few dB)."""
    cfg = SystemConfig.from_preset("kadirvelu2021")
    cfg.simulation_engine = "python"
    cfg.n_bits = 200
    cfg.random_seed = 1
    cfg.noise_enable = True

    res = run_python_simulation(cfg)
    nm = NoiseModel.from_config(cfg)
    bandwidth = cfg.data_rate_bps / 2
    T_total = len(res["time"]) * (res["time"][1] - res["time"][0])

    direct = link_budget_snr_db(res["I_ph"], bandwidth, nm, t_total_s=T_total)
    assert direct == pytest.approx(res["snr_link_budget_dB"], rel=0.01)


def test_eb_n0_only_uses_awgn_variance():
    """Eb/N0 must ignore mains flicker and 1/f — those break N0 being flat."""
    cfg = SystemConfig.from_preset("kadirvelu2021")
    cfg.simulation_engine = "python"
    cfg.n_bits = 100
    cfg.random_seed = 2
    cfg.noise_enable = True

    res = run_python_simulation(cfg)
    nm = NoiseModel.from_config(cfg)
    bw = cfg.data_rate_bps / 2

    # Reference value with the existing model
    ref = eb_n0_db_awgn(res["I_ph"], bw, 1.0 / cfg.data_rate_bps, nm)

    # Now turn flicker / pink on — Eb/N0 must NOT change (the helper only
    # reads compute_noise() which is the AWGN aggregate).
    cfg.enable_mains_flicker = True
    cfg.enable_amp_flicker = True
    nm2 = NoiseModel.from_config(cfg)
    after = eb_n0_db_awgn(res["I_ph"], bw, 1.0 / cfg.data_rate_bps, nm2)

    assert after == pytest.approx(ref, rel=1e-9)


# ---------------------------------------------------------------------------
# Two-run measured SNR
# ---------------------------------------------------------------------------

def test_notch_filter_makes_measured_snr_at_Vrx_better_than_at_Iph():
    """The chain's notch filter attenuates the mains-flicker fundamental, so
    measured SNR at V_rx (post-notch) must be materially better than
    measured SNR at I_ph_noisy (pre-chain). This is the diagnostic gap that
    the two-run technique exposes."""
    cfg = SystemConfig.from_preset("lifi_poc_breadboard")
    cfg.simulation_engine = "python"
    cfg.n_bits = 500
    cfg.random_seed = 11
    cfg.measure_snr_enable = True
    cfg.enable_mains_flicker = True
    cfg.mains_flicker_depth = 0.05
    cfg.ambient_illuminance_lux = 200.0
    cfg.notch_freq_hz = 100.0   # active notch rejection
    cfg.notch_Q = 30.0

    snrs = measure_snr_db(cfg, nodes=("rx.I_ph_noisy", "rx.V_rx"))
    iph_snr = snrs["rx.I_ph_noisy"]
    vrx_snr = snrs["rx.V_rx"]

    assert not math.isnan(iph_snr) and not math.isnan(vrx_snr)
    # The Q=30 notch should buy at least 5 dB of rejection at the fundamental;
    # in practice we see ~10 dB across this preset.
    assert vrx_snr > iph_snr + 5.0, (
        f"notch should improve SNR; got iph={iph_snr:.1f} dB, vrx={vrx_snr:.1f} dB"
    )


def test_measured_snr_matches_link_budget_when_only_awgn_is_on():
    """With only the AWGN sources enabled (no flicker, no pink, no chain
    filtering between I_ph and the noise injection), measured SNR at the
    I_ph_noisy node should agree with the link-budget number within a few
    dB — they're measuring the same thing."""
    cfg = SystemConfig.from_preset("kadirvelu2021")
    cfg.simulation_engine = "python"
    cfg.n_bits = 500
    cfg.random_seed = 13
    cfg.noise_enable = True
    cfg.enable_mains_flicker = False
    cfg.enable_amp_flicker = False
    cfg.measure_snr_enable = True

    snrs = measure_snr_db(cfg, nodes=("rx.I_ph_noisy",))
    res = run_python_simulation(cfg)

    delta = abs(snrs["rx.I_ph_noisy"] - res["snr_link_budget_dB"])
    # 4 dB is a generous tolerance — sample noise on a 500-bit run easily
    # accounts for 1-2 dB on its own.
    assert delta < 4.0, (
        f"measured={snrs['rx.I_ph_noisy']:.2f} dB vs "
        f"link-budget={res['snr_link_budget_dB']:.2f} dB"
    )


def test_measure_snr_does_not_recurse_when_called_via_pipeline():
    """When the result dict's measure-snr step calls measure_snr_db, the
    inner runs must NOT trigger another measurement (would be infinite)."""
    cfg = SystemConfig.from_preset("lifi_poc_breadboard")
    cfg.simulation_engine = "python"
    cfg.n_bits = 64
    cfg.random_seed = 17
    cfg.measure_snr_enable = True

    # This must complete in finite time and populate snr_measured_dB.
    res = run_python_simulation(cfg)
    assert "snr_measured_dB" in res
    assert not math.isnan(res["snr_measured_dB"])


def test_link_budget_matches_measured_at_iph_within_couple_db():
    """The link-budget SNR (now including analytical flicker + pink variances)
    must agree with the measured SNR at I_ph_noisy to within ~2 dB across an
    ambient-light sweep. This is the consistency check that proves the
    closed-form formula captures the dominant noise term."""
    deltas = []
    for lux in [10.0, 100.0, 500.0]:
        cfg = SystemConfig.from_preset("lifi_poc_breadboard")
        cfg.simulation_engine = "python"
        cfg.measure_snr_enable = True
        cfg.ambient_illuminance_lux = lux
        cfg.n_bits = 500
        cfg.random_seed = 21
        # Ensure flicker is in both the link-budget AND the measurement so we
        # can compare apples to apples; pink is small at these values either way.
        cfg.enable_mains_flicker = True
        cfg.mains_flicker_depth = 0.05

        res = run_python_simulation(cfg)
        delta = abs(res["snr_link_budget_dB"] - res["snr_measured_at_Iph_dB"])
        deltas.append((lux, delta))

    for lux, d in deltas:
        assert d < 2.5, (
            f"link-budget vs measured @ I_ph disagree by {d:.2f} dB at {lux} lux "
            f"— flicker variance must be wrong or not summed"
        )


def test_link_budget_drops_monotonically_with_ambient_lux():
    """Increasing ambient lux must lower the link-budget SNR — that's the
    physical observation that motivated this fix."""
    snrs = []
    for lux in [10.0, 100.0, 1000.0]:
        cfg = SystemConfig.from_preset("lifi_poc_breadboard")
        cfg.simulation_engine = "python"
        cfg.ambient_illuminance_lux = lux
        cfg.n_bits = 200
        cfg.random_seed = 31
        cfg.enable_mains_flicker = True
        cfg.mains_flicker_depth = 0.05
        res = run_python_simulation(cfg)
        snrs.append(res["snr_link_budget_dB"])

    # 10 → 100 → 1000 lux: each 10× lux increase should cost ~20 dB
    # (flicker variance ∝ I_amb² → SNR drops 20 dB per decade of lux).
    assert snrs[0] > snrs[1] > snrs[2], f"SNR should drop with lux, got {snrs}"
    assert snrs[0] - snrs[2] > 30.0, (
        f"expected >30 dB drop from 10 to 1000 lux, got {snrs[0]-snrs[2]:.1f} dB"
    )


def test_measure_snr_is_skipped_when_noise_disabled():
    """If noise is off there is nothing to measure — the field should stay
    NaN rather than spawn a costly reference run that compares zero to zero."""
    cfg = SystemConfig.from_preset("kadirvelu2021")
    cfg.simulation_engine = "python"
    cfg.n_bits = 32
    cfg.random_seed = 19
    cfg.noise_enable = False
    cfg.measure_snr_enable = True

    res = run_python_simulation(cfg)
    assert math.isnan(res["snr_measured_dB"])
