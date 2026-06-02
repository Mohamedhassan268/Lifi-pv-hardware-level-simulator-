"""Feature extraction (cosim/features.py) — record completeness + correctness."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from cosim.features import FEATURES, TAG_KEYS, extract_features  # noqa: E402
from cosim.python_engine import run_python_simulation  # noqa: E402
from cosim.system_config import SystemConfig  # noqa: E402


def _run(preset="kadirvelu2021", **over):
    cfg = SystemConfig.from_preset(preset)
    cfg.simulation_engine = "python"
    cfg.n_bits = 2000
    cfg.random_seed = 0
    for k, v in over.items():
        setattr(cfg, k, v)
    return run_python_simulation(cfg), cfg


def test_record_has_every_registered_feature():
    result, cfg = _run()
    rec = extract_features(result, cfg)
    for key in FEATURES:
        assert key in rec, f"feature {key!r} missing from record"
    for tag in TAG_KEYS:
        assert tag in rec


def test_core_features_are_sane():
    result, cfg = _run()
    rec = extract_features(result, cfg)
    assert 0.0 <= rec["ber"] <= 1.0
    assert rec["data_rate_bps"] == cfg.data_rate_bps
    # Goodput cannot exceed payload rate, which cannot exceed gross rate.
    assert rec["goodput_bps"] <= rec["payload_rate_bps"] + 1e-6
    assert rec["payload_rate_bps"] <= rec["data_rate_bps"] + 1e-6
    # Manchester is the kadirvelu scheme: 0.5 bps/Hz spectral efficiency.
    assert abs(rec["spectral_efficiency"] - 0.5) < 1e-6
    # Propagation delay = distance / c.
    assert math.isclose(rec["latency_propagation_s"], cfg.distance_m / 299_792_458.0, rel_tol=1e-9)
    # Frame latency must dominate the propagation + DSP components.
    assert rec["latency_frame_s"] > rec["latency_propagation_s"]
    assert rec["p_rx_avg_W"] > 0
    assert rec["path_loss_dB"] > 0  # always lossy


def test_q_factor_and_ci_consistency():
    result, cfg = _run()
    rec = extract_features(result, cfg)
    # CI brackets the point estimate.
    assert rec["ber_ci_low"] <= rec["ber"] <= rec["ber_ci_high"] + 1e-12
    # Clean link → high (or infinite) Q.
    if rec["ber"] == 0.0:
        assert math.isinf(rec["q_factor"]) or rec["q_factor"] > 5


def test_evm_only_for_ofdm():
    line, cfg_line = _run()
    assert math.isnan(extract_features(line, cfg_line)["evm_percent"])
    ofdm, cfg_ofdm = _run("sarwar2017")
    rec = extract_features(ofdm, cfg_ofdm)
    assert not math.isnan(rec["evm_percent"]) and rec["evm_percent"] >= 0


def test_fec_lowers_payload_rate():
    # code_rate < 1 when FEC on → payload rate below gross rate.
    rec = extract_features(*_run(fec_enable=True, fec_rate_num=1, fec_rate_den=2))
    assert abs(rec["code_rate"] - 0.5) < 1e-9
    assert rec["payload_rate_bps"] < rec["data_rate_bps"]


def test_never_raises_on_empty_result():
    cfg = SystemConfig.from_preset("kadirvelu2021")
    rec = extract_features({}, cfg)  # degenerate: no waveforms at all
    assert set(FEATURES).issubset(rec)
    assert math.isnan(rec["p_rx_avg_W"]) or rec["p_rx_avg_W"] == 0
