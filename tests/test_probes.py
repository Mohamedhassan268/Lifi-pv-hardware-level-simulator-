"""Tests for cosim/probes.py: registry, capture, hashing, persistence."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from cosim.probes import (
    PROBE_REGISTRY,
    ProbeCapture,
    all_probe_ids,
    config_hash,
    probes_for_edge,
    probes_for_stage,
    registry_export,
)
from cosim.python_engine import run_python_simulation
from cosim.system_config import SystemConfig


# ---------------------------------------------------------------------------
# Registry static properties
# ---------------------------------------------------------------------------

def test_registry_ids_are_unique_and_namespaced():
    ids = all_probe_ids()
    assert len(ids) == len(set(ids))
    for pid in ids:
        assert "." in pid, f"probe id {pid!r} must be <stage>.<name>"


def test_registry_export_is_json_serializable():
    payload = registry_export()
    json.dumps(payload)  # must not raise
    assert {p["id"] for p in payload} == set(PROBE_REGISTRY)


def test_every_edge_probe_has_a_valid_edge_id():
    # SchematicCanvas defines edges "tx-channel" and "channel-rx".
    valid_edges = {"tx-channel", "channel-rx"}
    edge_probes = [s for s in PROBE_REGISTRY.values() if s.edge_id is not None]
    for spec in edge_probes:
        assert spec.edge_id in valid_edges


def test_probes_for_stage_partitions_registry():
    tx = set(probes_for_stage("TX"))
    ch = set(probes_for_stage("Channel"))
    rx = set(probes_for_stage("RX"))
    assert tx.isdisjoint(ch) and tx.isdisjoint(rx) and ch.isdisjoint(rx)
    assert tx | ch | rx == set(PROBE_REGISTRY)


def test_probes_for_edge_returns_channel_edges():
    assert "tx.P_tx" in probes_for_edge("tx-channel")
    assert "channel.P_rx" in probes_for_edge("channel-rx")


# ---------------------------------------------------------------------------
# Capture mechanics
# ---------------------------------------------------------------------------

def test_capture_stores_array_and_scalar_with_correct_dtype():
    cap = ProbeCapture(config_hash="h")
    cap.capture("tx.bits", [0, 1, 1, 0])
    cap.capture("channel.G_los", 0.0123)

    bits = cap.get("tx.bits")
    assert isinstance(bits, np.ndarray)
    assert bits.dtype == np.uint8
    assert list(bits) == [0, 1, 1, 0]

    assert cap.get("channel.G_los") == pytest.approx(0.0123)
    assert cap.meta("channel.G_los")["units"] == "1"


def test_unknown_probe_id_is_silently_ignored():
    cap = ProbeCapture(config_hash="h")
    cap.capture("nonexistent.probe", np.zeros(4))
    assert cap.available_ids() == []


def test_enabled_ids_filter_is_respected():
    cap = ProbeCapture(config_hash="h", enabled_ids=["tx.P_tx"])
    cap.capture("tx.P_tx", np.zeros(8))
    cap.capture("rx.V_ina", np.ones(8))
    assert cap.has("tx.P_tx")
    assert not cap.has("rx.V_ina")


# ---------------------------------------------------------------------------
# config_hash
# ---------------------------------------------------------------------------

def test_config_hash_is_stable_across_equal_configs():
    a = SystemConfig.from_preset("kadirvelu2021")
    b = SystemConfig.from_preset("kadirvelu2021")
    assert config_hash(a) == config_hash(b)


def test_config_hash_changes_with_edits():
    a = SystemConfig.from_preset("kadirvelu2021")
    b = SystemConfig.from_preset("kadirvelu2021")
    b.distance_m = a.distance_m * 2
    assert config_hash(a) != config_hash(b)


# ---------------------------------------------------------------------------
# End-to-end pipeline capture
# ---------------------------------------------------------------------------

def test_python_engine_captures_all_expected_probes():
    cfg = SystemConfig.from_preset("kadirvelu2021")
    cfg.simulation_engine = "python"
    cfg.noise_enable = True
    cfg.n_bits = 128

    cap = ProbeCapture(config_hash=config_hash(cfg))
    hooks_fired = []

    run_python_simulation(cfg, probes=cap, stage_hook=hooks_fired.append)

    assert hooks_fired == ["TX", "Channel", "RX"]

    expected = {
        "tx.bits", "tx.P_tx",
        "channel.G_los", "channel.G_diffuse", "channel.beer_lambert",
        "channel.G_total", "channel.P_rx",
        "rx.I_ph", "rx.I_ph_noisy",
        "rx.noise.shot", "rx.noise.thermal", "rx.noise.ambient",
        "rx.noise.amplifier", "rx.noise.adc", "rx.noise.processing",
        "rx.V_sense", "rx.V_ina", "rx.V_bpf", "rx.V_comp",
        "rx.bits_rx", "rx.snr_db",
    }
    assert expected.issubset(set(cap.available_ids()))


def test_summed_per_source_noise_matches_clean_vs_noisy_diff():
    cfg = SystemConfig.from_preset("kadirvelu2021")
    cfg.simulation_engine = "python"
    cfg.noise_enable = True
    cfg.n_bits = 64
    cfg.random_seed = 42
    # Disable the non-AWGN sources so this test isolates the 6-source AWGN
    # sum — the I_ph_noisy difference equals the per-source AWGN sum only
    # when no extra terms are added on top.
    cfg.enable_mains_flicker = False
    cfg.enable_amp_flicker = False

    cap = ProbeCapture(config_hash=config_hash(cfg))
    run_python_simulation(cfg, probes=cap)

    summed = (
        cap.get("rx.noise.shot")
        + cap.get("rx.noise.thermal")
        + cap.get("rx.noise.ambient")
        + cap.get("rx.noise.amplifier")
        + cap.get("rx.noise.adc")
        + cap.get("rx.noise.processing")
    )
    diff = cap.get("rx.I_ph_noisy") - cap.get("rx.I_ph")
    # Floating-point equality is acceptable: both arrays are sums of the same
    # six independent gaussian draws produced once in generate_per_source_time_domain.
    assert np.allclose(summed, diff, atol=1e-18, rtol=0)


def test_probe_save_writes_npz_and_manifest(tmp_path: Path):
    cfg = SystemConfig.from_preset("kadirvelu2021")
    cfg.simulation_engine = "python"
    cfg.noise_enable = True
    cfg.n_bits = 64

    cap = ProbeCapture(session_dir=tmp_path, config_hash=config_hash(cfg))
    run_python_simulation(cfg, probes=cap)
    out_dir = cap.save()

    assert out_dir is not None
    assert (out_dir / "arrays.npz").exists()
    assert (out_dir / "manifest.json").exists()

    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert manifest["config_hash"] == config_hash(cfg)
    assert "channel.G_los" in manifest["scalars"]
