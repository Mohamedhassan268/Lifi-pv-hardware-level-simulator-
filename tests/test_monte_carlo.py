"""Tests for Monte Carlo determinism on the streaming (UI) path.

The batch path (cosim/monte_carlo.py) pins the inner simulator's noise/bit
realization so the only source of BER variation is the tolerance sampling.
The streaming path (backend/routers/sweep_ws.py) must do the same, otherwise
the live UI mixes tolerance spread with noise spread and never reproduces.

Most presets leave `random_seed` unset (None), so the streaming path must
force a fixed inner seed itself — it cannot rely on the preset carrying one.
"""

from __future__ import annotations

import threading

import cosim.monte_carlo as mc
import backend.routers.sweep_ws as sweep_ws


# gonzalez2024 has random_seed=None (n_bits=48 → fast). Using it proves the
# streaming path pins the seed itself rather than inheriting it from a preset.
_PRESET = "gonzalez2024"


def test_stream_pins_inner_seed(monkeypatch):
    """Every trial must run the inner sim with the same fixed seed, even for a
    preset whose own random_seed is None. Mirrors the batch path's
    `inner_seed = seed if seed is not None else 0`."""
    seen = []

    def fake_sim(cfg):
        seen.append(cfg.random_seed)
        return {"ber": 0.0, "snr_est_dB": 0.0, "P_rx_avg_uW": 0.0}

    # The sim call lives in the shared generator (cosim.monte_carlo), which the
    # streaming path delegates to — patch it there.
    monkeypatch.setattr(mc, "run_python_simulation", fake_sim)
    cancel = threading.Event()
    list(sweep_ws.stream_monte_carlo(_PRESET, {"sc_responsivity": 0.05}, 3, 42, cancel))

    assert seen == [42, 42, 42], f"inner seed not pinned: {seen}"


def test_stream_same_seed_reproduces():
    """Two real sweeps with the same seed must yield identical BER sequences."""
    spec = {"sc_responsivity": 0.05}

    def run():
        cancel = threading.Event()
        return [s["ber"] for s in sweep_ws.stream_monte_carlo(_PRESET, spec, 2, 7, cancel)]

    assert run() == run()
