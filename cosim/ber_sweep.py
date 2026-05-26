# cosim/ber_sweep.py
"""
BER-vs-SNR and BER-vs-distance sweep generators.

These complement the existing thermal_sweep and monte_carlo streamers:

  - stream_ber_vs_snr   — sweeps `ambient_illuminance_lux` over a log range
                          to vary SNR while keeping the signal path fixed.
                          (The simulator doesn't expose a direct SNR knob,
                          but ambient light is a clean way to load the noise
                          floor without changing the modulation chain.)
  - stream_ber_vs_distance — sweeps `distance_m` over a log range. Channel
                          gain falls as 1/r^(m+3); SNR drops monotonically.

Each point yields a dict with `(x_label, x_value, ber, n_errors, n_bits,
snr_est_dB, ci_low, ci_high)`. Confidence interval is **Wilson score 95%**
on the binomial proportion (errors / bits) — matches what the frontend
overlays as error bars.

Pattern mirrors stream_thermal() in backend/routers/sweep_ws.py.
"""

from __future__ import annotations

import math
import threading
from typing import Any, Iterator

import numpy as np

from cosim.python_engine import run_python_simulation
from cosim.system_config import SystemConfig


# ---------------------------------------------------------------------------
# Wilson score 95% confidence interval for a binomial proportion.
# Identical math to frontend wilsonCi() in lib/dsp.ts.
# ---------------------------------------------------------------------------

def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return (0.0, 1.0)
    p = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    centre = (p + z2 / (2 * n)) / denom
    radius = (z * math.sqrt(max(p * (1 - p) / n + z2 / (4 * n * n), 0.0))) / denom
    return (max(0.0, centre - radius), min(1.0, centre + radius))


# ---------------------------------------------------------------------------
# Sweep generators
# ---------------------------------------------------------------------------

def _logspace(lo: float, hi: float, n: int) -> list[float]:
    """log-spaced points; clamps lo to a positive floor."""
    lo = max(lo, 1e-12)
    return list(np.logspace(math.log10(lo), math.log10(max(hi, lo * 10)), n))


def stream_ber_vs_snr(
    preset: str,
    ambient_lux_lo: float,
    ambient_lux_hi: float,
    n_points: int,
    cancel: threading.Event,
) -> Iterator[dict[str, Any]]:
    """Sweep ambient illuminance (lux) to load the receiver noise floor.

    Yields one dict per point. Higher ambient → more shot noise → lower SNR.
    """
    xs = _logspace(ambient_lux_lo, ambient_lux_hi, n_points)
    for lux in xs:
        if cancel.is_set():
            return
        cfg = SystemConfig.from_preset(preset)
        cfg.noise_enable = True
        cfg.ambient_illuminance_lux = float(lux)

        try:
            res = run_python_simulation(cfg)
            ber = float(res.get("ber", 1.0))
            n_errors = int(res.get("n_errors", 0))
            n_bits = int(res.get("n_bits_tested", 0))
            snr_db = float(res.get("snr_est_dB", 0.0))
        except Exception:  # noqa: BLE001
            ber, n_errors, n_bits, snr_db = 1.0, 0, 0, 0.0

        ci_lo, ci_hi = wilson_ci(n_errors, n_bits)
        yield {
            "x_label": "ambient_lux",
            "x_value": float(lux),
            "snr_est_dB": snr_db,
            "ber": ber,
            "n_errors": n_errors,
            "n_bits": n_bits,
            "ci_low": ci_lo,
            "ci_high": ci_hi,
            "modulation": cfg.modulation,
        }


def stream_ber_vs_distance(
    preset: str,
    distance_lo_m: float,
    distance_hi_m: float,
    n_points: int,
    cancel: threading.Event,
) -> Iterator[dict[str, Any]]:
    """Sweep TX–RX distance (m). Channel gain falls roughly as 1/r^(m+3).
    """
    xs = _logspace(distance_lo_m, distance_hi_m, n_points)
    for d in xs:
        if cancel.is_set():
            return
        cfg = SystemConfig.from_preset(preset)
        cfg.distance_m = float(d)

        try:
            res = run_python_simulation(cfg)
            ber = float(res.get("ber", 1.0))
            n_errors = int(res.get("n_errors", 0))
            n_bits = int(res.get("n_bits_tested", 0))
            snr_db = float(res.get("snr_est_dB", 0.0))
        except Exception:  # noqa: BLE001
            ber, n_errors, n_bits, snr_db = 1.0, 0, 0, 0.0

        ci_lo, ci_hi = wilson_ci(n_errors, n_bits)
        yield {
            "x_label": "distance_m",
            "x_value": float(d),
            "snr_est_dB": snr_db,
            "ber": ber,
            "n_errors": n_errors,
            "n_bits": n_bits,
            "ci_low": ci_lo,
            "ci_high": ci_hi,
            "modulation": cfg.modulation,
        }
