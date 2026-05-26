# cosim/ac_analysis.py
"""
Small-signal AC analysis of the RX chain — closed-form Python (no SPICE).

The RX chain modeled here mirrors the python_engine.py topology:

    PV (current source @ photocurrent) →
        sense resistor (transimpedance) →
        INA (fixed-gain block w/ GBW pole) →
        BPF stages (cascaded 1st-order high-pass + low-pass) →
        OUTPUT (pre-comparator)

We construct one composite transfer function H(jω) by multiplying the
individual stages, then compute magnitude (dB), phase (deg), and the
stability margins (GM, PM) from the unity-gain and 180°-phase crossings.

We deliberately avoid SPICE here because:
  - The packaged Tauri build doesn't ship a SPICE binary.
  - This computation runs in <50 ms across 1000 frequency points.
  - The .ac netlist round-trip would add significant complexity for the
    same numeric result; the component models below come from the same
    parameters the .tran path uses.

The transfer function is approximate (single-pole INA, ideal cap/resistor
networks for the BPF). Good enough for stability margin verification and
band-pass shape inspection — the canonical EDA use cases.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from cosim.system_config import SystemConfig


@dataclass
class AcResult:
    frequency_hz: np.ndarray
    magnitude_db: np.ndarray
    phase_deg: np.ndarray
    gain_margin_db: Optional[float]
    phase_margin_deg: Optional[float]
    unity_gain_freq_hz: Optional[float]
    crossover_180_freq_hz: Optional[float]
    f_low_3dB_hz: Optional[float]
    f_high_3dB_hz: Optional[float]
    midband_gain_db: float

    def as_dict(self) -> dict:
        return {
            "frequency_hz": [float(x) for x in self.frequency_hz],
            "magnitude_db": [float(x) for x in self.magnitude_db],
            "phase_deg": [float(x) for x in self.phase_deg],
            "gain_margin_db": self.gain_margin_db,
            "phase_margin_deg": self.phase_margin_deg,
            "unity_gain_freq_hz": self.unity_gain_freq_hz,
            "crossover_180_freq_hz": self.crossover_180_freq_hz,
            "f_low_3dB_hz": self.f_low_3dB_hz,
            "f_high_3dB_hz": self.f_high_3dB_hz,
            "midband_gain_db": self.midband_gain_db,
        }


# ---------------------------------------------------------------------------
# Per-stage transfer functions (complex H(jω))
# ---------------------------------------------------------------------------

def _ina_tf(omega: np.ndarray, gain_linear: float, gbw_hz: float) -> np.ndarray:
    """INA / opamp with finite GBW: single-pole low-pass at GBW / gain."""
    f_pole = max(gbw_hz / max(gain_linear, 1.0), 1.0)
    return gain_linear / (1.0 + 1j * omega / (2.0 * math.pi * f_pole))


def _hp_first_order(omega: np.ndarray, f_c_hz: float) -> np.ndarray:
    """First-order high-pass: H = jω / (jω + ω_c)."""
    omega_c = 2.0 * math.pi * max(f_c_hz, 1e-6)
    s = 1j * omega
    return s / (s + omega_c)


def _lp_first_order(omega: np.ndarray, f_c_hz: float) -> np.ndarray:
    """First-order low-pass: H = ω_c / (jω + ω_c)."""
    omega_c = 2.0 * math.pi * max(f_c_hz, 1e-6)
    s = 1j * omega
    return omega_c / (s + omega_c)


# ---------------------------------------------------------------------------
# Composite RX chain
# ---------------------------------------------------------------------------

def run_ac_analysis(
    cfg: SystemConfig,
    f_start_hz: float = 1.0,
    f_stop_hz: float = 1e7,
    n_points: int = 400,
) -> AcResult:
    """Compute the small-signal frequency response of the RX chain.

    Stages:
      1. Sense resistor — flat gain = r_sense_ohm (V/A)
      2. INA — gain = 10^(ina_gain_dB/20), GBW pole = ina_gbw_kHz·1e3
      3. BPF cascade — `bpf_stages` first-order high-pass + low-pass pairs
         at bpf_f_low_Hz and bpf_f_high_Hz respectively.

    The output is the transfer function from photocurrent (A) to the
    pre-comparator analog voltage (V), i.e. units of Ω.
    """
    f = np.logspace(math.log10(max(f_start_hz, 0.1)), math.log10(f_stop_hz), n_points)
    omega = 2.0 * math.pi * f

    ina_gain_linear = 10.0 ** (cfg.ina_gain_dB / 20.0)
    ina_gbw_hz = cfg.ina_gbw_kHz * 1e3

    # Stage 1: sense resistor (flat, real gain)
    H = np.full_like(omega, cfg.r_sense_ohm, dtype=complex)
    # Stage 2: INA
    H = H * _ina_tf(omega, ina_gain_linear, ina_gbw_hz)
    # Stage 3: BPF cascade
    n_stages = max(1, int(cfg.bpf_stages))
    for _ in range(n_stages):
        H = H * _hp_first_order(omega, cfg.bpf_f_low_Hz)
        H = H * _lp_first_order(omega, cfg.bpf_f_high_Hz)

    mag_db = 20.0 * np.log10(np.abs(H) + 1e-30)
    phase_deg = np.unwrap(np.angle(H)) * 180.0 / math.pi

    midband_db = float(np.max(mag_db))

    # 3-dB bandwidth around the midband peak.
    threshold = midband_db - 3.0
    peak_idx = int(np.argmax(mag_db))
    f_low_3dB = _interp_crossing(f[:peak_idx + 1], mag_db[:peak_idx + 1], threshold, ascending=True)
    f_high_3dB = _interp_crossing(f[peak_idx:], mag_db[peak_idx:], threshold, ascending=False)

    # Unity gain (0 dB) crossing on the high-frequency falling edge → PM.
    f_unity, pm = _phase_margin(f, mag_db, phase_deg)
    # 180° phase crossing → GM.
    f_180, gm = _gain_margin(f, mag_db, phase_deg)

    return AcResult(
        frequency_hz=f,
        magnitude_db=mag_db,
        phase_deg=phase_deg,
        gain_margin_db=gm,
        phase_margin_deg=pm,
        unity_gain_freq_hz=f_unity,
        crossover_180_freq_hz=f_180,
        f_low_3dB_hz=f_low_3dB,
        f_high_3dB_hz=f_high_3dB,
        midband_gain_db=midband_db,
    )


# ---------------------------------------------------------------------------
# Margin extraction helpers
# ---------------------------------------------------------------------------

def _phase_margin(
    f: np.ndarray, mag_db: np.ndarray, phase_deg: np.ndarray,
) -> tuple[Optional[float], Optional[float]]:
    """Find the FIRST 0 dB crossing on the falling edge (high-side)
    and return (f_unity, phase_margin_deg = 180 + phase)."""
    peak_idx = int(np.argmax(mag_db))
    f_unity = _interp_crossing(f[peak_idx:], mag_db[peak_idx:], 0.0, ascending=False)
    if f_unity is None:
        return (None, None)
    phase_at = float(np.interp(f_unity, f, phase_deg))
    return (f_unity, 180.0 + phase_at)


def _gain_margin(
    f: np.ndarray, mag_db: np.ndarray, phase_deg: np.ndarray,
) -> tuple[Optional[float], Optional[float]]:
    """Find where phase crosses -180° on the descending edge (band-pass
    falls past midband). Return (f_180, GM = -magnitude at that frequency)."""
    target = -180.0
    crossings = []
    for i in range(len(phase_deg) - 1):
        a = phase_deg[i] - target
        b = phase_deg[i + 1] - target
        if a == 0:
            crossings.append(f[i])
        elif a * b < 0:
            frac = a / (a - b)
            crossings.append(f[i] + frac * (f[i + 1] - f[i]))
    if not crossings:
        return (None, None)
    f_180 = crossings[0]
    mag_at = float(np.interp(f_180, f, mag_db))
    return (f_180, -mag_at)


def _interp_crossing(
    xs: np.ndarray, ys: np.ndarray, target: float, ascending: bool,
) -> Optional[float]:
    """Return the first x at which ys crosses `target`. If ascending=True,
    we look for ys going UP through target; otherwise going DOWN."""
    if len(xs) < 2:
        return None
    for i in range(len(ys) - 1):
        a = ys[i] - target
        b = ys[i + 1] - target
        if a == 0:
            return float(xs[i])
        crossed = a * b < 0
        if not crossed:
            continue
        going_up = b > a
        if ascending and not going_up:
            continue
        if not ascending and going_up:
            continue
        frac = a / (a - b)
        return float(xs[i] + frac * (xs[i + 1] - xs[i]))
    return None
