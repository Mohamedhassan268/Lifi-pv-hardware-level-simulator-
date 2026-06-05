"""Compare a measured RX capture (from the real board) against the simulator.

Pure functions, no I/O: parse a CSV capture exported from the ESP serial
monitor, align it to a simulated RX waveform, and score the agreement. The
FastAPI router (backend/routers/measured.py) runs the Python sim for a preset
and feeds its V_rx trace into compare_waveforms() here.

Alignment is deliberately light (per the agreed scope):
  - amplitude: z-score both traces, so the ESP's ADC counts and the sim's volts
    compare regardless of absolute scale/offset;
  - time: resample both onto a common normalised grid, then a single coarse
    cross-correlation shift removes trigger skew.

Scores: NRMSE (RMS error / sim std) and Pearson correlation on the overlap.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

# Header keys (from `# key=value` comment lines) we recognise.
_RATE_KEYS = ("sample_rate_hz", "fs", "sample_rate")
_GRID = 2000  # common resample length for alignment


@dataclass
class MeasuredCapture:
    t: np.ndarray            # seconds, or a unitless index if no rate is known
    y: np.ndarray            # raw samples (ADC counts or volts)
    sample_rate_hz: float | None
    header: dict[str, str]   # parsed `# key=value` lines
    n: int


def parse_capture(csv_text: str, sample_rate_hz: float | None = None) -> MeasuredCapture:
    """Parse a CSV capture into a MeasuredCapture.

    Accepted shapes (comma / whitespace / semicolon separated):
      - two columns ``t_s, adc``  -> time taken from column 0;
      - one column ``adc``        -> time synthesised from the sample rate
        (``# sample_rate_hz=`` header, or the ``sample_rate_hz`` argument).
    ``# key=value`` comment lines are collected into ``header``; a leading
    non-numeric column-name row is skipped.
    """
    header: dict[str, str] = {}
    rows: list[list[float]] = []
    for raw in csv_text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            body = line[1:].strip()
            if "=" in body:
                k, v = body.split("=", 1)
                header[k.strip().lower()] = v.strip()
            continue
        parts = [p for p in re.split(r"[,;\t ]+", line) if p]
        try:
            rows.append([float(p) for p in parts])
        except ValueError:
            continue  # column-name row or stray text
    if not rows:
        raise ValueError("No numeric data rows found in the capture.")

    width = max(len(r) for r in rows)
    arr = np.array([r for r in rows if len(r) == width], dtype=float)
    if arr.ndim != 2 or arr.shape[0] < 4:
        raise ValueError("Need at least 4 samples with a consistent column count.")

    # Sample rate: header wins, then the argument.
    rate = sample_rate_hz
    for key in _RATE_KEYS:
        if key in header:
            try:
                rate = float(header[key])
            except ValueError:
                pass
            break

    if width >= 2:
        t, y = arr[:, 0], arr[:, 1]
        if rate is None and t[-1] > t[0]:
            rate = (len(t) - 1) / (t[-1] - t[0])
    else:
        y = arr[:, 0]
        if rate and rate > 0:
            t = np.arange(len(y), dtype=float) / rate
        else:
            t = np.arange(len(y), dtype=float)  # unitless index; alignment still works
    return MeasuredCapture(t=t, y=y, sample_rate_hz=rate, header=header, n=len(y))


def _zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    sd = x.std()
    return (x - x.mean()) / (sd if sd > 1e-12 else 1.0)


def _norm01(t: np.ndarray) -> np.ndarray:
    t = np.asarray(t, dtype=float)
    span = t[-1] - t[0]
    return (t - t[0]) / (span if span > 1e-12 else 1.0)


def _best_lag(a: np.ndarray, b: np.ndarray, max_lag: int) -> int:
    """Integer shift of ``a`` (vs ``b``) maximising cross-correlation, |lag|<=max_lag."""
    full = np.correlate(a, b, mode="full")
    n = len(a)
    lags = np.arange(-(n - 1), n)
    keep = np.abs(lags) <= max_lag
    full = np.where(keep, full, -np.inf)
    return int(lags[int(np.argmax(full))])


def compare_waveforms(meas: MeasuredCapture, sim_t, sim_y,
                      n_plot: int = 400) -> dict:
    """Align the measured capture to the simulated RX trace and score agreement.

    Returns ``{"series": {t, measured, simulated}, "metrics": {...}}`` with the
    plotted series normalised + aligned and downsampled to ``n_plot`` points.
    """
    m = _zscore(meas.y)
    s = _zscore(np.asarray(sim_y, dtype=float))

    grid = np.linspace(0.0, 1.0, _GRID)
    mi = np.interp(grid, _norm01(meas.t), m)
    si = np.interp(grid, _norm01(np.asarray(sim_t, dtype=float)), s)

    lag = _best_lag(mi, si, max_lag=_GRID // 4)
    mi = np.roll(mi, -lag)
    edge = abs(lag)
    sl = slice(edge, _GRID - edge) if edge else slice(None)
    a, b = mi[sl], si[sl]

    nrmse = float(np.sqrt(np.mean((a - b) ** 2)) / (b.std() + 1e-12))
    corr = float(np.corrcoef(a, b)[0, 1]) if a.size > 1 else 0.0
    if not np.isfinite(corr):
        corr = 0.0
    # Headline agreement = shape correlation (intuitive); NRMSE is the rigorous
    # secondary error (RMS error / sim std, on z-scored traces).
    agreement = max(0.0, corr) * 100.0

    idx = np.linspace(0, a.size - 1, min(n_plot, a.size)).astype(int)
    return {
        "series": {
            "t": grid[sl][idx].round(5).tolist(),
            "measured": a[idx].round(4).tolist(),
            "simulated": b[idx].round(4).tolist(),
        },
        "metrics": {
            "nrmse": round(nrmse, 4),
            "correlation": round(corr, 4),
            "agreement_pct": round(agreement, 1),
            "lag_frac": round(lag / _GRID, 4),
        },
    }
