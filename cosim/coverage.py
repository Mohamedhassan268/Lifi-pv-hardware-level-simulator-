"""cosim/coverage.py — fast analytical greenhouse coverage field.

Per floor position under one or more ceiling luminaires, computes the link
SNR/BER (closed-form, matching the link-budget router) and the steady-state PV
harvest (`PVCellModel.steady_state_voltage`, one root-solve). No transient ODE,
so a full grid is instant — this is what the interactive canvas calls. The
transient `scripts/coverage_*.py` remain the high-fidelity reference.

Model: the node takes data from the strongest luminaire; all luminaires' light
sums into harvest (and the non-data part adds shot noise) — same first-cut as the
transient multi-luminaire script.
"""

from __future__ import annotations

import math

import numpy as np

from cosim.channel import OpticalChannel
from cosim.pv_model import PVCellModel

_Q_E = 1.602e-19
_KT = 1.38e-23 * 300.0


def _ber_ook(snr_lin: float) -> float:
    """Analytical OOK BER from electrical SNR via the Q-factor: 0.5·erfc(Q/√2)."""
    q = math.sqrt(max(snr_lin, 0.0))
    return 0.5 * math.erfc(q / math.sqrt(2.0))


def coverage_field(cfg, luminaires: list[tuple[float, float]],
                   total_power_W: float, height_m: float, half_extent_m: float,
                   n: int, half_angle_deg: float, node_budget_uW: float,
                   dcdc_eff: float, ber_thresh: float) -> dict:
    """Return per-position grids (n×n) + summary fractions. JSON-serializable.

    `luminaires` are ceiling (x,y) positions; `total_power_W` is split across them.
    """
    xs = np.linspace(-half_extent_m, half_extent_m, n)
    area = cfg.sc_area_cm2
    resp = cfg.sc_responsivity
    rsense = cfg.r_sense_ohm
    fov = getattr(cfg, "fov_half_angle_deg", 90.0)
    bw = cfg.data_rate_bps / 2.0
    md = cfg.modulation_depth
    pv = PVCellModel.from_config(cfg)
    p_each = total_power_W / max(len(luminaires), 1)

    BER = np.empty((n, n)); HAR = np.empty((n, n)); SNR = np.empty((n, n))
    for iy, y in enumerate(xs):
        for ix, x in enumerate(xs):
            prx = []
            for (lx, ly) in luminaires:
                r = math.hypot(x - lx, y - ly)
                d = math.hypot(r, height_m)
                ang = math.degrees(math.atan2(r, height_m))
                g = OpticalChannel(distance_m=d, led_half_angle_deg=half_angle_deg,
                                   rx_area_cm2=area, tx_angle_deg=ang, rx_tilt_deg=ang,
                                   fov_half_angle_deg=fov).channel_gain()
                prx.append(p_each * g)
            prx = np.array(prx)
            p_data = float(prx.max())
            p_total = float(prx.sum())

            i_ph = resp * p_data
            i_sig = i_ph * md
            i_amb = resp * (p_total - p_data)        # other luminaires -> shot noise
            n_shot = math.sqrt(2 * _Q_E * max(i_ph + i_amb, 0.0) * bw)
            n_th = math.sqrt(4 * _KT * bw / max(rsense, 1e-6))
            n_tot = math.hypot(n_shot, n_th)
            snr = (i_sig / n_tot) ** 2 if (n_tot > 0 and i_sig > 0) else 0.0
            SNR[iy, ix] = 10.0 * math.log10(snr) if snr > 0 else -100.0
            BER[iy, ix] = _ber_ook(snr)

            v = pv.steady_state_voltage(p_total, R_load=rsense)   # total-light op point
            i_cell = v / (rsense + pv.R_s)
            HAR[iy, ix] = v * i_cell * 1e6

    closes = BER <= ber_thresh
    deployable = closes & ((HAR * dcdc_eff) >= node_budget_uW)
    return {
        "xs": xs.tolist(),
        "BER": BER.tolist(),
        "harvest_uW": HAR.tolist(),
        "snr_dB": SNR.tolist(),
        "deployable": deployable.tolist(),
        "data_cov_pct": float(100.0 * closes.mean()),
        "deployable_pct": float(100.0 * deployable.mean()),
        "harvest_min_uW": float(HAR.min()),
        "harvest_max_uW": float(HAR.max()),
    }
