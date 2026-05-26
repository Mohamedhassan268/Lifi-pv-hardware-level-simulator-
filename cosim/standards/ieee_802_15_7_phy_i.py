# cosim/standards/ieee_802_15_7_phy_i.py
"""
IEEE 802.15.7-2018 PHY-I — OOK with Manchester RLL.

Reference data rates (Table 73 of IEEE 802.15.7-2018):
    OOK / Manchester:  11.67, 24.44, 48.89, 73.3, 100.0,
                       200.0, 266.6 kbps

Compliance criteria (paraphrased — verify against the spec before
shipping a paper claim that uses these numbers):

    PASS condition for the receiver:
      - BER <= 1e-6 at SNR_est_dB >= 11 dB
      - Estimated SNR strictly positive
      - Detected modulation matches the OOK family
"""

from __future__ import annotations

from typing import Any

from cosim.standards.types import Criterion, CriterionResult, PhyProfile


def _ber_under(threshold: float):
    def _check(metrics: dict[str, Any]) -> CriterionResult:
        ber = metrics.get("ber")
        ok = isinstance(ber, (int, float)) and ber <= threshold
        return CriterionResult(
            label=f"BER ≤ {threshold:.0e}",
            ok=bool(ok),
            actual=float(ber) if isinstance(ber, (int, float)) else None,
            bound=threshold,
            detail="measured BER must be at or below the spec target",
        )
    return _check


def _snr_at_least(threshold_db: float):
    def _check(metrics: dict[str, Any]) -> CriterionResult:
        snr = metrics.get("snr_est_dB")
        ok = isinstance(snr, (int, float)) and snr >= threshold_db
        return CriterionResult(
            label=f"SNR_est ≥ {threshold_db:.0f} dB",
            ok=bool(ok),
            actual=float(snr) if isinstance(snr, (int, float)) else None,
            bound=threshold_db,
            detail="receiver must operate at or above the SNR floor",
        )
    return _check


def _modulation_in(allowed: set[str]):
    def _check(metrics: dict[str, Any]) -> CriterionResult:
        mod = metrics.get("modulation")
        ok = isinstance(mod, str) and mod in allowed
        return CriterionResult(
            label=f"Modulation ∈ {sorted(allowed)}",
            ok=bool(ok),
            actual=None,
            bound=None,
            detail=f"observed modulation: {mod!r}",
        )
    return _check


IEEE_802_15_7_PHY_I = PhyProfile(
    id="IEEE_802_15_7_PHY_I_OOK",
    label="IEEE 802.15.7 PHY-I (OOK / Manchester)",
    spec_section="IEEE 802.15.7-2018 §8 / Table 73",
    overrides={
        "modulation": "OOK_Manchester",
        "data_rate_bps": 100000.0,        # 100 kbps anchor data rate
        "noise_enable": True,
    },
    criteria=[
        Criterion(label="BER", check=_ber_under(1e-6)),
        Criterion(label="SNR", check=_snr_at_least(11.0)),
        Criterion(label="MOD", check=_modulation_in({"OOK", "OOK_Manchester"})),
    ],
    notes="Pinned data rate: 100 kbps. Other PHY-I rates "
          "(11.67 / 24.44 / 48.89 / 73.3 / 200 / 266.6 kbps) reuse the same chain.",
)
