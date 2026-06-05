"""POST /api/measured/compare — overlay a measured RX capture on the simulator.

Takes a CSV exported from the real board (ESP serial monitor / DMM), runs the
Python sim for the chosen preset, aligns the measured trace to the simulated
V_rx, and returns both series + agreement metrics. The alignment/scoring lives
in cosim.measured_compare; this router just wires the sim in and folds in a BER
comparison.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from cosim.measured_compare import compare_waveforms, parse_capture
from cosim.python_engine import run_python_simulation
from cosim.system_config import SystemConfig

router = APIRouter()


class CompareRequest(BaseModel):
    csv: str
    preset: str = "lifi_poc_breadboard"
    n_bits: int | None = None
    sample_rate_hz: float | None = None  # used only if the CSV has no time column
    measured_ber: float | None = None    # overrides a `# ber=` header line


class CompareResponse(BaseModel):
    ok: bool
    message: str
    series: dict[str, Any]
    metrics: dict[str, Any]
    meta: dict[str, Any]


@router.post("/compare", response_model=CompareResponse)
def compare(req: CompareRequest) -> CompareResponse:
    try:
        cap = parse_capture(req.csv, sample_rate_hz=req.sample_rate_hz)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e

    try:
        cfg = SystemConfig.from_preset(req.preset)
    except Exception as e:  # unknown preset / bad name
        raise HTTPException(status_code=422,
                            detail=f"Unknown preset '{req.preset}'.") from e
    if req.n_bits is not None:
        cfg.n_bits = int(min(max(req.n_bits, 8), 2000))
    cfg.random_seed = 1  # deterministic sim trace for a stable overlay

    sim = run_python_simulation(cfg)
    if "time" not in sim or "V_rx" not in sim:
        raise HTTPException(status_code=500,
                            detail="Sim produced no RX waveform to compare against.")

    out = compare_waveforms(cap, sim["time"], sim["V_rx"])

    # BER comparison: measured from the request or a `# ber=` header line.
    meas_ber = req.measured_ber
    if meas_ber is None and "ber" in cap.header:
        try:
            meas_ber = float(cap.header["ber"])
        except ValueError:
            meas_ber = None
    sim_ber = sim.get("ber")
    ber: dict[str, Any] = {"simulated": sim_ber, "measured": meas_ber}
    if sim_ber is not None and meas_ber is not None:
        ber["abs_error"] = round(abs(sim_ber - meas_ber), 5)
        ber["rel_error"] = (round(abs(sim_ber - meas_ber) / meas_ber, 4)
                            if meas_ber > 0 else None)
    out["metrics"]["ber"] = ber

    agree = out["metrics"]["agreement_pct"]
    return CompareResponse(
        ok=True,
        message=f"{agree:.0f}% waveform agreement over {cap.n} measured samples.",
        series=out["series"],
        metrics=out["metrics"],
        meta={
            "preset": req.preset,
            "n_measured": cap.n,
            "sample_rate_hz": cap.sample_rate_hz,
            "header": cap.header,
        },
    )
