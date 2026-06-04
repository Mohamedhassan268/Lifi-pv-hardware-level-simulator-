"""POST /api/coverage — fast analytical greenhouse coverage field for the canvas.

Reuses the node hardware from a preset (default slipt_node) and computes, over an
n×n floor grid under one or more ceiling luminaires, the link BER/SNR + steady-
state PV harvest + deployable region. Analytical (no transient), so it is fast
enough to drive an interactive canvas.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from cosim.coverage import coverage_field
from cosim.system_config import SystemConfig

router = APIRouter()


class Luminaire(BaseModel):
    x: float
    y: float


class CoverageRequest(BaseModel):
    preset: str = "slipt_node"
    luminaires: list[Luminaire] = Field(default_factory=lambda: [Luminaire(x=0.0, y=0.0)])
    total_power_W: float = 8.0
    height_m: float = 2.5
    half_extent_m: float = 2.0
    grid: int = Field(default=15, ge=3, le=41)
    half_angle_deg: float = 40.0
    r_sense_ohm: float = 500.0
    node_budget_uW: float = 10.0
    dcdc_eff: float = 0.75
    ber_thresh: float = 1e-2


class CoverageResponse(BaseModel):
    xs: list[float]
    BER: list[list[float]]
    harvest_uW: list[list[float]]
    snr_dB: list[list[float]]
    deployable: list[list[bool]]
    data_cov_pct: float
    deployable_pct: float
    harvest_min_uW: float
    harvest_max_uW: float


@router.post("", response_model=CoverageResponse)
def compute(req: CoverageRequest) -> CoverageResponse:
    try:
        cfg = SystemConfig.from_preset(req.preset)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    # Apply the deployment operating point (data-optimized PV load by default).
    d = cfg.to_dict()
    d["r_sense_ohm"] = req.r_sense_ohm
    cfg = SystemConfig(**d)

    lumins = [(l.x, l.y) for l in req.luminaires] or [(0.0, 0.0)]
    try:
        field = coverage_field(
            cfg, lumins, req.total_power_W, req.height_m, req.half_extent_m,
            req.grid, req.half_angle_deg, req.node_budget_uW, req.dcdc_eff,
            req.ber_thresh,
        )
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=422, detail=str(e)) from e
    return CoverageResponse(**field)
