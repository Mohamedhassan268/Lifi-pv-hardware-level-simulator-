"""Feature extraction endpoints — derived ML/analytics features for a link.

  POST /api/features          → run one Python simulation, return the full
                                feature record + the FEATURES schema.
  POST /api/features/dataset  → sweep one config field, return one feature row
                                per point. `format` = json (default) | csv | jsonl;
                                csv/jsonl come back as a downloadable text body.
  GET  /api/features/schema   → the FEATURES metadata + sweepable axes.

The single-run feature record is also attached to the /ws/pipeline `metrics`
message, so the in-app analytics panel gets it without an extra round trip;
these endpoints exist for programmatic / dataset use.
"""

from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

from cosim.features import FEATURES, extract_features
from cosim.feature_sweep import SWEEPABLE, run_feature_sweep, to_csv, to_jsonl
from cosim.python_engine import run_python_simulation
from cosim.system_config import SystemConfig

router = APIRouter()


class FeatureRequest(BaseModel):
    config: dict[str, Any] = {}
    n_bits: Optional[int] = None


class DatasetRequest(BaseModel):
    config: dict[str, Any] = {}
    param: str
    values: list[float]
    n_bits: Optional[int] = None
    format: str = "json"  # json | csv | jsonl


def _build_cfg(flat: dict[str, Any], n_bits: Optional[int]) -> SystemConfig:
    d = dict(flat)
    d["simulation_engine"] = "python"
    if n_bits is not None:
        d["n_bits"] = int(n_bits)
    try:
        return SystemConfig._from_flat_dict(d)
    except ValueError as e:
        raise HTTPException(status_code=422, detail={"errors": str(e).splitlines()}) from e


@router.get("/schema")
def feature_schema() -> dict:
    return {"features": FEATURES, "sweepable": list(SWEEPABLE)}


@router.post("")
def features(req: FeatureRequest) -> dict:
    cfg = _build_cfg(req.config, req.n_bits)
    result = run_python_simulation(cfg)
    return {"features": extract_features(result, cfg), "schema": FEATURES}


@router.post("/dataset")
def dataset(req: DatasetRequest):
    if req.param not in SWEEPABLE:
        raise HTTPException(
            status_code=422,
            detail={"errors": [f"param {req.param!r} not sweepable; choose from {list(SWEEPABLE)}"]},
        )
    if not req.values:
        raise HTTPException(status_code=422, detail={"errors": ["values must be non-empty"]})

    cfg = _build_cfg(req.config, req.n_bits)
    rows = run_feature_sweep(cfg, req.param, req.values, n_bits=req.n_bits)

    fmt = req.format.lower()
    if fmt == "csv":
        return PlainTextResponse(
            to_csv(rows),
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="features_{req.param}.csv"'},
        )
    if fmt == "jsonl":
        return PlainTextResponse(
            to_jsonl(rows),
            media_type="application/x-ndjson",
            headers={"Content-Disposition": f'attachment; filename="features_{req.param}.jsonl"'},
        )
    return {"param": req.param, "rows": rows, "schema": FEATURES}
