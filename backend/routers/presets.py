"""GET /api/presets — list and load preset configurations."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException

from cosim.system_config import SystemConfig

router = APIRouter()


@router.get("", response_model=list[str])
def list_presets() -> list[str]:
    return SystemConfig.list_presets()


@router.get("/{name}")
def get_preset(name: str) -> dict[str, Any]:
    try:
        cfg = SystemConfig.from_preset(name)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    return cfg.to_dict()
