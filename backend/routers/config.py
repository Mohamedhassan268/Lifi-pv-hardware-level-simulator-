"""POST /api/config/validate — round-trip a config through SystemConfig validation."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from backend.schemas import ConfigValidateResponse
from cosim.system_config import SystemConfig

router = APIRouter()


@router.post("/validate", response_model=ConfigValidateResponse)
def validate_config(payload: dict[str, Any]) -> ConfigValidateResponse:
    try:
        cfg = SystemConfig._from_flat_dict(payload)
    except ValueError as e:
        return ConfigValidateResponse(valid=False, errors=str(e).splitlines())
    return ConfigValidateResponse(valid=True, normalized=cfg.to_dict())
