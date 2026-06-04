"""POST /api/firmware/parse — extract PHY params from an ESP32 .ino.

Static parse only (no execution): given one ESP's Arduino source and its role
(tx / rx), return the SystemConfig fields it implies plus per-field provenance,
so the UI can show the user what was read and apply it to the link config.
"""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from cosim.firmware_parse import parse_firmware

router = APIRouter()


class FirmwareParseRequest(BaseModel):
    role: str            # "tx" or "rx"
    source: str          # raw .ino text
    filename: str | None = None


class FirmwareFinding(BaseModel):
    config_field: str
    value: float
    source: str
    label: str
    confidence: str


class FirmwareInfo(BaseModel):
    label: str
    value: str
    source: str


class FirmwareParseResponse(BaseModel):
    role: str
    params: dict[str, float]
    findings: list[FirmwareFinding]
    info: list[FirmwareInfo]
    warnings: list[str]


@router.post("/parse", response_model=FirmwareParseResponse)
def parse(req: FirmwareParseRequest) -> FirmwareParseResponse:
    r = parse_firmware(req.source, req.role)
    return FirmwareParseResponse(
        role=r.role,
        params=r.params,
        findings=[
            FirmwareFinding(
                config_field=f.config_field,
                value=f.value,
                source=f.source,
                label=f.label,
                confidence=f.confidence,
            )
            for f in r.findings
        ],
        info=[
            FirmwareInfo(label=i.label, value=i.value, source=i.source)
            for i in r.info
        ],
        warnings=r.warnings,
    )
