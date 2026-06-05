"""POST /api/firmware/parse — extract PHY params from an ESP32 .ino.

Static parse only (no execution): given one ESP's Arduino source and its role
(tx / rx), return the SystemConfig fields it implies plus per-field provenance,
so the UI can show the user what was read and apply it to the link config.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from cosim.firmware_gen import generate_firmware
from cosim.firmware_parse import parse_firmware
from cosim.system_config import SystemConfig

router = APIRouter()

# PHY fields the canvas may override when generating (else the preset's value).
_GEN_OVERRIDES = (
    "modulation", "carrier_freq_hz", "modulation_depth", "bfsk_f0_hz", "bfsk_f1_hz",
    "data_rate_bps", "prbs_order", "adc_bits", "adc_vref", "mcu_sample_rate_hz", "n_bits",
)


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


class FirmwareGenRequest(BaseModel):
    preset: str = "lifi_poc_breadboard"
    # Optional PHY overrides (from the canvas); anything omitted uses the preset.
    overrides: dict[str, float | str] | None = None


class FirmwareGenResponse(BaseModel):
    scheme: str
    tx: str
    rx: str
    notes: list[str]


@router.post("/generate", response_model=FirmwareGenResponse)
def generate(req: FirmwareGenRequest) -> FirmwareGenResponse:
    try:
        cfg = SystemConfig.from_preset(req.preset)
    except Exception as e:
        raise HTTPException(status_code=422,
                            detail=f"Unknown preset '{req.preset}'.") from e
    for k, v in (req.overrides or {}).items():
        if k in _GEN_OVERRIDES and hasattr(cfg, k):
            setattr(cfg, k, v)
    try:
        g = generate_firmware(cfg)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    return FirmwareGenResponse(scheme=g["scheme"], tx=g["tx"], rx=g["rx"], notes=g["notes"])


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
