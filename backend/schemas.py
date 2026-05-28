"""
Pydantic schemas used by the FastAPI routers.

SystemConfig is the existing dataclass from cosim/system_config.py; we don't
mirror its 75+ fields here. Instead, REST endpoints accept/return a flat
`dict[str, Any]` (the same shape as `SystemConfig.to_dict()`) and validate by
attempting to construct a SystemConfig from it. That keeps a single source of
truth on the Python side.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class PresetSummary(BaseModel):
    name: str
    paper_reference: Optional[str] = None
    modulation: Optional[str] = None
    data_rate_bps: Optional[float] = None


class ValidationIssueModel(BaseModel):
    level: str            # "error" | "warning" | "info"
    field: str            # dotted path, e.g. "tx.bias_current_A"
    message: str
    suggestion: str = ""
    rule_id: str = ""


class KicadExportRequest(BaseModel):
    preset: str


class KicadExportResponse(BaseModel):
    preset: str
    schematic_path: str
    bom_path: str
    component_count: int
    net_count: int
    warnings: list[str] = Field(default_factory=list)


class KicadPresetsResponse(BaseModel):
    presets: list[str] = Field(default_factory=list)


class ConfigValidateResponse(BaseModel):
    valid: bool
    errors: list[str] = Field(default_factory=list)
    # Structured issues from cosim.validation. Includes ERROR-level entries
    # (same content as `errors`) plus WARNING and INFO. UIs that want
    # color-coding / inline placement read `issues`; legacy clients can
    # still read `errors`.
    issues: list[ValidationIssueModel] = Field(default_factory=list)
    normalized: Optional[dict[str, Any]] = None


class LinkBudgetRequest(BaseModel):
    """Subset of SystemConfig fields needed for the link-budget calculation."""

    distance_m: float
    tx_angle_deg: float = 0.0
    rx_tilt_deg: float = 0.0
    fov_half_angle_deg: float = 90.0
    led_half_angle_deg: float = 9.0
    led_radiated_power_mW: float = 9.3
    sc_area_cm2: float = 9.0
    sc_responsivity: float = 0.457
    modulation_depth: float = 0.33
    data_rate_bps: float = 5000.0
    r_sense_ohm: float = 1.0
    beer_lambert_enabled: bool = False
    humidity_rh: Optional[float] = None
    n_reflections: int = 0
    room_length_m: float = 5.0
    room_width_m: float = 5.0
    room_height_m: float = 3.0
    wall_reflectivity: float = 0.7
    temperature_K: float = 300.0


class LinkBudgetResponse(BaseModel):
    channel_gain: float
    p_rx_W: float
    p_rx_uW: float
    i_ph_A: float
    i_ph_uA: float
    lambertian_order: float
    snr_estimate_dB: float


class ComponentSummary(BaseModel):
    part: str
    category: str
    parameters: dict[str, Any]
