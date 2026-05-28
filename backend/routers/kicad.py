"""KiCad export routes.

Surfaces the existing `kicad.export_kicad` machinery (CLI: `python cli.py
kicad`) over HTTP so the React UI can trigger a schematic + BOM export for
the currently configured preset.

Endpoints:
    GET  /api/kicad/presets                 — presets that have a graph builder
    POST /api/kicad/export                   — run an export, return paths/warnings
    GET  /api/kicad/download/{preset}/{kind} — download the produced file
                                               (kind ∈ {"sch", "bom"})
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from backend.schemas import (
    KicadExportRequest,
    KicadExportResponse,
    KicadPresetsResponse,
)

router = APIRouter()


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_OUT_DIR = _PROJECT_ROOT / "workspace" / "kicad"


def _export_dir() -> Path:
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    return _OUT_DIR


@router.get("/presets", response_model=KicadPresetsResponse)
def list_kicad_presets() -> KicadPresetsResponse:
    """Presets that currently have a registered KiCad graph builder.

    Today: only ``kadirvelu2021``. Adding a new builder = registering a
    callable via ``kicad.kicad_exporter.register_builder``.
    """
    from kicad.kicad_exporter import available_presets

    return KicadPresetsResponse(presets=available_presets())


@router.post("/export", response_model=KicadExportResponse)
def export_kicad_endpoint(req: KicadExportRequest) -> KicadExportResponse:
    """Run the KiCad export for one preset.

    Files are written to ``<project>/workspace/kicad/`` and can be retrieved
    via ``GET /api/kicad/download/{preset}/{kind}``.
    """
    from kicad import export_kicad
    from kicad.kicad_exporter import available_presets

    if req.preset not in available_presets():
        raise HTTPException(
            status_code=404,
            detail=(
                f"No KiCad graph builder registered for preset {req.preset!r}. "
                f"Available: {available_presets()}"
            ),
        )

    try:
        result = export_kicad(req.preset, _export_dir())
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"KiCad export failed: {exc}",
        ) from exc

    return KicadExportResponse(
        preset=result.preset,
        schematic_path=str(result.schematic_path),
        bom_path=str(result.bom_path),
        component_count=result.component_count,
        net_count=result.net_count,
        warnings=list(result.warnings),
    )


@router.get("/download/{preset}/{kind}")
def download_kicad_file(preset: str, kind: str) -> FileResponse:
    """Serve a previously-exported schematic or BOM file."""
    if kind not in ("sch", "bom"):
        raise HTTPException(status_code=400, detail="kind must be 'sch' or 'bom'")

    if kind == "sch":
        path = _export_dir() / f"{preset}.kicad_sch"
        media_type = "application/octet-stream"
        filename = f"{preset}.kicad_sch"
    else:
        path = _export_dir() / f"{preset}_bom.csv"
        media_type = "text/csv"
        filename = f"{preset}_bom.csv"

    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"{path.name} not found — run POST /api/kicad/export first.",
        )

    return FileResponse(path, media_type=media_type, filename=filename)
