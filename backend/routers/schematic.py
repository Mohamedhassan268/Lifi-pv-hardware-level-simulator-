"""Schematic rendering routes.

Reuses the existing schemdraw drawing registry at
`systems.paper_schematics.get_paper_info()` (8 + 5 + 6 + 5 + 5 + 5 = 34
hand-tuned drawings across 6 papers) and renders each one to SVG on demand
so the React UI can embed it.

Designed to be CircuitGraph-friendly: a future commit can register new
entries in the same shape — `{name: str, render: () -> schemdraw.Drawing}` —
so auto-layout entries slot in without router changes.

Endpoints:
    GET  /api/schematic/index             — registry shape (presets + drawings)
    GET  /api/schematic/{preset}/{idx}    — SVG for one drawing (idx into list)
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

router = APIRouter()


def _load_registry() -> dict[str, Any]:
    """Lazy-load the schemdraw paper registry.

    Returns {} if schemdraw isn't installed. Errors from individual paper
    modules don't take the whole registry down — they just omit that paper.
    """
    try:
        from systems.paper_schematics import get_paper_info
    except Exception:
        return {}
    try:
        return get_paper_info() or {}
    except Exception:
        return {}


@router.get("/index")
def schematic_index() -> dict[str, Any]:
    """List every paper and the schematics it has.

    Shape:
        {
          "presets": [
            {"key": "kadirvelu2021",
             "label": "Kadirvelu 2021 — ...",
             "drawings": [{"index": 0, "name": "Full System"}, ...]},
            ...
          ]
        }
    """
    registry = _load_registry()
    presets = []
    for key, info in registry.items():
        drawings = [
            {"index": i, "name": name}
            for i, (name, _func) in enumerate(info.get("schematics", []))
        ]
        presets.append(
            {
                "key": key,
                "label": info.get("label", key),
                "drawings": drawings,
            }
        )
    presets.sort(key=lambda p: p["key"])
    return {"presets": presets}


@router.get("/{preset}/{idx}")
def render_schematic(preset: str, idx: int) -> Response:
    """Render one drawing to SVG and return it inline."""
    registry = _load_registry()
    if preset not in registry:
        raise HTTPException(
            status_code=404,
            detail=f"Preset {preset!r} has no schematic registry entry.",
        )
    drawings = registry[preset].get("schematics", [])
    if idx < 0 or idx >= len(drawings):
        raise HTTPException(
            status_code=404,
            detail=(
                f"Drawing index {idx} out of range for {preset!r} "
                f"(0..{len(drawings) - 1})."
            ),
        )
    name, func = drawings[idx]

    try:
        # Some generic drawings accept a `preset` kwarg (e.g. draw_generic_channel)
        # — pass it when supported, else call no-args.
        try:
            drawing = func(preset=preset)
        except TypeError:
            drawing = func()
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Render failed for {preset}/{name!r}: {exc}",
        ) from exc

    if drawing is None:
        raise HTTPException(
            status_code=500,
            detail=(
                f"Renderer for {preset}/{name!r} returned None "
                "(schemdraw may not be installed)."
            ),
        )

    try:
        svg = drawing.get_imagedata("svg")
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"SVG export failed for {preset}/{name!r}: {exc}",
        ) from exc

    if isinstance(svg, str):
        svg = svg.encode("utf-8")

    return Response(content=svg, media_type="image/svg+xml")
