"""
REST endpoints for the equation registry and probe inventory.

These power the ScienceCard / Virtual-Probe UI: the frontend pulls the live
LaTeX, the symbol-value table, and the citation list for any equation, and
discovers the full probe registry so it knows which react-flow edges to
make probeable.
"""

from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, Body, HTTPException

from cosim.equations import EQUATIONS, equation_keys, resolve_for_config
from cosim.probes import PROBE_REGISTRY, registry_export
from cosim.system_config import SystemConfig

router = APIRouter()


@router.get("/equations")
def list_equations() -> dict:
    """List every equation key with a short summary (no live values)."""
    return {
        "keys": equation_keys(),
        "summaries": [
            {
                "key": spec.key,
                "title": spec.title,
                "references": list(spec.references),
            }
            for spec in EQUATIONS.values()
        ],
    }


@router.post("/equations/{key}")
def evaluate_equation(key: str, config: dict[str, Any] = Body(default_factory=dict)) -> dict:
    """
    Resolve an equation's symbols for a posted SystemConfig payload.

    The body must be the same flat dict the frontend already sends to
    /ws/pipeline (config). Returns the full ScienceCard payload:
    {key, title, latex, plain, symbols: [{symbol, name, value, units, source}], references}
    """
    if key not in EQUATIONS:
        raise HTTPException(status_code=404, detail=f"unknown equation key: {key}")

    try:
        cfg = SystemConfig._from_flat_dict(config)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=f"invalid config: {e}") from e

    payload = resolve_for_config(key, cfg)
    if payload is None:
        raise HTTPException(status_code=500, detail="resolver returned no payload")
    return payload


@router.get("/probes")
def list_probes() -> dict:
    """Return the full ProbeRegistry as a serializable list.

    The frontend uses this to mirror the registry as TypeScript and to
    decide which react-flow edges should be marked probeable.
    """
    return {"probes": registry_export(), "count": len(PROBE_REGISTRY)}
