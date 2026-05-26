"""
REST surface for PHY compliance profiles.

    GET /api/standards               → list of ProfileSummary
    GET /api/standards/{profile_id}  → full PhyProfile (overrides + criteria labels)
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from fastapi import APIRouter, HTTPException

from cosim.standards import available_profiles, get_profile

router = APIRouter()


@router.get("")
def list_profiles() -> list[dict[str, Any]]:
    return [asdict(p) for p in available_profiles()]


@router.get("/{profile_id}")
def fetch_profile(profile_id: str) -> dict[str, Any]:
    try:
        p = get_profile(profile_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return {
        "id": p.id,
        "label": p.label,
        "spec_section": p.spec_section,
        "notes": p.notes,
        "overrides": dict(p.overrides),
        "criteria": [c.label for c in p.criteria],
    }
