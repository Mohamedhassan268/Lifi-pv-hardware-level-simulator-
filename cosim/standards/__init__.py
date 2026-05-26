# cosim/standards/__init__.py
"""
PHY-layer compliance profiles.

Each profile pins a subset of SystemConfig fields to the values defined
by a published standard and provides PASS/FAIL criteria the simulator
checks after a run. Currently only IEEE 802.15.7 PHY-I (OOK) is wired up;
the architecture is open for IEEE 802.15.7 PHY-II/III, IEEE 802.11bb,
and ITU-T G.9991 follow-ups.

Public surface
==============
    PROFILES                 — dict[id -> PhyProfile]
    available_profiles()     — list[ProfileSummary] for the UI
    get_profile(id)          — fetch one
    evaluate(p, metrics)     — list[CriterionResult]
"""

from __future__ import annotations

from typing import Any

from cosim.standards.types import (
    Criterion,
    CriterionResult,
    PhyProfile,
    ProfileSummary,
)
from cosim.standards.ieee_802_15_7_phy_i import IEEE_802_15_7_PHY_I


# Registry — adding a new profile means appending here.
PROFILES: dict[str, PhyProfile] = {
    IEEE_802_15_7_PHY_I.id: IEEE_802_15_7_PHY_I,
}


def available_profiles() -> list[ProfileSummary]:
    return [
        ProfileSummary(
            id=p.id,
            label=p.label,
            spec_section=p.spec_section,
            notes=p.notes,
            overrides=dict(p.overrides),
        )
        for p in PROFILES.values()
    ]


def get_profile(profile_id: str) -> PhyProfile:
    if profile_id not in PROFILES:
        raise KeyError(f"unknown standards profile: {profile_id}")
    return PROFILES[profile_id]


def evaluate(profile: PhyProfile, metrics: dict[str, Any]) -> list[CriterionResult]:
    return [c.check(metrics) for c in profile.criteria]


__all__ = [
    "PROFILES",
    "PhyProfile",
    "Criterion",
    "CriterionResult",
    "ProfileSummary",
    "available_profiles",
    "get_profile",
    "evaluate",
]
