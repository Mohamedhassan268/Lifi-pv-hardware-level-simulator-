# cosim/standards/types.py
"""
Shared dataclasses for the standards profile system. Lives in its own
module so individual profile files can import the types without a
circular dependency with cosim/standards/__init__.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional


@dataclass
class CriterionResult:
    label: str
    ok: bool
    actual: Optional[float]
    bound: Optional[float]
    detail: str = ""


@dataclass
class Criterion:
    label: str
    check: Callable[[dict[str, Any]], CriterionResult]


@dataclass
class PhyProfile:
    id: str
    label: str
    spec_section: str
    overrides: dict[str, Any] = field(default_factory=dict)
    criteria: list[Criterion] = field(default_factory=list)
    notes: str = ""


@dataclass
class ProfileSummary:
    id: str
    label: str
    spec_section: str
    notes: str
    overrides: dict[str, Any]
