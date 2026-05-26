"""Top-level KiCad export orchestrator.

Usage:
    from kicad import export_kicad
    result = export_kicad("kadirvelu2021", Path("workspace/kicad/"))
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from . import bom_writer, sch_writer, validator
from .circuit_graph import CircuitGraph


@dataclass
class ExportResult:
    preset: str
    schematic_path: Path
    bom_path: Path
    warnings: list[str] = field(default_factory=list)
    component_count: int = 0
    net_count: int = 0


# Graph builder registry.  Populated lazily to avoid circular imports.
_BUILDERS: dict[str, Callable[[], CircuitGraph]] = {}


def register_builder(preset: str, builder: Callable[[], CircuitGraph]) -> None:
    """Register a per-paper graph builder."""
    _BUILDERS[preset] = builder


def _load_builders() -> None:
    """Import builders on first use (avoids circular imports)."""
    if _BUILDERS:
        return
    try:
        from systems.kadirvelu2021_graph import build_graph as kadirvelu_build
        register_builder("kadirvelu2021", kadirvelu_build)
    except ImportError:
        pass


def available_presets() -> list[str]:
    _load_builders()
    return sorted(_BUILDERS.keys())


def export_kicad(preset: str, out_dir: Path) -> ExportResult:
    """Export KiCad schematic + BOM for a named preset.

    Args:
        preset: Preset name (e.g. "kadirvelu2021").
        out_dir: Directory to write ``<preset>.kicad_sch`` and
            ``<preset>_bom.csv`` into. Created if missing.

    Returns:
        ExportResult with resolved paths and validator warnings.
    """
    _load_builders()
    if preset not in _BUILDERS:
        raise KeyError(
            f"No KiCad graph builder registered for preset '{preset}'. "
            f"Available: {available_presets()}"
        )

    graph = _BUILDERS[preset]()
    warnings = validator.validate(graph)

    out_dir = Path(out_dir).resolve()
    sch_path = sch_writer.write(graph, out_dir / f"{preset}.kicad_sch")
    bom_path = bom_writer.write(graph, out_dir / f"{preset}_bom.csv")

    return ExportResult(
        preset=preset,
        schematic_path=sch_path,
        bom_path=bom_path,
        warnings=warnings,
        component_count=len(graph.components),
        net_count=len(graph.nets),
    )
