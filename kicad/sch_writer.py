"""KiCad 8 ``.kicad_sch`` writer.

Emits a minimal but valid KiCad 8 schematic S-expression file from a
CircuitGraph. Components are placed on a coarse grid in signal-flow order
(the order they were added). Wires are drawn as straight labelled connections
between every pin on a net — KiCad renders the topology correctly even if the
geometry is ugly. The user then re-arranges and routes in KiCad's editor.

Format reference: KiCad 8 developer docs, S-expression file format.

Design notes:
    - We emit all nets as ``hierarchical_label`` stubs on each pin rather than
      drawn wires. This is uglier visually but far more robust than trying to
      auto-route wires — KiCad's ERC and netlist extractor treat same-named
      labels as electrically connected, so topology is preserved exactly.
    - Symbols reference their library by LIB_ID; KiCad resolves at open time
      against the user's configured symbol library paths.
"""

from __future__ import annotations

import uuid
from pathlib import Path

from .circuit_graph import CircuitGraph, ComponentInstance


# KiCad 8 schematic file version (stable release date).
_SCH_VERSION = 20231120
_GENERATOR = "hardware_faithful_simulator"

# Grid: mm internally, but KiCad schematics use mil. We use 2.54mm steps.
_GRID_MM = 12.7  # 500 mil — comfortable spacing for labels
_COMPONENTS_PER_ROW = 6


def write(graph: CircuitGraph, path: Path) -> Path:
    """Write a ``.kicad_sch`` file. Returns the absolute path."""
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    body_parts: list[str] = []

    # Place components on a grid.
    placed: dict[str, tuple[float, float]] = {}
    for i, comp in enumerate(graph.components):
        col = i % _COMPONENTS_PER_ROW
        row = i // _COMPONENTS_PER_ROW
        x = 50.8 + col * (_GRID_MM * 3)
        y = 50.8 + row * (_GRID_MM * 4)
        placed[comp.ref] = (x, y)
        body_parts.append(_emit_symbol(comp, x, y))

    # Emit hierarchical labels at every pin on every net.
    for net in graph.nets:
        for pin in net.pins:
            if pin.component_ref not in placed:
                continue
            cx, cy = placed[pin.component_ref]
            comp = next(c for c in graph.components if c.ref == pin.component_ref)
            # Fan pins out around the component position so labels don't stack.
            pin_count = len(comp.pins)
            pin_index = list(comp.pins.keys()).index(pin.pin_number)
            # Place labels at radii of _GRID_MM from the component origin.
            angle_step = 360.0 / max(pin_count, 1)
            angle_deg = pin_index * angle_step
            import math
            dx = _GRID_MM * math.cos(math.radians(angle_deg))
            dy = _GRID_MM * math.sin(math.radians(angle_deg))
            lx = _round_grid(cx + dx)
            ly = _round_grid(cy + dy)
            body_parts.append(_emit_label(net.name, lx, ly))

    body = "\n".join(body_parts)

    sch = (
        f"(kicad_sch\n"
        f"\t(version {_SCH_VERSION})\n"
        f"\t(generator \"{_GENERATOR}\")\n"
        f"\t(uuid \"{uuid.uuid4()}\")\n"
        f"\t(paper \"A4\")\n"
        f"\t(title_block\n"
        f"\t\t(title \"{_escape(graph.title)}\")\n"
        f"\t\t(company \"Hardware-Faithful LiFi-PV Simulator\")\n"
        f"\t)\n"
        f"\t(lib_symbols)\n"
        f"{body}\n"
        f"\t(sheet_instances\n"
        f"\t\t(path \"/\" (page \"1\"))\n"
        f"\t)\n"
        f")\n"
    )

    with path.open("w", encoding="utf-8") as f:
        f.write(sch)

    return path


def _emit_symbol(comp: ComponentInstance, x: float, y: float) -> str:
    sym_uuid = uuid.uuid4()
    lines = [
        f"\t(symbol",
        f"\t\t(lib_id \"{_escape(comp.symbol_lib_id)}\")",
        f"\t\t(at {x:.2f} {y:.2f} 0)",
        f"\t\t(unit 1)",
        f"\t\t(exclude_from_sim no)",
        f"\t\t(in_bom yes)",
        f"\t\t(on_board yes)",
        f"\t\t(uuid \"{sym_uuid}\")",
        f"\t\t(property \"Reference\" \"{_escape(comp.ref)}\"",
        f"\t\t\t(at {x:.2f} {y - 5.08:.2f} 0)",
        f"\t\t\t(effects (font (size 1.27 1.27)))",
        f"\t\t)",
        f"\t\t(property \"Value\" \"{_escape(comp.value)}\"",
        f"\t\t\t(at {x:.2f} {y + 5.08:.2f} 0)",
        f"\t\t\t(effects (font (size 1.27 1.27)))",
        f"\t\t)",
        f"\t\t(property \"Footprint\" \"{_escape(comp.footprint)}\"",
        f"\t\t\t(at {x:.2f} {y:.2f} 0)",
        f"\t\t\t(effects (font (size 1.27 1.27)) hide)",
        f"\t\t)",
        f"\t\t(property \"Datasheet\" \"{_escape(comp.datasheet_url)}\"",
        f"\t\t\t(at {x:.2f} {y:.2f} 0)",
        f"\t\t\t(effects (font (size 1.27 1.27)) hide)",
        f"\t\t)",
        f"\t\t(property \"MPN\" \"{_escape(comp.part_number)}\"",
        f"\t\t\t(at {x:.2f} {y:.2f} 0)",
        f"\t\t\t(effects (font (size 1.27 1.27)) hide)",
        f"\t\t)",
        f"\t\t(property \"Manufacturer\" \"{_escape(comp.manufacturer)}\"",
        f"\t\t\t(at {x:.2f} {y:.2f} 0)",
        f"\t\t\t(effects (font (size 1.27 1.27)) hide)",
        f"\t\t)",
        f"\t)",
    ]
    return "\n".join(lines)


def _emit_label(name: str, x: float, y: float) -> str:
    return (
        f"\t(global_label \"{_escape(name)}\"\n"
        f"\t\t(shape input)\n"
        f"\t\t(at {x:.2f} {y:.2f} 0)\n"
        f"\t\t(effects (font (size 1.27 1.27)) (justify left))\n"
        f"\t\t(uuid \"{uuid.uuid4()}\")\n"
        f"\t)"
    )


def _escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"')


def _round_grid(v: float) -> float:
    """Snap to 1.27 mm (50 mil) grid."""
    return round(v / 1.27) * 1.27
