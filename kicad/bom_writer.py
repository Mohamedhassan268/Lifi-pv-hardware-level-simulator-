"""BOM CSV writer for a CircuitGraph.

Merges identical parts by (part_number, value, footprint) and writes one
row per unique part with aggregated quantity and a comma-separated list of
reference designators.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

from .circuit_graph import CircuitGraph


BOM_COLUMNS = [
    "qty", "refs", "value", "part_number", "manufacturer",
    "footprint", "component_type", "datasheet",
]


def write(graph: CircuitGraph, path: Path) -> Path:
    """Write a merged BOM CSV. Returns the absolute path."""
    # Group refs by (value, part_number, footprint) so identical parts collapse.
    groups: dict[tuple[str, str, str, str, str, str], list[str]] = defaultdict(list)
    group_meta: dict[tuple, tuple[str, str]] = {}

    for comp in graph.components:
        # Skip power flags from BOM — they are not physical parts.
        if comp.component_type in ("VCC_FLAG", "GND_FLAG"):
            continue
        key = (
            comp.value,
            comp.part_number,
            comp.footprint,
            comp.manufacturer,
            comp.component_type,
            comp.datasheet_url,
        )
        groups[key].append(comp.ref)

    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(BOM_COLUMNS)
        # Stable ordering: by component_type, then value.
        for key in sorted(groups.keys(), key=lambda k: (k[4], k[0])):
            value, part_number, footprint, manufacturer, ctype, datasheet = key
            refs = sorted(groups[key], key=_ref_sort_key)
            writer.writerow([
                len(refs),
                ", ".join(refs),
                value,
                part_number,
                manufacturer,
                footprint,
                ctype,
                datasheet,
            ])

    return path


def _ref_sort_key(ref: str) -> tuple[str, int]:
    """Sort R1, R2, R10 correctly (prefix letters + numeric suffix)."""
    prefix = "".join(c for c in ref if c.isalpha())
    digits = "".join(c for c in ref if c.isdigit())
    return (prefix, int(digits) if digits else 0)
