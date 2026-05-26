# cosim/tolerance.py
"""
Per-component tolerance metadata.

Each entry maps a datasheet / process spread to a SystemConfig flat field
the Monte Carlo engine already knows how to vary. The metadata lives in
one place (this file) rather than scattered across each component class —
mirrors the kicad/part_library.PART_METADATA pattern.

To add a new component: append to PART_TOLERANCES with the SystemConfig
fields it influences. To opt-out of auto tolerance: pass a manual
`tolerance_spec` dict to `run_monte_carlo()` as before.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


@dataclass
class ToleranceSpec:
    """A single ±fractional tolerance band on a flat SystemConfig field."""
    field: str          # SystemConfig flat key, e.g. "sc_responsivity"
    fractional: float   # 0.08 = ±8 %
    source: str         # "datasheet" | "process" | "estimate"


# Keyed by SystemConfig field that PICKS the component (e.g. `pv_part`)
# and the chosen part value, mapping to the tolerances that part imposes.
# Generics (resistors, capacitors) are listed under PROCESS_DEFAULTS below.
PART_TOLERANCES: dict[tuple[str, str], list[ToleranceSpec]] = {
    # ---------------- PV / Photodiodes ----------------
    ("pv_part", "KXOB25-04X3F"): [
        ToleranceSpec("sc_responsivity", 0.08, "datasheet"),
        ToleranceSpec("sc_iph_uA", 0.15, "datasheet"),
        ToleranceSpec("sc_cj_nF", 0.20, "process"),
        ToleranceSpec("sc_rsh_kOhm", 0.20, "process"),
    ],
    ("pv_part", "BPW34"): [
        ToleranceSpec("sc_responsivity", 0.10, "datasheet"),
        ToleranceSpec("sc_cj_nF", 0.25, "process"),
    ],
    # ---------------- LEDs ----------------
    ("led_part", "LXM5-PD01"): [
        ToleranceSpec("led_radiated_power_mW", 0.15, "datasheet"),
        ToleranceSpec("led_half_angle_deg", 0.05, "datasheet"),
        ToleranceSpec("led_gled", 0.10, "datasheet"),
    ],
    # ---------------- Instrumentation amps ----------------
    ("ina_part", "INA322"): [
        ToleranceSpec("ina_gain_dB", 0.02, "datasheet"),  # ±2% gain error
        ToleranceSpec("ina_gbw_kHz", 0.20, "process"),
    ],
    ("ina_part", "TLV2379"): [
        ToleranceSpec("ina_gain_dB", 0.02, "datasheet"),
    ],
}

# Process spreads applied unconditionally to passive networks (the BPF
# resistors and capacitors are real parts on the board and inherit their
# class tolerance regardless of which active part sits between them).
PROCESS_DEFAULTS: list[ToleranceSpec] = [
    ToleranceSpec("bpf_rhp", 0.01, "process"),
    ToleranceSpec("bpf_rlp", 0.01, "process"),
    ToleranceSpec("bpf_chp_pF", 0.10, "process"),
    ToleranceSpec("bpf_clf_nF", 0.10, "process"),
    ToleranceSpec("r_sense_ohm", 0.01, "process"),
]


def derive_tolerance_spec(cfg: Any) -> dict[str, float]:
    """Walk the components referenced by `cfg` and return the merged
    tolerance dict for use with `run_monte_carlo()` /
    `stream_monte_carlo()`. Fields not present on the config are skipped.

    When two sources provide a tolerance for the same field, the larger
    fractional value wins (conservative).
    """
    out: dict[str, float] = {}
    items: Iterable[ToleranceSpec] = list(PROCESS_DEFAULTS)
    for picker_field, value in (
        ("pv_part", getattr(cfg, "pv_part", None)),
        ("led_part", getattr(cfg, "led_part", None)),
        ("ina_part", getattr(cfg, "ina_part", None)),
    ):
        if not value:
            continue
        items = [*items, *PART_TOLERANCES.get((picker_field, value), [])]

    for spec in items:
        if not hasattr(cfg, spec.field):
            continue
        prev = out.get(spec.field)
        out[spec.field] = max(spec.fractional, prev) if prev is not None else spec.fractional
    return out


def describe_tolerance_spec(cfg: Any) -> list[dict[str, Any]]:
    """Like derive_tolerance_spec but returns the human-readable rows for
    the UI: one entry per (field, source, fractional, picker)."""
    rows: list[dict[str, Any]] = []
    seen: dict[str, dict[str, Any]] = {}
    for picker_field, value, label in (
        ("pv_part", getattr(cfg, "pv_part", None), getattr(cfg, "pv_part", None)),
        ("led_part", getattr(cfg, "led_part", None), getattr(cfg, "led_part", None)),
        ("ina_part", getattr(cfg, "ina_part", None), getattr(cfg, "ina_part", None)),
    ):
        if not value:
            continue
        for spec in PART_TOLERANCES.get((picker_field, value), []):
            if not hasattr(cfg, spec.field):
                continue
            existing = seen.get(spec.field)
            if existing and existing["fractional"] >= spec.fractional:
                continue
            seen[spec.field] = {
                "field": spec.field,
                "fractional": spec.fractional,
                "source": spec.source,
                "via": f"{picker_field}={label}",
            }

    for spec in PROCESS_DEFAULTS:
        if not hasattr(cfg, spec.field):
            continue
        existing = seen.get(spec.field)
        if existing and existing["fractional"] >= spec.fractional:
            continue
        seen[spec.field] = {
            "field": spec.field,
            "fractional": spec.fractional,
            "source": spec.source,
            "via": "passive network",
        }

    rows = list(seen.values())
    rows.sort(key=lambda r: r["field"])
    return rows
