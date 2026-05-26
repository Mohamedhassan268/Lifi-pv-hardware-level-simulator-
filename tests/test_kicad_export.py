"""Tests for the KiCad schematic + BOM export feature."""

import csv
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kicad import export_kicad
from kicad.circuit_graph import CircuitGraph, ComponentInstance
from kicad.kicad_exporter import available_presets
from kicad.part_library import PART_METADATA, get_part_metadata
from kicad import bom_writer, validator
from systems.kadirvelu2021_graph import build_graph as build_kadirvelu


def _fake_component(ref, ctype="R", value="10k") -> ComponentInstance:
    meta = get_part_metadata(ctype)
    return ComponentInstance(
        ref=ref,
        component_type=ctype,
        value=value,
        footprint=meta.footprint,
        symbol_lib_id=meta.symbol_lib_id,
        pins=dict(meta.pins),
        part_number=meta.part_number,
        manufacturer=meta.manufacturer,
        datasheet_url=meta.datasheet_url,
    )


# --------------------------------------------------------------------------
# Metadata completeness
# --------------------------------------------------------------------------

def test_part_library_entries_have_required_fields():
    """Every registered KiCad part must carry symbol + pin metadata."""
    for name, meta in PART_METADATA.items():
        assert meta.symbol_lib_id, f"{name} missing symbol_lib_id"
        assert meta.pins, f"{name} missing pins"
        # Power flags are symbolic only and correctly have no footprint.
        if name not in ("VCC_FLAG", "GND_FLAG"):
            assert meta.footprint, f"{name} missing footprint"
        assert ":" in meta.symbol_lib_id, (
            f"{name} symbol_lib_id must be 'lib:symbol', got {meta.symbol_lib_id}"
        )


def test_kadirvelu_components_all_registered():
    """All parts referenced by the Kadirvelu graph builder must be registered."""
    graph = build_kadirvelu()
    for comp in graph.components:
        assert comp.component_type in PART_METADATA, (
            f"{comp.ref}: unregistered component_type '{comp.component_type}'"
        )


# --------------------------------------------------------------------------
# Validator
# --------------------------------------------------------------------------

def test_validator_accepts_well_formed_graph():
    g = CircuitGraph(title="t")
    g.add_component(_fake_component("R1"))
    g.connect("N1", "R1", 1)
    g.connect("N2", "R1", 2)
    warnings = validator.validate(g)
    assert warnings == []


def test_validator_rejects_unknown_component_ref():
    g = CircuitGraph(title="t")
    g.connect("N1", "R99", 1)
    with pytest.raises(validator.CircuitGraphValidationError, match="unknown component"):
        validator.validate(g)


def test_validator_rejects_unknown_pin():
    g = CircuitGraph(title="t")
    g.add_component(_fake_component("R1"))
    g.connect("N1", "R1", 99)
    with pytest.raises(validator.CircuitGraphValidationError, match="pin 99"):
        validator.validate(g)


def test_validator_rejects_pin_on_two_nets():
    g = CircuitGraph(title="t")
    g.add_component(_fake_component("R1"))
    g.connect("NET_A", "R1", 1)
    g.connect("NET_B", "R1", 1)
    with pytest.raises(validator.CircuitGraphValidationError, match="on two nets"):
        validator.validate(g)


def test_validator_warns_on_unconnected_pins():
    g = CircuitGraph(title="t")
    g.add_component(_fake_component("R1"))
    g.connect("N1", "R1", 1)
    # pin 2 left unconnected
    warnings = validator.validate(g)
    assert any("R1.2" in w for w in warnings)


# --------------------------------------------------------------------------
# BOM writer merges duplicates
# --------------------------------------------------------------------------

def test_bom_merges_identical_parts(tmp_path):
    g = CircuitGraph(title="t")
    g.add_component(_fake_component("R1", "R", "10k"))
    g.add_component(_fake_component("R2", "R", "10k"))
    g.add_component(_fake_component("R3", "R", "100k"))
    g.add_component(_fake_component("C1", "C", "10nF"))

    path = bom_writer.write(g, tmp_path / "bom.csv")
    rows = list(csv.DictReader(path.open()))

    # 3 unique (value, type) groups: 10k R, 100k R, 10nF C
    assert len(rows) == 3
    ten_k = next(r for r in rows if r["value"] == "10k")
    assert int(ten_k["qty"]) == 2
    assert ten_k["refs"] == "R1, R2"


def test_bom_excludes_power_flags(tmp_path):
    g = CircuitGraph(title="t")
    g.add_component(_fake_component("R1", "R", "10k"))
    vcc_meta = get_part_metadata("VCC_FLAG")
    g.add_component(ComponentInstance(
        ref="#PWR01", component_type="VCC_FLAG", value="VCC",
        footprint=vcc_meta.footprint, symbol_lib_id=vcc_meta.symbol_lib_id,
        pins=dict(vcc_meta.pins),
    ))

    path = bom_writer.write(g, tmp_path / "bom.csv")
    rows = list(csv.DictReader(path.open()))
    assert all(r["component_type"] != "VCC_FLAG" for r in rows)


# --------------------------------------------------------------------------
# End-to-end: Kadirvelu export roundtrip
# --------------------------------------------------------------------------

def test_kadirvelu_available_as_preset():
    assert "kadirvelu2021" in available_presets()


def test_kadirvelu_graph_passes_validation():
    g = build_kadirvelu()
    # Fatal issues raise; warnings are allowed (unconnected pins on quad op-amp).
    validator.validate(g)


def test_kadirvelu_full_export(tmp_path):
    result = export_kicad("kadirvelu2021", tmp_path)
    assert result.schematic_path.exists()
    assert result.bom_path.exists()
    assert result.component_count > 10
    assert result.net_count > 5

    # Schematic file is a valid S-expression shell.
    content = result.schematic_path.read_text()
    assert content.startswith("(kicad_sch")
    assert content.rstrip().endswith(")")
    assert "(generator " in content
    # Every component should have a symbol block.
    assert content.count("(symbol") >= result.component_count

    # BOM has a header row plus at least one data row.
    rows = list(csv.DictReader(result.bom_path.open()))
    assert len(rows) >= 1
    assert {"qty", "refs", "value", "part_number", "footprint"}.issubset(rows[0].keys())
