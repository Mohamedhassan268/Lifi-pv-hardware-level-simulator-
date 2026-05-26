"""Circuit graph builder for Kadirvelu 2021 LiFi-PV system.

Topology source: systems/kadirvelu2021_netlist.py (SubcircuitLibrary) +
KadirveluParams for component values. Produces a CircuitGraph the KiCad
exporter can turn into .kicad_sch + BOM.

Signal flow (receive path):
    LED (optical) -> Solar cell -> R_sense -> INA322 (+ R1, R2 gain)
    -> BPF stage 1 (C_hp, R_hp, R_in, R_fb, C_fb, TLV2379-A)
    -> BPF stage 2 (C_hp, R_hp, R_in, R_fb, C_fb, TLV2379-B)
    -> TLV7011 comparator -> DOUT

Power path:
    Solar cell anode -> C_in -> L1 -> NTS4409 switch / Schottky diode
    -> C_out -> V_BAT (load).

Transmit path (informational):
    V_MOD -> ADA4891 buffer -> R_E -> BSD235N gate
    -> LED LXM5-PD01 cathode (anode to VCC_LED).
"""

from __future__ import annotations

import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from kicad.circuit_graph import CircuitGraph, ComponentInstance
from kicad.part_library import get_part_metadata
from systems.kadirvelu2021 import KadirveluParams


def _make(ref: str, component_type: str, value: str) -> ComponentInstance:
    """Build a ComponentInstance from the part library."""
    meta = get_part_metadata(component_type)
    return ComponentInstance(
        ref=ref,
        component_type=component_type,
        value=value,
        footprint=meta.footprint,
        symbol_lib_id=meta.symbol_lib_id,
        pins=dict(meta.pins),
        part_number=meta.part_number,
        manufacturer=meta.manufacturer,
        datasheet_url=meta.datasheet_url,
    )


def build_graph() -> CircuitGraph:
    p = KadirveluParams()

    g = CircuitGraph(title="Kadirvelu 2021 LiFi-PV Receiver + DC-DC + TX")

    # ---- Transmitter ----
    u_tx = g.add_component(_make("U1", "ADA4891", "ADA4891"))
    r_e = g.add_component(_make("R1", "R", f"{p.LED_DRIVER_RE:.1f}"))
    q_tx = g.add_component(_make("Q1", "BSD235N", "BSD235N"))
    d_led = g.add_component(_make("D1", "LXM5_PD01", "LXM5-PD01"))

    # ---- Receiver front end ----
    d_pv = g.add_component(_make("D2", "KXOB25_04X3F", "KXOB25-04X3F"))
    r_sense = g.add_component(_make("R2", "R", f"{p.R_SENSE:.1f}"))

    # INA322 + gain-setting resistors (R1, R2 on pins 1, 2 of INA322)
    u_ina = g.add_component(_make("U2", "INA322", "INA322"))
    r_g1 = g.add_component(_make("R3", "R", f"{p.INA_R1/1e3:.0f}k"))  # 191k
    r_g2 = g.add_component(_make("R4", "R", f"{p.INA_R2/1e3:.0f}k"))  # 10k

    # ---- BPF Stage 1 ----
    c_hp1 = g.add_component(_make("C1", "C", f"{p.BPF_CHP_pF*1e-3:.0f}nF"))
    r_hp1 = g.add_component(_make("R5", "R", f"{p.BPF_RHP/1e3:.0f}k"))
    r_in1 = g.add_component(_make("R6", "R", f"{p.BPF_RLP/1e3:.0f}k"))
    r_fb1 = g.add_component(_make("R7", "R", f"{p.BPF_RLP/1e3:.0f}k"))
    c_fb1 = g.add_component(_make("C2", "C", f"{p.BPF_CLF_nF:.1f}nF"))

    # ---- BPF Stage 2 ----
    c_hp2 = g.add_component(_make("C3", "C", f"{p.BPF_CHP_pF*1e-3:.0f}nF"))
    r_hp2 = g.add_component(_make("R8", "R", f"{p.BPF_RHP/1e3:.0f}k"))
    r_in2 = g.add_component(_make("R9", "R", f"{p.BPF_RLP/1e3:.0f}k"))
    r_fb2 = g.add_component(_make("R10", "R", f"{p.BPF_RLP/1e3:.0f}k"))
    c_fb2 = g.add_component(_make("C4", "C", f"{p.BPF_CLF_nF:.1f}nF"))

    # Quad op-amp package housing both BPF stages.
    u_bpf = g.add_component(_make("U3", "TLV2379", "TLV2379"))

    # ---- Comparator ----
    u_cmp = g.add_component(_make("U4", "TLV7011", "TLV7011"))

    # ---- DC-DC Boost ----
    c_dcdc_in = g.add_component(_make("C5", "C_POL", f"{p.DCDC_CP_uF:.0f}uF"))
    l_dcdc = g.add_component(_make("L1", "L", f"{p.DCDC_L_uH:.0f}uH"))
    q_dcdc = g.add_component(_make("Q2", "NTS4409", "NTS4409"))
    d_dcdc = g.add_component(_make("D3", "D_SCHOTTKY", "B5819W"))
    c_dcdc_out = g.add_component(_make("C6", "C_POL", f"{p.DCDC_CL_uF:.0f}uF"))

    # =================================================================
    # Nets
    # =================================================================

    # Supplies
    g.connect("VCC", "U1", 5)   # ADA4891 V+
    g.connect("VCC", "U2", 7)   # INA322 V+
    g.connect("VCC", "U3", 4)   # TLV2379 V+
    g.connect("VCC", "U4", 5)   # TLV7011 V+

    g.connect("GND", "U1", 2)   # ADA4891 V-
    g.connect("GND", "U2", 4)   # INA322 V-
    g.connect("GND", "U3", 11)  # TLV2379 V-
    g.connect("GND", "U4", 2)   # TLV7011 V-
    g.connect("GND", "R2", 2)   # R_sense low side to GND
    g.connect("GND", "C5", 2)   # DC-DC input cap low side
    g.connect("GND", "C6", 2)   # DC-DC output cap low side
    g.connect("GND", "Q1", 2)   # BSD235N source
    g.connect("GND", "Q2", 2)   # NTS4409 source

    # TX path: V_MOD -> U1 buffer -> R_E -> Q1 gate -> LED cathode
    g.connect("V_MOD", "U1", 3)          # ADA4891 IN+
    g.connect("TX_FBK", "U1", 1)         # ADA4891 OUT
    g.connect("TX_FBK", "U1", 4)         # ADA4891 IN- (unity-gain feedback)
    g.connect("TX_FBK", "R1", 1)
    g.connect("LED_GATE", "R1", 2)
    g.connect("LED_GATE", "Q1", 1)       # BSD235N gate
    g.connect("LED_K", "Q1", 3)          # BSD235N drain
    g.connect("LED_K", "D1", 1)          # LED cathode
    g.connect("VCC_LED", "D1", 2)        # LED anode

    # PV front end: solar cell output across R_sense, into INA322
    g.connect("SC_ANODE", "D2", 1)
    g.connect("SC_CATHODE", "D2", 2)
    g.connect("SC_CATHODE", "R2", 1)     # R_sense high side = SC_CATHODE
    # INA322: IN+ on sense-low (GND side of Rsense), IN- on SC_CATHODE
    g.connect("GND", "U2", 3)            # IN+ (pin 3) = sense low = GND
    g.connect("SC_CATHODE", "U2", 8)     # IN- (pin 8) = sense high
    g.connect("R_G1", "U2", 1)           # R1 pin on INA322
    g.connect("R_G2", "U2", 2)           # R2 pin on INA322
    g.connect("R_G1", "R3", 1)
    g.connect("R_G1_R3_R4", "R3", 2)
    g.connect("R_G1_R3_R4", "R4", 1)
    g.connect("R_G2", "R4", 2)
    g.connect("VREF", "U2", 5)           # INA322 REF pin
    g.connect("INA_OUT", "U2", 6)

    # BPF Stage 1 — uses TLV2379 op-amp unit A (pins 1, 2, 3)
    g.connect("INA_OUT", "C1", 1)
    g.connect("BPF1_HP", "C1", 2)
    g.connect("BPF1_HP", "R5", 1)
    g.connect("VREF", "R5", 2)
    g.connect("BPF1_HP", "R6", 1)
    g.connect("BPF1_INN", "R6", 2)
    g.connect("BPF1_INN", "U3", 2)       # TLV2379 unit A IN-
    g.connect("VREF", "U3", 3)           # TLV2379 unit A IN+ bias
    g.connect("BPF1_OUT", "U3", 1)       # TLV2379 unit A OUT
    g.connect("BPF1_INN", "R7", 1)
    g.connect("BPF1_OUT", "R7", 2)
    g.connect("BPF1_INN", "C2", 1)
    g.connect("BPF1_OUT", "C2", 2)

    # BPF Stage 2 — uses TLV2379 op-amp unit B (pins 5, 6, 7)
    g.connect("BPF1_OUT", "C3", 1)
    g.connect("BPF2_HP", "C3", 2)
    g.connect("BPF2_HP", "R8", 1)
    g.connect("VREF", "R8", 2)
    g.connect("BPF2_HP", "R9", 1)
    g.connect("BPF2_INN", "R9", 2)
    g.connect("BPF2_INN", "U3", 6)       # TLV2379 unit B IN-
    g.connect("VREF", "U3", 5)           # TLV2379 unit B IN+ bias
    g.connect("BPF2_OUT", "U3", 7)       # TLV2379 unit B OUT
    g.connect("BPF2_INN", "R10", 1)
    g.connect("BPF2_OUT", "R10", 2)
    g.connect("BPF2_INN", "C4", 1)
    g.connect("BPF2_OUT", "C4", 2)

    # Comparator slicer
    g.connect("BPF2_OUT", "U4", 3)       # TLV7011 IN+
    g.connect("VREF", "U4", 4)           # TLV7011 IN-
    g.connect("DOUT", "U4", 1)           # TLV7011 OUT

    # DC-DC boost: SC_ANODE -> C5 -> L1 -> Q2 (switch to GND) / D3 -> C6 -> V_BAT
    g.connect("SC_ANODE", "C5", 1)
    g.connect("SC_ANODE", "L1", 1)
    g.connect("DCDC_SW", "L1", 2)
    g.connect("DCDC_SW", "Q2", 3)        # NTS4409 drain
    g.connect("DCDC_PHI", "Q2", 1)       # NTS4409 gate (from MCU PWM)
    g.connect("DCDC_SW", "D3", 2)        # Schottky anode
    g.connect("V_BAT", "D3", 1)          # Schottky cathode
    g.connect("V_BAT", "C6", 1)

    return g
