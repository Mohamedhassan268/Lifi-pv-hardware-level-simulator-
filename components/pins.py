"""Machine-readable SPICE port model for components.

A component's ``spice_subcircuit()`` emits a ``.SUBCKT NAME p1 p2 ...`` header.
``spice_ports()`` returns the same ports as structured ``Port`` objects so that
downstream tools (the graph -> netlist resolver, ERC, and a future drag-and-drop
canvas) know each port's name and electrical role without parsing SPICE text.

Kept here (low-level, no cosim dependency) so both ``components`` and ``cosim``
can import it.
"""

from dataclasses import dataclass
from enum import Enum


class PortRole(Enum):
    SIGNAL_IN = "signal_in"
    SIGNAL_OUT = "signal_out"
    SIGNAL = "signal"        # bidirectional / passive terminal
    SUPPLY_POS = "supply_pos"
    SUPPLY_NEG = "supply_neg"
    REF = "ref"              # reference / mid-rail bias
    OPTICAL_IN = "optical_in"
    OUTPUT = "output"        # circuit output marker (net read as V(dout))
    GROUND = "ground"        # ground / 0 reference marker


@dataclass(frozen=True)
class Port:
    name: str
    role: PortRole


# Canonical port list for a single-supply op-amp / comparator behavioral model.
OPAMP_PORTS = [
    Port("INP", PortRole.SIGNAL_IN),
    Port("INN", PortRole.SIGNAL_IN),
    Port("OUT", PortRole.SIGNAL_OUT),
    Port("VCC", PortRole.SUPPLY_POS),
    Port("VEE", PortRole.SUPPLY_NEG),
]
