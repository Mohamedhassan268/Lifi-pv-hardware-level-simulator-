"""Electrical rule checks for a user-drawn CircuitGraph.

Catches the circuits ngspice cannot converge on (or that are obviously wrong)
*before* they reach the solver, so the schematic editor can give actionable
feedback instead of a cryptic SPICE failure.

A graph here is the JSON shape emitted by the React editor and accepted by
``kicad.circuit_graph`` builders:

    {
      "title": str,
      "components": [{"ref","component_type","value","pins":{n:name}}],
      "nets":       [{"name", "pins":[{"component_ref","pin_number"}]}],
    }

Rules (each yields zero or more issues):
  - floating_pin    : a component pin not attached to any net
  - single_pin_net  : a net with only one pin (dangling wire)
  - no_ground       : no net named '0' / 'gnd'
  - no_supply       : an active part present but no 'vcc' net
"""

from dataclasses import dataclass
from typing import List, Dict, Any

# Component types that need supply rails to behave.
_ACTIVE_TYPES = {"INA", "INA322", "BPF_STAGE", "COMPARATOR", "TLV7011",
                 "TLV2379", "ADA4891", "OPA380"}
_GROUND_NAMES = {"0", "gnd", "GND"}


@dataclass
class ERCIssue:
    level: str   # "error" | "warning"
    rule: str
    message: str


def check_graph(graph: Dict[str, Any]) -> List[ERCIssue]:
    """Run all rules against a graph dict. Returns a list of issues."""
    issues: List[ERCIssue] = []
    components = graph.get("components", [])
    nets = graph.get("nets", [])

    # Build the set of (ref, pin_number) that ARE connected, and net membership.
    connected = set()
    net_pin_counts = {}
    net_names = set()
    for net in nets:
        name = net.get("name", "")
        net_names.add(name)
        pins = net.get("pins", [])
        net_pin_counts[name] = net_pin_counts.get(name, 0) + len(pins)
        for p in pins:
            connected.add((p["component_ref"], p["pin_number"]))

    # Rule: floating pins
    for comp in components:
        ref = comp["ref"]
        for pin_no in comp.get("pins", {}):
            # pins keys may be str (JSON) or int
            pn = int(pin_no)
            if (ref, pn) not in connected:
                issues.append(ERCIssue(
                    "error", "floating_pin",
                    f"{ref} pin {pn} ({comp['pins'][pin_no]}) is not connected.",
                ))

    # Rule: single-pin nets (dangling wires). A CHANNEL node's optical input and
    # an MCU node's unused GPIO/ADC legitimately have no electrical wire (the
    # LED's emission and the digital baseband aren't circuit nodes), so don't
    # flag those.
    ref_type = {c["ref"]: c.get("component_type") for c in components}
    marker_only_nets = set()
    for net in nets:
        pins = net.get("pins", [])
        if len(pins) == 1 and ref_type.get(pins[0]["component_ref"]) in ("CHANNEL", "MCU"):
            marker_only_nets.add(net.get("name", ""))
    for name, count in net_pin_counts.items():
        if count == 1 and name not in _GROUND_NAMES and name not in marker_only_nets:
            issues.append(ERCIssue(
                "warning", "single_pin_net",
                f"Net '{name}' connects only one pin (dangling).",
            ))

    # Rule: ground present
    if not (_GROUND_NAMES & net_names):
        issues.append(ERCIssue(
            "error", "no_ground",
            "Circuit has no ground net ('0'). SPICE needs a 0 reference node.",
        ))

    # Rule: supply present if any active part
    has_active = any(c.get("component_type") in _ACTIVE_TYPES for c in components)
    if has_active and "vcc" not in net_names:
        issues.append(ERCIssue(
            "warning", "no_supply",
            "Active components present but no 'vcc' supply net found.",
        ))

    return issues


def has_errors(issues: List[ERCIssue]) -> bool:
    return any(i.level == "error" for i in issues)
