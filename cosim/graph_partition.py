"""Split a drawn circuit graph into its TX and RX analog islands.

The single-canvas link is TX-analog -> Python channel -> RX-analog. Electrically
the TX and RX are two separate islands joined only by the shared power/ground
rails (and, optically, by the Python channel — which is not a wire). This module
finds those two islands by union-find over the *signal* nets (rails excluded, so
the two sides don't merge through Vcc/ground), identifies the LED that drives the
channel and the photodetector that receives it, and reports the nets the
orchestrator needs: the gate-drive net and the RX optical-injection net.

Returns ``None`` when the graph isn't a two-pass link (no LED, or no
photodetector) — the caller then falls back to the RX-only path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from components import get_component
from components.base import LEDBase, PhotodetectorBase

# Global nets that both islands legitimately share; excluded from connectivity so
# the TX and RX sides remain distinct islands.
RAIL_NETS = {"vcc", "vee", "vref", "0", "gnd", "vss", "vdd"}


@dataclass
class Partition:
    tx_refs: set[str]
    rx_refs: set[str]
    led_ref: str
    pv_ref: str
    drive_net: str
    rx_optical_net: str


def _classify(ctype: str) -> Optional[str]:
    """'LED' / 'PV' / None for a component type (registry part name)."""
    try:
        part = get_component(ctype)
    except Exception:  # noqa: BLE001
        return None
    if isinstance(part, LEDBase):
        return "LED"
    if isinstance(part, PhotodetectorBase):
        return "PV"
    return None


def _optical_pin_index(ctype: str) -> Optional[int]:
    """1-based index of a photodetector's optical-input port, or None."""
    try:
        ports = get_component(ctype).spice_ports()
    except Exception:  # noqa: BLE001
        return None
    for i, p in enumerate(ports, start=1):
        if p.role.value == "optical_in":
            return i
    return None


def partition_tx_rx(graph: dict) -> Optional[Partition]:
    comps = {c["ref"]: c for c in graph.get("components", [])}
    if not comps:
        return None

    parent = {ref: ref for ref in comps}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    net_of: dict[tuple[str, int], str] = {}
    for net in graph.get("nets", []):
        name = net["name"]
        pin_refs = []
        for p in net.get("pins", []):
            ref = p["component_ref"]
            net_of[(ref, int(p["pin_number"]))] = name
            # CHANNEL (optical bridge) and MCU (digital baseband) are logical
            # nodes, never electrical conductors: exclude their pins from
            # connectivity so they can't merge the TX and RX islands.
            if ref in parent and comps[ref]["component_type"] not in ("CHANNEL", "MCU"):
                pin_refs.append(ref)
        if name.lower() in RAIL_NETS:
            continue
        for r in pin_refs[1:]:
            union(pin_refs[0], r)

    led_ref = pv_ref = None
    for ref, c in comps.items():
        kind = _classify(c["component_type"])
        if kind == "LED" and led_ref is None:
            led_ref = ref
        elif kind == "PV" and pv_ref is None:
            pv_ref = ref
    if led_ref is None or pv_ref is None:
        return None

    tx_root, rx_root = find(led_ref), find(pv_ref)
    tx_refs = {r for r in comps if find(r) == tx_root}
    rx_refs = {r for r in comps if find(r) == rx_root}

    # Gate drive: the DRIVE marker node sitting in the TX island.
    drive_net = None
    for ref in tx_refs:
        if comps[ref]["component_type"] == "DRIVE":
            drive_net = net_of.get((ref, 1))
            break

    # RX optical injection net: the photodetector's optical-input pin.
    opt_idx = _optical_pin_index(comps[pv_ref]["component_type"])
    rx_optical_net = net_of.get((pv_ref, opt_idx)) if opt_idx else None

    if drive_net is None or rx_optical_net is None:
        return None
    return Partition(tx_refs, rx_refs, led_ref, pv_ref, drive_net, rx_optical_net)


def subgraph(graph: dict, refs: set[str]) -> dict:
    """Project ``graph`` onto the component subset ``refs`` (keeps nets that
    touch the subset, with only the subset's pins on them)."""
    comps = [c for c in graph["components"] if c["ref"] in refs]
    nets = []
    for net in graph["nets"]:
        pins = [p for p in net["pins"] if p["component_ref"] in refs]
        if pins:
            nets.append({"name": net["name"], "pins": pins})
    return {"title": graph.get("title", ""), "components": comps, "nets": nets}
