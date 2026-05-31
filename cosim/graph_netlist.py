"""Translate a CircuitGraph into SPICE element lines (the graph -> netlist
hand-off).

This generalises the fixed-topology data path in
``SimulationPipeline._generate_rx_netlist``: instead of hardcoding the RX
wiring, we walk a ``kicad.circuit_graph.CircuitGraph`` and emit one SPICE
element per node, ordering subcircuit nodes by each part's SPICE port list.

It is the headless foothold for a drag-and-drop schematic editor: a user-drawn
graph can be turned into the same SPICE netlist the validated pipeline produces.
The subcircuit *definitions* still come from the pipeline's ``_subckt_*``
generators; this module only produces the instance/wiring lines.
"""

import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from kicad.circuit_graph import CircuitGraph, ComponentInstance

# SPICE subcircuit port order per component type. A graph pin_number is the
# 1-based index into these lists, so node order is recoverable for X-cards.
PORT_ORDER = {
    'SOLAR_CELL': ['anode', 'cathode', 'photo_in'],
    'INA':        ['INP', 'INN', 'OUT', 'VCC', 'VEE', 'REF'],
    'BPF_STAGE':  ['inp', 'out', 'vcc', 'vee', 'vref'],
    'COMPARATOR': ['INP', 'INN', 'OUT', 'VCC', 'VEE'],
}


def _net_lookup(graph):
    """Map (component_ref, pin_number) -> net name."""
    lookup = {}
    for net in graph.nets:
        for pin in net.pins:
            lookup[(pin.component_ref, pin.pin_number)] = net.name
    return lookup


def graph_to_instances(graph):
    """Emit SPICE element lines for every component in ``graph``.

    Subcircuit instances (type in ``PORT_ORDER``) become ``Xref n... TYPE``;
    two-terminal primitives ``R`` and ``V`` become standard SPICE cards. The
    component ``ref`` already carries the SPICE prefix (``Xsc``, ``Rsense``,
    ``Vgnd_ref``).
    """
    nets = _net_lookup(graph)
    lines = []
    for comp in graph.components:
        ctype = comp.component_type
        if ctype in PORT_ORDER:
            nodes = [nets[(comp.ref, i + 1)] for i in range(len(PORT_ORDER[ctype]))]
            lines.append(f"{comp.ref} {' '.join(nodes)} {ctype}")
        elif ctype == 'R':
            n1, n2 = nets[(comp.ref, 1)], nets[(comp.ref, 2)]
            lines.append(f"{comp.ref} {n1} {n2} {comp.value}")
        elif ctype == 'V':
            n1, n2 = nets[(comp.ref, 1)], nets[(comp.ref, 2)]
            lines.append(f"{comp.ref} {n1} {n2} DC {comp.value}")
        else:
            raise ValueError(f"graph_to_instances: unsupported type {ctype!r}")
    return '\n'.join(lines)


def graph_dict_to_instances(graph):
    """Emit SPICE element lines from the JSON graph shape sent by the editor.

    ``graph`` is ``{"components":[{ref,component_type,value,pins:{n:name}}],
    "nets":[{name, pins:[{component_ref,pin_number}]}]}``. Node order for an
    X-card follows the component's pin_number order (1-based), matching the
    SPICE subcircuit port list.
    """
    nets = {}
    for net in graph.get('nets', []):
        for pin in net.get('pins', []):
            nets[(pin['component_ref'], int(pin['pin_number']))] = net['name']

    lines = []
    for comp in graph.get('components', []):
        ref = comp['ref']
        ctype = comp['component_type']
        # OUT/GND/DRIVE/CHANNEL/MCU and the power-rail flags (VCC/VEE/VREF) are
        # label-only markers; they name a net but emit no SPICE element (the
        # supply sources, drive source, optical bridge and digital-baseband MCU
        # are injected/modelled outside SPICE, not drawn as devices).
        if ctype in ('OUT', 'GND', 'DRIVE', 'CHANNEL', 'MCU', 'VCC', 'VEE', 'VREF'):
            continue
        pin_map = comp.get('pins', {})
        n_pins = len(pin_map)
        nodes = []
        for i in range(1, n_pins + 1):
            node = nets.get((ref, i))
            if node is None:
                raise ValueError(f"{ref} pin {i} is not connected to any net")
            nodes.append(node)
        if ctype == 'R':
            lines.append(f"{ref} {nodes[0]} {nodes[1]} {comp.get('value', '1k')}")
        elif ctype == 'C':
            lines.append(f"{ref} {nodes[0]} {nodes[1]} {comp.get('value', '1u')}")
        elif ctype == 'V':
            lines.append(f"{ref} {nodes[0]} {nodes[1]} DC {comp.get('value', '0')}")
        else:
            lines.append(f"{ref} {' '.join(nodes)} {ctype}")
    return '\n'.join(lines)


def build_rx_graph(cfg):
    """Build the ``ina_bpf_comp`` RX-chain graph from a SystemConfig.

    Mirrors the topology in ``_generate_rx_netlist``:
        solar cell -> R_sense -> INA -> BPF(xN) -> comparator
    """
    g = CircuitGraph(title=f"{getattr(cfg, 'preset_name', 'rx')} RX chain")

    def add(ref, ctype, value=""):
        g.add_component(ComponentInstance(
            ref=ref, component_type=ctype, value=value,
            footprint="", symbol_lib_id="", pins={}))

    n_bpf = cfg.bpf_stages
    has_comp = cfg.comparator_part != 'N/A'

    add('Xsc', 'SOLAR_CELL')
    add('Rsense', 'R', str(cfg.r_sense_ohm))
    add('Vgnd_ref', 'V', '0')
    add('Xina', 'INA')
    for i in range(n_bpf):
        add(f'Xbpf{i + 1}', 'BPF_STAGE')
    if has_comp:
        add('Xcomp', 'COMPARATOR')

    # Solar cell: anode(1), cathode(2), photo_in(3)
    g.connect('sc_anode', 'Xsc', 1)
    g.connect('sc_cathode', 'Xsc', 2)
    g.connect('optical_power', 'Xsc', 3)
    # R_sense: sc_cathode(1) -> sense_lo(2)
    g.connect('sc_cathode', 'Rsense', 1)
    g.connect('sense_lo', 'Rsense', 2)
    # Ground reference source: sense_lo(1) -> 0(2)
    g.connect('sense_lo', 'Vgnd_ref', 1)
    g.connect('0', 'Vgnd_ref', 2)
    # INA: INP(1), INN(2), OUT(3), VCC(4), VEE(5), REF(6)
    g.connect('sense_lo', 'Xina', 1)
    g.connect('sc_cathode', 'Xina', 2)
    g.connect('ina_out', 'Xina', 3)
    g.connect('vcc', 'Xina', 4)
    g.connect('vee', 'Xina', 5)
    g.connect('vref', 'Xina', 6)
    # BPF chain: inp(1), out(2), vcc(3), vee(4), vref(5)
    prev = 'ina_out'
    for i in range(n_bpf):
        out_node = f'bpf{i + 1}_out' if i < n_bpf - 1 else 'bpf_out'
        ref = f'Xbpf{i + 1}'
        g.connect(prev, ref, 1)
        g.connect(out_node, ref, 2)
        g.connect('vcc', ref, 3)
        g.connect('vee', ref, 4)
        g.connect('vref', ref, 5)
        prev = out_node
    # Comparator: INP(1), INN(2), OUT(3), VCC(4), VEE(5)
    if has_comp:
        comp_in = 'bpf_out' if n_bpf > 0 else 'ina_out'
        g.connect(comp_in, 'Xcomp', 1)
        g.connect('vref', 'Xcomp', 2)
        g.connect('dout', 'Xcomp', 3)
        g.connect('vcc', 'Xcomp', 4)
        g.connect('vee', 'Xcomp', 5)

    return g
