"""Structured circuit topology for KiCad schematic export.

A CircuitGraph captures component instances, their pins, and the nets that
wire them together. It is paper-agnostic — per-paper builders in
``systems/<paper>_graph.py`` populate it from ``SystemConfig`` + topology
knowledge, and the KiCad writers consume it.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Pin:
    component_ref: str
    pin_number: int

    def __str__(self) -> str:
        return f"{self.component_ref}.{self.pin_number}"


@dataclass
class ComponentInstance:
    """A placed component in the schematic.

    Attributes:
        ref: Schematic reference designator (e.g. "U1", "R5", "C3", "D1").
        component_type: Registry key or generic type (e.g. "INA322", "R", "C").
        value: Human-readable value ("191k", "2.5pF", "INA322IDGKR").
        footprint: KiCad footprint LIB_ID (e.g. "Package_SO:SOIC-8_3.9x4.9mm_P1.27mm").
        symbol_lib_id: KiCad symbol LIB_ID (e.g. "Amplifier_Operational:INA322").
        pins: Mapping pin number -> pin name/function.
        part_number: Manufacturer part number (for BOM).
        manufacturer: Manufacturer name (for BOM).
        datasheet_url: URL to datasheet (optional).
    """
    ref: str
    component_type: str
    value: str
    footprint: str
    symbol_lib_id: str
    pins: dict[int, str]
    part_number: str = ""
    manufacturer: str = ""
    datasheet_url: str = ""


@dataclass
class Net:
    """A named electrical net connecting one or more pins."""
    name: str
    pins: list[Pin] = field(default_factory=list)

    def add(self, component_ref: str, pin_number: int) -> None:
        self.pins.append(Pin(component_ref, pin_number))


@dataclass
class CircuitGraph:
    """Full circuit: components + nets. Single source of truth for export."""
    title: str
    components: list[ComponentInstance] = field(default_factory=list)
    nets: list[Net] = field(default_factory=list)

    def add_component(self, component: ComponentInstance) -> ComponentInstance:
        if any(c.ref == component.ref for c in self.components):
            raise ValueError(f"Duplicate ref: {component.ref}")
        self.components.append(component)
        return component

    def add_net(self, name: str) -> Net:
        net = Net(name=name)
        self.nets.append(net)
        return net

    def get_net(self, name: str) -> Net:
        for net in self.nets:
            if net.name == name:
                return net
        raise KeyError(f"Net not found: {name}")

    def connect(self, net_name: str, component_ref: str, pin_number: int) -> None:
        """Add a (component, pin) to the named net, creating it if needed."""
        try:
            net = self.get_net(net_name)
        except KeyError:
            net = self.add_net(net_name)
        net.add(component_ref, pin_number)
