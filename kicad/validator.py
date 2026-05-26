"""Sanity checks for a CircuitGraph before writing KiCad output."""

from __future__ import annotations

from .circuit_graph import CircuitGraph


class CircuitGraphValidationError(ValueError):
    """Raised when a CircuitGraph is inconsistent."""


def validate(graph: CircuitGraph) -> list[str]:
    """Validate a circuit graph. Returns a list of non-fatal warnings.

    Raises CircuitGraphValidationError on fatal problems (dangling refs,
    missing metadata, duplicate pin assignments to multiple nets).
    """
    warnings: list[str] = []

    refs = {c.ref for c in graph.components}
    if len(refs) != len(graph.components):
        raise CircuitGraphValidationError("Duplicate component refs detected")

    for comp in graph.components:
        if not comp.symbol_lib_id:
            raise CircuitGraphValidationError(
                f"Component {comp.ref} ({comp.component_type}) has no symbol_lib_id"
            )
        if not comp.footprint and comp.symbol_lib_id not in ("power:VCC", "power:GND"):
            raise CircuitGraphValidationError(
                f"Component {comp.ref} ({comp.component_type}) has no footprint"
            )
        if not comp.pins:
            raise CircuitGraphValidationError(
                f"Component {comp.ref} ({comp.component_type}) has no pins"
            )

    # Every pin on a net must reference a real component, and the pin number
    # must exist on that component.
    pin_to_net: dict[tuple[str, int], str] = {}
    for net in graph.nets:
        for pin in net.pins:
            if pin.component_ref not in refs:
                raise CircuitGraphValidationError(
                    f"Net '{net.name}' references unknown component '{pin.component_ref}'"
                )
            comp = next(c for c in graph.components if c.ref == pin.component_ref)
            if pin.pin_number not in comp.pins:
                raise CircuitGraphValidationError(
                    f"Net '{net.name}' references pin {pin.pin_number} on "
                    f"{pin.component_ref} but component has pins {list(comp.pins.keys())}"
                )
            key = (pin.component_ref, pin.pin_number)
            if key in pin_to_net and pin_to_net[key] != net.name:
                raise CircuitGraphValidationError(
                    f"Pin {pin} is on two nets: '{pin_to_net[key]}' and '{net.name}'"
                )
            pin_to_net[key] = net.name

    # Warn about unconnected pins (non-fatal).
    for comp in graph.components:
        for pin_num in comp.pins:
            if (comp.ref, pin_num) not in pin_to_net:
                warnings.append(
                    f"Pin {comp.ref}.{pin_num} ({comp.pins[pin_num]}) is unconnected"
                )

    return warnings
