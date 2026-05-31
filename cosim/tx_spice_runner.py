"""In-process libngspice runner for a user-drawn TX analog chain.

The transmitter half of the single-canvas co-simulation. Given the SPICE
pieces of a drawn TX subgraph — component .SUBCKT bodies + the graph's element
lines — plus a modulated gate-drive waveform, this assembles a TX deck, runs
one transient in libngspice, and returns the LED current ``I(LED, t)``.

That current is the hand-off to the Python comms layer (see
``cosim.optical_bridge``): it maps to optical watts and through
``cosim.channel`` to the received optical power that drives the RX deck.

Why libngspice (not LTspice): the RX runner already proved LTspice batch mode
nondeterministically hangs on these behavioral subcircuits; libngspice runs
in-process, is killable, and is what the .exe bundles.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_MAX_PWL_POINTS = 8000


def available() -> bool:
    """True if libngspice is importable and instantiable."""
    try:
        from PySpice.Spice.NgSpice.Shared import NgSpiceShared
        NgSpiceShared.new_instance().ngspice_version  # noqa: B018
        return True
    except Exception as e:  # noqa: BLE001
        logger.info("libngspice unavailable: %s", e)
        return False


def ook_gate_drive(bits: np.ndarray, bit_period: float, v_on: float,
                   v_off: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """OOK gate-drive waveform: ``v_on`` for bit=1, ``v_off`` for bit=0, as
    near-square steps (1%-of-bit transitions). Returns (t, v) arrays suitable
    for a PWL source on the MOSFET gate net."""
    eps = bit_period * 0.01
    t, v = [], []
    for i, b in enumerate(bits):
        level = v_on if b else v_off
        t.append(i * bit_period); v.append(level)
        t.append((i + 1) * bit_period - eps); v.append(level)
    return np.asarray(t, dtype=float), np.asarray(v, dtype=float)


def _pwl_string(t: np.ndarray, v: np.ndarray) -> str:
    """Format (t, v) as an ngspice PWL list, decimated to keep the deck small."""
    t = np.asarray(t, dtype=float)
    v = np.asarray(v, dtype=float)
    if len(t) > _MAX_PWL_POINTS:
        idx = np.linspace(0, len(t) - 1, _MAX_PWL_POINTS).astype(int)
        t, v = t[idx], v[idx]
    keep = np.concatenate(([True], np.diff(t) > 0))  # strictly increasing for ngspice
    t, v = t[keep], v[keep]
    return ' '.join(f'{ti:.9e} {vi:.6e}' for ti, vi in zip(t, v))


def _insert_led_ammeter(instance_lines: str, led_ref: str) -> str:
    """Rewrite the LED instance line to put a 0 V ammeter in series with its
    anode, so its branch current is recoverable as ``i(vsense_led)``.

    ``Xled anode cathode TYPE``  ->
        ``Vsense_led anode anode__s DC 0``
        ``Xled anode__s cathode TYPE``
    """
    out = []
    found = False
    for line in instance_lines.splitlines():
        toks = line.split()
        if toks and toks[0] == led_ref and len(toks) >= 4:
            anode = toks[1]
            sense = f"{anode}__s"
            out.append(f"Vsense_led {anode} {sense} DC 0")
            out.append(f"{toks[0]} {sense} {' '.join(toks[2:])}")
            found = True
        else:
            out.append(line)
    if not found:
        raise ValueError(f"LED ref {led_ref!r} not found in TX instance lines")
    return "\n".join(out)


def run_tx_graph(
    subckt_defs: str,
    instance_lines: str,
    *,
    drive_net: str,
    drive_t: np.ndarray,
    drive_v: np.ndarray,
    led_ref: str,
    vcc_volts: float,
    t_stop_s: float,
    t_step_s: Optional[float] = None,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Simulate a drawn TX chain and return ``(t, i_led)``, or None on failure.

    Args:
        subckt_defs: concatenated .SUBCKT bodies for placed library parts.
        instance_lines: the TX subgraph's element cards (Xled, Xmos, R, ...).
        drive_net: net name the modulated gate drive is injected on.
        drive_t, drive_v: the gate-drive PWL (e.g. from ``ook_gate_drive``).
        led_ref: ref of the LED to probe (its branch current is the optical src).
        vcc_volts: TX supply rail voltage (net ``vcc``; ground is ``0``).
        t_stop_s, t_step_s: transient window.
    """
    if not available():
        return None

    from PySpice.Spice.Netlist import Circuit

    t_step = float(t_step_s if t_step_s is not None else t_stop_s / 1000.0)
    circuit = Circuit("OptiSim_tx")
    if subckt_defs.strip():
        circuit.raw_spice += subckt_defs + "\n"

    # Implicit supply rail (drawn supply pins resolve to net 'vcc' by role).
    circuit.raw_spice += f"Vcc vcc 0 DC {vcc_volts}\n"
    # Modulated gate drive (Python owns bit-gen + modulation; SPICE plays it).
    circuit.raw_spice += f"Vdrive {drive_net} 0 PWL({_pwl_string(drive_t, drive_v)})\n"
    # The user's TX circuit, with an ammeter spliced in series with the LED.
    circuit.raw_spice += _insert_led_ammeter(instance_lines, led_ref) + "\n"

    try:
        sim = circuit.simulator(temperature=25, nominal_temperature=25,
                                simulator="ngspice-shared")
        sim.options("reltol=0.003", "abstol=1e-12", "vntol=1e-6",
                    "gmin=1e-9", "method=gear")
        analysis = sim.transient(step_time=t_step, end_time=t_stop_s)
    except Exception as e:  # noqa: BLE001
        logger.warning("libngspice TX transient failed: %s", e)
        return None

    try:
        i_led = np.array(analysis.branches["vsense_led"])
        t = np.array(analysis.time)
    except (KeyError, IndexError):
        logger.warning("I(LED) not found in TX result")
        return None
    return t, i_led
