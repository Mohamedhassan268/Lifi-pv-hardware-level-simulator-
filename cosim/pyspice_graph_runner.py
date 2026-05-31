"""In-process libngspice runner for a user-drawn schematic graph.

The LTspice subprocess path nondeterministically hangs on the behavioral
component subcircuits (batch-mode solver flakiness, uncontrollable from
Python). libngspice runs in-process via PySpice, so a run is fully
controllable and never spawns a detached worker. This runner takes the same
SPICE pieces the LTspice path uses — component .SUBCKT bodies + the graph's
element lines + an optical PWL bridge from the Python channel — assembles them
as a PySpice Circuit.raw_spice, runs one transient, and returns V(dout).

Used by the /api/schematic-sim endpoint.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

_MAX_PWL_POINTS = 8000


def available() -> bool:
    try:
        from PySpice.Spice.NgSpice.Shared import NgSpiceShared
        NgSpiceShared.new_instance().ngspice_version  # noqa: B018
        return True
    except Exception as e:  # noqa: BLE001
        logger.info("libngspice unavailable: %s", e)
        return False


def _pwl_string(t: np.ndarray, v: np.ndarray) -> str:
    """Format (t, v) arrays as an ngspice PWL argument list, decimated to a
    sane point count so the netlist string stays small."""
    t = np.asarray(t, dtype=float)
    v = np.asarray(v, dtype=float)
    if len(t) > _MAX_PWL_POINTS:
        idx = np.linspace(0, len(t) - 1, _MAX_PWL_POINTS).astype(int)
        t, v = t[idx], v[idx]
    # ngspice rejects non-increasing PWL time points; the source time vector can
    # carry duplicate breakpoint times. Keep only strictly increasing samples.
    keep = np.concatenate(([True], np.diff(t) > 0))
    t, v = t[keep], v[keep]
    return ' '.join(f'{ti:.6e} {vi:.6e}' for ti, vi in zip(t, v))


def run_graph(
    subckt_defs: str,
    instance_lines: str,
    optical_t: np.ndarray,
    optical_v: np.ndarray,
    vcc_volts: float,
    t_stop_s: float,
    t_step_s: Optional[float] = None,
    probe_nets: Optional[List[str]] = None,
) -> Optional[tuple]:
    """Simulate a user circuit and return ``(V(dout), t, extras)``, or None.

    Args:
        subckt_defs: concatenated .SUBCKT bodies for placed library parts.
        instance_lines: the graph's element cards (Xref n.. TYPE, R, V).
        optical_t, optical_v: optical-power PWL driving the `optical_power` node.
        vcc_volts: supply rail voltage (vcc; vee=0; vref=vcc/2).
        t_stop_s, t_step_s: transient window.
        probe_nets: extra net names to return in ``extras`` (e.g. the comparator
            input / threshold for post-processing). Missing nets are skipped.

    Returns:
        ``(v_dout, time, extras)`` where ``extras`` maps each found probe net to
        its waveform array, or ``None`` if V(dout) can't be produced.
    """
    if not available():
        return None

    from PySpice.Spice.Netlist import Circuit

    t_step = float(t_step_s if t_step_s is not None else t_stop_s / 1000.0)

    circuit = Circuit('OptiSim_schematic')
    if subckt_defs.strip():
        circuit.raw_spice += subckt_defs + '\n'

    # Power rails + reference (graph supply pins resolve to these net names).
    circuit.raw_spice += f'Vcc vcc 0 DC {vcc_volts}\n'
    circuit.raw_spice += 'Vee vee 0 DC 0\n'
    circuit.raw_spice += f'Vref vref 0 DC {vcc_volts / 2.0}\n'

    # Optical input bridge from the Python channel.
    pwl = _pwl_string(optical_t, optical_v)
    circuit.raw_spice += f'Voptical optical_power 0 DC 0 PWL({pwl})\n'

    # The user's circuit.
    circuit.raw_spice += instance_lines + '\n'

    try:
        sim = circuit.simulator(
            temperature=25, nominal_temperature=25,
            simulator='ngspice-shared',
        )
        sim.options('reltol=0.003', 'abstol=1e-12', 'vntol=1e-6',
                    'gmin=1e-9', 'method=gear')
        analysis = sim.transient(step_time=t_step, end_time=t_stop_s)
    except Exception as e:  # noqa: BLE001
        logger.warning("libngspice transient failed: %s", e)
        return None

    t = np.array(analysis.time)
    try:
        v_dout = np.array(analysis['dout'])
    except (KeyError, IndexError):
        # Analog RX front-ends (OFDM/BFSK) have no comparator / dout net; the
        # caller reads the amplifier output from `extras` instead.
        v_dout = None

    extras: Dict[str, np.ndarray] = {}
    for net in (probe_nets or []):
        try:
            extras[net] = np.array(analysis[net])
        except (KeyError, IndexError):
            logger.info("probe net %r not found in result", net)

    if v_dout is None and not extras:
        logger.warning("neither V(dout) nor any probe net found in result")
        return None

    # Debug aid: stash intermediate node swings for diagnosis.
    import os as _os
    if _os.environ.get('OPTISIM_SCH_DEBUG') == '1':
        for n in ('ina_out', 'sense_lo', 'sc_cathode'):
            try:
                v = np.array(analysis[n])
                logger.warning("DBG V(%s): [%.4f, %.4f] mean=%.4f", n, v.min(), v.max(), v.mean())
            except Exception:  # noqa: BLE001
                pass
    return v_dout, t, extras
