"""
backend/spice_waveforms.py — build a python_result-shaped waveforms dict
from a SPICE run.

The SPICE engine writes a `.raw` file but the Python-engine WebSocket payload
shape (time, P_tx, P_rx, I_ph, V_rx, bits_tx, bits_rx, modulation) is what
the frontend's result panels expect. This helper bridges the two so the
"Show Full Results" view is populated regardless of which engine ran.

Public entry:
    build_waveforms_from_spice(raw_path, bits_tx, bits_rx, cfg) -> dict | None

Returns None if the raw file can't be parsed, otherwise a dict shaped like
cosim.python_engine.run_python_simulation()'s return value (with arrays
that may be empty when the SPICE netlist didn't probe that node).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from cosim.spice_extract import extract_spice_waveforms

logger = logging.getLogger(__name__)


def build_waveforms_from_spice(
    raw_path: str | Path,
    bits_tx: np.ndarray | list | None,
    bits_rx: np.ndarray | list | None,
    cfg: Any,
) -> dict[str, Any] | None:
    """Assemble a python_result-shaped dict from a SPICE .raw file.

    Args:
        raw_path: Path to the .raw file produced by LTspice / ngspice.
        bits_tx: Transmitted bit sequence (from the pipeline). May be None.
        bits_rx: Received bits after threshold decision (from
            compute_ber_from_spice). May be None.
        cfg: The SystemConfig used for the run — only the `modulation` field
            is read.

    Returns:
        dict shaped like the Python engine result, or None on parse failure.
        Arrays that aren't available in the .raw file are returned as [].
    """
    try:
        wf = extract_spice_waveforms(raw_path)
    except Exception as e:  # noqa: BLE001 — parse failures must not crash the WS
        logger.warning("SPICE waveform extraction failed for %s: %s", raw_path, e)
        return None

    time = _ensure_array(wf.get("time"))
    # V_rx: prefer the post-amplifier / pre-comparator analog node so eye /
    # spectrum / decision panels see something meaningful. Fall back to the
    # comparator output if those aren't probed.
    v_rx = (
        _ensure_array(wf.get("V_bpf2"))
        if _has(wf, "V_bpf2")
        else _ensure_array(wf.get("V_ina"))
        if _has(wf, "V_ina")
        else _ensure_array(wf.get("V_comp"))
    )
    p_rx = _ensure_array(wf.get("P_rx"))
    i_ph = _ensure_array(wf.get("I_sense"))  # I through R_sense ≈ photocurrent

    return {
        "time": time,
        "P_tx": np.array([]),  # SPICE doesn't probe TX power directly
        "P_rx": p_rx,
        "I_ph": i_ph,
        "V_rx": v_rx,
        "bits_tx": _ensure_array(bits_tx),
        "bits_rx": _ensure_array(bits_rx),
        "modulation": getattr(cfg, "modulation", None),
    }


def _has(d: dict, key: str) -> bool:
    arr = d.get(key)
    return isinstance(arr, np.ndarray) and arr.size > 0


def _ensure_array(v: Any) -> np.ndarray:
    if v is None:
        return np.array([])
    if isinstance(v, np.ndarray):
        return v
    try:
        return np.asarray(v)
    except Exception:  # noqa: BLE001
        return np.array([])
