"""
Binary WebSocket frame for probe data.

Wire format
-----------
    [ 4 bytes  big-endian uint32 = header length in bytes ]
    [ N bytes  UTF-8 JSON header  ]
    [ M bytes  raw little-endian sample data ]

The header carries everything the frontend needs to decode the payload as a
TypedArray and route it to the right ProbeWindow:

    {
      "type": "probe_data",
      "id": "rx.V_ina",
      "dtype": "float64" | "uint8",
      "shape": [N] | [rows, cols],
      "units": "V",
      "config_hash": "abc1234567890def",
      "request_id": 17,             # echoes the client's probe_fetch.request_id
      "stage": "RX",
      "suggest_decimate": false     # set true when N > _DECIMATE_THRESHOLD
    }

Notes
-----
- Headers stay JSON for human-readable debugging in DevTools.
- Sample bytes are NEVER copied or re-encoded — `tobytes()` on a contiguous
  ndarray is zero-copy on the Python side and decodes directly into a
  TypedArray on the JS side.
- All multi-byte sample values are little-endian (matches `Float64Array` and
  `Uint8Array` defaults in browsers on every supported platform).
"""

from __future__ import annotations

import json
import struct
from typing import Any, Dict

import numpy as np

# Probe arrays larger than this hint the frontend to apply view-only decimation.
# The raw bytes are still sent in full — the frontend keeps the full-fidelity
# TypedArray for hover/export and only downsamples for the scope pixel grid.
_DECIMATE_THRESHOLD = 500_000

_DTYPE_NUMPY_TO_WIRE = {
    np.dtype("float64"): "float64",
    np.dtype("float32"): "float32",
    np.dtype("uint8"): "uint8",
    np.dtype("int32"): "int32",
}


def _wire_dtype(arr: np.ndarray) -> str:
    """Map a numpy dtype to the wire-format dtype string."""
    dt = arr.dtype
    if dt in _DTYPE_NUMPY_TO_WIRE:
        return _DTYPE_NUMPY_TO_WIRE[dt]
    # Anything else → upcast to float64 (lossless for everything we capture).
    return "float64"


def pack_probe(
    probe_id: str,
    arr: np.ndarray,
    *,
    units: str,
    config_hash: str,
    request_id: int = 0,
    stage: str = "",
    extra: Dict[str, Any] | None = None,
) -> bytes:
    """
    Pack a probe array into one binary WebSocket frame.

    Args:
        probe_id: Stable probe ID, e.g. "rx.V_ina".
        arr: Numpy array of sample values.
        units: SI units string, e.g. "V".
        config_hash: Tag the payload to a specific SystemConfig so the frontend
            can mark stale probes after a config edit.
        request_id: Echoes the request_id the client included in `probe_fetch`.
        stage: Pipeline stage owning the probe ("TX" | "Channel" | "RX").
        extra: Optional dict merged into the header (forward-compat hook).

    Returns:
        The full binary frame as bytes, ready for `ws.send_bytes(...)`.
    """
    a = np.ascontiguousarray(arr)
    wire_dtype = _wire_dtype(a)
    # Cast if we promoted (only happens for unsupported dtypes — see above).
    if wire_dtype == "float64" and a.dtype != np.float64:
        a = a.astype(np.float64, copy=False)

    # Sample data is always little-endian on the wire to match TypedArray defaults.
    if a.dtype.byteorder == ">":
        a = a.astype(a.dtype.newbyteorder("<"), copy=False)

    header: Dict[str, Any] = {
        "type": "probe_data",
        "id": probe_id,
        "dtype": wire_dtype,
        "shape": list(a.shape),
        "units": units,
        "config_hash": config_hash,
        "request_id": int(request_id),
        "stage": stage,
        "suggest_decimate": a.size > _DECIMATE_THRESHOLD,
    }
    if extra:
        header.update(extra)

    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    header_len = struct.pack(">I", len(header_bytes))

    return header_len + header_bytes + a.tobytes(order="C")


def unpack_probe(frame: bytes) -> Dict[str, Any]:
    """
    Decode a probe frame produced by `pack_probe`.

    Returns:
        Dict with keys: header (parsed JSON), payload (numpy ndarray reshaped
        to header['shape'] with the header's dtype).
    """
    if len(frame) < 4:
        raise ValueError("probe frame too short")
    (header_len,) = struct.unpack(">I", frame[:4])
    if header_len <= 0 or 4 + header_len > len(frame):
        raise ValueError(f"invalid header length {header_len}")

    header = json.loads(frame[4 : 4 + header_len].decode("utf-8"))
    raw = frame[4 + header_len:]

    dtype_str = header.get("dtype", "float64")
    np_dtype = {
        "float64": np.float64,
        "float32": np.float32,
        "uint8": np.uint8,
        "int32": np.int32,
    }.get(dtype_str, np.float64)

    arr = np.frombuffer(raw, dtype=np_dtype)
    shape = tuple(header.get("shape") or [arr.size])
    if shape and int(np.prod(shape)) == arr.size:
        arr = arr.reshape(shape)

    return {"header": header, "payload": arr}
