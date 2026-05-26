"""Round-trip + edge-case tests for backend.transport.binary_frame."""

from __future__ import annotations

import json
import struct

import numpy as np
import pytest

from backend.transport.binary_frame import pack_probe, unpack_probe


def test_roundtrip_float64_preserves_bytes():
    arr = np.linspace(0.0, 1.0, 100_000, dtype=np.float64)
    frame = pack_probe(
        "rx.V_ina", arr,
        units="V", config_hash="abc1234567890def", request_id=17, stage="RX",
    )
    out = unpack_probe(frame)
    assert out["header"]["id"] == "rx.V_ina"
    assert out["header"]["dtype"] == "float64"
    assert out["header"]["shape"] == [100_000]
    assert out["header"]["units"] == "V"
    assert out["header"]["request_id"] == 17
    assert out["header"]["config_hash"] == "abc1234567890def"
    assert out["payload"].shape == (100_000,)
    assert out["payload"].dtype == np.float64
    assert np.array_equal(arr, out["payload"])


def test_roundtrip_uint8_bit_stream():
    bits = np.random.randint(0, 2, 1024).astype(np.uint8)
    frame = pack_probe("rx.bits_rx", bits, units="1", config_hash="h", stage="RX")
    out = unpack_probe(frame)
    assert out["header"]["dtype"] == "uint8"
    assert out["payload"].dtype == np.uint8
    assert np.array_equal(bits, out["payload"])


def test_suggest_decimate_flag_triggers_above_threshold():
    small = np.zeros(1000, dtype=np.float64)
    big = np.zeros(600_000, dtype=np.float64)
    h_small = unpack_probe(pack_probe("p", small, units="V", config_hash="h"))["header"]
    h_big = unpack_probe(pack_probe("p", big, units="V", config_hash="h"))["header"]
    assert h_small["suggest_decimate"] is False
    assert h_big["suggest_decimate"] is True


def test_unsupported_dtype_promotes_to_float64_losslessly():
    arr = np.arange(64, dtype=np.float16) * 0.5  # not in the wire-type table
    frame = pack_probe("p", arr, units="V", config_hash="h")
    out = unpack_probe(frame)
    assert out["header"]["dtype"] == "float64"
    # All representable values should survive a float16 → float64 round-trip exactly.
    assert np.array_equal(out["payload"], arr.astype(np.float64))


def test_short_frame_raises():
    with pytest.raises(ValueError):
        unpack_probe(b"\x00")


def test_corrupt_header_length_raises():
    # Header claims to be 10 MB long, but the frame is only 20 bytes.
    bogus = struct.pack(">I", 10_000_000) + b"x" * 20
    with pytest.raises(ValueError):
        unpack_probe(bogus)


def test_extra_fields_appear_in_header():
    arr = np.zeros(4)
    frame = pack_probe("p", arr, units="V", config_hash="h", extra={"foo": "bar"})
    out = unpack_probe(frame)
    assert out["header"]["foo"] == "bar"


def test_header_is_strictly_json_after_length_prefix():
    arr = np.zeros(8, dtype=np.float64)
    frame = pack_probe("p", arr, units="V", config_hash="h")
    (header_len,) = struct.unpack(">I", frame[:4])
    header = json.loads(frame[4 : 4 + header_len].decode("utf-8"))
    assert header["type"] == "probe_data"
