"""
Text <-> bits codec with a simple frame for TX/RX demos.

Frame layout (MSB-first for every byte):
    [ PREAMBLE (4 B) | LENGTH (2 B) | PAYLOAD (N B) ]

PREAMBLE  = 0xAA 0x55 0xA5 0x5A   (alternating bit pattern for sync hunt)
LENGTH    = big-endian UTF-8 byte count of the payload (0..65535)
PAYLOAD   = the UTF-8 encoded message

The preamble gives the demodulator something to hunt for even if the first
bits are garbage, and the explicit length prevents trailing-junk from being
decoded past the real message. Bytes that fail UTF-8 validation are rendered
as U+FFFD so the receiver view can still show the partially-corrupted text.
"""

from __future__ import annotations

import numpy as np

PREAMBLE_BYTES = bytes([0xAA, 0x55, 0xA5, 0x5A])
PREAMBLE_BITS = 8 * len(PREAMBLE_BYTES)
LENGTH_BYTES = 2
LENGTH_BITS = 8 * LENGTH_BYTES
HEADER_BITS = PREAMBLE_BITS + LENGTH_BITS
MAX_PAYLOAD_BYTES = 0xFFFF


def _bytes_to_bits(data: bytes) -> np.ndarray:
    """Unpack a bytes object to an MSB-first bit array."""
    if not data:
        return np.zeros(0, dtype=np.uint8)
    return np.unpackbits(np.frombuffer(data, dtype=np.uint8))


def _bits_to_bytes(bits: np.ndarray) -> bytes:
    """Pack MSB-first bits back into bytes (truncates to whole bytes)."""
    bits = np.asarray(bits, dtype=np.uint8).ravel()
    n_full = (len(bits) // 8) * 8
    if n_full == 0:
        return b''
    return np.packbits(bits[:n_full]).tobytes()


def encode_text(message: str) -> np.ndarray:
    """Encode a string to a bit array wrapped in the sync frame.

    Raises ValueError if the UTF-8 payload exceeds MAX_PAYLOAD_BYTES.
    """
    payload = message.encode('utf-8')
    if len(payload) > MAX_PAYLOAD_BYTES:
        raise ValueError(
            f"Message too long: {len(payload)} bytes "
            f"(max {MAX_PAYLOAD_BYTES})")
    header = PREAMBLE_BYTES + len(payload).to_bytes(LENGTH_BYTES, 'big')
    return _bytes_to_bits(header + payload)


def _find_preamble(bits: np.ndarray) -> int:
    """Return the bit index where PREAMBLE ends, or -1 if not found.

    Allows up to 1 bit error inside the 32-bit preamble so a single flipped
    sync bit does not destroy the whole message.
    """
    preamble_bits = _bytes_to_bits(PREAMBLE_BYTES).astype(np.int8)
    if len(bits) < PREAMBLE_BITS:
        return -1

    bits_i8 = np.asarray(bits, dtype=np.int8)
    best_idx, best_err = -1, PREAMBLE_BITS
    for i in range(len(bits_i8) - PREAMBLE_BITS + 1):
        err = int(np.sum(bits_i8[i:i + PREAMBLE_BITS] != preamble_bits))
        if err < best_err:
            best_err = err
            best_idx = i
            if err == 0:
                break
    if best_idx < 0 or best_err > 1:
        return -1
    return best_idx + PREAMBLE_BITS


def decode_text(bits: np.ndarray) -> tuple[str, dict]:
    """Decode a received bit array back to text.

    Returns (text, info) where info reports how decoding went:
        - 'sync_found':       True if the preamble was located
        - 'declared_length':  bytes claimed by the LENGTH field (or None)
        - 'payload_bytes':    bytes actually available for the payload
        - 'truncated':        True if the stream ended mid-payload
        - 'utf8_errors':      count of invalid UTF-8 bytes (replaced by U+FFFD)
    """
    info = {
        'sync_found': False,
        'declared_length': None,
        'payload_bytes': 0,
        'truncated': False,
        'utf8_errors': 0,
    }
    bits = np.asarray(bits, dtype=np.uint8).ravel()

    payload_start = _find_preamble(bits)
    if payload_start < 0:
        return '', info
    info['sync_found'] = True

    len_end = payload_start + LENGTH_BITS
    if len_end > len(bits):
        return '', info
    length_bytes = _bits_to_bytes(bits[payload_start:len_end])
    declared = int.from_bytes(length_bytes, 'big')
    info['declared_length'] = declared

    payload_bits = bits[len_end:len_end + declared * 8]
    payload = _bits_to_bytes(payload_bits)
    info['payload_bytes'] = len(payload)
    info['truncated'] = len(payload) < declared

    # Replace invalid UTF-8 sequences with U+FFFD and count them.
    text = payload.decode('utf-8', errors='replace')
    info['utf8_errors'] = text.count('�')
    return text, info


def char_diff(tx: str, rx: str) -> list[tuple[str, str]]:
    """Align tx and rx character-by-character for a simple diff view.

    Returns a list of (char, status) tuples where status is:
        'ok'      rx char matches tx
        'bad'     rx char differs from tx
        'missing' rx is shorter than tx (tx char with no rx counterpart)
        'extra'   rx is longer than tx (rx char past the end of tx)
    """
    out: list[tuple[str, str]] = []
    n = max(len(tx), len(rx))
    for i in range(n):
        if i >= len(rx):
            out.append((tx[i], 'missing'))
        elif i >= len(tx):
            out.append((rx[i], 'extra'))
        elif tx[i] == rx[i]:
            out.append((rx[i], 'ok'))
        else:
            out.append((rx[i], 'bad'))
    return out
