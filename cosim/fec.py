"""
Forward Error Correction (FEC) layer.

Generic FEC interface that any preset can opt into via SimConfig.fec_enable.
Initial implementation is LDPC via the pyldpc library; the LDPCCodec class
exposes a minimal encode/decode_bits API that the simulation pipeline calls
around the existing modulate/demodulate chain.

Standards alignment caveat:
    802.11n/ax/bb define specific quasi-cyclic LDPC base matrices (n=648,
    1296, 1944 with rates 1/2, 2/3, 3/4, 5/6). pyldpc generates regular
    Gallager-construction LDPC matrices of the same n and rate, NOT the
    standard's QC-LDPC base matrices. This is a thesis-acceptable
    simplification — code rate is exact, decoder is standards-class
    belief-propagation, but the parity structure differs.

Soft-information caveat:
    The simulator's existing OFDM demodulator returns hard-decision bits.
    decode_bits() therefore feeds the BP decoder fake LLRs of magnitude
    derived from a fixed-SNR assumption (BSC-equivalent input). This gives
    correct error-correction behavior up to roughly half the code's
    minimum distance per codeword, which is sufficient to demonstrate
    FEC threshold crossing. True per-bit soft demapping (Euclidean
    distance per bit position in the QAM constellation) is a follow-up.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


# Lazy import — pyldpc pulls in numba, which is slow to import. Defer until
# a codec is actually constructed.
def _import_pyldpc():
    from pyldpc import make_ldpc, decode, get_message
    from pyldpc.utils import binaryproduct
    return make_ldpc, decode, get_message, binaryproduct


@dataclass
class FECParams:
    """LDPC FEC parameters. Mirrored from SimConfig fec_* fields."""
    enabled: bool = False
    rate_num: int = 5            # numerator of code rate
    rate_den: int = 6            # denominator of code rate
    codeword_n: int = 648        # codeword length (802.11 standard sizes: 648, 1296, 1944)
    d_v: int = 3                 # variable-node degree (regular LDPC)
    max_iter: int = 50           # BP iterations
    decode_snr_db: float = 5.0   # SNR assumption for BP LLR initialization
    seed: int = 42

    @property
    def rate(self) -> float:
        return self.rate_num / self.rate_den

    @property
    def d_c(self) -> int:
        # For regular LDPC: rate = 1 - d_v / d_c  =>  d_c = d_v / (1 - rate)
        return int(round(self.d_v / (1.0 - self.rate)))


class LDPCCodec:
    """Belief-propagation LDPC encoder/decoder wrapping pyldpc.

    Once constructed, exposes:
        codec.message_length   -> k, message bits per codeword
        codec.codeword_length  -> n, transmitted bits per codeword
        codec.encode(bits)     -> coded bits (length rounded up to k boundary)
        codec.decode_bits(bits, snr_db) -> decoded message bits
    """

    def __init__(self, params: FECParams):
        self.params = params
        if not params.enabled:
            self.H = self.G = None
            self._n = self._k = 0
            self._binaryproduct = None
            self._decode = None
            self._get_message = None
            return

        make_ldpc, decode, get_message, binaryproduct = _import_pyldpc()
        self._decode = decode
        self._get_message = get_message
        self._binaryproduct = binaryproduct

        self.H, self.G = make_ldpc(
            params.codeword_n,
            params.d_v,
            params.d_c,
            systematic=True,
            sparse=True,
            seed=params.seed,
        )
        # pyldpc returns G as tG with shape (n, k).
        self._n, self._k = self.G.shape

    @property
    def message_length(self) -> int:
        return self._k

    @property
    def codeword_length(self) -> int:
        return self._n

    @property
    def actual_rate(self) -> float:
        if self._n == 0:
            return 1.0
        return self._k / self._n

    def encode(self, bits: np.ndarray) -> np.ndarray:
        """Encode an arbitrary-length bit stream into LDPC codewords.

        Pads the message with zeros up to the next k-boundary before encoding.
        Returns the concatenated codewords (length = ceil(len(bits)/k) * n).
        """
        if not self.params.enabled:
            return bits
        bits = np.asarray(bits, dtype=int).ravel()
        pad = (-len(bits)) % self._k
        if pad:
            bits = np.concatenate([bits, np.zeros(pad, dtype=int)])
        n_blocks = len(bits) // self._k
        out = np.empty(n_blocks * self._n, dtype=int)
        for i in range(n_blocks):
            v = bits[i * self._k:(i + 1) * self._k]
            c = self._binaryproduct(self.G, v)
            out[i * self._n:(i + 1) * self._n] = c
        return out

    def decode_bits(self, rx_bits: np.ndarray,
                    snr_db: Optional[float] = None) -> np.ndarray:
        """Decode received hard bits back into message bits via BP.

        Args:
            rx_bits: received hard-decision bits, length should be multiple
                of n (extra bits are truncated).
            snr_db: SNR (dB) used to initialize BP LLRs. Higher values =
                more trust in received bits. Defaults to params.decode_snr_db.
        """
        if not self.params.enabled:
            return rx_bits
        rx_bits = np.asarray(rx_bits, dtype=int).ravel()
        if snr_db is None:
            snr_db = self.params.decode_snr_db

        n_blocks = len(rx_bits) // self._n
        out = np.empty(n_blocks * self._k, dtype=int)
        for i in range(n_blocks):
            c_hat = rx_bits[i * self._n:(i + 1) * self._n]
            # BPSK soft mapping: bit 0 -> +1, bit 1 -> -1.
            y = (1.0 - 2.0 * c_hat).astype(float)
            d = self._decode(self.H, y, snr=snr_db,
                             maxiter=self.params.max_iter)
            v_hat = self._get_message(self.G, d).astype(int)
            out[i * self._k:(i + 1) * self._k] = v_hat
        return out

    def decode_llrs(self, llrs: np.ndarray) -> np.ndarray:
        """Decode received per-bit LLRs back into message bits via BP.

        Soft-input counterpart to ``decode_bits``. Pass true LLRs (positive
        => bit 0 more likely, sign convention matching pyldpc internals)
        produced by e.g. ``cosim.modulation._qam_demap_soft``.

        Implementation: pyldpc's ``decode(H, y, snr, maxiter)`` computes
        Lc = 2*y / var with var = 10**(-snr/10). To pass true LLRs unchanged
        we use snr_db = 0 (var = 1) and y = llrs / 2 so that Lc = llrs.

        Args:
            llrs: per-bit log-likelihood ratios, length should be a multiple
                of n (extra entries are truncated).
        """
        if not self.params.enabled:
            # FEC disabled: return hard decisions from the LLR signs so the
            # caller gets back a sensible bit stream.
            return (np.asarray(llrs).ravel() < 0).astype(int)

        llrs = np.asarray(llrs, dtype=float).ravel()
        n_blocks = len(llrs) // self._n
        out = np.empty(n_blocks * self._k, dtype=int)
        for i in range(n_blocks):
            block_llrs = llrs[i * self._n:(i + 1) * self._n]
            y = block_llrs / 2.0
            d = self._decode(self.H, y, snr=0.0,
                             maxiter=self.params.max_iter)
            v_hat = self._get_message(self.G, d).astype(int)
            out[i * self._k:(i + 1) * self._k] = v_hat
        return out


def build_fec_codec(config) -> Optional[LDPCCodec]:
    """Construct an LDPCCodec from a SystemConfig.

    Returns None if FEC is disabled (sentinel that callers test with `is None`).
    """
    if not getattr(config, 'fec_enable', False):
        return None
    params = FECParams(
        enabled=True,
        rate_num=getattr(config, 'fec_rate_num', 5),
        rate_den=getattr(config, 'fec_rate_den', 6),
        codeword_n=getattr(config, 'fec_codeword_n', 648),
        d_v=getattr(config, 'fec_d_v', 3),
        max_iter=getattr(config, 'fec_max_iter', 50),
        decode_snr_db=getattr(config, 'fec_decode_snr_db', 5.0),
        seed=getattr(config, 'fec_seed', 42),
    )
    return LDPCCodec(params)
