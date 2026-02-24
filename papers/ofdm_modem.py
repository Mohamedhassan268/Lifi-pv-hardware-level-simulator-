"""
papers/ofdm_modem.py — Unified OFDM Engine for Paper Validation
================================================================

Self-contained OFDM modem for Sarwar 2017 and Oliveira 2024 validation.
Ported from lifi_pv_simulator/simulator/simulator_ofdm.py.

Provides:
  - OFDMModem: Full TX/RX chain with adaptive or fixed modulation
  - Per-subcarrier analysis (BER, EVM, SNR)
  - Gray-coded M-QAM (BPSK, QPSK, 16-QAM, 64-QAM)
  - ZF equalization
  - solar_panel_channel_response() for Sarwar's RC channel
  - allocate_bits(), ber_mqam(), gross_data_rate(), net_data_rate()
"""

import numpy as np
from scipy.special import erfc


# =============================================================================
# GRAY-CODED M-QAM CONSTELLATION
# =============================================================================

_GRAY_4PAM = {(0, 0): -3, (0, 1): -1, (1, 1): 1, (1, 0): 3}
_GRAY_4PAM_INV = {-3: (0, 0), -1: (0, 1), 1: (1, 1), 3: (1, 0)}

_GRAY_8PAM = {
    (0,0,0): -7, (0,0,1): -5, (0,1,1): -3, (0,1,0): -1,
    (1,1,0): 1,  (1,1,1): 3,  (1,0,1): 5,  (1,0,0): 7,
}
_GRAY_8PAM_INV = {v: k for k, v in _GRAY_8PAM.items()}


def _bits_to_qam(bits, n_bits):
    """Map bits to Gray-coded QAM symbol (normalized to unit average power)."""
    if n_bits == 0 or len(bits) < n_bits:
        return 0j
    if n_bits == 1:
        return (2.0 * bits[0] - 1.0) + 0j
    elif n_bits == 2:
        I = 2.0 * bits[0] - 1.0
        Q = 2.0 * bits[1] - 1.0
        return (I + 1j * Q) / np.sqrt(2)
    elif n_bits == 3:
        I = 2.0 * bits[0] - 1.0
        Q = _GRAY_4PAM.get((int(bits[1]), int(bits[2])), 0)
        return (I + 1j * Q) / np.sqrt(6)
    elif n_bits == 4:
        I = _GRAY_4PAM.get((int(bits[0]), int(bits[1])), 0)
        Q = _GRAY_4PAM.get((int(bits[2]), int(bits[3])), 0)
        return (I + 1j * Q) / np.sqrt(10)
    elif n_bits == 5:
        I = _GRAY_4PAM.get((int(bits[0]), int(bits[1])), 0)
        Q = _GRAY_8PAM.get((int(bits[2]), int(bits[3]), int(bits[4])), 0)
        return (I + 1j * Q) / np.sqrt(26)
    elif n_bits == 6:
        I = _GRAY_8PAM.get((int(bits[0]), int(bits[1]), int(bits[2])), 0)
        Q = _GRAY_8PAM.get((int(bits[3]), int(bits[4]), int(bits[5])), 0)
        return (I + 1j * Q) / np.sqrt(42)
    return 0j


def _qam_to_bits(symbol, n_bits):
    """Hard-decision demodulate QAM symbol to bits (Gray-coded)."""
    if n_bits == 0:
        return []
    if n_bits == 1:
        return [1 if symbol.real > 0 else 0]
    elif n_bits == 2:
        return [1 if symbol.real > 0 else 0,
                1 if symbol.imag > 0 else 0]
    elif n_bits == 3:
        s_I = symbol.real * np.sqrt(6)
        s_Q = symbol.imag * np.sqrt(6)
        b0 = 1 if s_I > 0 else 0
        def _dec4(v):
            if v < -2: return (0, 0)
            elif v < 0: return (0, 1)
            elif v < 2: return (1, 1)
            else:       return (1, 0)
        b12 = _dec4(s_Q)
        return [b0] + list(b12)
    elif n_bits == 4:
        s = symbol * np.sqrt(10)
        def _decide_4pam(v):
            if v < -2: return (0, 0)
            elif v < 0: return (0, 1)
            elif v < 2: return (1, 1)
            else:       return (1, 0)
        bi = _decide_4pam(s.real)
        bq = _decide_4pam(s.imag)
        return list(bi) + list(bq)
    elif n_bits == 5:
        s = symbol * np.sqrt(26)
        def _dec4(v):
            if v < -2: return (0, 0)
            elif v < 0: return (0, 1)
            elif v < 2: return (1, 1)
            else:       return (1, 0)
        def _dec8(v):
            levels = np.array([-7, -5, -3, -1, 1, 3, 5, 7])
            idx = np.argmin(np.abs(levels - v))
            mapping = [(0,0,0),(0,0,1),(0,1,1),(0,1,0),(1,1,0),(1,1,1),(1,0,1),(1,0,0)]
            return mapping[idx]
        bi = _dec4(s.real)
        bq = _dec8(s.imag)
        return list(bi) + list(bq)
    elif n_bits == 6:
        s = symbol * np.sqrt(42)
        def _decide_8pam(v):
            levels = np.array([-7, -5, -3, -1, 1, 3, 5, 7])
            idx = np.argmin(np.abs(levels - v))
            mapping = [(0,0,0),(0,0,1),(0,1,1),(0,1,0),(1,1,0),(1,1,1),(1,0,1),(1,0,0)]
            return mapping[idx]
        bi = _decide_8pam(s.real)
        bq = _decide_8pam(s.imag)
        return list(bi) + list(bq)
    return [0] * n_bits


# =============================================================================
# THEORETICAL BER
# =============================================================================

def ber_mqam(snr_db, M):
    """Theoretical BER for M-QAM over AWGN (Proakis)."""
    snr = 10 ** (snr_db / 10)
    if M == 2:
        return 0.5 * erfc(np.sqrt(snr))
    elif M == 4:
        return 0.5 * erfc(np.sqrt(snr / 2))
    elif M == 16:
        return (3/8) * erfc(np.sqrt(snr / 10))
    elif M == 64:
        return (7/24) * erfc(np.sqrt(snr / 42))
    else:
        k = np.log2(M)
        return (2/k) * (1 - 1/np.sqrt(M)) * erfc(np.sqrt(3*snr / (2*(M-1))))


# =============================================================================
# BIT ALLOCATION
# =============================================================================

BIT_ALLOCATION_TABLE = [
    (22.0, 6, 64),
    (16.0, 5, 32),
    (10.0, 4, 16),
    (7.0,  3, 8),
    (4.0,  2, 4),
    (1.0,  1, 2),
]


def allocate_bits(snr_per_sc, table=None):
    """Allocate bits per subcarrier based on SNR thresholds."""
    if table is None:
        table = BIT_ALLOCATION_TABLE
    allocation = np.zeros(len(snr_per_sc), dtype=int)
    for i, snr in enumerate(snr_per_sc):
        for threshold, bits, _ in table:
            if snr >= threshold:
                allocation[i] = bits
                break
    return allocation


# =============================================================================
# CHANNEL RESPONSE MODEL
# =============================================================================

def solar_panel_channel_response(n_subcarriers, signal_bandwidth_hz,
                                  junction_capacitance_pf, shunt_resistance_kohm,
                                  snr_base_db=28.0, snr_floor_db=17.0):
    """
    RC low-pass channel response for solar panel receiver.
    Returns (H, snr_db) per subcarrier.
    """
    freqs = np.linspace(0, signal_bandwidth_hz, n_subcarriers)
    C = junction_capacitance_pf * 1e-12
    R = shunt_resistance_kohm * 1e3
    f_3db = 1 / (2 * np.pi * R * C)
    H = 1 / np.sqrt(1 + (freqs / f_3db) ** 2)
    snr_db = snr_base_db - (snr_base_db - snr_floor_db) * (1 - H**2)
    return H, snr_db


# =============================================================================
# OFDM MODEM
# =============================================================================

class OFDMModem:
    """Unified OFDM modem with adaptive or fixed modulation."""

    def __init__(self, nfft=256, cp_length=32, n_subcarriers=80, bandwidth_hz=5e6):
        self.nfft = nfft
        self.cp_length = cp_length
        self.n_subcarriers = n_subcarriers
        self.bandwidth_hz = bandwidth_hz
        self.subcarrier_spacing = bandwidth_hz / n_subcarriers
        self.bit_allocation = np.zeros(n_subcarriers, dtype=int)
        self.power_allocation = np.ones(n_subcarriers)

    def set_fixed_modulation(self, qam_order=16):
        bps = int(np.log2(qam_order))
        self.bit_allocation = np.full(self.n_subcarriers, bps, dtype=int)
        self.power_allocation = np.ones(self.n_subcarriers)

    def set_adaptive_modulation(self, snr_per_sc):
        self.bit_allocation = allocate_bits(snr_per_sc[:self.n_subcarriers])
        self.power_allocation = np.ones(self.n_subcarriers)

    @property
    def total_bits_per_symbol(self):
        return int(np.sum(self.bit_allocation))

    def calculate_data_rate(self, sample_rate_hz):
        """Gross data rate in Mbps."""
        symbol_rate = sample_rate_hz / (self.nfft + self.cp_length)
        return symbol_rate * self.total_bits_per_symbol / 1e6

    def simulate(self, n_symbols, snr_per_sc, channel_H=None,
                 equalization='ZF', seed=None):
        """Full per-subcarrier Monte Carlo simulation."""
        if seed is not None:
            np.random.seed(seed)

        n_sc = self.n_subcarriers
        snr = snr_per_sc[:n_sc]
        H = channel_H[:n_sc] if channel_H is not None else np.ones(n_sc)

        err_per_sc = np.zeros(n_sc)
        tot_per_sc = np.zeros(n_sc)
        evm_per_sc = np.zeros(n_sc)
        sample_tx = []
        sample_rx = []

        for sym_idx in range(n_symbols):
            for sc in range(n_sc):
                nb = self.bit_allocation[sc]
                if nb == 0:
                    continue
                tx_bits = np.random.randint(0, 2, nb)
                tx_sym = _bits_to_qam(tx_bits, nb)
                rx_sym = tx_sym * H[sc]

                snr_lin = 10 ** (snr[sc] / 10)
                sig_power = np.abs(rx_sym) ** 2
                noise_power = sig_power / snr_lin if snr_lin > 0 else 1e-10
                noise_std = np.sqrt(noise_power / 2)
                noise = noise_std * (np.random.randn() + 1j * np.random.randn())
                rx_noisy = rx_sym + noise

                if equalization == 'ZF' and abs(H[sc]) > 1e-10:
                    rx_eq = rx_noisy / H[sc]
                else:
                    rx_eq = rx_noisy

                rx_bits = _qam_to_bits(rx_eq, nb)
                errors = sum(int(a) != int(b) for a, b in zip(tx_bits, rx_bits))
                err_per_sc[sc] += errors
                tot_per_sc[sc] += nb

                err_vec = rx_eq - tx_sym
                if abs(tx_sym) > 1e-10:
                    evm_per_sc[sc] += abs(err_vec) / abs(tx_sym) * 100

                if sym_idx % max(1, n_symbols // 100) == 0:
                    sample_tx.append(tx_sym)
                    sample_rx.append(rx_eq)

        ber_per_sc = err_per_sc / np.maximum(tot_per_sc, 1)
        evm_per_sc /= n_symbols
        total_errors = np.sum(err_per_sc)
        total_bits = np.sum(tot_per_sc)
        overall_ber = total_errors / total_bits if total_bits > 0 else 0

        return {
            'ber_per_sc': ber_per_sc,
            'evm_per_sc': evm_per_sc,
            'overall_ber': overall_ber,
            'total_errors': int(total_errors),
            'total_bits': int(total_bits),
            'n_symbols': n_symbols,
            'n_subcarriers': n_sc,
            'bit_allocation': self.bit_allocation.copy(),
            'snr_per_sc': snr.copy(),
            'channel_H': H.copy(),
            'tx_symbols': np.array(sample_tx),
            'rx_symbols': np.array(sample_rx),
        }


# =============================================================================
# DATA RATE HELPERS
# =============================================================================

def gross_data_rate(bit_allocation, nfft, cp_length, sample_rate_hz):
    """Gross data rate in Mbps."""
    symbol_rate = sample_rate_hz / (nfft + cp_length)
    return symbol_rate * np.sum(bit_allocation) / 1e6


def net_data_rate(gross_rate_mbps, fec_overhead=0.171):
    """Net data rate after FEC overhead."""
    return gross_rate_mbps * (1 - fec_overhead)
