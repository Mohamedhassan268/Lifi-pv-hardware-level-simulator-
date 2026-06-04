"""cosim/lora_budget.py — LoRa backhaul link budget (greenhouse Tier 3).

A *link-budget block*, not a faithful chirp PHY: standard Semtech LoRa
time-on-air, effective bit rate, receiver sensitivity (AN1200.13 / SX127x), and a
log-distance link margin. Used to chain the slow long-range backhaul onto the
fast optical edge tier in the end-to-end greenhouse budget.

All times in seconds, rates in bit/s, powers in dBm.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# Required demodulator SNR per spreading factor (dB), Semtech SX127x datasheet.
_SNR_REQ_DB = {7: -7.5, 8: -10.0, 9: -12.5, 10: -15.0, 11: -17.5, 12: -20.0}


@dataclass
class LoRaLink:
    """One LoRa configuration. coding_rate is the CR index 1..4 → code 4/(4+CR)."""
    sf: int = 7
    bw_hz: float = 125_000.0
    coding_rate: int = 1
    preamble_symbols: int = 8
    crc: bool = True
    explicit_header: bool = True

    def t_sym(self) -> float:
        """Symbol duration (s)."""
        return (2 ** self.sf) / self.bw_hz

    def low_data_rate_opt(self) -> int:
        """LDRO is enabled when the symbol time exceeds ~16 ms (SF11/12 @ 125 kHz)."""
        return 1 if self.t_sym() > 16e-3 else 0

    def airtime_s(self, payload_bytes: int) -> float:
        """Time-on-air for a payload (Semtech AN1200.13)."""
        ts = self.t_sym()
        de = self.low_data_rate_opt()
        ih = 0 if self.explicit_header else 1
        crc = 1 if self.crc else 0
        num = 8 * payload_bytes - 4 * self.sf + 28 + 16 * crc - 20 * ih
        den = 4 * (self.sf - 2 * de)
        n_payload = 8 + max(math.ceil(num / den) * (self.coding_rate + 4), 0)
        t_preamble = (self.preamble_symbols + 4.25) * ts
        return t_preamble + n_payload * ts

    def bitrate_bps(self) -> float:
        """Effective (coded) bit rate Rb = SF · BW/2^SF · 4/(4+CR)."""
        return self.sf * self.bw_hz / (2 ** self.sf) * (4.0 / (4 + self.coding_rate))

    def sensitivity_dBm(self, noise_figure_dB: float = 6.0) -> float:
        """RX sensitivity S = −174 + 10log10(BW) + NF + SNR_req(SF)."""
        return -174.0 + 10.0 * math.log10(self.bw_hz) + noise_figure_dB + _SNR_REQ_DB[self.sf]

    def link_margin_dB(self, tx_power_dBm: float, distance_m: float,
                       path_loss_exp: float = 2.7, pl0_dB: float = 40.0,
                       d0_m: float = 1.0, noise_figure_dB: float = 6.0) -> float:
        """Margin = P_rx − sensitivity, with log-distance path loss
        PL(d) = PL0 + 10·n·log10(d/d0). pl0_dB≈40 is a 1 m reference for ~868 MHz.
        """
        pl = pl0_dB + 10.0 * path_loss_exp * math.log10(max(distance_m / d0_m, 1.0))
        p_rx = tx_power_dBm - pl
        return p_rx - self.sensitivity_dBm(noise_figure_dB)
