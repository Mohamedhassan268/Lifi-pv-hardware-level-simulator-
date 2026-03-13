# cosim/tx_model.py
"""
LED Transmitter Model — Bandwidth Limiting and Optical Output.

Models the LED driver + LED + lens chain:
    Electrical signal → LED I-V → Optical power → Lens → P_tx(t)

The key addition over the simple modulation dispatch is the LED's
finite bandwidth, which acts as a single-pole low-pass filter on the
modulated signal.

References:
    - LXM5-PD01 datasheet: modulation bandwidth ~6.6 MHz
    - ADA4891 datasheet: GBW = 220 MHz (driver, not the bottleneck)
    - SPICE subcircuit TX_DRIVER in systems/kadirvelu2021_netlist.py
"""

import numpy as np
from scipy import signal as sp_signal
from dataclasses import dataclass
from typing import Optional


@dataclass
class TXResult:
    """Output of transmitter model."""
    t: np.ndarray               # Time array (s)
    P_electrical: np.ndarray    # Modulated electrical signal (normalized)
    P_optical: np.ndarray       # Optical power after LED (W)
    P_tx: np.ndarray            # Transmitted power after lens (W)


class LEDTransmitter:
    """
    LED transmitter with bandwidth limiting and optical output model.

    Usage:
        tx = LEDTransmitter.from_config(cfg)
        result = tx.process(P_mod_normalized, t)
    """

    def __init__(self, P_led_W: float = 9.3e-3,
                 modulation_depth: float = 0.33,
                 led_f3dB_Hz: float = 6.6e6,
                 lens_transmittance: float = 0.85,
                 bandwidth_limit_enable: bool = False):
        """
        Args:
            P_led_W: LED radiated power at bias point (W)
            modulation_depth: Modulation index (0-1)
            led_f3dB_Hz: LED -3dB bandwidth (Hz)
            lens_transmittance: Lens optical transmittance (0-1)
            bandwidth_limit_enable: Apply LED frequency response
        """
        self.P_led = P_led_W
        self.mod_depth = modulation_depth
        self.f3dB = led_f3dB_Hz
        self.T_lens = lens_transmittance
        self.bw_limit = bandwidth_limit_enable

    @classmethod
    def from_config(cls, config) -> 'LEDTransmitter':
        """Create from SystemConfig."""
        # Compute LED bandwidth from component parameters if available
        # Default: use datasheet value for LXM5-PD01
        f3dB = cls._estimate_led_bandwidth(config)

        return cls(
            P_led_W=config.led_radiated_power_mW * 1e-3,
            modulation_depth=config.modulation_depth,
            led_f3dB_Hz=f3dB,
            lens_transmittance=config.lens_transmittance,
            bandwidth_limit_enable=config.led_bandwidth_limit_enable,
        )

    @staticmethod
    def _estimate_led_bandwidth(config) -> float:
        """
        Estimate LED modulation bandwidth from driver and LED parameters.

        f_3dB = 1 / (2*pi*tau_LED)
        For LXM5-PD01 with ADA4891 driver: ~6.6 MHz

        The LED bandwidth is dominated by carrier recombination time,
        which depends on the drive current. Higher bias = faster response.
        """
        # Use component model if available, otherwise datasheet default
        # tau_LED ~ R_e * C_LED where R_e is the LED dynamic resistance
        R_e = config.led_driver_re  # ohm (dynamic resistance at bias)
        # LED capacitance approximation: C ~ 1/(2*pi*f_3dB*R_e)
        # For LXM5-PD01: f_3dB ~ 6.6 MHz at 350 mA bias
        # This gives C_LED ~ 1/(2*pi*6.6e6*12.1) ~ 2 nF
        f_3dB = 1.0 / (2 * np.pi * R_e * 2e-9)  # ~6.6 MHz
        return f_3dB

    def process(self, P_mod_normalized: np.ndarray,
                t: np.ndarray) -> TXResult:
        """
        Apply LED transmitter model to a modulated signal.

        Args:
            P_mod_normalized: Normalized modulated signal (0 to 1)
                representing the modulation envelope
            t: Time array (s)

        Returns:
            TXResult with electrical, optical, and transmitted power
        """
        P_electrical = P_mod_normalized.copy()

        # Apply LED bandwidth limiting if enabled
        if self.bw_limit:
            P_electrical = self._apply_bandwidth_limit(P_electrical, t)

        # LED I-V to optical power
        # P_opt = P_LED * (1 + m * (signal - 0.5) * 2)
        # where signal is 0-1, giving P_opt range: P_LED*(1-m) to P_LED*(1+m)
        # Simplified: P_opt = P_LED * signal for OOK-like modulation
        P_optical = self.P_led * P_electrical

        # Clamp to non-negative (LED can't emit negative power)
        P_optical = np.maximum(P_optical, 0.0)

        # Apply lens transmittance
        P_tx = self.T_lens * P_optical

        return TXResult(
            t=t,
            P_electrical=P_electrical,
            P_optical=P_optical,
            P_tx=P_tx,
        )

    def _apply_bandwidth_limit(self, signal: np.ndarray,
                               t: np.ndarray) -> np.ndarray:
        """Apply single-pole low-pass filter modeling LED response."""
        dt = np.mean(np.diff(t))
        fs = 1.0 / dt

        wn = self.f3dB / (fs / 2)
        if wn <= 0 or wn >= 1:
            return signal  # Nyquist limit exceeded, no filtering needed

        b, a = sp_signal.butter(1, wn, btype='low')
        filtered = sp_signal.filtfilt(b, a, signal)

        # Clamp to [0, max(signal)] — bandwidth limiting shouldn't create
        # values outside the original range
        filtered = np.clip(filtered, 0.0, np.max(signal))
        return filtered

    def __repr__(self) -> str:
        bw_str = "ON" if self.bw_limit else "OFF"
        return (f"LEDTransmitter(P={self.P_led*1e3:.1f}mW, "
                f"m={self.mod_depth:.2f}, "
                f"f3dB={self.f3dB/1e6:.1f}MHz, "
                f"BW_limit={bw_str})")
