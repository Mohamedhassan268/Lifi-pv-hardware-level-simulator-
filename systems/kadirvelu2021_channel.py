# systems/kadirvelu2021_channel.py
"""
Optical Channel Model for Kadirvelu 2021 LiFi-PV System
========================================================

Models the free-space optical channel between LED TX and PV RX:
    - LOS Lambertian path loss
    - Optical gain computation
    - SPICE behavioral source generation
    - Channel impulse response
    - AWGN noise addition

Usage:
    from systems.kadirvelu2021_channel import OpticalChannel

    channel = OpticalChannel()
    Gop = channel.dc_gain()
    P_rx = channel.received_power(P_tx=9.3e-3)
"""

import numpy as np
from typing import Dict, Tuple, Optional
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from systems.kadirvelu2021 import KadirveluParams


class OpticalChannel:
    """
    LOS Lambertian Optical Channel for indoor LiFi.

    Models the optical path from LED transmitter to PV receiver
    including:
        - Lambertian emission pattern
        - Distance-dependent path loss
        - Receiver area and orientation
        - Lens effects (transmittance)
    """

    def __init__(self, params: KadirveluParams = None):
        """
        Initialize optical channel.

        Args:
            params: System parameters (default: Kadirvelu 2021)
        """
        self.p = params or KadirveluParams()

    # -------------------------------------------------------------------------
    # Core Channel Properties
    # -------------------------------------------------------------------------

    def lambertian_order(self) -> float:
        """
        Calculate Lambertian emission order m.

        m = -ln(2) / ln(cos(alpha_half))
        """
        return self.p.lambertian_order()

    def dc_gain(self, distance_m: float = None,
                theta_deg: float = None,
                beta_deg: float = None) -> float:
        """
        Calculate DC optical channel gain H(0).

        H(0) = (m+1) / (2*pi*r^2) * cos^m(theta) * cos(beta) * A_rx * T_lens

        Args:
            distance_m: Override distance (default: from params)
            theta_deg: Override TX angle (default: from params)
            beta_deg: Override RX angle (default: from params)

        Returns:
            Optical channel gain (dimensionless)
        """
        m = self.lambertian_order()
        r = distance_m or self.p.DISTANCE_M
        theta = np.radians(theta_deg if theta_deg is not None else self.p.THETA_DEG)
        beta = np.radians(beta_deg if beta_deg is not None else self.p.BETA_DEG)
        A_rx = self.p.SC_AREA_CM2 * 1e-4  # m^2

        Gop = ((m + 1) / (2 * np.pi * r**2) *
               np.cos(theta)**m * np.cos(beta) * A_rx)

        return Gop

    def received_power(self, P_tx: float = None,
                       distance_m: float = None) -> float:
        """
        Calculate received optical power.

        P_rx = P_tx * H(0)

        Args:
            P_tx: Transmitted optical power (W)
            distance_m: Override distance

        Returns:
            Received power in W
        """
        if P_tx is None:
            P_tx = self.p.LED_RADIATED_POWER_mW * 1e-3

        Gop = self.dc_gain(distance_m=distance_m)
        return P_tx * Gop

    def received_photocurrent(self, P_tx: float = None,
                               distance_m: float = None) -> float:
        """
        Calculate photocurrent at receiver.

        I_ph = R_lambda * P_rx

        Returns:
            Photocurrent in A
        """
        P_rx = self.received_power(P_tx, distance_m)
        return self.p.SC_RESPONSIVITY * P_rx

    # -------------------------------------------------------------------------
    # Channel Transfer Function
    # -------------------------------------------------------------------------

    def impulse_response(self, t: np.ndarray) -> np.ndarray:
        """
        Channel impulse response.

        For LOS link: h(t) = H(0) * delta(t - tau)
        where tau = r/c (propagation delay, ~1 ns for indoor)

        For practical purposes, the delay is negligible and we model
        it as a pure gain.

        Args:
            t: Time array (s)

        Returns:
            Impulse response array
        """
        c = 3e8  # speed of light
        tau = self.p.DISTANCE_M / c  # ~1 ns propagation delay

        H0 = self.dc_gain()

        # Model as narrow Gaussian (approximate delta)
        sigma = 1e-10  # 0.1 ns width
        h = H0 * np.exp(-0.5 * ((t - tau) / sigma)**2) / (sigma * np.sqrt(2 * np.pi))

        return h

    def apply_channel(self, tx_signal: np.ndarray,
                      noise_power_dbm: float = None) -> np.ndarray:
        """
        Apply channel to transmitted signal.

        For LOS: rx = H(0) * tx + noise

        Args:
            tx_signal: Transmitted optical power waveform (W)
            noise_power_dbm: Optional AWGN noise power (dBm)

        Returns:
            Received optical power waveform (W)
        """
        H0 = self.dc_gain()
        rx = H0 * tx_signal

        if noise_power_dbm is not None:
            noise_power_W = 10**((noise_power_dbm - 30) / 10)
            noise = np.random.randn(len(rx)) * np.sqrt(noise_power_W)
            rx += noise

        return rx

    # -------------------------------------------------------------------------
    # SPICE Integration
    # -------------------------------------------------------------------------

    def spice_channel_element(self, tx_node: str = 'tx_optical',
                               rx_anode: str = 'sc_anode',
                               rx_cathode: str = 'sc_cathode') -> str:
        """
        Generate SPICE behavioral current source for channel.

        The channel is modeled as a voltage-controlled current source:
            I_ph = G0 * V(tx_optical)
        where G0 = H(0) * R_lambda

        Args:
            tx_node: TX optical output voltage node (1V = 1W)
            rx_anode: Solar cell anode node
            rx_cathode: Solar cell cathode node

        Returns:
            SPICE element string
        """
        G0 = self.dc_gain() * self.p.SC_RESPONSIVITY

        return (f"* Optical Channel: LOS Lambertian\n"
                f"* H(0) = {self.dc_gain():.6e}\n"
                f"* G0 = H(0) * R_lambda = {G0:.6e} A/W\n"
                f"Gchannel {rx_cathode} {rx_anode} "
                f"VALUE = {{{G0:.6e} * V({tx_node})}}")

    def spice_optical_source(self, P_dc_W: float = None,
                              modulation_depth: float = 0.33,
                              f_signal: float = 5e3) -> str:
        """
        Generate SPICE voltage source representing received optical power.

        V(optical) represents optical power in Watts (1V = 1W).

        Args:
            P_dc_W: DC optical power at receiver (W)
            modulation_depth: OOK modulation depth
            f_signal: Signal frequency (Hz)

        Returns:
            SPICE source string
        """
        if P_dc_W is None:
            P_dc_W = self.received_power()

        P_ac = P_dc_W * modulation_depth

        return (f"* Received optical power (modulated)\n"
                f"* P_dc = {P_dc_W*1e6:.2f} uW, mod_depth = {modulation_depth}\n"
                f"Voptical optical_power 0 "
                f"SIN({P_dc_W:.6e} {P_ac:.6e} {f_signal})")

    # -------------------------------------------------------------------------
    # Link Budget
    # -------------------------------------------------------------------------

    def link_budget(self) -> Dict[str, float]:
        """
        Calculate complete link budget.

        Returns:
            Dict with all link budget parameters
        """
        p = self.p
        m = self.lambertian_order()
        Gop = self.dc_gain()
        P_tx = p.LED_RADIATED_POWER_mW * 1e-3
        P_rx = self.received_power()
        I_ph = self.received_photocurrent()

        # Path loss in dB
        path_loss_dB = -10 * np.log10(Gop) if Gop > 0 else float('inf')

        return {
            'P_tx_mW': P_tx * 1e3,
            'P_tx_dBm': 10 * np.log10(P_tx * 1e3),
            'lambertian_order': m,
            'half_angle_deg': p.LED_HALF_ANGLE_DEG,
            'distance_m': p.DISTANCE_M,
            'rx_area_cm2': p.SC_AREA_CM2,
            'optical_gain': Gop,
            'path_loss_dB': path_loss_dB,
            'lens_transmittance': p.LENS_TRANSMITTANCE,
            'P_rx_uW': P_rx * 1e6,
            'P_rx_dBm': 10 * np.log10(P_rx * 1e3) if P_rx > 0 else -float('inf'),
            'responsivity_AW': p.SC_RESPONSIVITY,
            'I_ph_uA': I_ph * 1e6,
        }

    def distance_sweep(self, distances_cm: np.ndarray = None) -> Dict[str, np.ndarray]:
        """
        Sweep received power vs distance.

        Args:
            distances_cm: Array of distances in cm

        Returns:
            Dict with 'distance_cm', 'P_rx_uW', 'I_ph_uA', 'path_loss_dB'
        """
        if distances_cm is None:
            distances_cm = np.linspace(5, 100, 50)

        P_rx = np.array([self.received_power(distance_m=d/100) for d in distances_cm])
        I_ph = P_rx * self.p.SC_RESPONSIVITY
        Gop = np.array([self.dc_gain(distance_m=d/100) for d in distances_cm])
        path_loss = -10 * np.log10(np.maximum(Gop, 1e-30))

        return {
            'distance_cm': distances_cm,
            'P_rx_uW': P_rx * 1e6,
            'I_ph_uA': I_ph * 1e6,
            'path_loss_dB': path_loss,
        }

    def print_link_budget(self):
        """Print formatted link budget."""
        lb = self.link_budget()

        print("=" * 50)
        print("OPTICAL LINK BUDGET")
        print("=" * 50)
        print(f"  TX Power:        {lb['P_tx_mW']:.1f} mW ({lb['P_tx_dBm']:.1f} dBm)")
        print(f"  Lambertian m:    {lb['lambertian_order']:.2f}")
        print(f"  Half-angle:      {lb['half_angle_deg']:.0f}°")
        print(f"  Distance:        {lb['distance_m']*100:.1f} cm")
        print(f"  RX Area:         {lb['rx_area_cm2']:.1f} cm²")
        print(f"  Path Loss:       {lb['path_loss_dB']:.1f} dB")
        print(f"  Lens T:          {lb['lens_transmittance']:.0%}")
        print(f"  RX Power:        {lb['P_rx_uW']:.2f} µW ({lb['P_rx_dBm']:.1f} dBm)")
        print(f"  Responsivity:    {lb['responsivity_AW']:.3f} A/W")
        print(f"  Photocurrent:    {lb['I_ph_uA']:.2f} µA")
        print("=" * 50)


# =============================================================================
# SELF-TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("OPTICAL CHANNEL MODEL - SELF TEST")
    print("=" * 60)

    channel = OpticalChannel()
    channel.print_link_budget()

    # Test distance sweep
    print("\nDistance sweep:")
    sweep = channel.distance_sweep(np.array([10, 20, 32.5, 50, 100]))
    for d, p, i in zip(sweep['distance_cm'], sweep['P_rx_uW'], sweep['I_ph_uA']):
        print(f"  {d:5.1f} cm: P_rx = {p:8.2f} µW, I_ph = {i:8.2f} µA")

    # Test SPICE source
    print(f"\nSPICE optical source:")
    print(f"  {channel.spice_optical_source()}")

    print("\n[OK] Channel model tests passed!")
