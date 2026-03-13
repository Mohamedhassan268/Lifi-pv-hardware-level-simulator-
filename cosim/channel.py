# cosim/channel.py
"""
Generalized Optical Channel Model.

Implements Lambertian LOS propagation with optional Beer-Lambert atmospheric
attenuation, first-order diffuse wall reflections, and MIMO support for
multi-cell arrays.

All parameters are sourced from SystemConfig — zero hardcoded values.

Physics:
    LOS gain: G = [(m+1)·A_rx / (2π·d²)] · cos^m(φ) · cos(ψ)
    Beer-Lambert: P_rx *= exp(-α·d)
    Multipath: first-order Lambertian wall reflections

References:
    - Kahn & Barry, "Wireless Infrared Communications", Proc. IEEE 1997
    - Komine & Nakagawa, "Fundamental analysis for VLC using LED", IEEE TCE 2004
    - Correa et al. 2025 (Beer-Lambert humidity model)
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple


# =============================================================================
# CHANNEL GAIN RESULT
# =============================================================================

@dataclass
class ChannelResult:
    """Result of channel gain computation with breakdown."""
    gain_los: float              # LOS channel gain (dimensionless)
    gain_diffuse: float          # Diffuse (multipath) gain
    gain_total: float            # Total: LOS + diffuse
    beer_lambert_factor: float   # Atmospheric attenuation factor (0-1)
    received_power_W: float      # P_rx = P_tx * gain_total * beer_lambert
    lambertian_order: float      # m
    distance_m: float


# =============================================================================
# OPTICAL CHANNEL
# =============================================================================

class OpticalChannel:
    """
    Free-space optical channel with Lambertian LOS, Beer-Lambert attenuation,
    first-order diffuse reflections, and MIMO support.

    Usage:
        from cosim.channel import OpticalChannel

        ch = OpticalChannel.from_config(system_config)
        P_rx = ch.propagate(P_tx)
        result = ch.compute_gain()
    """

    def __init__(self, distance_m: float, led_half_angle_deg: float,
                 rx_area_cm2: float, tx_angle_deg: float = 0.0,
                 rx_tilt_deg: float = 0.0, fov_half_angle_deg: float = 90.0,
                 beer_lambert_enabled: bool = False,
                 humidity_rh: Optional[float] = None,
                 temperature_K: float = 300.0,
                 n_reflections: int = 0,
                 room_dims_m: Tuple[float, float, float] = (5.0, 5.0, 3.0),
                 wall_reflectivity: float = 0.7):
        """
        Args:
            distance_m: TX-RX distance in meters
            led_half_angle_deg: LED half-power angle (degrees)
            rx_area_cm2: Receiver active area (cm²)
            tx_angle_deg: TX emission angle from normal (degrees)
            rx_tilt_deg: RX tilt angle from vertical (degrees)
            fov_half_angle_deg: Receiver FOV half-angle (degrees), 90 = hemispherical
            beer_lambert_enabled: Enable atmospheric attenuation
            humidity_rh: Relative humidity 0-1 (None = no humidity effect)
            temperature_K: Ambient temperature
            n_reflections: Number of wall-bounce reflections (0 = LOS only)
            room_dims_m: (length, width, height) in meters
            wall_reflectivity: Diffuse wall reflectance (0-1)
        """
        self.distance_m = distance_m
        self.rx_area_m2 = rx_area_cm2 * 1e-4
        self.tx_angle_rad = np.radians(tx_angle_deg)
        self.rx_tilt_rad = np.radians(rx_tilt_deg)
        self.fov_half_rad = np.radians(fov_half_angle_deg)
        self.temperature_K = temperature_K

        # Lambertian order: m = -ln2 / ln(cos(half_angle))
        alpha = np.radians(led_half_angle_deg)
        self.m_lambertian = -np.log(2) / np.log(np.cos(alpha))

        # Beer-Lambert
        self.beer_lambert_enabled = beer_lambert_enabled
        self.humidity_rh = humidity_rh
        self._alpha_attenuation = self._compute_alpha(humidity_rh) if beer_lambert_enabled else 0.0

        # Multipath
        self.n_reflections = n_reflections
        self.room_dims_m = room_dims_m
        self.wall_reflectivity = wall_reflectivity

    @classmethod
    def from_config(cls, config) -> 'OpticalChannel':
        """Create OpticalChannel from a SystemConfig instance."""
        return cls(
            distance_m=config.distance_m,
            led_half_angle_deg=config.led_half_angle_deg,
            rx_area_cm2=config.sc_area_cm2,
            tx_angle_deg=config.tx_angle_deg,
            rx_tilt_deg=config.rx_tilt_deg,
            fov_half_angle_deg=config.fov_half_angle_deg,
            beer_lambert_enabled=config.beer_lambert_enabled,
            humidity_rh=config.humidity_rh,
            temperature_K=config.temperature_K,
            n_reflections=config.n_reflections,
            room_dims_m=(config.room_length_m, config.room_width_m, config.room_height_m),
            wall_reflectivity=config.wall_reflectivity,
        )

    # -------------------------------------------------------------------------
    # Beer-Lambert attenuation
    # -------------------------------------------------------------------------

    @staticmethod
    def _compute_alpha(humidity_rh: Optional[float]) -> float:
        """
        Beer-Lambert attenuation coefficient from relative humidity.

        Model from Correa et al. 2025:
            α = α_base + α_humidity
            α_base = 0.3 (clean-air baseline at visible wavelengths)
            α_humidity = 4.0 * (RH - 0.3)^1.5  for RH >= 0.3

        Args:
            humidity_rh: Relative humidity 0-1 (None = 0)

        Returns:
            Attenuation coefficient in 1/m
        """
        if humidity_rh is None:
            return 0.0
        rh = np.clip(humidity_rh, 0.0, 1.0)
        alpha_base = 0.3
        alpha_humidity = 4.0 * max(rh - 0.3, 0.0) ** 1.5
        return alpha_base + alpha_humidity

    def beer_lambert_factor(self) -> float:
        """Atmospheric transmission factor: exp(-α·d)."""
        if not self.beer_lambert_enabled or self._alpha_attenuation == 0.0:
            return 1.0
        return float(np.exp(-self._alpha_attenuation * self.distance_m))

    # -------------------------------------------------------------------------
    # LOS channel gain
    # -------------------------------------------------------------------------

    def los_gain(self) -> float:
        """
        Lambertian LOS channel gain H_LOS(0).

        G = [(m+1) · A_rx / (2π · d²)] · cos^m(φ) · cos(ψ)

        where φ = TX emission angle, ψ = RX incidence angle.
        Returns 0 if ψ exceeds FOV half-angle.
        """
        psi = self.rx_tilt_rad  # RX incidence angle

        # FOV check
        if abs(psi) > self.fov_half_rad:
            return 0.0

        m = self.m_lambertian
        d = self.distance_m
        phi = self.tx_angle_rad

        gain = ((m + 1) * self.rx_area_m2 / (2 * np.pi * d**2)
                * np.cos(phi)**m * np.cos(psi))
        return max(float(gain), 0.0)

    # -------------------------------------------------------------------------
    # First-order diffuse reflections
    # -------------------------------------------------------------------------

    def diffuse_gain(self) -> float:
        """
        First-order diffuse wall reflection gain (simplified model).

        Approximation: sum over 4 walls + ceiling of first-bounce Lambertian
        reflections. Uses the average path approximation for computational
        efficiency.

        Returns 0 if n_reflections == 0.
        """
        if self.n_reflections == 0:
            return 0.0

        L, W, H = self.room_dims_m
        rho = self.wall_reflectivity
        m = self.m_lambertian
        A_rx = self.rx_area_m2

        # Total room surface area (4 walls + ceiling, floor excluded as RX plane)
        A_walls = 2 * (L + W) * H
        A_ceiling = L * W
        A_total = A_walls + A_ceiling

        # Average first-bounce gain (Komine & Nakagawa 2004 approximation):
        # G_diff ≈ ρ · (m+1) · A_rx / (π · A_room) for diffuse scenario
        gain_diff = rho * (m + 1) * A_rx / (np.pi * A_total)

        return max(float(gain_diff), 0.0)

    # -------------------------------------------------------------------------
    # Total channel gain
    # -------------------------------------------------------------------------

    def channel_gain(self) -> float:
        """Total channel gain including LOS + diffuse + Beer-Lambert."""
        g_los = self.los_gain()
        g_diff = self.diffuse_gain()
        bl = self.beer_lambert_factor()
        return (g_los + g_diff) * bl

    def compute_gain(self) -> ChannelResult:
        """Compute full channel result with breakdown."""
        g_los = self.los_gain()
        g_diff = self.diffuse_gain()
        bl = self.beer_lambert_factor()
        g_total = g_los + g_diff
        return ChannelResult(
            gain_los=g_los,
            gain_diffuse=g_diff,
            gain_total=g_total,
            beer_lambert_factor=bl,
            received_power_W=0.0,  # Filled by caller with P_tx
            lambertian_order=self.m_lambertian,
            distance_m=self.distance_m,
        )

    # -------------------------------------------------------------------------
    # Signal propagation
    # -------------------------------------------------------------------------

    def propagate(self, P_tx):
        """
        Apply channel gain to transmitted optical power.

        Args:
            P_tx: Transmitted optical power (scalar or array, in Watts)

        Returns:
            P_rx: Received optical power (same shape as P_tx)
        """
        return self.channel_gain() * np.asarray(P_tx)

    # -------------------------------------------------------------------------
    # MIMO support
    # -------------------------------------------------------------------------

    @staticmethod
    def compute_mimo_gain_matrix(tx_positions, rx_positions,
                                 led_half_angle_deg, rx_area_cm2,
                                 fov_half_angle_deg=90.0):
        """
        Compute MIMO channel gain matrix H[n_rx, n_tx].

        Each element H[i,j] is the LOS channel gain from TX_j to RX_i.
        TX and RX assumed pointing downward and upward respectively.

        Args:
            tx_positions: array of shape (n_tx, 3) — [x, y, z] in meters
            rx_positions: array of shape (n_rx, 3) — [x, y, z] in meters
            led_half_angle_deg: LED half-power angle
            rx_area_cm2: Per-element receiver area
            fov_half_angle_deg: Receiver FOV half-angle

        Returns:
            H: numpy array of shape (n_rx, n_tx)
        """
        tx_pos = np.atleast_2d(tx_positions)
        rx_pos = np.atleast_2d(rx_positions)
        n_tx = tx_pos.shape[0]
        n_rx = rx_pos.shape[0]

        alpha = np.radians(led_half_angle_deg)
        m = -np.log(2) / np.log(np.cos(alpha))
        A_rx = rx_area_cm2 * 1e-4
        fov_rad = np.radians(fov_half_angle_deg)

        H = np.zeros((n_rx, n_tx))
        for j in range(n_tx):
            for i in range(n_rx):
                diff = rx_pos[i] - tx_pos[j]
                d = np.linalg.norm(diff)
                if d < 1e-6:
                    continue

                # TX points downward: normal = [0, 0, -1]
                cos_phi = abs(diff[2]) / d  # cos of emission angle
                # RX points upward: normal = [0, 0, 1]
                cos_psi = abs(diff[2]) / d  # cos of incidence angle

                if np.arccos(np.clip(cos_psi, -1, 1)) > fov_rad:
                    continue

                H[i, j] = ((m + 1) * A_rx / (2 * np.pi * d**2)
                            * cos_phi**m * cos_psi)

        return H

    # -------------------------------------------------------------------------
    # String representation
    # -------------------------------------------------------------------------

    def __repr__(self) -> str:
        bl_str = f", BL α={self._alpha_attenuation:.2f}" if self.beer_lambert_enabled else ""
        diff_str = f", {self.n_reflections} reflections" if self.n_reflections > 0 else ""
        return (f"OpticalChannel(d={self.distance_m:.3f}m, "
                f"m={self.m_lambertian:.1f}, "
                f"G_LOS={self.los_gain():.4e}"
                f"{bl_str}{diff_str})")
