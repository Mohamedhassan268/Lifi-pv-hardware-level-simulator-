# cosim/dcdc_model.py
"""
Boost DC-DC Converter — Steady-State and Time-Domain Model.

Models the boost converter used to step up the PV cell voltage to Vcc
for powering the receiver electronics.

Equations:
    CCM: V_out = V_in / (1 - D)
    CCM boundary: I_LB = V_in * D * (1-D)^2 / (2 * L * f_sw)
    Losses: conduction + switching + diode + inductor DCR

References:
    - Erickson & Maksimovic, "Fundamentals of Power Electronics", Ch. 2-3
    - SPICE subcircuit BOOST_DCDC in systems/kadirvelu2021_netlist.py
    - NTS4409 datasheet: R_ds_on = 52 mohm
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class BoostResult:
    """Output of boost converter computation."""
    V_in: float             # Input voltage (V)
    V_out: float            # Output voltage (V)
    D: float                # Duty cycle
    mode: str               # 'CCM' or 'DCM'
    I_load: float           # Load current (A)
    I_L_avg: float          # Average inductor current (A)
    I_L_peak: float         # Peak inductor current (A)
    P_in: float             # Input power (W)
    P_out: float            # Output power (W)
    P_loss_total: float     # Total losses (W)
    P_cond: float           # MOSFET conduction loss (W)
    P_sw: float             # Switching loss (W)
    P_diode: float          # Diode forward loss (W)
    P_dcr: float            # Inductor DCR loss (W)
    efficiency: float       # eta = P_out / P_in


class BoostConverter:
    """
    Boost DC-DC converter model with CCM/DCM detection and loss analysis.

    Usage:
        conv = BoostConverter.from_config(cfg)
        result = conv.compute(V_in=0.5, V_out_target=3.3)
    """

    def __init__(self, L_H: float = 22e-6,
                 f_sw_Hz: float = 50e3,
                 R_ds_on: float = 0.052,
                 V_diode: float = 0.3,
                 R_dcr: float = 0.5,
                 C_out_F: float = 47e-6,
                 R_load: float = 180e3,
                 t_rise_s: float = 10e-9,
                 t_fall_s: float = 10e-9):
        """
        Args:
            L_H: Inductance (H)
            f_sw_Hz: Switching frequency (Hz)
            R_ds_on: MOSFET on-resistance (ohm)
            V_diode: Schottky diode forward voltage (V)
            R_dcr: Inductor DC resistance (ohm)
            C_out_F: Output capacitance (F)
            R_load: Load resistance (ohm)
            t_rise_s: MOSFET rise time (s)
            t_fall_s: MOSFET fall time (s)
        """
        self.L = L_H
        self.f_sw = f_sw_Hz
        self.R_ds_on = R_ds_on
        self.V_diode = V_diode
        self.R_dcr = R_dcr
        self.C_out = C_out_F
        self.R_load = R_load
        self.t_rise = t_rise_s
        self.t_fall = t_fall_s

    @classmethod
    def from_config(cls, config) -> 'BoostConverter':
        """Create from SystemConfig."""
        return cls(
            L_H=config.dcdc_l_uH * 1e-6,
            f_sw_Hz=config.dcdc_fsw_kHz * 1e3,
            R_ds_on=config.dcdc_rds_on_mohm * 1e-3,
            V_diode=config.dcdc_diode_vf_V,
            R_dcr=config.dcdc_inductor_dcr_ohm,
            C_out_F=config.dcdc_cl_uF * 1e-6,
            R_load=config.r_load_ohm,
        )

    def ccm_boundary_current(self, V_in: float, D: float) -> float:
        """
        CCM/DCM boundary load current.
        I_LB = V_in * D * (1-D)^2 / (2 * L * f_sw)
        """
        return V_in * D * (1 - D) ** 2 / (2 * self.L * self.f_sw)

    def compute(self, V_in: float, V_out_target: float = 3.3) -> BoostResult:
        """
        Compute boost converter operating point.

        Args:
            V_in: Input voltage from PV cell (V)
            V_out_target: Desired output voltage (V)

        Returns:
            BoostResult with voltages, currents, losses, efficiency
        """
        if V_in <= 0 or V_in >= V_out_target:
            # Cannot boost: V_in too low or already above target
            return BoostResult(
                V_in=V_in, V_out=V_in, D=0.0, mode='BYPASS',
                I_load=V_in / self.R_load, I_L_avg=V_in / self.R_load,
                I_L_peak=V_in / self.R_load,
                P_in=V_in ** 2 / self.R_load,
                P_out=V_in ** 2 / self.R_load,
                P_loss_total=0.0, P_cond=0.0, P_sw=0.0,
                P_diode=0.0, P_dcr=0.0,
                efficiency=1.0 if V_in > 0 else 0.0,
            )

        # Ideal duty cycle (ignoring losses initially)
        D = 1.0 - V_in / V_out_target
        D = np.clip(D, 0.01, 0.95)

        # Load current
        I_load = V_out_target / self.R_load

        # Average inductor current (CCM): I_L = I_load / (1-D)
        I_L_avg = I_load / (1 - D)

        # CCM boundary current
        I_LB = self.ccm_boundary_current(V_in, D)

        # Determine operating mode
        if I_load > I_LB:
            mode = 'CCM'
            V_out = V_out_target
        else:
            mode = 'DCM'
            # DCM output voltage from energy balance:
            # V_out = V_in * (1 + sqrt(1 + 4*D^2 / (2*L*f_sw*I_load/V_in))) / 2
            # Simplified: V_out ≈ V_in * (1 + D * sqrt(R_load / (2*L*f_sw)))
            K = 2 * self.L * self.f_sw / self.R_load
            M = (1 + np.sqrt(1 + 4 * D ** 2 / K)) / 2
            V_out = V_in * M
            V_out = min(V_out, V_out_target)  # Clamp to target
            I_load = V_out / self.R_load
            I_L_avg = I_load / (1 - D) if D < 1 else I_load

        # Inductor current ripple (CCM)
        T_sw = 1.0 / self.f_sw
        delta_I_L = V_in * D * T_sw / self.L
        I_L_peak = I_L_avg + delta_I_L / 2

        # RMS inductor current (triangular ripple in CCM)
        I_L_rms = np.sqrt(I_L_avg ** 2 + (delta_I_L ** 2) / 12)

        # --- Loss model ---

        # 1. MOSFET conduction loss: P = I_rms^2 * R_ds_on * D
        P_cond = I_L_rms ** 2 * self.R_ds_on * D

        # 2. Switching loss: P = 0.5 * V_out * I_L_avg * (t_r + t_f) * f_sw
        P_sw = 0.5 * V_out * I_L_avg * (self.t_rise + self.t_fall) * self.f_sw

        # 3. Diode forward loss: P = V_f * I_load
        P_diode = self.V_diode * I_load

        # 4. Inductor DCR loss: P = I_rms^2 * R_dcr
        P_dcr = I_L_rms ** 2 * self.R_dcr

        P_loss_total = P_cond + P_sw + P_diode + P_dcr

        # Output power and efficiency
        P_out = V_out * I_load
        P_in = P_out + P_loss_total
        efficiency = P_out / P_in if P_in > 0 else 0.0

        return BoostResult(
            V_in=V_in, V_out=V_out, D=D, mode=mode,
            I_load=I_load, I_L_avg=I_L_avg, I_L_peak=I_L_peak,
            P_in=P_in, P_out=P_out,
            P_loss_total=P_loss_total,
            P_cond=P_cond, P_sw=P_sw, P_diode=P_diode, P_dcr=P_dcr,
            efficiency=efficiency,
        )

    def inductor_current_waveform(self, V_in: float, V_out: float,
                                  D: float, I_L_avg: float,
                                  n_cycles: int = 5,
                                  points_per_cycle: int = 200) -> tuple:
        """
        Generate time-domain inductor current waveform (triangular in CCM).

        Args:
            V_in: Input voltage (V)
            V_out: Output voltage (V)
            D: Duty cycle
            I_L_avg: Average inductor current (A)
            n_cycles: Number of switching cycles
            points_per_cycle: Time points per cycle

        Returns:
            (t, I_L) tuple of numpy arrays
        """
        T_sw = 1.0 / self.f_sw
        t_total = n_cycles * T_sw
        n_pts = n_cycles * points_per_cycle

        t = np.linspace(0, t_total, n_pts)
        I_L = np.zeros(n_pts)

        # Current slopes
        m_on = V_in / self.L                      # Rising slope (switch ON)
        m_off = -(V_out - V_in) / self.L          # Falling slope (switch OFF)

        delta_I = V_in * D * T_sw / self.L
        I_min = I_L_avg - delta_I / 2

        for i, ti in enumerate(t):
            t_in_cycle = ti % T_sw
            if t_in_cycle < D * T_sw:
                # Switch ON phase
                I_L[i] = I_min + m_on * t_in_cycle
            else:
                # Switch OFF phase
                t_off = t_in_cycle - D * T_sw
                I_L[i] = I_min + delta_I + m_off * t_off

        return t, I_L

    def __repr__(self) -> str:
        return (f"BoostConverter(L={self.L*1e6:.0f}uH, "
                f"fsw={self.f_sw/1e3:.0f}kHz, "
                f"Rds={self.R_ds_on*1e3:.0f}mohm)")
