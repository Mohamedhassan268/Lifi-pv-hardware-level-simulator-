# cosim/pv_model.py
"""
PV Cell Transient Model with Voltage-Dependent Capacitance.

Solves the single-diode equivalent circuit ODE:
    C_j(V) * dV/dt = I_ph(t) - I_dark(V) - V/R_sh - I_load

where:
    I_dark = I_s * (exp(V / (n * V_T)) - 1)
    C_j(V) = C_j0 / (1 - V/V_bi)^m     (m=0.5 for abrupt junction)

Uses scipy.integrate.solve_ivp with Radau (stiff-aware) solver.

References:
    - Sze & Ng, "Physics of Semiconductor Devices", Ch. 2
    - Kadirvelu 2021, Section III-A (solar cell equivalent circuit)
"""

import numpy as np
from scipy.integrate import solve_ivp
from dataclasses import dataclass
from typing import Optional

# Physical constants
Q_ELECTRON = 1.602176634e-19
K_BOLTZMANN = 1.380649e-23
EXP_CLIP = 40.0  # Prevent exp overflow


@dataclass
class PVCellResult:
    """Output of PV cell transient simulation."""
    t: np.ndarray           # Time array (s)
    V_cell: np.ndarray      # Cell terminal voltage (V)
    I_cell: np.ndarray      # Cell output current (A)
    I_ph: np.ndarray        # Photocurrent (A)
    I_dark: np.ndarray      # Diode dark current (A)
    I_shunt: np.ndarray     # Shunt current V/R_sh (A)


class PVCellModel:
    """
    Single-diode PV cell with transient ODE solver.

    Equivalent circuit:
        I_ph(t) --> [Diode] || [R_sh] || [C_j(V)] -- Rs -- output

    Usage:
        pv = PVCellModel.from_config(cfg)
        result = pv.simulate(t, P_rx, R_load=1.0)
    """

    def __init__(self, responsivity: float = 0.457,
                 C_j0_F: float = 798e-9,
                 R_sh_ohm: float = 138.8e3,
                 R_s_ohm: float = 2.5,
                 I_s_A: float = 1e-10,
                 n: float = 1.5,
                 V_bi_V: float = 1.1,
                 temperature_K: float = 300.0):
        """
        Args:
            responsivity: A/W
            C_j0_F: Zero-bias junction capacitance (F)
            R_sh_ohm: Shunt resistance (ohm)
            R_s_ohm: Series resistance (ohm)
            I_s_A: Diode saturation current (A)
            n: Diode ideality factor
            V_bi_V: Built-in voltage for C_j model (V)
            temperature_K: Temperature (K)
        """
        self.R_lambda = responsivity
        self.C_j0 = C_j0_F
        self.R_sh = R_sh_ohm
        self.R_s = R_s_ohm
        self.I_s = I_s_A
        self.n = n
        self.V_bi = V_bi_V
        self.T = temperature_K
        self.V_T = K_BOLTZMANN * temperature_K / Q_ELECTRON

    @classmethod
    def from_config(cls, config) -> 'PVCellModel':
        """Create from SystemConfig."""
        return cls(
            responsivity=config.sc_responsivity,
            C_j0_F=config.sc_cj_nF * 1e-9,
            R_sh_ohm=config.sc_rsh_kOhm * 1e3,
            R_s_ohm=config.pv_series_resistance_ohm,
            I_s_A=config.pv_dark_current_A,
            n=config.pv_ideality_factor,
            V_bi_V=config.pv_vbi_V,
            temperature_K=config.temperature_K,
        )

    def capacitance(self, V: float) -> float:
        """Voltage-dependent junction capacitance: C_j(V) = C_j0 / (1 - V/V_bi)^0.5"""
        ratio = V / self.V_bi
        # Clamp to avoid singularity at V = V_bi
        ratio = min(ratio, 0.95)
        return self.C_j0 / np.sqrt(max(1.0 - ratio, 0.05))

    def dark_current(self, V: float) -> float:
        """Diode dark current: I_dark = I_s * (exp(V / (n*V_T)) - 1)"""
        exponent = min(V / (self.n * self.V_T), EXP_CLIP)
        return self.I_s * (np.exp(exponent) - 1.0)

    def simulate(self, t: np.ndarray, P_rx: np.ndarray,
                 R_load: float = 1.0,
                 V0: float = 0.0) -> PVCellResult:
        """
        Run transient ODE simulation.

        Args:
            t: Time array (s), must be uniformly spaced
            P_rx: Received optical power array (W), same length as t
            R_load: Load resistance (ohm)
            V0: Initial cell voltage (V)

        Returns:
            PVCellResult with time-domain waveforms
        """
        dt = t[1] - t[0]

        # Interpolator for P_rx
        def P_rx_interp(ti):
            idx = min(int((ti - t[0]) / dt), len(P_rx) - 1)
            return P_rx[max(idx, 0)]

        def ode_rhs(ti, state):
            V = state[0]
            I_ph = self.R_lambda * P_rx_interp(ti)
            I_dark = self.dark_current(V)
            I_shunt = V / self.R_sh
            I_load = V / (R_load + self.R_s)
            C_j = self.capacitance(V)
            dVdt = (I_ph - I_dark - I_shunt - I_load) / C_j
            return [dVdt]

        # Solve with Radau (implicit, good for stiff systems)
        sol = solve_ivp(
            ode_rhs,
            t_span=(t[0], t[-1]),
            y0=[V0],
            method='Radau',
            t_eval=t,
            rtol=1e-6,
            atol=1e-9,
            max_step=dt * 10,
        )

        if not sol.success:
            # Fallback to simple Euler if Radau fails
            V_cell = self._euler_fallback(t, P_rx, R_load, V0)
        else:
            V_cell = sol.y[0]

        # Compute derived quantities
        I_ph = self.R_lambda * P_rx
        I_dark = np.array([self.dark_current(v) for v in V_cell])
        I_shunt = V_cell / self.R_sh
        I_cell = V_cell / (R_load + self.R_s)

        return PVCellResult(
            t=t, V_cell=V_cell, I_cell=I_cell,
            I_ph=I_ph, I_dark=I_dark, I_shunt=I_shunt,
        )

    def _euler_fallback(self, t, P_rx, R_load, V0):
        """Simple forward Euler fallback if Radau fails."""
        dt = t[1] - t[0]
        V = np.zeros(len(t))
        V[0] = V0
        for i in range(1, len(t)):
            I_ph = self.R_lambda * P_rx[i - 1]
            I_dark = self.dark_current(V[i - 1])
            I_shunt = V[i - 1] / self.R_sh
            I_load = V[i - 1] / (R_load + self.R_s)
            C_j = self.capacitance(V[i - 1])
            dVdt = (I_ph - I_dark - I_shunt - I_load) / C_j
            V[i] = V[i - 1] + dVdt * dt
            V[i] = np.clip(V[i], -0.5, self.V_bi * 0.95)
        return V

    def steady_state_voltage(self, P_rx_W: float, R_load: float = 1.0) -> float:
        """Find DC operating point V for a given optical power."""
        from scipy.optimize import brentq
        I_ph = self.R_lambda * P_rx_W

        def residual(V):
            return I_ph - self.dark_current(V) - V / self.R_sh - V / (R_load + self.R_s)

        try:
            return brentq(residual, 0.0, self.V_bi * 0.95)
        except ValueError:
            return 0.0

    def __repr__(self) -> str:
        return (f"PVCellModel(R={self.R_lambda} A/W, Cj0={self.C_j0*1e9:.0f}nF, "
                f"Rsh={self.R_sh/1e3:.1f}k, Is={self.I_s:.1e})")
