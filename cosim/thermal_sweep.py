# cosim/thermal_sweep.py
"""
Thermal sweep: run the unified pipeline at multiple temperatures and
report how dark current, responsivity, SNR, and BER move with temperature.

The production hardware (Cairo deployment, July 2026 roadmap) operates
over roughly -10C to 65C ambient; silicon junctions can reach 85C+ in
direct sun. This sweep surfaces the operating margin before a PCB is
committed.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from cosim.system_config import SystemConfig
from cosim.python_engine import run_python_simulation
from cosim.noise import NoiseModel


@dataclass
class ThermalPoint:
    temperature_C: float
    temperature_K: float
    dark_current_A: float
    responsivity_A_per_W: float
    I_ph_avg_uA: float
    snr_dB: float
    ber: float


def run_thermal_sweep(preset: str, temps_C: List[float],
                      output_dir: Optional[str] = None,
                      verbose: bool = True) -> List[ThermalPoint]:
    """Run the preset's pipeline at each temperature and collect metrics.

    Args:
        preset: Preset name (e.g. 'kadirvelu2021')
        temps_C: Temperatures in degrees Celsius
        output_dir: If set, save figures here

    Returns:
        List of ThermalPoint, one per temperature
    """
    points: List[ThermalPoint] = []
    for T_C in temps_C:
        T_K = T_C + 273.15
        cfg = SystemConfig.from_preset(preset)
        cfg.temperature_K = T_K

        nm = NoiseModel.from_config(cfg)
        I_dark = nm.dark_current_at_T()

        res = run_python_simulation(cfg)
        # Responsivity at T: recover from I_ph_avg / P_rx_avg
        P_rx_avg_W = res.get('P_rx_avg_uW', 0.0) * 1e-6
        I_ph_avg_A = res.get('I_ph_avg_uA', 0.0) * 1e-6
        R_eff = I_ph_avg_A / P_rx_avg_W if P_rx_avg_W > 0 else 0.0

        points.append(ThermalPoint(
            temperature_C=T_C,
            temperature_K=T_K,
            dark_current_A=I_dark,
            responsivity_A_per_W=R_eff,
            I_ph_avg_uA=res.get('I_ph_avg_uA', 0.0),
            snr_dB=res.get('snr_est_dB', 0.0),
            ber=res.get('ber', 1.0),
        ))

        if verbose:
            print(f"  T = {T_C:6.1f} C ({T_K:6.1f} K)  "
                  f"I_dark = {I_dark:.2e} A  "
                  f"R = {R_eff:.4f} A/W  "
                  f"SNR = {res.get('snr_est_dB', 0):6.2f} dB  "
                  f"BER = {res.get('ber', 1.0):.3e}")

    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        _plot_thermal_sweep(points, preset, out / f'thermal_sweep_{preset}.png')

    return points


def _plot_thermal_sweep(points: List[ThermalPoint], preset: str, path: Path):
    T = np.array([p.temperature_C for p in points])
    I_dark = np.array([p.dark_current_A for p in points])
    R = np.array([p.responsivity_A_per_W for p in points])
    snr = np.array([p.snr_dB for p in points])
    ber = np.array([max(p.ber, 1e-6) for p in points])

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    fig.suptitle(f'Thermal sweep: {preset}')

    axes[0, 0].semilogy(T, I_dark, 'o-', color='crimson')
    axes[0, 0].set_xlabel('Temperature (C)')
    axes[0, 0].set_ylabel('Dark current (A)')
    axes[0, 0].set_title('I_dark(T)')
    axes[0, 0].grid(True, which='both', alpha=0.3)

    axes[0, 1].plot(T, R, 's-', color='steelblue')
    axes[0, 1].set_xlabel('Temperature (C)')
    axes[0, 1].set_ylabel('Responsivity (A/W)')
    axes[0, 1].set_title('R(T)')
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(T, snr, '^-', color='darkgreen')
    axes[1, 0].set_xlabel('Temperature (C)')
    axes[1, 0].set_ylabel('SNR (dB)')
    axes[1, 0].set_title('SNR(T)')
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].semilogy(T, ber, 'd-', color='purple')
    axes[1, 1].set_xlabel('Temperature (C)')
    axes[1, 1].set_ylabel('BER')
    axes[1, 1].set_title('BER(T)')
    axes[1, 1].grid(True, which='both', alpha=0.3)

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
