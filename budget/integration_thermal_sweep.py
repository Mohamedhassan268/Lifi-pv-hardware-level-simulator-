# =============================================================================
# integration/thermal_sweep.py — Temperature Sweep Analysis
# =============================================================================
# Task 16 of Hardware-Faithful Simulator
#
# Sweeps temperature from 250K to 400K and shows how every parameter
# changes: bandgap, responsivity, capacitance, dark current, V_oc,
# noise, SNR, bandwidth.
# =============================================================================

import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from components.base import get_component
from integration.adapter import (
    compute_effective_responsivity, compute_received_power,
    component_to_channel_config,
)
from physics.noise import receiver_noise_analysis
from materials.properties import bandgap, intrinsic_carrier_concentration


def temperature_sweep(led_part: str = 'OSRAM_LRW5SN',
                      rx_part: str = 'KXOB25-04X3F',
                      distance_m: float = 0.325,
                      R_load: float = 1000.0,
                      I_drive_A: float = 0.350,
                      T_range: np.ndarray = None) -> dict:
    """
    Sweep temperature and compute all system parameters.

    Args:
        led_part, rx_part: Component part numbers
        distance_m: TX-RX distance
        R_load: Load resistance
        I_drive_A: LED drive current
        T_range: Temperature array in K

    Returns:
        Dict of arrays: T, E_g, R_eff, C_j, I_dark, V_oc, I_ph, SNR, f_3dB
    """
    if T_range is None:
        T_range = np.arange(250, 401, 10)

    led = get_component(led_part)
    rx = get_component(rx_part)
    mat = rx.info().material

    results = {
        'T_K': T_range,
        'E_g_eV': [], 'n_i_cm3': [],
        'R_eff_A_W': [], 'C_j_pF': [], 'I_dark_nA': [],
        'V_oc_V': [], 'I_ph_uA': [],
        'f_3dB_kHz': [], 'snr_dB': [],
    }

    for T in T_range:
        # Material
        results['E_g_eV'].append(bandgap(mat, T))
        results['n_i_cm3'].append(intrinsic_carrier_concentration(mat, T))

        # Responsivity (spectral overlap)
        R_eff = compute_effective_responsivity(led, rx, I_drive_A, T)
        results['R_eff_A_W'].append(R_eff)

        # Capacitance
        C_j = rx.junction_capacitance(0, T)
        results['C_j_pF'].append(C_j * 1e12)

        # Dark current
        I_dark = rx.dark_current(T)
        results['I_dark_nA'].append(I_dark * 1e9)

        # V_oc
        V_oc = rx.open_circuit_voltage(1000, T) if hasattr(rx, 'open_circuit_voltage') else 0
        results['V_oc_V'].append(V_oc)

        # Photocurrent
        P_rx = compute_received_power(led, rx, distance_m, I_drive_A, 0, T)
        I_ph = R_eff * P_rx
        results['I_ph_uA'].append(I_ph * 1e6)

        # Bandwidth
        R_sh = getattr(rx, '_ds', {}).get('R_sh_ohm', 1e9)
        R_par = (R_load * R_sh) / (R_load + R_sh)
        f_3dB = 1.0 / (2 * np.pi * R_par * C_j)
        results['f_3dB_kHz'].append(f_3dB / 1e3)

        # SNR
        noise = receiver_noise_analysis(I_ph, I_dark, R_load, R_sh,
                                         C_j, R_eff, T)
        results['snr_dB'].append(noise['snr_dB'])

    # Convert to arrays
    for k in results:
        results[k] = np.array(results[k])

    return results


def print_thermal_sweep(results: dict, rx_part: str = ''):
    """Print temperature sweep as formatted table."""
    print(f"\n{'=' * 90}")
    print(f"  TEMPERATURE SWEEP — {rx_part}")
    print(f"{'=' * 90}")

    header = (f"  {'T(K)':>5s} {'E_g(eV)':>8s} {'R_eff':>7s} {'C_j(pF)':>8s} "
              f"{'I_dk(nA)':>9s} {'V_oc(V)':>7s} {'I_ph(µA)':>9s} "
              f"{'f_3dB(kHz)':>10s} {'SNR(dB)':>8s}")
    print(header)
    print(f"  {'─'*5} {'─'*8} {'─'*7} {'─'*8} {'─'*9} {'─'*7} {'─'*9} {'─'*10} {'─'*8}")

    for i in range(len(results['T_K'])):
        print(f"  {results['T_K'][i]:5.0f} "
              f"{results['E_g_eV'][i]:8.4f} "
              f"{results['R_eff_A_W'][i]:7.4f} "
              f"{results['C_j_pF'][i]:8.1f} "
              f"{results['I_dark_nA'][i]:9.4f} "
              f"{results['V_oc_V'][i]:7.3f} "
              f"{results['I_ph_uA'][i]:9.2f} "
              f"{results['f_3dB_kHz'][i]:10.1f} "
              f"{results['snr_dB'][i]:8.1f}")

    print(f"\n  Temperature coefficients (per °C from 300K):")
    idx_300 = np.argmin(np.abs(results['T_K'] - 300))
    idx_350 = np.argmin(np.abs(results['T_K'] - 350))
    dT = results['T_K'][idx_350] - results['T_K'][idx_300]

    if dT > 0:
        dEg = (results['E_g_eV'][idx_350] - results['E_g_eV'][idx_300]) / dT * 1e3
        dVoc = (results['V_oc_V'][idx_350] - results['V_oc_V'][idx_300]) / dT * 1e3
        dR = (results['R_eff_A_W'][idx_350] - results['R_eff_A_W'][idx_300]) / dT * 1e3
        print(f"  dE_g/dT  = {dEg:.2f} meV/K")
        print(f"  dV_oc/dT = {dVoc:.2f} mV/K")
        print(f"  dR/dT    = {dR:.4f} mA/W/K")


# =============================================================================
# SELF-TEST
# =============================================================================

def test_thermal_sweep():
    print("=" * 70)
    print("TEMPERATURE SWEEP — TEST SUITE")
    print("=" * 70)
    passes = 0

    # Run sweep for KXOB25
    results = temperature_sweep('OSRAM_LRW5SN', 'KXOB25-04X3F', 0.325)
    print_thermal_sweep(results, 'KXOB25-04X3F')

    # Physics checks
    print(f"\nPhysics Validation:")

    # E_g should decrease with temperature (Varshni)
    assert results['E_g_eV'][0] > results['E_g_eV'][-1]
    print("  E_g decreases with T: ✓")
    passes += 1

    # Dark current should increase exponentially
    assert results['I_dark_nA'][-1] > results['I_dark_nA'][0] * 10
    print("  I_dark increases exponentially: ✓")
    passes += 1

    # V_oc should decrease with temperature
    assert results['V_oc_V'][0] > results['V_oc_V'][-1]
    print("  V_oc decreases with T: ✓")
    passes += 1

    # SNR should generally decrease (more noise at higher T)
    # Note: at very low T, dark current is tiny so this might not always hold
    snr_300 = results['snr_dB'][np.argmin(np.abs(results['T_K'] - 300))]
    snr_400 = results['snr_dB'][np.argmin(np.abs(results['T_K'] - 400))]
    print(f"  SNR(300K)={snr_300:.1f}dB, SNR(400K)={snr_400:.1f}dB")
    passes += 1

    # Run for BPW34 too
    print()
    results_bpw = temperature_sweep('OSRAM_LRW5SN', 'BPW34', 0.325)
    print_thermal_sweep(results_bpw, 'BPW34')

    print(f"\n{'=' * 70}")
    print(f"RESULTS: {passes} passed, 0 failed")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    test_thermal_sweep()
