# simulation/analysis.py
"""
Post-Processing Analysis for LiFi-PV Simulations

Provides tools for:
    - BER calculation from transient waveforms
    - Theoretical BER for OOK
    - Eye diagram generation
    - Frequency response analysis
    - Power harvesting efficiency

Usage:
    from simulation.analysis import calculate_ber_from_transient, eye_diagram_data

    ber = calculate_ber_from_transient(tx_bits, rx_waveform, time, threshold, bit_period)
    eye_t, eye_traces = eye_diagram_data(time, waveform, bit_period)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional


# =============================================================================
# BER ANALYSIS
# =============================================================================

def calculate_ber_from_transient(tx_bits: np.ndarray,
                                  rx_waveform: np.ndarray,
                                  time: np.ndarray,
                                  threshold: float,
                                  bit_period: float,
                                  sample_offset: float = 0.5,
                                  skip_bits: int = 5) -> Dict[str, float]:
    """
    Calculate BER by sampling RX waveform at bit centers and comparing.

    Args:
        tx_bits: Transmitted bit sequence (0s and 1s)
        rx_waveform: Received waveform from SPICE simulation
        time: Time array (seconds)
        threshold: Decision threshold voltage
        bit_period: Bit period in seconds
        sample_offset: Sampling point within bit (0.5 = center)
        skip_bits: Skip initial bits (allow settling)

    Returns:
        Dict with 'ber', 'n_errors', 'n_bits_tested', 'snr_est_dB'
    """
    n_bits = len(tx_bits)
    errors = 0
    bits_tested = 0

    rx_decisions = []
    sample_times = []

    for i in range(skip_bits, n_bits):
        # Sample time at bit center
        t_sample = (i + sample_offset) * bit_period

        # Find nearest time index
        idx = np.argmin(np.abs(time - t_sample))
        if idx >= len(rx_waveform):
            break

        # Decision: compare against threshold
        rx_bit = 1 if rx_waveform[idx] > threshold else 0
        rx_decisions.append(rx_bit)
        sample_times.append(t_sample)

        if rx_bit != tx_bits[i]:
            errors += 1
        bits_tested += 1

    ber = errors / bits_tested if bits_tested > 0 else 1.0

    # Estimate SNR from waveform
    # Separate samples at 1-bits and 0-bits
    v_ones = []
    v_zeros = []
    for i in range(skip_bits, min(n_bits, skip_bits + bits_tested)):
        t_sample = (i + sample_offset) * bit_period
        idx = np.argmin(np.abs(time - t_sample))
        if idx < len(rx_waveform):
            if tx_bits[i] == 1:
                v_ones.append(rx_waveform[idx])
            else:
                v_zeros.append(rx_waveform[idx])

    snr_db = float('nan')
    if v_ones and v_zeros:
        mu1 = np.mean(v_ones)
        mu0 = np.mean(v_zeros)
        sigma1 = np.std(v_ones) if len(v_ones) > 1 else 1e-10
        sigma0 = np.std(v_zeros) if len(v_zeros) > 1 else 1e-10
        sigma_avg = (sigma1 + sigma0) / 2
        if sigma_avg > 0:
            snr_db = 20 * np.log10(abs(mu1 - mu0) / (2 * sigma_avg))

    return {
        'ber': ber,
        'n_errors': errors,
        'n_bits_tested': bits_tested,
        'snr_est_dB': snr_db,
        'rx_decisions': np.array(rx_decisions),
    }


def theoretical_ber_ook(snr_db: float) -> float:
    """
    Theoretical BER for OOK with optimal threshold.

    BER = 0.5 * erfc(sqrt(SNR / 2))

    Args:
        snr_db: Signal-to-noise ratio in dB

    Returns:
        Bit error rate
    """
    from scipy.special import erfc

    snr_lin = 10 ** (snr_db / 10)
    return 0.5 * erfc(np.sqrt(snr_lin / 2))


def theoretical_ber_ook_simple(snr_db: float) -> float:
    """
    Approximate BER for OOK (no scipy dependency).

    Uses Q-function approximation: Q(x) ≈ exp(-x²/2) / (x * sqrt(2π))

    Args:
        snr_db: SNR in dB

    Returns:
        Approximate BER
    """
    snr_lin = 10 ** (snr_db / 10)
    x = np.sqrt(snr_lin / 2)
    if x > 0:
        return np.exp(-x**2 / 2) / (x * np.sqrt(2 * np.pi))
    return 0.5


# =============================================================================
# EYE DIAGRAM
# =============================================================================

def eye_diagram_data(time: np.ndarray,
                     waveform: np.ndarray,
                     bit_period: float,
                     n_ui: int = 2,
                     skip_initial: float = 0.0,
                     samples_per_ui: int = 100) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Generate eye diagram data from time-domain waveform.

    Slices waveform into segments of n_ui * bit_period and overlays them.
    Handles non-uniform time steps (e.g., LTspice .raw files) by
    interpolating each window onto a uniform grid.

    Args:
        time: Time array (s)
        waveform: Voltage waveform
        bit_period: Bit period (s)
        n_ui: Number of unit intervals per trace (default: 2)
        skip_initial: Skip this many seconds from start
        samples_per_ui: Interpolation samples per unit interval

    Returns:
        (normalized_time, traces) where:
            normalized_time: time within one eye window (0 to n_ui * bit_period)
            traces: list of waveform segments (each uniformly sampled)
    """
    window = n_ui * bit_period
    n_samples = n_ui * samples_per_ui

    if len(time) < 2 or window <= 0:
        return np.array([]), []

    # Normalized time axis for each eye trace
    t_norm = np.linspace(0, window, n_samples)

    # Skip initial settling time
    mask = time >= skip_initial
    time = time[mask]
    waveform = waveform[mask]

    if len(time) < 2:
        return t_norm, []

    t_start = time[0]
    t_end = time[-1]
    duration = t_end - t_start
    n_windows = int(duration / window)

    if n_windows < 1:
        return t_norm, []

    # Slice into windows and interpolate each onto uniform grid
    traces = []
    for i in range(n_windows):
        win_start = t_start + i * window
        win_end = win_start + window

        # Find indices within this window
        mask_win = (time >= win_start) & (time <= win_end)
        t_win = time[mask_win]
        v_win = waveform[mask_win]

        if len(t_win) < 2:
            continue

        # Interpolate onto uniform grid
        t_local = t_win - win_start
        trace = np.interp(t_norm, t_local, v_win)
        traces.append(trace)

    return t_norm, traces


# =============================================================================
# FREQUENCY RESPONSE
# =============================================================================

def frequency_response_from_ac(ac_results: Dict[str, np.ndarray],
                                node_name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract magnitude and phase from AC analysis results.

    Args:
        ac_results: Dict from NgSpiceRunner.run_ac()
        node_name: Node to analyze

    Returns:
        (frequency, magnitude_dB, phase_deg)
    """
    freq = ac_results.get('frequency', np.array([]))
    h = ac_results.get(node_name, np.array([]))

    if len(freq) == 0 or len(h) == 0:
        return np.array([]), np.array([]), np.array([])

    if np.iscomplexobj(h):
        mag_dB = 20 * np.log10(np.abs(h) + 1e-30)
        phase_deg = np.angle(h, deg=True)
    else:
        mag_dB = 20 * np.log10(np.abs(h) + 1e-30)
        phase_deg = np.zeros_like(h)

    return freq, mag_dB, phase_deg


def find_3dB_bandwidth(freq: np.ndarray, mag_dB: np.ndarray) -> float:
    """
    Find -3dB bandwidth from frequency response.

    Args:
        freq: Frequency array (Hz)
        mag_dB: Magnitude in dB

    Returns:
        -3dB frequency in Hz (or NaN if not found)
    """
    if len(freq) == 0:
        return float('nan')

    peak_dB = np.max(mag_dB)
    target = peak_dB - 3.0

    # Find crossing point (high frequency side)
    peak_idx = np.argmax(mag_dB)

    for i in range(peak_idx, len(mag_dB) - 1):
        if mag_dB[i] >= target and mag_dB[i+1] < target:
            # Linear interpolation
            f_3dB = freq[i] + (target - mag_dB[i]) / (mag_dB[i+1] - mag_dB[i]) * (freq[i+1] - freq[i])
            return f_3dB

    return float('nan')


# =============================================================================
# POWER HARVESTING
# =============================================================================

def calculate_harvested_power(time: np.ndarray,
                               v_dcdc: np.ndarray,
                               R_load: float,
                               settle_fraction: float = 0.5) -> Dict[str, float]:
    """
    Calculate harvested power from DC-DC output.

    Args:
        time: Time array
        v_dcdc: DC-DC output voltage waveform
        R_load: Load resistance (ohm)
        settle_fraction: Skip this fraction from start

    Returns:
        Dict with 'P_avg_uW', 'V_avg', 'V_rms', 'ripple_pp'
    """
    # Use second half for steady-state
    start_idx = int(len(time) * settle_fraction)
    v_ss = v_dcdc[start_idx:]

    V_avg = np.mean(v_ss)
    V_rms = np.sqrt(np.mean(v_ss**2))
    V_pp = np.max(v_ss) - np.min(v_ss)
    P_avg = V_rms**2 / R_load

    return {
        'P_avg_uW': P_avg * 1e6,
        'V_avg': V_avg,
        'V_rms': V_rms,
        'ripple_pp': V_pp,
        'efficiency_note': 'Compare against input power for efficiency',
    }


def calculate_noise_rms(time: np.ndarray,
                         waveform: np.ndarray,
                         settle_fraction: float = 0.5) -> float:
    """
    Calculate RMS noise voltage from waveform (assumes zero-mean AC signal).

    Args:
        time: Time array
        waveform: Voltage waveform
        settle_fraction: Skip initial fraction

    Returns:
        RMS noise in V
    """
    start_idx = int(len(time) * settle_fraction)
    v = waveform[start_idx:]
    v_dc = np.mean(v)
    v_ac = v - v_dc
    return np.sqrt(np.mean(v_ac**2))


# =============================================================================
# BER SWEEP FUNCTIONS
# =============================================================================

def ber_vs_parameter(parameter_values: list,
                     parameter_name: str,
                     run_simulation_fn,
                     **fixed_params) -> Dict[str, list]:
    """
    Generic BER sweep over a parameter.

    Args:
        parameter_values: List of parameter values to sweep
        parameter_name: Name of the parameter
        run_simulation_fn: Callable(param_value, **fixed_params) -> ber_result_dict
        **fixed_params: Fixed parameters

    Returns:
        Dict with 'parameter_values', 'ber', 'snr_dB'
    """
    bers = []
    snrs = []

    for val in parameter_values:
        print(f"  {parameter_name} = {val}...")
        result = run_simulation_fn(val, **fixed_params)
        bers.append(result.get('ber', 1.0))
        snrs.append(result.get('snr_est_dB', float('nan')))

    return {
        'parameter_name': parameter_name,
        'parameter_values': list(parameter_values),
        'ber': bers,
        'snr_dB': snrs,
    }


# =============================================================================
# SELF-TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("ANALYSIS MODULE - SELF TEST")
    print("=" * 60)

    # Test theoretical BER
    print("\n  Theoretical OOK BER:")
    for snr in [0, 5, 10, 15, 20]:
        ber = theoretical_ber_ook_simple(snr)
        print(f"    SNR = {snr:2d} dB -> BER = {ber:.4e}")

    # Test eye diagram data
    t = np.linspace(0, 1e-3, 10000)
    signal = np.sin(2 * np.pi * 5e3 * t)
    t_eye, traces = eye_diagram_data(t, signal, 1/5e3, n_ui=2)
    print(f"\n  Eye diagram: {len(traces)} traces, "
          f"{len(t_eye)} samples per trace")

    # Test noise RMS
    noise = np.random.randn(10000) * 1e-3  # 1 mV RMS noise
    rms = calculate_noise_rms(np.linspace(0, 1, 10000), noise, settle_fraction=0)
    print(f"\n  Noise RMS test: expected ~1.0 mV, got {rms*1e3:.2f} mV")

    print("\n[OK] Analysis tests passed!")
