# cosim/spice_extract.py
"""
SPICE Result Extraction & Alignment.

Extracts waveforms from SPICE .raw files, maps node names to standard keys,
resamples variable-timestep data to a uniform grid, and computes BER from
the comparator output.

Usage:
    from cosim.spice_extract import extract_spice_waveforms, compute_ber_from_spice

    waveforms = extract_spice_waveforms(raw_path)
    ber_result = compute_ber_from_spice(raw_path, bits_tx, data_rate_bps)
"""

import numpy as np
from pathlib import Path
from typing import Dict, Optional, List
from .raw_parser import LTSpiceRawParser


# ============================================================================
# SPICE node → standard key mapping
# ============================================================================

# Maps SPICE node names (case-insensitive) to standardized output keys.
# Multiple SPICE names may map to the same key (LTspice vs ngspice naming).
NODE_MAP = {
    # Solar cell
    'V(sc_anode)': 'V_cell',
    'V(sc_cathode)': 'V_cathode',
    # Sense resistor
    'V(sense_lo)': 'V_sense_lo',
    'I(Rsense)': 'I_sense',
    'I(rsense)': 'I_sense',
    # INA
    'V(ina_out)': 'V_ina',
    'V(ina_out_clean)': 'V_ina_clean',
    # BPF
    'V(bpf1_out)': 'V_bpf1',
    'V(bpf_out)': 'V_bpf2',
    # Comparator
    'V(dout)': 'V_comp',
    # DC-DC
    'V(dcdc_out)': 'V_dcdc',
    # Optical input
    'V(optical_power)': 'P_rx',
}


def extract_spice_waveforms(raw_path, node_map: Optional[Dict] = None) -> Dict:
    """
    Parse .raw file and return standardized waveform dict.

    Args:
        raw_path: Path to .raw file (LTspice or ngspice)
        node_map: Optional custom node name mapping (default: NODE_MAP)

    Returns:
        Dict with 'time' and standardized waveform keys.
        Only includes nodes that exist in the .raw file.
    """
    parser = LTSpiceRawParser(raw_path)
    available = parser.list_traces()
    mapping = node_map or NODE_MAP

    result = {
        'time': parser.get_time(),
        '_available_traces': available,
    }

    # Build case-insensitive lookup
    available_lower = {name.lower(): name for name in available}

    for spice_name, std_key in mapping.items():
        actual_name = available_lower.get(spice_name.lower())
        if actual_name:
            result[std_key] = parser.get_trace(actual_name)

    # Compute V_sense from circuit topology if not directly available
    # V_sense = V(sc_cathode) - V(sense_lo) = I_sense * R_sense
    if 'V_sense' not in result and 'I_sense' in result:
        # I_sense flows through R_sense, V_sense = I * R (R typically 1 ohm)
        result['V_sense'] = result['I_sense']

    return result


def resample_to_uniform(time: np.ndarray, signal: np.ndarray,
                        n_points: Optional[int] = None,
                        dt: Optional[float] = None) -> tuple:
    """
    Resample a variable-timestep signal to a uniform time grid.

    SPICE uses adaptive timestep, producing non-uniform time arrays.
    This resamples to uniform spacing for BER computation and comparison.

    Args:
        time: Non-uniform time array from SPICE
        signal: Signal values at each time point
        n_points: Number of output points (default: same as input)
        dt: Desired timestep (overrides n_points if given)

    Returns:
        (t_uniform, signal_uniform) tuple
    """
    t_start = time[0]
    t_end = time[-1]

    if dt is not None:
        n_points = int((t_end - t_start) / dt) + 1
    elif n_points is None:
        n_points = len(time)

    t_uniform = np.linspace(t_start, t_end, n_points)
    signal_uniform = np.interp(t_uniform, time, signal)
    return t_uniform, signal_uniform


def compute_ber_from_spice(raw_path, bits_tx: np.ndarray,
                           data_rate_bps: float,
                           vcc: float = 3.3,
                           skip_bits: int = 2,
                           sample_offset: float = 0.5) -> Dict:
    """
    Compute BER from SPICE comparator output V(dout).

    Extracts the digital output, resamples to uniform grid, samples at
    bit centers, and compares against transmitted bits.

    Args:
        raw_path: Path to .raw file
        bits_tx: Transmitted bit sequence
        data_rate_bps: Data rate in bits/second
        vcc: Supply voltage for threshold (threshold = vcc/2)
        skip_bits: Number of initial bits to skip (settling time)
        sample_offset: Sampling point within bit period (0.5 = center)

    Returns:
        Dict with 'ber', 'n_errors', 'n_bits_tested', 'bits_rx',
        'snr_est_dB', 'inverted'
    """
    waveforms = extract_spice_waveforms(raw_path)
    time = waveforms['time']

    # Get comparator output
    if 'V_comp' not in waveforms:
        available = waveforms.get('_available_traces', [])
        raise KeyError(f"V(dout) not found in .raw file. Available: {available}")

    v_comp = waveforms['V_comp']

    # Resample to uniform grid
    bit_period = 1.0 / data_rate_bps
    dt_target = bit_period / 100  # 100 samples per bit
    t_uniform, v_uniform = resample_to_uniform(time, v_comp, dt=dt_target)

    # Threshold at Vcc/2
    threshold = vcc / 2.0

    # Sample at bit centers
    t_max = t_uniform[-1]
    max_bits = int(t_max / bit_period)
    n_bits_available = min(max_bits, len(bits_tx))

    bits_rx = np.zeros(n_bits_available, dtype=int)
    for i in range(n_bits_available):
        t_sample = (i + sample_offset) * bit_period
        idx = int((t_sample - t_uniform[0]) / dt_target)
        idx = min(idx, len(v_uniform) - 1)
        bits_rx[i] = 1 if v_uniform[idx] > threshold else 0

    # Skip settling bits
    start = min(skip_bits, n_bits_available - 1)
    tx_clipped = bits_tx[start:n_bits_available]
    rx_clipped = bits_rx[start:]

    n_tested = len(tx_clipped)
    n_errors = int(np.sum(tx_clipped != rx_clipped))
    ber = n_errors / n_tested if n_tested > 0 else 1.0
    inverted = False

    # Check inverted polarity (BPF may invert)
    if ber > 0.4:
        rx_inv = 1 - rx_clipped
        n_errors_inv = int(np.sum(tx_clipped != rx_inv))
        ber_inv = n_errors_inv / n_tested if n_tested > 0 else 1.0
        if ber_inv < ber:
            ber = ber_inv
            n_errors = n_errors_inv
            bits_rx[start:] = 1 - bits_rx[start:]
            inverted = True

    # SNR estimate from BPF output if available
    snr_db = _estimate_snr(waveforms, bits_tx, data_rate_bps, vcc)

    return {
        'ber': ber,
        'n_errors': n_errors,
        'n_bits_tested': n_tested,
        'bits_rx': bits_rx,
        'snr_est_dB': snr_db,
        'inverted': inverted,
    }


def _estimate_snr(waveforms: Dict, bits_tx: np.ndarray,
                  data_rate_bps: float, vcc: float) -> float:
    """Estimate SNR from BPF output waveform."""
    # Try BPF output first (analog, pre-comparator)
    for key in ('V_bpf2', 'V_bpf1', 'V_ina'):
        if key in waveforms:
            signal = waveforms[key]
            time = waveforms['time']

            # Use second half to avoid transients
            mid = len(signal) // 2
            sig_half = signal[mid:]

            # Signal power = variance of the signal
            sig_power = np.var(sig_half)
            if sig_power <= 0:
                continue

            # Estimate noise: residual after removing ideal square wave
            # Simple approach: use variance around the mean of high/low levels
            threshold = np.mean(sig_half)
            high_samples = sig_half[sig_half > threshold]
            low_samples = sig_half[sig_half <= threshold]

            noise_var = 0.0
            if len(high_samples) > 10:
                noise_var += np.var(high_samples)
            if len(low_samples) > 10:
                noise_var += np.var(low_samples)
            noise_var /= 2  # Average of high/low noise

            if noise_var > 0:
                snr = sig_power / noise_var
                return float(10 * np.log10(max(snr, 1e-30)))

    return 0.0
