# simulation/prbs_generator.py
"""
PRBS and OOK Signal Generator for LiFi Simulation

Generates pseudo-random bit sequences (PRBS) and converts them to
time-domain OOK (On-Off Keying) waveforms suitable for SPICE simulation.

Usage:
    from simulation.prbs_generator import generate_prbs, generate_ook_waveform

    bits = generate_prbs(order=7, n_bits=1000)
    time, voltage = generate_ook_waveform(bits, bit_rate=5e3, mod_depth=0.33)
    write_pwl_file(time, voltage, 'ook_signal.pwl')
"""

import numpy as np
from typing import Tuple


# =============================================================================
# LFSR FEEDBACK MASKS for standard PRBS orders (Galois LFSR)
# =============================================================================

# Feedback taps (excluding the highest bit, which is implicit).
# These are the bit positions (1-indexed) where feedback is XORed in a Galois LFSR.
LFSR_TAPS = {
    7: [6],             # PRBS-7: x^7 + x^6 + 1 (ITU-T O.150)
    9: [5],             # PRBS-9: x^9 + x^5 + 1 (ITU-T O.150)
    11: [9],            # PRBS-11: x^11 + x^9 + 1
    15: [14],           # PRBS-15: x^15 + x^14 + 1 (ITU-T O.150)
    20: [17],           # PRBS-20: x^20 + x^17 + 1
    23: [18],           # PRBS-23: x^23 + x^18 + 1 (ITU-T O.150)
    31: [28],           # PRBS-31: x^31 + x^28 + 1 (ITU-T O.150)
}


def generate_prbs(order: int = 7, n_bits: int = 1000, seed: int = None) -> np.ndarray:
    """
    Generate pseudo-random bit sequence using Galois LFSR.

    Args:
        order: PRBS order (7, 9, 11, 15, 20, 23, 31)
        n_bits: Number of bits to generate
        seed: Random seed for initial state (None for all-ones)

    Returns:
        Binary array of shape (n_bits,) with values 0 or 1
    """
    if order not in LFSR_TAPS:
        raise ValueError(f"Unsupported PRBS order {order}. "
                        f"Available: {list(LFSR_TAPS.keys())}")

    taps = LFSR_TAPS[order]

    # Build feedback mask for Galois LFSR
    feedback_mask = 0
    for tap in taps:
        feedback_mask |= (1 << (tap - 1))

    # Initialize LFSR state
    if seed is not None:
        rng = np.random.RandomState(seed)
        state = rng.randint(1, 2**order)  # Non-zero initial state
    else:
        state = (1 << order) - 1  # All ones

    bits = np.zeros(n_bits, dtype=np.int8)

    for i in range(n_bits):
        # Output bit is LSB
        bits[i] = state & 1

        # Galois LFSR: if output bit is 1, XOR state with feedback mask
        fb = state & 1
        state >>= 1
        if fb:
            state ^= feedback_mask
            state |= (1 << (order - 1))

    return bits


def generate_ook_waveform(bits: np.ndarray,
                           bit_rate: float = 5e3,
                           samples_per_bit: int = 100,
                           modulation_depth: float = 0.33,
                           dc_level: float = 1.0,
                           rise_time_fraction: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert bit sequence to OOK time-domain waveform.

    OOK: bit=1 -> P_dc * (1 + m), bit=0 -> P_dc * (1 - m)
    where m = modulation_depth

    Args:
        bits: Binary array (0s and 1s)
        bit_rate: Bit rate in bits per second
        samples_per_bit: Time samples per bit period
        modulation_depth: Modulation depth (0 to 1)
        dc_level: DC bias level (represents average optical power)
        rise_time_fraction: Rise/fall time as fraction of bit period

    Returns:
        (time, voltage) tuple of numpy arrays
    """
    n_bits = len(bits)
    bit_period = 1.0 / bit_rate
    total_time = n_bits * bit_period
    n_samples = n_bits * samples_per_bit

    time = np.linspace(0, total_time, n_samples)
    voltage = np.zeros(n_samples)

    # High and low levels
    v_high = dc_level * (1 + modulation_depth)
    v_low = dc_level * (1 - modulation_depth)

    # Rise/fall time in samples
    rise_samples = max(1, int(rise_time_fraction * samples_per_bit))

    for i, bit in enumerate(bits):
        start = i * samples_per_bit
        end = (i + 1) * samples_per_bit
        target = v_high if bit == 1 else v_low

        voltage[start:end] = target

    # Apply smoothing for rise/fall times (simple moving average)
    if rise_samples > 1:
        kernel = np.ones(rise_samples) / rise_samples
        voltage = np.convolve(voltage, kernel, mode='same')

    return time, voltage


def generate_square_ook(bit_rate: float = 5e3,
                         n_bits: int = 100,
                         modulation_depth: float = 0.33,
                         dc_level: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate simple square-wave OOK for SPICE PWL source.

    Uses minimal points (2 per transition) for efficient SPICE PWL.

    Args:
        bit_rate: Bit rate (bps)
        n_bits: Number of bits
        modulation_depth: Modulation depth
        dc_level: DC level

    Returns:
        (time, voltage) for PWL source
    """
    bits = generate_prbs(order=7, n_bits=n_bits)
    bit_period = 1.0 / bit_rate
    rise_time = bit_period * 0.01  # 1% of bit period

    v_high = dc_level * (1 + modulation_depth)
    v_low = dc_level * (1 - modulation_depth)

    time_points = [0.0]
    voltage_points = [v_low if bits[0] == 0 else v_high]

    for i in range(1, len(bits)):
        if bits[i] != bits[i-1]:
            # Transition point
            t_trans = i * bit_period
            v_prev = v_high if bits[i-1] == 1 else v_low
            v_next = v_high if bits[i] == 1 else v_low

            time_points.append(t_trans - rise_time/2)
            voltage_points.append(v_prev)
            time_points.append(t_trans + rise_time/2)
            voltage_points.append(v_next)

    # Final point
    time_points.append(len(bits) * bit_period)
    voltage_points.append(v_high if bits[-1] == 1 else v_low)

    return np.array(time_points), np.array(voltage_points)


def write_pwl_file(time: np.ndarray, voltage: np.ndarray, filename: str):
    """
    Write time-voltage pairs as SPICE PWL file.

    Format: one pair per line, space-separated.

    Args:
        time: Time array (seconds)
        voltage: Voltage array (V)
        filename: Output file path
    """
    with open(filename, 'w') as f:
        for t, v in zip(time, voltage):
            f.write(f"{t:.10e} {v:.10e}\n")

    print(f"PWL file saved: {filename} ({len(time)} points)")


def bits_to_spice_pulse(bits: np.ndarray,
                         bit_rate: float,
                         v_high: float,
                         v_low: float) -> str:
    """
    Convert bit array to SPICE PWL source definition string.

    Args:
        bits: Binary array
        bit_rate: Bit rate (bps)
        v_high: High voltage level
        v_low: Low voltage level

    Returns:
        SPICE PWL source string (without "Vname node1 node2" prefix)
    """
    bit_period = 1.0 / bit_rate
    rise_time = bit_period * 0.01

    pairs = []
    for i, bit in enumerate(bits):
        t = i * bit_period
        v = v_high if bit == 1 else v_low
        if i > 0 and bits[i] != bits[i-1]:
            pairs.append(f"{t - rise_time:.6e} {v_high if bits[i-1]==1 else v_low:.6e}")
        pairs.append(f"{t + rise_time:.6e} {v:.6e}")

    return "PWL(" + " ".join(pairs) + ")"


# =============================================================================
# SELF-TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("PRBS GENERATOR - SELF TEST")
    print("=" * 60)

    # Test PRBS generation
    for order in [7, 9, 15]:
        bits = generate_prbs(order=order, n_bits=1000)
        ones = np.sum(bits)
        print(f"  PRBS-{order}: {len(bits)} bits, "
              f"{ones} ones ({ones/len(bits)*100:.1f}%), "
              f"{len(bits)-ones} zeros")

    # Test OOK waveform
    bits = generate_prbs(order=7, n_bits=20)
    time, voltage = generate_ook_waveform(bits, bit_rate=5e3, modulation_depth=0.33)
    print(f"\n  OOK waveform: {len(time)} samples, "
          f"duration = {time[-1]*1e3:.2f} ms")
    print(f"  V range: [{voltage.min():.3f}, {voltage.max():.3f}]")

    # Test PWL file
    write_pwl_file(time, voltage, 'test_ook.pwl')

    # Test square OOK
    t_sq, v_sq = generate_square_ook(bit_rate=5e3, n_bits=50)
    print(f"\n  Square OOK: {len(t_sq)} PWL points for 50 bits")

    print("\n[OK] PRBS generator tests passed!")
