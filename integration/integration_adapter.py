# =============================================================================
# integration/adapter.py — Component-to-Simulator Adapter
# =============================================================================
# Task 11 of Hardware-Faithful Simulator
#
# This is the KEY BRIDGE between Tier 2 (component library) and the
# existing behavioral simulator. It takes component objects and produces
# the exact config dicts that PVReceiver, Transmitter, and Channel expect.
#
# The adapter also handles cross-component interactions:
#   - LED emission × PV spectral response → effective responsivity
#   - LED power + distance + pattern → received optical power
#   - Complete link budget from component selection
# =============================================================================

import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from components.base import (
    get_component, SolarCell, PhotodetectorComponent, LEDComponent,
)
from materials.reference_data import GAAS_QE_CURVE, BPW34_SPECTRAL_RESPONSE
from physics.photodetection import (
    responsivity, quantum_efficiency_from_curve, effective_responsivity,
)
from physics.led_emission import irradiance_at_distance


# =============================================================================
# SPECTRAL MATCHING: LED × PV
# =============================================================================

def compute_effective_responsivity(led: LEDComponent,
                                    receiver: PhotodetectorComponent,
                                    I_drive_A: float = 0.350,
                                    T: float = 300.0) -> float:
    """
    Effective responsivity accounting for LED-PV spectral overlap.

    R_eff = ∫ R_PV(λ) · S_LED(λ) dλ / ∫ S_LED(λ) dλ

    This is more accurate than using R at the LED peak wavelength alone,
    because the LED has finite spectral width and the PV responsivity
    varies across that width.

    Args:
        led: LED component instance
        receiver: Photodetector component instance
        I_drive_A: LED drive current (affects spectrum via T_j)
        T: Ambient temperature

    Returns:
        R_eff in A/W
    """
    # Get LED spectrum
    spec = led.emission_spectrum(I_drive_A, T)
    wl = spec['wavelength_nm']
    s_led = spec['power_normalized']

    # Get PV responsivity at each LED wavelength
    r_at_wl = np.array([receiver.responsivity(w, T) for w in wl])

    # Weighted average
    numerator = np.trapezoid(r_at_wl * s_led, wl)
    denominator = np.trapezoid(s_led, wl)

    if denominator == 0:
        return receiver.responsivity(led.peak_wavelength(T), T)

    return numerator / denominator


# =============================================================================
# RECEIVED POWER CALCULATION
# =============================================================================

def compute_received_power(led: LEDComponent,
                            receiver: PhotodetectorComponent,
                            distance_m: float,
                            I_drive_A: float = 0.350,
                            theta_deg: float = 0.0,
                            T: float = 300.0) -> float:
    """
    Optical power received by the photodetector.

    P_rx = E(d, θ) · A_rx · cos(ψ)

    Where E is irradiance from the LED and A_rx is receiver area.
    Assumes on-axis alignment (ψ = 0).

    Args:
        led: LED component
        receiver: Photodetector component
        distance_m: TX-RX distance in meters
        I_drive_A: LED drive current
        theta_deg: Off-axis angle
        T: Temperature

    Returns:
        P_rx in Watts
    """
    P_opt = led.optical_power(I_drive_A, T)
    m = led.lambertian_order()
    E = irradiance_at_distance(P_opt, distance_m, m, theta_deg)

    # Receiver area
    if hasattr(receiver, 'active_area_cm2'):
        A_rx = receiver.active_area_cm2() * 1e-4  # cm² → m²
    else:
        ds = receiver._ds
        A_rx = ds.get('active_area_cm2', ds.get('active_area_mm2', 7.5) * 1e-2) * 1e-4

    return E * A_rx


# =============================================================================
# ADAPTER: Component → Receiver Config
# =============================================================================

def component_to_receiver_config(receiver_part: str,
                                  led_part: str = None,
                                  distance_m: float = 0.325,
                                  R_load: float = 1000.0,
                                  I_drive_A: float = 0.350,
                                  T: float = 300.0) -> dict:
    """
    Generate complete receiver config from component part numbers.

    This is the MAIN ENTRY POINT for the adapter. Specify part numbers,
    get a config dict that drops into the existing simulator.

    Args:
        receiver_part: PV/photodiode part number (e.g. 'KXOB25-04X3F')
        led_part: LED part number (e.g. 'OSRAM_LRW5SN'), optional
        distance_m: TX-RX distance
        R_load: Load resistance in Ohms
        I_drive_A: LED drive current
        T: Temperature in K

    Returns:
        Config dict compatible with PVReceiver.__init__()
    """
    rx = get_component(receiver_part)
    cfg = rx.to_receiver_config(T)

    # Override bandwidth with actual R_load
    cfg['bandwidth'] = rx.bandwidth(R_load, T)

    # If LED specified, compute effective responsivity from spectral overlap
    if led_part:
        led = get_component(led_part)
        R_eff = compute_effective_responsivity(led, rx, I_drive_A, T)
        cfg['responsivity_effective'] = R_eff
        cfg['led_peak_wavelength_nm'] = led.peak_wavelength(T)

    cfg['R_load'] = R_load
    cfg['temperature_K'] = T
    cfg['part_number'] = receiver_part

    return cfg


def component_to_transmitter_config(led_part: str,
                                     I_drive_A: float = 0.350,
                                     T: float = 300.0) -> dict:
    """
    Generate transmitter config from LED part number.

    Args:
        led_part: LED part number
        I_drive_A: Drive current
        T: Temperature

    Returns:
        Config dict compatible with Transmitter class
    """
    led = get_component(led_part)
    cfg = led.to_transmitter_config(T)
    cfg['drive_current'] = I_drive_A
    cfg['optical_power'] = led.optical_power(I_drive_A, T)
    cfg['part_number'] = led_part
    cfg['temperature_K'] = T
    return cfg


def component_to_channel_config(led_part: str, receiver_part: str,
                                 distance_m: float = 0.325,
                                 I_drive_A: float = 0.350,
                                 T: float = 300.0) -> dict:
    """
    Generate channel config from LED + receiver + geometry.

    Computes the complete link budget:
    LED power → channel loss → received power → photocurrent

    Returns:
        Channel config dict with link budget
    """
    led = get_component(led_part)
    rx = get_component(receiver_part)

    P_tx = led.optical_power(I_drive_A, T)
    P_rx = compute_received_power(led, rx, distance_m, I_drive_A, 0, T)
    R_eff = compute_effective_responsivity(led, rx, I_drive_A, T)
    I_ph = R_eff * P_rx

    channel_gain = P_rx / P_tx if P_tx > 0 else 0

    return {
        'distance_m': distance_m,
        'P_tx_W': P_tx,
        'P_rx_W': P_rx,
        'channel_gain': channel_gain,
        'channel_gain_dB': 10 * np.log10(channel_gain) if channel_gain > 0 else -np.inf,
        'R_effective_A_W': R_eff,
        'I_photocurrent_A': I_ph,
        'led_part': led_part,
        'receiver_part': receiver_part,
        'temperature_K': T,
    }


# =============================================================================
# LINK BUDGET REPORT
# =============================================================================

def link_budget_report(led_part: str, receiver_part: str,
                       distance_m: float = 0.325,
                       R_load: float = 1000.0,
                       I_drive_A: float = 0.350,
                       T: float = 300.0) -> dict:
    """
    Complete link budget combining all components.

    Returns a comprehensive dict with TX, channel, and RX parameters.
    """
    tx_cfg = component_to_transmitter_config(led_part, I_drive_A, T)
    rx_cfg = component_to_receiver_config(receiver_part, led_part,
                                           distance_m, R_load, I_drive_A, T)
    ch_cfg = component_to_channel_config(led_part, receiver_part,
                                          distance_m, I_drive_A, T)

    rx = get_component(receiver_part)
    C_j = rx.junction_capacitance(0, T)
    f_3dB = rx.bandwidth(R_load, T)

    return {
        'transmitter': tx_cfg,
        'channel': ch_cfg,
        'receiver': rx_cfg,
        'summary': {
            'P_tx_mW': tx_cfg['optical_power'] * 1e3,
            'P_rx_uW': ch_cfg['P_rx_W'] * 1e6,
            'channel_loss_dB': -ch_cfg['channel_gain_dB'],
            'R_eff_A_W': ch_cfg['R_effective_A_W'],
            'I_ph_uA': ch_cfg['I_photocurrent_A'] * 1e6,
            'C_j_pF': C_j * 1e12,
            'f_3dB_kHz': f_3dB / 1e3,
            'distance_m': distance_m,
        },
    }


# =============================================================================
# SELF-TEST
# =============================================================================

def test_adapter():
    print("=" * 70)
    print("COMPONENT ADAPTER — TEST SUITE")
    print("=" * 70)
    passes = 0
    fails = 0

    def check(name, val, expected, tol_pct=10.0, unit=""):
        nonlocal passes, fails
        err = abs(val - expected) / abs(expected) * 100 if expected else 0
        status = "PASS" if err < tol_pct else "FAIL"
        if status == "FAIL":
            fails += 1
        else:
            passes += 1
        print(f"  {name}: {val:.4g} {unit} (expected ~{expected:.4g}, err={err:.1f}%) [{status}]")

    # --- Effective responsivity ---
    print("\n1. SPECTRAL MATCHING: OSRAM LED × KXOB25 PV")
    led = get_component('OSRAM_LRW5SN')
    kxob = get_component('KXOB25-04X3F')
    R_eff = compute_effective_responsivity(led, kxob, 0.350)
    check("R_eff(OSRAM×KXOB25)", R_eff, 0.445, tol_pct=10, unit="A/W")
    # Should be close to R(625nm) since LED spectrum is narrow

    # With BPW34 — should be lower (Si has lower QE at 625nm)
    bpw = get_component('BPW34')
    R_eff_bpw = compute_effective_responsivity(led, bpw, 0.350)
    print(f"  R_eff(OSRAM×BPW34) = {R_eff_bpw:.4f} A/W (should be < KXOB25)")
    assert R_eff_bpw < R_eff, "BPW34 should have lower R at red wavelengths"
    passes += 1

    # --- Received power ---
    print("\n2. RECEIVED POWER (Kadirvelu setup)")
    P_rx = compute_received_power(led, kxob, 0.325, 0.350)
    P_rx_uW = P_rx * 1e6
    print(f"  P_rx(32.5cm) = {P_rx_uW:.1f} µW")
    assert P_rx_uW > 10, "Should receive measurable power"
    passes += 1

    # --- Receiver config ---
    print("\n3. RECEIVER CONFIG GENERATION")
    rx_cfg = component_to_receiver_config('KXOB25-04X3F', 'OSRAM_LRW5SN',
                                           distance_m=0.325, R_load=1000)
    print(f"  Keys: {sorted(rx_cfg.keys())}")
    assert 'responsivity' in rx_cfg
    assert 'responsivity_effective' in rx_cfg
    assert 'capacitance' in rx_cfg
    check("R_eff in config", rx_cfg['responsivity_effective'], 0.445, tol_pct=10, unit="A/W")

    # --- Transmitter config ---
    print("\n4. TRANSMITTER CONFIG GENERATION")
    tx_cfg = component_to_transmitter_config('OSRAM_LRW5SN', 0.350)
    check("P_opt", tx_cfg['optical_power'] * 1e3, 110, tol_pct=5, unit="mW")

    # --- Channel config ---
    print("\n5. CHANNEL CONFIG / LINK BUDGET")
    ch = component_to_channel_config('OSRAM_LRW5SN', 'KXOB25-04X3F', 0.325)
    print(f"  P_tx = {ch['P_tx_W']*1e3:.1f} mW")
    print(f"  P_rx = {ch['P_rx_W']*1e6:.1f} µW")
    print(f"  Channel gain = {ch['channel_gain_dB']:.1f} dB")
    print(f"  R_eff = {ch['R_effective_A_W']:.4f} A/W")
    print(f"  I_ph = {ch['I_photocurrent_A']*1e6:.2f} µA")
    assert ch['channel_gain_dB'] < 0, "Channel should have loss"
    passes += 1

    # --- Full link budget ---
    print("\n6. FULL LINK BUDGET REPORT")
    lb = link_budget_report('OSRAM_LRW5SN', 'KXOB25-04X3F', 0.325, 1000)
    s = lb['summary']
    print(f"  TX Power:     {s['P_tx_mW']:.1f} mW")
    print(f"  RX Power:     {s['P_rx_uW']:.1f} µW")
    print(f"  Channel Loss: {s['channel_loss_dB']:.1f} dB")
    print(f"  R_eff:        {s['R_eff_A_W']:.4f} A/W")
    print(f"  I_ph:         {s['I_ph_uA']:.2f} µA")
    print(f"  C_j:          {s['C_j_pF']:.0f} pF")
    print(f"  f_3dB:        {s['f_3dB_kHz']:.0f} kHz")
    passes += 1

    # --- Swappability via adapter ---
    print("\n7. COMPONENT SWAPPABILITY VIA ADAPTER")
    configs = {}
    for rx_part in ['KXOB25-04X3F', 'SM141K04LV', 'BPW34']:
        ch = component_to_channel_config('OSRAM_LRW5SN', rx_part, 0.325)
        configs[rx_part] = ch
        print(f"  {rx_part:15s}: R_eff={ch['R_effective_A_W']:.4f} A/W, "
              f"I_ph={ch['I_photocurrent_A']*1e6:.2f} µA")

    # All should produce different results
    i_phs = [c['I_photocurrent_A'] for c in configs.values()]
    assert len(set([f"{x:.6f}" for x in i_phs])) == 3, "All should differ"
    print("  All 3 receivers produce different photocurrents: ✓")
    passes += 1

    print(f"\n{'=' * 70}")
    print(f"RESULTS: {passes} passed, {fails} failed")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    test_adapter()
