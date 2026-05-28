#!/usr/bin/env python3
# demo.py
"""
Hardware-Faithful LiFi-PV Simulator - Demonstration Script

This script demonstrates the core concept: selecting real hardware components
and having their electrical parameters EMERGE from physics/datasheets.

Run this script to verify the installation and see the component library in action.
"""

import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    print("=" * 70)
    print("  HARDWARE-FAITHFUL LiFi-PV SIMULATOR")
    print("  Component Library Demonstration")
    print("=" * 70)
    
    # =========================================================================
    # 1. MATERIALS DATABASE
    # =========================================================================
    print("\n" + "─" * 70)
    print("1️⃣  MATERIALS DATABASE")
    print("─" * 70)
    
    from materials import get_material, GAAS, SILICON, InGaN
    
    print("\nAvailable semiconductor materials with temperature-dependent properties:")
    
    for name in ['Si', 'GaAs', 'GaN', 'InGaN_Blue']:
        mat = get_material(name)
        Eg = mat.bandgap(300)
        ni = mat.intrinsic_concentration(300)
        mu_e = mat.electron_mobility(300)
        print(f"  {mat.name:30} Eg={Eg:.3f}eV  n_i={ni:.2e}cm⁻³  μ_e={mu_e:.0f}cm²/Vs")
    
    # Temperature dependence
    print("\n  GaAs bandgap temperature dependence:")
    gaas = get_material('GaAs')
    for T in [250, 300, 350, 400]:
        print(f"    T={T}K: Eg={gaas.bandgap(T):.4f}eV")
    
    # InGaN composition tuning
    print("\n  InGaN bandgap tuning (for LED wavelength selection):")
    for x in [0.0, 0.15, 0.25, 0.40]:
        mat = InGaN(x)
        print(f"    In={x*100:.0f}%: Eg={mat.bandgap(300):.2f}eV → λ={mat.peak_wavelength(300):.0f}nm")
    
    # =========================================================================
    # 2. COMPONENT SELECTION - PARAMETERS EMERGE
    # =========================================================================
    print("\n" + "─" * 70)
    print("2️⃣  COMPONENT SELECTION → PARAMETERS EMERGE")
    print("─" * 70)
    
    from components import KXOB25_04X3F, BPW34, SFH206K, VEMD5510
    
    # --- Solar Cells ---
    print("\n📦 KXOB25-04X3F (GaAs Solar Cell - Primary Target)")
    print("   Just select the part number, all parameters derived automatically:")
    
    kxob = KXOB25_04X3F()
    params = kxob.get_parameters()
    
    print(f"\n   ✓ Responsivity:     {params['responsivity']:.3f} A/W  (from QE × λ/1240)")
    print(f"   ✓ Capacitance:      {params['capacitance']*1e12:.0f} pF  (from datasheet)")
    print(f"   ✓ Dark current:     {params['dark_current']*1e9:.1f} nA  (temperature-dependent)")
    print(f"   ✓ Shunt resistance: {params['shunt_resistance']:.1f} Ω")
    print(f"   ✓ Voc:              {params['open_circuit_voltage']:.2f} V")
    print(f"   ✓ Isc:              {params['short_circuit_current']*1e3:.1f} mA")
    
    print(f"\n   Bandwidth EMERGES from R×C:")
    for R in [100, 220, 1000, 10000]:
        bw = kxob.bandwidth(R)
        print(f"     R={R:5}Ω → BW = {bw/1e3:8.1f} kHz")
    
    # --- Photodiodes ---
    print("\n📦 Photodiode Comparison:")
    print("   " + "-" * 55)
    print(f"   {'Component':<12} {'Area':>8} {'R (A/W)':>8} {'C (pF)':>8} {'BW@1kΩ':>12}")
    print("   " + "-" * 55)
    
    for PD in [BPW34, SFH206K, VEMD5510]:
        pd = PD()
        p = pd.get_parameters()
        bw = pd.bandwidth(1000)  # Calculate directly
        print(f"   {pd.name:<12} {p['active_area_m2']*1e6:>7.2f}mm² {p['responsivity']:>8.2f} "
              f"{p['capacitance']*1e12:>8.1f} {bw/1e6:>10.2f}MHz")
    
    # =========================================================================
    # 3. PHYSICS-BASED DERIVATION
    # =========================================================================
    print("\n" + "─" * 70)
    print("3️⃣  PHYSICS-BASED PARAMETER DERIVATION")
    print("─" * 70)
    
    from components import GenericGaAsPV
    
    print("\n  Creating GaAs PV cell from physics (9cm², N_d=1e17, N_a=1e15):")
    generic = GenericGaAsPV(area_cm2=9.0, N_d=1e17, N_a=1e15)
    gp = generic.get_parameters()
    
    print(f"\n  Derived from semiconductor physics:")
    print(f"    Built-in voltage:    {gp['built_in_voltage']:.3f} V  (from V_bi = kT/q × ln(N_a×N_d/n_i²))")
    print(f"    Depletion width:     {gp['depletion_width_um']:.2f} µm  (from W = √(2ε(V_bi)/q × ...))")
    print(f"    Capacitance:         {gp['capacitance']*1e12:.0f} pF  (from C = ε×A/W)")
    print(f"    Responsivity:        {gp['responsivity']:.3f} A/W  (from QE × λ/1240)")
    
    # =========================================================================
    # 4. SYSTEM SIMULATION PREVIEW
    # =========================================================================
    print("\n" + "─" * 70)
    print("4️⃣  SYSTEM SIMULATION PREVIEW")
    print("─" * 70)
    
    print("\n  Complete LiFi system specification using component selection:")
    print("\n  Transmitter: OSRAM LR W5SN (Golden Dragon, 625nm)")
    print("  └─ I_drive = 100 mA")
    print("  └─ P_optical ≈ 180 mW")
    print("  └─ Lambertian m = 1")
    
    print("\n  Channel:")
    print("  └─ Distance = 0.5 m")
    print("  └─ Path loss ≈ 60 dB (Lambertian)")
    
    kxob = KXOB25_04X3F()
    R_load = 220  # Ω
    
    print(f"\n  Receiver: {kxob.name}")
    print(f"  └─ Responsivity = {kxob.responsivity:.3f} A/W")
    print(f"  └─ Capacitance = {kxob.capacitance*1e12:.0f} pF")
    print(f"  └─ Load R = {R_load} Ω")
    print(f"  └─ Bandwidth = {kxob.bandwidth(R_load)/1e3:.1f} kHz")
    
    # Simulate received signal
    P_tx = 180e-3  # 180 mW
    distance = 0.5  # m
    
    # Simplified Lambertian path loss
    m = 1  # Lambertian order
    A_rx = kxob.active_area_m2
    H_los = (m + 1) / (2 * np.pi * distance**2) * A_rx
    P_rx = P_tx * H_los
    
    I_ph = kxob.photocurrent(P_rx)
    V_out = I_ph * R_load
    
    print(f"\n  Signal chain calculation:")
    print(f"  └─ P_rx = {P_rx*1e6:.2f} µW")
    print(f"  └─ I_ph = {I_ph*1e6:.2f} µA")
    print(f"  └─ V_out = {V_out*1e3:.2f} mV")
    
    # SNR estimate
    B = kxob.bandwidth(R_load)
    snr = kxob.snr(P_rx, R_load, B)
    print(f"  └─ SNR ≈ {snr:.1f} dB")
    
    # =========================================================================
    # 5. VALIDATION AGAINST PAPER
    # =========================================================================
    print("\n" + "─" * 70)
    print("5️⃣  VALIDATION AGAINST CORREA 2025")
    print("─" * 70)
    
    print("\n  Paper target values vs our component model:")
    print()
    
    checks = [
        ("Responsivity", 0.457, kxob.responsivity, "A/W", 0.01),
        ("Capacitance", 798e-12, kxob.capacitance, "pF", 50e-12),
        ("Shunt R", 138.8, kxob.shunt_resistance, "Ω", 5),
        ("f_3dB @ 220Ω", 14e3, kxob.bandwidth(220), "kHz", 7e3),
    ]
    
    all_pass = True
    for name, target, actual, unit, tol in checks:
        status = "✓" if abs(actual - target) <= tol else "✗"
        if status == "✗":
            all_pass = False
        
        if 'pF' in unit:
            print(f"  {status} {name:15} Target: {target*1e12:>7.0f} {unit}  "
                  f"Model: {actual*1e12:>7.0f} {unit}")
        elif 'kHz' in unit:
            print(f"  {status} {name:15} Target: {target/1e3:>7.1f} {unit}  "
                  f"Model: {actual/1e3:>7.1f} {unit}")
        else:
            print(f"  {status} {name:15} Target: {target:>7.3f} {unit}  "
                  f"Model: {actual:>7.3f} {unit}")
    
    print()
    if all_pass:
        print("  ✅ All validation checks PASSED!")
    else:
        print("  ⚠️  Some checks outside tolerance (may need tuning)")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print("""
  The Hardware-Faithful Simulator allows you to:
  
  1. SELECT components by part number (KXOB25-04X3F, BPW34, etc.)
  2. DERIVE parameters automatically from physics and datasheets
  3. SIMULATE LiFi systems with realistic hardware behavior
  4. VALIDATE against published research papers
  
  Key files:
    materials/         - Semiconductor physics database
    components/        - Pre-configured hardware components  
    spice_parser/      - SPICE model file parser
    utils/             - Physical constants and helpers
    
  Next steps:
    - Add more components (LEDs, TIAs)
    - Integrate with existing simulator modules
    - Run full paper validation
""")


if __name__ == "__main__":
    main()
