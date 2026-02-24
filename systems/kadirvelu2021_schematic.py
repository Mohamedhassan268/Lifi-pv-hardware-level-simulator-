# systems/kadirvelu2021_schematic.py
"""
Kadirvelu 2021 - Detailed Component-Level Schematic Generation
==============================================================

Generates publication-quality visual circuit schematics using SchemDraw.
All component values sourced from KadirveluParams for consistency with
SPICE netlists.

Schematics:
    1. draw_full_system_detailed()    - Complete system (TX + Channel + RX)
    2. draw_solar_cell_detailed()     - Solar cell equivalent circuit
    3. draw_ina322_detailed()         - Instrumentation amplifier
    4. draw_bpf_detailed()            - 2-stage active band-pass filter
    5. draw_comparator_detailed()     - Data recovery comparator
    6. draw_dcdc_detailed()           - Boost DC-DC converter
    7. draw_tx_driver_detailed()      - LED transmitter driver
    8. draw_channel_model()           - Lambertian channel geometry

Requirements:
    pip install schemdraw

Usage:
    from systems.kadirvelu2021_schematic import draw_all_schematics
    draw_all_schematics('output_dir/')
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import schemdraw
    import schemdraw.elements as elm
    SCHEMDRAW_AVAILABLE = True
    SCHEMDRAW_VERSION = schemdraw.__version__
except ImportError:
    SCHEMDRAW_AVAILABLE = False
    SCHEMDRAW_VERSION = None
    print("SchemDraw not installed. Run: pip install schemdraw")

from systems.kadirvelu2021 import KadirveluParams


def _check_schemdraw():
    if not SCHEMDRAW_AVAILABLE:
        print("Error: SchemDraw not installed. Run: pip install schemdraw")
        return False
    return True


# =============================================================================
# 1. SOLAR CELL EQUIVALENT CIRCUIT (DETAILED)
# =============================================================================

def draw_solar_cell_detailed(filename=None, show=False, params=None):
    """
    Draw detailed solar cell equivalent circuit with all component values.

    Topology: Iph || Cj || Rsh || D1, with Rs in series.
    """
    if not _check_schemdraw():
        return None

    p = params or KadirveluParams()

    d = schemdraw.Drawing()
    d.config(unit=3, fontsize=10)

    # Title
    d += elm.Label().at((3, 5)).label(
        'Solar Cell Equivalent Circuit (KXOB25-04X3F)', fontsize=12)
    d += elm.Label().at((3, 4.5)).label(
        f'Kadirvelu 2021 — GaAs, Area = {p.SC_AREA_CM2} cm²', fontsize=9)

    # Top rail (anode_int)
    y_top = 3.0
    y_bot = 0.5

    # Photocurrent source
    d += elm.Dot().at((0.5, y_top))
    d += elm.SourceI().down().length(2.5).label(
        f'$I_{{ph}}$\n{p.SC_IPH_uA} µA', loc='left').reverse()
    d += elm.Dot().at((0.5, y_bot))

    # Junction capacitance
    d += elm.Line().at((0.5, y_top)).right().length(1.5)
    d += elm.Dot()
    cap_x = 2.0
    d += elm.Capacitor().down().length(2.5).label(
        f'$C_j$\n{p.SC_CJ_nF} nF', loc='right')
    d += elm.Dot()
    d += elm.Line().left().length(1.5)

    # Shunt resistance
    d += elm.Line().at((cap_x, y_top)).right().length(1.5)
    d += elm.Dot()
    rsh_x = 3.5
    d += elm.Resistor().down().length(2.5).label(
        f'$R_{{sh}}$\n{p.SC_RSH_kOhm} kΩ', loc='right')
    d += elm.Dot()
    d += elm.Line().left().length(1.5)

    # Diode
    d += elm.Line().at((rsh_x, y_top)).right().length(1.5)
    d += elm.Dot()
    diode_x = 5.0
    d += elm.Diode().down().length(2.5).label('$D_1$\nn=1.5', loc='right')
    d += elm.Dot()
    d += elm.Line().left().length(1.5)

    # Series resistance (Rs) to output
    d += elm.Line().at((diode_x, y_top)).right().length(0.5)
    d += elm.Resistor().right().label('$R_s$\n2.5 Ω', loc='top')
    rs_out = d.here
    d += elm.Dot().label('$V_{sc}$ (+)', loc='right')

    # Bottom rail
    d += elm.Line().at((0.5, y_bot)).right().length(6.5)
    d += elm.Dot().label('(−)', loc='right')

    # Node labels
    d += elm.Label().at((0.5, y_top + 0.3)).label('anode_int', fontsize=7, color='blue')
    d += elm.Label().at((rs_out[0], y_top + 0.3)).label('sc_anode', fontsize=7, color='blue')

    # Parameters box
    d += elm.Label().at((0, -0.5)).label(
        f'$R_λ$ = {p.SC_RESPONSIVITY} A/W @ 530nm  |  '
        f'$V_{{oc}}$ = 2.26 V  |  $I_{{sc}}$ = 78.9 mA',
        fontsize=8, halign='left')

    if filename:
        d.save(filename)
        print(f"Saved: {filename}")
    if show:
        d.draw()
    return d


# =============================================================================
# 2. INA322 INSTRUMENTATION AMPLIFIER (DETAILED)
# =============================================================================

def draw_ina322_detailed(filename=None, show=False, params=None):
    """
    Draw INA322 with gain-setting resistors and differential input.
    """
    if not _check_schemdraw():
        return None

    p = params or KadirveluParams()
    gain = 5 + 5 * (p.INA_R1 / p.INA_R2)

    d = schemdraw.Drawing()
    d.config(unit=3, fontsize=10)

    # Title
    d += elm.Label().at((3, 5)).label(
        f'INA322 Instrumentation Amplifier (G = {gain:.0f}x = {20*np.log10(gain):.0f} dB)',
        fontsize=12)

    # Inputs from Rsense
    d += elm.Dot().at((0, 3.5)).label('$V_{sense+}$', loc='left')
    d += elm.Line().right().length(1)
    d += elm.Dot().at((0, 2.5)).label('$V_{sense-}$', loc='left')
    d += elm.Line().right().length(1)

    # Op-amp triangle (INA322)
    d += elm.Opamp(anchor='in1').at((2, 3)).scale(0.8).label(
        'INA322', loc='center', fontsize=9)
    oa_out = d.here

    # Output
    d += elm.Line().at(oa_out).right().length(1)
    d += elm.Dot().label('ina_out', loc='right', fontsize=8, color='blue')

    # R1, R2 annotation
    d += elm.Label().at((4, 1.5)).label(
        f'$R_1$ = {p.INA_R1/1e3:.0f} kΩ\n$R_2$ = {p.INA_R2/1e3:.0f} kΩ\n'
        f'Gain = 5 + 5×($R_1$/$R_2$) = {gain:.1f}\n'
        f'GBW = {p.INA_GBW_kHz} kHz\n'
        f'$f_{{3dB}}$ = {p.INA_GBW_kHz*1e3/gain:.0f} Hz',
        fontsize=8, halign='left')

    if filename:
        d.save(filename)
        print(f"Saved: {filename}")
    if show:
        d.draw()
    return d


# =============================================================================
# 3. BAND-PASS FILTER (DETAILED, 2 STAGES)
# =============================================================================

def draw_bpf_detailed(filename=None, show=False, params=None):
    """
    Draw 2-stage active band-pass filter with component values.

    Each stage: HP (Chp + Rhp) + inverting active LP (Rin, Rfb, Cfb, TLV2379)
    """
    if not _check_schemdraw():
        return None

    p = params or KadirveluParams()
    f_hp = 1 / (2 * np.pi * p.BPF_RHP * p.BPF_CHP_pF * 1e-12)
    f_lp = 1 / (2 * np.pi * p.BPF_RLP * p.BPF_CLF_nF * 1e-9)

    d = schemdraw.Drawing()
    d.config(unit=2.5, fontsize=9)

    # Title
    d += elm.Label().at((6, 5.5)).label(
        f'2-Stage Active Band-Pass Filter (TLV2379)', fontsize=12)
    d += elm.Label().at((6, 5.0)).label(
        f'HP: {f_hp/1e3:.1f} kHz | LP: {f_lp:.0f} Hz | '
        f'Passband: 700 Hz – 10 kHz', fontsize=9)

    # ---- STAGE 1 ----
    d += elm.Label().at((0, 4.2)).label('Stage 1', fontsize=10, color='blue')

    # Input
    d += elm.Dot().at((0, 3.5)).label('$V_{in}$', loc='left')

    # HP cap
    d += elm.Capacitor().right().label(
        f'$C_{{HP}}$\n{p.BPF_CHP_pF} pF', loc='top')
    hp1 = d.here
    d += elm.Dot()

    # Rhp to Vref
    d += elm.Resistor().down().length(2).label(
        f'$R_{{HP}}$\n{p.BPF_RHP/1e3:.0f} kΩ', loc='left')
    d += elm.Dot().label('$V_{ref}$ 1.65V', loc='bottom', fontsize=8)

    # Rin
    d += elm.Line().at(hp1).right().length(0.3)
    d += elm.Resistor().right().label(f'$R_{{in}}$\n{p.BPF_RLP/1e3:.0f} kΩ', loc='top')
    inn1 = d.here
    d += elm.Dot()

    # Op-amp
    opamp1 = d.add(elm.Opamp(anchor='in2').at((inn1[0] + 1.2, inn1[1] + 0.5)).scale(0.6))

    # Feedback Rfb (top path)
    d += elm.Line().at(inn1).up().length(0.8)
    fb1_start = d.here
    d += elm.Resistor().right().length(2.5).label(
        f'$R_{{fb}}$ {p.BPF_RLP/1e3:.0f}kΩ', loc='top')
    fb1_end = d.here

    # Feedback Cfb (parallel, above)
    d += elm.Line().at(fb1_start).up().length(0.5)
    d += elm.Capacitor().right().length(2.5).label(
        f'$C_{{fb}}$ {p.BPF_CLF_nF}nF', loc='top')
    d += elm.Line().down().length(0.5)

    # Connect feedback to output
    d += elm.Line().at(fb1_end).down().length(0.8)
    out1 = d.here
    d += elm.Dot().label('bpf1_out', fontsize=7, color='blue', loc='bottom')

    # Non-inverting input to Vref
    d += elm.Line().at(opamp1.in1).left().length(0.3)
    d += elm.Dot().label('$V_{ref}$', loc='left', fontsize=8)

    # ---- STAGE 2 (same topology, shifted right) ----
    offset_x = 7
    d += elm.Label().at((offset_x, 4.2)).label('Stage 2', fontsize=10, color='blue')

    d += elm.Line().at(out1).right().length(1.5)

    # HP cap stage 2
    d += elm.Capacitor().right().label(
        f'$C_{{HP}}$\n{p.BPF_CHP_pF} pF', loc='top')
    hp2 = d.here
    d += elm.Dot()

    # Rhp to Vref
    d += elm.Resistor().down().length(2).label(
        f'$R_{{HP}}$\n{p.BPF_RHP/1e3:.0f} kΩ', loc='left')
    d += elm.Dot().label('$V_{ref}$', loc='bottom', fontsize=8)

    # Rin
    d += elm.Line().at(hp2).right().length(0.3)
    d += elm.Resistor().right().label(f'$R_{{in}}$\n{p.BPF_RLP/1e3:.0f}kΩ', loc='top')
    inn2 = d.here
    d += elm.Dot()

    # Op-amp
    opamp2 = d.add(elm.Opamp(anchor='in2').at((inn2[0] + 1.2, inn2[1] + 0.5)).scale(0.6))

    # Feedback
    d += elm.Line().at(inn2).up().length(0.8)
    fb2_start = d.here
    d += elm.Resistor().right().length(2.5).label(f'$R_{{fb}}$ {p.BPF_RLP/1e3:.0f}kΩ', loc='top')
    fb2_end = d.here

    d += elm.Line().at(fb2_start).up().length(0.5)
    d += elm.Capacitor().right().length(2.5).label(f'$C_{{fb}}$ {p.BPF_CLF_nF}nF', loc='top')
    d += elm.Line().down().length(0.5)

    d += elm.Line().at(fb2_end).down().length(0.8)
    d += elm.Dot().label('bpf_out', fontsize=7, color='blue', loc='right')

    # Non-inverting input to Vref
    d += elm.Line().at(opamp2.in1).left().length(0.3)
    d += elm.Dot().label('$V_{ref}$', loc='left', fontsize=8)

    if filename:
        d.save(filename)
        print(f"Saved: {filename}")
    if show:
        d.draw()
    return d


# =============================================================================
# 4. COMPARATOR (DETAILED)
# =============================================================================

def draw_comparator_detailed(filename=None, show=False, params=None):
    """
    Draw TLV7011 comparator with threshold reference.
    """
    if not _check_schemdraw():
        return None

    d = schemdraw.Drawing()
    d.config(unit=3, fontsize=10)

    d += elm.Label().at((2.5, 4)).label(
        'TLV7011 Comparator (Data Recovery)', fontsize=12)

    # BPF output input
    d += elm.Dot().at((0, 2.5)).label('$V_{bpf}$', loc='left')
    d += elm.Line().right().length(1)

    # Vref input
    d += elm.Dot().at((0, 1.5)).label('$V_{ref}$\n1.65V', loc='left')
    d += elm.Line().right().length(1)

    # Comparator symbol (op-amp triangle)
    d += elm.Opamp(anchor='in1').at((2, 2)).scale(0.8).label(
        'TLV7011', loc='center', fontsize=9)

    # Output
    d += elm.Line().right().length(1)
    d += elm.Dot().label('$d_{out}$', loc='right')

    # Specs
    d += elm.Label().at((1, 0.3)).label(
        '$t_{pd}$ = 260 ns  |  $I_q$ = 335 nA  |  Push-pull output',
        fontsize=8, halign='left')

    if filename:
        d.save(filename)
        print(f"Saved: {filename}")
    if show:
        d.draw()
    return d


# =============================================================================
# 5. DC-DC BOOST CONVERTER (DETAILED)
# =============================================================================

def draw_dcdc_detailed(filename=None, show=False, params=None):
    """
    Draw boost DC-DC converter with all component values.
    """
    if not _check_schemdraw():
        return None

    p = params or KadirveluParams()

    d = schemdraw.Drawing()
    d.config(unit=3, fontsize=10)

    d += elm.Label().at((3.5, 5)).label(
        'Boost DC-DC Converter', fontsize=12)
    d += elm.Label().at((3.5, 4.5)).label(
        f'$f_{{sw}}$ = {p.DCDC_FSW_kHz} kHz | η = 67% (@ 50 kHz)',
        fontsize=9)

    # Input from solar cell
    d += elm.Dot().at((0, 3)).label('$V_{sc}$', loc='left')

    # Input capacitor (down to ground)
    d += elm.Line().down().length(0.5)
    d += elm.Capacitor().down().length(1.5).label(
        f'$C_P$\n{p.DCDC_CP_uF} µF', loc='left')
    d += elm.Ground()

    # Main path: inductor
    d += elm.Line().at((0, 3)).right().length(0.5)
    d += elm.Inductor2().right().label(f'L\n{p.DCDC_L_uH} µH', loc='top')
    sw_node = d.here
    d += elm.Dot()

    # NMOS switch (down to ground)
    d += elm.Line().down().length(0.3)
    fet = d.add(elm.NFet(anchor='drain'))
    d += elm.Line().at(fet.source).down().length(0.3)
    d += elm.Ground()

    # Gate label
    d += elm.Label().at((sw_node[0] - 1, sw_node[1] - 1.2)).label(
        'NTS4409\nφ (PWM)', fontsize=8)

    # Schottky diode (right from switch node)
    d += elm.Line().at(sw_node).right().length(0.3)
    d += elm.Diode().right().label('$D_S$\nSchottky', loc='top')
    out_node = d.here
    d += elm.Dot()

    # Output capacitor
    d += elm.Line().down().length(0.3)
    d += elm.Capacitor().down().length(1.5).label(
        f'$C_L$\n{p.DCDC_CL_uF} µF', loc='right')
    d += elm.Ground()

    # Load resistor
    d += elm.Line().at(out_node).right().length(1.5)
    d += elm.Dot()
    d += elm.Resistor().down().length(1.8).label(
        f'$R_{{load}}$\n{p.DCDC_RLOAD_kOhm} kΩ', loc='right')
    d += elm.Ground()

    # Output label
    d += elm.Line().at(out_node).up().length(0.5)
    d += elm.Dot().label('$V_{out}$', loc='top')

    if filename:
        d.save(filename)
        print(f"Saved: {filename}")
    if show:
        d.draw()
    return d


# =============================================================================
# 6. TX DRIVER (DETAILED)
# =============================================================================

def draw_tx_driver_detailed(filename=None, show=False, params=None):
    """
    Draw LED transmitter driver with op-amp and MOSFET.
    """
    if not _check_schemdraw():
        return None

    p = params or KadirveluParams()

    d = schemdraw.Drawing()
    d.config(unit=3, fontsize=10)

    d += elm.Label().at((3, 5)).label(
        'LED Transmitter Driver', fontsize=12)
    d += elm.Label().at((3, 4.5)).label(
        f'ADA4891 + BSD235N + LXM5-PD01 + Fraen Lens', fontsize=9)

    # Signal source
    d += elm.Dot().at((0, 3)).label('$V_{mod}$', loc='left')
    d += elm.SourceSin().right().label('OOK\nSignal', loc='top', fontsize=8)

    # Driver resistance
    d += elm.Resistor().right().label(f'$R_e$\n{p.LED_DRIVER_RE} Ω', loc='top')

    # MOSFET gate drive point
    gate_pt = d.here
    d += elm.Dot()

    # MOSFET (down)
    d += elm.Line().right().length(0.5)
    d += elm.Line().down().length(0.3)
    fet = d.add(elm.NFet(anchor='gate'))

    # MOSFET drain -> LED
    d += elm.Line().at(fet.drain).up().length(0.5)
    d += elm.LED().up().label('LXM5-PD01', loc='right', fontsize=8).reverse()
    d += elm.Line().up().length(0.3)
    d += elm.Dot().label('$V_{CC}$', loc='top')

    # MOSFET source -> GND
    d += elm.Line().at(fet.source).down().length(0.3)
    d += elm.Ground()

    # MOSFET label
    d += elm.Label().at((gate_pt[0] + 1.5, gate_pt[1] - 1.5)).label(
        'BSD235N', fontsize=8)

    # Lens annotation
    d += elm.Label().at((gate_pt[0] + 2.5, gate_pt[1] + 2)).label(
        f'Fraen Lens\n$T$ = {p.LENS_TRANSMITTANCE}\n'
        f'θ½ = {p.LED_HALF_ANGLE_DEG}°\n'
        f'$P_e$ = {p.LED_RADIATED_POWER_mW} mW',
        fontsize=8, halign='left')

    if filename:
        d.save(filename)
        print(f"Saved: {filename}")
    if show:
        d.draw()
    return d


# =============================================================================
# 7. FULL SYSTEM BLOCK DIAGRAM (DETAILED)
# =============================================================================

def draw_full_system_detailed(filename='kadirvelu2021_system.png', show=False,
                               params=None):
    """
    Draw complete system with TX, channel, and RX paths.

    Layout:
        TX (left) -> Channel -> RX (right)
        RX splits: Data path (top) and Power path (bottom)
    """
    if not _check_schemdraw():
        return None

    p = params or KadirveluParams()
    Gop = p.optical_channel_gain()

    d = schemdraw.Drawing()
    d.config(unit=2, fontsize=8)

    # =========================================================================
    # TITLE
    # =========================================================================
    d += elm.Label().at((7, 8)).label(
        'Kadirvelu 2021 — Complete LiFi-PV System', fontsize=14)
    d += elm.Label().at((7, 7.4)).label(
        '"A Circuit for Simultaneous Reception of Data and Power '
        'Using a Solar Cell"', fontsize=8)

    # =========================================================================
    # TRANSMITTER
    # =========================================================================
    d += elm.Label().at((1, 6.5)).label('── TX ──', fontsize=10, color='blue')

    d += elm.Dot().at((0, 5.5))
    d += elm.SourceSin().right().label('$V_{mod}$', loc='top', fontsize=7)
    d += elm.Resistor().right().label(f'$R_e$\n{p.LED_DRIVER_RE}Ω', loc='top', fontsize=7)
    d += elm.Dot()

    # LED driver block
    d += elm.Line().right().length(0.3)
    d += elm.RBox(w=1.2, h=0.6).label('ADA4891\n+BSD235N', fontsize=6)
    d += elm.Line().right().length(0.3)

    # LED
    d += elm.LED().right().label('LXM5\n+Lens', loc='bottom', fontsize=6)

    # =========================================================================
    # OPTICAL CHANNEL
    # =========================================================================
    d += elm.Arrow().right().length(2).color('red').label(
        f'Optical Channel\n$G_{{op}}$ = {Gop:.2e}\n'
        f'd = {p.DISTANCE_M*100:.0f} cm',
        loc='top', fontsize=7, color='red')

    rx_in = d.here

    # =========================================================================
    # RECEIVER — SOLAR CELL
    # =========================================================================
    d += elm.Label().at((9, 6.5)).label('── RX ──', fontsize=10, color='green')

    d += elm.Dot().at(rx_in)
    d += elm.RBox(w=1.5, h=0.8).label(
        f'Solar Cell\nKXOB25\n$R_λ$={p.SC_RESPONSIVITY} A/W', fontsize=6)
    sc_out = d.here
    d += elm.Dot()

    # =========================================================================
    # DATA PATH (upper branch)
    # =========================================================================
    d += elm.Label().at((12, 6.5)).label('Data Path', fontsize=8, color='purple')

    # Branch up
    d += elm.Line().at(sc_out).up().length(0.8)
    d += elm.Dot()
    d += elm.Line().right().length(0.2)

    # Rsense
    d += elm.Resistor().right().label(f'$R_s$\n{p.R_SENSE}Ω', loc='top', fontsize=6)
    d += elm.Line().right().length(0.2)

    # INA322
    d += elm.RBox(w=1.2, h=0.6).label(
        f'INA322\n{p.INA_GAIN_DB:.0f}dB', fontsize=6)
    d += elm.Line().right().length(0.2)

    # BPF Stage 1
    d += elm.RBox(w=1, h=0.6).label('BPF\n×1', fontsize=6)
    d += elm.Line().right().length(0.2)

    # BPF Stage 2
    d += elm.RBox(w=1, h=0.6).label('BPF\n×2', fontsize=6)
    d += elm.Line().right().length(0.2)

    # Comparator
    d += elm.RBox(w=1, h=0.6).label('TLV7011\nCOMP', fontsize=6)
    d += elm.Line().right().length(0.3)
    d += elm.Dot().label('$d_{out}$', loc='right', fontsize=8)

    # =========================================================================
    # POWER PATH (lower branch)
    # =========================================================================
    d += elm.Label().at((12, 3.5)).label('Power Path', fontsize=8, color='orange')

    d += elm.Line().at(sc_out).down().length(0.8)
    d += elm.Dot()
    d += elm.Line().right().length(0.2)

    # DC-DC block
    d += elm.RBox(w=1.5, h=0.8).label(
        f'DC-DC\nBoost\n$f_{{sw}}$={p.DCDC_FSW_kHz}kHz', fontsize=6)
    d += elm.Line().right().length(0.3)
    d += elm.Dot()
    dcdc_out = d.here

    # Load
    d += elm.Line().right().length(0.5)
    d += elm.Resistor().down().length(1.2).label(
        f'$R_L$\n{p.DCDC_RLOAD_kOhm}kΩ', fontsize=6)
    d += elm.Ground()

    # Vout label
    d += elm.Dot().at(dcdc_out).label('$V_{out}$', loc='top')

    # =========================================================================
    # PARAMETER BOXES
    # =========================================================================
    d += elm.Label().at((0.5, 2.5)).label(
        '┌─────────────────────────────────┐\n'
        '│ System Parameters:              │\n'
        f'│ • $C_j$ = {p.SC_CJ_nF} nF                │\n'
        f'│ • $R_{{sh}}$ = {p.SC_RSH_kOhm} kΩ             │\n'
        '│ • BPF: 700 Hz – 10 kHz         │\n'
        f'│ • INA gain: {p.INA_GAIN_DB:.0f} dB              │\n'
        '└─────────────────────────────────┘',
        fontsize=7, halign='left')

    d += elm.Label().at((9, 2.5)).label(
        '┌─────────────────────────────────┐\n'
        '│ Validation Targets:             │\n'
        f'│ • $P_{{harv}}$ = {p.TARGET_HARVESTED_POWER_uW} µW            │\n'
        f'│ • BER = {p.TARGET_BER:.3e}            │\n'
        f'│ • Noise = {p.TARGET_NOISE_RMS_mV} mV$_{{rms}}$        │\n'
        '│ • η(50kHz) = 67%               │\n'
        '└─────────────────────────────────┘',
        fontsize=7, halign='left')

    if filename:
        d.save(filename)
        print(f"System schematic saved: {filename}")
    if show:
        d.draw()
    return d


# =============================================================================
# 8. CHANNEL MODEL DIAGRAM
# =============================================================================

def draw_channel_model(filename=None, show=False, params=None):
    """
    Draw annotated Lambertian channel geometry diagram.
    """
    if not _check_schemdraw():
        return None

    p = params or KadirveluParams()
    m = p.lambertian_order()
    Gop = p.optical_channel_gain()

    d = schemdraw.Drawing()
    d.config(unit=3, fontsize=10)

    d += elm.Label().at((3, 5)).label(
        'Optical Channel Model (Lambertian LOS)', fontsize=12)

    # TX point
    d += elm.Dot().at((1, 3)).label('TX (LED)', loc='left')
    d += elm.LED().right().length(1)

    # Channel arrow
    d += elm.Arrow().right().length(3).color('red').label(
        f'r = {p.DISTANCE_M*100:.1f} cm\n'
        f'θ = {p.THETA_DEG}°, β = {p.BETA_DEG}°',
        loc='top', fontsize=9, color='red')

    # RX point
    d += elm.RBox(w=1.5, h=0.6).label('RX (PV)', fontsize=8)
    d += elm.Dot().label('$A_{rx}$', loc='right')

    # Equations
    d += elm.Label().at((1, 1)).label(
        f'$m$ = −ln(2)/ln(cos(α½)) = {m:.2f}\n'
        f'$G_{{op}}$ = (m+1)/(2πr²) · cos$^m$(θ) · cos(β) · A = {Gop:.4e}\n'
        f'$A_{{rx}}$ = {p.SC_AREA_CM2} cm²  |  α½ = {p.LED_HALF_ANGLE_DEG}°',
        fontsize=9, halign='left')

    if filename:
        d.save(filename)
        print(f"Saved: {filename}")
    if show:
        d.draw()
    return d


# =============================================================================
# BACKWARD COMPATIBILITY (legacy function names)
# =============================================================================

def draw_full_system(filename='kadirvelu2021_system.png', show=False):
    """Legacy wrapper for draw_full_system_detailed."""
    return draw_full_system_detailed(filename, show)

def draw_solar_cell_equivalent(filename=None, show=False):
    """Legacy wrapper for draw_solar_cell_detailed."""
    return draw_solar_cell_detailed(filename, show)

def draw_dcdc_converter(filename=None, show=False):
    """Legacy wrapper for draw_dcdc_detailed."""
    return draw_dcdc_detailed(filename, show)

def draw_bandpass_filter(filename=None, show=False):
    """Legacy wrapper for draw_bpf_detailed."""
    return draw_bpf_detailed(filename, show)

def draw_receiver_detail(filename=None, show=False):
    """Legacy wrapper — now redirects to full system."""
    return draw_full_system_detailed(filename, show)


# =============================================================================
# GENERATE ALL SCHEMATICS
# =============================================================================

def draw_all_schematics(output_dir='.', fmt='png'):
    """
    Generate all schematic diagrams.

    Args:
        output_dir: Output directory
        fmt: File format ('png', 'svg', 'pdf')
    """
    if not _check_schemdraw():
        return

    os.makedirs(output_dir, exist_ok=True)

    print(f"Generating schematics (schemdraw v{SCHEMDRAW_VERSION})...")
    print("=" * 60)

    schematics = [
        ('full_system', draw_full_system_detailed),
        ('solar_cell', draw_solar_cell_detailed),
        ('ina322', draw_ina322_detailed),
        ('bpf', draw_bpf_detailed),
        ('comparator', draw_comparator_detailed),
        ('dcdc', draw_dcdc_detailed),
        ('tx_driver', draw_tx_driver_detailed),
        ('channel', draw_channel_model),
    ]

    for name, func in schematics:
        filepath = os.path.join(output_dir, f'kadirvelu2021_{name}.{fmt}')
        try:
            func(filepath, show=False)
        except Exception as e:
            print(f"  Warning: Failed to generate {name}: {e}")

    print("=" * 60)
    print(f"All schematics saved to: {output_dir}/")


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Generate Kadirvelu 2021 schematics')
    parser.add_argument('--output', '-o', type=str, default='.',
                        help='Output directory')
    parser.add_argument('--format', '-f', type=str, default='png',
                        choices=['png', 'svg', 'pdf'],
                        help='Output format')
    parser.add_argument('--all', action='store_true',
                        help='Generate all schematics')
    parser.add_argument('--system', action='store_true',
                        help='System block diagram')
    parser.add_argument('--solarcell', action='store_true',
                        help='Solar cell equivalent')
    parser.add_argument('--ina', action='store_true',
                        help='INA322 schematic')
    parser.add_argument('--bpf', action='store_true',
                        help='Band-pass filter')
    parser.add_argument('--comp', action='store_true',
                        help='Comparator')
    parser.add_argument('--dcdc', action='store_true',
                        help='DC-DC converter')
    parser.add_argument('--tx', action='store_true',
                        help='TX driver')
    parser.add_argument('--channel', action='store_true',
                        help='Channel model')

    args = parser.parse_args()

    if not _check_schemdraw():
        exit(1)

    if args.all or not any([args.system, args.solarcell, args.ina, args.bpf,
                            args.comp, args.dcdc, args.tx, args.channel]):
        draw_all_schematics(args.output, args.format)
    else:
        if args.system:
            draw_full_system_detailed(f'{args.output}/kadirvelu2021_full_system.{args.format}')
        if args.solarcell:
            draw_solar_cell_detailed(f'{args.output}/kadirvelu2021_solar_cell.{args.format}')
        if args.ina:
            draw_ina322_detailed(f'{args.output}/kadirvelu2021_ina322.{args.format}')
        if args.bpf:
            draw_bpf_detailed(f'{args.output}/kadirvelu2021_bpf.{args.format}')
        if args.comp:
            draw_comparator_detailed(f'{args.output}/kadirvelu2021_comparator.{args.format}')
        if args.dcdc:
            draw_dcdc_detailed(f'{args.output}/kadirvelu2021_dcdc.{args.format}')
        if args.tx:
            draw_tx_driver_detailed(f'{args.output}/kadirvelu2021_tx.{args.format}')
        if args.channel:
            draw_channel_model(f'{args.output}/kadirvelu2021_channel.{args.format}')
