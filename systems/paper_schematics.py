# systems/paper_schematics.py
"""
Multi-Paper Schematic Generation
=================================

Generic schemdraw generators that work with any paper's configuration.
Each paper gets a Full System block diagram plus component-specific schematics
appropriate to its architecture.

For Kadirvelu 2021, delegates to the existing detailed implementations
in kadirvelu2021_schematic.py.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import schemdraw
    import schemdraw.elements as elm
    SCHEMDRAW_AVAILABLE = True
except ImportError:
    SCHEMDRAW_AVAILABLE = False

from cosim.system_config import SystemConfig
from systems.schematic_style import create_styled_drawing


def _check():
    if not SCHEMDRAW_AVAILABLE:
        raise ImportError("schemdraw not installed. Run: pip install schemdraw")


# =============================================================================
# Paper Registry - maps preset name to available schematics
# =============================================================================

def get_paper_info():
    """Return dict of paper_key -> {label, schematics: [(name, func), ...]}."""
    return {
        'kadirvelu2021': {
            'label': 'Kadirvelu 2021 - Indoor Desktop LiFi',
            'schematics': [
                ('Full System', _kadirvelu_full),
                ('Solar Cell (KXOB25-04X3F)', _kadirvelu_solar),
                ('INA322 Amplifier', _kadirvelu_ina),
                ('Band-Pass Filter (TLV2379)', _kadirvelu_bpf),
                ('Comparator (TLV7011)', _kadirvelu_comp),
                ('DC-DC Boost Converter', _kadirvelu_dcdc),
                ('LED Driver (TX)', _kadirvelu_tx),
                ('Channel Model', _kadirvelu_channel),
            ],
        },
        'fakidis2020': {
            'label': 'Fakidis 2020 - Indoor OWP at Nighttime',
            'schematics': [
                ('Full System', draw_fakidis_full),
                ('Solar Cell (SM141K)', draw_fakidis_solar),
                ('INA322 Amplifier', draw_fakidis_ina),
                ('Band-Pass Filter', draw_fakidis_bpf),
                ('DC-DC Boost Converter', draw_fakidis_dcdc),
                ('LED TX Driver', draw_fakidis_tx),
                ('Channel Model', draw_generic_channel),
            ],
        },
        'sarwar2017': {
            'label': 'Sarwar 2017 - High-Speed OFDM VLC',
            'schematics': [
                ('Full System', draw_sarwar_full),
                ('LED Transmitter', draw_sarwar_tx),
                ('Solar Panel Receiver', draw_sarwar_rx),
                ('OFDM Signal Chain', draw_sarwar_ofdm),
                ('Channel Model', draw_generic_channel),
            ],
        },
        'gonzalez2024': {
            'label': 'González 2024 - Low-Cost VLC Receiver',
            'schematics': [
                ('Full System', draw_gonzalez_full),
                ('LED Transmitter', draw_gonzalez_tx),
                ('Solar Panel (Poly-Si 66 cm2)', draw_gonzalez_solar),
                ('Amplifier + Notch Filter', draw_gonzalez_amp),
                ('Comparator (TLV7011)', draw_gonzalez_comp),
                ('Channel Model', draw_generic_channel),
            ],
        },
        'oliveira2024': {
            'label': 'Oliveira 2024 - Laser MIMO VLC',
            'schematics': [
                ('Full System', draw_oliveira_full),
                ('Laser Transmitter (658 nm)', draw_oliveira_tx),
                ('Photodiode Array (3x3)', draw_oliveira_rx),
                ('OFDM Signal Chain', draw_oliveira_ofdm),
                ('Channel Model', draw_generic_channel),
            ],
        },
        'xu2024': {
            'label': 'Xu 2024 - Sunlight-Duo Harvesting',
            'schematics': [
                ('Full System', draw_xu_full),
                ('LC Shutter Transmitter', draw_xu_tx),
                ('GaAs Solar Array (16-cell)', draw_xu_solar),
                ('BFSK Demodulator', draw_xu_bfsk),
                ('Channel Model', draw_generic_channel),
            ],
        },
        'correa2025': {
            'label': 'Correa 2025 - Greenhouse VLC',
            'schematics': [
                ('Full System', draw_correa_full),
                ('LED Panel Transmitter (30 W)', draw_correa_tx),
                ('Solar Panel (Poly-Si)', draw_correa_solar),
                ('TL072 Amplifier', draw_correa_amp),
                ('Channel Model', draw_generic_channel),
            ],
        },
    }


# =============================================================================
# Kadirvelu 2021 - delegates to existing detailed schematics
# =============================================================================

def _kadirvelu_full():
    from systems.kadirvelu2021_schematic import draw_full_system_detailed
    return draw_full_system_detailed()

def _kadirvelu_solar():
    from systems.kadirvelu2021_schematic import draw_solar_cell_detailed
    return draw_solar_cell_detailed()

def _kadirvelu_ina():
    from systems.kadirvelu2021_schematic import draw_ina322_detailed
    return draw_ina322_detailed()

def _kadirvelu_bpf():
    from systems.kadirvelu2021_schematic import draw_bpf_detailed
    return draw_bpf_detailed()

def _kadirvelu_comp():
    from systems.kadirvelu2021_schematic import draw_comparator_detailed
    return draw_comparator_detailed()

def _kadirvelu_dcdc():
    from systems.kadirvelu2021_schematic import draw_dcdc_detailed
    return draw_dcdc_detailed()

def _kadirvelu_tx():
    from systems.kadirvelu2021_schematic import draw_tx_driver_detailed
    return draw_tx_driver_detailed()

def _kadirvelu_channel():
    from systems.kadirvelu2021_schematic import draw_channel_model
    return draw_channel_model()


# =============================================================================
# Helper: load config for a preset
# =============================================================================

def _cfg(preset):
    return SystemConfig.from_preset(preset)


def _lambertian_order(half_angle_deg):
    a = np.radians(half_angle_deg)
    if np.cos(a) <= 0:
        return 1.0
    return -np.log(2) / np.log(np.cos(a))


def _channel_gain(cfg):
    m = _lambertian_order(cfg.led_half_angle_deg)
    r = cfg.distance_m
    theta = np.radians(cfg.tx_angle_deg)
    beta = np.radians(cfg.rx_tilt_deg)
    A = cfg.sc_area_cm2 * 1e-4
    T = cfg.lens_transmittance
    if r <= 0:
        return 0
    G = (m + 1) / (2 * np.pi * r**2) * np.cos(theta)**m * np.cos(beta) * A * T
    return max(G, 0)


# =============================================================================
# Generic Channel Model - works for any paper
# =============================================================================

def draw_generic_channel(preset='kadirvelu2021'):
    _check()
    c = _cfg(preset)
    m = _lambertian_order(c.led_half_angle_deg)
    G = _channel_gain(c)

    d = create_styled_drawing(unit=3, fontsize=10)

    d += elm.Label().at((3, 5.5)).label(
        f'Optical Channel - {preset}', fontsize=12)
    d += elm.Label().at((3, 5.0)).label(
        'Lambertian Line-of-Sight Model', fontsize=9)

    # TX
    d += elm.Dot().at((0.5, 3)).label('TX', loc='left')
    d += elm.LED().right().length(1)

    # Arrow
    d += elm.Arrow().right().length(3.5).color('red').label(
        f'r = {c.distance_m*100:.1f} cm\n'
        f'theta = {c.tx_angle_deg} deg, beta = {c.rx_tilt_deg} deg',
        loc='top', fontsize=9, color='red')

    # RX
    d += elm.RBox(w=1.5, h=0.6).label('RX (PV)', fontsize=8)
    d += elm.Dot().label(f'A = {c.sc_area_cm2} cm2', loc='right', fontsize=8)

    # Equations
    d += elm.Label().at((0.5, 1.2)).label(
        f'm = {m:.2f}  (a_half = {c.led_half_angle_deg} deg)\n'
        f'G_op = {G:.4e}\n'
        f'T_lens = {c.lens_transmittance}',
        fontsize=9, halign='left')

    return d


# =============================================================================
# FAKIDIS 2020 - Indoor OWP (similar to Kadirvelu, higher power)
# =============================================================================

def draw_fakidis_full():
    _check()
    c = _cfg('fakidis2020')
    G = _channel_gain(c)
    d = create_styled_drawing(unit=3, fontsize=10)

    y_data, y_main, y_power = 14, 10, 6

    # Title
    d += elm.Label().at((10, 17)).label(
        'Fakidis 2020 - Indoor Optical Wireless Power', fontsize=15)
    d += elm.Label().at((10, 16.2)).label(
        'OOK @ 2 kbps  |  d = 1.0 m  |  Si Solar Cell', fontsize=10)

    # TX row
    d += elm.Label().at((2, y_main + 1.5)).label('TRANSMITTER', fontsize=11)
    d += elm.Dot().at((0, y_main))
    d += elm.SourceSin().right().length(2.5).label('OOK Signal', loc='top')
    d += elm.Resistor().right().length(3).label('Re = 12.1 Ohm', loc='top')
    d += elm.Dot()
    d += elm.Line().right().length(0.8)
    d += elm.RBox(w=2.5, h=1.0).label('ADA4891\n+ MOSFET', fontsize=9)
    d += elm.Line().right().length(0.8)
    d += elm.LED().right().length(2.5).label('LXM5 (50 mW)', loc='bottom', fontsize=9)
    d += elm.Dot()
    d += elm.Line().right().length(0.5)
    d += elm.Arrow().right().length(3.5).color('red').label(
        f'Channel\nG = {G:.2e}', loc='top', fontsize=9, color='red')
    d += elm.Line().right().length(0.5)
    d += elm.Label().at((d.here[0] + 2, y_main + 1.5)).label('RECEIVER', fontsize=11)
    d += elm.RBox(w=3.0, h=1.2).label('SM141K\nSi Solar Cell\n25 cm2', fontsize=9)
    sc_out = d.here
    d += elm.Dot()

    # Branch up to data path
    d += elm.Line().at(sc_out).up().length(y_data - y_main)
    d += elm.Dot()
    d += elm.Label().at((sc_out[0] + 5, y_data + 1.5)).label(
        'DATA PATH', fontsize=11, color='#4B0082')
    d += elm.Line().right().length(0.8)
    d += elm.Resistor().right().length(3).label('Rs = 2.2 Ohm', loc='top')
    d += elm.Line().right().length(0.8)
    d += elm.RBox(w=2.0, h=1.0).label('INA322\n(40 dB)', fontsize=9)
    d += elm.Line().right().length(0.8)
    d += elm.RBox(w=2.0, h=1.0).label('BPF\nStage 1', fontsize=9)
    d += elm.Line().right().length(0.8)
    d += elm.RBox(w=2.0, h=1.0).label('BPF\nStage 2', fontsize=9)
    d += elm.Line().right().length(0.8)
    d += elm.Dot().label('Data Out', loc='right', fontsize=10)

    # Branch down to power path
    d += elm.Line().at(sc_out).down().length(y_main - y_power)
    d += elm.Dot()
    d += elm.Label().at((sc_out[0] + 5, y_power + 1.5)).label(
        'POWER PATH', fontsize=11, color='#B8860B')
    d += elm.Line().right().length(0.8)
    d += elm.RBox(w=3.0, h=1.2).label('DC-DC Boost\n100 kHz', fontsize=9)
    d += elm.Line().right().length(1.0)
    d += elm.Dot()
    dcdc_out = d.here
    d += elm.Line().right().length(1.5)
    d += elm.Resistor().down().length(2.5).label('RL = 100 kOhm', loc='right')
    d += elm.Ground()
    d += elm.Dot().at(dcdc_out).label('Vout', loc='top', fontsize=10)

    return d


def draw_fakidis_solar():
    _check()
    d = create_styled_drawing(unit=3, fontsize=10)

    d += elm.Label().at((3, 5)).label(
        'Solar Cell - SM141K (IXYS Silicon)', fontsize=12)
    d += elm.Label().at((3, 4.5)).label(
        'Fakidis 2020 - Area = 25 cm2, R_resp = 0.40 A/W', fontsize=9)

    y_top, y_bot = 3.0, 0.5

    # Photocurrent
    d += elm.Dot().at((0.5, y_top))
    d += elm.SourceI().down().length(2.5).label(
        '$I_{ph}$\n800 uA', loc='left').reverse()
    d += elm.Dot().at((0.5, y_bot))

    # Cj
    d += elm.Line().at((0.5, y_top)).right().length(1.5)
    d += elm.Dot()
    d += elm.Capacitor().down().length(2.5).label(
        '$C_j$\n1200 nF', loc='right')
    d += elm.Dot()
    d += elm.Line().left().length(1.5)

    # Rsh
    d += elm.Line().at((2.0, y_top)).right().length(1.5)
    d += elm.Dot()
    d += elm.Resistor().down().length(2.5).label(
        '$R_{sh}$\n50 kOhm', loc='right')
    d += elm.Dot()
    d += elm.Line().left().length(1.5)

    # Diode
    d += elm.Line().at((3.5, y_top)).right().length(1.5)
    d += elm.Dot()
    d += elm.Diode().down().length(2.5).label('$D_1$', loc='right')
    d += elm.Dot()
    d += elm.Line().left().length(1.5)

    # Rs
    d += elm.Line().at((5.0, y_top)).right().length(0.5)
    d += elm.Resistor().right().label('$R_s$', loc='top')
    d += elm.Dot().label('$V_{sc}$ (+)', loc='right')

    d += elm.Line().at((0.5, y_bot)).right().length(6.5)
    d += elm.Dot().label('(−)', loc='right')

    return d


def draw_fakidis_ina():
    _check()
    d = create_styled_drawing(unit=3, fontsize=10)

    d += elm.Label().at((3, 5)).label(
        'INA322 Instrumentation Amplifier (G = 40 dB)', fontsize=12)
    d += elm.Label().at((3, 4.5)).label(
        'Fakidis 2020 - Differential current sense', fontsize=9)

    d += elm.Dot().at((0, 3.5)).label('$V_{sense+}$', loc='left')
    d += elm.Line().right().length(1)
    d += elm.Dot().at((0, 2.5)).label('$V_{sense-}$', loc='left')
    d += elm.Line().right().length(1)

    d += elm.Opamp(anchor='in1').at((2, 3)).scale(0.8).label(
        'INA322', loc='center', fontsize=9)
    oa_out = d.here
    d += elm.Line().at(oa_out).right().length(1)
    d += elm.Dot().label('ina_out', loc='right', fontsize=8, color='blue')

    d += elm.Label().at((4, 1.5)).label(
        'Gain = 40 dB (100x)\n'
        'GBW = 700 kHz\n'
        'f_3dB = 7 kHz',
        fontsize=8, halign='left')

    return d


def draw_fakidis_bpf():
    _check()
    d = create_styled_drawing(unit=2.5, fontsize=9)

    d += elm.Label().at((5, 5)).label(
        '2-Stage Active Band-Pass Filter', fontsize=12)
    d += elm.Label().at((5, 4.5)).label(
        'Fakidis 2020 - Passband: 500 Hz - 5 kHz', fontsize=9)

    # Stage 1
    d += elm.Label().at((0, 3.7)).label('Stage 1', fontsize=10, color='blue')
    d += elm.Dot().at((0, 3))
    d += elm.Capacitor().right().label('$C_{HP}$\n470 pF', loc='top')
    hp1 = d.here
    d += elm.Dot()
    d += elm.Resistor().down().length(1.5).label('$R_{HP}$\n100 kOhm', loc='left')
    d += elm.Dot().label('$V_{ref}$', loc='bottom', fontsize=8)

    d += elm.Line().at(hp1).right().length(0.5)
    d += elm.Resistor().right().label('$R_{in}$\n10 kOhm', loc='top')
    inn = d.here
    d += elm.Dot()
    d.add(elm.Opamp(anchor='in2').at((inn[0] + 1.2, inn[1] + 0.5)).scale(0.6))

    d += elm.Line().at(inn).up().length(0.6)
    fb_s = d.here
    d += elm.Resistor().right().length(2.5).label('$R_{fb}$ 10 kOhm', loc='top')
    fb_e = d.here
    d += elm.Line().at(fb_s).up().length(0.4)
    d += elm.Capacitor().right().length(2.5).label('$C_{fb}$ 3.3 nF', loc='top')
    d += elm.Line().down().length(0.4)
    d += elm.Line().at(fb_e).down().length(0.6)
    d += elm.Dot().label('out1', fontsize=7, color='blue', loc='right')

    # Stage 2 arrow
    d += elm.Arrow().right().length(1.5).label('Stage 2\n(identical)', loc='top', fontsize=8)
    d += elm.Dot().label('bpf_out', fontsize=8, loc='right', color='blue')

    return d


def draw_fakidis_dcdc():
    _check()
    d = create_styled_drawing(unit=3, fontsize=10)

    d += elm.Label().at((3.5, 5)).label(
        'Boost DC-DC Converter', fontsize=12)
    d += elm.Label().at((3.5, 4.5)).label(
        'Fakidis 2020 - f_sw = 100 kHz', fontsize=9)

    d += elm.Dot().at((0, 3)).label('$V_{sc}$', loc='left')
    d += elm.Line().down().length(0.5)
    d += elm.Capacitor().down().length(1.5).label('$C_P$\n22 uF', loc='left')
    d += elm.Ground()

    d += elm.Line().at((0, 3)).right().length(0.5)
    d += elm.Inductor2().right().label('L\n10 uH', loc='top')
    sw = d.here
    d += elm.Dot()

    d += elm.Line().down().length(0.5)
    fet = d.add(elm.NFet(anchor='drain'))
    d += elm.Line().at(fet.source).down().length(0.5)
    d += elm.Ground()

    d += elm.Line().at(sw).right().length(0.5)
    d += elm.Diode().right().label('$D_S$', loc='top')
    out = d.here
    d += elm.Dot()

    d += elm.Line().down().length(0.5)
    d += elm.Capacitor().down().length(1.5).label('$C_L$\n100 uF', loc='right')
    d += elm.Ground()

    d += elm.Line().at(out).right().length(1.5)
    d += elm.Dot()
    d += elm.Resistor().down().length(1.8).label('$R_L$\n100 kOhm', loc='right')
    d += elm.Ground()
    d += elm.Line().at(out).up().length(0.5)
    d += elm.Dot().label('$V_{out}$', loc='top')

    return d


def draw_fakidis_tx():
    _check()
    d = create_styled_drawing(unit=3, fontsize=10)

    d += elm.Label().at((3, 5)).label('LED Transmitter Driver', fontsize=12)
    d += elm.Label().at((3, 4.5)).label(
        'Fakidis 2020 - LXM5 LED, 50 mW, a_half = 30 deg', fontsize=9)

    d += elm.Dot().at((0, 3)).label('$V_{mod}$', loc='left')
    d += elm.SourceSin().right().label('OOK', loc='top', fontsize=8)
    d += elm.Resistor().right().label('$R_e$\n12.1 Ohm', loc='top')
    gate = d.here
    d += elm.Dot()
    d += elm.Line().right().length(0.5)
    d += elm.Line().down().length(0.5)
    fet = d.add(elm.NFet(anchor='gate'))
    d += elm.Line().at(fet.drain).up().length(0.5)
    d += elm.LED().up().label('LXM5\n50 mW', loc='right', fontsize=8).reverse()
    d += elm.Line().up().length(0.5)
    d += elm.Dot().label('$V_{CC}$', loc='top')
    d += elm.Line().at(fet.source).down().length(0.5)
    d += elm.Ground()

    d += elm.Label().at((gate[0] + 2, gate[1] + 1.5)).label(
        'a_half = 30 deg\nP = 50 mW\nI_bias = 0.7 A',
        fontsize=8, halign='left')

    return d


# =============================================================================
# SARWAR 2017 - High-Speed OFDM VLC
# =============================================================================

def draw_sarwar_full():
    _check()
    c = _cfg('sarwar2017')
    G = _channel_gain(c)
    d = create_styled_drawing(unit=3, fontsize=10)

    y_tx = 10
    y_rx = 6

    # Title
    d += elm.Label().at((10, 14)).label(
        'Sarwar 2017 - High-Speed OFDM VLC', fontsize=15)
    d += elm.Label().at((10, 13.2)).label(
        'OFDM 16-QAM @ 15 Mbps  |  d = 2.0 m  |  Si Solar Panel', fontsize=10)

    # TX row
    d += elm.Label().at((2, y_tx + 1.5)).label('TRANSMITTER', fontsize=11)
    d += elm.Dot().at((0, y_tx))
    d += elm.RBox(w=2.5, h=1.0).label('OFDM\nModulator', fontsize=9)
    d += elm.Line().right().length(0.8)
    d += elm.RBox(w=1.5, h=0.8).label('DAC', fontsize=9)
    d += elm.Line().right().length(0.8)
    d += elm.RBox(w=2.5, h=1.0).label('LED Driver', fontsize=9)
    d += elm.Line().right().length(0.8)
    d += elm.LED().right().length(2.5).label('Blue 3W LED', loc='bottom', fontsize=9)
    d += elm.Dot()

    # Channel arrow (vertical, down)
    d += elm.Arrow().down().length(y_tx - y_rx).color('red').label(
        f'LOS Channel\nG = {G:.2e}\nd = 2.0 m',
        loc='right', fontsize=9, color='red')
    d += elm.Dot()

    # RX row
    d += elm.Label().at((d.here[0] + 2, y_rx + 1.5)).label('RECEIVER', fontsize=11)
    d += elm.RBox(w=2.5, h=1.0).label('Si Solar Panel\n7.5 cm2', fontsize=9)
    d += elm.Line().right().length(0.8)
    d += elm.RBox(w=1.8, h=0.8).label('TIA', fontsize=9)
    d += elm.Line().right().length(0.8)
    d += elm.RBox(w=1.8, h=0.8).label('ADC', fontsize=9)
    d += elm.Line().right().length(0.8)
    d += elm.RBox(w=2.5, h=1.0).label('OFDM\nDemod', fontsize=9)
    d += elm.Line().right().length(0.8)
    d += elm.Dot().label('Data Out', loc='right', fontsize=10)

    # Info text
    d += elm.Label().at((1, 3.5)).label(
        'OFDM: FFT=256  |  QAM=16  |  80 subcarriers  |  CP=32  |  '
        'Sample Rate=15 MHz  |  Data Rate=15.03 Mbps',
        fontsize=9, halign='left')

    return d


def draw_sarwar_tx():
    _check()
    d = create_styled_drawing(unit=3, fontsize=10)

    d += elm.Label().at((3, 5)).label(
        'Blue LED Transmitter - 3 W', fontsize=12)
    d += elm.Label().at((3, 4.5)).label(
        'Sarwar 2017 - OFDM-modulated output', fontsize=9)

    d += elm.Dot().at((0, 3)).label('OFDM\nSignal', loc='left')
    d += elm.RBox(w=1.5, h=0.6).right().label('DAC', fontsize=9)
    d += elm.Line().right().length(0.5)
    d += elm.RBox(w=1.5, h=0.6).label('Driver\nAmplifier', fontsize=8)
    d += elm.Line().right().length(0.5)

    d += elm.Line().down().length(0.5)
    fet = d.add(elm.NFet(anchor='gate'))
    d += elm.Line().at(fet.drain).up().length(0.5)
    d += elm.LED().up().label('Blue LED\n3 W\na_half = 30 deg', loc='right', fontsize=8).reverse()
    d += elm.Line().up().length(0.5)
    d += elm.Dot().label('$V_{CC}$', loc='top')
    d += elm.Line().at(fet.source).down().length(0.5)
    d += elm.Ground()

    return d


def draw_sarwar_rx():
    _check()
    d = create_styled_drawing(unit=3, fontsize=10)

    d += elm.Label().at((3, 5)).label(
        'Solar Panel Receiver - Si, 7.5 cm2', fontsize=12)
    d += elm.Label().at((3, 4.5)).label(
        'Sarwar 2017 - Direct photodetection', fontsize=9)

    y_top, y_bot = 3.0, 0.5
    d += elm.Dot().at((0.5, y_top))
    d += elm.SourceI().down().length(2.5).label(
        '$I_{ph}$', loc='left').reverse()
    d += elm.Dot().at((0.5, y_bot))

    d += elm.Line().at((0.5, y_top)).right().length(1.5)
    d += elm.Dot()
    d += elm.Capacitor().down().length(2.5).label('$C_j$\n0.1 nF', loc='right')
    d += elm.Dot()
    d += elm.Line().left().length(1.5)

    d += elm.Line().at((2.0, y_top)).right().length(1.5)
    d += elm.Dot()
    d += elm.Resistor().down().length(2.5).label('$R_{sh}$\n10 kOhm', loc='right')
    d += elm.Dot()
    d += elm.Line().left().length(1.5)

    d += elm.Line().at((3.5, y_top)).right().length(1.5)
    d += elm.Dot()
    d += elm.Diode().down().length(2.5).label('$D_1$', loc='right')
    d += elm.Dot()
    d += elm.Line().left().length(1.5)

    d += elm.Line().at((5.0, y_top)).right().length(1)
    d += elm.Dot().label('To TIA', loc='right')
    d += elm.Line().at((0.5, y_bot)).right().length(5.5)
    d += elm.Dot().label('GND', loc='right')

    d += elm.Label().at((0, -0.5)).label(
        'R_resp = 0.40 A/W  |  Low C_j for high BW  |  No energy harvesting',
        fontsize=8, halign='left')

    return d


def draw_sarwar_ofdm():
    _check()
    d = create_styled_drawing(unit=2, fontsize=9)

    d += elm.Label().at((5, 5.5)).label(
        'OFDM Signal Processing Chain', fontsize=12)
    d += elm.Label().at((5, 5.0)).label(
        'Sarwar 2017 - 256-FFT, 16-QAM', fontsize=9)

    # TX chain
    d += elm.Label().at((0, 4.2)).label('TX', fontsize=10, color='blue')
    d += elm.Dot().at((0, 3.5)).label('bits', loc='left', fontsize=8)
    d += elm.RBox(w=1.5, h=0.6).label('QAM\nMap', fontsize=7)
    d += elm.Line().right().length(0.4)
    d += elm.RBox(w=1.5, h=0.6).label('IFFT\n256', fontsize=7)
    d += elm.Line().right().length(0.4)
    d += elm.RBox(w=1.6, h=0.8).label('Add\nCP', fontsize=7)
    d += elm.Line().right().length(0.4)
    d += elm.RBox(w=1.0, h=0.6).label('DAC', fontsize=7)
    d += elm.Line().right().length(0.4)
    d += elm.RBox(w=1.6, h=0.8).label('LED\n+Bias', fontsize=7)
    d += elm.Arrow().right().length(0.5).label('lambda', fontsize=8)

    # RX chain
    d += elm.Label().at((0, 1.7)).label('RX', fontsize=10, color='green')
    d += elm.Dot().at((0, 1)).label('optical', loc='left', fontsize=8)
    d += elm.RBox(w=1.6, h=0.8).label('PV\nCell', fontsize=7)
    d += elm.Line().right().length(0.4)
    d += elm.RBox(w=1.0, h=0.6).label('ADC', fontsize=7)
    d += elm.Line().right().length(0.4)
    d += elm.RBox(w=1.6, h=0.8).label('Rm\nCP', fontsize=7)
    d += elm.Line().right().length(0.4)
    d += elm.RBox(w=1.5, h=0.6).label('FFT\n256', fontsize=7)
    d += elm.Line().right().length(0.4)
    d += elm.RBox(w=1.5, h=0.6).label('QAM\nDemap', fontsize=7)
    d += elm.Arrow().right().length(0.5).label('bits', fontsize=8)

    return d


# =============================================================================
# GONZÁLEZ 2024 - Low-Cost VLC Receiver
# =============================================================================

def draw_gonzalez_full():
    _check()
    c = _cfg('gonzalez2024')
    G = _channel_gain(c)
    d = create_styled_drawing(unit=3, fontsize=10)

    y_tx = 10
    y_rx = 6

    # Title
    d += elm.Label().at((10, 14)).label(
        'Gonzalez 2024 - Low-Cost VLC Receiver', fontsize=15)
    d += elm.Label().at((10, 13.2)).label(
        'OOK Manchester @ 4.8 kbps  |  d = 0.60 m  |  Off-the-shelf', fontsize=10)

    # TX row
    d += elm.Label().at((2, y_tx + 1.5)).label('TRANSMITTER', fontsize=11)
    d += elm.Dot().at((0, y_tx))
    d += elm.SourceSin().right().length(2.5).label('OOK Manchester', loc='top')
    d += elm.Line().right().length(0.8)
    d += elm.RBox(w=2.0, h=1.0).label('LED Driver', fontsize=9)
    d += elm.Line().right().length(0.8)
    d += elm.LED().right().length(2.5).label('White 3W LED', loc='bottom', fontsize=9)
    d += elm.Dot()

    # Channel (vertical down)
    d += elm.Arrow().down().length(y_tx - y_rx).color('red').label(
        f'Channel\nG = {G:.2e}\nd = 60 cm',
        loc='right', fontsize=9, color='red')
    d += elm.Dot()

    # RX row (single signal chain, no data/power split)
    d += elm.Label().at((d.here[0] + 2, y_rx + 1.5)).label('RECEIVER', fontsize=11)
    d += elm.RBox(w=2.5, h=1.0).label('Poly-Si Panel\n66 cm2', fontsize=9)
    d += elm.Line().right().length(0.8)
    d += elm.Resistor().right().length(3).label('Rs = 220 Ohm', loc='top')
    d += elm.Line().right().length(0.8)
    d += elm.RBox(w=2.0, h=1.0).label('Amp\n(165x)', fontsize=9)
    d += elm.Line().right().length(0.8)
    d += elm.RBox(w=2.5, h=1.0).label('Notch Filter\n100 Hz', fontsize=9)
    d += elm.Line().right().length(0.8)
    d += elm.RBox(w=2.5, h=1.0).label('TLV7011\nComparator', fontsize=9)
    d += elm.Line().right().length(0.8)
    d += elm.Dot().label('Data Out', loc='right', fontsize=10)

    # Info
    d += elm.Label().at((1, 3.5)).label(
        'Minimal design: No INA322, no BPF, no DC-DC  |  '
        'Notch filter removes 100 Hz AC line interference',
        fontsize=9, halign='left')

    return d


def draw_gonzalez_tx():
    _check()
    d = create_styled_drawing(unit=3, fontsize=10)

    d += elm.Label().at((3, 5)).label(
        'White LED Transmitter - 3 W', fontsize=12)
    d += elm.Label().at((3, 4.5)).label(
        'González 2024 - OOK Manchester encoding', fontsize=9)

    d += elm.Dot().at((0, 3)).label('Data\nBits', loc='left')
    d += elm.RBox(w=1.8, h=0.6).right().label('Manchester\nEncoder', fontsize=8)
    d += elm.Line().right().length(0.5)
    d += elm.RBox(w=1.5, h=0.6).label('LED\nDriver', fontsize=8)
    d += elm.Line().right().length(0.5)
    d += elm.LED().right().label('White LED\n3 W\na_half = 30 deg', loc='bottom', fontsize=8)
    d += elm.Arrow().right().length(1).label('lambda', fontsize=10)

    return d


def draw_gonzalez_solar():
    _check()
    d = create_styled_drawing(unit=3, fontsize=10)

    d += elm.Label().at((3, 5)).label(
        'Solar Panel - Poly-Si, 66 cm2', fontsize=12)
    d += elm.Label().at((3, 4.5)).label(
        'González 2024 - Largest receiver area', fontsize=9)

    y_top, y_bot = 3.0, 0.5
    d += elm.Dot().at((0.5, y_top))
    d += elm.SourceI().down().length(2.5).label(
        '$I_{ph}$', loc='left').reverse()
    d += elm.Dot().at((0.5, y_bot))

    d += elm.Line().at((0.5, y_top)).right().length(1.5)
    d += elm.Dot()
    d += elm.Capacitor().down().length(2.5).label('$C_j$\n14.5 nF', loc='right')
    d += elm.Dot()
    d += elm.Line().left().length(1.5)

    d += elm.Line().at((2.0, y_top)).right().length(1.5)
    d += elm.Dot()
    d += elm.Resistor().down().length(2.5).label('$R_{sh}$\n200 kOhm', loc='right')
    d += elm.Dot()
    d += elm.Line().left().length(1.5)

    d += elm.Line().at((3.5, y_top)).right().length(1.5)
    d += elm.Dot()
    d += elm.Diode().down().length(2.5).label('$D_1$', loc='right')
    d += elm.Dot()
    d += elm.Line().left().length(1.5)

    # Rs (high value!)
    d += elm.Line().at((5.0, y_top)).right().length(0.5)
    d += elm.Resistor().right().label('$R_{sense}$\n220 Ohm', loc='top')
    d += elm.Dot().label('To Amp', loc='right')

    d += elm.Line().at((0.5, y_bot)).right().length(6.5)
    d += elm.Dot().label('GND', loc='right')

    d += elm.Label().at((0, -0.5)).label(
        'R_resp = 0.40 A/W  |  A = 66 cm2  |  R_{sh} = 200 kOhm (high)',
        fontsize=8, halign='left')

    return d


def draw_gonzalez_amp():
    _check()
    d = create_styled_drawing(unit=3, fontsize=10)

    d += elm.Label().at((4, 5)).label(
        'Amplifier + Notch Filter', fontsize=12)
    d += elm.Label().at((4, 4.5)).label(
        'González 2024 - Gain = 165x + Notch @ 100 Hz (Q=30)', fontsize=9)

    # Amp
    d += elm.Dot().at((0, 3)).label('$V_{sense}$', loc='left')
    d += elm.Line().right().length(1)
    d += elm.Opamp(anchor='in1').at((2, 3)).scale(0.8).label(
        'Amp\n165x', loc='center', fontsize=9)
    amp_out = d.here

    d += elm.Line().at(amp_out).right().length(1)
    d += elm.Dot()

    # Notch (block)
    d += elm.RBox(w=2, h=0.8).right().label(
        'Notch Filter\nf0 = 100 Hz\nQ = 30', fontsize=8)
    d += elm.Line().right().length(0.5)
    d += elm.Dot().label('To Comparator', loc='right', fontsize=8)

    d += elm.Label().at((0, 1)).label(
        'Purpose: Removes 50/60 Hz power line harmonics\n'
        'No dedicated BPF - relies on amp bandwidth',
        fontsize=8, halign='left')

    return d


def draw_gonzalez_comp():
    _check()
    d = create_styled_drawing(unit=3, fontsize=10)

    d += elm.Label().at((2.5, 4)).label(
        'TLV7011 Comparator', fontsize=12)
    d += elm.Label().at((2.5, 3.5)).label(
        'González 2024 - Data Recovery', fontsize=9)

    d += elm.Dot().at((0, 2.5)).label('$V_{filtered}$', loc='left')
    d += elm.Line().right().length(1)
    d += elm.Dot().at((0, 1.5)).label('$V_{th}$', loc='left')
    d += elm.Line().right().length(1)

    d += elm.Opamp(anchor='in1').at((2, 2)).scale(0.8).label(
        'TLV7011', loc='center', fontsize=9)
    d += elm.Line().right().length(1)
    d += elm.Dot().label('$d_{out}$\n(Manchester)', loc='right')

    d += elm.Label().at((1, 0)).label(
        't_pd = 260 ns  |  I_q = 335 nA  |  Push-pull output',
        fontsize=8, halign='left')

    return d


# =============================================================================
# OLIVEIRA 2024 - Laser MIMO VLC
# =============================================================================

def draw_oliveira_full():
    _check()
    c = _cfg('oliveira2024')
    G = _channel_gain(c)
    d = create_styled_drawing(unit=3, fontsize=10)

    y_tx = 10
    y_rx = 6

    # Title
    d += elm.Label().at((10, 14)).label(
        'Oliveira 2024 - Laser MIMO VLC', fontsize=15)
    d += elm.Label().at((10, 13.2)).label(
        'OFDM 64-QAM @ 25.7 Mbps  |  d = 0.5 m  |  3x3 PD Array', fontsize=10)

    # TX row
    d += elm.Label().at((2, y_tx + 1.5)).label('TRANSMITTER', fontsize=11)
    d += elm.Dot().at((0, y_tx))
    d += elm.RBox(w=2.5, h=1.0).label('OFDM\n64-QAM', fontsize=9)
    d += elm.Line().right().length(0.8)
    d += elm.RBox(w=1.5, h=0.8).label('DAC', fontsize=9)
    d += elm.Line().right().length(0.8)
    d += elm.RBox(w=2.5, h=1.0).label('Laser Driver', fontsize=9)
    d += elm.Line().right().length(0.8)
    d += elm.Diode().right().length(2.5).label('Red Laser\n658 nm, 100 mW',
                                                loc='bottom', fontsize=9)
    d += elm.Dot()

    # Channel (vertical down)
    d += elm.Arrow().down().length(y_tx - y_rx).color('red').label(
        f'LOS Channel\nG = {G:.2e}\nd = 0.5 m',
        loc='right', fontsize=9, color='red')
    d += elm.Dot()

    # RX row
    d += elm.Label().at((d.here[0] + 2, y_rx + 1.5)).label('RECEIVER', fontsize=11)
    d += elm.RBox(w=3.0, h=1.2).label('PDBC171SM\n3x3 PD Array\n0.693 cm2', fontsize=9)
    d += elm.Line().right().length(0.8)
    d += elm.RBox(w=1.8, h=0.8).label('ADC', fontsize=9)
    d += elm.Line().right().length(0.8)
    d += elm.RBox(w=2.5, h=1.0).label('OFDM\nDemod', fontsize=9)
    d += elm.Line().right().length(0.8)
    d += elm.Dot().label('Data Out', loc='right', fontsize=10)

    # Info
    d += elm.Label().at((1, 3.5)).label(
        '1024-FFT  |  500 subcarriers  |  No power harvesting  |  '
        'Highest data rate: 25.7 Mbps',
        fontsize=9, halign='left')

    return d


def draw_oliveira_tx():
    _check()
    d = create_styled_drawing(unit=3, fontsize=10)

    d += elm.Label().at((3, 5)).label(
        'Red Laser Transmitter - 658 nm', fontsize=12)
    d += elm.Label().at((3, 4.5)).label(
        'Oliveira 2024 - 100 mW, a_half = 10 deg', fontsize=9)

    d += elm.Dot().at((0, 3)).label('OFDM\nSignal', loc='left')
    d += elm.RBox(w=1.5, h=0.6).right().label('DAC', fontsize=9)
    d += elm.Line().right().length(0.5)
    d += elm.RBox(w=1.5, h=0.6).label('Current\nDriver', fontsize=8)
    d += elm.Line().right().length(0.5)
    d += elm.Diode().right().label('Laser Diode\n658 nm\n100 mW', loc='bottom', fontsize=8)
    d += elm.Arrow().right().length(1).color('red').label('lambda = 658 nm', fontsize=8, color='red')

    d += elm.Label().at((0, 1)).label(
        'Narrow beam (a_half = 10 deg)  |  Class 3B laser safety  |  Direct modulation',
        fontsize=8, halign='left')

    return d


def draw_oliveira_rx():
    _check()
    d = create_styled_drawing(unit=2.5, fontsize=9)

    d += elm.Label().at((4, 6)).label(
        'PDBC171SM 3x3 Photodiode Array', fontsize=12)
    d += elm.Label().at((4, 5.5)).label(
        'Oliveira 2024 - A = 0.693 cm2, R_resp = 0.36 A/W', fontsize=9)

    # Draw 3x3 grid
    for row in range(3):
        for col in range(3):
            x = 1.5 + col * 2
            y = 4 - row * 1.2
            d += elm.Diode().at((x, y)).right().length(1.5).label(
                f'PD{row*3+col+1}', loc='top', fontsize=7)

    # Connections
    d += elm.Label().at((0.5, 0.2)).label(
        'Config: 3x3 array for spatial diversity andMIMO\n'
        'Each PD: ~0.077 cm2  |  C_j = 1.0 nF (total)\n'
        'Direct to ADC - no analog amplification',
        fontsize=8, halign='left')

    return d


def draw_oliveira_ofdm():
    _check()
    d = create_styled_drawing(unit=2, fontsize=9)

    d += elm.Label().at((5, 5.5)).label(
        'OFDM Signal Chain - 1024-FFT, 64-QAM', fontsize=12)
    d += elm.Label().at((5, 5.0)).label(
        'Oliveira 2024 - 500 subcarriers, 25.7 Mbps', fontsize=9)

    # TX
    d += elm.Label().at((0, 4.2)).label('TX', fontsize=10, color='blue')
    d += elm.Dot().at((0, 3.5)).label('bits', loc='left', fontsize=8)
    d += elm.RBox(w=1.5, h=0.6).label('64-QAM\nMap', fontsize=7)
    d += elm.Line().right().length(0.4)
    d += elm.RBox(w=1.5, h=0.6).label('IFFT\n1024', fontsize=7)
    d += elm.Line().right().length(0.4)
    d += elm.RBox(w=1.6, h=0.8).label('Add\nCP', fontsize=7)
    d += elm.Line().right().length(0.4)
    d += elm.RBox(w=1.0, h=0.6).label('DAC', fontsize=7)
    d += elm.Line().right().length(0.4)
    d += elm.RBox(w=1.6, h=0.8).label('Laser\nDiode', fontsize=7)
    d += elm.Arrow().right().length(0.5).label('lambda', fontsize=8)

    # RX
    d += elm.Label().at((0, 1.7)).label('RX', fontsize=10, color='green')
    d += elm.Dot().at((0, 1)).label('optical', loc='left', fontsize=8)
    d += elm.RBox(w=1.6, h=0.8).label('PD\nArray', fontsize=7)
    d += elm.Line().right().length(0.4)
    d += elm.RBox(w=1.0, h=0.6).label('ADC', fontsize=7)
    d += elm.Line().right().length(0.4)
    d += elm.RBox(w=1.6, h=0.8).label('Rm\nCP', fontsize=7)
    d += elm.Line().right().length(0.4)
    d += elm.RBox(w=1.5, h=0.6).label('FFT\n1024', fontsize=7)
    d += elm.Line().right().length(0.4)
    d += elm.RBox(w=1.5, h=0.6).label('64-QAM\nDemap', fontsize=7)
    d += elm.Arrow().right().length(0.5).label('bits', fontsize=8)

    return d


# =============================================================================
# XU 2024 - Sunlight-Duo
# =============================================================================

def draw_xu_full():
    _check()
    c = _cfg('xu2024')
    G = _channel_gain(c)
    d = create_styled_drawing(unit=3, fontsize=10)

    y_data, y_main, y_power = 14, 10, 6

    # Title
    d += elm.Label().at((10, 17)).label(
        'Xu 2024 - Sunlight-Duo Harvesting', fontsize=15)
    d += elm.Label().at((10, 16.2)).label(
        'BFSK @ 400 bps  |  d = 5.0 m  |  Sunlight modulation via LC shutter',
        fontsize=10)

    # TX row (sunlight + LC shutter)
    d += elm.Label().at((2, y_main + 1.5)).label('TRANSMITTER', fontsize=11)
    d += elm.Dot().at((0, y_main))
    d += elm.RBox(w=2.5, h=1.0).label('Sunlight\n600 mW', fontsize=9)
    d += elm.Line().right().length(0.8)
    d += elm.RBox(w=3.0, h=1.2).label('LC Shutter\n(BFSK modulation)', fontsize=9)
    d += elm.Line().right().length(0.5)
    d += elm.Arrow().right().length(3.5).color('red').label(
        f'Free Space\nG = {G:.2e}\nd = 5.0 m',
        loc='top', fontsize=9, color='red')
    d += elm.Line().right().length(0.5)
    d += elm.Label().at((d.here[0] + 2, y_main + 1.5)).label('RECEIVER', fontsize=11)
    d += elm.RBox(w=3.0, h=1.2).label('GaAs 16-cell\nSolar Array\n16 cm2', fontsize=9)
    sc_out = d.here
    d += elm.Dot()

    # Branch up to data path
    d += elm.Line().at(sc_out).up().length(y_data - y_main)
    d += elm.Dot()
    d += elm.Label().at((sc_out[0] + 4, y_data + 1.5)).label(
        'DATA PATH', fontsize=11, color='#4B0082')
    d += elm.Line().right().length(0.8)
    d += elm.RBox(w=2.5, h=1.0).label('BFSK\nDemodulator', fontsize=9)
    d += elm.Line().right().length(0.8)
    d += elm.Dot().label('Data Out\n(400 bps)', loc='right', fontsize=10)

    # Branch down to power path
    d += elm.Line().at(sc_out).down().length(y_main - y_power)
    d += elm.Dot()
    d += elm.Label().at((sc_out[0] + 4, y_power + 1.5)).label(
        'POWER PATH', fontsize=11, color='#B8860B')
    d += elm.Line().right().length(0.8)
    d += elm.RBox(w=2.5, h=1.0).label('Energy\nHarvesting', fontsize=9)
    d += elm.Line().right().length(0.8)
    d += elm.Dot().label('P_harv', loc='right', fontsize=10)

    # Info
    d += elm.Label().at((1, 3.5)).label(
        'BFSK: f0=1600 Hz, f1=2000 Hz  |  '
        'Passive TX (sunlight)  |  Farthest range: 5.0 m',
        fontsize=9, halign='left')

    return d


def draw_xu_tx():
    _check()
    d = create_styled_drawing(unit=3, fontsize=10)

    d += elm.Label().at((3.5, 5.5)).label(
        'LC Shutter Transmitter (Sunlight Modulation)', fontsize=12)
    d += elm.Label().at((3.5, 5.0)).label(
        'Xu 2024 - Passive optical source', fontsize=9)

    # Sunlight
    d += elm.Label().at((0, 3.5)).label('SUN', fontsize=12)
    d += elm.Arrow().at((0.8, 3)).right().length(1.5).label(
        'Sunlight\n600 mW', loc='top', fontsize=9)

    # LC shutter (drawn as switch)
    d += elm.RBox(w=2, h=1).label(
        'LC Shutter\n(Liquid Crystal)', fontsize=9)
    lc_out = d.here

    # BFSK control
    d += elm.Line().at((lc_out[0] - 1, lc_out[1] - 0.8)).down().length(1)
    d += elm.Dot().label('BFSK Control\nf0 = 1600 Hz\nf1 = 2000 Hz',
                         loc='bottom', fontsize=8)

    # Output
    d += elm.Arrow().at(lc_out).right().length(2).color('red').label(
        'Modulated\nSunlight', loc='top', fontsize=9, color='red')

    d += elm.Label().at((0, 0.5)).label(
        'Principle: LC shutter varies optical transmittance\n'
        'at BFSK frequencies to encode data onto sunlight',
        fontsize=8, halign='left')

    return d


def draw_xu_solar():
    _check()
    d = create_styled_drawing(unit=2.5, fontsize=9)

    d += elm.Label().at((4, 6)).label(
        'GaAs 16-Cell Solar Array', fontsize=12)
    d += elm.Label().at((4, 5.5)).label(
        'Xu 2024 - 2S x 8P config, 16 cm2 total', fontsize=9)

    # Draw 2x8 array (simplified as 2 rows)
    for row in range(2):
        y = 4 - row * 1.5
        d += elm.Dot().at((0.5, y))
        for col in range(4):
            x = 0.5 + col * 1.8
            d += elm.Diode().at((x, y)).right().length(1.5).label(
                f'PV{row*4+col+1}', loc='top', fontsize=7)
        d += elm.Dot()

    # Labels
    d += elm.Label().at((0.5, 0.5)).label(
        '2 cells in series (higher voltage) x 8 parallel (higher current)\n'
        'R_resp = 0.457 A/W  |  C_j = 50 nF  |  R_sh = 100 kOhm\n'
        'GaAs for outdoor sunlight spectrum matching',
        fontsize=8, halign='left')

    return d


def draw_xu_bfsk():
    _check()
    d = create_styled_drawing(unit=2, fontsize=9)

    d += elm.Label().at((5, 5.5)).label(
        'BFSK Demodulation Chain', fontsize=12)
    d += elm.Label().at((5, 5.0)).label(
        'Xu 2024 - f0 = 1600 Hz (bit 0), f1 = 2000 Hz (bit 1)', fontsize=9)

    # Signal flow
    d += elm.Dot().at((0, 3.5)).label('$V_{pv}$', loc='left', fontsize=8)
    d += elm.RBox(w=1.5, h=0.6).label('BPF\n1-3 kHz', fontsize=7)
    d += elm.Line().right().length(0.4)
    d += elm.Dot()
    branch = d.here

    # Upper path (f0)
    d += elm.Line().at(branch).up().length(1.2)
    d += elm.Dot()
    d += elm.RBox(w=1.5, h=0.5).right().label('Corr f0\n1600 Hz', fontsize=7)
    d += elm.Line().right().length(0.4)
    d += elm.RBox(w=1.0, h=0.5).label('Energy', fontsize=7)
    f0_out = d.here

    # Lower path (f1)
    d += elm.Line().at(branch).down().length(1.2)
    d += elm.Dot()
    d += elm.RBox(w=1.5, h=0.5).right().label('Corr f1\n2000 Hz', fontsize=7)
    d += elm.Line().right().length(0.4)
    d += elm.RBox(w=1.0, h=0.5).label('Energy', fontsize=7)
    f1_out = d.here

    # Decision
    d += elm.Line().at(f0_out).right().length(0.5)
    d += elm.Line().down().length(1.2)
    d += elm.Dot()
    d += elm.Line().at(f1_out).right().length(0.5)
    d += elm.Line().up().length(1.2)

    d += elm.RBox(w=1.5, h=0.6).right().label('Compare\nDecide', fontsize=7)
    d += elm.Arrow().right().length(0.5).label('bits', fontsize=8)

    return d


# =============================================================================
# CORREA 2025 - Greenhouse VLC
# =============================================================================

def draw_correa_full():
    _check()
    c = _cfg('correa2025')
    G = _channel_gain(c)
    d = create_styled_drawing(unit=3, fontsize=10)

    y_tx = 10
    y_rx = 6

    # Title
    d += elm.Label().at((10, 14)).label(
        'Correa 2025 - Greenhouse VLC', fontsize=15)
    d += elm.Label().at((10, 13.2)).label(
        'PWM-ASK @ 10 kbps  |  d = 0.85 m  |  30 W LED panel', fontsize=10)

    # TX row
    d += elm.Label().at((2, y_tx + 1.5)).label('TRANSMITTER', fontsize=11)
    d += elm.Dot().at((0, y_tx))
    d += elm.SourceSin().right().length(2.5).label('PWM-ASK Signal', loc='top')
    d += elm.Line().right().length(0.8)
    d += elm.RBox(w=2.0, h=1.0).label('LED Driver', fontsize=9)
    d += elm.Line().right().length(0.8)
    d += elm.RBox(w=3.0, h=1.2).label('30 W LED Panel\na_half = 60 deg', fontsize=9)
    d += elm.Dot()

    # Channel (vertical down)
    d += elm.Arrow().down().length(y_tx - y_rx).color('red').label(
        f'Channel\nG = {G:.2e}\nd = 85 cm',
        loc='right', fontsize=9, color='red')
    d += elm.Dot()

    # RX row
    d += elm.Label().at((d.here[0] + 2, y_rx + 1.5)).label('RECEIVER', fontsize=11)
    d += elm.RBox(w=2.5, h=1.0).label('Poly-Si Panel\n66 cm2', fontsize=9)
    d += elm.Line().right().length(0.8)
    d += elm.Resistor().right().length(3).label('Rs = 1 Ohm', loc='top')
    d += elm.Line().right().length(0.8)
    d += elm.RBox(w=2.5, h=1.0).label('TL072 Amp\n(20.8 dB)', fontsize=9)
    d += elm.Line().right().length(0.8)
    d += elm.Dot().label('Data Out', loc='right', fontsize=10)

    # Info
    d += elm.Label().at((1, 3.5)).label(
        'Dual-use: plant growth lighting + data  |  '
        'PWM @ 10 Hz dimming  |  ASK carrier @ 10 kHz  |  '
        'TL072 op-amp (3 MHz GBW)',
        fontsize=9, halign='left')

    return d


def draw_correa_tx():
    _check()
    d = create_styled_drawing(unit=3, fontsize=10)

    d += elm.Label().at((3.5, 5.5)).label(
        '30 W LED Panel Transmitter', fontsize=12)
    d += elm.Label().at((3.5, 5.0)).label(
        'Correa 2025 - Greenhouse grow-light + VLC', fontsize=9)

    d += elm.Dot().at((0, 3)).label('Data', loc='left')
    d += elm.RBox(w=1.8, h=0.6).right().label('PWM-ASK\nEncoder', fontsize=8)
    d += elm.Line().right().length(0.5)
    d += elm.RBox(w=1.5, h=0.6).label('LED Panel\nDriver', fontsize=8)
    d += elm.Line().right().length(0.5)

    # LED panel (array representation)
    d += elm.RBox(w=2, h=1.2).label(
        '30 W LED Panel\n(multiple LEDs)\na_half = 60 deg\n3000 mW optical',
        fontsize=8)
    d += elm.Arrow().right().length(1).label('lambda', fontsize=10)

    d += elm.Label().at((0, 1)).label(
        'PWM @ 10 Hz for dimming  |  ASK carrier @ 10 kHz for data\n'
        'Dual-purpose: illumination + communication',
        fontsize=8, halign='left')

    return d


def draw_correa_solar():
    _check()
    d = create_styled_drawing(unit=3, fontsize=10)

    d += elm.Label().at((3, 5)).label(
        'Solar Panel - Poly-Si, 66 cm2', fontsize=12)
    d += elm.Label().at((3, 4.5)).label(
        'Correa 2025 - R_resp = 0.50 A/W', fontsize=9)

    y_top, y_bot = 3.0, 0.5
    d += elm.Dot().at((0.5, y_top))
    d += elm.SourceI().down().length(2.5).label(
        '$I_{ph}$', loc='left').reverse()
    d += elm.Dot().at((0.5, y_bot))

    d += elm.Line().at((0.5, y_top)).right().length(1.5)
    d += elm.Dot()
    d += elm.Capacitor().down().length(2.5).label('$C_j$\n50 nF', loc='right')
    d += elm.Dot()
    d += elm.Line().left().length(1.5)

    d += elm.Line().at((2.0, y_top)).right().length(1.5)
    d += elm.Dot()
    d += elm.Resistor().down().length(2.5).label('$R_{sh}$\n200 kOhm', loc='right')
    d += elm.Dot()
    d += elm.Line().left().length(1.5)

    d += elm.Line().at((3.5, y_top)).right().length(1.5)
    d += elm.Dot()
    d += elm.Diode().down().length(2.5).label('$D_1$', loc='right')
    d += elm.Dot()
    d += elm.Line().left().length(1.5)

    d += elm.Line().at((5.0, y_top)).right().length(0.5)
    d += elm.Resistor().right().label('$R_{sense}$\n1 Ohm', loc='top')
    d += elm.Dot().label('To TL072', loc='right')
    d += elm.Line().at((0.5, y_bot)).right().length(6.5)
    d += elm.Dot().label('GND', loc='right')

    return d


def draw_correa_amp():
    _check()
    d = create_styled_drawing(unit=3, fontsize=10)

    d += elm.Label().at((3, 5)).label(
        'TL072 Dual Op-Amp - Signal Conditioning', fontsize=12)
    d += elm.Label().at((3, 4.5)).label(
        'Correa 2025 - Gain = 20.83 dB (11x), GBW = 3 MHz', fontsize=9)

    d += elm.Dot().at((0, 3)).label('$V_{sense}$', loc='left')
    d += elm.Line().right().length(0.5)

    # Coupling cap
    d += elm.Capacitor().right().label('$C_{in}$\nAC couple', loc='top', fontsize=8)
    d += elm.Line().right().length(0.5)

    d += elm.Opamp(anchor='in1').at((3.5, 3)).scale(0.8).label(
        'TL072', loc='center', fontsize=9)
    oa_out = d.here

    d += elm.Line().at(oa_out).right().length(1)
    d += elm.Dot().label('$V_{out}$\nTo ADC', loc='right')

    d += elm.Label().at((1, 1)).label(
        'Gain: 20.83 dB (~11x)  |  GBW: 3 MHz\n'
        'f_3dB ~ 273 kHz  |  Low-cost, widely available\n'
        'Direct amplification - no INA322 needed',
        fontsize=8, halign='left')

    return d
