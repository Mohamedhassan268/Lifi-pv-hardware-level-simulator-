"""
Kadirvelu 2021 - Schematic Generation using SchemDraw
======================================================

Generates visual circuit schematics for the LiFi-PV system.

Requirements:
    pip install schemdraw

Usage:
    from kadirvelu2021_schematic import draw_full_system, draw_all_schematics
    
    draw_full_system('schematic.png')
"""

try:
    import schemdraw
    import schemdraw.elements as elm
    SCHEMDRAW_AVAILABLE = True
    SCHEMDRAW_VERSION = schemdraw.__version__
except ImportError:
    SCHEMDRAW_AVAILABLE = False
    SCHEMDRAW_VERSION = None
    print("SchemDraw not installed. Run: pip install schemdraw")


def draw_solar_cell_equivalent(filename=None, show=False):
    """
    Draw solar cell equivalent circuit.
    
    Args:
        filename: Output file (optional)
        show: Display the drawing
    
    Returns:
        schemdraw.Drawing object
    """
    if not SCHEMDRAW_AVAILABLE:
        print("SchemDraw not available")
        return None
    
    d = schemdraw.Drawing()
    d.config(unit=3, fontsize=10)
    
    # Title
    d += elm.Label().at((0, 4)).label('Solar Cell Equivalent Circuit\n(Kadirvelu 2021)', fontsize=12)
    
    # Start point - Photocurrent source
    d += elm.Dot().at((0, 2))
    d += elm.Line().right().length(0.5)
    
    # Photocurrent source
    d += elm.SourceI().down().label('$I_{ph}$\n508µA', loc='left').reverse()
    gnd_start = d.here
    
    # Junction capacitance branch
    d += elm.Line().at((0.5, 2)).right().length(1.5)
    d += elm.Dot()
    d += elm.Capacitor().down().label('$C_j$\n798nF', loc='right')
    d += elm.Dot()
    d += elm.Line().left().length(1.5)
    d += elm.Line().left().length(0.5)
    
    # Shunt resistance branch
    d += elm.Line().at((2, 2)).right().length(1.5)
    d += elm.Dot()
    d += elm.Resistor().down().label('$R_{sh}$\n138.8kΩ', loc='right')
    d += elm.Dot()
    d += elm.Line().left().length(1.5)
    
    # Diode branch
    d += elm.Line().at((3.5, 2)).right().length(1.5)
    d += elm.Dot()
    d += elm.Diode().down().label('$D_1$', loc='right')
    d += elm.Dot()
    d += elm.Line().left().length(1.5)
    
    # Output terminals
    d += elm.Line().at((5, 2)).right().length(1)
    d += elm.Dot().label('$V_{sc}$ +', loc='right')
    
    d += elm.Line().at((5, 0)).right().length(1)
    d += elm.Dot().label('−', loc='right')
    
    # Ground line
    d += elm.Line().at(gnd_start).right().length(6)
    
    if filename:
        d.save(filename)
        print(f"Saved: {filename}")
    
    if show:
        d.draw()
    
    return d


def draw_dcdc_converter(filename=None, show=False):
    """
    Draw boost DC-DC converter.
    
    Args:
        filename: Output file (optional)
        show: Display the drawing
    
    Returns:
        schemdraw.Drawing object
    """
    if not SCHEMDRAW_AVAILABLE:
        print("SchemDraw not available")
        return None
    
    d = schemdraw.Drawing()
    d.config(unit=3, fontsize=10)
    
    # Title
    d += elm.Label().at((0, 4.5)).label('Boost DC-DC Converter\n(Kadirvelu 2021)', fontsize=12)
    
    # Input
    d += elm.Dot().at((0, 2.5)).label('$V_{sc}$', loc='left')
    
    # Input capacitor
    d += elm.Line().down().length(0.5)
    d += elm.Capacitor().down().label('$C_P$\n10µF', loc='left')
    d += elm.Ground()
    
    # Main line
    d += elm.Line().at((0, 2.5)).right().length(0.5)
    
    # Inductor
    d += elm.Inductor2().right().label('L\n22µH', loc='top')
    sw_node = d.here
    d += elm.Dot()
    
    # NMOS switch (down)
    d += elm.Line().down().length(0.3)
    d += elm.NFet(anchor='drain').label('M1\nNTS4409', loc='right', fontsize=9)
    d += elm.Line().down().length(0.3)
    d += elm.Ground()
    
    # Clock label at gate
    d += elm.Label().at((sw_node[0] - 0.8, sw_node[1] - 0.8)).label('φ', fontsize=10)
    
    # Schottky diode (continuing right)
    d += elm.Line().at(sw_node).right().length(0.3)
    d += elm.Diode().right().label('$D_s$', loc='top')
    out_node = d.here
    d += elm.Dot()
    
    # Output capacitor
    d += elm.Line().down().length(0.3)
    d += elm.Capacitor().down().label('$C_L$\n47µF', loc='right')
    d += elm.Ground()
    
    # Load resistor
    d += elm.Line().at(out_node).right().length(1.5)
    d += elm.Dot()
    d += elm.Resistor().down().label('$R_{load}$\n180kΩ', loc='right')
    d += elm.Ground()
    
    # Output voltage label
    d += elm.Line().at(out_node).up().length(0.5)
    d += elm.Dot().label('$V_{out}$', loc='top')
    
    if filename:
        d.save(filename)
        print(f"Saved: {filename}")
    
    if show:
        d.draw()
    
    return d


def draw_bandpass_filter(filename=None, show=False):
    """
    Draw band-pass filter stage.
    
    Args:
        filename: Output file (optional)
        show: Display the drawing
    
    Returns:
        schemdraw.Drawing object
    """
    if not SCHEMDRAW_AVAILABLE:
        print("SchemDraw not available")
        return None
    
    d = schemdraw.Drawing()
    d.config(unit=2.5, fontsize=9)
    
    # Title
    d += elm.Label().at((0, 4)).label('Band-Pass Filter Stage\n(TLV2379)', fontsize=11)
    
    # Input
    d += elm.Dot().at((0, 2)).label('$V_{in}$', loc='left')
    
    # High-pass capacitor
    d += elm.Capacitor().right().label('$C_{HP}$\n482pF', loc='top')
    hp_node = d.here
    d += elm.Dot()
    
    # Resistor to Vref
    d += elm.Resistor().down().label('$R_{HP}$\n33kΩ', loc='left')
    d += elm.Dot().label('$V_{ref}$', loc='bottom')
    
    # Input resistor
    d += elm.Line().at(hp_node).right().length(0.5)
    d += elm.Resistor().right().label('$R_{in}$\n10kΩ', loc='top')
    opamp_in = d.here
    d += elm.Dot()
    
    # Op-amp
    d += elm.Opamp(anchor='in2').scale(0.7)
    opamp_out = d.here
    
    # Feedback path - resistor
    d += elm.Line().at(opamp_in).up().length(1)
    fb_top = d.here
    d += elm.Resistor().right().length(1.5).label('$R_{fb}$ 10kΩ', loc='top')
    d += elm.Dot()
    
    # Feedback capacitor (parallel)
    d += elm.Line().at(fb_top).up().length(0.5)
    d += elm.Capacitor().right().length(1.5).label('$C_{fb}$ 64nF', loc='top')
    d += elm.Line().down().length(0.5)
    
    # Connect to output
    d += elm.Line().down().length(1.5)
    d += elm.Dot().label('$V_{out}$', loc='right')
    
    if filename:
        d.save(filename)
        print(f"Saved: {filename}")
    
    if show:
        d.draw()
    
    return d


def draw_full_system(filename='kadirvelu2021_system.png', show=False):
    """
    Draw complete system block diagram.
    
    Args:
        filename: Output filename (supports .png, .svg, .pdf)
        show: Whether to display the drawing
        
    Returns:
        schemdraw.Drawing object
    """
    if not SCHEMDRAW_AVAILABLE:
        print("Error: SchemDraw not installed. Run: pip install schemdraw")
        return None
    
    d = schemdraw.Drawing()
    d.config(unit=2, fontsize=9)
    
    # =========================================================================
    # TITLE
    # =========================================================================
    d += elm.Label().at((0, 7)).label(
        'Kadirvelu 2021 - LiFi-PV System',
        fontsize=14
    )
    d += elm.Label().at((0, 6.3)).label(
        '"A Circuit for Simultaneous Reception of Data and Power Using a Solar Cell"',
        fontsize=9
    )
    
    # =========================================================================
    # TRANSMITTER SECTION (left side)
    # =========================================================================
    d += elm.Label().at((0, 5.2)).label('── TRANSMITTER ──', fontsize=10, color='blue')
    
    # Signal source
    d += elm.Dot().at((0, 4))
    d += elm.SourceSin().right().label('$V_{mod}$', loc='top')
    
    # Arrow to LED driver (representing the driver block)
    d += elm.Line().right().length(0.5)
    d += elm.Arrow().right().length(1).label('LED\nDriver', loc='top', fontsize=8)
    
    # LED symbol
    d += elm.LED().right().label('LXM5-PD01', loc='bottom', fontsize=8)
    led_pos = d.here
    
    # Optical beam (dashed line with arrow)
    d += elm.Line().right().length(0.5)
    d += elm.Arrow().right().length(2).color('red').label('Optical\n$P_e$=9.3mW', loc='top', fontsize=8, color='red')
    
    # =========================================================================
    # RECEIVER SECTION (right side)
    # =========================================================================
    d += elm.Label().at((8, 5.2)).label('── RECEIVER ──', fontsize=10, color='green')
    
    # Solar cell (box representation using basic elements)
    d += elm.Dot().at((8, 4)).label('', loc='left')
    
    # Draw a simple box for solar cell
    d += elm.Line().right().length(0.3)
    sc_start = d.here
    d += elm.RBox(w=1.5, h=0.8).label('Solar\nCell', fontsize=8)
    d += elm.Line().right().length(0.3)
    sc_out = d.here
    d += elm.Dot()
    
    # =========================================================================
    # DATA PATH (top branch)
    # =========================================================================
    d += elm.Label().at((10, 5)).label('Data Path', fontsize=8, color='purple')
    
    # Branch up
    d += elm.Line().at(sc_out).up().length(0.8)
    d += elm.Dot()
    d += elm.Line().right().length(0.3)
    
    # Rsense
    d += elm.Resistor().right().label('$R_s$\n1Ω', loc='top', fontsize=7)
    d += elm.Line().right().length(0.3)
    
    # INA322 block
    d += elm.RBox(w=1.2, h=0.6).label('INA322', fontsize=7)
    d += elm.Line().right().length(0.3)
    
    # BPF block
    d += elm.RBox(w=1.2, h=0.6).label('BPF', fontsize=7)
    d += elm.Line().right().length(0.3)
    
    # Comparator block
    d += elm.RBox(w=1, h=0.6).label('COMP', fontsize=7)
    d += elm.Line().right().length(0.3)
    
    # Output
    d += elm.Dot().label('$d_{out}$', loc='right')
    
    # =========================================================================
    # POWER PATH (bottom branch)
    # =========================================================================
    d += elm.Label().at((10, 2.5)).label('Power Path', fontsize=8, color='orange')
    
    # Branch down
    d += elm.Line().at(sc_out).down().length(0.8)
    d += elm.Dot()
    d += elm.Line().right().length(0.3)
    
    # DC-DC block
    d += elm.RBox(w=1.5, h=0.8).label('DC-DC\nBoost', fontsize=7)
    d += elm.Line().right().length(0.3)
    d += elm.Dot()
    dcdc_out = d.here
    
    # Load
    d += elm.Line().right().length(0.5)
    d += elm.Resistor().down().label('$R_L$\n180kΩ', fontsize=7)
    d += elm.Ground()
    
    # Vout label
    d += elm.Dot().at(dcdc_out).label('$V_{out}$', loc='top')
    
    # =========================================================================
    # PARAMETERS BOX
    # =========================================================================
    d += elm.Label().at((0, 1.5)).label(
        '┌─────────────────────────────┐\n'
        '│ System Parameters:          │\n'
        '│ • Distance: 32.5 cm         │\n'
        '│ • Rλ = 0.457 A/W            │\n'
        '│ • Cj = 798 nF               │\n'
        '│ • Rsh = 138.8 kΩ            │\n'
        '│ • BW: 700Hz - 10kHz         │\n'
        '└─────────────────────────────┘',
        fontsize=8, halign='left'
    )
    
    # =========================================================================
    # TARGETS BOX
    # =========================================================================
    d += elm.Label().at((8, 1.5)).label(
        '┌─────────────────────────────┐\n'
        '│ Validation Targets:         │\n'
        '│ • P_harv = 223 µW           │\n'
        '│ • BER = 1.008×10⁻³          │\n'
        '│ • Noise = 7.77 mVrms        │\n'
        '│ • η(50kHz) = 67%            │\n'
        '└─────────────────────────────┘',
        fontsize=8, halign='left'
    )
    
    # Save
    if filename:
        d.save(filename)
        print(f"Schematic saved to: {filename}")
    
    if show:
        try:
            d.draw()
        except:
            pass
    
    return d


def draw_receiver_detail(filename='kadirvelu2021_receiver.png', show=False):
    """
    Draw detailed receiver circuit.
    
    Args:
        filename: Output filename
        show: Whether to display
        
    Returns:
        schemdraw.Drawing object
    """
    if not SCHEMDRAW_AVAILABLE:
        print("Error: SchemDraw not installed")
        return None
    
    d = schemdraw.Drawing()
    d.config(unit=2.5, fontsize=9)
    
    # Title
    d += elm.Label().at((0, 5)).label('Data Receiver Chain (Kadirvelu 2021)', fontsize=12)
    
    # Solar cell current source
    d += elm.Dot().at((0, 3))
    d += elm.SourceI().down().label('$I_{sc}$', loc='left').reverse()
    d += elm.Ground()
    
    # Current sense
    d += elm.Line().at((0, 3)).right().length(0.5)
    d += elm.Resistor().right().label('$R_{sense}$\n1Ω', loc='top')
    sense_out = d.here
    d += elm.Dot()
    
    # Ground return
    d += elm.Line().down().length(0.5)
    d += elm.Ground()
    
    # INA322
    d += elm.Line().at(sense_out).right().length(0.5)
    d += elm.Opamp(anchor='in1').label('INA322\n(40dB)', loc='bottom', fontsize=8)
    
    # BPF Stage 1
    d += elm.Line().right().length(0.5)
    d += elm.Opamp(anchor='in1').label('BPF-1\nTLV2379', loc='bottom', fontsize=8)
    
    # BPF Stage 2
    d += elm.Line().right().length(0.5)
    d += elm.Opamp(anchor='in1').label('BPF-2\nTLV2379', loc='bottom', fontsize=8)
    
    # Comparator
    d += elm.Line().right().length(0.5)
    d += elm.Opamp(anchor='in1').label('COMP\nTLV7011', loc='bottom', fontsize=8)
    
    # Output
    d += elm.Line().right().length(0.5)
    d += elm.Dot().label('$d_{out}$', loc='right')
    
    # Filter specs
    d += elm.Label().at((4, 1)).label('BPF: 700Hz - 10kHz\nTotal Gain: 60dB', fontsize=9)
    
    if filename:
        d.save(filename)
        print(f"Saved: {filename}")
    
    if show:
        try:
            d.draw()
        except:
            pass
    
    return d


def draw_all_schematics(output_dir='.'):
    """
    Generate all schematic diagrams.
    
    Args:
        output_dir: Directory to save files
    """
    import os
    
    if not SCHEMDRAW_AVAILABLE:
        print("Error: SchemDraw not installed. Run: pip install schemdraw")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Generating schematics (schemdraw v{SCHEMDRAW_VERSION})...")
    print("=" * 50)
    
    # Full system
    draw_full_system(os.path.join(output_dir, 'kadirvelu2021_system.png'), show=False)
    
    # Solar cell equivalent
    draw_solar_cell_equivalent(os.path.join(output_dir, 'kadirvelu2021_solarcell.png'), show=False)
    
    # DC-DC converter
    draw_dcdc_converter(os.path.join(output_dir, 'kadirvelu2021_dcdc.png'), show=False)
    
    # Band-pass filter
    draw_bandpass_filter(os.path.join(output_dir, 'kadirvelu2021_bpf.png'), show=False)
    
    # Receiver detail
    draw_receiver_detail(os.path.join(output_dir, 'kadirvelu2021_receiver.png'), show=False)
    
    print("=" * 50)
    print(f"All schematics saved to: {output_dir}")


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate Kadirvelu 2021 schematics')
    parser.add_argument('--output', '-o', type=str, default='.',
                        help='Output directory (default: current)')
    parser.add_argument('--format', '-f', type=str, default='png',
                        choices=['png', 'svg', 'pdf'],
                        help='Output format (default: png)')
    parser.add_argument('--all', action='store_true',
                        help='Generate all schematics')
    parser.add_argument('--system', action='store_true',
                        help='Generate system block diagram only')
    parser.add_argument('--solarcell', action='store_true',
                        help='Generate solar cell equivalent circuit')
    parser.add_argument('--dcdc', action='store_true',
                        help='Generate DC-DC converter schematic')
    parser.add_argument('--bpf', action='store_true',
                        help='Generate band-pass filter schematic')
    parser.add_argument('--receiver', action='store_true',
                        help='Generate receiver chain schematic')
    
    args = parser.parse_args()
    
    if not SCHEMDRAW_AVAILABLE:
        print("Error: SchemDraw not installed.")
        print("Install with: pip install schemdraw")
        exit(1)
    
    print(f"SchemDraw version: {SCHEMDRAW_VERSION}")
    
    if args.all:
        draw_all_schematics(args.output)
    elif args.system:
        draw_full_system(f'{args.output}/kadirvelu2021_system.{args.format}')
    elif args.solarcell:
        draw_solar_cell_equivalent(f'{args.output}/kadirvelu2021_solarcell.{args.format}')
    elif args.dcdc:
        draw_dcdc_converter(f'{args.output}/kadirvelu2021_dcdc.{args.format}')
    elif args.bpf:
        draw_bandpass_filter(f'{args.output}/kadirvelu2021_bpf.{args.format}')
    elif args.receiver:
        draw_receiver_detail(f'{args.output}/kadirvelu2021_receiver.{args.format}')
    else:
        # Default: generate system schematic
        draw_full_system('kadirvelu2021_system.png')
