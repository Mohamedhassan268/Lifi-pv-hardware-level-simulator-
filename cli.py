#!/usr/bin/env python
"""
LiFi-PV Simulator CLI
=====================
Usage:  python cli.py <command> [options]
"""

import sys, os, argparse, numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ── helpers ──────────────────────────────────────────────────────────────
def _header(title):
    print(f"\n{'='*50}")
    print(f"  {title}")
    print(f"{'='*50}\n")


# ── commands ─────────────────────────────────────────────────────────────

def cmd_test(args):
    """Run all module self-tests."""
    _header("TEST SUITE")
    ok = fail = 0

    def check(name, fn):
        nonlocal ok, fail
        try:
            fn(); ok += 1; print(f"  PASS  {name}")
        except Exception as e:
            fail += 1; print(f"  FAIL  {name}: {e}")

    check("Materials",    lambda: __import__('materials', fromlist=['GAAS']))
    check("Solar Cells",  lambda: __import__('components.solar_cells', fromlist=['KXOB25_04X3F']).KXOB25_04X3F())
    check("LEDs",         lambda: __import__('components.leds', fromlist=['LXM5_PD01']).LXM5_PD01())
    check("Amplifiers",   lambda: [getattr(__import__('components.amplifiers', fromlist=[c]),c)() for c in ['INA322','TLV2379','ADA4891']])
    check("Comparators",  lambda: __import__('components.comparators', fromlist=['TLV7011']).TLV7011())
    check("MOSFETs",      lambda: [getattr(__import__('components.mosfets', fromlist=[c]),c)() for c in ['BSD235N','NTS4409']])
    check("Registry",     lambda: __import__('components', fromlist=['get_component']).get_component('KXOB25-04X3F'))
    check("PRBS-7",       lambda: None if np.sum(__import__('simulation.prbs_generator', fromlist=['generate_prbs']).generate_prbs(7,127))==64 else (_ for _ in ()).throw(ValueError("bad balance")))
    check("OOK waveform", lambda: __import__('simulation.prbs_generator', fromlist=['generate_ook_waveform']).generate_ook_waveform(np.array([1,0,1]), bit_rate=5e3, modulation_depth=0.33, samples_per_bit=100))
    check("BER theory",   lambda: None if 1e-4 < __import__('simulation.analysis', fromlist=['theoretical_ber_ook']).theoretical_ber_ook(10) < 1e-2 else (_ for _ in ()).throw(ValueError("out of range")))
    check("Channel",      lambda: __import__('systems.kadirvelu2021_channel', fromlist=['OpticalChannel']).OpticalChannel().link_budget())
    check("Netlist",      lambda: __import__('systems.kadirvelu2021_netlist', fromlist=['FullSystemNetlist']).FullSystemNetlist().generate())
    check("Simulation",   lambda: __import__('systems.kadirvelu2021', fromlist=['KadirveluSimulation']).KadirveluSimulation().calculate_theoretical_values())

    print(f"\n  {ok} passed, {fail} failed")


def cmd_components(args):
    """List all registered components or inspect one."""
    from components import COMPONENT_REGISTRY, get_component

    if args.name:
        comp = get_component(args.name)
        _header(comp.name)
        for k, v in comp.get_parameters().items():
            print(f"  {k:30s}  {v}")
    else:
        _header("COMPONENT REGISTRY")
        for name in sorted(set(COMPONENT_REGISTRY.keys())):
            cls = COMPONENT_REGISTRY[name]
            print(f"  {name:20s}  {cls.__name__}")


def cmd_channel(args):
    """Show optical channel link budget."""
    from systems.kadirvelu2021_channel import OpticalChannel

    _header("OPTICAL CHANNEL")
    ch = OpticalChannel()
    ch.print_link_budget()

    if args.sweep:
        distances = [10, 20, 30, 40, 50, 60, 80, 100]
        print("\n  Distance sweep:")
        results = ch.distance_sweep(distances)
        for d, p, i in zip(distances, results['P_rx_uW'], results['I_ph_uA']):
            print(f"    {d:4d} cm  ->  P_rx = {p:8.2f} uW,  I_ph = {i:8.2f} uA")


def cmd_netlist(args):
    """Generate a SPICE netlist."""
    from systems.kadirvelu2021_netlist import FullSystemNetlist

    _header("NETLIST GENERATOR")
    gen = FullSystemNetlist()

    if args.type == 'ook':
        netlist = gen.generate_ook_test(
            bit_rate=args.bitrate, n_bits=args.bits,
            modulation_depth=args.mod, fsw=args.fsw)
    elif args.type == 'rx':
        netlist = gen.generate_receiver_only(f_signal=args.freq)
    else:
        netlist = gen.generate(
            source_type='sine', f_signal=args.freq,
            modulation_depth=args.mod, fsw=args.fsw)

    if args.output:
        gen.save(args.output, netlist)
    else:
        print(netlist)


def cmd_params(args):
    """Show all Kadirvelu 2021 paper parameters."""
    from systems.kadirvelu2021 import KadirveluParams

    _header("KADIRVELU 2021 PARAMETERS")
    p = KadirveluParams()

    sections = {
        'Transmitter':  ['LED_RADIATED_POWER_mW','LED_HALF_ANGLE_DEG','LED_DRIVER_RE','LED_GLED','LENS_TRANSMITTANCE'],
        'Channel':      ['DISTANCE_M','THETA_DEG','BETA_DEG'],
        'Solar Cell':   ['SC_AREA_CM2','SC_RESPONSIVITY','SC_CJ_nF','SC_RSH_kOhm','SC_IPH_uA','SC_VMPP_mV','SC_IMPP_uA','SC_PMPP_uW'],
        'INA322':       ['R_SENSE','INA_R1','INA_R2','INA_GAIN_DB','INA_GBW_kHz'],
        'BPF':          ['BPF_RHP','BPF_CHP_pF','BPF_RLP','BPF_CLF_nF','BPF_STAGES'],
        'DC-DC':        ['DCDC_L_uH','DCDC_CP_uF','DCDC_CL_uF','DCDC_RLOAD_kOhm','DCDC_FSW_kHz'],
        'Targets':      ['TARGET_HARVESTED_POWER_uW','TARGET_BER','TARGET_NOISE_RMS_mV'],
    }

    for section, attrs in sections.items():
        print(f"  --- {section} ---")
        for a in attrs:
            print(f"    {a:30s}  {getattr(p, a)}")
        print()

    print(f"  --- Derived ---")
    print(f"    {'Lambertian order m':30s}  {p.lambertian_order():.2f}")
    print(f"    {'Optical gain Gop':30s}  {p.optical_channel_gain():.6e}")


def cmd_simulate(args):
    """Run ngspice transient simulation."""
    from systems.kadirvelu2021 import KadirveluSimulation

    _header("TRANSIENT SIMULATION")
    sim = KadirveluSimulation()
    sim.run_transient(
        t_stop=args.tstop, fsw=args.fsw,
        modulation_depth=args.mod, f_signal=args.freq,
        output_dir=args.output)


def cmd_schematic(args):
    """Generate schematics (requires schemdraw)."""
    from systems.kadirvelu2021_schematic import draw_all_schematics

    out = args.output or './schematics'
    fmt = args.format or 'svg'
    _header(f"SCHEMATICS -> {out}/ ({fmt})")
    draw_all_schematics(out, fmt=fmt)


def cmd_ber(args):
    """Show theoretical BER curve for OOK."""
    from simulation.analysis import theoretical_ber_ook

    _header("THEORETICAL BER (OOK)")
    print(f"  {'SNR (dB)':>10}  {'BER':>12}")
    print(f"  {'-'*10}  {'-'*12}")
    for snr in range(0, 25):
        ber = theoretical_ber_ook(snr)
        bar = '#' * max(1, int(-np.log10(max(ber, 1e-30))))
        print(f"  {snr:10d}  {ber:12.4e}  {bar}")


def cmd_prbs(args):
    """Generate and preview PRBS sequence."""
    from simulation.prbs_generator import generate_prbs

    _header(f"PRBS-{args.order}")
    bits = generate_prbs(order=args.order, n_bits=args.bits)
    ones = int(np.sum(bits))
    print(f"  Bits: {len(bits)},  Ones: {ones} ({ones/len(bits)*100:.1f}%)")
    preview = ''.join(str(b) for b in bits[:80])
    print(f"  Preview: {preview}{'...' if len(bits)>80 else ''}")


def cmd_gui(args):
    """Launch the PyQt6 desktop GUI."""
    _header("LAUNCHING GUI")
    from PyQt6.QtWidgets import QApplication
    from gui.main_window import MainWindow

    app = QApplication(sys.argv)
    app.setApplicationName('LiFi-PV Simulator')
    app.setStyle('Fusion')
    window = MainWindow()
    window.show()
    window.raise_()
    window.activateWindow()
    print("  GUI window opened. Close the window to exit.")
    sys.exit(app.exec())


def cmd_pipeline(args):
    """Run the full TX->Channel->RX co-simulation pipeline."""
    from cosim.system_config import SystemConfig
    from cosim.session import SessionManager
    from cosim.pipeline import SimulationPipeline

    preset = args.preset or 'kadirvelu2021'
    _header(f"COSIM PIPELINE ({preset})")

    cfg = SystemConfig.from_preset(preset)
    if args.bits:
        cfg.n_bits = args.bits
    if args.tstop:
        cfg.t_stop_s = args.tstop

    sm = SessionManager()
    session = sm.create_session(label=preset)
    sm.save_config(session, cfg)
    print(f"  Session: {session}")
    print(f"  Config:  {cfg}")
    print()

    def on_progress(step, status, msg):
        print(f"  [{step}] {status}: {msg}")

    pipe = SimulationPipeline(cfg, session, on_progress=on_progress)
    results = pipe.run_all()

    print()
    for step, r in results.items():
        symbol = {'done': 'OK', 'error': 'FAIL'}.get(r.status, '??')
        print(f"  {symbol}  {step}: {r.message} ({r.duration_s:.1f}s)")
    print(f"\n  Session dir: {session}")


def cmd_presets(args):
    """List or inspect available presets."""
    from cosim.system_config import SystemConfig

    if args.name:
        _header(f"PRESET: {args.name}")
        cfg = SystemConfig.from_preset(args.name)
        import json
        print(json.dumps(cfg.to_dict(), indent=2))
    else:
        _header("AVAILABLE PRESETS")
        for name in SystemConfig.list_presets():
            cfg = SystemConfig.from_preset(name)
            print(f"  {name:20s}  {cfg.paper_reference or '(no reference)'}")


def cmd_validate(args):
    """Run paper validation & generate figures."""
    from papers import PAPERS, run_paper, run_all_papers, list_papers

    if args.list:
        _header("AVAILABLE PAPER VALIDATIONS")
        for name, label, ref in list_papers():
            print(f"  {name:20s}  {label}")
            print(f"  {' '*20}  {ref}")
        return

    if args.paper:
        _header(f"VALIDATE: {args.paper}")
        out = args.output or os.path.join('workspace', f'validation_{args.paper}')
        passed = run_paper(args.paper, out)
        print(f"\n  Result: {'PASS' if passed else 'FAIL'}")
    else:
        _header("VALIDATE ALL PAPERS")
        base = args.output or os.path.join('workspace', 'validation')
        results = run_all_papers(base)
        print("\n" + "=" * 50)
        print("  SUMMARY")
        print("=" * 50)
        for name, passed in results.items():
            label = PAPERS[name]['label']
            print(f"  {'PASS' if passed else 'FAIL'}  {label}")
        total = sum(results.values())
        print(f"\n  {total}/{len(results)} papers passed")


def cmd_ltspice(args):
    """Check LTspice installation."""
    from cosim.ltspice_runner import LTSpiceRunner, find_ltspice

    _header("LTSPICE STATUS")
    path = find_ltspice()
    if path:
        print(f"  Found: {path}")
        runner = LTSpiceRunner(path)
        print(f"  Available: {runner.available}")
    else:
        print("  LTspice not found.")
        print("  Install from: https://www.analog.com/en/resources/design-tools-and-calculators/ltspice-simulator.html")


# ── parser ───────────────────────────────────────────────────────────────

def build_parser():
    p = argparse.ArgumentParser(
        prog='lifi-pv',
        description='Hardware-Faithful LiFi-PV Simulator CLI')
    sub = p.add_subparsers(dest='command', help='command')

    # test
    sub.add_parser('test', help='Run all module self-tests')

    # components
    c = sub.add_parser('components', help='List/inspect components')
    c.add_argument('name', nargs='?', help='Component name to inspect')

    # channel
    ch = sub.add_parser('channel', help='Show optical channel link budget')
    ch.add_argument('--sweep', action='store_true', help='Distance sweep')

    # netlist
    n = sub.add_parser('netlist', help='Generate SPICE netlist')
    n.add_argument('--type', choices=['full','rx','ook'], default='full')
    n.add_argument('--freq', type=float, default=5e3, help='Signal freq (Hz)')
    n.add_argument('--mod', type=float, default=0.33, help='Modulation depth')
    n.add_argument('--fsw', type=float, default=50e3, help='DC-DC fsw (Hz)')
    n.add_argument('--bitrate', type=float, default=5e3, help='OOK bit rate')
    n.add_argument('--bits', type=int, default=100, help='OOK num bits')
    n.add_argument('-o', '--output', help='Save to file')

    # params
    sub.add_parser('params', help='Show paper parameters')

    # simulate
    s = sub.add_parser('simulate', help='Run ngspice transient')
    s.add_argument('--tstop', type=float, default=1e-3)
    s.add_argument('--fsw', type=float, default=50e3)
    s.add_argument('--mod', type=float, default=0.33)
    s.add_argument('--freq', type=float, default=5e3)
    s.add_argument('-o', '--output', help='Output directory')

    # schematic
    sc = sub.add_parser('schematic', help='Generate schematics')
    sc.add_argument('-o', '--output', help='Output directory')
    sc.add_argument('--format', choices=['svg','png','pdf'], default='svg')

    # ber
    sub.add_parser('ber', help='Theoretical BER curve (OOK)')

    # prbs
    pr = sub.add_parser('prbs', help='Generate PRBS sequence')
    pr.add_argument('--order', type=int, default=7)
    pr.add_argument('--bits', type=int, default=127)

    # gui
    sub.add_parser('gui', help='Launch PyQt6 desktop GUI')

    # pipeline
    pl = sub.add_parser('pipeline', help='Run TX->Channel->RX co-simulation')
    pl.add_argument('--preset', default='kadirvelu2021', help='Preset name')
    pl.add_argument('--bits', type=int, help='Override n_bits')
    pl.add_argument('--tstop', type=float, help='Override t_stop (s)')

    # presets
    ps = sub.add_parser('presets', help='List/inspect presets')
    ps.add_argument('name', nargs='?', help='Preset to inspect')

    # ltspice
    sub.add_parser('ltspice', help='Check LTspice installation')

    # validate
    va = sub.add_parser('validate', help='Run paper validation & generate figures')
    va.add_argument('paper', nargs='?', help='Paper key (e.g. kadirvelu2021)')
    va.add_argument('--list', action='store_true', help='List available papers')
    va.add_argument('-o', '--output', help='Output directory')

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()

    commands = {
        'test':       cmd_test,
        'components': cmd_components,
        'channel':    cmd_channel,
        'netlist':    cmd_netlist,
        'params':     cmd_params,
        'simulate':   cmd_simulate,
        'schematic':  cmd_schematic,
        'ber':        cmd_ber,
        'prbs':       cmd_prbs,
        'gui':        cmd_gui,
        'pipeline':   cmd_pipeline,
        'presets':    cmd_presets,
        'ltspice':    cmd_ltspice,
        'validate':   cmd_validate,
    }

    if args.command in commands:
        commands[args.command](args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
