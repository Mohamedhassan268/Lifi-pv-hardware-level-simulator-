#!/usr/bin/env python
"""
LiFi-PV Simulator CLI
=====================
Usage:  python cli.py <command> [options]
"""

import sys, os, argparse, logging, numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── logging setup ───────────────────────────────────────────────────────
def _setup_logging(verbose=False):
    """Configure logging for CLI and all submodules."""
    level = logging.DEBUG if verbose else logging.INFO
    log_dir = Path(__file__).parent / 'workspace'
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / 'simulator.log'

    logging.basicConfig(
        level=level,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout) if verbose else logging.NullHandler(),
        ],
    )

logger = logging.getLogger('lifi_pv')


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

    # Phase 4: Preset smoke tests
    def _check_presets():
        from cosim.system_config import SystemConfig
        from cosim.python_engine import run_python_simulation
        for name in SystemConfig.list_presets():
            cfg = SystemConfig.from_preset(name)
            assert cfg.rx_topology in ('ina_bpf_comp', 'amp_slicer', 'direct'), f"{name}: bad topology"
            d = cfg.to_dict()
            d['simulation_engine'] = 'python'
            d['n_bits'] = 20
            cfg_small = SystemConfig(**d)
            result = run_python_simulation(cfg_small)
            assert 'ber' in result and 'time' in result, f"{name}: missing result keys"
    check("Presets (6)", _check_presets)

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
    from PyQt6.QtCore import QLocale
    from gui.main_window import MainWindow

    app = QApplication(sys.argv)
    app.setApplicationName('LiFi-PV Simulator')
    app.setStyle('Fusion')

    # Force English (Latin) numerals regardless of system locale
    QLocale.setDefault(QLocale(QLocale.Language.English, QLocale.Country.UnitedStates))
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

    if not _preflight_validate(cfg):
        print("\nABORT: configuration has ERROR-level issues. "
              "Run `python cli.py validate-config --preset " + preset + "` for details.")
        sys.exit(1)

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


def _preflight_validate(cfg, *, strict_warnings: bool = False) -> bool:
    """Run validator on a config and print issues. Returns True if OK to run."""
    from cosim.validation import format_issues, has_errors, IssueLevel
    issues = cfg.validate()
    if not issues:
        return True
    print(format_issues(issues))
    if has_errors(issues):
        return False
    if strict_warnings and any(i.level == IssueLevel.WARNING for i in issues):
        return False
    return True


def cmd_validate_config(args):
    """Check one or all presets for parameter issues."""
    from cosim.system_config import SystemConfig
    from cosim.validation import format_issues, has_errors, IssueLevel

    if args.preset:
        names = [args.preset]
    else:
        names = SystemConfig.list_presets()

    exit_code = 0
    for name in names:
        _header(f"VALIDATE-CONFIG: {name}")
        try:
            cfg = SystemConfig.from_preset(name)
        except Exception as exc:
            print(f"ERROR   load: {exc}")
            exit_code = 1
            continue
        issues = cfg.validate()
        print(format_issues(issues))
        if has_errors(issues):
            exit_code = 1
        elif args.strict and any(i.level == IssueLevel.WARNING for i in issues):
            exit_code = 1
        print()
    sys.exit(exit_code)


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
        out = args.output or str(Path('workspace') / f'validation_{args.paper}')
        passed = run_paper(args.paper, out)
        print(f"\n  Result: {'PASS' if passed else 'FAIL'}")
    else:
        _header("VALIDATE ALL PAPERS")
        base = args.output or str(Path('workspace') / 'validation')
        results = run_all_papers(base)
        print("\n" + "=" * 50)
        print("  SUMMARY")
        print("=" * 50)
        for name, passed in results.items():
            label = PAPERS[name]['label']
            print(f"  {'PASS' if passed else 'FAIL'}  {label}")
        total = sum(results.values())
        print(f"\n  {total}/{len(results)} papers passed")


def cmd_compare(args):
    """Run pipeline validation: compare all presets against paper targets."""
    from papers.pipeline_validation import validate_all, validate_preset, cross_validate

    out = args.output or str(Path('workspace') / 'validation_pipeline')

    if args.cross:
        _header("CROSS-VALIDATION: Standalone vs Pipeline")
        cross_validate(output_dir=out, verbose=True)
    elif args.preset:
        _header(f"PIPELINE VALIDATION: {args.preset}")
        result = validate_preset(args.preset, verbose=True)
        status = "PASS" if result['passed'] else "REVIEW"
        print(f"\n  Result: {status}")
    else:
        _header("PIPELINE VALIDATION: All Presets")
        validate_all(output_dir=out, verbose=True)


def cmd_sensitivity(args):
    """Run parameter sensitivity analysis."""
    from scripts.sensitivity_analysis import run_sensitivity_analysis

    presets = [args.preset] if args.preset else None
    _header("SENSITIVITY ANALYSIS")
    run_sensitivity_analysis(
        presets=presets,
        output_dir=args.output,
        fig=args.fig or 'all',
    )


def cmd_thermal(args):
    """Run a thermal sweep of one preset across a list of temperatures."""
    from cosim.thermal_sweep import run_thermal_sweep

    temps = [float(x) for x in args.temps.split(',')] if args.temps else \
            [-10, 0, 25, 45, 65, 85]
    out = args.output or str(Path('workspace') / 'thermal_sweep')

    _header(f"THERMAL SWEEP: {args.preset}")
    print(f"  Temperatures (C): {temps}\n")
    run_thermal_sweep(args.preset, temps, output_dir=out, verbose=True)
    print(f"\n  Figure saved under: {out}")


def cmd_tolerance(args):
    """Run a Monte Carlo tolerance sweep of one preset.

    Tolerance spec formats:
      --spec path/to/spec.json   (JSON dict of {field: frac_tol})
      --tol field1:0.05,field2:0.10,...
      (default): sc_responsivity:0.05,pv_dark_current_A:0.20,distance_m:0.02
    """
    import json as _json
    from cosim.monte_carlo import run_monte_carlo

    if args.spec:
        with open(args.spec, 'r') as fh:
            spec = _json.load(fh)
    elif args.tol:
        spec = {}
        for pair in args.tol.split(','):
            name, val = pair.split(':')
            spec[name.strip()] = float(val)
    else:
        spec = {
            'sc_responsivity': 0.05,
            'pv_dark_current_A': 0.20,
            'distance_m': 0.02,
        }

    out = args.output or str(Path('workspace') / 'monte_carlo')

    _header(f"MONTE CARLO: {args.preset}  (N={args.runs})")
    print(f"  Tolerance spec:")
    for k, v in spec.items():
        print(f"    {k}: +/-{v*100:.1f}%")
    print()

    result = run_monte_carlo(
        preset=args.preset,
        tolerance_spec=spec,
        n_runs=args.runs,
        ber_threshold=args.ber_threshold,
        seed=args.seed,
        output_dir=out,
        verbose=True,
    )

    print(f"\n  Yield (BER < {args.ber_threshold:.0e}): {result.yield_fraction*100:.1f}%")
    print(f"  Median BER:   {result.ber_median:.3e}")
    print(f"  95%-ile BER:  {result.ber_p95:.3e}")
    print(f"  Median SNR:   {result.snr_median_dB:.2f} dB")
    print(f"\n  Figure saved under: {out}")


def cmd_kicad(args):
    """Export KiCad schematic + BOM for a preset."""
    from kicad import export_kicad, ExportResult
    from kicad.kicad_exporter import available_presets

    presets = [args.preset] if args.preset else available_presets()
    if not presets:
        print("  No KiCad graph builders registered.")
        return

    out_dir = Path(args.output) if args.output else Path('workspace') / 'kicad'

    _header(f"KICAD EXPORT -> {out_dir}")
    for preset in presets:
        try:
            result: ExportResult = export_kicad(preset, out_dir)
        except KeyError as exc:
            print(f"  SKIP  {preset}: {exc}")
            continue
        print(f"  OK    {preset}")
        print(f"         schematic: {result.schematic_path}")
        print(f"         BOM:       {result.bom_path}")
        print(f"         {result.component_count} components, {result.net_count} nets")
        for w in result.warnings:
            print(f"         warn: {w}")


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
    p.add_argument('-v', '--verbose', action='store_true',
                   help='Enable verbose logging to console')
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

    # validate-config
    vc = sub.add_parser('validate-config',
                        help='Check a preset (or all) for parameter issues')
    vc.add_argument('--preset', help='Preset name (default: all)')
    vc.add_argument('--strict', action='store_true',
                    help='Exit non-zero on WARNING as well as ERROR')

    # kicad
    kc = sub.add_parser('kicad', help='Export KiCad schematic + BOM for a preset')
    kc.add_argument('--preset', help='Preset name (default: all registered)')
    kc.add_argument('-o', '--output', help='Output directory (default: workspace/kicad)')

    # ltspice
    sub.add_parser('ltspice', help='Check LTspice installation')

    # validate
    va = sub.add_parser('validate', help='Run paper validation & generate figures')
    va.add_argument('paper', nargs='?', help='Paper key (e.g. kadirvelu2021)')
    va.add_argument('--list', action='store_true', help='List available papers')
    va.add_argument('-o', '--output', help='Output directory')

    # compare (pipeline validation)
    cp = sub.add_parser('compare', help='Compare pipeline results against paper targets')
    cp.add_argument('--preset', help='Single preset to validate')
    cp.add_argument('--cross', action='store_true', help='Cross-validate standalone vs pipeline')
    cp.add_argument('-o', '--output', help='Output directory')

    # thermal
    th = sub.add_parser('thermal', help='Thermal sweep across temperatures')
    th.add_argument('--preset', required=True, help='Preset name')
    th.add_argument('--temps', help='Comma-separated C values (default: -10,0,25,45,65,85)')
    th.add_argument('-o', '--output', help='Output directory')

    # tolerance (Monte Carlo)
    to = sub.add_parser('tolerance', help='Monte Carlo component tolerance sweep')
    to.add_argument('--preset', required=True, help='Preset name')
    to.add_argument('--runs', type=int, default=100, help='Number of MC trials (default: 100)')
    to.add_argument('--spec', help='Path to JSON tolerance spec file')
    to.add_argument('--tol', help='Inline spec: field1:0.05,field2:0.10,...')
    to.add_argument('--ber-threshold', type=float, default=1e-3,
                    help='BER threshold for yield calc (default: 1e-3)')
    to.add_argument('--seed', type=int, help='RNG seed for reproducibility')
    to.add_argument('-o', '--output', help='Output directory')

    # sensitivity
    se = sub.add_parser('sensitivity', help='Parameter sensitivity analysis')
    se.add_argument('--preset', help='Single preset (default: all 3 representative)')
    se.add_argument('--fig', choices=['all', 'sweeps', 'tornado', 'comparison', 'temperature', 'table'],
                    default='all', help='Which figure to generate')
    se.add_argument('-o', '--output', help='Output directory')

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()

    # Initialize logging (always logs to file, verbose adds console output)
    _setup_logging(verbose=getattr(args, 'verbose', False))

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
        'validate-config': cmd_validate_config,
        'kicad':      cmd_kicad,
        'ltspice':    cmd_ltspice,
        'validate':   cmd_validate,
        'compare':    cmd_compare,
        'sensitivity': cmd_sensitivity,
        'thermal':    cmd_thermal,
        'tolerance':  cmd_tolerance,
    }

    if args.command in commands:
        commands[args.command](args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
