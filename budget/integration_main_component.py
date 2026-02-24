#!/usr/bin/env python3
# =============================================================================
# integration/main_component.py — Component-Driven Simulation Entry Point
# =============================================================================
# Task 13 of Hardware-Faithful Simulator
#
# Usage:
#   python main_component.py --pv KXOB25-04X3F --led OSRAM_LRW5SN --distance 0.325
#   python main_component.py --pv SM141K04LV --led OSRAM_LRW5SN --distance 0.5
#   python main_component.py --pv BPW34 --led OSRAM_LRW5SN --distance 0.1
#   python main_component.py --list  (show available components)
#   python main_component.py --compare --distance 0.325  (compare all receivers)
# =============================================================================

import argparse
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from components.base import get_component, list_components
from integration.adapter import (
    link_budget_report, component_to_channel_config,
    compute_effective_responsivity,
)


def print_banner():
    print("=" * 72)
    print("  LiFi-PV HARDWARE-FAITHFUL SIMULATOR — Component-Driven Mode")
    print("=" * 72)


def cmd_list():
    """List all available components."""
    print_banner()
    print("\nAvailable Components:\n")
    for pn, desc in list_components().items():
        comp = get_component(pn)
        info = comp.info()
        print(f"  {pn:20s}  [{info.material:8s}]  {info.component_type:12s}  {desc}")
    print()


def cmd_simulate(args):
    """Run a single simulation with specified components."""
    print_banner()
    print(f"\n  TX: {args.led}")
    print(f"  RX: {args.pv}")
    print(f"  Distance: {args.distance} m")
    print(f"  R_load: {args.r_load} Ω")
    print(f"  I_drive: {args.i_drive*1e3:.0f} mA")
    print(f"  Temperature: {args.temperature} K")

    lb = link_budget_report(
        args.led, args.pv, args.distance,
        args.r_load, args.i_drive, args.temperature,
    )
    s = lb['summary']

    print(f"\n{'─' * 72}")
    print(f"  LINK BUDGET RESULTS")
    print(f"{'─' * 72}")
    print(f"  TX Optical Power:    {s['P_tx_mW']:10.2f} mW")
    print(f"  RX Optical Power:    {s['P_rx_uW']:10.2f} µW")
    print(f"  Channel Loss:        {s['channel_loss_dB']:10.1f} dB")
    print(f"  Effective R(λ):      {s['R_eff_A_W']:10.4f} A/W")
    print(f"  Photocurrent:        {s['I_ph_uA']:10.2f} µA")
    print(f"  Junction Cap:        {s['C_j_pF']:10.0f} pF")
    print(f"  3dB Bandwidth:       {s['f_3dB_kHz']:10.0f} kHz")

    # Derived parameters
    rx = get_component(args.pv)
    dp = rx.derived_parameters(args.temperature)
    print(f"\n{'─' * 72}")
    print(f"  RECEIVER DERIVED PARAMETERS ({args.pv})")
    print(f"{'─' * 72}")
    for k, v in dp.items():
        if isinstance(v, float):
            print(f"  {k:30s}: {v:.4g}")
        else:
            print(f"  {k:30s}: {v}")

    led = get_component(args.led)
    dp_led = led.derived_parameters(args.temperature)
    print(f"\n{'─' * 72}")
    print(f"  TRANSMITTER DERIVED PARAMETERS ({args.led})")
    print(f"{'─' * 72}")
    for k, v in dp_led.items():
        if isinstance(v, float):
            print(f"  {k:30s}: {v:.4g}")
        else:
            print(f"  {k:30s}: {v}")

    print(f"\n{'=' * 72}")
    return lb


def cmd_compare(args):
    """Compare all receivers with the same LED and distance."""
    print_banner()
    print(f"\n  TX: {args.led}")
    print(f"  Distance: {args.distance} m")
    print(f"  I_drive: {args.i_drive*1e3:.0f} mA")

    # Find all non-LED components
    all_comps = list_components()
    receivers = []
    for pn in all_comps:
        comp = get_component(pn)
        if comp.info().component_type in ('solar_cell', 'photodiode'):
            receivers.append(pn)

    print(f"\n{'─' * 72}")
    header = (f"  {'Receiver':15s} {'Material':8s} {'R_eff':>8s} {'P_rx':>10s} "
              f"{'I_ph':>10s} {'C_j':>8s} {'f_3dB':>10s}")
    print(header)
    units = (f"  {'':15s} {'':8s} {'A/W':>8s} {'µW':>10s} "
             f"{'µA':>10s} {'pF':>8s} {'kHz':>10s}")
    print(units)
    print(f"  {'─'*15} {'─'*8} {'─'*8} {'─'*10} {'─'*10} {'─'*8} {'─'*10}")

    for rx_pn in receivers:
        ch = component_to_channel_config(args.led, rx_pn, args.distance,
                                          args.i_drive, args.temperature)
        rx = get_component(rx_pn)
        C_j = rx.junction_capacitance(0, args.temperature)
        f_3dB = rx.bandwidth(args.r_load, args.temperature)
        mat = rx.info().material

        print(f"  {rx_pn:15s} {mat:8s} {ch['R_effective_A_W']:8.4f} "
              f"{ch['P_rx_W']*1e6:10.1f} {ch['I_photocurrent_A']*1e6:10.2f} "
              f"{C_j*1e12:8.0f} {f_3dB/1e3:10.0f}")

    print(f"{'─' * 72}")

    # Distance sweep
    print(f"\n  DISTANCE SWEEP (0.1 - 1.0 m)")
    distances = [0.1, 0.2, 0.325, 0.5, 0.75, 1.0]
    print(f"\n  {'d(m)':>6s}", end="")
    for rx_pn in receivers:
        print(f"  {rx_pn:>15s}", end="")
    print("  (I_ph in µA)")

    for d in distances:
        print(f"  {d:6.3f}", end="")
        for rx_pn in receivers:
            ch = component_to_channel_config(args.led, rx_pn, d,
                                              args.i_drive, args.temperature)
            print(f"  {ch['I_photocurrent_A']*1e6:15.2f}", end="")
        print()

    print(f"\n{'=' * 72}")


def main():
    parser = argparse.ArgumentParser(
        description="LiFi-PV Hardware-Faithful Simulator — Component Mode",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --pv KXOB25-04X3F --led OSRAM_LRW5SN --distance 0.325
  %(prog)s --pv BPW34 --led OSRAM_LRW5SN --distance 0.1
  %(prog)s --compare --distance 0.325
  %(prog)s --list
        """
    )

    parser.add_argument('--list', action='store_true',
                        help='List available components')
    parser.add_argument('--compare', action='store_true',
                        help='Compare all receivers')
    parser.add_argument('--pv', type=str, default='KXOB25-04X3F',
                        help='PV receiver part number')
    parser.add_argument('--led', type=str, default='OSRAM_LRW5SN',
                        help='LED transmitter part number')
    parser.add_argument('--distance', type=float, default=0.325,
                        help='TX-RX distance in meters')
    parser.add_argument('--r_load', type=float, default=1000.0,
                        help='Load resistance in Ohms')
    parser.add_argument('--i_drive', type=float, default=0.350,
                        help='LED drive current in Amps')
    parser.add_argument('--temperature', type=float, default=300.0,
                        help='Temperature in Kelvin')

    args = parser.parse_args()

    if args.list:
        cmd_list()
    elif args.compare:
        cmd_compare(args)
    else:
        cmd_simulate(args)


if __name__ == '__main__':
    main()
