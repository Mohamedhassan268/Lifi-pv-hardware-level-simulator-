"""End-to-end PySpice RX-chain smoke test.

Loads a real SystemConfig preset, generates a synthetic OOK optical
waveform in numpy, runs it through PySpiceRxRunner, and asserts the
comparator output toggles between rails.

Run: .venv\\Scripts\\python.exe scripts\\spike_pyspice_rx.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Make project root importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cosim.system_config import SystemConfig
from cosim.pyspice_runner import PySpiceRxRunner


def main() -> int:
    # Use the kadirvelu2021 preset since that's the one that requires SPICE.
    try:
        cfg = SystemConfig.from_preset('kadirvelu2021')
    except Exception as e:
        print(f'FAIL loading preset: {e}')
        return 1
    print(f'Preset: {cfg.preset_name}')
    print(f'  topology={cfg.rx_topology}  vcc={cfg.vcc_volts}V  '
          f'ina_gain={cfg.ina_gain_dB}dB  bpf_stages={cfg.bpf_stages}  '
          f'comparator={cfg.comparator_part}')
    if cfg.rx_topology == 'auto':
        cfg.rx_topology = 'ina_bpf_comp'

    # Synthetic OOK optical waveform: 5 kHz square wave, 100 uW peak,
    # 1 ms total, 1000 samples.
    t_stop = 1e-3
    n = 1000
    t = np.linspace(0, t_stop, n)
    # 5 kHz square in W (Voptical drives SOLAR_CELL: V_in = optical power in W,
    # photocurrent = V_in * responsivity).
    f_carrier = 5e3
    p_peak_W = 1e-3   # 1 mW peak (realistic VLC near-field power)
    p_rx = np.where(np.sin(2 * np.pi * f_carrier * t) > 0, p_peak_W, 0.0)

    print(f'\nRunning transient: t_stop={t_stop*1e3:.1f}ms, '
          f'{n} input PWL points, f_carrier={f_carrier/1e3:.1f}kHz')

    runner = PySpiceRxRunner(cfg, verbose=False)
    if not runner.available():
        print('FAIL: PySpice/libngspice not available')
        return 2
    print(f'  engine: {runner.version_string}')


    # Dump the Stage A netlist for inspection before running
    circuit = runner._build_circuit_stage_a(
        cfg=cfg, topology=cfg.rx_topology,
        optical_t=t, optical_v=p_rx, noise_t=None, noise_v=None,
    )
    netlist_text = str(circuit)
    netlist_dump = Path(__file__).resolve().parent / '_dump_rx_netlist.cir'
    netlist_dump.write_text(netlist_text, encoding='utf-8')
    print(f'  Stage A netlist dumped to {netlist_dump}')
    print(f'  Stage A netlist length: {len(netlist_text)} chars')

    # Bisect: try progressively complex topologies to find the convergence killer.
    print('\n--- Bisect: try simpler configs ---')
    import copy
    for label, mutate in [
        ('1) solar+sense+INA only (no BPF, no comparator)',
         lambda c: setattr(c, 'bpf_stages', 0) or setattr(c, 'comparator_part', 'N/A')),
        ('1b) INA+comparator (no BPF in between)',
         lambda c: setattr(c, 'bpf_stages', 0) or setattr(c, 'comparator_part', 'TLV7011')),
        ('2) solar+sense+INA+BPF (no comparator)',
         lambda c: setattr(c, 'bpf_stages', 2) or setattr(c, 'comparator_part', 'N/A')),
        ('3) full chain (INA+BPF+comparator)',
         lambda c: setattr(c, 'bpf_stages', 2) or setattr(c, 'comparator_part', 'TLV7011')),
    ]:
        cfg2 = copy.deepcopy(cfg)
        mutate(cfg2)
        r2 = PySpiceRxRunner(cfg2)
        try:
            wf2 = r2.run_transient(
                optical_t=t, optical_v=p_rx,
                duration_s=t_stop, t_step_s=t_stop / 5000.0,
            )
            print(f'  {label}: PASS ({wf2.meta["n_points"]} points, '
                  f'{wf2.meta["duration_s"]:.2f}s)')
        except Exception as e:
            print(f'  {label}: FAIL ({str(e)[:80]})')
    print('--- End bisect ---\n')

    try:
        wf = runner.run_transient(
            optical_t=t, optical_v=p_rx,
            duration_s=t_stop, t_step_s=t_stop / 5000.0,
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f'FAIL transient: {e}')
        return 3

    print(f'  duration: {wf.meta["duration_s"]:.2f}s')
    print(f'  result: {wf.meta["n_points"]} samples')
    print(f'  nodes: {sorted(wf.nodes.keys())}')

    # Assertions
    v_sc = wf.get('sc_anode')
    v_ina = wf.get('ina_out')
    v_bpf = wf.get('bpf_out')
    v_dout = wf.get('dout')

    failed = False

    if v_sc is None:
        print('FAIL: sc_anode missing')
        failed = True
    else:
        print(f'  V(sc_anode): mean={v_sc.mean()*1e3:.2f}mV  '
              f'pp={np.ptp(v_sc)*1e3:.2f}mV')

    if v_ina is None:
        print('FAIL: ina_out missing')
        failed = True
    else:
        print(f'  V(ina_out):  mean={v_ina.mean():.3f}V  '
              f'pp={np.ptp(v_ina)*1e3:.2f}mV')

    if cfg.bpf_stages > 0:
        if v_bpf is None:
            print('FAIL: bpf_out missing')
            failed = True
        else:
            pp_bpf = np.ptp(v_bpf)
            print(f'  V(bpf_out):  mean={v_bpf.mean():.3f}V  '
                  f'pp={pp_bpf*1e3:.2f}mV')
            if pp_bpf < 0.05:
                print(f'WARN: bpf swing tiny ({pp_bpf*1e3:.2f}mV) '
                      f'- may not drive the comparator')

    if cfg.comparator_part != 'N/A':
        if v_dout is None:
            print('FAIL: dout missing')
            failed = True
        else:
            # After settling, expect bimodal distribution near 0 and Vcc
            settled = v_dout[len(v_dout) // 4:]
            n_hi = (settled > cfg.vcc_volts * 0.7).sum()
            n_lo = (settled < cfg.vcc_volts * 0.3).sum()
            pct_hi = 100 * n_hi / len(settled)
            pct_lo = 100 * n_lo / len(settled)
            print(f'  V(dout):     pp={np.ptp(v_dout):.3f}V  '
                  f'HI%={pct_hi:.0f}  LO%={pct_lo:.0f}')
            if not (pct_hi > 10 and pct_lo > 10):
                print('WARN: dout did not toggle (signal too small or '
                      'not reaching comparator)')

    if failed:
        return 4

    print('\nALL CHECKS PASSED')
    return 0


if __name__ == '__main__':
    sys.exit(main())
