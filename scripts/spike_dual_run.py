"""Dual-run verification: same pipeline with engine_compare=True.

Confirms run_step_rx exercises both the existing LTspice/ngspice path
AND the new PySpice path on a single channel-step input.

Run: .venv\\Scripts\\python.exe scripts\\spike_dual_run.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cosim.system_config import SystemConfig
from cosim.pipeline import SimulationPipeline


def main() -> int:
    cfg = SystemConfig.from_preset('kadirvelu2021')
    cfg.engine_compare = True
    cfg.n_bits = 200
    # 50 ms @ 2 kbps = 100 bits. Statistically meaningful agreement
    # number (~10% confidence interval on the percentage). LTspice
    # subprocess timeout is now 600 s via cfg.ltspice_timeout_s.
    cfg.t_stop_s = 50e-3
    if cfg.rx_topology == 'auto':
        cfg.rx_topology = 'ina_bpf_comp'

    print(f'Preset: {cfg.preset_name}')
    print(f'  simulation_engine={cfg.simulation_engine}  '
          f'engine_compare={cfg.engine_compare}')
    print(f'  topology={cfg.rx_topology}  bpf_stages={cfg.bpf_stages}')

    session_dir = Path(__file__).resolve().parent / '_dual_run_session'
    for sub in ('netlists', 'pwl', 'raw'):
        (session_dir / sub).mkdir(parents=True, exist_ok=True)
    pipe = SimulationPipeline(cfg, session_dir=session_dir)

    print('\n--- Step 1: TX ---')
    s = pipe.run_step_tx()
    print(f'  {s.status}: {s.message}')
    if s.status != 'done':
        return 1

    print('\n--- Step 2: Channel ---')
    s = pipe.run_step_channel()
    print(f'  {s.status}: {s.message}')
    if s.status != 'done':
        return 2

    print('\n--- Step 3: RX (dual-run) ---')
    s = pipe.run_step_rx()
    print(f'  {s.status}: {s.message}')
    print(f'  duration: {s.duration_s:.2f}s')

    print('\n--- Outputs ---')
    for key, val in s.outputs.items():
        if isinstance(val, dict):
            print(f'  {key}:')
            for k2, v2 in val.items():
                print(f'    {k2}: {v2}')
        else:
            print(f'  {key}: {val}')

    py = s.outputs.get('pyspice')
    if py is None:
        print('\nNOTE: pyspice path did not produce output')
        return 3
    if 'error' in py:
        print(f'\nFAIL pyspice error: {py["error"]}')
        return 4
    npz = py.get('npz_file')
    if not npz or not Path(npz).exists():
        print('\nFAIL pyspice npz not written')
        return 5

    import numpy as np
    arrs = np.load(npz)
    print(f'\nPySpice npz at {npz}')
    print(f'  arrays: {list(arrs.files)}')
    print(f'  t.shape: {arrs["t"].shape}')
    if 'dout' in arrs:
        print(f'  V(dout) pp: {float(np.ptp(arrs["dout"])):.3f} V')
    if 'sc_anode' in arrs:
        print(f'  V(sc_anode) pp: {float(np.ptp(arrs["sc_anode"]))*1e3:.2f} mV')

    print('\nDual-run smoke PASS')
    return 0


if __name__ == '__main__':
    sys.exit(main())
