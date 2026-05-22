"""LTspice .raw vs PySpice .npz trace-equivalence diff.

Phase B verification: takes a session that was produced with
engine_compare=True (i.e. has both `receiver.raw` and `pyspice_rx.npz`)
and prints per-node metrics + an overlay plot.

Usage:
    .venv\\Scripts\\python.exe scripts\\diff_traces.py <session_dir>
    .venv\\Scripts\\python.exe scripts\\diff_traces.py            # defaults to scripts/_dual_run_session
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cosim.raw_parser import LTSpiceRawParser


# Standard node mapping: PySpice node name -> LTspice trace name
NODE_MAP = {
    'sc_anode':       'V(sc_anode)',
    'sc_cathode':     'V(sc_cathode)',
    'sense_lo':       'V(sense_lo)',
    'optical_power':  'V(optical_power)',
    'ina_out':        'V(ina_out)',
    'bpf1_out':       'V(bpf1_out)',
    'bpf2_out':       'V(bpf_out)',   # LTspice last stage = V(bpf_out)
    'bpf_out':        'V(bpf_out)',
    'dout':           'V(dout)',
}


def _resample(t_src: np.ndarray, y_src: np.ndarray,
              t_dst: np.ndarray) -> np.ndarray:
    """Linear-interpolate y_src(t_src) onto t_dst."""
    return np.interp(t_dst, t_src, y_src)


def _metric(a: np.ndarray, b: np.ndarray) -> dict:
    """Quantitative diff between two aligned arrays."""
    diff = a - b
    rms = float(np.sqrt(np.mean(diff ** 2)))
    max_abs = float(np.max(np.abs(diff)))
    # Pearson correlation; guard against zero variance
    if a.std() < 1e-12 or b.std() < 1e-12:
        corr = float('nan')
    else:
        corr = float(np.corrcoef(a, b)[0, 1])
    return {
        'lt_pp': float(np.ptp(a)),
        'py_pp': float(np.ptp(b)),
        'rms_diff': rms,
        'max_diff': max_abs,
        'corr': corr,
    }


def main(session_dir: Path) -> int:
    raw_path = session_dir / 'raw' / 'receiver.raw'
    npz_path = session_dir / 'raw' / 'pyspice_rx.npz'
    if not raw_path.exists():
        print(f'FAIL: LTspice raw not found at {raw_path}')
        return 1
    if not npz_path.exists():
        print(f'FAIL: PySpice npz not found at {npz_path}')
        return 1

    print(f'Loading LTspice: {raw_path.name} ({raw_path.stat().st_size/1024:.1f} kB)')
    parser = LTSpiceRawParser(str(raw_path))
    t_lt = parser.get_time()
    lt_traces = {name.lower(): parser.get_trace(name)
                 for name in parser.list_traces() if name.lower() != 'time'}

    print(f'Loading PySpice: {npz_path.name} ({npz_path.stat().st_size/1024:.1f} kB)')
    npz = np.load(npz_path)
    t_py = npz['t']
    py_nodes = {k: npz[k] for k in npz.files if k != 't'}

    print(f'\nLTspice: {len(t_lt)} samples, t in [{t_lt[0]*1e3:.3f}, {t_lt[-1]*1e3:.3f}] ms')
    print(f'PySpice: {len(t_py)} samples, t in [{t_py[0]*1e3:.3f}, {t_py[-1]*1e3:.3f}] ms')

    # Resample LTspice onto PySpice's uniform grid for fair comparison
    t_common = t_py
    t_min = max(t_lt[0], t_py[0])
    t_max = min(t_lt[-1], t_py[-1])
    mask = (t_common >= t_min) & (t_common <= t_max)
    t_common = t_common[mask]
    print(f'Common window: {len(t_common)} samples on [{t_min*1e3:.3f}, '
          f'{t_max*1e3:.3f}] ms')

    rows = []
    plotable = []
    for py_name, lt_name in NODE_MAP.items():
        if py_name not in py_nodes:
            continue
        lt_key = lt_name.lower()
        if lt_key not in lt_traces:
            rows.append((py_name, lt_name, None,
                         f'LTspice trace {lt_name!r} missing'))
            continue
        lt_y_full = lt_traces[lt_key]
        py_y_full = py_nodes[py_name]
        # Resample both onto the common window
        lt_y = _resample(t_lt, lt_y_full, t_common)
        py_y = _resample(t_py, py_y_full, t_common)
        m = _metric(lt_y, py_y)
        rows.append((py_name, lt_name, m, None))
        plotable.append((py_name, lt_name, lt_y, py_y))

    # Print table
    print(f'\n{"Node":<14} {"LT-pp":>10} {"PY-pp":>10} {"RMS-diff":>10} '
          f'{"Max-diff":>10} {"Corr":>8}  Notes')
    print('-' * 90)
    for py_name, lt_name, m, note in rows:
        if m is None:
            print(f'{py_name:<14} {"-":>10} {"-":>10} {"-":>10} {"-":>10} '
                  f'{"-":>8}  {note}')
            continue
        # Pick unit per node
        if 'sc_anode' in py_name or 'sc_cathode' in py_name or 'sense_lo' in py_name \
           or 'optical' in py_name:
            scale, unit = 1e3, 'mV'
        else:
            scale, unit = 1.0, ' V'
        verdict = ''
        if m['py_pp'] < 1e-6 and m['lt_pp'] > m['py_pp'] * 100 + 1e-3:
            verdict = '<- PySpice flat'
        elif m['lt_pp'] < 1e-6 and m['py_pp'] > m['lt_pp'] * 100 + 1e-3:
            verdict = '<- LTspice flat'
        elif abs(m['corr']) > 0.95 and m['rms_diff'] / max(m['lt_pp'], 1e-9) < 0.10:
            verdict = 'OK'
        else:
            verdict = 'differs'
        print(f'{py_name:<14} {m["lt_pp"]*scale:>9.3f}{unit} '
              f'{m["py_pp"]*scale:>9.3f}{unit} '
              f'{m["rms_diff"]*scale:>9.3f}{unit} '
              f'{m["max_diff"]*scale:>9.3f}{unit} '
              f'{m["corr"]:>8.4f}  {verdict}')

    # BER-level decision agreement: sample dout at bit centers, threshold
    # at vref, count disagreements between engines. This is the metric that
    # actually matters for the comm-link thesis; trace-level diff is misled
    # by DC-DC switching noise that the INA later filters out anyway.
    if 'dout' in py_nodes and 'v(dout)' in lt_traces:
        try:
            preset_path = (Path(__file__).resolve().parent.parent /
                           'presets' / 'kadirvelu2021.json')
            import json as _json
            preset = _json.loads(preset_path.read_text())
            data_rate = float(preset.get('data_rate_bps', 2000))
            vcc = float(preset.get('vcc_volts', 3.3))
            t_bit = 1.0 / data_rate
            vth = vcc / 2.0
            t_window = t_max - t_min
            n_bits_possible = max(1, int(t_window / t_bit))
            if n_bits_possible >= 4:
                lt_dout = _resample(t_lt, lt_traces['v(dout)'], t_common)
                py_dout = _resample(t_py, py_nodes['dout'], t_common)
                # Sample at bit centers (skip the first bit -- transient settle)
                bit_centers = t_min + (np.arange(n_bits_possible) + 0.5) * t_bit
                bit_centers = bit_centers[(bit_centers >= t_min) & (bit_centers <= t_max)]
                lt_at_centers = np.interp(bit_centers, t_common, lt_dout)
                py_at_centers = np.interp(bit_centers, t_common, py_dout)
                lt_bits = (lt_at_centers > vth).astype(int)
                py_bits = (py_at_centers > vth).astype(int)
                disagreements = int(np.sum(lt_bits != py_bits))
                n = len(lt_bits)
                pct_agree = 100.0 * (n - disagreements) / n if n > 0 else 0.0
                print(f'\nBER-decision agreement (V(dout) thresholded at vcc/2):')
                print(f'  bit period: {t_bit*1e6:.0f} us, '
                      f'samples per bit: {int(t_bit / np.mean(np.diff(t_common)))}')
                print(f'  bits compared: {n}  agreements: {n - disagreements}  '
                      f'disagreements: {disagreements}')
                print(f'  per-bit agreement: {pct_agree:.1f}%')
                if disagreements == 0:
                    print('  -> Engines AGREE on every bit decision.')
                elif pct_agree >= 90:
                    print(f'  -> Engines mostly agree ({pct_agree:.0f}%); '
                          f'within useful BER tolerance.')
                else:
                    print(f'  -> Engines disagree ({pct_agree:.0f}%); '
                          f'longer run / DC-DC re-injection needed.')
            else:
                print(f'\nNot enough bits in window for BER comparison '
                      f'({n_bits_possible} bits; need >= 4). '
                      f'Re-run with longer t_stop_s.')
        except Exception as e:
            print(f'\nBER agreement check failed: {e}')

    # Generate overlay plot
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        n = len(plotable)
        if n > 0:
            fig, axes = plt.subplots(n, 1, figsize=(10, max(3 * n, 3)),
                                     sharex=True)
            if n == 1:
                axes = [axes]
            for ax, (py_name, lt_name, lt_y, py_y) in zip(axes, plotable):
                ax.plot(t_common * 1e3, lt_y, label=f'LTspice {lt_name}',
                        lw=1.0, alpha=0.85)
                ax.plot(t_common * 1e3, py_y, label=f'PySpice {py_name}',
                        lw=1.0, alpha=0.85, linestyle='--')
                ax.set_ylabel(py_name)
                ax.legend(loc='upper right', fontsize=8)
                ax.grid(alpha=0.3)
            axes[-1].set_xlabel('Time (ms)')
            fig.suptitle('Trace overlay: LTspice vs PySpice (hybrid)')
            plot_path = session_dir / 'trace_diff_overlay.png'
            fig.savefig(plot_path, dpi=120, bbox_inches='tight')
            plt.close(fig)
            print(f'\nOverlay plot saved: {plot_path}')
    except ImportError:
        print('\n(matplotlib unavailable; skipping overlay plot)')

    return 0


if __name__ == '__main__':
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    default = Path(__file__).resolve().parent / '_dual_run_session'
    session = Path(arg) if arg else default
    sys.exit(main(session))
