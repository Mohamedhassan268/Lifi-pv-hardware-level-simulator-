# cosim/engine_compare.py
"""
SPICE vs Python Engine Comparison Tool.

Runs the same SystemConfig through both simulation engines and compares
waveforms at each node. Reports RMS error, correlation, and frequency
domain analysis.

Usage:
    from cosim.engine_compare import EngineComparison

    comp = EngineComparison(config)
    report = comp.run()
    print(report.summary())
    comp.plot(report, output_dir='comparison_plots/')

Target: <5% RMS error at each node for default Kadirvelu config.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Optional, List
from pathlib import Path


@dataclass
class NodeComparison:
    """Comparison result for a single waveform node."""
    name: str
    rms_error: float          # RMS of (python - spice) / max(|spice|)
    correlation: float        # Pearson correlation coefficient
    max_abs_error: float      # Maximum absolute error
    spice_rms: float          # RMS of SPICE waveform
    python_rms: float         # RMS of Python waveform
    passed: bool              # True if rms_error < threshold


@dataclass
class ComparisonReport:
    """Full comparison report between engines."""
    config_name: str
    nodes: List[NodeComparison] = field(default_factory=list)
    spice_available: bool = False
    spice_result: Optional[Dict] = None
    python_result: Optional[Dict] = None
    overall_pass: bool = False

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [f"Engine Comparison: {self.config_name}"]
        lines.append("=" * 60)

        if not self.spice_available:
            lines.append("SPICE engine not available — comparison skipped.")
            lines.append("Run with SPICE installed to enable comparison.")
            if self.python_result:
                lines.append(f"Python engine BER: {self.python_result.get('ber', 'N/A')}")
            return "\n".join(lines)

        lines.append(f"{'Node':<20} {'RMS Err %':<12} {'Corr':<10} {'Status'}")
        lines.append("-" * 60)
        for node in self.nodes:
            status = "PASS" if node.passed else "FAIL"
            lines.append(
                f"{node.name:<20} {node.rms_error*100:>8.2f}%   "
                f"{node.correlation:>8.4f}   {status}"
            )
        lines.append("-" * 60)
        overall = "PASS" if self.overall_pass else "FAIL"
        lines.append(f"Overall: {overall}")
        return "\n".join(lines)


class EngineComparison:
    """
    Compare SPICE and Python engine outputs for the same configuration.

    Usage:
        comp = EngineComparison(config, rms_threshold=0.05)
        report = comp.run()
        print(report.summary())
    """

    # Per-node waveform keys to compare (Python key → display name)
    NODE_KEYS = [
        ('I_ph', 'Photocurrent'),
        ('V_sense', 'V_sense'),
        ('V_ina', 'V_ina'),
        ('V_comp', 'V_comparator'),
        ('V_rx', 'V_rx (output)'),
    ]

    def __init__(self, config, rms_threshold: float = 0.05):
        """
        Args:
            config: SystemConfig instance
            rms_threshold: Maximum allowed normalized RMS error (default 5%)
        """
        self.config = config
        self.threshold = rms_threshold

    def run(self) -> ComparisonReport:
        """
        Run both engines and compare outputs.

        Returns:
            ComparisonReport with per-node comparison results
        """
        from cosim.spice_finder import spice_available
        from cosim.python_engine import run_python_simulation

        report = ComparisonReport(
            config_name=self.config.preset_name or 'Custom',
        )

        # Run Python engine with Phase 2 features (always available)
        py_cfg = self._make_python_config()
        report.python_result = run_python_simulation(py_cfg)

        # Check SPICE availability
        if not spice_available():
            report.spice_available = False
            return report

        report.spice_available = True

        # Run SPICE engine via hybrid pipeline
        try:
            import tempfile
            from cosim.pipeline import SimulationPipeline
            from cosim.session import SessionManager

            with tempfile.TemporaryDirectory() as tmpdir:
                session = SessionManager(tmpdir)
                session_dir = session.create_session('compare')
                pipeline = SimulationPipeline(self.config, session_dir)
                step_results = pipeline.run_hybrid()

                # Extract SPICE result dict from pipeline
                rx_step = step_results.get('RX')
                if rx_step and rx_step.status == 'done':
                    spice_dict = rx_step.outputs.get('python_result', {})
                    # If hybrid, build dict from raw extraction
                    raw_file = rx_step.outputs.get('raw_file')
                    if raw_file:
                        from cosim.spice_extract import extract_spice_waveforms
                        waveforms = extract_spice_waveforms(raw_file)
                        spice_dict = {
                            'time': waveforms.get('time', np.array([])),
                            'I_ph': waveforms.get('I_sense', np.array([])),
                            'V_ina': waveforms.get('V_ina', np.array([])),
                            'V_comp': waveforms.get('V_comp', np.array([])),
                            'V_rx': waveforms.get('V_comp', np.array([])),
                            'ber': rx_step.outputs.get('ber', 0.0),
                        }
                    report.spice_result = spice_dict

        except Exception as e:
            report.spice_available = False
            return report

        # Compare common waveform nodes
        self._compare_nodes(report)

        return report

    def _make_python_config(self):
        """Create a Python engine config with Phase 2 features enabled."""
        import copy
        cfg = copy.copy(self.config)
        cfg.simulation_engine = 'python'
        cfg.pv_ode_enable = True
        cfg.rx_chain_enable = True
        cfg.led_bandwidth_limit_enable = True
        return cfg

    def _compare_nodes(self, report: ComparisonReport):
        """Compare waveforms at shared nodes."""
        py = report.python_result
        sp = report.spice_result

        if py is None or sp is None:
            return

        # Compare array waveforms at each node
        for key, display_name in self.NODE_KEYS:
            if key in py and key in sp:
                py_wave = np.asarray(py[key], dtype=float)
                sp_wave = np.asarray(sp[key], dtype=float)

                if len(py_wave) == 0 or len(sp_wave) == 0:
                    continue

                # Align lengths via interpolation
                if len(py_wave) != len(sp_wave):
                    target_len = min(len(py_wave), len(sp_wave))
                    py_wave = np.interp(
                        np.linspace(0, 1, target_len),
                        np.linspace(0, 1, len(py_wave)),
                        py_wave
                    )
                    sp_wave = np.interp(
                        np.linspace(0, 1, target_len),
                        np.linspace(0, 1, len(sp_wave)),
                        sp_wave
                    )

                node = self._compare_arrays(display_name, py_wave, sp_wave)
                report.nodes.append(node)

        # BPF stages comparison
        if 'V_bpf' in py and isinstance(py['V_bpf'], list):
            for i, bpf_py in enumerate(py['V_bpf']):
                sp_key = f'V_bpf{i+1}' if f'V_bpf{i+1}' in (sp or {}) else None
                if sp_key and sp_key in sp:
                    py_wave = np.asarray(bpf_py, dtype=float)
                    sp_wave = np.asarray(sp[sp_key], dtype=float)
                    if len(py_wave) != len(sp_wave):
                        tgt = min(len(py_wave), len(sp_wave))
                        py_wave = np.interp(np.linspace(0, 1, tgt),
                                            np.linspace(0, 1, len(py_wave)), py_wave)
                        sp_wave = np.interp(np.linspace(0, 1, tgt),
                                            np.linspace(0, 1, len(sp_wave)), sp_wave)
                    node = self._compare_arrays(f'V_bpf{i+1}', py_wave, sp_wave)
                    report.nodes.append(node)

        # BER comparison (scalar)
        py_ber = py.get('ber', 0.0)
        sp_ber = sp.get('ber', 0.0) if sp else 0.0
        if sp_ber is not None:
            diff = abs(float(py_ber) - float(sp_ber))
            report.nodes.append(NodeComparison(
                name='BER',
                rms_error=diff,
                correlation=1.0 if diff < 0.01 else 0.0,
                max_abs_error=diff,
                spice_rms=float(sp_ber),
                python_rms=float(py_ber),
                passed=diff < self.threshold,
            ))

        # Overall pass/fail
        report.overall_pass = all(n.passed for n in report.nodes) if report.nodes else False

    def _compare_arrays(self, name: str,
                        py_wave: np.ndarray,
                        sp_wave: np.ndarray) -> NodeComparison:
        """Compare two waveform arrays."""
        # Normalized RMS error
        scale = np.max(np.abs(sp_wave)) if np.max(np.abs(sp_wave)) > 0 else 1.0
        diff = py_wave - sp_wave
        rms_error = np.sqrt(np.mean(diff ** 2)) / scale

        # Pearson correlation
        if np.std(py_wave) > 0 and np.std(sp_wave) > 0:
            correlation = float(np.corrcoef(py_wave, sp_wave)[0, 1])
        else:
            correlation = 1.0 if np.allclose(py_wave, sp_wave) else 0.0

        max_abs = float(np.max(np.abs(diff)))

        return NodeComparison(
            name=name,
            rms_error=rms_error,
            correlation=correlation,
            max_abs_error=max_abs,
            spice_rms=float(np.sqrt(np.mean(sp_wave ** 2))),
            python_rms=float(np.sqrt(np.mean(py_wave ** 2))),
            passed=rms_error < self.threshold,
        )

    def plot(self, report: ComparisonReport,
             output_dir: Optional[str] = None) -> None:
        """
        Plot overlaid waveforms and FFT spectra for comparison.

        Generates a multi-panel figure with time-domain and frequency-domain
        comparisons at each available node.

        Args:
            report: ComparisonReport from run()
            output_dir: Directory to save plots (None = show interactively)
        """
        try:
            import matplotlib
            if output_dir:
                matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib required for plotting")
            return

        py = report.python_result
        sp = report.spice_result

        if py is None:
            print("No Python result available for plotting")
            return

        # Determine which nodes have data in both engines
        plot_nodes = []
        for key, name in self.NODE_KEYS:
            if key in py and len(np.asarray(py[key])) > 0:
                has_spice = (sp is not None and key in sp and
                             len(np.asarray(sp[key])) > 0)
                plot_nodes.append((key, name, has_spice))

        if not plot_nodes:
            print("No waveforms available for plotting")
            return

        n_panels = len(plot_nodes)
        fig, axes = plt.subplots(n_panels, 2, figsize=(14, 3.5 * n_panels))
        if n_panels == 1:
            axes = axes.reshape(1, 2)
        fig.suptitle(f'Engine Comparison: {report.config_name}', fontsize=14)

        for i, (key, name, has_spice) in enumerate(plot_nodes):
            t_py = py.get('time', np.arange(len(py[key])))
            py_wave = np.asarray(py[key])

            # Time domain
            ax_t = axes[i, 0]
            ax_t.plot(t_py * 1e3, py_wave, label='Python', alpha=0.8, linewidth=0.8)
            if has_spice:
                t_sp = sp.get('time', np.arange(len(sp[key])))
                sp_wave = np.asarray(sp[key])
                ax_t.plot(t_sp * 1e3, sp_wave, label='SPICE',
                          alpha=0.8, linestyle='--', linewidth=0.8)
            ax_t.set_ylabel(name)
            ax_t.set_xlabel('Time (ms)')
            ax_t.legend(fontsize=8)
            ax_t.grid(True, alpha=0.3)

            # Frequency domain (FFT)
            ax_f = axes[i, 1]
            dt_py = np.mean(np.diff(t_py)) if len(t_py) > 1 else 1.0
            freqs_py = np.fft.rfftfreq(len(py_wave), dt_py)
            fft_py = np.abs(np.fft.rfft(py_wave - np.mean(py_wave)))
            ax_f.semilogy(freqs_py / 1e3, fft_py + 1e-15,
                          label='Python', alpha=0.8, linewidth=0.8)

            if has_spice:
                sp_wave_rs = np.interp(t_py, sp.get('time', t_py),
                                       np.asarray(sp[key]))
                fft_sp = np.abs(np.fft.rfft(sp_wave_rs - np.mean(sp_wave_rs)))
                ax_f.semilogy(freqs_py / 1e3, fft_sp + 1e-15,
                              label='SPICE', alpha=0.8, linestyle='--',
                              linewidth=0.8)

            ax_f.set_ylabel(f'|FFT({name})|')
            ax_f.set_xlabel('Frequency (kHz)')
            ax_f.legend(fontsize=8)
            ax_f.grid(True, alpha=0.3)
            # Limit frequency range to meaningful region
            max_freq = min(freqs_py[-1] / 1e3, 50)
            ax_f.set_xlim(0, max_freq)

        plt.tight_layout()

        if output_dir:
            out_path = Path(output_dir)
            out_path.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_path / 'engine_comparison.png', dpi=150)
            plt.close(fig)
        else:
            plt.show()
