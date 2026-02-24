# gui/tab_results.py
"""
Tab 5: Results & Analysis

Sub-tabs: Waveforms, Eye Diagram, BER/SNR, Energy Harvest
Parses .raw files from LTspice and displays real simulation data.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QTabWidget, QLabel, QPushButton,
    QHBoxLayout, QComboBox, QTableWidget, QTableWidgetItem,
    QHeaderView, QGroupBox, QFileDialog,
)
from PyQt6.QtCore import Qt

from gui.widgets import MplCanvas
from cosim.raw_parser import LTSpiceRawParser


class ResultsTab(QWidget):
    """Tab 5: Post-simulation results analysis."""

    def __init__(self, config=None, parent=None):
        super().__init__(parent)
        self._config = config
        self._results = {}
        self._parser = None
        self._build_ui()

    def update_config(self, config):
        self._config = config

    def _build_ui(self):
        main = QVBoxLayout(self)

        self._subtabs = QTabWidget()

        # --- Waveforms sub-tab ---
        self._waveforms_tab = QWidget()
        wf_layout = QVBoxLayout(self._waveforms_tab)
        self._wf_canvas = MplCanvas(width=8, height=6, nrows=3, ncols=2)
        wf_layout.addWidget(self._wf_canvas)
        self._wf_placeholder = QLabel('Run simulation first to see waveforms')
        self._wf_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._wf_placeholder.setStyleSheet('font-size: 14px; color: #999;')
        wf_layout.addWidget(self._wf_placeholder)
        self._subtabs.addTab(self._waveforms_tab, 'Waveforms')

        # --- Eye Diagram sub-tab ---
        self._eye_tab = QWidget()
        eye_layout = QVBoxLayout(self._eye_tab)
        self._eye_canvas = MplCanvas(width=6, height=4)
        eye_layout.addWidget(self._eye_canvas)

        eye_ctrl = QHBoxLayout()
        eye_ctrl.addWidget(QLabel('Signal:'))
        self._eye_signal_combo = QComboBox()
        self._eye_signal_combo.addItems(['V(ina_out)', 'V(bpf_out)', 'V(dout)'])
        eye_ctrl.addWidget(self._eye_signal_combo)
        btn_eye = QPushButton('Generate Eye Diagram')
        btn_eye.clicked.connect(self._plot_eye_diagram)
        eye_ctrl.addWidget(btn_eye)
        eye_ctrl.addStretch()
        eye_layout.addLayout(eye_ctrl)
        self._subtabs.addTab(self._eye_tab, 'Eye Diagram')

        # --- BER/SNR sub-tab ---
        self._ber_tab = QWidget()
        ber_layout = QVBoxLayout(self._ber_tab)
        self._ber_canvas = MplCanvas(width=6, height=4)
        ber_layout.addWidget(self._ber_canvas)

        ber_ctrl = QHBoxLayout()
        btn_plot_ber = QPushButton('Plot Theoretical BER')
        btn_plot_ber.clicked.connect(self._plot_theoretical_ber)
        ber_ctrl.addWidget(btn_plot_ber)
        btn_ber_dist = QPushButton('BER vs Distance')
        btn_ber_dist.clicked.connect(self._plot_ber_vs_distance)
        ber_ctrl.addWidget(btn_ber_dist)
        ber_ctrl.addStretch()
        ber_layout.addLayout(ber_ctrl)

        # Simulated BER info label
        self._ber_info = QLabel('')
        self._ber_info.setStyleSheet('font-size: 12px; padding: 5px;')
        ber_layout.addWidget(self._ber_info)
        self._subtabs.addTab(self._ber_tab, 'BER / SNR')

        # --- Energy Harvest sub-tab ---
        self._harvest_tab = QWidget()
        harvest_layout = QVBoxLayout(self._harvest_tab)
        self._harvest_canvas = MplCanvas(width=6, height=4)
        harvest_layout.addWidget(self._harvest_canvas)

        # Harvest metrics table
        self._harvest_table = QTableWidget()
        self._harvest_table.setColumnCount(2)
        self._harvest_table.setHorizontalHeaderLabels(['Metric', 'Value'])
        self._harvest_table.horizontalHeader().setStretchLastSection(True)
        self._harvest_table.setEditTriggers(
            QTableWidget.EditTrigger.NoEditTriggers)
        self._harvest_table.setMaximumHeight(150)
        harvest_layout.addWidget(self._harvest_table)
        self._subtabs.addTab(self._harvest_tab, 'Energy Harvest')

        # --- Signal Explorer sub-tab ---
        self._explorer_tab = QWidget()
        exp_layout = QVBoxLayout(self._explorer_tab)

        exp_ctrl = QHBoxLayout()
        exp_ctrl.addWidget(QLabel('Signal:'))
        self._exp_signal_combo = QComboBox()
        exp_ctrl.addWidget(self._exp_signal_combo)
        btn_plot_sig = QPushButton('Plot')
        btn_plot_sig.clicked.connect(self._plot_explorer_signal)
        exp_ctrl.addWidget(btn_plot_sig)
        exp_ctrl.addStretch()
        exp_layout.addLayout(exp_ctrl)

        self._exp_canvas = MplCanvas(width=8, height=4)
        exp_layout.addWidget(self._exp_canvas)
        self._subtabs.addTab(self._explorer_tab, 'Signal Explorer')

        main.addWidget(self._subtabs)

        # Export row
        export_row = QHBoxLayout()
        export_row.addStretch()
        btn_export_png = QPushButton('Export Current Plot (PNG)')
        btn_export_png.clicked.connect(lambda: self._export_current_plot('png'))
        export_row.addWidget(btn_export_png)
        btn_export_pdf = QPushButton('Export Current Plot (PDF)')
        btn_export_pdf.clicked.connect(lambda: self._export_current_plot('pdf'))
        export_row.addWidget(btn_export_pdf)
        btn_summary = QPushButton('Export Session Summary')
        btn_summary.clicked.connect(self._export_session_summary)
        export_row.addWidget(btn_summary)
        main.addLayout(export_row)

    def update_results(self, results: dict):
        """Called when simulation completes with new results."""
        self._results = results
        self._wf_placeholder.hide()

        # Load .raw file if available
        rx = results.get('RX')
        if rx and hasattr(rx, 'outputs'):
            raw_path = rx.outputs.get('raw_file')
            if raw_path:
                try:
                    self._parser = LTSpiceRawParser(raw_path)
                    # Populate signal explorer dropdown
                    self._exp_signal_combo.clear()
                    self._exp_signal_combo.addItems(self._parser.list_traces())
                except Exception:
                    self._parser = None

        # Update BER info from pipeline
        if rx and hasattr(rx, 'outputs'):
            ber = rx.outputs.get('ber')
            n_err = rx.outputs.get('ber_n_errors')
            n_bits = rx.outputs.get('ber_n_bits')
            snr = rx.outputs.get('bpf_snr_dB')
            if ber is not None:
                info = f'Simulated BER: {ber:.4e} ({n_err}/{n_bits} errors)'
                if snr is not None and not (snr != snr):  # not NaN
                    info += f'  |  BPF SNR: {snr:.1f} dB'
                self._ber_info.setText(info)

        self._update_waveforms()
        self._update_harvest()

    def _update_waveforms(self):
        """Display 6-panel waveform overview from simulation data."""
        axes = self._wf_canvas.fig.get_axes()
        for ax in axes:
            ax.clear()

        # Panel layout:
        # [0] Optical Power P_rx(t)     [1] Solar Cell V(sc_anode)
        # [2] INA Output V(ina_out)     [3] BPF Output V(bpf_out)
        # [4] Comparator V(dout)        [5] DC-DC V(dcdc_out)

        if self._parser is not None:
            time = self._parser.get_time()
            t_ms = time * 1e3

            signal_map = [
                ('V(optical_power)', 'P_rx (W)', 'Optical Power', 'b'),
                ('V(sc_anode)', 'V (V)', 'Solar Cell', 'orange'),
                ('V(ina_out)', 'V (V)', 'INA322 Output', 'g'),
                ('V(bpf_out)', 'V (V)', 'BPF Output', 'purple'),
                ('V(dout)', 'V (V)', 'Comparator Out', 'r'),
                ('V(dcdc_out)', 'V (V)', 'DC-DC Output', 'brown'),
            ]

            for i, (trace_name, ylabel, title, color) in enumerate(signal_map):
                if i >= len(axes):
                    break
                try:
                    trace = self._parser.get_trace(trace_name)
                    axes[i].plot(t_ms, trace, color=color, linewidth=0.3)
                    axes[i].set_title(title, fontsize=9)
                    axes[i].set_ylabel(ylabel, fontsize=7)
                    axes[i].grid(True, alpha=0.3)
                    axes[i].tick_params(labelsize=7)
                    if i >= 4:
                        axes[i].set_xlabel('Time (ms)', fontsize=7)
                except KeyError:
                    axes[i].text(0.5, 0.5, f'{trace_name}\nnot found',
                                 ha='center', va='center',
                                 transform=axes[i].transAxes, fontsize=9, color='gray')
        else:
            # Fallback: show TX/Channel PWL data
            tx = self._results.get('TX')
            if tx and hasattr(tx, 'outputs'):
                pwl = tx.outputs.get('P_tx_pwl')
                if pwl:
                    try:
                        data = np.loadtxt(pwl, comments=';')
                        axes[0].plot(data[:, 0] * 1e3, data[:, 1] * 1e3,
                                     'b-', linewidth=0.5)
                        axes[0].set_title('TX: P_tx', fontsize=9)
                        axes[0].set_ylabel('mW', fontsize=8)
                        axes[0].grid(True, alpha=0.3)
                    except Exception:
                        pass

            ch = self._results.get('Channel')
            if ch and hasattr(ch, 'outputs'):
                pwl = ch.outputs.get('optical_pwl')
                if pwl:
                    try:
                        data = np.loadtxt(pwl, comments=';')
                        axes[1].plot(data[:, 0] * 1e3, data[:, 1] * 1e6,
                                     'g-', linewidth=0.5)
                        axes[1].set_title('Channel: P_rx', fontsize=9)
                        axes[1].set_ylabel('uW', fontsize=8)
                        axes[1].grid(True, alpha=0.3)
                    except Exception:
                        pass

            for i in range(2, len(axes)):
                axes[i].text(0.5, 0.5, 'Run RX simulation\nfor SPICE results',
                             ha='center', va='center',
                             transform=axes[i].transAxes, fontsize=9, color='#999')

        self._wf_canvas.fig.tight_layout()
        self._wf_canvas.draw()

    def _update_harvest(self):
        """Update energy harvesting metrics from simulation data."""
        if self._parser is None:
            return

        ax = self._harvest_canvas.ax
        ax.clear()

        try:
            time = self._parser.get_time()
            v_dcdc = self._parser.get_trace('V(dcdc_out)')
            v_sc = self._parser.get_trace('V(sc_anode)')

            t_ms = time * 1e3

            ax.plot(t_ms, v_dcdc, 'b-', linewidth=0.5, label='V(dcdc_out)')
            ax.plot(t_ms, v_sc, 'r-', linewidth=0.5, label='V(sc_anode)')
            ax.set_xlabel('Time (ms)')
            ax.set_ylabel('Voltage (V)')
            ax.set_title('DC-DC Output & Solar Cell Voltage')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

            # Compute metrics from second half (steady state)
            n_half = len(time) // 2
            v_ss = v_dcdc[n_half:]
            v_sc_ss = v_sc[n_half:]

            from cosim.system_config import SystemConfig
            cfg = SystemConfig.from_preset('kadirvelu2021')
            R_load = cfg.r_load_ohm

            V_avg = np.mean(v_ss)
            V_rms = np.sqrt(np.mean(v_ss**2))
            P_harvest = V_rms**2 / R_load
            ripple = np.max(v_ss) - np.min(v_ss)

            items = [
                ('V_dcdc (avg)', f'{V_avg:.4f} V'),
                ('V_dcdc (rms)', f'{V_rms:.4f} V'),
                ('P_harvest', f'{P_harvest*1e6:.2f} uW'),
                ('Ripple (pp)', f'{ripple*1e3:.2f} mV'),
                ('V_sc (avg)', f'{np.mean(v_sc_ss):.4f} V'),
                ('R_load', f'{R_load/1e3:.0f} kOhm'),
            ]

            self._harvest_table.setRowCount(len(items))
            for i, (k, v) in enumerate(items):
                self._harvest_table.setItem(i, 0, QTableWidgetItem(k))
                self._harvest_table.setItem(i, 1, QTableWidgetItem(v))

        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {e}',
                    ha='center', va='center', transform=ax.transAxes)

        self._harvest_canvas.draw()

    def _plot_eye_diagram(self):
        """Generate eye diagram from simulation data."""
        ax = self._eye_canvas.ax
        ax.clear()

        if self._parser is None:
            ax.text(0.5, 0.5, 'No simulation data',
                    ha='center', va='center', transform=ax.transAxes)
            self._eye_canvas.draw()
            return

        signal_name = self._eye_signal_combo.currentText()

        try:
            from simulation.analysis import eye_diagram_data

            time = self._parser.get_time()
            waveform = self._parser.get_trace(signal_name)

            data_rate = 5000.0
            if self._config is not None:
                data_rate = self._config.data_rate_bps
            bit_period = 1.0 / data_rate
            t_eye, traces = eye_diagram_data(
                time, waveform, bit_period, n_ui=2,
                skip_initial=time[-1] * 0.2)

            if traces:
                for trace in traces:
                    ax.plot(t_eye * 1e6, trace, 'b-', alpha=0.1, linewidth=0.3)
                ax.set_xlabel('Time (us)')
                ax.set_ylabel('Voltage (V)')
                ax.set_title(f'Eye Diagram: {signal_name}')
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, 'Not enough data for eye diagram',
                        ha='center', va='center', transform=ax.transAxes)

        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {e}',
                    ha='center', va='center', transform=ax.transAxes)

        self._eye_canvas.draw()

    def _plot_theoretical_ber(self):
        """Plot theoretical BER curve for OOK."""
        ax = self._ber_canvas.ax
        ax.clear()

        try:
            from simulation.analysis import theoretical_ber_ook
            snr_range = np.linspace(0, 24, 200)
            ber = [theoretical_ber_ook(s) for s in snr_range]

            ax.semilogy(snr_range, ber, 'b-', linewidth=2)
            ax.set_xlabel('SNR (dB)')
            ax.set_ylabel('BER')
            ax.set_title('Theoretical BER for OOK')
            ax.set_ylim([1e-10, 1])
            ax.grid(True, which='both', alpha=0.3)

            for target, label in [(1e-3, 'BER=1e-3'), (1e-6, 'BER=1e-6')]:
                ax.axhline(target, color='r', linestyle='--', alpha=0.5)
                ax.text(1, target * 1.5, label, fontsize=8, color='r')

        except ImportError:
            ax.text(0.5, 0.5, 'scipy not available',
                    ha='center', va='center', transform=ax.transAxes)

        self._ber_canvas.draw()

    def _plot_ber_vs_distance(self):
        """Plot analytical BER vs. distance using link budget + noise model."""
        ax = self._ber_canvas.ax
        ax.clear()

        if self._config is None:
            ax.text(0.5, 0.5, 'No config loaded',
                    ha='center', va='center', transform=ax.transAxes)
            self._ber_canvas.draw()
            return

        try:
            from simulation.analysis import theoretical_ber_ook
            from dataclasses import replace

            distances = np.linspace(0.05, 2.0, 200)
            ber_values = []
            snr_values = []

            for d in distances:
                cfg_d = replace(self._config, distance_m=d)
                snr = cfg_d.snr_estimate_dB()
                snr_values.append(snr)
                ber_values.append(theoretical_ber_ook(snr))

            ber_values = np.array(ber_values)
            snr_values = np.array(snr_values)

            ax.semilogy(distances * 100, ber_values, 'b-', linewidth=2,
                        label='Predicted BER')
            ax.set_xlabel('Distance (cm)')
            ax.set_ylabel('BER')
            ax.set_title('Predicted BER vs. Distance (Analytical)')
            ax.set_ylim([1e-12, 1])
            ax.grid(True, which='both', alpha=0.3)

            # Mark current config distance
            cur_d = self._config.distance_m
            cur_snr = self._config.snr_estimate_dB()
            cur_ber = theoretical_ber_ook(cur_snr)
            ax.axvline(cur_d * 100, color='r', linestyle='--', alpha=0.7)
            ax.plot(cur_d * 100, cur_ber, 'ro', markersize=8,
                    label=f'd={cur_d*100:.1f}cm, BER={cur_ber:.2e}')

            # Mark simulated BER if available
            rx = self._results.get('RX')
            if rx and hasattr(rx, 'outputs'):
                sim_ber = rx.outputs.get('ber')
                if sim_ber is not None and sim_ber > 0:
                    ax.plot(cur_d * 100, sim_ber, 'g^', markersize=10,
                            label=f'Simulated BER={sim_ber:.2e}')

            # BER thresholds
            for target, label in [(1e-3, 'FEC limit'), (1e-6, 'Error-free')]:
                ax.axhline(target, color='gray', linestyle=':', alpha=0.5)
                ax.text(distances[-1] * 100 * 0.95, target * 1.5, label,
                        fontsize=7, color='gray', ha='right')

            ax.legend(fontsize=8)

            # Add SNR secondary axis
            ax2 = ax.twinx()
            ax2.plot(distances * 100, snr_values, 'g--', linewidth=1, alpha=0.5)
            ax2.set_ylabel('SNR (dB)', color='green', fontsize=8)
            ax2.tick_params(axis='y', labelcolor='green', labelsize=7)

        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {e}',
                    ha='center', va='center', transform=ax.transAxes)

        self._ber_canvas.draw()

    def _plot_explorer_signal(self):
        """Plot arbitrary signal from .raw file."""
        ax = self._exp_canvas.ax
        ax.clear()

        if self._parser is None:
            ax.text(0.5, 0.5, 'No simulation data',
                    ha='center', va='center', transform=ax.transAxes)
            self._exp_canvas.draw()
            return

        signal_name = self._exp_signal_combo.currentText()

        try:
            time = self._parser.get_time()
            trace = self._parser.get_trace(signal_name)

            ax.plot(time * 1e3, trace, 'b-', linewidth=0.5)
            ax.set_xlabel('Time (ms)')
            ax.set_ylabel(signal_name)
            ax.set_title(f'Signal: {signal_name}')
            ax.grid(True, alpha=0.3)

        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {e}',
                    ha='center', va='center', transform=ax.transAxes)

        self._exp_canvas.draw()

    def _export_current_plot(self, fmt='png'):
        """Export the currently visible plot to a file."""
        canvas_map = {
            0: self._wf_canvas,
            1: self._eye_canvas,
            2: self._ber_canvas,
            3: self._harvest_canvas,
            4: self._exp_canvas,
        }
        idx = self._subtabs.currentIndex()
        canvas = canvas_map.get(idx)
        if canvas is None:
            return

        tab_name = self._subtabs.tabText(idx).replace(' ', '_').replace('/', '_')
        default_name = f'lifi_pv_{tab_name}.{fmt}'

        ext_filter = f'{fmt.upper()} (*.{fmt})'
        path, _ = QFileDialog.getSaveFileName(
            self, f'Export Plot ({fmt.upper()})', default_name, ext_filter)
        if path:
            canvas.fig.savefig(path, dpi=150, bbox_inches='tight')

    def _export_session_summary(self):
        """Export a text summary of the simulation session."""
        path, _ = QFileDialog.getSaveFileName(
            self, 'Export Session Summary', 'session_summary.txt',
            'Text (*.txt)')
        if not path:
            return

        lines = ['LiFi-PV Simulation Session Summary', '=' * 40, '']

        if self._config:
            lines.append(f'Preset: {self._config.preset_name or "Custom"}')
            lines.append(f'LED: {self._config.led_part}')
            lines.append(f'PV: {self._config.pv_part}')
            lines.append(f'Distance: {self._config.distance_m * 100:.1f} cm')
            lines.append(f'Data rate: {self._config.data_rate_bps / 1e3:.0f} kbps')
            lines.append(f'Modulation: {self._config.modulation} (depth={self._config.modulation_depth})')
            lines.append('')

        for step_name in ['TX', 'Channel', 'RX']:
            r = self._results.get(step_name)
            if r and hasattr(r, 'status'):
                lines.append(f'--- {step_name} ---')
                lines.append(f'  Status: {r.status}')
                lines.append(f'  Message: {r.message}')
                lines.append(f'  Duration: {r.duration_s:.1f}s')
                if r.outputs:
                    for k, v in r.outputs.items():
                        if not isinstance(v, (list, np.ndarray)):
                            lines.append(f'  {k}: {v}')
                lines.append('')

        rx = self._results.get('RX')
        if rx and hasattr(rx, 'outputs'):
            ber = rx.outputs.get('ber')
            if ber is not None:
                lines.append('--- BER Results ---')
                lines.append(f'  BER: {ber:.4e}')
                lines.append(f'  Errors: {rx.outputs.get("ber_n_errors")}/{rx.outputs.get("ber_n_bits")}')
                lines.append(f'  BPF SNR: {rx.outputs.get("bpf_snr_dB", "N/A")} dB')
                lines.append('')

        with open(path, 'w') as f:
            f.write('\n'.join(lines))
