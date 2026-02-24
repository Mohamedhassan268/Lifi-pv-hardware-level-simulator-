# gui/tab_simulation_engine.py
"""
Tab 4: Simulation Engine (THE CRITICAL TAB)

3-step pipeline visualization: TX -> Channel -> RX
Each step has: description, status indicator, [Run] button
Shows PWL bridge between steps 2->3
[Run All 3 Steps] master button
Log panel + 4-panel waveform preview
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QPushButton, QPlainTextEdit, QSplitter, QProgressBar,
    QListWidget, QListWidgetItem,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QElapsedTimer

from cosim.system_config import SystemConfig
from cosim.session import SessionManager
from cosim.pipeline import SimulationPipeline
from cosim.ltspice_runner import LTSpiceRunner
from gui.widgets import StatusIndicator, MplCanvas


class _PipelineWorker(QThread):
    """Background thread for running the simulation pipeline."""
    step_update = pyqtSignal(str, str, str)  # step_name, status, message
    finished = pyqtSignal(dict)               # results dict

    def __init__(self, config, session_dir, ltspice_runner, steps='all'):
        super().__init__()
        self._config = config
        self._session_dir = session_dir
        self._ltspice = ltspice_runner
        self._steps = steps  # 'all', 'tx', 'channel', 'rx'

    def run(self):
        def on_progress(step, status, msg):
            self.step_update.emit(step, status, msg)

        pipe = SimulationPipeline(
            self._config, self._session_dir,
            ltspice_runner=self._ltspice,
            on_progress=on_progress,
        )

        if self._steps == 'all':
            results = pipe.run_all()
        elif self._steps == 'tx':
            pipe.run_step_tx()
            results = {'TX': pipe.step_tx}
        elif self._steps == 'channel':
            pipe.run_step_tx()
            pipe.run_step_channel()
            results = {'TX': pipe.step_tx, 'Channel': pipe.step_channel}
        elif self._steps == 'rx':
            pipe.run_step_tx()
            pipe.run_step_channel()
            pipe.run_step_rx()
            results = {'TX': pipe.step_tx, 'Channel': pipe.step_channel,
                        'RX': pipe.step_rx}

        self.finished.emit(results)


class SimulationEngineTab(QWidget):
    """Tab 4: 3-step simulation pipeline with live status."""

    simulation_done = pyqtSignal(dict)  # emits results when complete

    def __init__(self, config: SystemConfig, ltspice_runner=None, parent=None):
        super().__init__(parent)
        self._config = config
        self._session_mgr = SessionManager()
        self._ltspice = ltspice_runner or LTSpiceRunner()
        self._worker = None
        self._last_results = {}

        # Elapsed time tracking
        self._elapsed_timer = QElapsedTimer()
        self._ui_timer = QTimer()
        self._ui_timer.setInterval(500)  # Update every 500ms
        self._ui_timer.timeout.connect(self._update_elapsed)

        self._build_ui()

    def _build_ui(self):
        main = QVBoxLayout(self)

        # ---- Engine status bar ----
        status_row = QHBoxLayout()
        ltspice_status = ('Available' if self._ltspice.available
                          else 'Not found')
        self._ltspice_label = QLabel(f'LTspice: {ltspice_status}')
        status_row.addWidget(self._ltspice_label)
        self._engine_mode_label = QLabel('')
        self._engine_mode_label.setStyleSheet('font-weight: bold; padding: 2px 8px;')
        status_row.addWidget(self._engine_mode_label)
        status_row.addStretch()
        self._session_label = QLabel('Session: (none)')
        status_row.addWidget(self._session_label)
        main.addLayout(status_row)
        self._update_engine_label()

        # ---- Pipeline steps ----
        pipe_grp = QGroupBox('Simulation Pipeline')
        pipe_layout = QVBoxLayout(pipe_grp)

        # Step 1: TX
        self._tx_row, self._tx_status, self._tx_label = self._make_step_row(
            'Step 1: TX', 'Generate OOK waveform P_tx(t)')
        self._tx_btn = QPushButton('Run TX')
        self._tx_btn.clicked.connect(lambda: self._run_steps('tx'))
        self._tx_row.addWidget(self._tx_btn)
        pipe_layout.addLayout(self._tx_row)

        # Arrow
        pipe_layout.addWidget(self._arrow_label())

        # Step 2: Channel
        self._ch_row, self._ch_status, self._ch_label = self._make_step_row(
            'Step 2: Channel', 'Apply channel model -> write i_ph.pwl')
        self._ch_btn = QPushButton('Run Channel')
        self._ch_btn.clicked.connect(lambda: self._run_steps('channel'))
        self._ch_row.addWidget(self._ch_btn)
        pipe_layout.addLayout(self._ch_row)

        # PWL bridge indicator
        pwl_label = QLabel('     [i_ph.pwl bridge file]')
        pwl_label.setStyleSheet('color: #888; font-style: italic;')
        pipe_layout.addWidget(pwl_label)

        # Arrow
        pipe_layout.addWidget(self._arrow_label())

        # Step 3: RX
        self._rx_row, self._rx_status, self._rx_label = self._make_step_row(
            'Step 3: RX', 'Generate netlist -> run SPICE -> parse results')
        self._rx_btn = QPushButton('Run RX')
        self._rx_btn.clicked.connect(lambda: self._run_steps('rx'))
        self._rx_row.addWidget(self._rx_btn)
        pipe_layout.addLayout(self._rx_row)

        # Master button + elapsed time
        btn_row = QHBoxLayout()
        self._run_all_btn = QPushButton('Run All 3 Steps')
        self._run_all_btn.setStyleSheet(
            'font-weight: bold; padding: 8px 20px; font-size: 13px;')
        self._run_all_btn.clicked.connect(lambda: self._run_steps('all'))
        btn_row.addStretch()
        btn_row.addWidget(self._run_all_btn)
        self._elapsed_label = QLabel('')
        self._elapsed_label.setStyleSheet('color: #666; font-size: 11px; padding-left: 10px;')
        btn_row.addWidget(self._elapsed_label)
        btn_row.addStretch()
        pipe_layout.addLayout(btn_row)

        main.addWidget(pipe_grp)

        # ---- Bottom: log + preview + session browser ----
        bottom_splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left: Log + session browser stacked
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)

        # Log panel
        log_grp = QGroupBox('Log')
        log_layout = QVBoxLayout(log_grp)
        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)
        self._log.setMaximumBlockCount(500)
        log_layout.addWidget(self._log)
        left_layout.addWidget(log_grp, stretch=2)

        # Session browser
        session_grp = QGroupBox('Past Sessions')
        session_layout = QVBoxLayout(session_grp)
        self._session_list = QListWidget()
        self._session_list.setMaximumHeight(120)
        session_layout.addWidget(self._session_list)
        sess_btn_row = QHBoxLayout()
        btn_refresh = QPushButton('Refresh')
        btn_refresh.clicked.connect(self._refresh_sessions)
        btn_load = QPushButton('Load Results')
        btn_load.clicked.connect(self._load_session_results)
        sess_btn_row.addWidget(btn_refresh)
        sess_btn_row.addWidget(btn_load)
        sess_btn_row.addStretch()
        session_layout.addLayout(sess_btn_row)
        left_layout.addWidget(session_grp, stretch=1)

        # Waveform preview
        preview_grp = QGroupBox('Waveform Preview')
        preview_layout = QVBoxLayout(preview_grp)
        self._preview_canvas = MplCanvas(width=6, height=4, nrows=2, ncols=2)
        preview_layout.addWidget(self._preview_canvas)

        bottom_splitter.addWidget(left_widget)
        bottom_splitter.addWidget(preview_grp)
        bottom_splitter.setSizes([350, 500])
        main.addWidget(bottom_splitter)

        # Populate sessions on load
        self._refresh_sessions()

    def _make_step_row(self, title, description):
        layout = QHBoxLayout()
        indicator = StatusIndicator()
        label = QLabel(f'<b>{title}</b>: {description}')
        status_label = QLabel('')
        status_label.setStyleSheet('color: #666;')
        layout.addWidget(indicator)
        layout.addWidget(label)
        layout.addStretch()
        layout.addWidget(status_label)
        return layout, indicator, status_label

    def _arrow_label(self):
        lbl = QLabel('        |')
        lbl.setStyleSheet('color: #999; font-size: 14px;')
        return lbl

    def update_config(self, config: SystemConfig):
        self._config = config
        self._update_engine_label()

    def _update_engine_label(self):
        """Update the engine mode indicator based on current config."""
        engine = getattr(self._config, 'simulation_engine', 'spice')
        mod = getattr(self._config, 'modulation', 'OOK')
        if engine == 'python':
            self._engine_mode_label.setText(f'Engine: Python ({mod})')
            self._engine_mode_label.setStyleSheet(
                'font-weight: bold; color: #4CAF50; padding: 2px 8px;'
                'background: #E8F5E9; border-radius: 3px;')
        else:
            self._engine_mode_label.setText(f'Engine: SPICE ({mod})')
            self._engine_mode_label.setStyleSheet(
                'font-weight: bold; color: #2196F3; padding: 2px 8px;'
                'background: #E3F2FD; border-radius: 3px;')

    def set_ltspice_runner(self, runner: LTSpiceRunner):
        """Update LTspice runner (e.g. after user sets a new path)."""
        self._ltspice = runner

    def _run_steps(self, steps):
        if self._worker and self._worker.isRunning():
            self._log_msg('Simulation already running...')
            return

        # Create session
        session_dir = self._session_mgr.create_session(
            label=self._config.preset_name or 'sim')
        self._session_label.setText(f'Session: {session_dir.name}')
        self._session_mgr.save_config(session_dir, self._config)

        self._log_msg(f'Starting pipeline ({steps})...')
        self._log_msg(f'Session: {session_dir}')

        # Reset indicators
        for indicator in [self._tx_status, self._ch_status, self._rx_status]:
            indicator.status = 'pending'
        for label in [self._tx_label, self._ch_label, self._rx_label]:
            label.setText('')

        # Disable buttons and start timer
        self._set_buttons_enabled(False)
        self._elapsed_timer.start()
        self._ui_timer.start()

        # Launch worker thread
        self._worker = _PipelineWorker(
            self._config, session_dir, self._ltspice, steps)
        self._worker.step_update.connect(self._on_step_update)
        self._worker.finished.connect(self._on_finished)
        self._worker.start()

    def _on_step_update(self, step, status, message):
        self._log_msg(f'[{step}] {status}: {message}')

        indicator_map = {
            'TX': self._tx_status,
            'Channel': self._ch_status,
            'RX': self._rx_status,
        }
        label_map = {
            'TX': self._tx_label,
            'Channel': self._ch_label,
            'RX': self._rx_label,
        }

        if step in indicator_map:
            indicator_map[step].status = status
        if step in label_map:
            label_map[step].setText(message)

    def _on_finished(self, results):
        self._ui_timer.stop()
        elapsed_s = self._elapsed_timer.elapsed() / 1000.0
        self._elapsed_label.setText(f'Completed in {elapsed_s:.1f}s')

        self._last_results = results
        self._set_buttons_enabled(True)
        self._log_msg(f'Pipeline complete. ({elapsed_s:.1f}s elapsed)')

        # Update preview plots
        self._update_preview(results)
        self.simulation_done.emit(results)

    def _update_preview(self, results):
        axes = self._preview_canvas.fig.get_axes()
        for ax in axes:
            ax.clear()

        # Check if Python engine result is available for TX/Channel
        rx_res = results.get('RX')
        py_result = (rx_res.outputs.get('python_result')
                     if rx_res and rx_res.status == 'done' else None)

        # TX waveform
        tx_result = results.get('TX')
        if tx_result and tx_result.status == 'done':
            try:
                if py_result and 'P_tx' in py_result:
                    t_ms = py_result['time'] * 1e3
                    p_mw = py_result['P_tx'] * 1e3
                    axes[0].plot(t_ms, p_mw, 'b-', linewidth=0.5)
                    axes[0].set_title('TX: P_tx(t)', fontsize=9)
                    axes[0].set_ylabel('mW', fontsize=8)
                    axes[0].grid(True, alpha=0.3)
                else:
                    pwl_path = tx_result.outputs.get('P_tx_pwl')
                    if pwl_path:
                        data = np.loadtxt(pwl_path, comments=';')
                        t_ms = data[:, 0] * 1e3
                        p_mw = data[:, 1] * 1e3
                        axes[0].plot(t_ms, p_mw, 'b-', linewidth=0.5)
                        axes[0].set_title('TX: P_tx(t)', fontsize=9)
                        axes[0].set_ylabel('mW', fontsize=8)
                        axes[0].grid(True, alpha=0.3)
            except Exception:
                axes[0].text(0.5, 0.5, 'TX data not available',
                             ha='center', va='center', transform=axes[0].transAxes)

        # Channel / I_ph
        ch_result = results.get('Channel')
        if ch_result and ch_result.status == 'done':
            try:
                if py_result and 'I_ph' in py_result:
                    t_ms = py_result['time'] * 1e3
                    i_ua = py_result['I_ph'] * 1e6
                    axes[1].plot(t_ms, i_ua, 'g-', linewidth=0.5)
                    axes[1].set_title('Channel: I_ph(t)', fontsize=9)
                    axes[1].set_ylabel('uA', fontsize=8)
                    axes[1].grid(True, alpha=0.3)
                else:
                    pwl_path = ch_result.outputs.get('optical_pwl')
                    if pwl_path:
                        data = np.loadtxt(pwl_path, comments=';')
                        t_ms = data[:, 0] * 1e3
                        i_ua = data[:, 1] * 1e6
                        axes[1].plot(t_ms, i_ua, 'g-', linewidth=0.5)
                        axes[1].set_title('Channel: I_ph(t)', fontsize=9)
                        axes[1].set_ylabel('uA', fontsize=8)
                        axes[1].grid(True, alpha=0.3)
            except Exception:
                axes[1].text(0.5, 0.5, 'Channel data not available',
                             ha='center', va='center', transform=axes[1].transAxes)

        # RX results from .raw file or Python engine
        rx = results.get('RX')
        if rx and rx.status == 'done':
            python_result = rx.outputs.get('python_result')
            raw_path = rx.outputs.get('raw_file')

            if python_result:
                # Python engine: plot from in-memory arrays
                try:
                    t_ms = python_result['time'] * 1e3
                    V_rx = python_result['V_rx']

                    # Panel 3: Received signal
                    axes[2].plot(t_ms, V_rx * 1e3, 'g-', linewidth=0.3)
                    axes[2].set_title('RX: V_rx(t)', fontsize=9)
                    axes[2].set_ylabel('mV', fontsize=8)
                    axes[2].grid(True, alpha=0.3)

                    # Panel 4: BER info text
                    ber = python_result['ber']
                    snr = python_result['snr_est_dB']
                    mod = python_result['modulation']
                    info_text = (
                        f"Engine: Python\n"
                        f"Modulation: {mod}\n"
                        f"BER: {ber:.4e}\n"
                        f"Errors: {python_result['n_errors']}/{python_result['n_bits_tested']}\n"
                        f"SNR est: {snr:.1f} dB\n"
                        f"P_rx avg: {python_result['P_rx_avg_uW']:.2f} uW"
                    )
                    axes[3].text(0.1, 0.5, info_text,
                                 ha='left', va='center', transform=axes[3].transAxes,
                                 fontsize=9, fontfamily='monospace',
                                 bbox=dict(boxstyle='round', facecolor='lightyellow'))
                    axes[3].set_title('Results Summary', fontsize=9)
                    axes[3].set_axis_off()
                except Exception as e:
                    for i in [2, 3]:
                        axes[i].text(0.5, 0.5, f'Python result error: {e}',
                                     ha='center', va='center', transform=axes[i].transAxes)

            elif raw_path:
                # SPICE engine: parse .raw file
                try:
                    from cosim.raw_parser import LTSpiceRawParser
                    parser = LTSpiceRawParser(raw_path)
                    time = parser.get_time()
                    t_ms = time * 1e3

                    # Panel 3: INA output
                    try:
                        v_ina = parser.get_trace('V(ina_out)')
                        axes[2].plot(t_ms, v_ina * 1e3, 'g-', linewidth=0.3)
                        axes[2].set_title('RX: V(ina_out)', fontsize=9)
                        axes[2].set_ylabel('mV', fontsize=8)
                        axes[2].grid(True, alpha=0.3)
                    except KeyError:
                        axes[2].text(0.5, 0.5, 'V(ina_out) not found',
                                     ha='center', va='center', transform=axes[2].transAxes)

                    # Panel 4: DC-DC output
                    try:
                        v_dcdc = parser.get_trace('V(dcdc_out)')
                        axes[3].plot(t_ms, v_dcdc, 'r-', linewidth=0.3)
                        axes[3].set_title('RX: V(dcdc_out)', fontsize=9)
                        axes[3].set_ylabel('V', fontsize=8)
                        axes[3].grid(True, alpha=0.3)
                    except KeyError:
                        axes[3].text(0.5, 0.5, 'V(dcdc_out) not found',
                                     ha='center', va='center', transform=axes[3].transAxes)
                except Exception as e:
                    for i in [2, 3]:
                        axes[i].text(0.5, 0.5, f'Parse error: {e}',
                                     ha='center', va='center', transform=axes[i].transAxes)
            else:
                for i in [2, 3]:
                    axes[i].text(0.5, 0.5, f'RX: {rx.message}',
                                 ha='center', va='center', transform=axes[i].transAxes,
                                 fontsize=8, color='green')
        else:
            for i in [2, 3]:
                axes[i].text(0.5, 0.5, 'Run simulation to see results',
                             ha='center', va='center', transform=axes[i].transAxes,
                             fontsize=9, color='gray')

        self._preview_canvas.fig.tight_layout()
        self._preview_canvas.draw()

    def _set_buttons_enabled(self, enabled):
        for btn in [self._tx_btn, self._ch_btn, self._rx_btn,
                     self._run_all_btn]:
            btn.setEnabled(enabled)

    def _update_elapsed(self):
        """Update the elapsed time label during simulation."""
        elapsed_s = self._elapsed_timer.elapsed() / 1000.0
        self._elapsed_label.setText(f'Running... {elapsed_s:.0f}s')

    def _log_msg(self, text):
        self._log.appendPlainText(text)

    def _refresh_sessions(self):
        """Populate session list from workspace."""
        self._session_list.clear()
        sessions = self._session_mgr.list_sessions()
        for s in sessions[:20]:  # Show last 20
            summary = self._session_mgr.session_summary(s)
            raw_icon = 'R' if summary['n_raw'] > 0 else '-'
            label = f"{s.name}  [{raw_icon}]"
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, str(s))
            self._session_list.addItem(item)

    def _load_session_results(self):
        """Load results from a selected past session."""
        from cosim.pipeline import StepResult

        item = self._session_list.currentItem()
        if item is None:
            self._log_msg('No session selected')
            return

        from pathlib import Path
        session_dir = Path(item.data(Qt.ItemDataRole.UserRole))
        self._log_msg(f'Loading session: {session_dir.name}')

        # Try to load session config for BER recomputation
        session_config = None
        config_path = session_dir / 'config.json'
        if config_path.exists():
            try:
                session_config = SystemConfig.load(config_path)
                self._log_msg(f'  Config: {session_config.preset_name or "custom"}')
            except Exception as e:
                self._log_msg(f'  Config load failed: {e}')

        # Build results dict from session files
        results = {}

        # TX step
        tx_pwl = session_dir / 'pwl' / 'P_tx.pwl'
        if tx_pwl.exists():
            tx = StepResult('TX')
            tx.status = 'done'
            tx.message = 'Loaded from session'
            tx.outputs = {'P_tx_pwl': str(tx_pwl)}
            results['TX'] = tx

        # Channel step
        ch_pwl = session_dir / 'pwl' / 'optical_power.pwl'
        if ch_pwl.exists():
            ch = StepResult('Channel')
            ch.status = 'done'
            ch.message = 'Loaded from session'
            ch.outputs = {'optical_pwl': str(ch_pwl)}
            results['Channel'] = ch

        # RX step
        raw_files = list((session_dir / 'raw').glob('*.raw')) if (session_dir / 'raw').exists() else []
        if not raw_files:
            raw_files = list(session_dir.rglob('*.raw'))
        if raw_files:
            rx = StepResult('RX')
            rx.status = 'done'
            rx.message = 'Loaded from session'
            rx.outputs = {'raw_file': str(raw_files[0])}

            # Recompute BER from loaded data if we have TX bits and config
            if session_config and tx_pwl.exists():
                try:
                    self._recompute_ber(rx, tx_pwl, raw_files[0], session_config)
                except Exception as e:
                    self._log_msg(f'  BER recompute failed: {e}')
            results['RX'] = rx

        if results:
            self._last_results = results
            self._update_preview(results)
            self.simulation_done.emit(results)
            self._log_msg(f'Loaded {len(results)} steps from session')
            self._session_label.setText(f'Session: {session_dir.name}')
        else:
            self._log_msg('No result files found in session')

    def _recompute_ber(self, rx_result, tx_pwl_path, raw_path, config):
        """Recompute BER from saved TX PWL and RX .raw data."""
        from cosim.raw_parser import LTSpiceRawParser
        from simulation.analysis import calculate_ber_from_transient

        parser = LTSpiceRawParser(str(raw_path))
        time = parser.get_time()

        try:
            v_dout = parser.get_trace('V(dout)')
        except KeyError:
            return

        # Reconstruct TX bits from PWL file
        data = np.loadtxt(str(tx_pwl_path), comments=';')
        bit_period = 1.0 / config.data_rate_bps
        t_max = time[-1]
        max_bits = int(t_max / bit_period)

        # Extract transmitted bits from PWL transitions
        tx_bits = []
        p_max = data[:, 1].max()
        for i in range(max_bits):
            t_center = (i + 0.5) * bit_period
            idx = np.searchsorted(data[:, 0], t_center)
            if idx >= len(data):
                idx = len(data) - 1
            tx_bits.append(1 if data[idx, 1] > p_max * 0.5 else 0)

        tx_bits = np.array(tx_bits)
        threshold = 1.65  # Comparator threshold (Vref)

        # Calculate BER
        ber_info = calculate_ber_from_transient(
            tx_bits, v_dout, time, threshold, bit_period)

        # If BER > 0.4, try inverted polarity
        if ber_info['ber'] > 0.4:
            inverted_bits = 1 - tx_bits
            ber_inv = calculate_ber_from_transient(
                inverted_bits, v_dout, time, threshold, bit_period)
            if ber_inv['ber'] < ber_info['ber']:
                ber_info = ber_inv

        rx_result.outputs['ber'] = ber_info['ber']
        rx_result.outputs['ber_n_errors'] = ber_info['n_errors']
        rx_result.outputs['ber_n_bits'] = ber_info['n_bits_tested']
        self._log_msg(f'  BER: {ber_info["ber"]:.6f} ({ber_info["n_errors"]}/{ber_info["n_bits_tested"]})')
