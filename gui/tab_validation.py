# gui/tab_validation.py
"""
Tab 7: Validation

Paper preset selector + validation table comparing
simulated values against published targets.
Reads .raw files to extract measured values after simulation.
"""

import sys, os, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QComboBox, QPushButton, QTableWidget, QTableWidgetItem,
    QHeaderView, QScrollArea, QSplitter, QListWidget,
    QListWidgetItem,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QColor, QPixmap

from cosim.system_config import SystemConfig
from cosim.raw_parser import LTSpiceRawParser


class ValidationTab(QWidget):
    """Tab 7: Compare simulation results against paper targets."""

    def __init__(self, config: SystemConfig, parent=None):
        super().__init__(parent)
        self._config = config
        self._results = {}
        self._parser = None
        self._build_ui()
        self._update_targets()

    def _build_ui(self):
        main = QVBoxLayout(self)

        # Preset selector
        top_row = QHBoxLayout()
        top_row.addWidget(QLabel('Paper Preset:'))
        self._preset_combo = QComboBox()
        for name in SystemConfig.list_presets():
            self._preset_combo.addItem(name)
        self._preset_combo.currentTextChanged.connect(self._on_preset_changed)
        top_row.addWidget(self._preset_combo)
        top_row.addStretch()
        main.addLayout(top_row)

        # Paper info
        self._paper_label = QLabel()
        self._paper_label.setWordWrap(True)
        self._paper_label.setStyleSheet('color: #555; padding: 5px;')
        main.addWidget(self._paper_label)

        # Validation table
        self._table = QTableWidget()
        self._table.setColumnCount(5)
        self._table.setHorizontalHeaderLabels([
            'Parameter', 'Target', 'Simulated', 'Error (%)', 'Status'
        ])
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        main.addWidget(self._table)

        # Summary
        self._summary_label = QLabel()
        self._summary_label.setStyleSheet(
            'font-size: 13px; font-weight: bold; padding: 10px;')
        main.addWidget(self._summary_label)

        # ── Paper Figure Generation ──
        fig_group = QGroupBox('Paper Figure Generation')
        fig_layout = QVBoxLayout(fig_group)

        fig_top = QHBoxLayout()
        fig_top.addWidget(QLabel('Paper:'))
        self._fig_paper_combo = QComboBox()
        self._fig_paper_combo.addItem('-- All Papers --', 'all')
        try:
            from papers import list_papers
            for key, label, ref in list_papers():
                self._fig_paper_combo.addItem(f'{label}', key)
        except ImportError:
            pass
        fig_top.addWidget(self._fig_paper_combo)

        self._fig_run_btn = QPushButton('Generate Figures')
        self._fig_run_btn.clicked.connect(self._run_figure_generation)
        fig_top.addWidget(self._fig_run_btn)
        fig_top.addStretch()
        fig_layout.addLayout(fig_top)

        self._fig_status = QLabel('Ready')
        self._fig_status.setStyleSheet('color: #555; padding: 2px;')
        fig_layout.addWidget(self._fig_status)

        # Image list + preview
        fig_splitter = QSplitter(Qt.Orientation.Horizontal)
        self._fig_list = QListWidget()
        self._fig_list.currentItemChanged.connect(self._on_figure_selected)
        fig_splitter.addWidget(self._fig_list)

        self._fig_preview = QLabel('Select a figure to preview')
        self._fig_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._fig_preview.setMinimumSize(400, 300)
        self._fig_preview.setStyleSheet('background: #f0f0f0; border: 1px solid #ccc;')
        scroll = QScrollArea()
        scroll.setWidget(self._fig_preview)
        scroll.setWidgetResizable(True)
        fig_splitter.addWidget(scroll)
        fig_splitter.setSizes([200, 500])

        fig_layout.addWidget(fig_splitter)
        main.addWidget(fig_group)

    def update_config(self, config: SystemConfig):
        self._config = config
        self._update_targets()

    def update_results(self, results: dict):
        """Called when simulation completes."""
        self._results = results

        # Load .raw parser
        rx = results.get('RX')
        if rx and hasattr(rx, 'outputs'):
            raw_path = rx.outputs.get('raw_file')
            if raw_path:
                try:
                    self._parser = LTSpiceRawParser(raw_path)
                except Exception:
                    self._parser = None

        self._update_table_with_results()

    def _on_preset_changed(self, name):
        if name:
            try:
                self._config = SystemConfig.from_preset(name)
                self._update_targets()
            except Exception:
                pass

    def _update_targets(self):
        cfg = self._config

        if cfg.paper_reference:
            self._paper_label.setText(f'Reference: {cfg.paper_reference}')
        else:
            self._paper_label.setText('No paper reference set')

        # Build target rows
        rows = self._build_validation_rows()

        self._table.setRowCount(len(rows))
        for i, row in enumerate(rows):
            self._table.setItem(i, 0, QTableWidgetItem(row['param']))
            self._table.setItem(i, 1, QTableWidgetItem(row['target_str']))
            self._table.setItem(i, 2, QTableWidgetItem('--'))
            self._table.setItem(i, 3, QTableWidgetItem('--'))
            item = QTableWidgetItem('Pending')
            item.setForeground(QColor(150, 150, 150))
            self._table.setItem(i, 4, item)

        self._summary_label.setText(
            'Run simulation (Tab 4) then return here to see validation results')

    def _build_validation_rows(self):
        """Build list of validation parameter rows."""
        cfg = self._config
        rows = []

        # Paper targets
        if cfg.target_harvested_power_uW is not None:
            rows.append({
                'param': 'Harvested Power (uW)',
                'target': cfg.target_harvested_power_uW,
                'target_str': f'{cfg.target_harvested_power_uW:.1f}',
                'extract': 'harvest_power',
            })
        if cfg.target_ber is not None:
            rows.append({
                'param': 'BER',
                'target': cfg.target_ber,
                'target_str': f'{cfg.target_ber:.4e}',
                'extract': 'ber',
            })
        if cfg.target_noise_rms_mV is not None:
            rows.append({
                'param': 'Noise RMS (mV)',
                'target': cfg.target_noise_rms_mV,
                'target_str': f'{cfg.target_noise_rms_mV:.3f}',
                'extract': 'noise_rms',
            })

        # Link budget (always show)
        try:
            G_ch = cfg.optical_channel_gain()
            P_rx = cfg.received_power_W() * 1e6
            I_ph = cfg.photocurrent_A() * 1e6
            m = cfg.lambertian_order()

            rows.extend([
                {'param': 'Lambertian Order m', 'target': m,
                 'target_str': f'{m:.2f}', 'extract': 'lambertian'},
                {'param': 'Channel Gain', 'target': G_ch,
                 'target_str': f'{G_ch:.4e}', 'extract': 'channel_gain'},
                {'param': 'P_rx (uW)', 'target': P_rx,
                 'target_str': f'{P_rx:.2f}', 'extract': 'p_rx'},
                {'param': 'I_ph (uA)', 'target': I_ph,
                 'target_str': f'{I_ph:.2f}', 'extract': 'i_ph'},
                {'param': 'V_dcdc (V)', 'target': None,
                 'target_str': '--', 'extract': 'v_dcdc'},
                {'param': 'V_INA RMS (mV)', 'target': None,
                 'target_str': '--', 'extract': 'v_ina_rms'},
            ])
        except Exception:
            pass

        return rows

    def _update_table_with_results(self):
        """Fill in simulated column from .raw data."""
        rows = self._build_validation_rows()
        cfg = self._config

        # Extract simulated values
        simulated = {}

        # From channel step
        ch = self._results.get('Channel')
        if ch and hasattr(ch, 'outputs'):
            simulated['channel_gain'] = ch.outputs.get('G_ch')
            simulated['p_rx'] = ch.outputs.get('P_rx_avg_uW')
            simulated['i_ph'] = ch.outputs.get('I_ph_avg_uA')

        # From .raw file
        if self._parser is not None:
            time = self._parser.get_time()
            n_half = len(time) // 2

            try:
                v_dcdc = self._parser.get_trace('V(dcdc_out)')
                v_dcdc_avg = np.mean(v_dcdc[n_half:])
                simulated['v_dcdc'] = v_dcdc_avg

                R_load = cfg.r_load_ohm
                v_rms = np.sqrt(np.mean(v_dcdc[n_half:]**2))
                simulated['harvest_power'] = v_rms**2 / R_load * 1e6
            except KeyError:
                pass

            try:
                v_ina = self._parser.get_trace('V(ina_out)')
                v_ina_ss = v_ina[n_half:]
                v_ina_ac = v_ina_ss - np.mean(v_ina_ss)
                simulated['v_ina_rms'] = np.sqrt(np.mean(v_ina_ac**2)) * 1e3
            except KeyError:
                pass

            try:
                v_bpf = self._parser.get_trace('V(bpf_out)')
                v_bpf_ss = v_bpf[n_half:]
                v_bpf_ac = v_bpf_ss - np.mean(v_bpf_ss)
                simulated['noise_rms'] = np.sqrt(np.mean(v_bpf_ac**2)) * 1e3
            except KeyError:
                pass

        # From pipeline BER computation
        rx = self._results.get('RX')
        if rx and hasattr(rx, 'outputs'):
            ber_val = rx.outputs.get('ber')
            if ber_val is not None:
                simulated['ber'] = ber_val
            snr_val = rx.outputs.get('bpf_snr_dB')
            if snr_val is not None:
                simulated['snr_dB'] = snr_val

        # Derived
        try:
            simulated['lambertian'] = cfg.lambertian_order()
        except Exception:
            pass

        # Fill table
        n_pass = 0
        n_fail = 0
        n_total = 0

        self._table.setRowCount(len(rows))
        for i, row in enumerate(rows):
            self._table.setItem(i, 0, QTableWidgetItem(row['param']))
            self._table.setItem(i, 1, QTableWidgetItem(row['target_str']))

            key = row['extract']
            sim_val = simulated.get(key)

            if sim_val is not None:
                # Format simulated value
                if abs(sim_val) < 0.01 and sim_val != 0:
                    sim_str = f'{sim_val:.4e}'
                else:
                    sim_str = f'{sim_val:.4f}'
                self._table.setItem(i, 2, QTableWidgetItem(sim_str))

                # Compute error
                target = row['target']
                if target is not None and target != 0:
                    error_pct = abs(sim_val - target) / abs(target) * 100
                    self._table.setItem(i, 3,
                                         QTableWidgetItem(f'{error_pct:.1f}%'))

                    n_total += 1
                    if error_pct < 20:
                        status = 'PASS'
                        color = QColor(40, 160, 40)
                        n_pass += 1
                    else:
                        status = 'FAIL'
                        color = QColor(200, 40, 40)
                        n_fail += 1

                    item = QTableWidgetItem(status)
                    item.setForeground(color)
                    self._table.setItem(i, 4, item)
                else:
                    self._table.setItem(i, 3, QTableWidgetItem('--'))
                    item = QTableWidgetItem('Measured')
                    item.setForeground(QColor(0, 100, 200))
                    self._table.setItem(i, 4, item)
            else:
                self._table.setItem(i, 2, QTableWidgetItem('--'))
                self._table.setItem(i, 3, QTableWidgetItem('--'))
                item = QTableWidgetItem('No data')
                item.setForeground(QColor(150, 150, 150))
                self._table.setItem(i, 4, item)

        if n_total > 0:
            self._summary_label.setText(
                f'Validation: {n_pass}/{n_total} passed, '
                f'{n_fail}/{n_total} failed '
                f'(threshold: 20% error)')
        else:
            self._summary_label.setText(
                'Simulation data loaded. '
                'Some targets have no comparison reference.')

    # ── Figure Generation ──

    def _run_figure_generation(self):
        """Run paper validation to generate figures in a background thread."""
        paper_key = self._fig_paper_combo.currentData()
        self._fig_run_btn.setEnabled(False)
        self._fig_status.setText('Running validation...')
        self._fig_status.setStyleSheet('color: #0066cc;')

        self._fig_thread = _FigureWorker(paper_key)
        self._fig_thread.finished.connect(self._on_figures_done)
        self._fig_thread.start()

    def _on_figures_done(self, output_dir, success, message):
        """Called when figure generation thread completes."""
        self._fig_run_btn.setEnabled(True)
        if success:
            self._fig_status.setText(f'Done: {message}')
            self._fig_status.setStyleSheet('color: green;')
            self._load_figure_list(output_dir)
        else:
            self._fig_status.setText(f'Error: {message}')
            self._fig_status.setStyleSheet('color: red;')

    def _load_figure_list(self, output_dir):
        """Populate figure list from generated PNGs."""
        self._fig_list.clear()
        if not os.path.isdir(output_dir):
            return
        png_files = sorted(glob.glob(os.path.join(output_dir, '**', '*.png'),
                                      recursive=True))
        for path in png_files:
            rel = os.path.relpath(path, output_dir)
            item = QListWidgetItem(rel)
            item.setData(Qt.ItemDataRole.UserRole, path)
            self._fig_list.addItem(item)

        if self._fig_list.count() > 0:
            self._fig_list.setCurrentRow(0)

    def _on_figure_selected(self, current, previous):
        """Show selected figure in preview."""
        if current is None:
            return
        path = current.data(Qt.ItemDataRole.UserRole)
        if path and os.path.isfile(path):
            pixmap = QPixmap(path)
            if not pixmap.isNull():
                scaled = pixmap.scaled(
                    self._fig_preview.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation)
                self._fig_preview.setPixmap(scaled)
            else:
                self._fig_preview.setText('Could not load image')


class _FigureWorker(QThread):
    """Background thread for running paper validation."""
    finished = pyqtSignal(str, bool, str)

    def __init__(self, paper_key):
        super().__init__()
        self._paper_key = paper_key

    def run(self):
        try:
            from papers import PAPERS, run_paper, run_all_papers
            base_dir = os.path.join('workspace', 'validation')

            if self._paper_key == 'all':
                results = run_all_papers(base_dir)
                n_pass = sum(results.values())
                msg = f'{n_pass}/{len(results)} papers passed'
                self.finished.emit(base_dir, True, msg)
            else:
                out = os.path.join(base_dir, self._paper_key)
                passed = run_paper(self._paper_key, out)
                msg = f'{"PASS" if passed else "FAIL"}'
                self.finished.emit(out, True, msg)
        except Exception as e:
            self.finished.emit('', False, str(e))
