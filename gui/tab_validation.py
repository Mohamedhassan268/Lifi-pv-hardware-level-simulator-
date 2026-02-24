# gui/tab_validation.py
"""
Tab 7: Validation

Paper preset selector + validation table comparing
simulated values against published targets.
Reads .raw files to extract measured values after simulation.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QComboBox, QPushButton, QTableWidget, QTableWidgetItem,
    QHeaderView,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor

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
