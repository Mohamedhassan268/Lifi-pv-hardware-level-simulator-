# gui/tab_channel_config.py
"""
Tab 3: Channel Configuration

- Distance / angle / area spinboxes
- Link budget display table
- Distance sweep plot
- Shows selected LED + PV from SystemConfig
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QDoubleSpinBox, QFormLayout, QTableWidget, QTableWidgetItem,
    QPushButton, QSplitter, QHeaderView,
)
from PyQt6.QtCore import Qt

from cosim.system_config import SystemConfig
from gui.theme import COLORS
from gui.widgets import MplCanvas


class ChannelConfigTab(QWidget):
    """Tab 3: Optical channel link budget and distance sweep."""

    def __init__(self, config: SystemConfig, parent=None):
        super().__init__(parent)
        self._config = config
        self._build_ui()
        self._update_link_budget()

    def _build_ui(self):
        main = QHBoxLayout(self)
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # ---- Left: controls + link budget ----
        left = QWidget()
        left_layout = QVBoxLayout(left)

        # Component info
        self._info_label = QLabel()
        self._info_label.setStyleSheet(f'font-size: 11px; color: {COLORS["text_dim"]};')
        left_layout.addWidget(self._info_label)

        # Controls
        ctrl_grp = QGroupBox('Channel Parameters')
        ctrl_form = QFormLayout(ctrl_grp)

        self._distance_spin = QDoubleSpinBox()
        self._distance_spin.setRange(0.01, 10.0)
        self._distance_spin.setSingleStep(0.01)
        self._distance_spin.setDecimals(3)
        self._distance_spin.setSuffix(' m')
        self._distance_spin.setValue(self._config.distance_m)
        self._distance_spin.valueChanged.connect(self._on_param_changed)
        ctrl_form.addRow('Distance:', self._distance_spin)

        self._tx_angle_spin = QDoubleSpinBox()
        self._tx_angle_spin.setRange(0, 89)
        self._tx_angle_spin.setSuffix(' deg')
        self._tx_angle_spin.setValue(self._config.tx_angle_deg)
        self._tx_angle_spin.valueChanged.connect(self._on_param_changed)
        ctrl_form.addRow('TX Angle:', self._tx_angle_spin)

        self._rx_tilt_spin = QDoubleSpinBox()
        self._rx_tilt_spin.setRange(0, 89)
        self._rx_tilt_spin.setSuffix(' deg')
        self._rx_tilt_spin.setValue(self._config.rx_tilt_deg)
        self._rx_tilt_spin.valueChanged.connect(self._on_param_changed)
        ctrl_form.addRow('RX Tilt:', self._rx_tilt_spin)

        self._area_spin = QDoubleSpinBox()
        self._area_spin.setRange(0.01, 100.0)
        self._area_spin.setSuffix(' cm2')
        self._area_spin.setDecimals(2)
        self._area_spin.setValue(self._config.sc_area_cm2)
        self._area_spin.valueChanged.connect(self._on_param_changed)
        ctrl_form.addRow('RX Area:', self._area_spin)

        left_layout.addWidget(ctrl_grp)

        # Link budget table
        budget_grp = QGroupBox('Link Budget')
        budget_lay = QVBoxLayout(budget_grp)
        self._budget_table = QTableWidget()
        self._budget_table.setColumnCount(2)
        self._budget_table.setHorizontalHeaderLabels(['Parameter', 'Value'])
        self._budget_table.horizontalHeader().setStretchLastSection(True)
        self._budget_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.ResizeToContents)
        self._budget_table.setEditTriggers(
            QTableWidget.EditTrigger.NoEditTriggers)
        budget_lay.addWidget(self._budget_table)
        left_layout.addWidget(budget_grp)
        left_layout.addStretch()

        # ---- Right: distance sweep plot ----
        right = QWidget()
        right_layout = QVBoxLayout(right)

        self._canvas = MplCanvas(width=6, height=5, nrows=2, ncols=1)
        right_layout.addWidget(self._canvas)

        btn_sweep = QPushButton('Refresh Sweep Plot')
        btn_sweep.clicked.connect(self._plot_sweep)
        right_layout.addWidget(btn_sweep)

        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setSizes([350, 450])
        main.addWidget(splitter)

    def update_config(self, config: SystemConfig):
        """Called when config changes externally."""
        self._config = config
        self._distance_spin.setValue(config.distance_m)
        self._tx_angle_spin.setValue(config.tx_angle_deg)
        self._rx_tilt_spin.setValue(config.rx_tilt_deg)
        self._area_spin.setValue(config.sc_area_cm2)
        self._update_link_budget()

    def _on_param_changed(self):
        self._config.distance_m = self._distance_spin.value()
        self._config.tx_angle_deg = self._tx_angle_spin.value()
        self._config.rx_tilt_deg = self._rx_tilt_spin.value()
        self._config.sc_area_cm2 = self._area_spin.value()
        self._update_link_budget()

    def _update_link_budget(self):
        cfg = self._config
        self._info_label.setText(
            f'LED: {cfg.led_part}  |  PV: {cfg.pv_part}  |  '
            f'Responsivity: {cfg.sc_responsivity:.3f} A/W')

        try:
            G_ch = cfg.optical_channel_gain()
            P_rx = cfg.received_power_W()
            I_ph = cfg.photocurrent_A()
            m = cfg.lambertian_order()
            path_loss = -10 * np.log10(G_ch) if G_ch > 0 else float('inf')

            items = [
                ('TX Power', f'{cfg.led_radiated_power_mW:.1f} mW'),
                ('Lambertian m', f'{m:.2f}'),
                ('Half-angle', f'{cfg.led_half_angle_deg:.0f} deg'),
                ('Distance', f'{cfg.distance_m*100:.1f} cm'),
                ('RX Area', f'{cfg.sc_area_cm2:.1f} cm2'),
                ('Channel Gain', f'{G_ch:.4e}'),
                ('Path Loss', f'{path_loss:.1f} dB'),
                ('P_rx', f'{P_rx*1e6:.2f} uW'),
                ('Responsivity', f'{cfg.sc_responsivity:.3f} A/W'),
                ('I_ph', f'{I_ph*1e6:.2f} uA'),
            ]

            self._budget_table.setRowCount(len(items))
            for i, (k, v) in enumerate(items):
                self._budget_table.setItem(i, 0, QTableWidgetItem(k))
                self._budget_table.setItem(i, 1, QTableWidgetItem(v))

        except Exception:
            pass

        self._plot_sweep()

    def _plot_sweep(self):
        cfg = self._config
        distances = np.linspace(5, 150, 100)  # cm

        P_rx = []
        I_ph = []
        for d in distances:
            cfg_copy_dist = d / 100.0
            m = cfg.lambertian_order()
            A = cfg.sc_area_cm2 * 1e-4
            theta = np.radians(cfg.tx_angle_deg)
            beta = np.radians(cfg.rx_tilt_deg)
            G = (m + 1) / (2 * np.pi * cfg_copy_dist**2) * \
                np.cos(theta)**m * np.cos(beta) * A
            P = cfg.led_radiated_power_mW * 1e-3 * G
            P_rx.append(P * 1e6)
            I_ph.append(P * cfg.sc_responsivity * 1e6)

        ax_top, ax_bot = self._canvas.fig.get_axes()
        ax_top.clear()
        ax_bot.clear()

        # Current distance marker
        d_cur = cfg.distance_m * 100

        ax_top.plot(distances, P_rx, 'b-', linewidth=1.5)
        ax_top.axvline(d_cur, color='r', linestyle='--', alpha=0.7,
                        label=f'd = {d_cur:.1f} cm')
        ax_top.set_ylabel('P_rx (uW)')
        ax_top.set_title('Received Power vs Distance')
        ax_top.grid(True, alpha=0.3)
        ax_top.legend(fontsize=8)

        ax_bot.plot(distances, I_ph, 'g-', linewidth=1.5)
        ax_bot.axvline(d_cur, color='r', linestyle='--', alpha=0.7)
        ax_bot.set_xlabel('Distance (cm)')
        ax_bot.set_ylabel('I_ph (uA)')
        ax_bot.set_title('Photocurrent vs Distance')
        ax_bot.grid(True, alpha=0.3)

        self._canvas.fig.tight_layout()
        self._canvas.draw()
