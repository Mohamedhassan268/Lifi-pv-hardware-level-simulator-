# gui/channel_canvas.py
"""
Inline channel canvas — replaces the windowed ChannelConfigTab.

Shown in the central workspace as the always-visible TX -> Channel -> RX
diagram with the four key controls (distance, TX angle, RX tilt, area)
inline. Link budget and sweep plot are kept directly below — no popup,
no separate tab to switch to.

Public API:
    ChannelCanvasWidget(config)
    .update_config(config)   # called from MainWindow on config_changed
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QDoubleSpinBox,
    QFormLayout, QFrame, QSizePolicy, QTableWidget, QTableWidgetItem,
    QHeaderView, QSplitter,
)
from PyQt6.QtCore import Qt, QRectF, QPointF
from PyQt6.QtGui import QPainter, QPen, QBrush, QColor, QFont, QPainterPath

from cosim.system_config import SystemConfig
from gui.theme import COLORS
from gui.widgets import MplCanvas


class _LinkDiagram(QWidget):
    """Schematic TX -> beam cone -> RX diagram, painted with QPainter."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(180)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._distance_m = 1.0
        self._tx_angle = 0.0
        self._rx_tilt = 0.0
        self._half_angle = 60.0
        self._led_label = ''
        self._pv_label = ''
        self._gain_db = 0.0

    def update_state(self, distance_m, tx_angle, rx_tilt, half_angle,
                     led_label, pv_label, gain_db):
        self._distance_m = distance_m
        self._tx_angle = tx_angle
        self._rx_tilt = rx_tilt
        self._half_angle = half_angle
        self._led_label = led_label
        self._pv_label = pv_label
        self._gain_db = gain_db
        self.update()

    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        w, h = self.width(), self.height()
        pad_x = 60
        cy = h / 2

        c_bg = QColor(COLORS['surface'])
        c_text = QColor(COLORS['text'])
        c_dim = QColor(COLORS['text_dim'])
        c_border = QColor(COLORS['border_strong'])
        c_accent = QColor(COLORS['accent'])

        p.fillRect(self.rect(), c_bg)

        tx_x = pad_x
        rx_x = w - pad_x

        # --- Beam cone (Lambertian footprint hint) ---
        cone_path = QPainterPath()
        half_rad = np.radians(min(self._half_angle, 75))
        spread = (rx_x - tx_x) * np.tan(half_rad) * 0.35
        cone_path.moveTo(tx_x, cy)
        cone_path.lineTo(rx_x, cy - spread)
        cone_path.lineTo(rx_x, cy + spread)
        cone_path.closeSubpath()

        cone_color = QColor(COLORS['accent'])
        cone_color.setAlpha(28)
        p.fillPath(cone_path, QBrush(cone_color))

        # --- Center axis ---
        pen = QPen(c_border, 1, Qt.PenStyle.DashLine)
        p.setPen(pen)
        p.drawLine(int(tx_x), int(cy), int(rx_x), int(cy))

        # --- TX node ---
        tx_size = 28
        tx_rect = QRectF(tx_x - tx_size/2, cy - tx_size/2, tx_size, tx_size)
        p.setPen(QPen(c_accent, 2))
        p.setBrush(QBrush(QColor(COLORS['accent_bg'])))
        p.drawEllipse(tx_rect)

        # --- RX node ---
        rx_size = 32
        rx_rect = QRectF(rx_x - rx_size/2, cy - rx_size/2, rx_size, rx_size)
        p.setPen(QPen(c_text, 2))
        p.setBrush(QBrush(QColor(COLORS['surface_alt'])))
        p.drawRect(rx_rect)

        # --- Labels ---
        font_main = QFont('Inter', 10, QFont.Weight.DemiBold)
        font_dim = QFont('Inter', 9)
        p.setFont(font_main)
        p.setPen(c_text)
        p.drawText(QRectF(tx_x - 80, cy + tx_size/2 + 6, 160, 18),
                   Qt.AlignmentFlag.AlignCenter, 'TX')
        p.drawText(QRectF(rx_x - 80, cy + rx_size/2 + 6, 160, 18),
                   Qt.AlignmentFlag.AlignCenter, 'RX')

        p.setFont(font_dim)
        p.setPen(c_dim)
        if self._led_label:
            p.drawText(QRectF(tx_x - 100, cy + tx_size/2 + 24, 200, 16),
                       Qt.AlignmentFlag.AlignCenter, self._led_label)
        if self._pv_label:
            p.drawText(QRectF(rx_x - 100, cy + rx_size/2 + 24, 200, 16),
                       Qt.AlignmentFlag.AlignCenter, self._pv_label)

        # Distance label centered above axis
        dist_text = f'{self._distance_m * 100:.1f} cm   |   {self._gain_db:+.1f} dB'
        p.setFont(font_main)
        p.setPen(c_text)
        p.drawText(QRectF(tx_x, cy - 36, rx_x - tx_x, 18),
                   Qt.AlignmentFlag.AlignCenter, dist_text)

        # Angle hints
        p.setFont(font_dim)
        p.setPen(c_dim)
        p.drawText(QRectF(tx_x - 60, cy - 56, 160, 14),
                   Qt.AlignmentFlag.AlignCenter,
                   f'theta = {self._tx_angle:.0f}°')
        p.drawText(QRectF(rx_x - 100, cy - 56, 160, 14),
                   Qt.AlignmentFlag.AlignCenter,
                   f'beta = {self._rx_tilt:.0f}°')

        p.end()


class ChannelCanvasWidget(QWidget):
    """Inline channel UI: diagram + 4 controls + link budget + sweep plot."""

    def __init__(self, config: SystemConfig, parent=None):
        super().__init__(parent)
        self._config = config
        self._build_ui()
        self._refresh()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 20, 24, 20)
        root.setSpacing(18)

        # --- Header ---
        header = QLabel('Optical Channel')
        header.setStyleSheet(
            f'font-size: 11px; font-weight: 700; letter-spacing: 1px; '
            f'text-transform: uppercase; color: {COLORS["text_dim"]};')
        root.addWidget(header)

        # --- Diagram ---
        self._diagram = _LinkDiagram()
        root.addWidget(self._diagram)

        # --- Inline controls (single horizontal row) ---
        ctrl_row = QHBoxLayout()
        ctrl_row.setSpacing(20)

        self._distance_spin = self._make_spin(
            0.01, 10.0, 0.01, 3, ' m', self._config.distance_m)
        self._tx_angle_spin = self._make_spin(
            0, 89, 1, 0, ' deg', self._config.tx_angle_deg)
        self._rx_tilt_spin = self._make_spin(
            0, 89, 1, 0, ' deg', self._config.rx_tilt_deg)
        self._area_spin = self._make_spin(
            0.01, 100.0, 0.1, 2, ' cm2', self._config.sc_area_cm2)

        for label, spin in [
            ('Distance', self._distance_spin),
            ('TX Angle', self._tx_angle_spin),
            ('RX Tilt', self._rx_tilt_spin),
            ('RX Area', self._area_spin),
        ]:
            cell = QVBoxLayout()
            cell.setSpacing(4)
            lbl = QLabel(label.upper())
            lbl.setStyleSheet(
                f'font-size: 10px; font-weight: 600; letter-spacing: 0.6px; '
                f'color: {COLORS["text_dim"]};')
            cell.addWidget(lbl)
            cell.addWidget(spin)
            wrap = QWidget()
            wrap.setLayout(cell)
            ctrl_row.addWidget(wrap)
            spin.valueChanged.connect(self._on_param_changed)

        ctrl_row.addStretch()
        root.addLayout(ctrl_row)

        # Hairline separator
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet(f'color: {COLORS["border"]}; max-height: 1px;')
        root.addWidget(sep)

        # --- Bottom: link budget table + sweep plot side-by-side ---
        split = QSplitter(Qt.Orientation.Horizontal)
        split.setHandleWidth(1)

        # Link budget table
        budget_wrap = QWidget()
        budget_lay = QVBoxLayout(budget_wrap)
        budget_lay.setContentsMargins(0, 0, 0, 0)
        budget_lay.setSpacing(6)
        budget_lbl = QLabel('LINK BUDGET')
        budget_lbl.setStyleSheet(
            f'font-size: 10px; font-weight: 600; letter-spacing: 0.6px; '
            f'color: {COLORS["text_dim"]};')
        budget_lay.addWidget(budget_lbl)

        self._budget_table = QTableWidget()
        self._budget_table.setColumnCount(2)
        self._budget_table.setHorizontalHeaderLabels(['Parameter', 'Value'])
        self._budget_table.horizontalHeader().setStretchLastSection(True)
        self._budget_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.ResizeToContents)
        self._budget_table.verticalHeader().setVisible(False)
        self._budget_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._budget_table.setShowGrid(False)
        self._budget_table.setAlternatingRowColors(True)
        budget_lay.addWidget(self._budget_table)
        split.addWidget(budget_wrap)

        # Sweep plot
        plot_wrap = QWidget()
        plot_lay = QVBoxLayout(plot_wrap)
        plot_lay.setContentsMargins(0, 0, 0, 0)
        plot_lay.setSpacing(6)
        plot_lbl = QLabel('DISTANCE SWEEP')
        plot_lbl.setStyleSheet(
            f'font-size: 10px; font-weight: 600; letter-spacing: 0.6px; '
            f'color: {COLORS["text_dim"]};')
        plot_lay.addWidget(plot_lbl)

        self._canvas = MplCanvas(width=6, height=4, nrows=2, ncols=1)
        plot_lay.addWidget(self._canvas)
        split.addWidget(plot_wrap)

        split.setSizes([320, 520])
        root.addWidget(split, 1)

    def _make_spin(self, lo, hi, step, decimals, suffix, value):
        s = QDoubleSpinBox()
        s.setRange(lo, hi)
        s.setSingleStep(step)
        s.setDecimals(decimals)
        s.setSuffix(suffix)
        s.setValue(value)
        s.setMinimumWidth(110)
        return s

    # ---- Public API (kept compatible with old ChannelConfigTab) ----

    def update_config(self, config: SystemConfig):
        self._config = config
        for spin, val in [
            (self._distance_spin, config.distance_m),
            (self._tx_angle_spin, config.tx_angle_deg),
            (self._rx_tilt_spin, config.rx_tilt_deg),
            (self._area_spin, config.sc_area_cm2),
        ]:
            spin.blockSignals(True)
            spin.setValue(val)
            spin.blockSignals(False)
        self._refresh()

    # ---- Internals ----

    def _on_param_changed(self):
        self._config.distance_m = self._distance_spin.value()
        self._config.tx_angle_deg = self._tx_angle_spin.value()
        self._config.rx_tilt_deg = self._rx_tilt_spin.value()
        self._config.sc_area_cm2 = self._area_spin.value()
        self._refresh()

    def _refresh(self):
        cfg = self._config
        try:
            G_ch = cfg.optical_channel_gain()
            P_rx = cfg.received_power_W()
            I_ph = cfg.photocurrent_A()
            m = cfg.lambertian_order()
            path_loss = -10 * np.log10(G_ch) if G_ch > 0 else float('inf')
        except Exception:
            G_ch = P_rx = I_ph = m = 0.0
            path_loss = 0.0

        self._diagram.update_state(
            distance_m=cfg.distance_m,
            tx_angle=cfg.tx_angle_deg,
            rx_tilt=cfg.rx_tilt_deg,
            half_angle=cfg.led_half_angle_deg,
            led_label=cfg.led_part,
            pv_label=cfg.pv_part,
            gain_db=-path_loss if np.isfinite(path_loss) else 0.0,
        )

        items = [
            ('TX Power',     f'{cfg.led_radiated_power_mW:.1f} mW'),
            ('Lambertian m', f'{m:.2f}'),
            ('Half-angle',   f'{cfg.led_half_angle_deg:.0f} deg'),
            ('RX Area',      f'{cfg.sc_area_cm2:.1f} cm2'),
            ('Channel Gain', f'{G_ch:.4e}'),
            ('Path Loss',    f'{path_loss:.1f} dB'),
            ('P_rx',         f'{P_rx*1e6:.2f} uW'),
            ('Responsivity', f'{cfg.sc_responsivity:.3f} A/W'),
            ('I_ph',         f'{I_ph*1e6:.2f} uA'),
        ]
        self._budget_table.setRowCount(len(items))
        for i, (k, v) in enumerate(items):
            self._budget_table.setItem(i, 0, QTableWidgetItem(k))
            self._budget_table.setItem(i, 1, QTableWidgetItem(v))

        self._plot_sweep()

    def _plot_sweep(self):
        cfg = self._config
        distances = np.linspace(5, 150, 100)

        m = cfg.lambertian_order()
        A = cfg.sc_area_cm2 * 1e-4
        theta = np.radians(cfg.tx_angle_deg)
        beta = np.radians(cfg.rx_tilt_deg)
        d_m = distances / 100.0

        G = (m + 1) / (2 * np.pi * d_m**2) * np.cos(theta)**m * np.cos(beta) * A
        P = cfg.led_radiated_power_mW * 1e-3 * G
        P_rx = P * 1e6
        I_ph = P * cfg.sc_responsivity * 1e6

        ax_top, ax_bot = self._canvas.fig.get_axes()
        ax_top.clear(); ax_bot.clear()

        d_cur = cfg.distance_m * 100
        accent = COLORS['accent']
        info = COLORS['info']

        ax_top.plot(distances, P_rx, color=info, linewidth=1.4)
        ax_top.axvline(d_cur, color=accent, linestyle='--', alpha=0.8, linewidth=1.0,
                        label=f'd = {d_cur:.1f} cm')
        ax_top.set_ylabel('P_rx (uW)', fontsize=9)
        ax_top.grid(True, alpha=0.2, linewidth=0.5)
        ax_top.legend(fontsize=8, frameon=False)
        for s in ('top', 'right'):
            ax_top.spines[s].set_visible(False)

        ax_bot.plot(distances, I_ph, color=COLORS['success'], linewidth=1.4)
        ax_bot.axvline(d_cur, color=accent, linestyle='--', alpha=0.8, linewidth=1.0)
        ax_bot.set_xlabel('Distance (cm)', fontsize=9)
        ax_bot.set_ylabel('I_ph (uA)', fontsize=9)
        ax_bot.grid(True, alpha=0.2, linewidth=0.5)
        for s in ('top', 'right'):
            ax_bot.spines[s].set_visible(False)

        self._canvas.fig.tight_layout()
        self._canvas.draw()
