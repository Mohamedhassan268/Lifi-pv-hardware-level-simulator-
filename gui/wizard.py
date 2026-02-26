# gui/wizard.py
"""
Quick Start Wizard — step-by-step guided setup for new users.

3 pages:
  1. Choose a scenario (clickable cards)
  2. Adjust key parameters (distance, power, data rate)
  3. Summary & launch

Returns the configured SystemConfig on accept.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QStackedWidget, QWidget, QDoubleSpinBox, QSlider,
    QGroupBox, QFormLayout, QSizePolicy,
)
from PyQt6.QtCore import Qt, pyqtSignal

from cosim.system_config import SystemConfig
from gui.theme import COLORS


# ── Scenario definitions ──────────────────────────────────────────────

_SCENARIOS = [
    {
        'key': 'kadirvelu2021',
        'title': 'Indoor Desktop LiFi',
        'desc': (
            'Short-range communication + energy harvesting.\n'
            '30 cm distance, 5 kbps OOK modulation.\n'
            'Good starting point for SLIPT research.'),
        'icon': 'TX -- 30cm -- RX',
    },
    {
        'key': 'sarwar2017',
        'title': 'High-Speed OFDM Link',
        'desc': (
            'Multi-carrier OFDM with 16-QAM modulation.\n'
            '1.25 m distance, high data throughput.\n'
            'For advanced modulation experiments.'),
        'icon': 'TX ~~~ 1.25m ~~~ RX',
    },
    {
        'key': 'xu2024',
        'title': 'Low-Power Sensor Node',
        'desc': (
            'BFSK modulation for energy-constrained IoT.\n'
            'Optimized for reliable low-rate communication\n'
            'while maximizing energy harvesting.'),
        'icon': 'TX --- IoT --- RX',
    },
    {
        'key': None,
        'title': 'Custom Setup',
        'desc': (
            'Start with default parameters and\n'
            'configure everything manually.\n'
            'For experienced users.'),
        'icon': '[  ?  ]',
    },
]


class _ScenarioCard(QWidget):
    """Clickable card representing a scenario choice."""

    clicked = pyqtSignal()

    def __init__(self, title, desc, icon_text, parent=None):
        super().__init__(parent)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._selected = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)

        # Icon / diagram line
        icon = QLabel(icon_text)
        icon.setStyleSheet(
            f'color: {COLORS["accent"]}; font-family: monospace; '
            f'font-size: 12px; padding: 4px 0;')
        icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(icon)

        # Title
        title_lbl = QLabel(title)
        title_lbl.setStyleSheet(
            f'font-weight: bold; font-size: 13px; color: {COLORS["text"]};')
        title_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title_lbl)

        # Description
        desc_lbl = QLabel(desc)
        desc_lbl.setWordWrap(True)
        desc_lbl.setStyleSheet(
            f'color: {COLORS["text_dim"]}; font-size: 11px;')
        desc_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(desc_lbl)

        self._update_style()

    def _update_style(self):
        border_color = COLORS['accent'] if self._selected else COLORS['border']
        bg = COLORS['surface'] if self._selected else COLORS['surface_alt']
        self.setStyleSheet(
            f'_ScenarioCard {{ '
            f'  background: {bg}; border: 2px solid {border_color}; '
            f'  border-radius: 8px; '
            f'}}')

    @property
    def selected(self):
        return self._selected

    @selected.setter
    def selected(self, val):
        self._selected = val
        self._update_style()

    def mousePressEvent(self, event):
        self.clicked.emit()
        super().mousePressEvent(event)


class QuickStartWizard(QDialog):
    """3-page guided setup wizard."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Quick Start Guide')
        self.setMinimumSize(650, 480)
        self.resize(700, 520)
        self.setStyleSheet(f'background: {COLORS["bg"]};')

        self._chosen_preset = 'kadirvelu2021'
        self._config = None

        main = QVBoxLayout(self)

        # Header
        header = QLabel('Quick Start Guide')
        header.setStyleSheet(
            f'font-size: 18px; font-weight: bold; color: {COLORS["accent"]}; '
            f'padding: 10px 0 5px 0;')
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main.addWidget(header)

        # Page stack
        self._stack = QStackedWidget()
        main.addWidget(self._stack, stretch=1)

        self._build_page1()
        self._build_page2()
        self._build_page3()

        # Navigation buttons
        nav = QHBoxLayout()
        self._btn_back = QPushButton('Back')
        self._btn_back.clicked.connect(self._go_back)
        self._btn_back.setVisible(False)
        self._btn_next = QPushButton('Next')
        self._btn_next.clicked.connect(self._go_next)
        self._btn_next.setStyleSheet(
            f'font-weight: bold; padding: 6px 20px;')

        nav.addStretch()
        nav.addWidget(self._btn_back)
        nav.addWidget(self._btn_next)
        main.addLayout(nav)

    # ── Page 1: Choose Scenario ──────────────────────────────────────

    def _build_page1(self):
        page = QWidget()
        layout = QVBoxLayout(page)

        subtitle = QLabel('What would you like to simulate?')
        subtitle.setStyleSheet(
            f'color: {COLORS["text"]}; font-size: 13px; padding: 5px 0;')
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(subtitle)

        cards_layout = QHBoxLayout()
        cards_layout.setSpacing(10)

        self._cards = []
        for scenario in _SCENARIOS:
            card = _ScenarioCard(
                scenario['title'], scenario['desc'], scenario['icon'])
            card.clicked.connect(
                lambda s=scenario: self._select_scenario(s))
            cards_layout.addWidget(card)
            self._cards.append(card)

        # Select first by default
        self._cards[0].selected = True
        layout.addLayout(cards_layout)
        layout.addStretch()
        self._stack.addWidget(page)

    def _select_scenario(self, scenario):
        self._chosen_preset = scenario['key']
        for card in self._cards:
            card.selected = False
        # Find and select the clicked card
        idx = _SCENARIOS.index(scenario)
        self._cards[idx].selected = True

    # ── Page 2: Adjust Parameters ────────────────────────────────────

    def _build_page2(self):
        page = QWidget()
        layout = QVBoxLayout(page)

        subtitle = QLabel('Adjust the key parameters')
        subtitle.setStyleSheet(
            f'color: {COLORS["text"]}; font-size: 13px; padding: 5px 0;')
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(subtitle)

        hint = QLabel(
            'These are the parameters that matter most.\n'
            'You can fine-tune everything else in System Setup later.')
        hint.setStyleSheet(
            f'color: {COLORS["text_dim"]}; font-size: 11px;')
        hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(hint)

        form_grp = QGroupBox('Key Parameters')
        form = QFormLayout(form_grp)

        # Distance
        self._wiz_distance = QDoubleSpinBox()
        self._wiz_distance.setRange(0.01, 5.0)
        self._wiz_distance.setSingleStep(0.05)
        self._wiz_distance.setDecimals(2)
        self._wiz_distance.setSuffix(' m')
        self._wiz_distance.setToolTip(
            'Distance between LED and PV cell.\n'
            'This has the biggest impact on signal strength.')
        self._wiz_distance.valueChanged.connect(self._update_snr_preview)
        form.addRow('Distance:', self._wiz_distance)

        # LED Power
        self._wiz_power = QDoubleSpinBox()
        self._wiz_power.setRange(0.1, 100.0)
        self._wiz_power.setSingleStep(1.0)
        self._wiz_power.setSuffix(' mW')
        self._wiz_power.setToolTip(
            'LED optical output power.\n'
            'More power = longer range but higher consumption.')
        self._wiz_power.valueChanged.connect(self._update_snr_preview)
        form.addRow('LED Power:', self._wiz_power)

        # Data Rate
        self._wiz_rate = QDoubleSpinBox()
        self._wiz_rate.setRange(100, 1e6)
        self._wiz_rate.setDecimals(0)
        self._wiz_rate.setSuffix(' bps')
        self._wiz_rate.setToolTip(
            'Communication bit rate.\n'
            'Higher rates need more SNR to work reliably.')
        self._wiz_rate.valueChanged.connect(self._update_snr_preview)
        form.addRow('Data Rate:', self._wiz_rate)

        layout.addWidget(form_grp)

        # SNR preview bar
        snr_grp = QGroupBox('Link Quality Preview')
        snr_layout = QVBoxLayout(snr_grp)
        self._snr_label = QLabel('SNR: -- dB')
        self._snr_label.setStyleSheet(
            f'font-weight: bold; font-size: 14px; color: {COLORS["text"]};')
        self._snr_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        snr_layout.addWidget(self._snr_label)

        self._snr_verdict = QLabel('')
        self._snr_verdict.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._snr_verdict.setStyleSheet('font-size: 12px;')
        snr_layout.addWidget(self._snr_verdict)
        layout.addWidget(snr_grp)

        layout.addStretch()
        self._stack.addWidget(page)

    def _update_snr_preview(self):
        """Recalculate SNR from wizard parameters and update the preview."""
        if self._config is None:
            return
        try:
            from dataclasses import replace
            cfg = replace(
                self._config,
                distance_m=self._wiz_distance.value(),
                led_radiated_power_mW=self._wiz_power.value(),
                data_rate_bps=self._wiz_rate.value(),
            )
            snr = cfg.snr_estimate_dB()
            self._snr_label.setText(f'SNR: {snr:.1f} dB')
            if snr > 15:
                self._snr_verdict.setText('Excellent link quality')
                self._snr_verdict.setStyleSheet(
                    f'font-size: 12px; color: {COLORS["success"]};')
            elif snr > 10:
                self._snr_verdict.setText('Good link quality')
                self._snr_verdict.setStyleSheet(
                    f'font-size: 12px; color: {COLORS["success"]};')
            elif snr > 6:
                self._snr_verdict.setText(
                    'Marginal \u2014 may have bit errors')
                self._snr_verdict.setStyleSheet(
                    f'font-size: 12px; color: {COLORS["warning"]};')
            else:
                self._snr_verdict.setText(
                    'Too weak \u2014 reduce distance or increase power')
                self._snr_verdict.setStyleSheet(
                    f'font-size: 12px; color: {COLORS["error"]};')
            # Store tweaked config
            self._config = cfg
        except Exception:
            self._snr_label.setText('SNR: --')
            self._snr_verdict.setText('')

    # ── Page 3: Summary ──────────────────────────────────────────────

    def _build_page3(self):
        page = QWidget()
        layout = QVBoxLayout(page)

        subtitle = QLabel('Ready to simulate!')
        subtitle.setStyleSheet(
            f'color: {COLORS["accent"]}; font-size: 15px; '
            f'font-weight: bold; padding: 10px 0;')
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(subtitle)

        self._summary_label = QLabel('')
        self._summary_label.setWordWrap(True)
        self._summary_label.setStyleSheet(
            f'color: {COLORS["text"]}; font-size: 12px; '
            f'background: {COLORS["surface"]}; padding: 15px; '
            f'border-radius: 6px; border: 1px solid {COLORS["border"]};')
        layout.addWidget(self._summary_label)

        hint = QLabel(
            'Click "Apply" to load this configuration into the simulator.\n'
            'You can fine-tune any parameter in the System Setup tab.')
        hint.setStyleSheet(
            f'color: {COLORS["text_dim"]}; font-size: 11px; padding: 10px 0;')
        hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(hint)

        layout.addStretch()
        self._stack.addWidget(page)

    def _update_summary(self):
        if self._config is None:
            return
        c = self._config
        try:
            snr = c.snr_estimate_dB()
            snr_text = f'{snr:.1f} dB'
        except Exception:
            snr_text = 'N/A'

        preset_name = c.preset_name or 'Custom'
        engine = getattr(c, 'simulation_engine', 'spice')
        mod = getattr(c, 'modulation', 'OOK')

        text = (
            f'<b>Preset:</b> {preset_name}<br>'
            f'<b>Engine:</b> {engine.upper()} | <b>Modulation:</b> {mod}<br>'
            f'<br>'
            f'<b>Distance:</b> {c.distance_m * 100:.1f} cm<br>'
            f'<b>LED Power:</b> {c.led_radiated_power_mW:.1f} mW<br>'
            f'<b>Data Rate:</b> {c.data_rate_bps:.0f} bps<br>'
            f'<b>Estimated SNR:</b> {snr_text}<br>'
            f'<br>'
            f'<b>LED:</b> {c.led_part} | <b>PV:</b> {c.pv_part}')
        self._summary_label.setText(text)

    # ── Navigation ───────────────────────────────────────────────────

    def _go_next(self):
        idx = self._stack.currentIndex()
        if idx == 0:
            # Load the chosen preset
            if self._chosen_preset:
                self._config = SystemConfig.from_preset(self._chosen_preset)
            else:
                self._config = SystemConfig.from_preset('kadirvelu2021')
            # Populate page 2 from config
            self._wiz_distance.setValue(self._config.distance_m)
            self._wiz_power.setValue(self._config.led_radiated_power_mW)
            self._wiz_rate.setValue(self._config.data_rate_bps)
            self._update_snr_preview()
            self._stack.setCurrentIndex(1)
            self._btn_back.setVisible(True)
        elif idx == 1:
            self._update_summary()
            self._stack.setCurrentIndex(2)
            self._btn_next.setText('Apply')
        elif idx == 2:
            self.accept()

    def _go_back(self):
        idx = self._stack.currentIndex()
        if idx > 0:
            self._stack.setCurrentIndex(idx - 1)
            self._btn_next.setText('Next')
            if idx - 1 == 0:
                self._btn_back.setVisible(False)

    def get_config(self):
        """Return the configured SystemConfig after wizard completes."""
        return self._config
