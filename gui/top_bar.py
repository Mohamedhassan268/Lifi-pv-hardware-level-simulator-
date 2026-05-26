# gui/top_bar.py
"""
Top control bar — replaces the legacy "System Setup" window.

Anchors only the two controls that drive most of the workflow:
    - Simulation Preset
    - Modulation scheme
plus quick actions: Wizard, Save, Load, Run.

Emits signals; MainWindow wires them to existing handlers.
"""

from PyQt6.QtWidgets import (
    QToolBar, QWidget, QLabel, QComboBox, QPushButton,
    QHBoxLayout, QSizePolicy, QFrame,
)
from PyQt6.QtCore import Qt, pyqtSignal

from cosim.system_config import SystemConfig


_MODULATIONS = ['OOK', 'OOK_Manchester', 'OFDM', 'BFSK', 'PWM_ASK']


def _separator() -> QFrame:
    line = QFrame()
    line.setFrameShape(QFrame.Shape.VLine)
    line.setFrameShadow(QFrame.Shadow.Plain)
    line.setStyleSheet('color: #E5E5E0; background: #E5E5E0; max-width: 1px;')
    return line


class TopBar(QToolBar):
    """Slim top control bar anchored to MainWindow."""

    preset_changed = pyqtSignal(str)        # preset name
    modulation_changed = pyqtSignal(str)    # modulation name
    wizard_requested = pyqtSignal()
    workbench_requested = pyqtSignal()      # open Workbench window
    load_requested = pyqtSignal()
    save_requested = pyqtSignal()
    run_requested = pyqtSignal()

    def __init__(self, config: SystemConfig, parent=None):
        super().__init__(parent)
        self.setMovable(False)
        self.setFloatable(False)

        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(4, 0, 4, 0)
        layout.setSpacing(14)

        # --- Brand ---
        brand = QLabel('LiFi-PV')
        brand.setStyleSheet(
            'font-size: 14px; font-weight: 700; '
            'letter-spacing: 1.5px; color: #1A1A1A; '
            'text-transform: uppercase; padding-right: 8px;')
        layout.addWidget(brand)

        layout.addWidget(_separator())

        # --- Preset ---
        preset_lbl = QLabel('Preset')
        layout.addWidget(preset_lbl)

        self._preset_combo = QComboBox()
        self._preset_combo.setMinimumWidth(180)
        self._preset_combo.addItem('(Custom)')
        for name in SystemConfig.list_presets():
            self._preset_combo.addItem(name)
        if config.preset_name:
            idx = self._preset_combo.findText(config.preset_name)
            if idx >= 0:
                self._preset_combo.setCurrentIndex(idx)
        self._preset_combo.currentTextChanged.connect(self._on_preset_changed)
        layout.addWidget(self._preset_combo)

        # --- Modulation ---
        mod_lbl = QLabel('Modulation')
        layout.addWidget(mod_lbl)

        self._mod_combo = QComboBox()
        self._mod_combo.setMinimumWidth(150)
        self._mod_combo.addItems(_MODULATIONS)
        current_mod = getattr(config, 'modulation', 'OOK')
        idx = self._mod_combo.findText(current_mod)
        if idx >= 0:
            self._mod_combo.setCurrentIndex(idx)
        self._mod_combo.currentTextChanged.connect(self.modulation_changed.emit)
        layout.addWidget(self._mod_combo)

        # --- Spacer ---
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        layout.addWidget(spacer)

        # --- Secondary actions ---
        self._btn_workbench = QPushButton('Edit System...')
        self._btn_workbench.setToolTip(
            'Open the Workbench to build a custom system\n'
            '(parameters + component library).')
        self._btn_workbench.clicked.connect(self.workbench_requested.emit)
        layout.addWidget(self._btn_workbench)

        self._btn_wizard = QPushButton('Wizard')
        self._btn_wizard.clicked.connect(self.wizard_requested.emit)
        layout.addWidget(self._btn_wizard)

        self._btn_load = QPushButton('Load')
        self._btn_load.clicked.connect(self.load_requested.emit)
        layout.addWidget(self._btn_load)

        self._btn_save = QPushButton('Save')
        self._btn_save.clicked.connect(self.save_requested.emit)
        layout.addWidget(self._btn_save)

        layout.addWidget(_separator())

        # --- Primary action ---
        self._btn_run = QPushButton('Run Simulation')
        self._btn_run.setProperty('role', 'primary')
        self._btn_run.clicked.connect(self.run_requested.emit)
        layout.addWidget(self._btn_run)

        self.addWidget(container)

    # ---- Public API ----

    def set_preset(self, name: str | None):
        """Sync preset selector without re-emitting."""
        self._preset_combo.blockSignals(True)
        target = name or '(Custom)'
        idx = self._preset_combo.findText(target)
        self._preset_combo.setCurrentIndex(idx if idx >= 0 else 0)
        self._preset_combo.blockSignals(False)

    def set_modulation(self, modulation: str):
        self._mod_combo.blockSignals(True)
        idx = self._mod_combo.findText(modulation)
        if idx >= 0:
            self._mod_combo.setCurrentIndex(idx)
        self._mod_combo.blockSignals(False)

    def set_run_enabled(self, enabled: bool):
        self._btn_run.setEnabled(enabled)
        self._btn_run.setText('Running...' if not enabled else 'Run Simulation')

    def set_run_visible(self, visible: bool):
        """Hide the Run button on views where running is irrelevant (e.g. Setup)."""
        self._btn_run.setVisible(visible)

    def _on_preset_changed(self, text: str):
        if text == '(Custom)':
            return
        self.preset_changed.emit(text)
