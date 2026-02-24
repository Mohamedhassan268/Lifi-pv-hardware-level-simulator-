# gui/tab_system_setup.py
"""
Tab 1: System Setup

Left sidebar:  preset dropdown, Load/Save, Quick Info panel
Right area:    3 group boxes (TX, Channel, RX) with component dropdowns
               from COMPONENT_REGISTRY, filtered by type.

Changing any parameter triggers _recalculate() -> updates Quick Info.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QComboBox, QDoubleSpinBox, QSpinBox, QPushButton,
    QFileDialog, QFormLayout, QSplitter, QMessageBox, QCheckBox,
)
from PyQt6.QtCore import Qt, pyqtSignal

from cosim.system_config import SystemConfig
from gui.widgets import QuickInfoPanel


# Map component types to their registry keys
_LED_PARTS = ['LXM5-PD01']
_DRIVER_PARTS = ['ADA4891', 'TLV2379']
_PV_PARTS = ['KXOB25-04X3F', 'SM141K']
_INA_PARTS = ['INA322']
_COMP_PARTS = ['TLV7011']


class SystemSetupTab(QWidget):
    """Tab 1: System configuration with component selection."""

    config_changed = pyqtSignal(object)  # emits SystemConfig

    def __init__(self, config: SystemConfig, parent=None):
        super().__init__(parent)
        self._config = config
        self._building = True  # suppress signals during construction
        self._build_ui()
        self._load_config_to_ui()
        self._building = False

    def _build_ui(self):
        main = QHBoxLayout(self)
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # ---- Left sidebar ----
        left = QWidget()
        left_layout = QVBoxLayout(left)

        # Preset selector
        preset_grp = QGroupBox('Preset')
        preset_lay = QVBoxLayout(preset_grp)
        self._preset_combo = QComboBox()
        self._preset_combo.addItem('(Custom)')
        for name in SystemConfig.list_presets():
            self._preset_combo.addItem(name)
        self._preset_combo.currentTextChanged.connect(self._on_preset_changed)
        preset_lay.addWidget(self._preset_combo)

        btn_row = QHBoxLayout()
        self._btn_load = QPushButton('Load...')
        self._btn_load.clicked.connect(self._load_file)
        self._btn_save = QPushButton('Save...')
        self._btn_save.clicked.connect(self._save_file)
        btn_row.addWidget(self._btn_load)
        btn_row.addWidget(self._btn_save)
        preset_lay.addLayout(btn_row)
        left_layout.addWidget(preset_grp)

        # Quick Info
        self._quick_info = QuickInfoPanel()
        left_layout.addWidget(self._quick_info)
        left_layout.addStretch()

        # ---- Right area ----
        right = QWidget()
        right_layout = QVBoxLayout(right)

        # TX group
        tx_grp = QGroupBox('Transmitter')
        tx_form = QFormLayout(tx_grp)
        self._led_combo = QComboBox()
        self._led_combo.addItems(_LED_PARTS)
        tx_form.addRow('LED:', self._led_combo)

        self._driver_combo = QComboBox()
        self._driver_combo.addItems(_DRIVER_PARTS)
        tx_form.addRow('Driver:', self._driver_combo)

        self._mod_depth = QDoubleSpinBox()
        self._mod_depth.setRange(0.01, 1.0)
        self._mod_depth.setSingleStep(0.05)
        self._mod_depth.setDecimals(2)
        tx_form.addRow('Mod. Depth:', self._mod_depth)

        self._led_power = QDoubleSpinBox()
        self._led_power.setRange(0.1, 100.0)
        self._led_power.setSingleStep(0.5)
        self._led_power.setSuffix(' mW')
        tx_form.addRow('Radiated Power:', self._led_power)

        self._half_angle = QDoubleSpinBox()
        self._half_angle.setRange(1.0, 90.0)
        self._half_angle.setSuffix(' deg')
        tx_form.addRow('Half Angle:', self._half_angle)

        right_layout.addWidget(tx_grp)

        # Channel group
        ch_grp = QGroupBox('Channel')
        ch_form = QFormLayout(ch_grp)

        self._distance = QDoubleSpinBox()
        self._distance.setRange(0.01, 10.0)
        self._distance.setSingleStep(0.05)
        self._distance.setDecimals(3)
        self._distance.setSuffix(' m')
        ch_form.addRow('Distance:', self._distance)

        self._tx_angle = QDoubleSpinBox()
        self._tx_angle.setRange(0.0, 89.0)
        self._tx_angle.setSuffix(' deg')
        ch_form.addRow('TX Angle:', self._tx_angle)

        self._rx_tilt = QDoubleSpinBox()
        self._rx_tilt.setRange(0.0, 89.0)
        self._rx_tilt.setSuffix(' deg')
        ch_form.addRow('RX Tilt:', self._rx_tilt)

        self._lens_t = QDoubleSpinBox()
        self._lens_t.setRange(0.0, 1.0)
        self._lens_t.setSingleStep(0.05)
        self._lens_t.setDecimals(2)
        ch_form.addRow('Lens T:', self._lens_t)

        right_layout.addWidget(ch_grp)

        # RX group
        rx_grp = QGroupBox('Receiver')
        rx_form = QFormLayout(rx_grp)

        self._pv_combo = QComboBox()
        self._pv_combo.addItems(_PV_PARTS)
        rx_form.addRow('PV Cell:', self._pv_combo)

        self._ina_combo = QComboBox()
        self._ina_combo.addItems(_INA_PARTS)
        rx_form.addRow('INA:', self._ina_combo)

        self._comp_combo = QComboBox()
        self._comp_combo.addItems(_COMP_PARTS)
        rx_form.addRow('Comparator:', self._comp_combo)

        self._r_sense = QDoubleSpinBox()
        self._r_sense.setRange(0.01, 1000.0)
        self._r_sense.setSuffix(' Ohm')
        rx_form.addRow('R_sense:', self._r_sense)

        self._bpf_stages = QSpinBox()
        self._bpf_stages.setRange(1, 4)
        rx_form.addRow('BPF Stages:', self._bpf_stages)

        self._dcdc_fsw = QDoubleSpinBox()
        self._dcdc_fsw.setRange(1.0, 1000.0)
        self._dcdc_fsw.setSuffix(' kHz')
        rx_form.addRow('DC-DC fsw:', self._dcdc_fsw)

        right_layout.addWidget(rx_grp)

        # Simulation controls
        sim_grp = QGroupBox('Simulation')
        sim_form = QFormLayout(sim_grp)

        self._data_rate = QDoubleSpinBox()
        self._data_rate.setRange(100, 1e6)
        self._data_rate.setSuffix(' bps')
        self._data_rate.setDecimals(0)
        sim_form.addRow('Data Rate:', self._data_rate)

        self._n_bits = QSpinBox()
        self._n_bits.setRange(10, 100000)
        sim_form.addRow('PRBS Bits:', self._n_bits)

        self._t_stop = QDoubleSpinBox()
        self._t_stop.setRange(1e-6, 1.0)
        self._t_stop.setDecimals(6)
        self._t_stop.setSuffix(' s')
        sim_form.addRow('t_stop:', self._t_stop)

        self._noise_enable = QCheckBox('Enable transient noise sources')
        self._noise_enable.setToolTip(
            'Inject shot noise, thermal noise, and INA input noise\n'
            'into the SPICE simulation for realistic BER')
        sim_form.addRow('Noise:', self._noise_enable)

        right_layout.addWidget(sim_grp)
        right_layout.addStretch()

        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setSizes([250, 550])
        main.addWidget(splitter)

        # Connect all change signals
        for combo in [self._led_combo, self._driver_combo, self._pv_combo,
                       self._ina_combo, self._comp_combo]:
            combo.currentTextChanged.connect(self._on_value_changed)
        for spin in [self._mod_depth, self._led_power, self._half_angle,
                      self._distance, self._tx_angle, self._rx_tilt,
                      self._lens_t, self._r_sense, self._dcdc_fsw,
                      self._data_rate, self._t_stop]:
            spin.valueChanged.connect(self._on_value_changed)
        self._bpf_stages.valueChanged.connect(self._on_value_changed)
        self._n_bits.valueChanged.connect(self._on_value_changed)
        self._noise_enable.stateChanged.connect(self._on_value_changed)

    def _load_config_to_ui(self):
        """Push config values into UI widgets."""
        c = self._config
        self._building = True

        self._set_combo(self._led_combo, c.led_part)
        self._set_combo(self._driver_combo, c.driver_part)
        self._mod_depth.setValue(c.modulation_depth)
        self._led_power.setValue(c.led_radiated_power_mW)
        self._half_angle.setValue(c.led_half_angle_deg)

        self._distance.setValue(c.distance_m)
        self._tx_angle.setValue(c.tx_angle_deg)
        self._rx_tilt.setValue(c.rx_tilt_deg)
        self._lens_t.setValue(c.lens_transmittance)

        self._set_combo(self._pv_combo, c.pv_part)
        self._set_combo(self._ina_combo, c.ina_part)
        self._set_combo(self._comp_combo, c.comparator_part)
        self._r_sense.setValue(c.r_sense_ohm)
        self._bpf_stages.setValue(c.bpf_stages)
        self._dcdc_fsw.setValue(c.dcdc_fsw_kHz)

        self._data_rate.setValue(c.data_rate_bps)
        self._n_bits.setValue(c.n_bits)
        self._t_stop.setValue(c.t_stop_s)
        self._noise_enable.setChecked(c.noise_enable)

        if c.preset_name:
            self._set_combo(self._preset_combo, c.preset_name)

        self._building = False
        self._recalculate()

    def _set_combo(self, combo, text):
        idx = combo.findText(text)
        if idx >= 0:
            combo.setCurrentIndex(idx)

    def _read_ui_to_config(self):
        """Read UI values back into config."""
        c = self._config
        c.led_part = self._led_combo.currentText()
        c.driver_part = self._driver_combo.currentText()
        c.modulation_depth = self._mod_depth.value()
        c.led_radiated_power_mW = self._led_power.value()
        c.led_half_angle_deg = self._half_angle.value()

        c.distance_m = self._distance.value()
        c.tx_angle_deg = self._tx_angle.value()
        c.rx_tilt_deg = self._rx_tilt.value()
        c.lens_transmittance = self._lens_t.value()

        c.pv_part = self._pv_combo.currentText()
        c.ina_part = self._ina_combo.currentText()
        c.comparator_part = self._comp_combo.currentText()
        c.r_sense_ohm = self._r_sense.value()
        c.bpf_stages = self._bpf_stages.value()
        c.dcdc_fsw_kHz = self._dcdc_fsw.value()

        c.data_rate_bps = self._data_rate.value()
        c.n_bits = self._n_bits.value()
        c.t_stop_s = self._t_stop.value()
        c.noise_enable = self._noise_enable.isChecked()

    def _on_value_changed(self, *args):
        if self._building:
            return
        self._read_ui_to_config()
        self._recalculate()
        self.config_changed.emit(self._config)

    def _recalculate(self):
        self._quick_info.update_from_config(self._config)

    def _on_preset_changed(self, name):
        if self._building or name == '(Custom)':
            return
        try:
            self._config = SystemConfig.from_preset(name)
            self._load_config_to_ui()
            self.config_changed.emit(self._config)
        except Exception as e:
            QMessageBox.warning(self, 'Preset Error', str(e))

    def _load_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, 'Load Config', '', 'JSON (*.json)')
        if path:
            try:
                self._config = SystemConfig.load(path)
                self._load_config_to_ui()
                self.config_changed.emit(self._config)
            except Exception as e:
                QMessageBox.warning(self, 'Load Error', str(e))

    def _save_file(self):
        path, _ = QFileDialog.getSaveFileName(
            self, 'Save Config', 'config.json', 'JSON (*.json)')
        if path:
            self._read_ui_to_config()
            self._config.save(path)

    def get_config(self) -> SystemConfig:
        self._read_ui_to_config()
        return self._config

    def set_config(self, config: SystemConfig):
        self._config = config
        self._load_config_to_ui()
