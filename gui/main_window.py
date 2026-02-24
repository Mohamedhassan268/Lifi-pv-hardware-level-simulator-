# gui/main_window.py
"""
MainWindow — 7-tab LiFi-PV Simulator GUI

Menu bar:
    File:    Load/Save Config, Export Session, Quit
    Presets: Load preset from presets/ directory
    Tools:   Set LTspice path
    Help:    About

Status bar: LTspice status, active session path
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt6.QtWidgets import (
    QMainWindow, QTabWidget, QMenuBar, QMenu, QStatusBar,
    QFileDialog, QMessageBox, QApplication, QLabel,
)
from PyQt6.QtCore import Qt, QSettings
from PyQt6.QtGui import QAction

from cosim.system_config import SystemConfig
from cosim.ltspice_runner import LTSpiceRunner

from gui.tab_system_setup import SystemSetupTab
from gui.tab_component_library import ComponentLibraryTab
from gui.tab_channel_config import ChannelConfigTab
from gui.tab_simulation_engine import SimulationEngineTab
from gui.tab_results import ResultsTab
from gui.tab_schematics import SchematicsTab
from gui.tab_validation import ValidationTab


class MainWindow(QMainWindow):
    """Main application window with 7 tabs."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle('LiFi-PV Simulator')
        self.setMinimumSize(1000, 700)
        self.resize(1200, 800)

        # Persistent settings
        self._settings = QSettings('LiFi-PV-Simulator', 'HardwareFaithful')

        # Shared state
        self._config = SystemConfig.from_preset('kadirvelu2021')

        # Restore persisted LTspice path
        saved_ltspice = self._settings.value('ltspice_path', '')
        if saved_ltspice and os.path.isfile(saved_ltspice):
            self._ltspice = LTSpiceRunner(saved_ltspice)
        else:
            self._ltspice = LTSpiceRunner()

        self._build_menu()
        self._build_tabs()
        self._build_status_bar()
        self._connect_signals()

        # Center on screen and ensure window comes to foreground
        self._center_on_screen()

    def _center_on_screen(self):
        """Center the window on the primary screen and raise to foreground."""
        screen = QApplication.primaryScreen()
        if screen:
            geo = screen.availableGeometry()
            x = (geo.width() - self.width()) // 2 + geo.x()
            y = (geo.height() - self.height()) // 2 + geo.y()
            self.move(x, y)
        self.raise_()
        self.activateWindow()

    def _build_menu(self):
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu('File')

        load_act = QAction('Load Config...', self)
        load_act.setShortcut('Ctrl+O')
        load_act.triggered.connect(self._load_config)
        file_menu.addAction(load_act)

        save_act = QAction('Save Config...', self)
        save_act.setShortcut('Ctrl+S')
        save_act.triggered.connect(self._save_config)
        file_menu.addAction(save_act)

        file_menu.addSeparator()

        quit_act = QAction('Quit', self)
        quit_act.setShortcut('Ctrl+Q')
        quit_act.triggered.connect(self.close)
        file_menu.addAction(quit_act)

        # Presets menu
        presets_menu = menubar.addMenu('Presets')
        for name in SystemConfig.list_presets():
            act = QAction(name, self)
            act.triggered.connect(lambda checked, n=name: self._load_preset(n))
            presets_menu.addAction(act)

        # Tools menu
        tools_menu = menubar.addMenu('Tools')

        ltspice_act = QAction('Set LTspice Path...', self)
        ltspice_act.triggered.connect(self._set_ltspice_path)
        tools_menu.addAction(ltspice_act)

        # Help menu
        help_menu = menubar.addMenu('Help')
        about_act = QAction('About', self)
        about_act.triggered.connect(self._show_about)
        help_menu.addAction(about_act)

    def _build_tabs(self):
        self._tabs = QTabWidget()
        self.setCentralWidget(self._tabs)

        self._tab_setup = SystemSetupTab(self._config)
        self._tab_components = ComponentLibraryTab()
        self._tab_channel = ChannelConfigTab(self._config)
        self._tab_simulation = SimulationEngineTab(self._config, ltspice_runner=self._ltspice)
        self._tab_results = ResultsTab(config=self._config)
        self._tab_schematics = SchematicsTab()
        self._tab_validation = ValidationTab(self._config)

        self._tabs.addTab(self._tab_setup, '1. System Setup')
        self._tabs.addTab(self._tab_components, '2. Components')
        self._tabs.addTab(self._tab_channel, '3. Channel')
        self._tabs.addTab(self._tab_simulation, '4. Simulation')
        self._tabs.addTab(self._tab_results, '5. Results')
        self._tabs.addTab(self._tab_schematics, '6. Schematics')
        self._tabs.addTab(self._tab_validation, '7. Validation')

    def _build_status_bar(self):
        sb = self.statusBar()

        # Permanent LTspice status widget
        ltspice_msg = (f'LTspice: OK'
                       if self._ltspice.available
                       else 'LTspice: Not found (Tools > Set LTspice Path)')
        ltspice_style = ('color: green;' if self._ltspice.available
                         else 'color: orange;')
        self._ltspice_status_label = QLabel(ltspice_msg)
        self._ltspice_status_label.setStyleSheet(ltspice_style + ' padding: 0 10px;')
        sb.addPermanentWidget(self._ltspice_status_label)

        # Config info
        self._config_status_label = QLabel(
            f'Config: {self._config.preset_name or "Custom"}')
        self._config_status_label.setStyleSheet('padding: 0 10px;')
        sb.addPermanentWidget(self._config_status_label)

    def _connect_signals(self):
        # Config changes from Setup tab propagate to other tabs
        self._tab_setup.config_changed.connect(self._on_config_changed)

        # Simulation results propagate to Results and Validation
        self._tab_simulation.simulation_done.connect(
            self._tab_results.update_results)
        self._tab_simulation.simulation_done.connect(
            self._tab_validation.update_results)

    def _on_config_changed(self, config):
        self._config = config
        self._tab_channel.update_config(config)
        self._tab_simulation.update_config(config)
        self._tab_results.update_config(config)
        self._tab_validation.update_config(config)
        self._config_status_label.setText(
            f'Config: {config.preset_name or "Custom"}')

    def _load_config(self):
        path, _ = QFileDialog.getOpenFileName(
            self, 'Load Configuration', '', 'JSON (*.json)')
        if path:
            try:
                self._config = SystemConfig.load(path)
                self._tab_setup.set_config(self._config)
                self.statusBar().showMessage(f'Loaded: {path}', 3000)
            except Exception as e:
                QMessageBox.warning(self, 'Error', f'Failed to load: {e}')

    def _save_config(self):
        path, _ = QFileDialog.getSaveFileName(
            self, 'Save Configuration', 'config.json', 'JSON (*.json)')
        if path:
            try:
                config = self._tab_setup.get_config()
                config.save(path)
                self.statusBar().showMessage(f'Saved: {path}', 3000)
            except Exception as e:
                QMessageBox.warning(self, 'Error', f'Failed to save: {e}')

    def _load_preset(self, name):
        try:
            self._config = SystemConfig.from_preset(name)
            self._tab_setup.set_config(self._config)
            self.statusBar().showMessage(f'Loaded preset: {name}', 3000)
        except Exception as e:
            QMessageBox.warning(self, 'Preset Error', str(e))

    def _set_ltspice_path(self):
        path, _ = QFileDialog.getOpenFileName(
            self, 'Select LTspice Executable', '',
            'Executable (*.exe);;All files (*)')
        if path:
            self._ltspice = LTSpiceRunner(path)
            if self._ltspice.available:
                # Persist path for future sessions
                self._settings.setValue('ltspice_path', path)
                self._tab_simulation.set_ltspice_runner(self._ltspice)
                self._ltspice_status_label.setText('LTspice: OK')
                self._ltspice_status_label.setStyleSheet('color: green; padding: 0 10px;')
                self.statusBar().showMessage(
                    f'LTspice set: {path} (saved)', 3000)
            else:
                QMessageBox.warning(self, 'Error',
                                     f'LTspice not found at: {path}')

    def _show_about(self):
        QMessageBox.about(
            self, 'About LiFi-PV Simulator',
            'Hardware-Faithful LiFi-PV Simulator\n\n'
            'Paper-agnostic simulation platform for\n'
            'simultaneous light communication and\n'
            'energy harvesting via photovoltaic cells.\n\n'
            'Architecture: TX -> Channel -> RX\n'
            'Engines: LTspice / ngspice\n\n'
            'Built with PyQt6 + matplotlib')
