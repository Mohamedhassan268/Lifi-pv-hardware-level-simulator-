# gui/main_window.py
"""
MainWindow — single-shell LiFi-PV Simulator GUI.

Layout (no popups):

    +-----------------------------------------------------------+
    |  TopBar  [LiFi-PV] Preset Modulation     [Wizard][Run]    |
    +----------+--------------------------------+---------------+
    |          |  Inline channel canvas         |  Inspector    |
    | Left:    |  (TX -> channel -> RX)         |  accordion:   |
    | Compo-   |                                |   Schematics  |
    | nents    |--------------------------------|   Validation  |
    |          |  Bottom dock tabs:             |   Message     |
    |          |  [ Setup | Engine | Results ]  |               |
    +----------+--------------------------------+---------------+
    |  status bar: SPICE / preset / session                     |
    +-----------------------------------------------------------+
"""

import sys, os, logging
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'workspace')
os.makedirs(_log_dir, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[logging.FileHandler(os.path.join(_log_dir, 'simulator.log'), encoding='utf-8')],
)
logger = logging.getLogger(__name__)

from PyQt6.QtWidgets import (
    QMainWindow, QStackedWidget, QToolBox, QDockWidget,
    QFileDialog, QMessageBox, QApplication, QLabel, QWidget, QVBoxLayout,
)
from PyQt6.QtCore import Qt, QSettings, QLocale
from PyQt6.QtGui import QAction

QLocale.setDefault(QLocale(QLocale.Language.English, QLocale.Country.UnitedStates))

from cosim.system_config import SystemConfig
from cosim.ltspice_runner import LTSpiceRunner
from cosim.spice_finder import find_ngspice

from gui.theme import COLORS, STYLESHEET
from gui.top_bar import TopBar
from gui.view_setup import SetupView
from gui.view_engine import EngineView
from gui.tab_simulation_engine import SimulationEngineTab
from gui.tab_results import ResultsTab
from gui.tab_schematics import SchematicsTab
from gui.tab_validation import ValidationTab
from gui.tab_message import MessageTab
from gui.workbench_window import WorkbenchWindow

VIEW_SETUP = 0
VIEW_ENGINE = 1


class MainWindow(QMainWindow):
    """Single-shell main window — no popups, no separate tab pages."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle('LiFi-PV Simulator')
        self.setMinimumSize(1280, 820)
        self.resize(1480, 920)

        self._settings = QSettings('LiFi-PV-Simulator', 'HardwareFaithful')
        self._config = SystemConfig.from_preset('kadirvelu2021')

        saved_ltspice = self._settings.value('ltspice_path', '')
        if saved_ltspice and os.path.isfile(saved_ltspice):
            self._ltspice = LTSpiceRunner(saved_ltspice)
        else:
            self._ltspice = LTSpiceRunner()

        self.setStyleSheet(STYLESHEET)

        self._build_menu()
        self._build_top_bar()
        self._build_shell()
        self._view_menu.addAction(self._toggle_inspector_act)
        self._view_menu.addAction(self._top_bar.toggleViewAction())
        self._build_status_bar()
        self._connect_signals()

        # Initial UI state matches the starting view (Setup).
        self._top_bar.set_run_visible(False)

        self._center_on_screen()

        logger.info("MainWindow initialized (LTspice=%s, ngspice=%s)",
                     'OK' if self._ltspice.available else 'N/A',
                     'OK' if find_ngspice() else 'N/A')

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _build_menu(self):
        menubar = self.menuBar()

        file_menu = menubar.addMenu('File')
        for label, slot, shortcut in [
            ('Load Config...', self._load_config, 'Ctrl+O'),
            ('Save Config...', self._save_config, 'Ctrl+S'),
        ]:
            act = QAction(label, self)
            act.setShortcut(shortcut)
            act.triggered.connect(slot)
            file_menu.addAction(act)

        file_menu.addSeparator()
        quit_act = QAction('Quit', self)
        quit_act.setShortcut('Ctrl+Q')
        quit_act.triggered.connect(self.close)
        file_menu.addAction(quit_act)

        presets_menu = menubar.addMenu('Presets')
        for name in SystemConfig.list_presets():
            act = QAction(name, self)
            act.triggered.connect(lambda checked, n=name: self._load_preset(n))
            presets_menu.addAction(act)

        tools_menu = menubar.addMenu('Tools')
        ltspice_act = QAction('Set LTspice Path...', self)
        ltspice_act.triggered.connect(self._set_ltspice_path)
        tools_menu.addAction(ltspice_act)
        tools_menu.addSeparator()
        paper_reader_act = QAction('PaperLens...', self)
        paper_reader_act.setShortcut('Ctrl+Shift+P')
        paper_reader_act.triggered.connect(self._show_paper_reader)
        tools_menu.addAction(paper_reader_act)
        tools_menu.addSeparator()
        kicad_act = QAction('Export KiCad Schematic...', self)
        kicad_act.setShortcut('Ctrl+Shift+K')
        kicad_act.triggered.connect(self._export_kicad)
        tools_menu.addAction(kicad_act)

        view_menu = menubar.addMenu('View')
        # Inspector toggle is added after _build_shell() runs
        self._view_menu = view_menu

        help_menu = menubar.addMenu('Help')
        wizard_act = QAction('Quick Start Guide...', self)
        wizard_act.setShortcut('F1')
        wizard_act.triggered.connect(self._show_wizard)
        help_menu.addAction(wizard_act)
        help_menu.addSeparator()
        about_act = QAction('About', self)
        about_act.triggered.connect(self._show_about)
        help_menu.addAction(about_act)

    def _build_top_bar(self):
        self._top_bar = TopBar(self._config, parent=self)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, self._top_bar)

    def _build_shell(self):
        # Leaf widgets shared across views (Setup + Components live in Workbench)
        self._tab_simulation = SimulationEngineTab(self._config, ltspice_runner=self._ltspice)
        self._tab_results = ResultsTab(config=self._config)
        self._tab_schematics = SchematicsTab()
        self._tab_validation = ValidationTab(self._config)
        self._tab_message = MessageTab(self._config)
        self._workbench = None  # lazy: created on first Edit System... click

        # ---- View 1: Setup (welcoming landing) ----
        self._view_setup = SetupView(self._config)

        # ---- View 2: Engine + Results ----
        self._view_engine = EngineView(self._tab_simulation, self._tab_results)

        # ---- Router: stacked widget swaps the two views ----
        self._router = QStackedWidget()
        self._router.addWidget(self._view_setup)    # index 0 = VIEW_SETUP
        self._router.addWidget(self._view_engine)   # index 1 = VIEW_ENGINE
        self._router.setCurrentIndex(VIEW_SETUP)
        self.setCentralWidget(self._router)

        # ---- Inspector: hidden-by-default dock with accordion ----
        self._inspector = QToolBox()
        self._inspector.addItem(self._wrap(self._tab_schematics), 'Schematics')
        self._inspector.addItem(self._wrap(self._tab_validation), 'Validation')
        self._inspector.addItem(self._wrap(self._tab_message),    'Message TX/RX')

        self._inspector_dock = QDockWidget('Inspector', self)
        self._inspector_dock.setWidget(self._inspector)
        self._inspector_dock.setAllowedAreas(
            Qt.DockWidgetArea.RightDockWidgetArea | Qt.DockWidgetArea.LeftDockWidgetArea)
        self._inspector_dock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable |
            QDockWidget.DockWidgetFeature.DockWidgetFloatable |
            QDockWidget.DockWidgetFeature.DockWidgetClosable)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._inspector_dock)
        self._inspector_dock.hide()

        # Toggle action for the View menu / status bar
        self._toggle_inspector_act = self._inspector_dock.toggleViewAction()
        self._toggle_inspector_act.setText('Inspector')
        self._toggle_inspector_act.setShortcut('Ctrl+I')

    @staticmethod
    def _wrap(widget: QWidget) -> QWidget:
        """Pad a hosted widget with consistent gutters."""
        host = QWidget()
        lay = QVBoxLayout(host)
        lay.setContentsMargins(16, 14, 16, 14)
        lay.setSpacing(0)
        lay.addWidget(widget)
        return host

    def _build_status_bar(self):
        sb = self.statusBar()
        sb.setSizeGripEnabled(False)

        ltspice_ok = self._ltspice.available
        ngspice_ok = find_ngspice() is not None

        if ltspice_ok and ngspice_ok:
            spice_msg = 'LTspice OK  ·  ngspice OK'
            spice_color = COLORS['success']
        elif ltspice_ok or ngspice_ok:
            parts = []
            parts.append('LTspice OK' if ltspice_ok else 'LTspice —')
            parts.append('ngspice OK' if ngspice_ok else 'ngspice —')
            spice_msg = '  ·  '.join(parts)
            spice_color = COLORS['success']
        else:
            spice_msg = 'No SPICE engine'
            spice_color = COLORS['warning']

        self._spice_status_label = QLabel(spice_msg)
        self._spice_status_label.setStyleSheet(
            f'color: {spice_color}; padding: 0 14px; font-size: 11px;')
        sb.addPermanentWidget(self._spice_status_label)

        self._config_status_label = QLabel(
            f'Config — {self._config.preset_name or "Custom"}')
        self._config_status_label.setStyleSheet(
            f'color: {COLORS["text_dim"]}; padding: 0 14px; font-size: 11px;')
        sb.addPermanentWidget(self._config_status_label)

    def _connect_signals(self):
        # Top bar
        self._top_bar.preset_changed.connect(self._load_preset)
        self._top_bar.modulation_changed.connect(self._on_modulation_changed)
        self._top_bar.wizard_requested.connect(self._show_wizard)
        self._top_bar.workbench_requested.connect(self._show_workbench)
        self._top_bar.load_requested.connect(self._load_config)
        self._top_bar.save_requested.connect(self._save_config)
        self._top_bar.run_requested.connect(self._on_run_clicked)

        # View navigation
        self._view_setup.continue_requested.connect(self._goto_engine)
        self._view_setup.workbench_requested.connect(self._show_workbench)
        self._view_engine.back_requested.connect(self._goto_setup)
        self._router.currentChanged.connect(self._on_view_changed)

        # Simulation results fan out to Results and Validation
        self._tab_simulation.simulation_done.connect(self._tab_results.update_results)
        self._tab_simulation.simulation_done.connect(self._tab_validation.update_results)
        self._tab_simulation.simulation_done.connect(self._on_simulation_done)

    def _center_on_screen(self):
        screen = QApplication.primaryScreen()
        if screen:
            geo = screen.availableGeometry()
            x = (geo.width() - self.width()) // 2 + geo.x()
            y = (geo.height() - self.height()) // 2 + geo.y()
            self.move(x, y)
        self.raise_()
        self.activateWindow()

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_config_changed(self, config):
        """Single fan-out for any config update (preset, workbench, file, paper)."""
        self._config = config
        self._view_setup.update_config(config)
        self._tab_simulation.update_config(config)
        self._tab_results.update_config(config)
        self._tab_validation.update_config(config)
        self._tab_message.update_config(config)
        if self._workbench is not None and self._workbench.isVisible():
            self._workbench.set_config(config)
        self._top_bar.set_preset(config.preset_name)
        self._top_bar.set_modulation(getattr(config, 'modulation', 'OOK'))
        self._config_status_label.setText(
            f'Config — {config.preset_name or "Custom"}')
        logger.info("Config changed to: %s", config.preset_name or 'Custom')

    # ---- View navigation ----

    def _goto_setup(self):
        self._router.setCurrentIndex(VIEW_SETUP)

    def _goto_engine(self):
        self._router.setCurrentIndex(VIEW_ENGINE)

    def _on_view_changed(self, index: int):
        # Run button only makes sense when the engine is visible.
        in_engine = (index == VIEW_ENGINE)
        self._top_bar.set_run_visible(in_engine)

    def _show_workbench(self):
        """Open (or raise) the Workbench window for editing System + Components."""
        if self._workbench is None:
            self._workbench = WorkbenchWindow(self._config, parent=self)
            self._workbench.config_committed.connect(self._on_config_changed)
            self._workbench.wizard_requested.connect(self._show_wizard)
        else:
            self._workbench.set_config(self._config)
        self._workbench.show()
        self._workbench.raise_()
        self._workbench.activateWindow()

    def _on_modulation_changed(self, modulation: str):
        self._config.modulation = modulation
        self._tab_simulation.update_config(self._config)
        logger.info("Modulation changed to: %s", modulation)

    def _on_run_clicked(self):
        self._top_bar.set_run_enabled(False)
        if self._router.currentIndex() != VIEW_ENGINE:
            self._goto_engine()
        self._view_engine.show_engine()
        try:
            self._tab_simulation.run_all()
        except Exception as e:
            logger.error("Run failed to start: %s", e)
            self._top_bar.set_run_enabled(True)
            QMessageBox.warning(self, 'Run Error', str(e))

    def _on_simulation_done(self, _results):
        self._top_bar.set_run_enabled(True)
        self._view_engine.show_results()

    def _load_config(self):
        path, _ = QFileDialog.getOpenFileName(
            self, 'Load Configuration', '', 'JSON (*.json)')
        if path:
            try:
                cfg = SystemConfig.load(path)
                self._on_config_changed(cfg)
                self.statusBar().showMessage(f'Loaded: {path}', 3000)
                logger.info("Config loaded from: %s", path)
            except Exception as e:
                logger.warning("Failed to load config from %s: %s", path, e)
                QMessageBox.warning(self, 'Error', f'Failed to load: {e}')

    def _save_config(self):
        path, _ = QFileDialog.getSaveFileName(
            self, 'Save Configuration', 'config.json', 'JSON (*.json)')
        if path:
            try:
                self._config.save(path)
                self.statusBar().showMessage(f'Saved: {path}', 3000)
                logger.info("Config saved to: %s", path)
            except Exception as e:
                logger.warning("Failed to save config to %s: %s", path, e)
                QMessageBox.warning(self, 'Error', f'Failed to save: {e}')

    def _load_preset(self, name):
        try:
            cfg = SystemConfig.from_preset(name)
            self._on_config_changed(cfg)
            self.statusBar().showMessage(f'Loaded preset: {name}', 3000)
            logger.info("Preset loaded: %s", name)
        except Exception as e:
            logger.warning("Failed to load preset '%s': %s", name, e)
            QMessageBox.warning(self, 'Preset Error', str(e))

    def _set_ltspice_path(self):
        path, _ = QFileDialog.getOpenFileName(
            self, 'Select LTspice Executable', '',
            'Executable (*.exe);;All files (*)')
        if path:
            self._ltspice = LTSpiceRunner(path)
            if self._ltspice.available:
                self._settings.setValue('ltspice_path', path)
                self._tab_simulation.set_ltspice_runner(self._ltspice)
                self._spice_status_label.setText('LTspice OK')
                self._spice_status_label.setStyleSheet(
                    f'color: {COLORS["success"]}; padding: 0 14px; font-size: 11px;')
                self.statusBar().showMessage(f'LTspice set: {path}', 3000)
                logger.info("LTspice path set: %s", path)
            else:
                logger.warning("LTspice not found at user-selected path: %s", path)
                QMessageBox.warning(self, 'Error', f'LTspice not found at: {path}')

    def _show_wizard(self):
        from gui.wizard import QuickStartWizard
        wizard = QuickStartWizard(self)
        if wizard.exec() == QuickStartWizard.DialogCode.Accepted:
            config = wizard.get_config()
            if config:
                self._on_config_changed(config)
                self.statusBar().showMessage(
                    f'Loaded from Quick Start: {config.preset_name or "Custom"}', 3000)
                logger.info("Wizard applied preset: %s", config.preset_name or 'Custom')

    def _show_paper_reader(self):
        from gui.dialog_paper_reader import PaperReaderDialog
        dialog = PaperReaderDialog(self)
        dialog.config_extracted.connect(self._on_paper_extracted)
        dialog.exec()

    def _export_kicad(self):
        """Export KiCad schematic + BOM for the active preset."""
        from pathlib import Path
        from kicad import export_kicad, available_presets

        preset = self._config.preset_name
        if not preset:
            QMessageBox.warning(
                self, 'KiCad Export',
                'KiCad export requires a named preset.\n'
                'Load a preset first (Presets menu or top-bar selector), '
                'then try again.')
            return

        supported = available_presets()
        if preset not in supported:
            QMessageBox.warning(
                self, 'KiCad Export',
                f'No KiCad graph builder is registered for "{preset}".\n\n'
                f'Currently supported: {", ".join(supported) or "(none)"}.')
            return

        default_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'workspace', 'kicad')
        os.makedirs(default_dir, exist_ok=True)

        out_dir = QFileDialog.getExistingDirectory(
            self, 'Choose KiCad output directory', default_dir)
        if not out_dir:
            return

        try:
            result = export_kicad(preset, Path(out_dir))
        except Exception as e:
            logger.error("KiCad export failed: %s", e)
            QMessageBox.warning(self, 'KiCad Export Failed', str(e))
            return

        msg = (
            f'Exported "{preset}".\n\n'
            f'Schematic: {result.schematic_path}\n'
            f'BOM:        {result.bom_path}\n\n'
            f'Components: {result.component_count}    Nets: {result.net_count}'
        )
        if result.warnings:
            msg += '\n\nValidator warnings:\n  · ' + '\n  · '.join(result.warnings)

        QMessageBox.information(self, 'KiCad Export', msg)
        self.statusBar().showMessage(
            f'KiCad exported: {result.schematic_path.name}', 5000)
        logger.info("KiCad exported preset=%s to %s (warnings=%d)",
                    preset, out_dir, len(result.warnings))

    def _on_paper_extracted(self, params: dict):
        try:
            valid_fields = {f.name for f in self._config.__dataclass_fields__.values()}
            filtered = {k: v for k, v in params.items() if k in valid_fields}
            cfg = SystemConfig(**filtered)
            self._on_config_changed(cfg)
            self.statusBar().showMessage(
                f'PaperLens: loaded {len(filtered)} parameters', 5000)
            logger.info("Paper reader loaded %d parameters (preset=%s)",
                         len(filtered), cfg.preset_name or 'Custom')
        except Exception as e:
            logger.error("Failed to apply AI-extracted params: %s", e)
            QMessageBox.warning(self, 'Error',
                f'Failed to apply extracted parameters:\n{e}')

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
