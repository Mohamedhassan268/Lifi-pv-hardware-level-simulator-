# gui/workbench_window.py
"""
Workbench — secondary, non-modal window for building a custom system.

Hosts the full System Setup form and the Components library side-by-side,
so the main Simulator window stays focused on Channel + Run + Results.

Lifecycle:
    - Opened from the top bar (Edit System...).
    - Stays open across runs (non-modal); user can leave it floating.
    - Emits `config_committed(SystemConfig)` when the user clicks "Apply".
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QSplitter, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFrame, QApplication, QScrollArea,
)
from PyQt6.QtCore import Qt, pyqtSignal

from cosim.system_config import SystemConfig
from gui.theme import COLORS, STYLESHEET
from gui.tab_system_setup import SystemSetupTab
from gui.tab_component_library import ComponentLibraryTab


class WorkbenchWindow(QMainWindow):
    """Non-modal workbench: System Setup + Components."""

    config_committed = pyqtSignal(object)   # emits SystemConfig
    wizard_requested = pyqtSignal()         # forwarded from inner SetupTab

    def __init__(self, config: SystemConfig, parent=None):
        # Pass an explicit Window flag so the OS treats this as a top-level
        # window with full chrome (minimize/maximize/close). When a QMainWindow
        # has a parent on Windows, Qt sometimes strips the minimize button —
        # setting Qt.WindowType.Window restores standard window decorations.
        super().__init__(parent, Qt.WindowType.Window)
        self.setWindowTitle('LiFi-PV — Workbench')
        self.setMinimumSize(1080, 640)
        self.resize(1320, 820)
        self.setStyleSheet(STYLESHEET)

        self._build_ui(config)
        self._center_on_screen()

    def _build_ui(self, config: SystemConfig):
        root = QWidget()
        root_lay = QVBoxLayout(root)
        root_lay.setContentsMargins(0, 0, 0, 0)
        root_lay.setSpacing(0)

        # --- Header strip ---
        header = QWidget()
        header_lay = QHBoxLayout(header)
        header_lay.setContentsMargins(20, 14, 20, 14)
        header_lay.setSpacing(10)

        title = QLabel('Workbench')
        title.setStyleSheet(
            'font-size: 14px; font-weight: 700; '
            'letter-spacing: 1.5px; text-transform: uppercase; '
            f'color: {COLORS["text"]};')
        header_lay.addWidget(title)

        subtitle = QLabel('Build a custom system')
        subtitle.setStyleSheet(
            f'color: {COLORS["text_dim"]}; font-size: 12px; padding-left: 6px;')
        header_lay.addWidget(subtitle)
        header_lay.addStretch()

        self._btn_revert = QPushButton('Revert')
        self._btn_revert.clicked.connect(self._on_revert)
        header_lay.addWidget(self._btn_revert)

        self._btn_apply = QPushButton('Apply && Close')
        self._btn_apply.setProperty('role', 'primary')
        self._btn_apply.clicked.connect(self._on_apply)
        header_lay.addWidget(self._btn_apply)

        root_lay.addWidget(header)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet(f'color: {COLORS["border"]}; max-height: 1px;')
        root_lay.addWidget(sep)

        # --- Body: Setup (left) | Components (right) ---
        self._setup_tab = SystemSetupTab(config)
        self._components_tab = ComponentLibraryTab()

        # Forward wizard requests upward
        self._setup_tab.wizard_requested.connect(self.wizard_requested.emit)

        body_split = QSplitter(Qt.Orientation.Horizontal)
        body_split.setHandleWidth(1)
        body_split.addWidget(self._wrap(self._setup_tab))
        body_split.addWidget(self._wrap(self._components_tab))
        body_split.setStretchFactor(0, 1)
        body_split.setStretchFactor(1, 1)
        body_split.setSizes([640, 640])

        root_lay.addWidget(body_split, 1)
        self.setCentralWidget(root)

    @staticmethod
    def _wrap(widget: QWidget) -> QWidget:
        """Wrap content in padding + a vertical QScrollArea so tall forms
        stay reachable when the Workbench window is small."""
        inner = QWidget()
        inner_lay = QVBoxLayout(inner)
        inner_lay.setContentsMargins(16, 14, 16, 14)
        inner_lay.setSpacing(0)
        inner_lay.addWidget(widget)

        scroll = QScrollArea()
        scroll.setWidget(inner)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        return scroll

    # ---- Public API ----

    def set_config(self, config: SystemConfig):
        """Sync the inner setup form to a new config without re-emitting."""
        self._setup_tab.set_config(config)

    # ---- Slots ----

    def _on_apply(self):
        try:
            cfg = self._setup_tab.get_config()
        except Exception as e:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(self, 'Invalid Config', str(e))
            return
        self.config_committed.emit(cfg)
        self.close()

    def _on_revert(self):
        # Re-pull whatever the inner tab currently considers "self._config"
        cfg = self._setup_tab.get_config()
        self._setup_tab.set_config(cfg)

    def _center_on_screen(self):
        screen = QApplication.primaryScreen()
        if screen:
            geo = screen.availableGeometry()
            x = (geo.width() - self.width()) // 2 + geo.x()
            y = (geo.height() - self.height()) // 2 + geo.y()
            self.move(x, y)
