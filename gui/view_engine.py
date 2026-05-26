# gui/view_engine.py
"""
View 2 — Simulation Engine & Analysis Workspace.

Receives existing SimulationEngineTab and ResultsTab instances from the router
(so signals stay wired centrally in MainWindow). Provides the page chrome:
hero header, vertical split (Engine on top, Results below), and a
"Back to Setup" navigation hook.
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFrame,
    QSplitter, QTabWidget,
)
from PyQt6.QtCore import Qt, pyqtSignal

from gui.theme import COLORS


class EngineView(QWidget):
    """View 2 — Engine + Results host."""

    back_requested = pyqtSignal()       # → router switches to View 1

    def __init__(self, engine_tab: QWidget, results_tab: QWidget, parent=None):
        super().__init__(parent)
        self._engine_tab = engine_tab
        self._results_tab = results_tab
        self._build_ui()

    def _build_ui(self):
        body = QVBoxLayout(self)
        body.setContentsMargins(40, 24, 40, 24)
        body.setSpacing(16)

        # ---- Hero header ----
        hero = QHBoxLayout()
        hero.setSpacing(16)

        nav_block = QVBoxLayout()
        nav_block.setSpacing(2)

        eyebrow = QLabel('STEP 02  ·  EXECUTE')
        eyebrow.setStyleSheet(
            f'color: {COLORS["accent"]}; font-size: 11px; font-weight: 700; '
            'letter-spacing: 2.5px;')
        nav_block.addWidget(eyebrow)

        title = QLabel('Run the engine. Read the result.')
        title.setStyleSheet(
            f'color: {COLORS["text"]}; font-size: 22px; font-weight: 700; '
            'letter-spacing: -0.4px;')
        nav_block.addWidget(title)
        hero.addLayout(nav_block, 1)

        self._btn_back = QPushButton('←  Back to Setup')
        self._btn_back.setMinimumHeight(32)
        self._btn_back.clicked.connect(self.back_requested.emit)
        hero.addWidget(self._btn_back)

        body.addLayout(hero)

        # Hairline
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet(f'color: {COLORS["border"]}; max-height: 1px;')
        body.addWidget(sep)

        # ---- Body: Engine | Results in a tab strip (clean, no nesting) ----
        self._tabs = QTabWidget()
        self._tabs.setDocumentMode(True)
        self._tabs.addTab(self._wrap(self._engine_tab),  'Engine')
        self._tabs.addTab(self._wrap(self._results_tab), 'Results')
        body.addWidget(self._tabs, 1)

    @staticmethod
    def _wrap(widget: QWidget) -> QWidget:
        host = QWidget()
        lay = QVBoxLayout(host)
        lay.setContentsMargins(4, 8, 4, 4)
        lay.setSpacing(0)
        lay.addWidget(widget)
        return host

    # ---- Public API ----

    def show_results(self):
        """Auto-switch to Results tab (e.g. after a run completes)."""
        self._tabs.setCurrentIndex(1)

    def show_engine(self):
        self._tabs.setCurrentIndex(0)
