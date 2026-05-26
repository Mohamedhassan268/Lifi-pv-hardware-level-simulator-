# gui/view_setup.py
"""
View 1 — Welcome & Interactive Configuration.

The landing page. Hosts the channel canvas (always-visible diagram + sliders +
link budget + sweep plot) plus a calm hero header and a single primary action:
"Continue to Simulation".

This view never runs anything. It only shapes the optical link parameters.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFrame,
    QScrollArea, QSizePolicy,
)
from PyQt6.QtCore import Qt, pyqtSignal

from cosim.system_config import SystemConfig
from gui.theme import COLORS
from gui.channel_canvas import ChannelCanvasWidget


class SetupView(QWidget):
    """View 1 — welcoming setup page."""

    continue_requested = pyqtSignal()       # → router switches to View 2
    workbench_requested = pyqtSignal()      # → opens Workbench window

    def __init__(self, config: SystemConfig, parent=None):
        super().__init__(parent)
        self._config = config
        self._build_ui()

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # Scrollable container so the page is comfortable on small windows.
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        outer.addWidget(scroll)

        page = QWidget()
        scroll.setWidget(page)

        body = QVBoxLayout(page)
        body.setContentsMargins(40, 36, 40, 28)
        body.setSpacing(24)

        # ---- Hero header ----
        hero = QHBoxLayout()
        hero.setSpacing(20)

        title_block = QVBoxLayout()
        title_block.setSpacing(6)

        eyebrow = QLabel('STEP 01  ·  CONFIGURE')
        eyebrow.setStyleSheet(
            f'color: {COLORS["accent"]}; font-size: 11px; font-weight: 700; '
            'letter-spacing: 2.5px;')
        title_block.addWidget(eyebrow)

        title = QLabel('Shape the optical link.')
        title.setStyleSheet(
            f'color: {COLORS["text"]}; font-size: 30px; font-weight: 700; '
            'letter-spacing: -0.5px;')
        title_block.addWidget(title)

        subtitle = QLabel(
            'Choose a preset or set your own geometry. The diagram updates as '
            'you tune distance, angles, and area. When the link looks right, '
            'continue to the engine.')
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet(
            f'color: {COLORS["text_dim"]}; font-size: 13px; line-height: 1.5;')
        subtitle.setMaximumWidth(620)
        title_block.addWidget(subtitle)

        hero.addLayout(title_block, 1)

        # Right side of hero: secondary actions stacked
        hero_actions = QVBoxLayout()
        hero_actions.setSpacing(8)
        hero_actions.addStretch()

        self._btn_workbench = QPushButton('Edit Full System...')
        self._btn_workbench.setToolTip(
            'Open the Workbench for advanced control:\n'
            'TX/RX components, modulation, noise, electrical chain.')
        self._btn_workbench.clicked.connect(self.workbench_requested.emit)
        hero_actions.addWidget(self._btn_workbench)

        hero.addLayout(hero_actions, 0)
        body.addLayout(hero)

        # Hairline under hero
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet(f'color: {COLORS["border"]}; max-height: 1px;')
        body.addWidget(sep)

        # ---- Centerpiece: channel canvas (diagram + sliders + budget + sweep) ----
        self._canvas = ChannelCanvasWidget(self._config)
        # The canvas already has its own internal margins; trim the outer
        # because we provide our page-level gutter.
        canvas_host = QFrame()
        canvas_host.setFrameShape(QFrame.Shape.NoFrame)
        canvas_lay = QVBoxLayout(canvas_host)
        canvas_lay.setContentsMargins(0, 0, 0, 0)
        canvas_lay.addWidget(self._canvas)
        canvas_host.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        body.addWidget(canvas_host, 1)

        # ---- Footer: continue CTA ----
        footer_sep = QFrame()
        footer_sep.setFrameShape(QFrame.Shape.HLine)
        footer_sep.setStyleSheet(f'color: {COLORS["border"]}; max-height: 1px;')
        body.addWidget(footer_sep)

        footer = QHBoxLayout()
        footer.setSpacing(12)

        helper = QLabel('Ready to simulate? The engine will use the geometry above.')
        helper.setStyleSheet(f'color: {COLORS["text_dim"]}; font-size: 12px;')
        footer.addWidget(helper)
        footer.addStretch()

        self._btn_continue = QPushButton('Continue to Simulation  →')
        self._btn_continue.setProperty('role', 'primary')
        self._btn_continue.setMinimumHeight(36)
        self._btn_continue.setMinimumWidth(220)
        self._btn_continue.clicked.connect(self.continue_requested.emit)
        footer.addWidget(self._btn_continue)

        body.addLayout(footer)

    # ---- Public API ----

    def update_config(self, config: SystemConfig):
        self._config = config
        self._canvas.update_config(config)
