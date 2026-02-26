# gui/tab_component_library.py
"""
Tab 2: Component Library

Left:  filter combo (All/Solar Cells/LEDs/Amplifiers/...) + searchable list
Right: parameter table + circuit symbol drawing
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QComboBox, QListWidget, QTableWidget, QTableWidgetItem,
    QSplitter, QHeaderView,
)
from PyQt6.QtCore import Qt

from components import COMPONENT_REGISTRY, get_component
from gui.theme import COLORS
from gui.widgets import MplCanvas


# Category mapping
_CATEGORIES = {
    'All':          list(set(COMPONENT_REGISTRY.keys())),
    'Solar Cells':  ['KXOB25-04X3F', 'SM141K'],
    'Photodiodes':  ['BPW34', 'SFH206K', 'VEMD5510'],
    'LEDs':         ['LXM5-PD01'],
    'Amplifiers':   ['INA322', 'TLV2379', 'ADA4891'],
    'Comparators':  ['TLV7011'],
    'MOSFETs':      ['BSD235N', 'NTS4409'],
}

# Map component names to their symbol category
_COMP_CATEGORY = {}
for _cat, _names in _CATEGORIES.items():
    if _cat == 'All':
        continue
    for _n in _names:
        _COMP_CATEGORY[_n] = _cat


class ComponentLibraryTab(QWidget):
    """Tab 2: Browse and inspect hardware components."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()
        self._populate_list()

    def _build_ui(self):
        main = QHBoxLayout(self)
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # ---- Left panel: filter + list ----
        left = QWidget()
        left_layout = QVBoxLayout(left)

        self._filter_combo = QComboBox()
        self._filter_combo.addItems(_CATEGORIES.keys())
        self._filter_combo.currentTextChanged.connect(self._on_filter_changed)
        left_layout.addWidget(QLabel('Filter:'))
        left_layout.addWidget(self._filter_combo)

        self._comp_list = QListWidget()
        self._comp_list.currentTextChanged.connect(self._on_component_selected)
        left_layout.addWidget(self._comp_list)

        # ---- Right panel: details + symbol ----
        right = QWidget()
        right_layout = QVBoxLayout(right)

        self._name_label = QLabel('Select a component')
        self._name_label.setStyleSheet('font-size: 14px; font-weight: bold;')
        right_layout.addWidget(self._name_label)

        self._param_table = QTableWidget()
        self._param_table.setColumnCount(2)
        self._param_table.setHorizontalHeaderLabels(['Parameter', 'Value'])
        self._param_table.horizontalHeader().setStretchLastSection(True)
        self._param_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.ResizeToContents)
        self._param_table.setEditTriggers(
            QTableWidget.EditTrigger.NoEditTriggers)
        right_layout.addWidget(self._param_table)

        self._canvas = MplCanvas(width=5, height=3)
        right_layout.addWidget(self._canvas)

        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setSizes([200, 600])
        main.addWidget(splitter)

    def _populate_list(self, category='All'):
        self._comp_list.clear()
        names = _CATEGORIES.get(category, [])
        for name in sorted(set(names)):
            self._comp_list.addItem(name)

    def _on_filter_changed(self, category):
        self._populate_list(category)

    def _on_component_selected(self, name):
        if not name:
            return

        try:
            comp = get_component(name)
        except KeyError:
            self._name_label.setText(f'{name} (not found)')
            return

        self._name_label.setText(f'{comp.name}')

        # Fill parameter table
        params = comp.get_parameters()
        self._param_table.setRowCount(len(params))
        for i, (k, v) in enumerate(params.items()):
            self._param_table.setItem(i, 0, QTableWidgetItem(str(k)))
            self._param_table.setItem(i, 1, QTableWidgetItem(str(v)))

        # Draw circuit symbol
        self._canvas.clear()
        ax = self._canvas.ax
        ax.set_xlim(-2, 2)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect('equal')
        ax.axis('off')

        category = _COMP_CATEGORY.get(name, 'Unknown')

        if category == 'Solar Cells':
            self._draw_solar_cell(ax, comp.name)
        elif category == 'Photodiodes':
            self._draw_photodiode(ax, comp.name)
        elif category == 'LEDs':
            self._draw_led(ax, comp.name)
        elif category == 'Amplifiers':
            self._draw_amplifier(ax, comp.name)
        elif category == 'Comparators':
            self._draw_comparator(ax, comp.name)
        elif category == 'MOSFETs':
            self._draw_mosfet(ax, comp.name)
        else:
            ax.text(0, 0, comp.name, ha='center', va='center',
                    fontsize=14, fontweight='bold', color=COLORS['text'],
                    bbox=dict(boxstyle='round,pad=0.5',
                              fc=COLORS['surface'], ec=COLORS['accent']))

        self._canvas.draw()

    # -----------------------------------------------------------------
    # Circuit symbol drawing methods
    # -----------------------------------------------------------------

    def _draw_solar_cell(self, ax, name):
        """Draw a solar cell / PV cell symbol."""
        lw = 2.0
        clr = '#2E5090'

        # Diode triangle
        tri_x = [-0.4, 0.4, 0.0, -0.4]
        tri_y = [-0.4, -0.4, 0.4, -0.4]
        ax.fill(tri_x, tri_y, fc='#D0E0F0', ec=clr, lw=lw)

        # Cathode bar
        ax.plot([-0.5, 0.5], [0.4, 0.4], color=clr, lw=lw)

        # Leads
        ax.plot([0, 0], [0.4, 1.0], color=clr, lw=lw)
        ax.plot([0, 0], [-0.4, -1.0], color=clr, lw=lw)

        # Incoming light arrows (indicating photovoltaic)
        for dy in [0.15, -0.15]:
            ax.annotate('', xy=(-0.3, dy), xytext=(-1.1, dy + 0.4),
                        arrowprops=dict(arrowstyle='->', color='#D4A017',
                                        lw=1.5))

        # Circle around diode (cell enclosure)
        circle = plt_Circle((0, 0), 0.7, fill=False, ec=clr, lw=1.2,
                             linestyle='--')
        ax.add_patch(circle)

        # Terminal labels
        ax.text(0.15, 1.05, '+', fontsize=12, fontweight='bold', color=clr)
        ax.text(0.15, -1.15, '-', fontsize=12, fontweight='bold', color=clr)

        # Component name
        ax.text(0, -1.4, name, ha='center', va='top', fontsize=11,
                fontweight='bold', color=COLORS['text'])

    def _draw_photodiode(self, ax, name):
        """Draw a photodiode symbol."""
        lw = 2.0
        clr = '#2E5090'

        # Diode triangle (pointing up)
        tri_x = [-0.4, 0.4, 0.0, -0.4]
        tri_y = [-0.4, -0.4, 0.4, -0.4]
        ax.fill(tri_x, tri_y, fc='#D0E0F0', ec=clr, lw=lw)

        # Cathode bar
        ax.plot([-0.5, 0.5], [0.4, 0.4], color=clr, lw=lw)

        # Leads
        ax.plot([0, 0], [0.4, 1.0], color=clr, lw=lw)
        ax.plot([0, 0], [-0.4, -1.0], color=clr, lw=lw)

        # Incoming light arrows
        for dy in [0.15, -0.15]:
            ax.annotate('', xy=(-0.3, dy), xytext=(-1.1, dy + 0.4),
                        arrowprops=dict(arrowstyle='->', color='#D4A017',
                                        lw=1.5))

        # Terminal labels
        ax.text(0.15, 1.05, 'K', fontsize=10, fontweight='bold', color=clr)
        ax.text(0.15, -1.15, 'A', fontsize=10, fontweight='bold', color=clr)

        ax.text(0, -1.4, name, ha='center', va='top', fontsize=11,
                fontweight='bold', color=COLORS['text'])

    def _draw_led(self, ax, name):
        """Draw an LED symbol."""
        lw = 2.0
        clr = '#CC3333'

        # Diode triangle (pointing up)
        tri_x = [-0.4, 0.4, 0.0, -0.4]
        tri_y = [-0.4, -0.4, 0.4, -0.4]
        ax.fill(tri_x, tri_y, fc='#FFD0D0', ec=clr, lw=lw)

        # Cathode bar
        ax.plot([-0.5, 0.5], [0.4, 0.4], color=clr, lw=lw)

        # Leads
        ax.plot([0, 0], [0.4, 1.0], color=clr, lw=lw)
        ax.plot([0, 0], [-0.4, -1.0], color=clr, lw=lw)

        # Outgoing light arrows (emitting)
        for dy in [0.15, -0.15]:
            ax.annotate('', xy=(1.1, dy + 0.4), xytext=(0.3, dy),
                        arrowprops=dict(arrowstyle='->', color='#D4A017',
                                        lw=1.5))

        # Terminal labels
        ax.text(0.15, 1.05, 'A', fontsize=10, fontweight='bold', color=clr)
        ax.text(0.15, -1.15, 'K', fontsize=10, fontweight='bold', color=clr)

        ax.text(0, -1.4, name, ha='center', va='top', fontsize=11,
                fontweight='bold', color=COLORS['text'])

    def _draw_amplifier(self, ax, name):
        """Draw an op-amp / amplifier triangle symbol."""
        lw = 2.0
        clr = '#2E7D32'

        # Triangle body
        tri_x = [-0.7, -0.7, 0.8, -0.7]
        tri_y = [-0.7, 0.7, 0.0, -0.7]
        ax.fill(tri_x, tri_y, fc='#E0F0E0', ec=clr, lw=lw)

        # Input leads
        ax.plot([-1.4, -0.7], [0.35, 0.35], color=clr, lw=lw)   # +
        ax.plot([-1.4, -0.7], [-0.35, -0.35], color=clr, lw=lw)  # -

        # Output lead
        ax.plot([0.8, 1.5], [0.0, 0.0], color=clr, lw=lw)

        # +/- labels inside triangle
        ax.text(-0.5, 0.35, '+', fontsize=12, fontweight='bold',
                ha='center', va='center', color=clr)
        ax.text(-0.5, -0.35, '-', fontsize=12, fontweight='bold',
                ha='center', va='center', color=clr)

        # Power rails (V+ and V-)
        ax.plot([-0.1, -0.1], [0.45, 0.85], color=COLORS['text_dim'], lw=1.2,
                linestyle='--')
        ax.plot([-0.1, -0.1], [-0.45, -0.85], color=COLORS['text_dim'], lw=1.2,
                linestyle='--')
        ax.text(-0.1, 0.95, 'V+', fontsize=9, ha='center', color=COLORS['text_dim'])
        ax.text(-0.1, -0.95, 'V-', fontsize=9, ha='center', color=COLORS['text_dim'])

        # Terminal labels
        ax.text(-1.5, 0.35, 'IN+', fontsize=9, ha='right', va='center',
                color=clr)
        ax.text(-1.5, -0.35, 'IN-', fontsize=9, ha='right', va='center',
                color=clr)
        ax.text(1.6, 0.0, 'OUT', fontsize=9, ha='left', va='center',
                color=clr)

        ax.text(0, -1.3, name, ha='center', va='top', fontsize=11,
                fontweight='bold', color=COLORS['text'])

    def _draw_comparator(self, ax, name):
        """Draw a comparator symbol (similar to op-amp but with digital output)."""
        lw = 2.0
        clr = '#6A1B9A'

        # Triangle body
        tri_x = [-0.7, -0.7, 0.8, -0.7]
        tri_y = [-0.7, 0.7, 0.0, -0.7]
        ax.fill(tri_x, tri_y, fc='#F0E0F8', ec=clr, lw=lw)

        # Input leads
        ax.plot([-1.4, -0.7], [0.35, 0.35], color=clr, lw=lw)
        ax.plot([-1.4, -0.7], [-0.35, -0.35], color=clr, lw=lw)

        # Output lead
        ax.plot([0.8, 1.5], [0.0, 0.0], color=clr, lw=lw)

        # +/- labels
        ax.text(-0.5, 0.35, '+', fontsize=12, fontweight='bold',
                ha='center', va='center', color=clr)
        ax.text(-0.5, -0.35, '-', fontsize=12, fontweight='bold',
                ha='center', va='center', color=clr)

        # Digital output indicator (small square wave)
        sq_x = [1.0, 1.0, 1.2, 1.2, 1.4, 1.4]
        sq_y = [-0.15, 0.15, 0.15, -0.15, -0.15, 0.15]
        ax.plot(sq_x, sq_y, color=clr, lw=1.2)

        # Terminal labels
        ax.text(-1.5, 0.35, 'IN+', fontsize=9, ha='right', va='center',
                color=clr)
        ax.text(-1.5, -0.35, 'IN-', fontsize=9, ha='right', va='center',
                color=clr)
        ax.text(1.6, 0.0, 'OUT', fontsize=9, ha='left', va='center',
                color=clr)

        ax.text(0, -1.3, name, ha='center', va='top', fontsize=11,
                fontweight='bold', color=COLORS['text'])

    def _draw_mosfet(self, ax, name):
        """Draw an N-channel MOSFET symbol."""
        lw = 2.0
        clr = '#E65100'

        # Vertical channel line
        ax.plot([0, 0], [-0.6, 0.6], color=clr, lw=lw + 0.5)

        # Gate line (left of channel)
        ax.plot([-0.5, -0.5], [-0.4, 0.4], color=clr, lw=lw)

        # Gate lead
        ax.plot([-1.3, -0.5], [0, 0], color=clr, lw=lw)

        # Insulating gap between gate and channel
        ax.plot([-0.25, -0.25], [-0.4, 0.4], color=clr, lw=1.0,
                linestyle=':')

        # Source (bottom) - horizontal line + lead down
        ax.plot([0, 0.6], [-0.4, -0.4], color=clr, lw=lw)
        ax.plot([0.6, 0.6], [-0.4, -1.0], color=clr, lw=lw)

        # Drain (top) - horizontal line + lead up
        ax.plot([0, 0.6], [0.4, 0.4], color=clr, lw=lw)
        ax.plot([0.6, 0.6], [0.4, 1.0], color=clr, lw=lw)

        # Body connection (center, with arrow indicating N-channel)
        ax.plot([0, 0.6], [0.0, 0.0], color=clr, lw=lw)
        ax.annotate('', xy=(0.45, 0.0), xytext=(0.1, 0.0),
                    arrowprops=dict(arrowstyle='->', color=clr, lw=1.5))

        # Terminal labels
        ax.text(-1.4, 0.0, 'G', fontsize=10, fontweight='bold',
                ha='right', va='center', color=clr)
        ax.text(0.75, 1.05, 'D', fontsize=10, fontweight='bold',
                ha='left', va='center', color=clr)
        ax.text(0.75, -1.05, 'S', fontsize=10, fontweight='bold',
                ha='left', va='center', color=clr)

        # Body diode (small)
        bd_x = [0.45, 0.75, 0.60, 0.45]
        bd_y = [-0.15, -0.15, 0.05, -0.15]
        ax.fill(bd_x, bd_y, fc=clr, ec=clr, lw=0.8, alpha=0.4)

        ax.text(0, -1.4, name, ha='center', va='top', fontsize=11,
                fontweight='bold', color=COLORS['text'])


# Matplotlib Circle patch helper
from matplotlib.patches import Circle as plt_Circle
