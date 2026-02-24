# gui/tab_component_library.py
"""
Tab 2: Component Library

Left:  filter combo (All/Solar Cells/LEDs/Amplifiers/...) + searchable list
Right: parameter table + matplotlib plot area
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QComboBox, QListWidget, QTableWidget, QTableWidgetItem,
    QSplitter, QHeaderView,
)
from PyQt6.QtCore import Qt

from components import COMPONENT_REGISTRY, get_component
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

        # ---- Right panel: details + plot ----
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

        # Plot spectral response if available
        self._canvas.clear()
        ax = self._canvas.ax
        try:
            if hasattr(comp, 'spectral_response'):
                wl, resp = comp.spectral_response()
                ax.plot(wl, resp, 'b-')
                ax.set_xlabel('Wavelength (nm)')
                ax.set_ylabel('Responsivity (A/W)')
                ax.set_title(f'{comp.name} Spectral Response')
                ax.grid(True, alpha=0.3)
            elif hasattr(comp, 'emission_spectrum'):
                wl, power = comp.emission_spectrum()
                ax.plot(wl, power, 'r-')
                ax.set_xlabel('Wavelength (nm)')
                ax.set_ylabel('Relative Power')
                ax.set_title(f'{comp.name} Emission Spectrum')
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, 'No spectral data available',
                        ha='center', va='center', transform=ax.transAxes,
                        fontsize=12, color='gray')
        except Exception:
            ax.text(0.5, 0.5, 'Plot not available',
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=12, color='gray')

        self._canvas.draw()
