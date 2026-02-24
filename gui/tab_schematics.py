# gui/tab_schematics.py
"""
Tab 6: Schematics

Dropdown to select which schematic, Generate + Export buttons.
Renders schemdraw output as image in a QLabel via pixmap.
"""

import sys, os, io, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QPushButton, QFileDialog, QScrollArea, QMessageBox,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap, QImage
from pathlib import Path


_SCHEMATIC_CHOICES = [
    'Full System',
    'Solar Cell + Sense Resistor',
    'INA322 Amplifier',
    'Band-Pass Filter',
    'Comparator (TLV7011)',
    'DC-DC Boost Converter',
    'LED Driver (TX)',
    'Signal Flow Diagram',
]


class SchematicsTab(QWidget):
    """Tab 6: Generate and display circuit schematics."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._current_pixmap = None
        self._build_ui()

    def _build_ui(self):
        main = QVBoxLayout(self)

        # Controls
        ctrl = QHBoxLayout()
        ctrl.addWidget(QLabel('Schematic:'))
        self._schematic_combo = QComboBox()
        self._schematic_combo.addItems(_SCHEMATIC_CHOICES)
        ctrl.addWidget(self._schematic_combo)

        self._btn_generate = QPushButton('Generate')
        self._btn_generate.clicked.connect(self._generate)
        ctrl.addWidget(self._btn_generate)

        self._btn_export = QPushButton('Export...')
        self._btn_export.clicked.connect(self._export)
        self._btn_export.setEnabled(False)
        ctrl.addWidget(self._btn_export)
        ctrl.addStretch()
        main.addLayout(ctrl)

        # Image display
        self._scroll = QScrollArea()
        self._image_label = QLabel('Click "Generate" to create a schematic')
        self._image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._image_label.setStyleSheet('font-size: 14px; color: #999;')
        self._scroll.setWidget(self._image_label)
        self._scroll.setWidgetResizable(True)
        main.addWidget(self._scroll)

    def _generate(self):
        choice = self._schematic_combo.currentText()
        try:
            from systems.kadirvelu2021_schematic import (
                draw_full_system, draw_solar_cell_circuit,
                draw_ina322_circuit, draw_bandpass_filter,
                draw_comparator_circuit, draw_dcdc_converter,
                draw_led_driver, draw_signal_flow,
            )
        except ImportError as e:
            QMessageBox.warning(self, 'Import Error',
                                f'schemdraw not available: {e}')
            return

        func_map = {
            'Full System': draw_full_system,
            'Solar Cell + Sense Resistor': draw_solar_cell_circuit,
            'INA322 Amplifier': draw_ina322_circuit,
            'Band-Pass Filter': draw_bandpass_filter,
            'Comparator (TLV7011)': draw_comparator_circuit,
            'DC-DC Boost Converter': draw_dcdc_converter,
            'LED Driver (TX)': draw_led_driver,
            'Signal Flow Diagram': draw_signal_flow,
        }

        func = func_map.get(choice)
        if not func:
            return

        try:
            # Generate to temp PNG
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                tmp_path = tmp.name

            drawing = func()
            drawing.save(tmp_path, dpi=150)

            pixmap = QPixmap(tmp_path)
            self._current_pixmap = pixmap
            self._image_label.setPixmap(pixmap)
            self._image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._btn_export.setEnabled(True)

            # Clean up temp file
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

        except Exception as e:
            QMessageBox.warning(self, 'Schematic Error', str(e))

    def _export(self):
        if not self._current_pixmap:
            return

        path, _ = QFileDialog.getSaveFileName(
            self, 'Export Schematic', 'schematic.png',
            'PNG (*.png);;SVG (*.svg);;PDF (*.pdf)')

        if path:
            try:
                if path.endswith('.png'):
                    self._current_pixmap.save(path, 'PNG')
                else:
                    # Re-generate in requested format
                    choice = self._schematic_combo.currentText()
                    from systems.kadirvelu2021_schematic import (
                        draw_full_system, draw_solar_cell_circuit,
                        draw_ina322_circuit, draw_bandpass_filter,
                        draw_comparator_circuit, draw_dcdc_converter,
                        draw_led_driver, draw_signal_flow,
                    )
                    func_map = {
                        'Full System': draw_full_system,
                        'Solar Cell + Sense Resistor': draw_solar_cell_circuit,
                        'INA322 Amplifier': draw_ina322_circuit,
                        'Band-Pass Filter': draw_bandpass_filter,
                        'Comparator (TLV7011)': draw_comparator_circuit,
                        'DC-DC Boost Converter': draw_dcdc_converter,
                        'LED Driver (TX)': draw_led_driver,
                        'Signal Flow Diagram': draw_signal_flow,
                    }
                    func = func_map.get(choice)
                    if func:
                        drawing = func()
                        drawing.save(path)
            except Exception as e:
                QMessageBox.warning(self, 'Export Error', str(e))
