# gui/tab_schematics.py
"""
Tab 6: Schematics

Paper selector + schematic dropdown. Supports all 7 papers.
Renders schemdraw output as image in a QLabel via pixmap.
"""

import sys, os, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QPushButton, QFileDialog, QScrollArea, QMessageBox,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap

from gui.theme import COLORS


class SchematicsTab(QWidget):
    """Tab 6: Generate and display circuit schematics for any paper."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._current_pixmap = None
        self._current_draw_func = None
        self._paper_info = {}
        self._build_ui()
        self._load_paper_registry()

    def _build_ui(self):
        main = QVBoxLayout(self)

        # Paper selector row
        paper_row = QHBoxLayout()
        paper_row.addWidget(QLabel('Paper:'))
        self._paper_combo = QComboBox()
        self._paper_combo.setMinimumWidth(300)
        self._paper_combo.currentIndexChanged.connect(self._on_paper_changed)
        paper_row.addWidget(self._paper_combo)
        paper_row.addStretch()
        main.addLayout(paper_row)

        # Paper description
        self._paper_desc = QLabel('')
        self._paper_desc.setStyleSheet(
            f'color: {COLORS["text_dim"]}; font-size: 11px; padding: 2px 4px;')
        self._paper_desc.setWordWrap(True)
        main.addWidget(self._paper_desc)

        # Schematic selector + buttons row
        ctrl = QHBoxLayout()
        ctrl.addWidget(QLabel('Schematic:'))
        self._schematic_combo = QComboBox()
        self._schematic_combo.setMinimumWidth(250)
        ctrl.addWidget(self._schematic_combo)

        self._btn_generate = QPushButton('Generate')
        self._btn_generate.clicked.connect(self._generate)
        ctrl.addWidget(self._btn_generate)

        self._btn_export = QPushButton('Export...')
        self._btn_export.clicked.connect(self._export)
        self._btn_export.setEnabled(False)
        ctrl.addWidget(self._btn_export)

        self._btn_kicad = QPushButton('Export KiCad\u2026')
        self._btn_kicad.setToolTip(
            'Export this paper as a KiCad schematic (.kicad_sch) + BOM (CSV)')
        self._btn_kicad.clicked.connect(self._export_kicad)
        ctrl.addWidget(self._btn_kicad)

        ctrl.addStretch()
        main.addLayout(ctrl)

        # Image display
        self._scroll = QScrollArea()
        self._image_label = QLabel('Select a paper and schematic, then click "Generate"')
        self._image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._image_label.setStyleSheet(
            f'font-size: 14px; color: {COLORS["text_dim"]};')
        self._scroll.setWidget(self._image_label)
        self._scroll.setWidgetResizable(True)
        main.addWidget(self._scroll)

    def _load_paper_registry(self):
        """Load the paper schematic registry."""
        try:
            from systems.paper_schematics import get_paper_info
            self._paper_info = get_paper_info()
        except Exception as e:
            self._paper_info = {}
            print(f"Warning: Could not load paper schematics registry: {e}")

        if not self._paper_info:
            self._paper_combo.addItem('(no papers found)')
            return

        # Populate paper combo
        for key, info in self._paper_info.items():
            self._paper_combo.addItem(info['label'], key)

        # Trigger initial population
        self._on_paper_changed(0)

    def _on_paper_changed(self, index):
        """Update schematic choices when paper selection changes."""
        self._schematic_combo.clear()

        paper_key = self._paper_combo.currentData()
        if not paper_key or paper_key not in self._paper_info:
            return

        info = self._paper_info[paper_key]
        for name, _func in info['schematics']:
            self._schematic_combo.addItem(name)

        # Description hints
        _descs = {
            'kadirvelu2021': 'OOK @ 5 kbps, 32.5 cm — Full analog RX chain with INA322 + BPF + DC-DC',
            'sarwar2017': 'OFDM 16-QAM @ 15 Mbps, 2.0 m — High-speed, minimal analog, direct demod',
            'gonzalez2024': 'OOK Manchester @ 4.8 kbps, 0.6 m — Minimal design: amp + notch + comparator',
            'oliveira2024': 'OFDM 64-QAM @ 25.7 Mbps, 0.5 m — Red laser + 3×3 photodiode array',
            'xu2024': 'BFSK @ 400 bps, 5.0 m — Sunlight modulation via LC shutter',
            'correa2025': 'PWM-ASK @ 10 kbps, 0.85 m — 30 W LED panel for greenhouse dual-use',
        }
        self._paper_desc.setText(_descs.get(paper_key, ''))

    def _get_current_func(self):
        """Get the drawing function for the current selection."""
        paper_key = self._paper_combo.currentData()
        if not paper_key or paper_key not in self._paper_info:
            return None

        schematic_idx = self._schematic_combo.currentIndex()
        schematics = self._paper_info[paper_key]['schematics']
        if schematic_idx < 0 or schematic_idx >= len(schematics):
            return None

        _name, func = schematics[schematic_idx]
        return func

    def _generate(self):
        func = self._get_current_func()
        if not func:
            return

        try:
            # Generate to temp PNG
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                tmp_path = tmp.name

            # Some functions take a preset arg for the generic channel
            paper_key = self._paper_combo.currentData()
            schematic_name = self._schematic_combo.currentText()

            if schematic_name == 'Channel Model' and paper_key != 'kadirvelu2021':
                drawing = func(preset=paper_key)
            else:
                drawing = func()

            if drawing is None:
                QMessageBox.warning(self, 'Schematic Error',
                                    'schemdraw not available.\nInstall: pip install schemdraw')
                return

            # Apply Proteus-style rendering (grid + professional colors)
            try:
                from systems.schematic_style import save_styled
                save_styled(drawing, tmp_path, dpi=150)
            except ImportError:
                drawing.save(tmp_path, dpi=150)

            pixmap = QPixmap(tmp_path)
            self._current_pixmap = pixmap
            self._current_draw_func = func
            self._image_label.setPixmap(pixmap)
            self._image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._btn_export.setEnabled(True)

            try:
                os.unlink(tmp_path)
            except Exception:
                pass

        except ImportError as e:
            QMessageBox.warning(self, 'Import Error',
                                f'schemdraw not available: {e}')
        except Exception as e:
            QMessageBox.warning(self, 'Schematic Error', str(e))

    def _export_kicad(self):
        """Export current paper as .kicad_sch + BOM."""
        paper_key = self._paper_combo.currentData()
        if not paper_key:
            return

        try:
            from kicad import export_kicad
            from kicad.kicad_exporter import available_presets
        except ImportError as exc:
            QMessageBox.warning(self, 'KiCad Export', f'KiCad module not available: {exc}')
            return

        if paper_key not in available_presets():
            QMessageBox.information(
                self, 'KiCad Export',
                f"No KiCad graph builder for '{paper_key}' yet.\n"
                f"Available: {', '.join(available_presets()) or '(none)'}")
            return

        out_dir = QFileDialog.getExistingDirectory(
            self, 'Choose output folder for KiCad files')
        if not out_dir:
            return

        try:
            from pathlib import Path
            result = export_kicad(paper_key, Path(out_dir))
        except Exception as exc:
            QMessageBox.warning(self, 'KiCad Export Failed', str(exc))
            return

        msg = (
            f"Wrote:\n"
            f"  {result.schematic_path}\n"
            f"  {result.bom_path}\n\n"
            f"{result.component_count} components, {result.net_count} nets."
        )
        if result.warnings:
            msg += "\n\nWarnings:\n  " + "\n  ".join(result.warnings[:10])
            if len(result.warnings) > 10:
                msg += f"\n  ... and {len(result.warnings) - 10} more"
        QMessageBox.information(self, 'KiCad Export Complete', msg)

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
                    # Re-generate in requested format (SVG/PDF)
                    func = self._get_current_func()
                    if func:
                        paper_key = self._paper_combo.currentData()
                        schematic_name = self._schematic_combo.currentText()
                        if schematic_name == 'Channel Model' and paper_key != 'kadirvelu2021':
                            drawing = func(preset=paper_key)
                        else:
                            drawing = func()
                        if drawing:
                            try:
                                from systems.schematic_style import save_styled
                                save_styled(drawing, path)
                            except ImportError:
                                drawing.save(path)
                    else:
                        self._current_pixmap.save(path, 'PNG')
            except Exception as e:
                QMessageBox.warning(self, 'Export Error', str(e))
