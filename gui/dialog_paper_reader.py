# gui/dialog_paper_reader.py
"""
AI Paper Reader Dialog — Extract LiFi-PV parameters from PDF.

Supports two backends:
    - Ollama (local, free, no API key) — recommended
    - Google Gemini API (cloud, needs API key)

Features:
    - Backend selector (Ollama / Gemini)
    - PDF file picker
    - Progress bar with step messages
    - Results table with color-coded confidence
    - Validation report
    - Save as preset / Load into simulator buttons
"""

import os
import logging

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QLabel, QLineEdit, QFileDialog,
    QProgressBar, QTableWidget, QTableWidgetItem,
    QTextEdit, QGroupBox, QMessageBox, QHeaderView,
    QSplitter, QWidget, QComboBox, QStackedWidget,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSettings
from PyQt6.QtGui import QColor

from gui.theme import COLORS

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Worker thread — runs pipeline off the UI thread
# ─────────────────────────────────────────────────────────────────────────────

class _ExtractionWorker(QThread):
    """Background thread for PDF extraction pipeline."""

    progress = pyqtSignal(int, str)     # step_number, message
    finished = pyqtSignal(dict)         # result dict
    error = pyqtSignal(str)             # error message

    def __init__(self, pdf_path: str, backend: str,
                 api_key: str = '', ollama_model: str = 'qwen2.5:3b'):
        super().__init__()
        self.pdf_path = pdf_path
        self.backend = backend
        self.api_key = api_key
        self.ollama_model = ollama_model

    def run(self):
        try:
            from ai.paper_pipeline import PaperReaderPipeline
            pipeline = PaperReaderPipeline(
                backend=self.backend,
                api_key=self.api_key,
                ollama_model=self.ollama_model,
            )
            pipeline.set_progress_callback(
                lambda step, msg: self.progress.emit(step, msg))
            result = pipeline.run(self.pdf_path)
            self.finished.emit(result)
        except Exception as e:
            logger.error("Extraction failed: %s", e, exc_info=True)
            self.error.emit(str(e))


# ─────────────────────────────────────────────────────────────────────────────
# Main Dialog
# ─────────────────────────────────────────────────────────────────────────────

class PaperReaderDialog(QDialog):
    """Dialog for AI-powered paper parameter extraction."""

    # Emitted when user clicks "Load into Simulator" with cleaned params
    config_extracted = pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('AI Paper Reader')
        self.setMinimumSize(800, 600)
        self.resize(950, 700)

        self._settings = QSettings('LiFi-PV-Simulator', 'HardwareFaithful')
        self._result = None
        self._worker = None

        self._build_ui()
        self._restore_settings()
        self._on_backend_changed()  # Update visibility

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # ── Title ──
        title = QLabel('AI Paper Reader')
        title.setStyleSheet(
            f'font-size: 18px; font-weight: bold; color: {COLORS["accent"]};')
        subtitle = QLabel(
            'Extract LiFi-PV parameters from a research paper PDF')
        subtitle.setStyleSheet(f'color: {COLORS["text_dim"]}; margin-bottom: 8px;')
        layout.addWidget(title)
        layout.addWidget(subtitle)

        # ── Input Section ──
        input_group = QGroupBox('Input')
        input_layout = QGridLayout(input_group)

        # Backend selector row
        input_layout.addWidget(QLabel('Backend:'), 0, 0)
        self._backend_combo = QComboBox()
        self._backend_combo.addItem('Ollama (Local - Free)', 'ollama')
        self._backend_combo.addItem('Google Gemini (Cloud - API Key)', 'gemini')
        self._backend_combo.currentIndexChanged.connect(self._on_backend_changed)
        input_layout.addWidget(self._backend_combo, 0, 1)

        # Ollama model selector (shown when Ollama selected)
        self._ollama_status = QLabel('')
        input_layout.addWidget(self._ollama_status, 0, 2)

        # Ollama model row
        self._ollama_label = QLabel('Model:')
        input_layout.addWidget(self._ollama_label, 1, 0)
        self._ollama_model_combo = QComboBox()
        self._ollama_model_combo.setEditable(True)
        self._ollama_model_combo.addItem('qwen2.5:3b')
        self._ollama_model_combo.addItem('qwen2.5:7b')
        self._ollama_model_combo.addItem('qwen2.5:14b')
        self._ollama_model_combo.addItem('llama3.1:8b')
        self._ollama_model_combo.addItem('mistral:7b')
        input_layout.addWidget(self._ollama_model_combo, 1, 1)
        self._refresh_models_btn = QPushButton('Refresh')
        self._refresh_models_btn.clicked.connect(self._refresh_ollama_models)
        input_layout.addWidget(self._refresh_models_btn, 1, 2)

        # Gemini API key row (shown when Gemini selected)
        self._key_label = QLabel('API Key:')
        input_layout.addWidget(self._key_label, 2, 0)
        self._key_edit = QLineEdit()
        self._key_edit.setPlaceholderText('Paste your Google AI API key...')
        self._key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        input_layout.addWidget(self._key_edit, 2, 1)
        self._show_key_btn = QPushButton('Show')
        self._show_key_btn.setCheckable(True)
        self._show_key_btn.toggled.connect(self._toggle_key_visibility)
        input_layout.addWidget(self._show_key_btn, 2, 2)

        # PDF file row
        input_layout.addWidget(QLabel('PDF File:'), 3, 0)
        self._pdf_edit = QLineEdit()
        self._pdf_edit.setPlaceholderText('Select a research paper PDF...')
        self._pdf_edit.setReadOnly(True)
        input_layout.addWidget(self._pdf_edit, 3, 1)
        self._browse_btn = QPushButton('Browse...')
        self._browse_btn.clicked.connect(self._browse_pdf)
        input_layout.addWidget(self._browse_btn, 3, 2)

        # Run button
        self._run_btn = QPushButton('Extract Parameters')
        self._run_btn.setStyleSheet(
            f'background-color: {COLORS["accent"]}; color: {COLORS["bg"]}; '
            f'font-size: 14px; padding: 10px;')
        self._run_btn.clicked.connect(self._run_extraction)
        input_layout.addWidget(self._run_btn, 4, 0, 1, 3)

        layout.addWidget(input_group)

        # ── Progress ──
        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 5)
        self._progress_bar.setValue(0)
        self._progress_bar.setTextVisible(True)
        self._progress_bar.setFormat('Ready')
        layout.addWidget(self._progress_bar)

        self._status_label = QLabel('')
        self._status_label.setStyleSheet(f'color: {COLORS["text_dim"]};')
        layout.addWidget(self._status_label)

        # ── Results (splitter: table left, report right) ──
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Parameter table
        table_widget = QWidget()
        table_layout = QVBoxLayout(table_widget)
        table_layout.setContentsMargins(0, 0, 0, 0)
        table_label = QLabel('Extracted Parameters')
        table_label.setStyleSheet(
            f'font-weight: bold; color: {COLORS["accent"]}; font-size: 13px;')
        table_layout.addWidget(table_label)

        self._table = QTableWidget()
        self._table.setColumnCount(4)
        self._table.setHorizontalHeaderLabels(
            ['Parameter', 'Value', 'Confidence', 'Status'])
        self._table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch)
        self._table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.ResizeToContents)
        self._table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.ResizeMode.ResizeToContents)
        self._table.horizontalHeader().setSectionResizeMode(
            3, QHeaderView.ResizeMode.ResizeToContents)
        self._table.setAlternatingRowColors(True)
        table_layout.addWidget(self._table)
        splitter.addWidget(table_widget)

        # Validation report
        report_widget = QWidget()
        report_layout = QVBoxLayout(report_widget)
        report_layout.setContentsMargins(0, 0, 0, 0)
        report_label = QLabel('Validation Report')
        report_label.setStyleSheet(
            f'font-weight: bold; color: {COLORS["accent"]}; font-size: 13px;')
        report_layout.addWidget(report_label)

        self._report_text = QTextEdit()
        self._report_text.setReadOnly(True)
        report_layout.addWidget(self._report_text)
        splitter.addWidget(report_widget)

        splitter.setSizes([550, 350])
        layout.addWidget(splitter, stretch=1)

        # ── Action Buttons ──
        btn_layout = QHBoxLayout()

        self._save_btn = QPushButton('Save as Preset')
        self._save_btn.setEnabled(False)
        self._save_btn.clicked.connect(self._save_preset)
        btn_layout.addWidget(self._save_btn)

        self._load_btn = QPushButton('Load into Simulator')
        self._load_btn.setEnabled(False)
        self._load_btn.setStyleSheet(
            f'background-color: {COLORS["success"]}; color: white; font-weight: bold;')
        self._load_btn.clicked.connect(self._load_into_simulator)
        btn_layout.addWidget(self._load_btn)

        btn_layout.addStretch()

        self._close_btn = QPushButton('Close')
        self._close_btn.clicked.connect(self.close)
        btn_layout.addWidget(self._close_btn)

        layout.addLayout(btn_layout)

    # ─── Backend switching ────────────────────────────────────────────────

    def _on_backend_changed(self):
        """Show/hide fields based on selected backend."""
        is_ollama = self._backend_combo.currentData() == 'ollama'

        # Ollama fields
        self._ollama_label.setVisible(is_ollama)
        self._ollama_model_combo.setVisible(is_ollama)
        self._refresh_models_btn.setVisible(is_ollama)

        # Gemini fields
        self._key_label.setVisible(not is_ollama)
        self._key_edit.setVisible(not is_ollama)
        self._show_key_btn.setVisible(not is_ollama)

        # Check Ollama status
        if is_ollama:
            self._check_ollama_status()

    def _check_ollama_status(self):
        """Check if Ollama is running and update status label."""
        try:
            from ai.ollama_client import is_ollama_running, list_models, get_gpu_vram_mb
            if is_ollama_running():
                models = list_models()
                vram = get_gpu_vram_mb()
                vram_info = f' | GPU: {vram} MB' if vram else ''
                if models:
                    self._ollama_status.setText(
                        f'{len(models)} model(s) ready{vram_info}')
                    self._ollama_status.setStyleSheet(
                        f'color: {COLORS["success"]};')
                    # VRAM estimates per model size tag (MB)
                    vram_needed = {
                        '0.5b': 800, '1.5b': 1500, '3b': 2500,
                        '7b': 5500, '8b': 6000, '13b': 9000, '14b': 10000,
                    }
                    # Update model combo with available models
                    current = self._ollama_model_combo.currentText()
                    self._ollama_model_combo.clear()
                    best_gpu_model = None
                    for m in models:
                        # Check if model fits in GPU
                        size_tag = m.split(':')[-1] if ':' in m else ''
                        needed = vram_needed.get(size_tag, 0)
                        if vram and needed > vram:
                            label = f'{m}  (CPU only - needs {needed} MB)'
                        elif vram and needed > 0:
                            label = f'{m}  (GPU - {needed} MB)'
                            if best_gpu_model is None:
                                best_gpu_model = m
                        else:
                            label = m
                            if best_gpu_model is None:
                                best_gpu_model = m
                        self._ollama_model_combo.addItem(label, m)
                    # Restore selection if it fits GPU, otherwise pick best GPU model
                    restored = False
                    for i in range(self._ollama_model_combo.count()):
                        if self._ollama_model_combo.itemData(i) == current:
                            # Check if saved model fits GPU
                            size_tag = current.split(':')[-1] if ':' in current else ''
                            needed = vram_needed.get(size_tag, 0)
                            if not vram or needed <= vram:
                                self._ollama_model_combo.setCurrentIndex(i)
                                restored = True
                            break
                    if not restored and best_gpu_model:
                        for i in range(self._ollama_model_combo.count()):
                            if self._ollama_model_combo.itemData(i) == best_gpu_model:
                                self._ollama_model_combo.setCurrentIndex(i)
                                break
                else:
                    self._ollama_status.setText(
                        f'No models - pull one first{vram_info}')
                    self._ollama_status.setStyleSheet(
                        f'color: {COLORS["warning"]};')
            else:
                self._ollama_status.setText('Ollama not running')
                self._ollama_status.setStyleSheet(
                    f'color: {COLORS["error"]};')
        except Exception:
            self._ollama_status.setText('Ollama not available')
            self._ollama_status.setStyleSheet(f'color: {COLORS["error"]};')

    def _refresh_ollama_models(self):
        """Refresh the list of available Ollama models."""
        self._check_ollama_status()

    # ─── Settings ─────────────────────────────────────────────────────────

    def _restore_settings(self):
        """Restore last-used settings."""
        # Backend
        backend = self._settings.value('paper_reader_backend', 'ollama')
        idx = self._backend_combo.findData(backend)
        if idx >= 0:
            self._backend_combo.setCurrentIndex(idx)

        # Ollama model
        model = self._settings.value('paper_reader_ollama_model', 'qwen2.5:3b')
        self._ollama_model_combo.setCurrentText(model)

        # API key
        key = self._settings.value('gemini_api_key', '')
        if key:
            self._key_edit.setText(key)

        # Last directory
        last_dir = self._settings.value('paper_reader_last_dir', '')
        self._last_dir = last_dir if last_dir else os.path.expanduser('~')

    def _save_settings(self):
        """Persist settings."""
        self._settings.setValue('paper_reader_backend',
                                self._backend_combo.currentData())
        idx = self._ollama_model_combo.currentIndex()
        model_name = self._ollama_model_combo.itemData(idx)
        if not model_name:
            model_name = self._ollama_model_combo.currentText()
        self._settings.setValue('paper_reader_ollama_model', model_name)
        self._settings.setValue('gemini_api_key', self._key_edit.text().strip())
        if self._pdf_edit.text():
            self._settings.setValue(
                'paper_reader_last_dir',
                os.path.dirname(self._pdf_edit.text()))

    # ─── Actions ──────────────────────────────────────────────────────────

    def _browse_pdf(self):
        path, _ = QFileDialog.getOpenFileName(
            self, 'Select Research Paper PDF',
            self._last_dir, 'PDF Files (*.pdf)')
        if path:
            self._pdf_edit.setText(path)
            self._last_dir = os.path.dirname(path)

    def _toggle_key_visibility(self, show):
        if show:
            self._key_edit.setEchoMode(QLineEdit.EchoMode.Normal)
            self._show_key_btn.setText('Hide')
        else:
            self._key_edit.setEchoMode(QLineEdit.EchoMode.Password)
            self._show_key_btn.setText('Show')

    def _run_extraction(self):
        pdf_path = self._pdf_edit.text().strip()
        backend = self._backend_combo.currentData()

        if not pdf_path:
            QMessageBox.warning(self, 'Missing PDF', 'Please select a PDF file first.')
            return
        if not os.path.isfile(pdf_path):
            QMessageBox.warning(self, 'File Not Found', f'PDF not found:\n{pdf_path}')
            return

        api_key = ''
        ollama_model = 'qwen2.5:3b'

        if backend == 'gemini':
            api_key = self._key_edit.text().strip()
            if not api_key:
                QMessageBox.warning(self, 'Missing API Key',
                    'Please enter your Gemini API key.\n\n'
                    'Get one free at: https://aistudio.google.com/apikey')
                return
        else:
            # Use itemData (actual model name) if available, fall back to text
            idx = self._ollama_model_combo.currentIndex()
            ollama_model = self._ollama_model_combo.itemData(idx)
            if not ollama_model:
                ollama_model = self._ollama_model_combo.currentText().strip()
            if not ollama_model:
                QMessageBox.warning(self, 'No Model',
                    'Please select or enter an Ollama model name.\n\n'
                    'Pull one with: ollama pull qwen2.5:3b')
                return

        # Save settings before running
        self._save_settings()

        # Disable controls
        self._run_btn.setEnabled(False)
        self._browse_btn.setEnabled(False)
        self._save_btn.setEnabled(False)
        self._load_btn.setEnabled(False)
        self._table.setRowCount(0)
        self._report_text.clear()
        self._progress_bar.setValue(0)
        self._progress_bar.setFormat('Starting...')

        # Launch worker thread
        self._worker = _ExtractionWorker(
            pdf_path, backend, api_key, ollama_model)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_progress(self, step: int, message: str):
        self._progress_bar.setValue(step)
        self._progress_bar.setFormat(f'Step {step}/5: {message}')
        self._status_label.setText(message)

    def _on_finished(self, result: dict):
        self._result = result
        self._progress_bar.setValue(5)
        self._progress_bar.setFormat('Done!')
        backend_label = result.get('backend', 'unknown')
        self._status_label.setText(
            f'Extracted {len(result["parameters"])} parameters '
            f'(score: {result["validation"]["score"]:.0f}%) '
            f'via {backend_label}')

        # Populate table
        self._populate_table(result)

        # Populate report
        from ai.parameter_validator import format_validation_report
        report = format_validation_report(result['validation'])
        if result.get('notes'):
            report += f"\n\n--- AI Notes ---\n{result['notes']}"
        report += (f"\n\nBackend: {backend_label}"
                   f"\nPDF: {result.get('pdf_path', '?')}"
                   f"\nText extracted: {result.get('raw_text_length', 0):,} chars"
                   f"\nTables found: {result.get('tables_found', 0)}")
        self._report_text.setPlainText(report)

        # Enable action buttons
        self._save_btn.setEnabled(True)
        self._load_btn.setEnabled(True)
        self._run_btn.setEnabled(True)
        self._browse_btn.setEnabled(True)

        logger.info("Extraction complete: %d params, score=%.1f%% (%s)",
                     len(result['parameters']),
                     result['validation']['score'], backend_label)

    def _on_error(self, error_msg: str):
        self._progress_bar.setValue(0)
        self._progress_bar.setFormat('Error')
        self._status_label.setText(f'Error: {error_msg}')
        self._run_btn.setEnabled(True)
        self._browse_btn.setEnabled(True)

        QMessageBox.critical(self, 'Extraction Error',
            f'Failed to extract parameters:\n\n{error_msg}')

    def _populate_table(self, result: dict):
        """Fill the table with extracted parameters."""
        params = result['parameters']
        confidence = result.get('confidence', {})
        validation = result.get('validation', {})

        # Build a lookup for validation status
        error_fields = {name for name, _, _ in validation.get('errors', [])}
        warning_fields = {name for name, _, _ in validation.get('warnings', [])}

        # All parameter keys
        all_keys = sorted(params.keys())
        self._table.setRowCount(len(all_keys))

        for row, key in enumerate(all_keys):
            value = params[key]
            conf = confidence.get(key, '')

            # Status
            if key in error_fields:
                status = 'ERROR'
                status_color = COLORS['error']
            elif key in warning_fields:
                status = 'WARN'
                status_color = COLORS['warning']
            else:
                status = 'OK'
                status_color = COLORS['success']

            # Confidence color
            if conf == 'extracted':
                conf_color = COLORS['success']
            elif conf == 'estimated':
                conf_color = COLORS['warning']
            elif conf == 'missing':
                conf_color = COLORS['error']
            else:
                conf_color = COLORS['text_dim']

            # Parameter name
            name_item = QTableWidgetItem(key)
            name_item.setFlags(name_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self._table.setItem(row, 0, name_item)

            # Value
            val_str = str(value) if value is not None else '(null)'
            val_item = QTableWidgetItem(val_str)
            val_item.setFlags(val_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self._table.setItem(row, 1, val_item)

            # Confidence
            conf_item = QTableWidgetItem(conf or '-')
            conf_item.setForeground(QColor(conf_color))
            conf_item.setFlags(conf_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self._table.setItem(row, 2, conf_item)

            # Status
            status_item = QTableWidgetItem(status)
            status_item.setForeground(QColor(status_color))
            status_item.setFlags(status_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self._table.setItem(row, 3, status_item)

    def _save_preset(self):
        """Save extracted parameters as a preset JSON file."""
        if not self._result:
            return

        try:
            from ai.paper_pipeline import save_as_preset
            filepath = save_as_preset(self._result['parameters'])
            QMessageBox.information(self, 'Preset Saved',
                f'Parameters saved as preset:\n{filepath}')
            logger.info("Preset saved: %s", filepath)
        except Exception as e:
            QMessageBox.warning(self, 'Save Error', f'Failed to save preset:\n{e}')

    def _load_into_simulator(self):
        """Emit extracted parameters to load into the simulator."""
        if not self._result:
            return

        self.config_extracted.emit(self._result['parameters'])
        self._status_label.setText('Parameters loaded into simulator!')
        logger.info("Parameters emitted to simulator")
        self.accept()

    def closeEvent(self, event):
        """Clean up worker thread on close."""
        if self._worker and self._worker.isRunning():
            self._worker.quit()
            self._worker.wait(3000)
        super().closeEvent(event)
