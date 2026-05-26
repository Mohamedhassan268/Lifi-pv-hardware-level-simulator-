"""
Tab 8: Message TX/RX

Type a text message, simulate it through the full TX -> Channel -> RX chain,
and see what the receiver decodes. Wrong characters are highlighted red, and
the simulation runs in a background thread so the GUI stays responsive.

Uses the Python engine (cosim.python_engine.run_python_simulation) with a
user-supplied bit sequence built by cosim.text_codec.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import html

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QPushButton, QPlainTextEdit, QTextEdit, QGridLayout,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal

from cosim.system_config import SystemConfig
from cosim.python_engine import run_python_simulation
from cosim.text_codec import encode_text, decode_text, char_diff, MAX_PAYLOAD_BYTES
from gui.theme import COLORS


class _MessageWorker(QThread):
    """Runs the Python simulation with user-supplied bits off the UI thread."""
    finished = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(self, config, bits_tx):
        super().__init__()
        self._config = config
        self._bits_tx = bits_tx

    def run(self):
        try:
            result = run_python_simulation(self._config,
                                           bits_override=self._bits_tx)
            self.finished.emit(result)
        except Exception as exc:
            self.failed.emit(f'{type(exc).__name__}: {exc}')


class MessageTab(QWidget):
    """Tab: encode a text message and send it through the simulated link."""

    def __init__(self, config: SystemConfig, parent=None):
        super().__init__(parent)
        self._config = config
        self._worker = None
        self._build_ui()

    # ------------------------------------------------------------------
    # External API (used by MainWindow)
    # ------------------------------------------------------------------
    def update_config(self, config: SystemConfig):
        self._config = config
        self._refresh_info()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self):
        main = QVBoxLayout(self)

        header = QLabel(
            'Type a message, press <b>Send</b>, and watch it go through the '
            'full TX → Channel → RX chain. Corrupted characters are shown in '
            'red on the receiver.')
        header.setWordWrap(True)
        header.setStyleSheet(f'color: {COLORS["text_dim"]}; padding: 4px;')
        main.addWidget(header)

        # ---- Config summary row ----
        self._info_label = QLabel('')
        self._info_label.setStyleSheet(
            f'color: {COLORS["text_dim"]}; padding: 2px 6px; '
            f'background: {COLORS["surface_alt"]}; border-radius: 3px;')
        main.addWidget(self._info_label)

        # ---- TX group ----
        tx_group = QGroupBox('Transmitter')
        tx_lay = QVBoxLayout(tx_group)
        self._tx_edit = QPlainTextEdit('I am mohamed')
        self._tx_edit.setPlaceholderText('Type the message to send…')
        self._tx_edit.setFixedHeight(80)
        tx_lay.addWidget(self._tx_edit)

        btn_row = QHBoxLayout()
        self._send_btn = QPushButton('Send')
        self._send_btn.clicked.connect(self._on_send)
        btn_row.addWidget(self._send_btn)
        self._status_label = QLabel('Idle')
        self._status_label.setStyleSheet(f'color: {COLORS["text_dim"]};')
        btn_row.addWidget(self._status_label)
        btn_row.addStretch()
        tx_lay.addLayout(btn_row)
        main.addWidget(tx_group)

        # ---- RX group ----
        rx_group = QGroupBox('Receiver (decoded)')
        rx_lay = QVBoxLayout(rx_group)
        self._rx_view = QTextEdit()
        self._rx_view.setReadOnly(True)
        self._rx_view.setFixedHeight(80)
        rx_lay.addWidget(self._rx_view)
        main.addWidget(rx_group)

        # ---- Stats grid ----
        stats_group = QGroupBox('Link statistics')
        grid = QGridLayout(stats_group)
        self._stat_labels = {}
        fields = [
            ('bits_total', 'Bits transmitted:'),
            ('bit_errors', 'Bit errors:'),
            ('ber', 'Bit error rate:'),
            ('char_errors', 'Character errors:'),
            ('snr', 'Estimated SNR (dB):'),
            ('sync', 'Frame sync:'),
        ]
        for row, (key, label) in enumerate(fields):
            grid.addWidget(QLabel(label), row, 0)
            val = QLabel('—')
            val.setStyleSheet(f'color: {COLORS["accent"]}; font-weight: bold;')
            grid.addWidget(val, row, 1)
            self._stat_labels[key] = val
        grid.setColumnStretch(2, 1)
        main.addWidget(stats_group)

        main.addStretch()
        self._refresh_info()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _refresh_info(self):
        cfg = self._config
        engine = getattr(cfg, 'simulation_engine', 'python')
        note = ''
        if engine != 'python':
            note = ('  (note: this tab always uses the Python engine — '
                    'SPICE is not used for the text link)')
        self._info_label.setText(
            f'Preset: <b>{cfg.preset_name or "Custom"}</b>  |  '
            f'Modulation: <b>{cfg.modulation}</b>  |  '
            f'Rate: <b>{cfg.data_rate_bps/1e3:.1f} kbps</b>  |  '
            f'Distance: <b>{cfg.distance_m:.2f} m</b>{note}')

    def _set_busy(self, busy: bool):
        self._send_btn.setEnabled(not busy)
        self._tx_edit.setReadOnly(busy)
        if busy:
            self._status_label.setText('Simulating…')
            self._status_label.setStyleSheet(f'color: {COLORS["accent"]};')
        else:
            self._status_label.setText('Idle')
            self._status_label.setStyleSheet(f'color: {COLORS["text_dim"]};')

    # ------------------------------------------------------------------
    # Send / receive flow
    # ------------------------------------------------------------------
    def _on_send(self):
        message = self._tx_edit.toPlainText()
        if not message:
            self._status_label.setText('Type something first.')
            return
        try:
            bits_tx = encode_text(message)
        except ValueError as exc:
            self._status_label.setText(str(exc))
            self._status_label.setStyleSheet(f'color: {COLORS["error"]};')
            return

        # Modulation-specific minimum bit counts (OFDM needs one full symbol).
        mod = self._config.modulation.upper().replace('-', '_')
        if mod == 'OFDM':
            bps = self._stats_min_bits_ofdm()
            if len(bits_tx) < bps:
                self._status_label.setText(
                    f'OFDM needs ≥{bps} bits/symbol — use OOK for short '
                    f'messages, or type a longer message.')
                self._status_label.setStyleSheet(
                    f'color: {COLORS["error"]};')
                return

        self._set_busy(True)
        self._worker = _MessageWorker(self._config, bits_tx)
        self._worker.finished.connect(self._on_result)
        self._worker.failed.connect(self._on_failure)
        self._worker.start()

    def _stats_min_bits_ofdm(self) -> int:
        cfg = self._config
        import numpy as np
        bps = int(np.log2(getattr(cfg, 'ofdm_qam_order', 16)))
        n_sc = min(getattr(cfg, 'ofdm_n_subcarriers', 80),
                   getattr(cfg, 'ofdm_nfft', 256) // 2 - 1)
        return bps * n_sc

    def _on_failure(self, msg: str):
        self._set_busy(False)
        self._status_label.setText('Error')
        self._status_label.setStyleSheet(f'color: {COLORS["error"]};')
        self._rx_view.setPlainText(f'Simulation failed:\n{msg}')

    def _on_result(self, result: dict):
        self._set_busy(False)
        tx_text = self._tx_edit.toPlainText()
        bits_rx = result.get('bits_rx')
        if bits_rx is None:
            self._on_failure('Engine returned no bits_rx field.')
            return

        rx_text, info = decode_text(bits_rx)
        self._render_rx(tx_text, rx_text, info)
        self._update_stats(tx_text, rx_text, result, info)

    def _render_rx(self, tx_text: str, rx_text: str, info: dict):
        """Render the received text with red highlights on wrong characters."""
        if not info['sync_found']:
            self._rx_view.setHtml(
                f'<span style="color:{COLORS["error"]};">'
                f'[frame sync lost — preamble not found in received bits]'
                f'</span>')
            return

        parts = []
        for ch, status in char_diff(tx_text, rx_text):
            safe = html.escape(ch).replace(' ', '&nbsp;')
            if status == 'ok':
                parts.append(
                    f'<span style="color:{COLORS["text"]};">{safe}</span>')
            elif status == 'bad':
                parts.append(
                    f'<span style="background:{COLORS["error_bg"]}; '
                    f'color:{COLORS["error"]}; font-weight:bold;">'
                    f'{safe}</span>')
            elif status == 'missing':
                parts.append(
                    f'<span style="background:{COLORS["warning_bg"]}; '
                    f'color:{COLORS["warning"]};">·</span>')
            else:  # extra
                parts.append(
                    f'<span style="background:{COLORS["warning_bg"]}; '
                    f'color:{COLORS["warning"]};">{safe}</span>')
        if info['truncated']:
            parts.append(
                f'<span style="color:{COLORS["warning"]};"> '
                f'[truncated]</span>')
        self._rx_view.setHtml(
            f'<div style="font-family:Consolas,monospace;font-size:13px;">'
            f'{"".join(parts)}</div>')

    def _update_stats(self, tx_text: str, rx_text: str,
                      result: dict, info: dict):
        bits_tx = result.get('bits_tx')
        bits_rx = result.get('bits_rx')
        n_bits = int(min(len(bits_tx), len(bits_rx))) if bits_tx is not None \
            else 0
        bit_errors = int(result.get('n_errors', 0))
        ber = result.get('ber', 0.0)

        # Character-level comparison (based on actual decoded text).
        char_errors = 0
        for _, status in char_diff(tx_text, rx_text):
            if status != 'ok':
                char_errors += 1

        self._stat_labels['bits_total'].setText(str(n_bits))
        self._stat_labels['bit_errors'].setText(str(bit_errors))
        self._stat_labels['ber'].setText(f'{ber:.2e}')
        self._stat_labels['char_errors'].setText(
            f'{char_errors} / {len(tx_text)}')
        self._stat_labels['snr'].setText(
            f'{result.get("snr_est_dB", 0.0):.1f}')
        self._stat_labels['sync'].setText(
            'locked' if info['sync_found'] else 'LOST')
        sync_color = COLORS['success'] if info['sync_found'] \
            else COLORS['error']
        self._stat_labels['sync'].setStyleSheet(
            f'color: {sync_color}; font-weight: bold;')
