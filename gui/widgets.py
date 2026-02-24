# gui/widgets.py
"""
Shared GUI widgets used across multiple tabs.

- MplCanvas:        Matplotlib canvas embedded in PyQt6
- StatusIndicator:  Colored dot showing step status
- NetlistViewer:    Monospace read-only text viewer for SPICE netlists
"""

from PyQt6.QtWidgets import (
    QWidget, QLabel, QPlainTextEdit, QVBoxLayout, QHBoxLayout,
    QSizePolicy,
)
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QFont, QColor, QPainter, QPen

import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure


# =============================================================================
# MplCanvas — Matplotlib Figure embedded in Qt
# =============================================================================

class MplCanvas(FigureCanvasQTAgg):
    """Matplotlib canvas widget for embedding plots in PyQt6."""

    def __init__(self, parent=None, width=6, height=4, dpi=100, nrows=1, ncols=1):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        if nrows == 1 and ncols == 1:
            self.ax = self.fig.add_subplot(111)
        else:
            self.axes = self.fig.subplots(nrows, ncols)
            self.ax = self.axes.flat[0] if hasattr(self.axes, 'flat') else self.axes
        self.fig.set_tight_layout(True)
        super().__init__(self.fig)

    def clear(self):
        """Clear all axes."""
        for ax in self.fig.get_axes():
            ax.clear()
        self.draw()


# =============================================================================
# StatusIndicator — Colored status dot
# =============================================================================

_STATUS_COLORS = {
    'idle':    QColor(160, 160, 160),    # gray
    'pending': QColor(160, 160, 160),    # gray
    'running': QColor(255, 200, 0),      # yellow
    'done':    QColor(80, 200, 80),      # green
    'error':   QColor(220, 60, 60),      # red
}


class StatusIndicator(QWidget):
    """Small colored circle indicating status (idle/running/done/error)."""

    def __init__(self, parent=None, size=14):
        super().__init__(parent)
        self._status = 'idle'
        self._size = size
        self.setFixedSize(QSize(size, size))

    @property
    def status(self):
        return self._status

    @status.setter
    def status(self, value: str):
        self._status = value
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        color = _STATUS_COLORS.get(self._status, _STATUS_COLORS['idle'])
        painter.setBrush(color)
        painter.setPen(QPen(color.darker(130), 1))
        margin = 1
        painter.drawEllipse(margin, margin,
                            self._size - 2 * margin,
                            self._size - 2 * margin)
        painter.end()


# =============================================================================
# NetlistViewer — Monospace text viewer for SPICE netlists
# =============================================================================

class NetlistViewer(QPlainTextEdit):
    """Read-only monospace text display for SPICE netlists."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        font = QFont('Consolas', 9)
        font.setStyleHint(QFont.StyleHint.Monospace)
        self.setFont(font)
        self.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)

    def set_netlist(self, text: str):
        """Set the netlist text content."""
        self.setPlainText(text)


# =============================================================================
# QuickInfoPanel — Live parameter readout
# =============================================================================

class QuickInfoPanel(QWidget):
    """Displays key derived values (P_tx, P_rx, I_ph, SNR) that update live."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        title = QLabel('Quick Info')
        title.setStyleSheet('font-weight: bold; font-size: 12px;')
        layout.addWidget(title)

        self._labels = {}
        for key, label_text in [
            ('P_tx', 'P_tx (mW):'),
            ('G_ch', 'Channel Gain:'),
            ('P_rx', 'P_rx (uW):'),
            ('I_ph', 'I_ph (uA):'),
            ('SNR', 'SNR (dB):'),
        ]:
            row = QHBoxLayout()
            lbl = QLabel(label_text)
            lbl.setFixedWidth(100)
            val = QLabel('--')
            val.setStyleSheet('font-weight: bold;')
            self._labels[key] = val
            row.addWidget(lbl)
            row.addWidget(val)
            row.addStretch()
            layout.addLayout(row)

        layout.addStretch()

    def update_from_config(self, config):
        """Update displayed values from a SystemConfig."""
        try:
            P_tx = config.led_radiated_power_mW
            G_ch = config.optical_channel_gain()
            P_rx = config.received_power_W() * 1e6
            I_ph = config.photocurrent_A() * 1e6
            snr = config.snr_estimate_dB()

            self._labels['P_tx'].setText(f'{P_tx:.2f}')
            self._labels['G_ch'].setText(f'{G_ch:.4e}')
            self._labels['P_rx'].setText(f'{P_rx:.2f}')
            self._labels['I_ph'].setText(f'{I_ph:.2f}')
            self._labels['SNR'].setText(f'{snr:.1f}')
        except Exception:
            for lbl in self._labels.values():
                lbl.setText('--')
