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

from gui.theme import COLORS


# =============================================================================
# MplCanvas — Matplotlib Figure embedded in Qt
# =============================================================================

class MplCanvas(FigureCanvasQTAgg):
    """Matplotlib canvas widget for embedding plots in PyQt6."""

    def __init__(self, parent=None, width=6, height=4, dpi=100, nrows=1, ncols=1):
        self.fig = Figure(figsize=(width, height), dpi=dpi,
                          facecolor=COLORS['surface'])
        if nrows == 1 and ncols == 1:
            self.ax = self.fig.add_subplot(111)
        else:
            self.axes = self.fig.subplots(nrows, ncols)
            self.ax = self.axes.flat[0] if hasattr(self.axes, 'flat') else self.axes
        # Apply dark theme to all axes
        for ax in self.fig.get_axes():
            ax.set_facecolor(COLORS['input_bg'])
            ax.tick_params(colors=COLORS['text_dim'], labelsize=8)
            ax.xaxis.label.set_color(COLORS['text_dim'])
            ax.yaxis.label.set_color(COLORS['text_dim'])
            ax.title.set_color(COLORS['text'])
            for spine in ax.spines.values():
                spine.set_color(COLORS['border'])
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
    'idle':    QColor(COLORS['idle']),
    'pending': QColor(COLORS['idle']),
    'running': QColor(COLORS['running']),
    'done':    QColor(COLORS['done']),
    'error':   QColor(COLORS['error']),
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


# =============================================================================
# WarningsPanel — Live configuration warnings & recommendations
# =============================================================================

_LEVEL_STYLES = {
    'error':   (COLORS['error'],   COLORS['error_bg']),
    'warning': (COLORS['warning'], COLORS['warning_bg']),
    'info':    (COLORS['info'],    COLORS['surface_alt']),
    'success': (COLORS['success'], COLORS['success_bg']),
}


class WarningsPanel(QWidget):
    """Displays live warnings and recommendations based on current config."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        title = QLabel('Diagnostics')
        title.setStyleSheet('font-weight: bold; font-size: 12px;')
        layout.addWidget(title)

        self._container = QVBoxLayout()
        self._container.setSpacing(4)
        layout.addLayout(self._container)
        layout.addStretch()

        self._msg_labels = []

    def update_from_config(self, config):
        """Run all checks and update displayed warnings."""
        # Clear previous messages
        for lbl in self._msg_labels:
            lbl.deleteLater()
        self._msg_labels.clear()

        messages = self._check_config(config)

        if not messages:
            messages = [('success', 'Configuration looks good')]

        for level, text in messages:
            lbl = QLabel(text)
            lbl.setWordWrap(True)
            fg, bg = _LEVEL_STYLES.get(level, _LEVEL_STYLES['info'])
            lbl.setStyleSheet(
                f'color: {fg}; background: {bg}; '
                f'padding: 4px 6px; border-radius: 3px; font-size: 11px; '
                f'border-left: 3px solid {fg};')
            self._container.addWidget(lbl)
            self._msg_labels.append(lbl)

    def _check_config(self, config):
        """Return list of (level, message) tuples."""
        msgs = []
        try:
            snr = config.snr_estimate_dB()
            P_rx_uW = config.received_power_W() * 1e6

            # SNR checks
            if snr < 6:
                msgs.append(('error',
                    f'Very weak signal (SNR: {snr:.1f} dB). '
                    f'Reduce distance or increase LED power.'))
            elif snr < 12:
                msgs.append(('warning',
                    f'Marginal SNR ({snr:.1f} dB) \u2014 expect high BER. '
                    f'Try reducing distance.'))
            else:
                msgs.append(('success',
                    f'Good link budget (SNR: {snr:.1f} dB)'))

            # Received power check
            if P_rx_uW < 0.01:
                msgs.append(('error',
                    f'P_rx = {P_rx_uW:.4f} \u00b5W \u2014 '
                    f'too weak to detect.'))

            # Geometry checks
            half_angle = getattr(config, 'led_half_angle_deg', 60)
            tx_angle = getattr(config, 'tx_angle_deg', 0)
            if tx_angle > half_angle:
                msgs.append(('warning',
                    f'TX angle ({tx_angle}\u00b0) exceeds LED half-angle '
                    f'({half_angle}\u00b0) \u2014 receiver outside main beam.'))

            # Simulation duration check
            data_rate = getattr(config, 'data_rate_bps', 5000)
            n_bits = getattr(config, 'n_bits', 1000)
            t_stop = getattr(config, 't_stop_s', 0.2)
            min_t = n_bits / data_rate if data_rate > 0 else 0
            if t_stop < min_t * 0.95:
                msgs.append(('warning',
                    f'Simulation too short ({t_stop:.4f}s) for {n_bits} bits '
                    f'at {data_rate:.0f} bps (need {min_t:.4f}s).'))

            # SPICE availability
            engine = getattr(config, 'simulation_engine', 'spice')
            if engine == 'spice':
                try:
                    from cosim.spice_finder import spice_available
                    if not spice_available():
                        msgs.append(('warning',
                            'No SPICE engine found \u2014 will auto-fall '
                            'back to Python engine.'))
                except ImportError:
                    pass

            # Distance sanity
            distance = getattr(config, 'distance_m', 0.3)
            if distance < 0.02:
                msgs.append(('info',
                    f'Very close range ({distance*100:.1f} cm) \u2014 '
                    f'near-field effects may not be modeled.'))
            elif distance > 3.0:
                msgs.append(('info',
                    f'Long distance ({distance:.1f} m) \u2014 '
                    f'signal attenuation will be significant.'))

        except Exception:
            msgs.append(('info', 'Unable to evaluate configuration'))

        return msgs
