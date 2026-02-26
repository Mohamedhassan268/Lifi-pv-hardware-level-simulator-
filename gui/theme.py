# gui/theme.py
"""
Midnight Blue Theme — Centralized color palette and stylesheet.

All GUI colors are defined here. Import COLORS dict or STYLESHEET string.

Usage:
    from gui.theme import COLORS, STYLESHEET
    app.setStyleSheet(STYLESHEET)
"""

# =============================================================================
# Color Palette — Midnight Blue
# =============================================================================

COLORS = {
    # Backgrounds
    'bg':           '#0d1b2a',   # Main background (midnight blue)
    'surface':      '#1b2838',   # Cards, group boxes
    'surface_alt':  '#15202e',   # Alternate surface (slightly darker)
    'border':       '#2c3e50',   # Borders
    'input_bg':     '#162230',   # Input fields, text areas

    # Text
    'text':         '#e8e8e8',   # Primary text (off-white)
    'text_dim':     '#7f8c9b',   # Secondary/dimmed text
    'text_bright':  '#ffffff',   # Bright text (headings)

    # Accent
    'accent':       '#f0b429',   # Primary accent (amber/gold)
    'accent_dim':   '#c99a20',   # Dimmed accent
    'accent_bg':    '#2a2210',   # Accent background tint

    # Status
    'success':      '#4caf50',   # Green (pass, done)
    'success_bg':   '#1a2e1a',   # Green background tint
    'error':        '#ff5252',   # Red (fail, error)
    'error_bg':     '#2e1a1a',   # Red background tint
    'warning':      '#ff9800',   # Orange (warning)
    'warning_bg':   '#2e2210',   # Warning background tint
    'info':         '#42a5f5',   # Blue (info, measured)

    # Status indicator specific
    'idle':         '#556677',   # Gray (pending/idle)
    'running':      '#f0b429',   # Gold (running)
    'done':         '#4caf50',   # Green (done)

    # Tab bar
    'tab_bg':       '#15202e',   # Inactive tab
    'tab_active':   '#1b2838',   # Active tab (matches surface)
    'tab_hover':    '#1e2d3d',   # Tab hover
    'tab_border':   '#f0b429',   # Active tab accent line

    # Status bar
    'statusbar_bg': '#0a1520',   # Darkest
}


# =============================================================================
# QSS Stylesheet
# =============================================================================

STYLESHEET = f"""
/* === Global === */
QMainWindow, QWidget {{
    background-color: {COLORS['bg']};
    color: {COLORS['text']};
    font-family: "Segoe UI", "Helvetica", "Arial", sans-serif;
    font-size: 13px;
}}

/* === Tab Widget === */
QTabWidget::pane {{
    border: 1px solid {COLORS['border']};
    background: {COLORS['surface']};
    border-radius: 4px;
}}

QTabBar::tab {{
    background: {COLORS['tab_bg']};
    color: {COLORS['text_dim']};
    border: 1px solid {COLORS['border']};
    border-bottom: none;
    padding: 8px 18px;
    margin-right: 2px;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
}}

QTabBar::tab:selected {{
    background: {COLORS['tab_active']};
    color: {COLORS['accent']};
    border-top: 2px solid {COLORS['accent']};
}}

QTabBar::tab:hover:!selected {{
    background: {COLORS['tab_hover']};
    color: {COLORS['text']};
}}

/* === Group Boxes === */
QGroupBox {{
    background-color: {COLORS['surface']};
    border: 1px solid {COLORS['border']};
    border-radius: 6px;
    margin-top: 14px;
    padding-top: 16px;
    font-weight: bold;
    color: {COLORS['accent']};
}}

QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 2px 10px;
    color: {COLORS['accent']};
}}

/* === Labels === */
QLabel {{
    color: {COLORS['text']};
    background: transparent;
}}

/* === Buttons === */
QPushButton {{
    background-color: {COLORS['border']};
    color: {COLORS['text']};
    border: 1px solid {COLORS['border']};
    border-radius: 4px;
    padding: 6px 16px;
    font-weight: bold;
}}

QPushButton:hover {{
    background-color: {COLORS['accent']};
    color: {COLORS['bg']};
    border-color: {COLORS['accent']};
}}

QPushButton:pressed {{
    background-color: {COLORS['accent_dim']};
    color: {COLORS['bg']};
}}

QPushButton:disabled {{
    background-color: {COLORS['surface']};
    color: {COLORS['idle']};
    border-color: {COLORS['surface']};
}}

/* === Combo Boxes === */
QComboBox {{
    background-color: {COLORS['input_bg']};
    color: {COLORS['text']};
    border: 1px solid {COLORS['border']};
    border-radius: 4px;
    padding: 4px 8px;
}}

QComboBox::drop-down {{
    border: none;
    width: 24px;
}}

QComboBox QAbstractItemView {{
    background-color: {COLORS['surface']};
    color: {COLORS['text']};
    border: 1px solid {COLORS['border']};
    selection-background-color: {COLORS['accent_bg']};
    selection-color: {COLORS['accent']};
}}

/* === Line Edits / Spin Boxes === */
QLineEdit, QSpinBox, QDoubleSpinBox {{
    background-color: {COLORS['input_bg']};
    color: {COLORS['text']};
    border: 1px solid {COLORS['border']};
    border-radius: 4px;
    padding: 4px 8px;
}}

QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
    border-color: {COLORS['accent']};
}}

/* === Text Edit / Plain Text Edit === */
QPlainTextEdit, QTextEdit {{
    background-color: {COLORS['input_bg']};
    color: {COLORS['text']};
    border: 1px solid {COLORS['border']};
    border-radius: 4px;
    font-family: "Consolas", "Courier New", monospace;
    font-size: 12px;
}}

/* === Tables === */
QTableWidget, QTableView {{
    background-color: {COLORS['surface']};
    color: {COLORS['text']};
    border: 1px solid {COLORS['border']};
    gridline-color: {COLORS['border']};
    alternate-background-color: {COLORS['surface_alt']};
}}

QHeaderView::section {{
    background-color: {COLORS['surface_alt']};
    color: {COLORS['accent']};
    border: 1px solid {COLORS['border']};
    padding: 4px 8px;
    font-weight: bold;
}}

QTableWidget::item:selected {{
    background-color: {COLORS['accent_bg']};
    color: {COLORS['accent']};
}}

/* === List Widgets === */
QListWidget {{
    background-color: {COLORS['input_bg']};
    color: {COLORS['text']};
    border: 1px solid {COLORS['border']};
    border-radius: 4px;
}}

QListWidget::item:selected {{
    background-color: {COLORS['accent_bg']};
    color: {COLORS['accent']};
}}

QListWidget::item:hover {{
    background-color: {COLORS['tab_hover']};
}}

/* === Scroll Bars === */
QScrollBar:vertical {{
    background: {COLORS['surface']};
    width: 10px;
    margin: 0;
}}

QScrollBar::handle:vertical {{
    background: {COLORS['border']};
    border-radius: 5px;
    min-height: 30px;
}}

QScrollBar::handle:vertical:hover {{
    background: {COLORS['accent_dim']};
}}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0;
}}

QScrollBar:horizontal {{
    background: {COLORS['surface']};
    height: 10px;
    margin: 0;
}}

QScrollBar::handle:horizontal {{
    background: {COLORS['border']};
    border-radius: 5px;
    min-width: 30px;
}}

QScrollBar::handle:horizontal:hover {{
    background: {COLORS['accent_dim']};
}}

QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
    width: 0;
}}

/* === Splitter === */
QSplitter::handle {{
    background: {COLORS['border']};
    width: 2px;
    height: 2px;
}}

/* === Status Bar === */
QStatusBar {{
    background-color: {COLORS['statusbar_bg']};
    color: {COLORS['text_dim']};
    border-top: 1px solid {COLORS['border']};
}}

QStatusBar QLabel {{
    color: {COLORS['text_dim']};
}}

/* === Menu Bar === */
QMenuBar {{
    background-color: {COLORS['statusbar_bg']};
    color: {COLORS['text']};
    border-bottom: 1px solid {COLORS['border']};
}}

QMenuBar::item:selected {{
    background-color: {COLORS['accent_bg']};
    color: {COLORS['accent']};
}}

QMenu {{
    background-color: {COLORS['surface']};
    color: {COLORS['text']};
    border: 1px solid {COLORS['border']};
}}

QMenu::item:selected {{
    background-color: {COLORS['accent_bg']};
    color: {COLORS['accent']};
}}

/* === Progress Bar === */
QProgressBar {{
    background-color: {COLORS['input_bg']};
    border: 1px solid {COLORS['border']};
    border-radius: 4px;
    text-align: center;
    color: {COLORS['text']};
}}

QProgressBar::chunk {{
    background-color: {COLORS['accent']};
    border-radius: 3px;
}}

/* === Scroll Area === */
QScrollArea {{
    background-color: {COLORS['surface']};
    border: 1px solid {COLORS['border']};
}}

/* === Check Box / Radio === */
QCheckBox, QRadioButton {{
    color: {COLORS['text']};
}}

QCheckBox::indicator, QRadioButton::indicator {{
    width: 16px;
    height: 16px;
}}

/* === Tooltips === */
QToolTip {{
    background-color: {COLORS['surface']};
    color: {COLORS['text']};
    border: 1px solid {COLORS['accent_dim']};
    border-radius: 4px;
    padding: 6px 8px;
    font-size: 12px;
}}

/* === Slider (wizard) === */
QSlider::groove:horizontal {{
    background: {COLORS['border']};
    height: 6px;
    border-radius: 3px;
}}

QSlider::handle:horizontal {{
    background: {COLORS['accent']};
    width: 16px;
    height: 16px;
    margin: -5px 0;
    border-radius: 8px;
}}
"""
