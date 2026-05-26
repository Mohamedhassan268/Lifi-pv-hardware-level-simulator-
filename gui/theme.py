# gui/theme.py
"""
Clinical Light theme — minimalist palette and stylesheet.

Design principles:
- Ample whitespace (16-24px gutters)
- Hairline borders only (1px, used structurally)
- No drop shadows
- Single amber accent, used sparingly for active/Run state
- Paper-white background, near-black text

Usage:
    from gui.theme import COLORS, STYLESHEET
    app.setStyleSheet(STYLESHEET)
"""

COLORS = {
    'bg':           '#FAFAF7',
    'surface':      '#FFFFFF',
    'surface_alt':  '#F4F3EE',
    'border':       '#E5E5E0',
    'border_strong':'#D4D4CE',
    'input_bg':     '#FFFFFF',

    'text':         '#1A1A1A',
    'text_dim':     '#6B6B6B',
    'text_bright':  '#000000',

    'accent':       '#D97706',
    'accent_dim':   '#B45309',
    'accent_bg':    '#FEF3E2',

    'success':      '#059669',
    'success_bg':   '#ECFDF5',
    'error':        '#DC2626',
    'error_bg':     '#FEF2F2',
    'warning':      '#D97706',
    'warning_bg':   '#FEF3E2',
    'info':         '#2563EB',

    'idle':         '#9CA3AF',
    'running':      '#D97706',
    'done':         '#059669',

    'tab_bg':       '#FAFAF7',
    'tab_active':   '#FFFFFF',
    'tab_hover':    '#F4F3EE',
    'tab_border':   '#D97706',

    'statusbar_bg': '#F4F3EE',
}


STYLESHEET = f"""
QMainWindow, QWidget {{
    background-color: {COLORS['bg']};
    color: {COLORS['text']};
    font-family: "Inter", "Segoe UI", "Helvetica", "Arial", sans-serif;
    font-size: 13px;
}}

QTabWidget::pane {{
    border: 1px solid {COLORS['border']};
    background: {COLORS['surface']};
    border-radius: 0px;
    top: -1px;
}}

QTabBar::tab {{
    background: transparent;
    color: {COLORS['text_dim']};
    border: none;
    border-bottom: 2px solid transparent;
    padding: 10px 18px;
    margin-right: 4px;
    font-weight: 500;
}}

QTabBar::tab:selected {{
    background: transparent;
    color: {COLORS['text']};
    border-bottom: 2px solid {COLORS['accent']};
}}

QTabBar::tab:hover:!selected {{
    color: {COLORS['text']};
}}

QGroupBox {{
    background-color: transparent;
    border: none;
    border-top: 1px solid {COLORS['border']};
    border-radius: 0px;
    margin-top: 22px;
    padding-top: 14px;
    font-weight: 600;
    color: {COLORS['text_dim']};
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}}

QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 0 4px 0;
    color: {COLORS['text_dim']};
}}

QLabel {{
    color: {COLORS['text']};
    background: transparent;
}}

QPushButton {{
    background-color: {COLORS['surface']};
    color: {COLORS['text']};
    border: 1px solid {COLORS['border_strong']};
    border-radius: 4px;
    padding: 7px 14px;
    font-weight: 500;
}}

QPushButton:hover {{
    background-color: {COLORS['surface_alt']};
    border-color: {COLORS['text_dim']};
}}

QPushButton:pressed {{
    background-color: {COLORS['border']};
}}

QPushButton:disabled {{
    background-color: {COLORS['surface']};
    color: {COLORS['idle']};
    border-color: {COLORS['border']};
}}

QPushButton[role="primary"] {{
    background-color: {COLORS['accent']};
    color: #FFFFFF;
    border-color: {COLORS['accent']};
    font-weight: 600;
}}

QPushButton[role="primary"]:hover {{
    background-color: {COLORS['accent_dim']};
    border-color: {COLORS['accent_dim']};
}}

QPushButton[role="primary"]:disabled {{
    background-color: {COLORS['border']};
    color: {COLORS['text_dim']};
    border-color: {COLORS['border']};
}}

QComboBox {{
    background-color: {COLORS['input_bg']};
    color: {COLORS['text']};
    border: 1px solid {COLORS['border_strong']};
    border-radius: 4px;
    padding: 6px 10px;
    min-height: 18px;
}}

QComboBox:hover {{
    border-color: {COLORS['text_dim']};
}}

QComboBox:focus {{
    border-color: {COLORS['accent']};
}}

QComboBox::drop-down {{
    border: none;
    width: 22px;
}}

QComboBox QAbstractItemView {{
    background-color: {COLORS['surface']};
    color: {COLORS['text']};
    border: 1px solid {COLORS['border_strong']};
    selection-background-color: {COLORS['accent_bg']};
    selection-color: {COLORS['accent_dim']};
    outline: 0;
    padding: 2px;
}}

QLineEdit, QSpinBox, QDoubleSpinBox {{
    background-color: {COLORS['input_bg']};
    color: {COLORS['text']};
    border: 1px solid {COLORS['border_strong']};
    border-radius: 4px;
    padding: 6px 10px;
}}

QLineEdit:hover, QSpinBox:hover, QDoubleSpinBox:hover {{
    border-color: {COLORS['text_dim']};
}}

QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
    border-color: {COLORS['accent']};
}}

QPlainTextEdit, QTextEdit {{
    background-color: {COLORS['surface']};
    color: {COLORS['text']};
    border: 1px solid {COLORS['border']};
    border-radius: 4px;
    font-family: "JetBrains Mono", "Consolas", "Courier New", monospace;
    font-size: 12px;
    padding: 8px;
}}

QTableWidget, QTableView {{
    background-color: {COLORS['surface']};
    color: {COLORS['text']};
    border: 1px solid {COLORS['border']};
    gridline-color: {COLORS['border']};
    alternate-background-color: {COLORS['surface_alt']};
    selection-background-color: {COLORS['accent_bg']};
    selection-color: {COLORS['accent_dim']};
}}

QHeaderView::section {{
    background-color: {COLORS['surface_alt']};
    color: {COLORS['text_dim']};
    border: none;
    border-bottom: 1px solid {COLORS['border_strong']};
    padding: 8px 10px;
    font-weight: 600;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.3px;
}}

QTableWidget::item:selected {{
    background-color: {COLORS['accent_bg']};
    color: {COLORS['accent_dim']};
}}

QListWidget, QTreeWidget {{
    background-color: {COLORS['surface']};
    color: {COLORS['text']};
    border: 1px solid {COLORS['border']};
    border-radius: 4px;
    outline: 0;
}}

QListWidget::item, QTreeWidget::item {{
    padding: 5px 6px;
    border: none;
}}

QListWidget::item:selected, QTreeWidget::item:selected {{
    background-color: {COLORS['accent_bg']};
    color: {COLORS['accent_dim']};
}}

QListWidget::item:hover, QTreeWidget::item:hover {{
    background-color: {COLORS['surface_alt']};
}}

QScrollBar:vertical {{
    background: transparent;
    width: 10px;
    margin: 0;
}}

QScrollBar::handle:vertical {{
    background: {COLORS['border_strong']};
    border-radius: 5px;
    min-height: 30px;
}}

QScrollBar::handle:vertical:hover {{
    background: {COLORS['text_dim']};
}}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0;
}}

QScrollBar:horizontal {{
    background: transparent;
    height: 10px;
    margin: 0;
}}

QScrollBar::handle:horizontal {{
    background: {COLORS['border_strong']};
    border-radius: 5px;
    min-width: 30px;
}}

QScrollBar::handle:horizontal:hover {{
    background: {COLORS['text_dim']};
}}

QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
    width: 0;
}}

QSplitter::handle {{
    background: {COLORS['border']};
}}

QSplitter::handle:horizontal {{ width: 1px; }}
QSplitter::handle:vertical   {{ height: 1px; }}

QStatusBar {{
    background-color: {COLORS['statusbar_bg']};
    color: {COLORS['text_dim']};
    border-top: 1px solid {COLORS['border']};
}}

QStatusBar::item {{ border: none; }}

QStatusBar QLabel {{
    color: {COLORS['text_dim']};
    font-size: 12px;
}}

QMenuBar {{
    background-color: {COLORS['bg']};
    color: {COLORS['text']};
    border-bottom: 1px solid {COLORS['border']};
    padding: 2px;
}}

QMenuBar::item {{
    padding: 6px 12px;
    background: transparent;
}}

QMenuBar::item:selected {{
    background-color: {COLORS['surface_alt']};
}}

QMenu {{
    background-color: {COLORS['surface']};
    color: {COLORS['text']};
    border: 1px solid {COLORS['border_strong']};
    padding: 4px;
}}

QMenu::item {{
    padding: 6px 22px 6px 18px;
    border-radius: 3px;
}}

QMenu::item:selected {{
    background-color: {COLORS['surface_alt']};
    color: {COLORS['text']};
}}

QMenu::separator {{
    height: 1px;
    background: {COLORS['border']};
    margin: 4px 6px;
}}

QProgressBar {{
    background-color: {COLORS['surface_alt']};
    border: none;
    border-radius: 2px;
    text-align: center;
    color: {COLORS['text_dim']};
    font-size: 11px;
    height: 4px;
}}

QProgressBar::chunk {{
    background-color: {COLORS['accent']};
    border-radius: 2px;
}}

QScrollArea {{
    background-color: transparent;
    border: none;
}}

QCheckBox, QRadioButton {{
    color: {COLORS['text']};
    spacing: 8px;
}}

QCheckBox::indicator, QRadioButton::indicator {{
    width: 16px;
    height: 16px;
}}

QToolTip {{
    background-color: {COLORS['text']};
    color: {COLORS['surface']};
    border: none;
    border-radius: 4px;
    padding: 6px 10px;
    font-size: 12px;
}}

QSlider::groove:horizontal {{
    background: {COLORS['border']};
    height: 4px;
    border-radius: 2px;
}}

QSlider::handle:horizontal {{
    background: {COLORS['accent']};
    width: 14px;
    height: 14px;
    margin: -5px 0;
    border-radius: 7px;
}}

QSlider::sub-page:horizontal {{
    background: {COLORS['accent']};
    border-radius: 2px;
}}

QToolBar {{
    background: {COLORS['bg']};
    border: none;
    border-bottom: 1px solid {COLORS['border']};
    padding: 8px 16px;
    spacing: 12px;
}}

QToolBar QLabel {{
    color: {COLORS['text_dim']};
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}}

QDockWidget {{
    color: {COLORS['text']};
    font-weight: 500;
}}

QDockWidget::title {{
    background: {COLORS['surface_alt']};
    border-bottom: 1px solid {COLORS['border']};
    padding: 6px 12px;
    text-align: left;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: {COLORS['text_dim']};
}}

QToolBox::tab {{
    background: {COLORS['surface_alt']};
    border: none;
    border-bottom: 1px solid {COLORS['border']};
    color: {COLORS['text']};
    padding: 10px 12px;
    font-weight: 600;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    text-align: left;
}}

QToolBox::tab:selected {{
    background: {COLORS['surface']};
    color: {COLORS['accent']};
    border-left: 2px solid {COLORS['accent']};
}}

QToolBox::tab:hover {{
    background: {COLORS['border']};
}}
"""
