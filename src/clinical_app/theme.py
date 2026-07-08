"""
Color palette and Qt stylesheet (QSS) for the Angio-AI Clinical Dashboard.

Palette (as specified): pastel green accent (#BAED91) on a black/ash/white
dark theme. Widgets opt into the accent/secondary button styling via a
"variant" dynamic property, and panels opt into the card look via a "card"
dynamic property, e.g.:

    btn.setProperty("variant", "primary")
    frame.setProperty("card", "true")
"""

ACCENT = "#BAED91"
ACCENT_HOVER = "#CDF3AC"
ACCENT_PRESSED = "#9FD873"
ACCENT_TEXT = "#0B0C0E"  # text drawn on top of the accent color

BG = "#0B0C0E"
SURFACE = "#151719"
SURFACE_ALT = "#1D2024"
BORDER = "#2A2D32"

ASH = "#8A8F98"
ASH_DIM = "#5C6067"
WHITE = "#F2F3F5"

DANGER = "#E5484D"
SUCCESS = ACCENT


def build_stylesheet() -> str:
    return f"""
    QMainWindow, QWidget {{
        background-color: {BG};
        color: {WHITE};
        font-family: 'Segoe UI', sans-serif;
        font-size: 13px;
    }}

    QScrollArea {{
        background-color: transparent;
        border: none;
    }}
    QScrollArea > QWidget > QWidget {{
        background-color: transparent;
    }}

    /* ── Sidebar ─────────────────────────────────────────────────── */
    QFrame#sidebar {{
        background-color: {SURFACE};
        border-right: 1px solid {BORDER};
    }}
    QLabel#brand {{
        color: {ACCENT};
        font-size: 20px;
        font-weight: 700;
        letter-spacing: 1px;
    }}
    QLabel#brandSubtitle {{
        color: {ASH};
        font-size: 11px;
    }}
    QLabel#versionLabel {{
        color: {ASH_DIM};
        font-size: 10px;
    }}
    QListWidget#navList {{
        background-color: transparent;
        border: none;
        outline: none;
    }}
    QListWidget#navList::item {{
        color: {WHITE};
        padding: 10px 12px;
        border-radius: 8px;
        margin-bottom: 2px;
    }}
    QListWidget#navList::item:selected {{
        background-color: {SURFACE_ALT};
        color: {ACCENT};
        font-weight: 600;
    }}
    QListWidget#navList::item:disabled {{
        color: {ASH_DIM};
    }}

    /* ── Cards / panels ──────────────────────────────────────────── */
    QFrame[card="true"] {{
        background-color: {SURFACE};
        border: 1px solid {BORDER};
        border-radius: 10px;
    }}
    QLabel[role="pageTitle"] {{
        font-size: 22px;
        font-weight: 700;
        color: {WHITE};
    }}
    QLabel[role="pageSubtitle"] {{
        color: {ASH};
        font-size: 12.5px;
    }}
    QLabel[role="sectionHeader"] {{
        color: {ACCENT};
        font-size: 13.5px;
        font-weight: 700;
        letter-spacing: 0.5px;
    }}
    QLabel[role="hint"] {{
        color: {ASH_DIM};
        font-size: 10.5px;
    }}
    QLabel[role="fieldLabel"] {{
        color: {ASH};
        font-size: 11.5px;
    }}
    QLabel[role="statusSuccess"] {{
        color: {SUCCESS};
        font-weight: 600;
    }}
    QLabel[role="statusError"] {{
        color: {DANGER};
        font-weight: 600;
    }}

    /* ── Inputs ──────────────────────────────────────────────────── */
    QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QDateEdit, QTextEdit {{
        background-color: {SURFACE_ALT};
        color: {WHITE};
        border: 1px solid {BORDER};
        border-radius: 6px;
        padding: 6px 10px;
        selection-background-color: {ACCENT};
        selection-color: {ACCENT_TEXT};
    }}
    QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus,
    QDateEdit:focus, QTextEdit:focus {{
        border: 1px solid {ACCENT};
    }}
    QLineEdit[error="true"] {{
        border: 1px solid {DANGER};
    }}
    QComboBox::drop-down {{
        border: none;
        width: 22px;
    }}
    QComboBox::down-arrow {{
        width: 0;
        height: 0;
        border-left: 4px solid transparent;
        border-right: 4px solid transparent;
        border-top: 5px solid {ASH};
        margin-right: 10px;
    }}
    QComboBox::down-arrow:on {{
        border-top: 5px solid {ACCENT};
    }}
    QDateEdit::drop-down {{
        border: none;
        width: 22px;
    }}
    QDateEdit::down-arrow {{
        width: 0;
        height: 0;
        border-left: 4px solid transparent;
        border-right: 4px solid transparent;
        border-top: 5px solid {ASH};
        margin-right: 10px;
    }}
    QComboBox QAbstractItemView {{
        background-color: {SURFACE_ALT};
        color: {WHITE};
        border: 1px solid {BORDER};
        selection-background-color: {BORDER};
        selection-color: {ACCENT};
        outline: none;
    }}

    QCheckBox {{
        color: {WHITE};
        spacing: 8px;
        padding: 3px 0;
    }}
    QCheckBox::indicator {{
        width: 16px;
        height: 16px;
        border-radius: 4px;
        border: 1px solid {ASH};
        background-color: {SURFACE_ALT};
    }}
    QCheckBox::indicator:checked {{
        background-color: {ACCENT};
        border: 1px solid {ACCENT};
    }}

    /* ── Buttons ─────────────────────────────────────────────────── */
    QPushButton {{
        background-color: {SURFACE_ALT};
        color: {WHITE};
        border: 1px solid {BORDER};
        border-radius: 8px;
        padding: 9px 18px;
        font-weight: 600;
    }}
    QPushButton:hover {{
        border: 1px solid {ASH};
    }}
    QPushButton:disabled {{
        color: {ASH_DIM};
        border: 1px solid {BORDER};
    }}
    QPushButton[variant="primary"] {{
        background-color: {ACCENT};
        color: {ACCENT_TEXT};
        border: 1px solid {ACCENT};
    }}
    QPushButton[variant="primary"]:hover {{
        background-color: {ACCENT_HOVER};
        border: 1px solid {ACCENT_HOVER};
    }}
    QPushButton[variant="primary"]:pressed {{
        background-color: {ACCENT_PRESSED};
    }}
    QPushButton[variant="primary"]:disabled {{
        background-color: {SURFACE_ALT};
        color: {ASH_DIM};
        border: 1px solid {BORDER};
    }}
    QPushButton[variant="ghost"] {{
        background-color: transparent;
        border: 1px solid {BORDER};
        color: {ASH};
    }}
    QPushButton[variant="ghost"]:hover {{
        color: {WHITE};
        border: 1px solid {ASH};
    }}

    /* ── Lists (e.g. selected DICOM files) ───────────────────────── */
    QListWidget {{
        background-color: {SURFACE_ALT};
        border: 1px solid {BORDER};
        border-radius: 6px;
        color: {WHITE};
    }}
    QListWidget::item {{
        padding: 4px 6px;
    }}
    QListWidget::item:selected {{
        background-color: {BORDER};
        color: {ACCENT};
    }}

    /* ── Status bar ──────────────────────────────────────────────── */
    QStatusBar {{
        background-color: {SURFACE};
        color: {ASH};
        border-top: 1px solid {BORDER};
    }}

    QScrollBar:vertical {{
        background: transparent;
        width: 10px;
    }}
    QScrollBar::handle:vertical {{
        background: {BORDER};
        border-radius: 5px;
        min-height: 24px;
    }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
        height: 0px;
    }}
    """
