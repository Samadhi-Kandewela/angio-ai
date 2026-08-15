"""
auth_dialog.py
--------------
Login / Register dialog for the Cardexa Clinical Dashboard.

Shows before the main AppWindow so only authenticated cardiologists can
access the system. Credentials are checked against the `users` table in
patients.db (see patient_db.py).  No third-party dependency — uses only
Python stdlib + PySide6.
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QIcon, QPixmap, QFont
from PySide6.QtWidgets import (
    QDialog, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QFrame,
    QTabWidget, QSizePolicy, QSpacerItem,
)

import patient_db
import theme

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOGO_PNG     = PROJECT_ROOT / "logo" / "logo.png"
LOGO_ICO     = PROJECT_ROOT / "logo" / "logo.ico"


# ─────────────────────────────────────────────────────────────────────────────
# Small helper widgets
# ─────────────────────────────────────────────────────────────────────────────

def _field_label(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setProperty("role", "fieldLabel")
    return lbl


def _error_label() -> QLabel:
    lbl = QLabel("")
    lbl.setProperty("role", "statusError")
    lbl.setWordWrap(True)
    lbl.hide()
    return lbl


def _success_label() -> QLabel:
    lbl = QLabel("")
    lbl.setProperty("role", "statusSuccess")
    lbl.setWordWrap(True)
    lbl.hide()
    return lbl


def _password_field(placeholder: str = "Password") -> tuple[QLineEdit, QPushButton]:
    """Returns (QLineEdit, show/hide toggle button) for a password row."""
    field = QLineEdit()
    field.setPlaceholderText(placeholder)
    field.setEchoMode(QLineEdit.Password)
    field.setMinimumHeight(38)

    toggle = QPushButton("👁")
    toggle.setFixedSize(38, 38)
    toggle.setCheckable(True)
    toggle.setToolTip("Show / hide password")
    toggle.setProperty("variant", "ghost")
    toggle.setStyleSheet("font-size: 16px; border-radius: 6px;")

    def _on_toggle(checked):
        field.setEchoMode(QLineEdit.Normal if checked else QLineEdit.Password)

    toggle.toggled.connect(_on_toggle)
    return field, toggle


def _pw_row(field: QLineEdit, toggle: QPushButton) -> QHBoxLayout:
    row = QHBoxLayout()
    row.setSpacing(6)
    row.addWidget(field)
    row.addWidget(toggle)
    return row


# ─────────────────────────────────────────────────────────────────────────────
# Login tab
# ─────────────────────────────────────────────────────────────────────────────

class _LoginTab(QWidget):
    def __init__(self, dialog: "AuthDialog"):
        super().__init__()
        self._dialog = dialog
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(32, 28, 32, 28)
        layout.setSpacing(14)

        # Username
        layout.addWidget(_field_label("Username"))
        self._username = QLineEdit()
        self._username.setPlaceholderText("Enter your username")
        self._username.setMinimumHeight(38)
        layout.addWidget(self._username)

        # Password
        layout.addWidget(_field_label("Password"))
        self._pw, self._pw_toggle = _password_field("Enter your password")
        layout.addLayout(_pw_row(self._pw, self._pw_toggle))

        # Error feedback
        self._error = _error_label()
        layout.addWidget(self._error)

        layout.addSpacing(6)

        # Login button
        self._btn = QPushButton("Login")
        self._btn.setProperty("variant", "primary")
        self._btn.setMinimumHeight(42)
        self._btn.setFont(QFont("Segoe UI", 11, QFont.Bold))
        self._btn.clicked.connect(self._attempt_login)
        layout.addWidget(self._btn)

        layout.addStretch()

        # Allow Enter key to submit
        self._username.returnPressed.connect(self._attempt_login)
        self._pw.returnPressed.connect(self._attempt_login)

    def _attempt_login(self):
        self._error.hide()
        username = self._username.text().strip()
        password = self._pw.text()

        if not username or not password:
            self._show_error("Please enter both username and password.")
            return

        user = patient_db.verify_user(username, password)
        if user is None:
            self._show_error("Incorrect username or password. Please try again.")
            self._pw.clear()
            self._pw.setFocus()
            return

        self._dialog._on_login_success(user)

    def _show_error(self, msg: str):
        self._error.setText(msg)
        self._error.show()

    def focus_username(self):
        self._username.setFocus()


# ─────────────────────────────────────────────────────────────────────────────
# Register tab
# ─────────────────────────────────────────────────────────────────────────────

class _RegisterTab(QWidget):
    def __init__(self, dialog: "AuthDialog"):
        super().__init__()
        self._dialog = dialog
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(32, 28, 32, 28)
        layout.setSpacing(10)

        # Full name
        layout.addWidget(_field_label("Full Name  *"))
        self._name = QLineEdit()
        self._name.setPlaceholderText("Dr. Firstname Lastname")
        self._name.setMinimumHeight(38)
        layout.addWidget(self._name)

        # Username
        layout.addWidget(_field_label("Username  *"))
        self._username = QLineEdit()
        self._username.setPlaceholderText("Choose a username (unique)")
        self._username.setMinimumHeight(38)
        layout.addWidget(self._username)

        # Specialisation (optional)
        layout.addWidget(_field_label("Specialisation  (optional)"))
        self._spec = QLineEdit()
        self._spec.setPlaceholderText("e.g. Interventional Cardiology")
        self._spec.setMinimumHeight(38)
        layout.addWidget(self._spec)

        # Password
        layout.addWidget(_field_label("Password  *"))
        self._pw, self._pw_toggle = _password_field("At least 6 characters")
        layout.addLayout(_pw_row(self._pw, self._pw_toggle))

        # Confirm password
        layout.addWidget(_field_label("Confirm Password  *"))
        self._pw2, self._pw2_toggle = _password_field("Re-enter password")
        layout.addLayout(_pw_row(self._pw2, self._pw2_toggle))

        # Feedback labels
        self._error   = _error_label()
        self._success = _success_label()
        layout.addWidget(self._error)
        layout.addWidget(self._success)

        layout.addSpacing(6)

        # Register button
        self._btn = QPushButton("Create Account")
        self._btn.setProperty("variant", "primary")
        self._btn.setMinimumHeight(42)
        self._btn.setFont(QFont("Segoe UI", 11, QFont.Bold))
        self._btn.clicked.connect(self._attempt_register)
        layout.addWidget(self._btn)

        layout.addStretch()

        self._pw2.returnPressed.connect(self._attempt_register)

    def _attempt_register(self):
        self._error.hide()
        self._success.hide()

        full_name      = self._name.text().strip()
        username       = self._username.text().strip()
        specialisation = self._spec.text().strip()
        password       = self._pw.text()
        confirm        = self._pw2.text()

        if not full_name or not username or not password or not confirm:
            self._show_error("Please fill in all required fields (marked with *).")
            return

        if len(password) < 6:
            self._show_error("Password must be at least 6 characters.")
            return

        if password != confirm:
            self._show_error("Passwords do not match.")
            self._pw2.clear()
            self._pw2.setFocus()
            return

        try:
            patient_db.register_user(
                username=username,
                full_name=full_name,
                password=password,
                specialisation=specialisation,
            )
        except ValueError as exc:
            self._show_error(str(exc))
            return

        # Success — clear fields and nudge user to login tab
        self._pw.clear()
        self._pw2.clear()
        self._name.clear()
        self._username.clear()
        self._spec.clear()
        self._show_success(
            f"Account created for {full_name}. You can now log in."
        )
        self._dialog._on_register_success(username)

    def _show_error(self, msg: str):
        self._error.setText(msg)
        self._error.show()

    def _show_success(self, msg: str):
        self._success.setText(msg)
        self._success.show()


# ─────────────────────────────────────────────────────────────────────────────
# Main dialog
# ─────────────────────────────────────────────────────────────────────────────

class AuthDialog(QDialog):
    """
    Blocking login / register dialog.  Call exec() before showing AppWindow.
    After the dialog is accepted, read `.current_user` for the logged-in user.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_user: dict | None = None
        self.setWindowTitle("Cardexa — Sign In")
        if LOGO_ICO.exists():
            self.setWindowIcon(QIcon(str(LOGO_ICO)))
        self.setFixedSize(480, 620)
        self.setWindowFlags(Qt.Dialog | Qt.WindowTitleHint | Qt.WindowCloseButtonHint)
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Header ──────────────────────────────────────────────────────────
        header = QFrame()
        header.setObjectName("authHeader")
        header.setStyleSheet(f"""
            QFrame#authHeader {{
                background-color: {theme.SURFACE};
                border-bottom: 1px solid {theme.BORDER};
            }}
        """)
        header_layout = QVBoxLayout(header)
        header_layout.setContentsMargins(32, 28, 32, 24)
        header_layout.setSpacing(6)

        if LOGO_PNG.exists():
            logo_lbl = QLabel()
            logo_lbl.setAlignment(Qt.AlignCenter)
            logo_lbl.setPixmap(
                QPixmap(str(LOGO_PNG)).scaledToWidth(160, Qt.SmoothTransformation)
            )
            header_layout.addWidget(logo_lbl)
        else:
            brand = QLabel("Cardexa")
            brand.setAlignment(Qt.AlignCenter)
            brand.setStyleSheet(f"color: {theme.ACCENT}; font-size: 28px; font-weight: 800;")
            header_layout.addWidget(brand)

        subtitle = QLabel("Clinical AI Dashboard  ·  Cardiologist Access")
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setProperty("role", "pageSubtitle")
        header_layout.addWidget(subtitle)

        root.addWidget(header)

        # ── Tabs ─────────────────────────────────────────────────────────────
        self._tabs = QTabWidget()
        self._tabs.setStyleSheet(f"""
            QTabWidget::pane {{
                border: none;
                background-color: {theme.BG};
            }}
            QTabBar::tab {{
                background-color: {theme.SURFACE};
                color: {theme.ASH};
                padding: 10px 28px;
                font-size: 13px;
                font-weight: 600;
                border: none;
                border-bottom: 2px solid transparent;
            }}
            QTabBar::tab:selected {{
                color: {theme.ACCENT};
                border-bottom: 2px solid {theme.ACCENT};
                background-color: {theme.BG};
            }}
            QTabBar::tab:hover:!selected {{
                color: {theme.WHITE};
            }}
        """)

        self._login_tab    = _LoginTab(self)
        self._register_tab = _RegisterTab(self)

        self._tabs.addTab(self._login_tab,    "  Login  ")
        self._tabs.addTab(self._register_tab, "  Register  ")
        self._tabs.currentChanged.connect(self._on_tab_changed)
        root.addWidget(self._tabs, stretch=1)

        # ── Footer ───────────────────────────────────────────────────────────
        footer = QFrame()
        footer.setStyleSheet(f"""
            QFrame {{
                background-color: {theme.SURFACE};
                border-top: 1px solid {theme.BORDER};
            }}
        """)
        footer_layout = QHBoxLayout(footer)
        footer_layout.setContentsMargins(24, 12, 24, 12)
        disclaimer = QLabel("Research system — not for independent clinical diagnosis.")
        disclaimer.setProperty("role", "hint")
        disclaimer.setAlignment(Qt.AlignCenter)
        footer_layout.addWidget(disclaimer)
        root.addWidget(footer)

        # Focus username on open
        self._login_tab.focus_username()

    def _on_tab_changed(self, index: int):
        if index == 0:
            self._login_tab.focus_username()

    def _on_login_success(self, user: dict):
        self.current_user = user
        self.accept()

    def _on_register_success(self, username: str):
        """Switch to login tab after successful registration."""
        self._tabs.setCurrentIndex(0)
