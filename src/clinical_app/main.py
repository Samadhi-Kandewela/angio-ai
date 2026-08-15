"""
Entry point for the Angio-AI Clinical Dashboard (PySide6).

Run with:
    python src/clinical_app/main.py

This is a new, standalone application -- it does not modify or replace
desktop_app_qca.py / demo_app.py, which remain available as-is.
"""
import os
import sys

# !! CRITICAL: import matplotlib before PySide6 !!
# qca.py (used by the DICOM analysis page) imports matplotlib, whose import
# chain pulls in dateutil -> six.moves. PySide6's shiboken signature-loader
# installs an import hook that crashes on six's lazy module proxies
# (AttributeError: '_SixMetaPathImporter' object has no attribute '_path')
# if PySide6 is imported first. Importing matplotlib here, before PySide6,
# avoids the conflict. Must be matplotlib.pyplot specifically -- that's what
# pulls in the dateutil/six chain that trips the hook.
import matplotlib.pyplot  # noqa: F401

APP_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(APP_DIR)  # angio-ai/src
for _p in (APP_DIR, SRC_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QFont

import theme
import patient_db
from app_window import AppWindow
from auth_dialog import AuthDialog


def _show_auth() -> dict | None:
    """
    Shows the login/register dialog.
    Returns the logged-in user dict on success, or None if the dialog
    was dismissed without logging in.
    """
    auth = AuthDialog()
    if auth.exec() != AuthDialog.Accepted or auth.current_user is None:
        return None
    return auth.current_user


def main():
    # Rebuild the SQLite case index from disk on every launch so it stays
    # in sync with the actual patient_data/ folders.
    patient_db.rebuild_from_disk()

    app = QApplication(sys.argv)
    app.setFont(QFont("Segoe UI", 10))
    app.setStyleSheet(theme.build_stylesheet())

    # ── Authentication loop ──────────────────────────────────────────────────
    # After a logout the loop re-shows the auth dialog so a different
    # clinician can log in without restarting the application.
    while True:
        current_user = _show_auth()
        if current_user is None:
            # User closed the dialog without logging in — exit.
            sys.exit(0)

        window = AppWindow(current_user=current_user)

        # Track whether the window was closed via Logout (vs the X button).
        _logged_out = []
        window.logout_requested.connect(lambda: _logged_out.append(True))

        window.showMaximized()
        app.exec()  # blocks until window closes

        if not _logged_out:
            # Closed via the X button — exit normally.
            sys.exit(0)
        # Otherwise, loop back and show the auth dialog again.


if __name__ == "__main__":
    main()
