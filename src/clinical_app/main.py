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
SRC_DIR = os.path.dirname(APP_DIR)  # angio-ai/src -- qca.py, frame_pipeline.py, localization.py live here
for _p in (APP_DIR, SRC_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QFont

import theme
import patient_db
from app_window import AppWindow


def main():
    # The SQLite case index is just a disposable, rebuildable cache over each
    # case's real metadata.json -- rebuilding it on every launch keeps it
    # always in sync with what's actually on disk (including cases created
    # before this feature existed, or a patients.db that got deleted/corrupted).
    patient_db.rebuild_from_disk()

    app = QApplication(sys.argv)
    app.setFont(QFont("Segoe UI", 10))
    app.setStyleSheet(theme.build_stylesheet())

    window = AppWindow()
    window.showMaximized()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
