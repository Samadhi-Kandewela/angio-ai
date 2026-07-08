"""
Patient Records page.

Browse, search, and inspect every patient case on file. Backed by the
SQLite case index (patient_db.py, via patient_store.list_cases()) for fast
querying -- each case's own metadata.json remains the authoritative record,
this page just gives a searchable table over all of them plus quick access
to a case's folder or its DICOM Analysis workflow.
"""
import os

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QFrame, QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView,
    QScrollArea
)

import patient_store

# (metadata/row key, column header)
COLUMNS = [
    ("patient_id", "Patient ID"),
    ("full_name", "Full Name"),
    ("age", "Age"),
    ("gender", "Gender"),
    ("study_date", "Study Date"),
    ("indication", "Indication"),
    ("risk_factors", "Risk Factors"),
    ("dicom_file_count", "DICOM Files"),
    ("created_at", "Created"),
]


def _card() -> QFrame:
    frame = QFrame()
    frame.setProperty("card", "true")
    return frame


def _field_label(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setProperty("role", "fieldLabel")
    return lbl


class PatientRecordsPage(QWidget):
    """Searchable table of every patient case, with a details panel for the current selection."""

    go_to_dicom_analysis = Signal(str)  # case_id

    def __init__(self, parent=None):
        super().__init__(parent)
        self._cases = []
        self._build_ui()
        self.refresh()

    # ── UI construction ────────────────────────────────────────────
    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        outer.addWidget(scroll)

        content = QWidget()
        scroll.setWidget(content)

        layout = QVBoxLayout(content)
        layout.setContentsMargins(28, 24, 28, 20)
        layout.setSpacing(14)

        title = QLabel("Patient Records")
        title.setProperty("role", "pageTitle")
        layout.addWidget(title)

        subtitle = QLabel(
            "Search and browse every patient case on file. Select a row to see its full "
            "record, open its folder, or jump straight to DICOM analysis."
        )
        subtitle.setProperty("role", "pageSubtitle")
        subtitle.setWordWrap(True)
        layout.addWidget(subtitle)

        layout.addWidget(self._build_search_card())
        layout.addWidget(self._build_table_card(), stretch=1)
        layout.addWidget(self._build_details_card())

    def _build_search_card(self) -> QFrame:
        card = _card()
        h = QHBoxLayout(card)
        h.setContentsMargins(20, 14, 20, 14)
        h.setSpacing(10)

        h.addWidget(_field_label("Search"))
        self.txt_search = QLineEdit()
        self.txt_search.setPlaceholderText("Patient ID, name, or indication...")
        self.txt_search.textChanged.connect(self._on_search_changed)
        h.addWidget(self.txt_search, stretch=1)

        self.btn_refresh = QPushButton("Refresh")
        self.btn_refresh.setProperty("variant", "ghost")
        self.btn_refresh.clicked.connect(self.refresh)
        h.addWidget(self.btn_refresh)

        self.lbl_count = QLabel("0 case(s)")
        self.lbl_count.setProperty("role", "hint")
        h.addWidget(self.lbl_count)

        return card

    def _build_table_card(self) -> QFrame:
        card = _card()
        v = QVBoxLayout(card)
        v.setContentsMargins(20, 16, 20, 16)
        v.setSpacing(10)

        header = QLabel("ALL PATIENT CASES")
        header.setProperty("role", "sectionHeader")
        v.addWidget(header)

        self.table = QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels([label for _, label in COLUMNS])
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.table.setMinimumHeight(280)
        self.table.itemSelectionChanged.connect(self._on_row_selected)
        v.addWidget(self.table, stretch=1)

        return card

    def _build_details_card(self) -> QFrame:
        card = _card()
        v = QVBoxLayout(card)
        v.setContentsMargins(20, 16, 20, 16)
        v.setSpacing(10)

        header = QLabel("CASE DETAILS")
        header.setProperty("role", "sectionHeader")
        v.addWidget(header)

        self.lbl_details = QLabel("Select a case above to see its full record.")
        self.lbl_details.setProperty("role", "hint")
        self.lbl_details.setWordWrap(True)
        self.lbl_details.setTextFormat(Qt.RichText)
        v.addWidget(self.lbl_details)

        btn_row = QHBoxLayout()
        self.btn_open_folder = QPushButton("Open Case Folder")
        self.btn_open_folder.setEnabled(False)
        self.btn_open_folder.clicked.connect(self._open_case_folder)
        btn_row.addWidget(self.btn_open_folder)

        self.btn_go_analysis = QPushButton("Go to DICOM Analysis")
        self.btn_go_analysis.setProperty("variant", "primary")
        self.btn_go_analysis.setEnabled(False)
        self.btn_go_analysis.clicked.connect(self._go_to_analysis)
        btn_row.addWidget(self.btn_go_analysis)
        btn_row.addStretch()
        v.addLayout(btn_row)

        return card

    # ── Data ─────────────────────────────────────────────────────────
    def refresh(self):
        self._load(self.txt_search.text().strip())

    def _on_search_changed(self, text):
        self._load(text.strip())

    def _load(self, search):
        self._cases = patient_store.list_cases(search=search or None)
        self.table.setRowCount(len(self._cases))
        for r, case in enumerate(self._cases):
            for c, (key, _label) in enumerate(COLUMNS):
                value = case.get(key)
                text = "" if value is None else str(value)
                self.table.setItem(r, c, QTableWidgetItem(text))
        self.table.resizeColumnsToContents()
        self.lbl_count.setText(f"{len(self._cases)} case(s)")
        self.lbl_details.setText("Select a case above to see its full record.")
        self.btn_open_folder.setEnabled(False)
        self.btn_go_analysis.setEnabled(False)

    def _current_case_id(self):
        row = self.table.currentRow()
        if row < 0 or row >= len(self._cases):
            return None
        return self._cases[row]["case_id"]

    def _on_row_selected(self):
        case_id = self._current_case_id()
        if case_id is None:
            return
        metadata = patient_store.load_metadata(case_id)
        self._show_details(case_id, metadata)
        self.btn_open_folder.setEnabled(True)
        self.btn_go_analysis.setEnabled(True)

    def _show_details(self, case_id, metadata):
        risk_factors = metadata.get("risk_factors") or []
        risk_str = ", ".join(risk_factors) if risk_factors else "None recorded"
        dicom_sources = metadata.get("dicom_sources") or []
        sources_str = ", ".join(s.get("name", "?") for s in dicom_sources) or "—"

        html = (
            f"<b>Case ID:</b> {case_id}<br>"
            f"<b>Patient:</b> {metadata.get('full_name') or '—'} ({metadata.get('patient_id') or '—'})<br>"
            f"<b>Age / Gender:</b> {metadata.get('age', '—')} / {metadata.get('gender') or '—'}<br>"
            f"<b>Height / Weight:</b> {metadata.get('height_cm') or '—'} cm / {metadata.get('weight_kg') or '—'} kg<br>"
            f"<b>Study Date:</b> {metadata.get('study_date') or '—'}<br>"
            f"<b>Operator:</b> {metadata.get('operator') or '—'}<br>"
            f"<b>Referring Physician:</b> {metadata.get('referring_physician') or '—'}<br>"
            f"<b>Indication:</b> {metadata.get('indication') or '—'}<br>"
            f"<b>Risk Factors:</b> {risk_str}<br>"
            f"<b>Notes:</b> {metadata.get('notes') or '—'}<br>"
            f"<b>DICOM Source(s):</b> {sources_str} ({metadata.get('dicom_file_count', 0)} file(s))<br>"
            f"<b>Created:</b> {metadata.get('created_at') or '—'}"
        )
        self.lbl_details.setText(html)

    def _open_case_folder(self):
        case_id = self._current_case_id()
        if case_id is None:
            return
        case_dir = patient_store.get_case_dir(case_id)
        if hasattr(os, "startfile"):
            try:
                os.startfile(case_dir)
            except OSError:
                pass

    def _go_to_analysis(self):
        case_id = self._current_case_id()
        if case_id is not None:
            self.go_to_dicom_analysis.emit(case_id)
