"""
ECG Analysis page.

Lets the analyst pick a patient case, load a 12-lead ECG image, run the
composite ECG pipeline (segmentation -> digitization -> territory-based
diagnosis -> ML classification), and save the result into that case's
ecg_results/ folder. Once the case has both a saved ECG result and a saved
QCA angiogram view, a Compare-with-Angiogram section becomes available.
"""
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QPixmap, QFont, QColor
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QPushButton,
    QFrame, QFileDialog, QMessageBox, QSizePolicy, QScrollArea, QListWidget,
    QListWidgetItem
)

import patient_store
import analysis_results_store
import ecg_qca_concordance
import ecg_results_store

# angio-ai/src/clinical_app/ecg_analysis_page.py -> parents[2] == angio-ai/
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_ECG_SRC_12LEAD = _PROJECT_ROOT / "ECG" / "src" / "src_12lead"
if str(_ECG_SRC_12LEAD) not in sys.path:
    sys.path.insert(0, str(_ECG_SRC_12LEAD))

from pipeline_composite import run_pipeline, _DEFAULT_MODEL  # noqa: E402

VERDICT_COLORS = {
    "CONCORDANT": "#2ecc71",
    "PARTIALLY CONCORDANT": "#f1c40f",
    "DISCORDANT": "#e74c3c",
    "INCONCLUSIVE": "#7f8c8d",
    "UNCONFIRMED_BY_ECG": "#e67e22",
    "UNMAPPED": "#7f8c8d",
}


def _card() -> QFrame:
    frame = QFrame()
    frame.setProperty("card", "true")
    return frame


def _field_label(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setProperty("role", "fieldLabel")
    return lbl


class _EcgAnalysisThread(QThread):
    """
    Runs the composite ECG pipeline (segment -> digitize -> diagnose ->
    classify) off the UI thread -- it loads a model checkpoint and processes
    a full-resolution image, which is too slow to block the UI.
    """
    finished_ok = Signal(object)  # report dict
    error = Signal(str)

    def __init__(self, image_path: str, output_dir: str):
        super().__init__()
        self.image_path = image_path
        self.output_dir = output_dir

    def run(self):
        try:
            report = run_pipeline(self.image_path, model_path=str(_DEFAULT_MODEL),
                                  output_dir=self.output_dir)
            self.finished_ok.emit(report)
        except Exception as e:
            self.error.emit(str(e))


class EcgAnalysisPage(QWidget):
    """ECG analysis page: case picker + load ECG image + analyze/save + compare with angiogram."""

    go_to_new_patient = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._cases = []
        self._image_path = None
        self._analysis_thread = None
        self._pending_case_id = None
        self._pending_view_dir = None
        self._last_report = None
        self._last_report_dir = None

        self._build_ui()
        self.refresh_cases()

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

        title = QLabel("ECG Analysis")
        title.setProperty("role", "pageTitle")
        layout.addWidget(title)

        subtitle = QLabel(
            "Choose a patient case, load an ECG image, and save the analysis."
        )
        subtitle.setProperty("role", "pageSubtitle")
        subtitle.setWordWrap(True)
        layout.addWidget(subtitle)

        layout.addWidget(self._build_selection_card())
        layout.addWidget(self._build_viewer_card())
        layout.addWidget(self._build_save_report_card())
        layout.addWidget(self._build_compare_card())

    def _build_selection_card(self) -> QFrame:
        card = _card()
        v = QVBoxLayout(card)
        v.setContentsMargins(20, 16, 20, 16)
        v.setSpacing(10)

        header = QLabel("CASE & ECG IMAGE")
        header.setProperty("role", "sectionHeader")
        v.addWidget(header)

        row = QHBoxLayout()
        row.addWidget(_field_label("Patient Case"))
        self.combo_case = QComboBox()
        self.combo_case.setMinimumWidth(320)
        self.combo_case.currentIndexChanged.connect(self._on_case_selected)
        row.addWidget(self.combo_case, stretch=1)

        self.btn_refresh_cases = QPushButton("Refresh")
        self.btn_refresh_cases.setProperty("variant", "ghost")
        self.btn_refresh_cases.clicked.connect(self.refresh_cases)
        row.addWidget(self.btn_refresh_cases)

        self.btn_goto_new_patient = QPushButton("+ New Patient")
        self.btn_goto_new_patient.clicked.connect(self.go_to_new_patient.emit)
        row.addWidget(self.btn_goto_new_patient)
        v.addLayout(row)

        btn_row = QHBoxLayout()
        self.btn_load_image = QPushButton("Load ECG Image...")
        self.btn_load_image.setProperty("variant", "primary")
        self.btn_load_image.clicked.connect(self._browse_ecg_image)
        btn_row.addWidget(self.btn_load_image)

        self.btn_clear_image = QPushButton("Clear")
        self.btn_clear_image.setProperty("variant", "ghost")
        self.btn_clear_image.setVisible(False)
        self.btn_clear_image.clicked.connect(self._clear_loaded_image)
        btn_row.addWidget(self.btn_clear_image)

        self.lbl_image_status = QLabel("No ECG image loaded.")
        self.lbl_image_status.setProperty("role", "hint")
        btn_row.addWidget(self.lbl_image_status, stretch=1)
        v.addLayout(btn_row)

        return card

    def _build_viewer_card(self) -> QFrame:
        card = _card()
        v = QVBoxLayout(card)
        v.setContentsMargins(20, 16, 20, 16)
        v.setSpacing(10)

        header = QLabel("ECG PREVIEW & RESULT")
        header.setProperty("role", "sectionHeader")
        v.addWidget(header)

        panels = QHBoxLayout()
        panels.setSpacing(10)

        self.panel_source, self.label_source = self._build_image_panel("Source ECG")
        panels.addWidget(self.panel_source, stretch=1)

        self.panel_result, self.label_result = self._build_image_panel("Result Card")
        panels.addWidget(self.panel_result, stretch=1)

        v.addLayout(panels, stretch=1)

        self.lbl_findings = QLabel("No analysis yet — load an ECG image, then press Analyze & Save.")
        self.lbl_findings.setFont(QFont("Consolas", 10))
        self.lbl_findings.setWordWrap(True)
        self.lbl_findings.setProperty("role", "hint")
        v.addWidget(self.lbl_findings)

        return card

    def _build_image_panel(self, title: str):
        box = _card()
        v = QVBoxLayout(box)
        v.setContentsMargins(10, 10, 10, 10)
        v.setSpacing(6)

        header = QLabel(title)
        header.setProperty("role", "fieldLabel")
        v.addWidget(header)

        label = QLabel("No image loaded")
        label.setAlignment(Qt.AlignCenter)
        label.setMinimumSize(300, 260)
        label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        label.setStyleSheet("background-color: #0B0C0E; border-radius: 6px; color: #5C6067;")
        v.addWidget(label, stretch=1)

        return box, label

    def _build_save_report_card(self) -> QFrame:
        card = _card()
        v = QVBoxLayout(card)
        v.setContentsMargins(20, 16, 20, 16)
        v.setSpacing(10)

        header = QLabel("ANALYZE & SAVE")
        header.setProperty("role", "sectionHeader")
        v.addWidget(header)

        hint = QLabel("Analyze the ECG and save the result with this patient case.")
        hint.setProperty("role", "hint")
        hint.setWordWrap(True)
        v.addWidget(hint)

        row1 = QHBoxLayout()
        self.btn_analyze_save = QPushButton("Analyze && Save ECG Report")
        self.btn_analyze_save.setProperty("variant", "primary")
        self.btn_analyze_save.clicked.connect(self._analyze_and_save)
        row1.addWidget(self.btn_analyze_save)

        self.btn_open_report = QPushButton("Open Report")
        self.btn_open_report.setVisible(False)
        self.btn_open_report.clicked.connect(self._open_report)
        row1.addWidget(self.btn_open_report)
        row1.addStretch()
        v.addLayout(row1)

        self.lbl_save_status = QLabel("Select a case and load an ECG image first.")
        self.lbl_save_status.setProperty("role", "hint")
        self.lbl_save_status.setWordWrap(True)
        v.addWidget(self.lbl_save_status)

        self.lbl_ecg_results_status = QLabel("No ECG results saved yet for this case.")
        self.lbl_ecg_results_status.setProperty("role", "hint")
        self.lbl_ecg_results_status.setWordWrap(True)
        v.addWidget(self.lbl_ecg_results_status)

        self.list_ecg_results = QListWidget()
        self.list_ecg_results.setMinimumHeight(90)
        v.addWidget(self.list_ecg_results)

        del_row = QHBoxLayout()
        self.btn_delete_ecg_result = QPushButton("Delete Selected Result")
        self.btn_delete_ecg_result.setProperty("variant", "ghost")
        self.btn_delete_ecg_result.clicked.connect(self._delete_selected_ecg_result)
        del_row.addWidget(self.btn_delete_ecg_result)
        del_row.addStretch()
        v.addLayout(del_row)

        return card

    def _build_compare_card(self) -> QFrame:
        card = _card()
        v = QVBoxLayout(card)
        v.setContentsMargins(20, 16, 20, 16)
        v.setSpacing(10)

        header = QLabel("COMPARE WITH ANGIOGRAM")
        header.setProperty("role", "sectionHeader")
        v.addWidget(header)

        hint = QLabel("Compare the ECG result with saved angiogram findings for this case.")
        hint.setProperty("role", "hint")
        hint.setWordWrap(True)
        v.addWidget(hint)

        self.btn_compare = QPushButton("Compare ECG vs. Angiogram Results")
        self.btn_compare.setEnabled(False)
        self.btn_compare.clicked.connect(self._run_comparison)
        v.addWidget(self.btn_compare)

        self.lbl_compare_status = QLabel(
            "Save an ECG result and an angiogram view for this case to enable comparison."
        )
        self.lbl_compare_status.setProperty("role", "hint")
        self.lbl_compare_status.setWordWrap(True)
        v.addWidget(self.lbl_compare_status)

        self.list_compare = QListWidget()
        self.list_compare.setMinimumHeight(140)
        self.list_compare.setWordWrap(True)
        self.list_compare.setVisible(False)
        v.addWidget(self.list_compare)

        return card

    # ── Case selection ─────────────────────────────────────────────
    def refresh_cases(self):
        self._cases = patient_store.list_cases()
        self.combo_case.blockSignals(True)
        self.combo_case.clear()
        for case in self._cases:
            label = f"{case.get('patient_id', '?')} — {case.get('full_name', '?')} ({case.get('study_date', '?')})"
            self.combo_case.addItem(label)
        self.combo_case.blockSignals(False)

        if self._cases:
            self._on_case_selected(0)
        else:
            self.lbl_save_status.setText("No patient cases yet. Click \"+ New Patient\" to create one.")
            self._refresh_ecg_results_status()
            self._refresh_compare_status()

    def select_case_by_id(self, case_id: str):
        """Refreshes the case list and selects the given case, mirroring
        LocalDicomAnalysisPage.select_case_by_id."""
        self.refresh_cases()
        for i, case in enumerate(self._cases):
            if case["case_id"] == case_id:
                self.combo_case.setCurrentIndex(i)
                return

    def _current_case(self):
        row = self.combo_case.currentIndex()
        if row < 0 or row >= len(self._cases):
            return None
        return self._cases[row]

    def _on_case_selected(self, index: int):
        self._refresh_ecg_results_status()
        self._refresh_compare_status()

    # ── Load ECG image ─────────────────────────────────────────────
    def _browse_ecg_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select ECG Image", "", "Image files (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"
        )
        if not path:
            return
        self._image_path = path
        self.lbl_image_status.setText(f"Loaded: {Path(path).name}")
        self.btn_clear_image.setVisible(True)
        pix = QPixmap(path)
        if not pix.isNull():
            self.label_source.setPixmap(
                pix.scaled(self.label_source.width(), self.label_source.height(),
                          Qt.KeepAspectRatio, Qt.SmoothTransformation)
            )
        self.label_result.setText("Not analyzed yet")
        self.btn_open_report.setVisible(False)
        self.lbl_save_status.setText("Ready to analyze.")

    def _clear_loaded_image(self):
        self._image_path = None
        self.btn_clear_image.setVisible(False)
        self.lbl_image_status.setText("No ECG image loaded.")
        self.label_source.setText("No image loaded")
        self.label_source.setPixmap(QPixmap())
        self.label_result.setText("No image loaded")
        self.label_result.setPixmap(QPixmap())
        self.btn_open_report.setVisible(False)
        self.lbl_findings.setText("No analysis yet — load an ECG image, then press Analyze & Save.")
        self.lbl_save_status.setText("Select a case and load an ECG image first.")

    # ── Analyze & Save ─────────────────────────────────────────────
    def _analyze_and_save(self):
        if self._analysis_thread is not None and self._analysis_thread.isRunning():
            return
        if not self._image_path:
            self.lbl_save_status.setText("Load an ECG image first.")
            return
        case = self._current_case()
        if case is None:
            self.lbl_save_status.setText("Select a patient case first.")
            return

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        label = f"{Path(self._image_path).stem}_{ts}"
        ecg_dir = patient_store.get_case_ecg_dir(case["case_id"])
        view_dir = ecg_dir / label
        view_dir.mkdir(parents=True, exist_ok=True)

        try:
            shutil.copy2(self._image_path, view_dir / f"source_ecg{Path(self._image_path).suffix}")
        except OSError:
            pass

        self._pending_case_id = case["case_id"]
        self._pending_view_dir = view_dir
        self.btn_analyze_save.setEnabled(False)
        self.btn_open_report.setVisible(False)
        self.lbl_save_status.setText(f"Analyzing '{Path(self._image_path).name}'...")

        self._analysis_thread = _EcgAnalysisThread(self._image_path, str(view_dir))
        self._analysis_thread.finished_ok.connect(self._on_analysis_finished)
        self._analysis_thread.error.connect(self._on_analysis_error)
        self._analysis_thread.start()

    def _on_analysis_finished(self, report: dict):
        self.btn_analyze_save.setEnabled(True)
        self._last_report = report
        self._last_report_dir = self._pending_view_dir

        diag = report.get("diagnosis") or {}
        cls = report.get("classification") or {}
        lines = [
            f"Severity: {diag.get('severity', '?')}   Rhythm: {diag.get('rhythm', '?')}   "
            f"Axis: {diag.get('axis', '?')}",
            f"Classifier: {cls.get('label', '?')} ({cls.get('confidence', 0) * 100:.1f}%)",
        ]
        for finding in diag.get("findings", [])[:6]:
            lines.append(f"  • {finding}")
        self.lbl_findings.setText("\n".join(lines))

        result_card = self._pending_view_dir / "result_card.png"
        if result_card.exists():
            pix = QPixmap(str(result_card))
            if not pix.isNull():
                self.label_result.setPixmap(
                    pix.scaled(self.label_result.width(), self.label_result.height(),
                              Qt.KeepAspectRatio, Qt.SmoothTransformation)
                )

        self.lbl_save_status.setText(f"Saved to {self._pending_view_dir}")
        self.btn_open_report.setVisible(True)
        self._refresh_ecg_results_status()
        self._refresh_compare_status()

    def _on_analysis_error(self, message: str):
        self.btn_analyze_save.setEnabled(True)
        self.lbl_save_status.setText(f"Analysis failed: {message}")
        QMessageBox.warning(self, "ECG Analysis Failed", message)

    def _open_report(self):
        if self._last_report_dir and hasattr(os, "startfile"):
            report_txt = self._last_report_dir / "diagnosis_report.txt"
            try:
                os.startfile(report_txt if report_txt.exists() else self._last_report_dir)
            except OSError:
                pass

    def _refresh_ecg_results_status(self):
        self.list_ecg_results.clear()
        case = self._current_case()
        if case is None:
            self.lbl_ecg_results_status.setText("No ECG results saved yet for this case.")
            return
        ecg_dir = patient_store.get_case_ecg_dir(case["case_id"])
        results = ecg_results_store.list_ecg_results(ecg_dir)
        if not results:
            self.lbl_ecg_results_status.setText("No ECG results saved yet for this case.")
            return

        self.lbl_ecg_results_status.setText(f"{len(results)} ECG result(s) saved for this case:")
        for r in results:
            diag = r.get("diagnosis") or {}
            cls = r.get("classification") or {}
            name = Path(r.get("image_path", "?")).stem
            text = f"{name} — {diag.get('severity', '?')} — {cls.get('label', '?')}"
            item = QListWidgetItem(text)
            item.setData(Qt.UserRole, str(r["_view_dir"]))
            self.list_ecg_results.addItem(item)

    def _delete_selected_ecg_result(self):
        item = self.list_ecg_results.currentItem()
        if item is None:
            QMessageBox.information(self, "Delete ECG Result", "Select a saved ECG result from the list first.")
            return

        view_dir = Path(item.data(Qt.UserRole))
        proceed = QMessageBox.question(
            self, "Delete This ECG Result?",
            f"Permanently delete the saved ECG result '{item.text()}'?\n\n"
            f"This removes {view_dir} and everything in it (report, result card, source image). "
            "This cannot be undone.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        if proceed != QMessageBox.Yes:
            return

        ecg_results_store.delete_ecg_result(view_dir)
        if self._last_report_dir == view_dir:
            self._last_report_dir = None
            self._last_report = None
            self.btn_open_report.setVisible(False)
        self._refresh_ecg_results_status()
        self._refresh_compare_status()

    # ── Compare with Angiogram ──────────────────────────────────────
    def _refresh_compare_status(self):
        case = self._current_case()
        self.list_compare.setVisible(False)
        self.list_compare.clear()
        if case is None:
            self.btn_compare.setEnabled(False)
            self.lbl_compare_status.setText("Select a patient case first.")
            return

        ecg_results = ecg_results_store.list_ecg_results(patient_store.get_case_ecg_dir(case["case_id"]))
        view_results = analysis_results_store.list_view_results(
            patient_store.get_case_analysis_dir(case["case_id"])
        )

        if not ecg_results or not view_results:
            self.btn_compare.setEnabled(False)
            self.lbl_compare_status.setText(
                f"Need at least one saved ECG result ({len(ecg_results)} saved) and one saved "
                f"angiogram view ({len(view_results)} saved) for this case to compare."
            )
            return

        if not ecg_qca_concordance.has_any_known_artery(view_results):
            self.btn_compare.setEnabled(False)
            self.lbl_compare_status.setText(
                "This case's saved angiogram view(s) have no identified vessel (no localization "
                "model was loaded during DICOM analysis) — comparison needs at least one lesion "
                "with a known artery. Re-analyze the angiogram with a localization model loaded."
            )
            return

        self.btn_compare.setEnabled(True)
        self.lbl_compare_status.setText(
            f"{len(ecg_results)} ECG result(s), {len(view_results)} angiogram view(s) available."
        )

    def _run_comparison(self):
        case = self._current_case()
        if case is None:
            return
        ecg_results = ecg_results_store.list_ecg_results(patient_store.get_case_ecg_dir(case["case_id"]))
        view_results = analysis_results_store.list_view_results(
            patient_store.get_case_analysis_dir(case["case_id"])
        )
        if not ecg_results or not view_results:
            return

        # Compare the most recently saved ECG result against every saved angiogram view.
        latest_ecg = max(ecg_results, key=lambda r: r.get("timestamp", ""))
        verdicts = ecg_qca_concordance.compare(latest_ecg, view_results)

        self.list_compare.clear()
        self.list_compare.setVisible(True)
        if not verdicts:
            self.list_compare.addItem(QListWidgetItem("No ECG territories or angiogram lesions to compare."))
            return

        for v in verdicts:
            territory = v["territory"] or "-"
            vessel = v["vessel"] or "-"
            text = f"[{v['verdict']}]  territory={territory}  vessel={vessel}\n{v['detail']}"
            item = QListWidgetItem(text)
            color_hex = VERDICT_COLORS.get(v["verdict"])
            if color_hex:
                item.setForeground(QColor(color_hex))
            self.list_compare.addItem(item)

    # ── Cleanup ──────────────────────────────────────────────────────
    def shutdown(self):
        if self._analysis_thread is not None and self._analysis_thread.isRunning():
            self._analysis_thread.wait(3000)
