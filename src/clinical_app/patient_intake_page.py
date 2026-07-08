"""
Patient Intake page -- the dashboard's home page.

Collects patient demographics, study details, and clinical risk-factor
history, attaches the DICOM file(s) for the study, and on submit creates
a dedicated case folder (metadata.json + dicom/ + analysis_results/) via
patient_store.create_patient_case().
"""
import os

from PySide6.QtCore import Qt, QDate, QThread, Signal, QEvent, QTimer, QObject
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QLineEdit,
    QComboBox, QSpinBox, QDoubleSpinBox, QDateEdit, QTextEdit, QCheckBox,
    QPushButton, QFrame, QFileDialog, QListWidget, QListWidgetItem,
    QScrollArea, QSizePolicy, QMessageBox, QProgressBar
)

from patient_store import create_patient_case

RISK_FACTORS = [
    "Hypertension", "Diabetes Mellitus", "Dyslipidemia", "Current Smoker",
    "Family History of CAD", "Prior MI", "Prior PCI", "Prior CABG",
    "Chronic Kidney Disease",
]

INDICATIONS = [
    "Stable Angina", "Unstable Angina / NSTEMI", "STEMI",
    "Positive Stress Test", "Pre-operative Evaluation",
    "Valve Disease Workup", "Routine Surveillance", "Other",
]


class _SelectAllOnFocus(QObject):
    """
    Event filter for numeric spin boxes: selects the field's current value
    as soon as it gains focus (click or Tab), so the next keystroke
    overwrites the default "0" instead of requiring a manual select-all
    first. Queued via QTimer.singleShot(0, ...) so it runs after the click
    itself has placed the cursor, rather than being immediately undone by it.
    """
    def eventFilter(self, obj, event):
        if event.type() == QEvent.FocusIn:
            QTimer.singleShot(0, obj.selectAll)
        return False


def _card() -> QFrame:
    frame = QFrame()
    frame.setProperty("card", "true")
    return frame


def _field_label(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setProperty("role", "fieldLabel")
    return lbl


class _CaseCreationThread(QThread):
    """
    Runs create_patient_case() off the UI thread -- real DICOM study exports
    (a whole CD/DVD folder, not a single file) can be hundreds of MB, and
    copying that synchronously would freeze the form.
    """
    progress = Signal(str)
    finished_ok = Signal(object)  # PatientCase
    error = Signal(str)

    def __init__(self, metadata, dicom_paths):
        super().__init__()
        self.metadata = metadata
        self.dicom_paths = dicom_paths

    def run(self):
        try:
            case = create_patient_case(
                self.metadata, self.dicom_paths,
                progress_cb=lambda name: self.progress.emit(f"Copying {name}..."),
            )
            self.finished_ok.emit(case)
        except OSError as e:
            self.error.emit(str(e))


class PatientIntakePage(QWidget):
    """Home page: new-patient-case form. Emits case_created(PatientCase) on success."""

    case_created = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._dicom_paths = []
        self._case_thread = None
        self._select_all_filter = _SelectAllOnFocus(self)
        self._build_ui()

    # ── UI construction ────────────────────────────────────────────
    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        outer.addWidget(scroll)

        content = QWidget()
        scroll.setWidget(content)

        layout = QVBoxLayout(content)
        layout.setContentsMargins(36, 32, 36, 32)
        layout.setSpacing(18)

        title = QLabel("New Patient Case")
        title.setProperty("role", "pageTitle")
        layout.addWidget(title)

        subtitle = QLabel(
            "Add the patient details, attach the angiogram files, and create a new case."
        )
        subtitle.setProperty("role", "pageSubtitle")
        subtitle.setWordWrap(True)
        layout.addWidget(subtitle)

        layout.addWidget(self._build_patient_card())
        layout.addWidget(self._build_study_card())
        layout.addWidget(self._build_history_card())
        layout.addWidget(self._build_imaging_card())
        layout.addWidget(self._build_action_bar())

        layout.addStretch()

    def _build_patient_card(self) -> QFrame:
        card = _card()
        v = QVBoxLayout(card)
        v.setContentsMargins(20, 18, 20, 18)
        v.setSpacing(12)

        header = QLabel("PATIENT INFORMATION")
        header.setProperty("role", "sectionHeader")
        v.addWidget(header)

        grid = QGridLayout()
        grid.setHorizontalSpacing(16)
        grid.setVerticalSpacing(4)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(3, 1)

        self.txt_patient_id = QLineEdit()
        self.txt_patient_id.setPlaceholderText("e.g. CAD65887")
        self.txt_full_name = QLineEdit()
        self.txt_full_name.setPlaceholderText("Patient full name")
        self.spin_age = QSpinBox()
        self.spin_age.setRange(0, 120)
        self.spin_age.setSuffix(" yrs")
        self.spin_age.installEventFilter(self._select_all_filter)
        self.combo_gender = QComboBox()
        self.combo_gender.addItems(["Male", "Female", "Other"])
        self.spin_height = QDoubleSpinBox()
        self.spin_height.setRange(0, 250)
        self.spin_height.setSuffix(" cm")
        self.spin_height.setSpecialValueText("Not specified")
        self.spin_height.installEventFilter(self._select_all_filter)
        self.spin_weight = QDoubleSpinBox()
        self.spin_weight.setRange(0, 300)
        self.spin_weight.setSuffix(" kg")
        self.spin_weight.setSpecialValueText("Not specified")
        self.spin_weight.installEventFilter(self._select_all_filter)

        row = 0
        grid.addWidget(_field_label("Patient ID *"), row, 0)
        grid.addWidget(_field_label("Full Name *"), row, 2)
        row += 1
        grid.addWidget(self.txt_patient_id, row, 0, 1, 2)
        grid.addWidget(self.txt_full_name, row, 2, 1, 2)
        row += 1
        grid.addWidget(_field_label("Age"), row, 0)
        grid.addWidget(_field_label("Gender"), row, 1)
        grid.addWidget(_field_label("Height"), row, 2)
        grid.addWidget(_field_label("Weight"), row, 3)
        row += 1
        grid.addWidget(self.spin_age, row, 0)
        grid.addWidget(self.combo_gender, row, 1)
        grid.addWidget(self.spin_height, row, 2)
        grid.addWidget(self.spin_weight, row, 3)

        v.addLayout(grid)
        return card

    def _build_study_card(self) -> QFrame:
        card = _card()
        v = QVBoxLayout(card)
        v.setContentsMargins(20, 18, 20, 18)
        v.setSpacing(12)

        header = QLabel("STUDY INFORMATION")
        header.setProperty("role", "sectionHeader")
        v.addWidget(header)

        grid = QGridLayout()
        grid.setHorizontalSpacing(16)
        grid.setVerticalSpacing(4)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(3, 1)

        self.date_study = QDateEdit()
        self.date_study.setCalendarPopup(True)
        self.date_study.setDate(QDate.currentDate())
        self.txt_operator = QLineEdit()
        self.txt_operator.setPlaceholderText("Performing cardiologist")
        self.txt_referring = QLineEdit()
        self.txt_referring.setPlaceholderText("Referring physician")

        self.combo_indication = QComboBox()
        self.combo_indication.addItems(INDICATIONS)
        self.txt_indication_other = QLineEdit()
        self.txt_indication_other.setPlaceholderText("Specify indication...")
        self.txt_indication_other.setEnabled(False)
        self.combo_indication.currentTextChanged.connect(
            lambda text: self.txt_indication_other.setEnabled(text == "Other")
        )

        row = 0
        grid.addWidget(_field_label("Study Date"), row, 0)
        grid.addWidget(_field_label("Operator"), row, 1)
        grid.addWidget(_field_label("Referring Physician"), row, 2, 1, 2)
        row += 1
        grid.addWidget(self.date_study, row, 0)
        grid.addWidget(self.txt_operator, row, 1)
        grid.addWidget(self.txt_referring, row, 2, 1, 2)
        row += 1
        grid.addWidget(_field_label("Indication"), row, 0, 1, 2)
        row += 1
        grid.addWidget(self.combo_indication, row, 0, 1, 2)
        grid.addWidget(self.txt_indication_other, row, 2, 1, 2)

        v.addLayout(grid)
        return card

    def _build_history_card(self) -> QFrame:
        card = _card()
        v = QVBoxLayout(card)
        v.setContentsMargins(20, 18, 20, 18)
        v.setSpacing(12)

        header = QLabel("CLINICAL HISTORY / RISK FACTORS")
        header.setProperty("role", "sectionHeader")
        v.addWidget(header)

        grid = QGridLayout()
        grid.setHorizontalSpacing(24)
        grid.setVerticalSpacing(8)
        self.chk_risk_factors = {}
        cols = 3
        for i, name in enumerate(RISK_FACTORS):
            chk = QCheckBox(name)
            self.chk_risk_factors[name] = chk
            grid.addWidget(chk, i // cols, i % cols)
        v.addLayout(grid)

        echo_row = QHBoxLayout()
        self.chk_echo_done = QCheckBox("Echocardiogram Conducted")
        self.chk_echo_done.toggled.connect(self._on_echo_toggled)
        echo_row.addWidget(self.chk_echo_done)

        echo_row.addWidget(_field_label("EF"))
        self.spin_ef = QDoubleSpinBox()
        self.spin_ef.setRange(0, 100)
        self.spin_ef.setSuffix(" % EF")
        self.spin_ef.setSpecialValueText("Not specified")
        self.spin_ef.setEnabled(False)
        self.spin_ef.installEventFilter(self._select_all_filter)
        echo_row.addWidget(self.spin_ef)
        echo_row.addStretch()
        v.addLayout(echo_row)

        v.addWidget(_field_label("Additional Notes"))
        self.txt_notes = QTextEdit()
        self.txt_notes.setPlaceholderText("Any other clinically relevant notes...")
        self.txt_notes.setFixedHeight(70)
        v.addWidget(self.txt_notes)

        return card

    def _build_imaging_card(self) -> QFrame:
        card = _card()
        v = QVBoxLayout(card)
        v.setContentsMargins(20, 18, 20, 18)
        v.setSpacing(12)

        header = QLabel("IMAGING DATA")
        header.setProperty("role", "sectionHeader")
        v.addWidget(header)

        hint = QLabel(
            "Add DICOM files or a full study folder. The app will keep the files with this case."
        )
        hint.setProperty("role", "hint")
        hint.setWordWrap(True)
        v.addWidget(hint)

        row = QHBoxLayout()
        self.list_dicom = QListWidget()
        self.list_dicom.setFixedHeight(90)
        self.list_dicom.setSelectionMode(QListWidget.ExtendedSelection)
        row.addWidget(self.list_dicom, stretch=1)

        btns = QVBoxLayout()
        self.btn_add_dicom_folder = QPushButton("Add DICOM Folder")
        self.btn_add_dicom_folder.setProperty("variant", "primary")
        self.btn_add_dicom_folder.clicked.connect(self._add_dicom_folder)
        btns.addWidget(self.btn_add_dicom_folder)

        self.btn_add_dicom = QPushButton("Add Individual File(s)")
        self.btn_add_dicom.clicked.connect(self._add_dicom_files)
        btns.addWidget(self.btn_add_dicom)

        self.btn_remove_dicom = QPushButton("Remove Selected")
        self.btn_remove_dicom.setProperty("variant", "ghost")
        self.btn_remove_dicom.clicked.connect(self._remove_selected_dicom)
        btns.addWidget(self.btn_remove_dicom)
        btns.addStretch()
        row.addLayout(btns)

        v.addLayout(row)
        return card

    def _build_action_bar(self) -> QFrame:
        card = _card()
        v = QVBoxLayout(card)
        v.setContentsMargins(20, 16, 20, 16)
        v.setSpacing(10)

        self.lbl_status = QLabel("Fields marked * are required.")
        self.lbl_status.setProperty("role", "hint")
        self.lbl_status.setWordWrap(True)
        v.addWidget(self.lbl_status)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)  # indeterminate -- copy size isn't known upfront
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedHeight(6)
        self.progress_bar.setVisible(False)
        v.addWidget(self.progress_bar)

        btn_row = QHBoxLayout()
        self.btn_create = QPushButton("Create Patient Case")
        self.btn_create.setProperty("variant", "primary")
        self.btn_create.setFixedHeight(38)
        self.btn_create.clicked.connect(self._create_case)
        btn_row.addWidget(self.btn_create)

        self.btn_reset = QPushButton("Reset Form")
        self.btn_reset.setProperty("variant", "ghost")
        self.btn_reset.setFixedHeight(38)
        self.btn_reset.clicked.connect(self._reset_form)
        btn_row.addWidget(self.btn_reset)

        btn_row.addStretch()

        self.btn_open_folder = QPushButton("Open Case Folder")
        self.btn_open_folder.setFixedHeight(38)
        self.btn_open_folder.setVisible(False)
        self.btn_open_folder.clicked.connect(self._open_last_case_folder)
        btn_row.addWidget(self.btn_open_folder)

        v.addLayout(btn_row)
        return card

    # ── DICOM source list (files and/or whole study folders) ────────
    def _add_dicom_folder(self):
        path = QFileDialog.getExistingDirectory(self, "Select DICOM Study Folder", "")
        if path:
            self._add_dicom_source(path, is_folder=True)

    def _add_dicom_files(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select DICOM File(s)", "", "DICOM Files (*.dcm);;All Files (*)"
        )
        for p in paths:
            self._add_dicom_source(p, is_folder=False)

    def _add_dicom_source(self, path: str, is_folder: bool):
        if path in self._dicom_paths:
            return
        self._dicom_paths.append(path)
        label = f"📁 {os.path.basename(path)}/" if is_folder else os.path.basename(path)
        self.list_dicom.addItem(QListWidgetItem(label))

    def _remove_selected_dicom(self):
        for item in self.list_dicom.selectedItems():
            row = self.list_dicom.row(item)
            self.list_dicom.takeItem(row)
            del self._dicom_paths[row]

    # ── Submit / validate ──────────────────────────────────────────
    def _validate(self):
        errors = []
        required = [
            (self.txt_patient_id, "Patient ID"),
            (self.txt_full_name, "Full Name"),
        ]
        for field, name in required:
            ok = bool(field.text().strip())
            field.setProperty("error", not ok)
            field.style().unpolish(field)
            field.style().polish(field)
            if not ok:
                errors.append(f"{name} is required.")

        if not self._dicom_paths:
            errors.append("At least one DICOM file or study folder must be attached.")

        return errors

    def _collect_metadata(self) -> dict:
        indication = self.combo_indication.currentText()
        if indication == "Other" and self.txt_indication_other.text().strip():
            indication = self.txt_indication_other.text().strip()

        return {
            "patient_id": self.txt_patient_id.text().strip(),
            "full_name": self.txt_full_name.text().strip(),
            "age": self.spin_age.value(),
            "gender": self.combo_gender.currentText(),
            "height_cm": self.spin_height.value() or None,
            "weight_kg": self.spin_weight.value() or None,
            "study_date": self.date_study.date().toString("yyyy-MM-dd"),
            "operator": self.txt_operator.text().strip(),
            "referring_physician": self.txt_referring.text().strip(),
            "indication": indication,
            "risk_factors": [name for name, chk in self.chk_risk_factors.items() if chk.isChecked()],
            "echo_conducted": self.chk_echo_done.isChecked(),
            "ef_percent": self.spin_ef.value() if self.chk_echo_done.isChecked() and self.spin_ef.value() > 0 else None,
            "notes": self.txt_notes.toPlainText().strip(),
        }

    def _create_case(self):
        if self._case_thread is not None and self._case_thread.isRunning():
            return

        errors = self._validate()
        if errors:
            self.lbl_status.setProperty("role", "statusError")
            self.lbl_status.setText("• " + "\n• ".join(errors))
            self._repolish(self.lbl_status)
            return

        metadata = self._collect_metadata()

        self._set_busy(True)
        self.lbl_status.setProperty("role", "hint")
        self.lbl_status.setText("Creating case folder...")
        self._repolish(self.lbl_status)

        self._case_thread = _CaseCreationThread(metadata, list(self._dicom_paths))
        self._case_thread.progress.connect(self._on_case_progress)
        self._case_thread.finished_ok.connect(self._on_case_finished)
        self._case_thread.error.connect(self._on_case_error)
        self._case_thread.start()

    def _on_case_progress(self, message: str):
        self.lbl_status.setText(message)

    def _on_case_finished(self, case):
        self._last_case = case
        self._set_busy(False)
        self.lbl_status.setProperty("role", "statusSuccess")
        self.lbl_status.setText(
            f"Case created: {case.case_dir}  ({len(case.dicom_files)} DICOM file(s) copied)"
        )
        self._repolish(self.lbl_status)
        self.btn_open_folder.setVisible(True)

        self.case_created.emit(case)

    def _on_case_error(self, message: str):
        self._set_busy(False)
        self.lbl_status.setProperty("role", "statusError")
        self.lbl_status.setText(f"Failed to create patient case: {message}")
        self._repolish(self.lbl_status)
        QMessageBox.warning(self, "Case Creation Failed", message)

    def _set_busy(self, busy: bool):
        self.btn_create.setEnabled(not busy)
        self.btn_reset.setEnabled(not busy)
        self.btn_add_dicom_folder.setEnabled(not busy)
        self.btn_add_dicom.setEnabled(not busy)
        self.btn_remove_dicom.setEnabled(not busy)
        self.progress_bar.setVisible(busy)

    def _on_echo_toggled(self, checked: bool):
        self.spin_ef.setEnabled(checked)
        if not checked:
            self.spin_ef.setValue(0)

    def _open_last_case_folder(self):
        case = getattr(self, "_last_case", None)
        if case is None:
            return
        if hasattr(os, "startfile"):
            try:
                os.startfile(case.case_dir)
            except OSError:
                pass

    def _reset_form(self):
        self.txt_patient_id.clear()
        self.txt_full_name.clear()
        self.spin_age.setValue(0)
        self.combo_gender.setCurrentIndex(0)
        self.spin_height.setValue(0)
        self.spin_weight.setValue(0)
        self.date_study.setDate(QDate.currentDate())
        self.txt_operator.clear()
        self.txt_referring.clear()
        self.combo_indication.setCurrentIndex(0)
        self.txt_indication_other.clear()
        for chk in self.chk_risk_factors.values():
            chk.setChecked(False)
        self.chk_echo_done.setChecked(False)
        self.spin_ef.setValue(0)
        self.txt_notes.clear()
        self._dicom_paths = []
        self.list_dicom.clear()
        self.btn_open_folder.setVisible(False)
        self.lbl_status.setProperty("role", "hint")
        self.lbl_status.setText("Fields marked * are required.")
        self._repolish(self.lbl_status)
        for field in (self.txt_patient_id, self.txt_full_name):
            field.setProperty("error", False)
            self._repolish(field)

    @staticmethod
    def _repolish(widget):
        widget.style().unpolish(widget)
        widget.style().polish(widget)
