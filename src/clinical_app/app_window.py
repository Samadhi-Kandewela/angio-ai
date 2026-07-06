"""
Main window shell: a left navigation rail plus a stacked content area.

"New Patient" and "Local DICOM Analysis" are implemented. Live Stream
Analysis and 3D Viewer are implemented. Live Stream Analysis, Reports, and
Settings remain disabled placeholders -- this shell is the scaffold they plug
into as they're built in later iterations.
"""
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QLabel,
    QListWidget, QListWidgetItem, QStackedWidget, QFrame, QMessageBox, QApplication
)

from patient_intake_page import PatientIntakePage
from local_dicom_analysis_page import LocalDicomAnalysisPage
from three_d_viewer_page import ThreeDViewerPage

# (label, enabled) -- enabled rows get a real page; disabled ones are future work.
NAV_ITEMS = [
    ("New Patient", True),
    ("DICOM Analysis", True),
    ("Live Stream Analysis", False),
    ("3D Viewer", True),
    ("Reports", False),
    ("Settings", False),
]


class AppWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Angio-AI Clinical Dashboard")
        self.resize(1440, 900)
        self.setMinimumSize(1180, 720)

        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QHBoxLayout(central)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        root_layout.addWidget(self._build_sidebar())

        self.pages = QStackedWidget()
        root_layout.addWidget(self.pages, stretch=1)

        self.patient_intake_page = PatientIntakePage()
        self.patient_intake_page.case_created.connect(self._on_case_created)
        self._add_page(0, self.patient_intake_page)

        self.dicom_analysis_page = LocalDicomAnalysisPage()
        self.dicom_analysis_page.go_to_new_patient.connect(lambda: self.nav_list.setCurrentRow(0))
        self.dicom_analysis_page.view_3d_requested.connect(self._open_3d_viewer_for_case)
        self._add_page(1, self.dicom_analysis_page)

        self.three_d_viewer_page = ThreeDViewerPage()
        self._add_page(3, self.three_d_viewer_page)

        self.nav_list.currentRowChanged.connect(self._on_nav_changed)

        self.statusBar().showMessage("Ready.")

    def _build_sidebar(self) -> QFrame:
        sidebar = QFrame()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(240)

        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(20, 24, 20, 20)
        layout.setSpacing(2)

        brand = QLabel("ANGIO-AI")
        brand.setObjectName("brand")
        layout.addWidget(brand)

        subtitle = QLabel("Clinical Dashboard")
        subtitle.setObjectName("brandSubtitle")
        layout.addWidget(subtitle)

        layout.addSpacing(28)

        self.nav_list = QListWidget()
        self.nav_list.setObjectName("navList")
        self.nav_list.setFocusPolicy(Qt.NoFocus)
        for label, enabled in NAV_ITEMS:
            item = QListWidgetItem(label)
            if not enabled:
                item.setFlags(item.flags() & ~Qt.ItemIsEnabled)
            self.nav_list.addItem(item)
        self.nav_list.setCurrentRow(0)
        layout.addWidget(self.nav_list)
        layout.addStretch()

        version = QLabel("v0.2 — patient intake + DICOM analysis")
        version.setObjectName("versionLabel")
        layout.addWidget(version)

        return sidebar

    def _add_page(self, nav_row: int, page: QWidget):
        """Registers a page widget for the given sidebar row."""
        if not hasattr(self, "_page_by_row"):
            self._page_by_row = {}
        index = self.pages.addWidget(page)
        self._page_by_row[nav_row] = index

    def _on_nav_changed(self, row: int):
        index = getattr(self, "_page_by_row", {}).get(row)
        if index is not None:
            self.pages.setCurrentIndex(index)

    def _on_case_created(self, case):
        self.statusBar().showMessage(f"Case created: {case.case_dir}")
        self.dicom_analysis_page.refresh_cases()

    def _open_3d_viewer_for_case(self, case_id: str):
        self.nav_list.setCurrentRow(3)
        QApplication.processEvents()
        if self.three_d_viewer_page.load_case(case_id):
            self.statusBar().showMessage(f"3D Viewer loaded case: {case_id}")
        else:
            message = self.three_d_viewer_page.last_error or f"3D reconstruction is not ready for case: {case_id}"
            self.statusBar().showMessage(message)
            QMessageBox.warning(self, "3D Viewer", message)

    def closeEvent(self, event):
        self.dicom_analysis_page.shutdown()
        event.accept()
