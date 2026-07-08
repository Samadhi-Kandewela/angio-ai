"""Main window shell: a left navigation rail plus a stacked content area."""
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QLabel,
    QListWidget, QListWidgetItem, QStackedWidget, QFrame, QMessageBox, QApplication
)

from patient_intake_page import PatientIntakePage
from patient_records_page import PatientRecordsPage
from local_dicom_analysis_page import LocalDicomAnalysisPage
from three_d_viewer_page import ThreeDViewerPage
from ecg_analysis_page import EcgAnalysisPage
from live_stream_page import LiveStreamPage


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOGO_DIR = PROJECT_ROOT / "logo"
LOGO_PNG = LOGO_DIR / "logo.png"
LOGO_ICO = LOGO_DIR / "logo.ico"

NAV_ITEMS = [
    ("New Patient", True),
    ("Patient Records", True),
    ("DICOM Analysis", True),
    ("ECG Analysis", True),
    ("Live Stream Analysis", True),
    ("3D Viewer", True),
]

NAV_NEW_PATIENT = 0
NAV_PATIENT_RECORDS = 1
NAV_DICOM_ANALYSIS = 2
NAV_ECG_ANALYSIS = 3
NAV_LIVE_STREAM = 4
NAV_3D_VIEWER = 5


class AppWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Cardexa Clinical Dashboard")
        if LOGO_ICO.exists():
            self.setWindowIcon(QIcon(str(LOGO_ICO)))
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
        self._add_page(NAV_NEW_PATIENT, self.patient_intake_page)

        self.patient_records_page = PatientRecordsPage()
        self.patient_records_page.go_to_dicom_analysis.connect(self._on_go_to_dicom_analysis)
        self._add_page(NAV_PATIENT_RECORDS, self.patient_records_page)

        self.dicom_analysis_page = LocalDicomAnalysisPage()
        self.dicom_analysis_page.go_to_new_patient.connect(lambda: self.nav_list.setCurrentRow(NAV_NEW_PATIENT))
        if hasattr(self.dicom_analysis_page, "view_3d_requested"):
            self.dicom_analysis_page.view_3d_requested.connect(self._open_3d_viewer_for_case)
        self._add_page(NAV_DICOM_ANALYSIS, self.dicom_analysis_page)

        self.ecg_analysis_page = EcgAnalysisPage()
        self.ecg_analysis_page.go_to_new_patient.connect(lambda: self.nav_list.setCurrentRow(NAV_NEW_PATIENT))
        self._add_page(NAV_ECG_ANALYSIS, self.ecg_analysis_page)

        self.live_stream_page = LiveStreamPage()
        self.live_stream_page.go_to_new_patient.connect(lambda: self.nav_list.setCurrentRow(NAV_NEW_PATIENT))
        self._add_page(NAV_LIVE_STREAM, self.live_stream_page)

        self.three_d_viewer_page = ThreeDViewerPage()
        if hasattr(self.three_d_viewer_page, "create_reconstruction_requested"):
            self.three_d_viewer_page.create_reconstruction_requested.connect(
                lambda: self.nav_list.setCurrentRow(NAV_DICOM_ANALYSIS)
            )
        self._add_page(NAV_3D_VIEWER, self.three_d_viewer_page)

        self.nav_list.currentRowChanged.connect(self._on_nav_changed)

        self.statusBar().showMessage("Ready.")

    def _build_sidebar(self) -> QFrame:
        sidebar = QFrame()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(240)

        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(20, 24, 20, 20)
        layout.setSpacing(2)

        brand_block = QFrame()
        brand_block.setObjectName("brandBlock")
        brand_layout = QVBoxLayout(brand_block)
        brand_layout.setContentsMargins(4, 4, 4, 4)
        brand_layout.setSpacing(0)

        if LOGO_PNG.exists():
            brand_logo = QLabel()
            brand_logo.setObjectName("brandLogo")
            brand_logo.setAlignment(Qt.AlignCenter)
            brand_logo.setFixedHeight(72)
            brand_logo.setPixmap(
                QPixmap(str(LOGO_PNG)).scaledToWidth(180, Qt.SmoothTransformation)
            )
            brand_layout.addWidget(brand_logo)
        else:
            brand = QLabel("Cardexa")
            brand.setObjectName("brand")
            brand.setAlignment(Qt.AlignCenter)
            brand_layout.addWidget(brand)

        layout.addWidget(brand_block)

        layout.addSpacing(30)

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
        self.ecg_analysis_page.refresh_cases()
        self.live_stream_page.refresh_cases()
        self.patient_records_page.refresh()

    def _on_go_to_dicom_analysis(self, case_id: str):
        self.nav_list.setCurrentRow(NAV_DICOM_ANALYSIS)
        self.dicom_analysis_page.select_case_by_id(case_id)

    def _open_3d_viewer_for_case(self, case_id: str):
        self.nav_list.setCurrentRow(NAV_3D_VIEWER)
        QApplication.processEvents()
        if self.three_d_viewer_page.load_case(case_id):
            self.statusBar().showMessage(f"3D Viewer loaded case: {case_id}")
        else:
            message = self.three_d_viewer_page.last_error or f"3D reconstruction is not ready for case: {case_id}"
            self.statusBar().showMessage(message)
            QMessageBox.warning(self, "3D Viewer", message)

    def closeEvent(self, event):
        self.dicom_analysis_page.shutdown()
        self.ecg_analysis_page.shutdown()
        self.live_stream_page.shutdown()
        event.accept()
