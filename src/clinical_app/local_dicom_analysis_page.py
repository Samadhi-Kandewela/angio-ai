"""
Local DICOM Analysis page.

Lets the analyst pick a previously created patient case, choose one of its
DICOM series, load AI models, then play/scrub through the cine loop with the
original frame and QCA stenosis analysis always visible. The segmentation
mask is optional -- shown only if the analyst toggles it on -- since it's a
diagnostic aid, not the primary read.
"""
import os
from pathlib import Path

import cv2

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QImage, QPixmap, QFont
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QLineEdit,
    QComboBox, QSlider, QPushButton, QFrame, QFileDialog, QListWidget,
    QListWidgetItem, QCheckBox, QMessageBox, QSizePolicy, QScrollArea, QInputDialog
)

import patient_store
import analysis_results_store
from dicom_loader import discover_series, load_series_frames
from dicom_analysis_thread import DicomAnalysisThread
from frame_pipeline import SegmentationModel, LocalizationModel
from report_engine import KEY_FRAME_MAX_COUNT, analyze_frame_list, draw_frame_stenosis_only

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # angio-ai/
DEFAULT_SEGMENTATION_MODEL_PATHS = [
    PROJECT_ROOT / "checkpoints" / "mobileunetv3" / "mobileunetv3_augmented_best.onnx",
    PROJECT_ROOT / "checkpoints" / "mobileunetv3" / "mobileunetv3_augmented_best.pth",
]
DEFAULT_LOCALIZATION_MODEL_PATHS = [
    PROJECT_ROOT / "checkpoints" / "mask_localization_v2" / "best.onnx",
    PROJECT_ROOT / "checkpoints" / "mask_localization_v2" / "best.pth",
    PROJECT_ROOT / "checkpoints" / "mask_localization_v2" / "latest.pth",
    PROJECT_ROOT / "checkpoints" / "multitask_localization_v2" / "multitask_latest.onnx",
    PROJECT_ROOT / "checkpoints" / "multitask_localization_v2" / "multitask_latest.pth",
    PROJECT_ROOT / "checkpoints" / "multitask_localization_v2" / "multitask_best.onnx",
    PROJECT_ROOT / "checkpoints" / "multitask_localization_v2" / "multitask_best.pth",
]


def _first_existing(paths):
    for p in paths:
        if Path(p).exists():
            return str(p)
    return ""


def _card() -> QFrame:
    frame = QFrame()
    frame.setProperty("card", "true")
    return frame


def _field_label(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setProperty("role", "fieldLabel")
    return lbl


def ndarray_to_qpixmap(img_rgb, target_w: int, target_h: int) -> QPixmap:
    h, w, ch = img_rgb.shape
    bytes_per_line = ch * w
    qimg = QImage(img_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg).scaled(
        target_w, target_h, Qt.KeepAspectRatio, Qt.SmoothTransformation
    )


class _SeriesLoadThread(QThread):
    loaded = Signal(list)
    error = Signal(str)

    def __init__(self, path):
        super().__init__()
        self.path = path

    def run(self):
        try:
            frames = load_series_frames(self.path)
            self.loaded.emit(frames)
        except Exception as e:
            self.error.emit(str(e))


class _ViewAnalysisThread(QThread):
    """
    Runs the whole-series QCA analysis (analyze_frame_list) off the UI
    thread -- aggregating detections across every frame of a series into
    authoritative per-lesion tracks takes much longer than analyzing one
    live frame, so it must not block the UI.
    """
    progress = Signal(str)
    finished_ok = Signal(object)  # AngleResult
    error = Signal(str)

    def __init__(self, frames, view_label, seg_model, loc_model, cfg, threshold):
        super().__init__()
        self.frames = frames
        self.view_label = view_label
        self.seg_model = seg_model
        self.loc_model = loc_model
        self.cfg = cfg
        self.threshold = threshold

    def run(self):
        try:
            def _cb(i, n):
                if n:
                    self.progress.emit(f"Analyzing frame {i}/{n}...")

            result = analyze_frame_list(
                self.frames, self.view_label, self.seg_model, self.loc_model,
                self.cfg, threshold=self.threshold, progress_cb=_cb,
            )
            self.finished_ok.emit(result)
        except Exception as e:
            self.error.emit(str(e))


class LocalDicomAnalysisPage(QWidget):
    """DICOM-file analysis page: case/series picker + AI-model viewer."""

    go_to_new_patient = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._cases = []          # raw case_index entries
        self._series = []         # DicomSeriesInfo list for the selected case
        self._series_frames = []  # loaded BGR frames for the selected series
        self._series_load_thread = None
        self._view_analysis_thread = None
        self._view_analysis_is_preview_only = False
        self._pending_case_id = None
        self._last_view_report = None
        self._last_final_report = None
        self._current_angle_result = None
        self._current_key_frame_page = 0
        self.analysis_thread = DicomAnalysisThread()
        self.analysis_thread.frame_ready.connect(self._on_frame_ready)
        self.analysis_thread.playback_finished.connect(self._on_playback_finished)
        self.analysis_thread.error.connect(self._on_analysis_error)
        self.analysis_thread.start()  # idle until play()/seek() is called

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
        self._page_scroll_area = scroll

        content = QWidget()
        scroll.setWidget(content)

        layout = QVBoxLayout(content)
        layout.setContentsMargins(28, 24, 28, 20)
        layout.setSpacing(14)

        title = QLabel("Local DICOM Analysis")
        title.setProperty("role", "pageTitle")
        layout.addWidget(title)

        subtitle = QLabel(
            "Select a patient case and DICOM series, load the AI models, then play or scrub through "
            "the cine loop. Each frame is fully re-analyzed (segmentation + QCA), so playback is "
            "paced by analysis time, not the original frame rate."
        )
        subtitle.setProperty("role", "pageSubtitle")
        subtitle.setWordWrap(True)
        layout.addWidget(subtitle)

        layout.addWidget(self._build_selection_card())
        layout.addWidget(self._build_model_card())
        layout.addWidget(self._build_viewer_card())
        layout.addWidget(self._build_playback_card())
        layout.addWidget(self._build_key_frames_card())
        layout.addWidget(self._build_save_report_card())

    def _build_selection_card(self) -> QFrame:
        card = _card()
        v = QVBoxLayout(card)
        v.setContentsMargins(20, 16, 20, 16)
        v.setSpacing(10)

        header = QLabel("CASE & SERIES")
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

        v.addWidget(_field_label("DICOM Series"))
        self.list_series = QListWidget()
        self.list_series.setFixedHeight(72)
        v.addWidget(self.list_series)

        btn_row = QHBoxLayout()
        self.btn_load_series = QPushButton("Load Series")
        self.btn_load_series.setProperty("variant", "primary")
        self.btn_load_series.clicked.connect(self._load_selected_series)
        btn_row.addWidget(self.btn_load_series)

        self.lbl_series_status = QLabel("No series loaded.")
        self.lbl_series_status.setProperty("role", "hint")
        btn_row.addWidget(self.lbl_series_status, stretch=1)
        v.addLayout(btn_row)

        return card

    def _build_model_card(self) -> QFrame:
        card = _card()
        v = QVBoxLayout(card)
        v.setContentsMargins(20, 16, 20, 16)
        v.setSpacing(10)

        header = QLabel("AI MODELS")
        header.setProperty("role", "sectionHeader")
        v.addWidget(header)

        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setColumnStretch(1, 1)

        grid.addWidget(_field_label("Segmentation Model"), 0, 0)
        self.txt_seg_model = QLineEdit(_first_existing(DEFAULT_SEGMENTATION_MODEL_PATHS))
        grid.addWidget(self.txt_seg_model, 0, 1)
        self.btn_browse_seg = QPushButton("Browse...")
        self.btn_browse_seg.setProperty("variant", "ghost")
        self.btn_browse_seg.clicked.connect(self._browse_seg_model)
        grid.addWidget(self.btn_browse_seg, 0, 2)

        grid.addWidget(_field_label("Localization Model (optional)"), 1, 0)
        self.txt_loc_model = QLineEdit(_first_existing(DEFAULT_LOCALIZATION_MODEL_PATHS))
        grid.addWidget(self.txt_loc_model, 1, 1)
        self.btn_browse_loc = QPushButton("Browse...")
        self.btn_browse_loc.setProperty("variant", "ghost")
        self.btn_browse_loc.clicked.connect(self._browse_loc_model)
        grid.addWidget(self.btn_browse_loc, 1, 2)

        v.addLayout(grid)

        btn_row = QHBoxLayout()
        self.btn_load_models = QPushButton("Load Models")
        self.btn_load_models.setProperty("variant", "primary")
        self.btn_load_models.clicked.connect(self._load_models)
        btn_row.addWidget(self.btn_load_models)

        self.lbl_model_status = QLabel("Models not loaded.")
        self.lbl_model_status.setProperty("role", "hint")
        btn_row.addWidget(self.lbl_model_status, stretch=1)
        v.addLayout(btn_row)

        return card

    def _build_viewer_card(self) -> QFrame:
        card = _card()
        v = QVBoxLayout(card)
        v.setContentsMargins(20, 16, 20, 16)
        v.setSpacing(10)

        top_row = QHBoxLayout()
        header = QLabel("ANALYSIS VIEWER")
        header.setProperty("role", "sectionHeader")
        top_row.addWidget(header)
        top_row.addStretch()

        self.chk_show_mask = QCheckBox("Show Segmentation Mask")
        self.chk_show_mask.toggled.connect(self._on_toggle_mask)
        top_row.addWidget(self.chk_show_mask)
        v.addLayout(top_row)

        panels = QHBoxLayout()
        panels.setSpacing(10)

        self.panel_original, self.label_original = self._build_image_panel("Original Frame")
        panels.addWidget(self.panel_original, stretch=1)

        self.panel_qca, self.label_qca = self._build_image_panel("QCA Stenosis Analysis")
        panels.addWidget(self.panel_qca, stretch=1)

        self.panel_mask, self.label_mask = self._build_image_panel("Segmentation Mask")
        self.panel_mask.setVisible(False)
        panels.addWidget(self.panel_mask, stretch=1)

        v.addLayout(panels, stretch=1)

        self.lbl_stenosis = QLabel("No analysis yet — load a series and models, then press Play.")
        self.lbl_stenosis.setFont(QFont("Consolas", 11, QFont.Bold))
        self.lbl_stenosis.setWordWrap(True)
        self.lbl_stenosis.setProperty("role", "hint")
        v.addWidget(self.lbl_stenosis)

        return card

    def _build_image_panel(self, title: str):
        box = _card()
        v = QVBoxLayout(box)
        v.setContentsMargins(10, 10, 10, 10)
        v.setSpacing(6)

        header = QLabel(title)
        header.setProperty("role", "fieldLabel")
        v.addWidget(header)

        label = QLabel("No frame loaded")
        label.setAlignment(Qt.AlignCenter)
        label.setMinimumSize(520, 460)
        label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        label.setStyleSheet("background-color: #0B0C0E; border-radius: 6px; color: #5C6067;")
        v.addWidget(label, stretch=1)

        return box, label

    def _build_playback_card(self) -> QFrame:
        card = _card()
        v = QVBoxLayout(card)
        v.setContentsMargins(20, 14, 20, 14)
        v.setSpacing(10)

        slider_row = QHBoxLayout()
        self.slider_frame = QSlider(Qt.Horizontal)
        self.slider_frame.setRange(0, 0)
        self.slider_frame.sliderReleased.connect(self._on_slider_released)
        self.slider_frame.valueChanged.connect(self._on_slider_value_changed)
        slider_row.addWidget(self.slider_frame, stretch=1)

        self.lbl_frame_count = QLabel("Frame 0 / 0")
        self.lbl_frame_count.setFont(QFont("Consolas", 10))
        slider_row.addWidget(self.lbl_frame_count)
        v.addLayout(slider_row)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)

        self.btn_prev_frame = QPushButton("◀ Prev")
        self.btn_prev_frame.clicked.connect(self._step_prev)
        btn_row.addWidget(self.btn_prev_frame)

        self.btn_play = QPushButton("Play")
        self.btn_play.setProperty("variant", "primary")
        self.btn_play.clicked.connect(self._play)
        btn_row.addWidget(self.btn_play)

        self.btn_pause = QPushButton("Pause")
        self.btn_pause.clicked.connect(self._pause)
        btn_row.addWidget(self.btn_pause)

        self.btn_next_frame = QPushButton("Next ▶")
        self.btn_next_frame.clicked.connect(self._step_next)
        btn_row.addWidget(self.btn_next_frame)

        btn_row.addSpacing(20)
        btn_row.addWidget(_field_label("Threshold:"))
        self.slider_threshold = QSlider(Qt.Horizontal)
        self.slider_threshold.setRange(0, 100)
        self.slider_threshold.setValue(50)
        self.slider_threshold.setFixedWidth(110)
        self.slider_threshold.valueChanged.connect(self._on_threshold_changed)
        btn_row.addWidget(self.slider_threshold)
        self.lbl_threshold_val = QLabel("0.50")
        self.lbl_threshold_val.setFont(QFont("Consolas", 10))
        btn_row.addWidget(self.lbl_threshold_val)

        btn_row.addStretch()

        self.lbl_latency = QLabel("Latency: -- ms")
        self.lbl_latency.setFont(QFont("Consolas", 10))
        btn_row.addWidget(self.lbl_latency)

        v.addLayout(btn_row)
        self._set_playback_enabled(False)
        return card

    def _build_save_report_card(self) -> QFrame:
        card = _card()
        v = QVBoxLayout(card)
        v.setContentsMargins(20, 16, 20, 16)
        v.setSpacing(10)

        header = QLabel("SAVE & REPORT")
        header.setProperty("role", "sectionHeader")
        v.addWidget(header)

        hint = QLabel(
            "Analyzing the full series aggregates stenosis detections across every frame into one "
            "authoritative reading per lesion (not a single noisy frame) and saves the result into "
            "this case's analysis_results/ folder, alongside a per-view explainable report."
        )
        hint.setProperty("role", "hint")
        hint.setWordWrap(True)
        v.addWidget(hint)

        row1 = QHBoxLayout()
        self.btn_save_view = QPushButton("Save Results && Generate View Report")
        self.btn_save_view.setProperty("variant", "primary")
        self.btn_save_view.clicked.connect(self._save_view_results)
        row1.addWidget(self.btn_save_view)

        self.btn_open_view_report = QPushButton("Open View Report")
        self.btn_open_view_report.setVisible(False)
        self.btn_open_view_report.clicked.connect(self._open_view_report)
        row1.addWidget(self.btn_open_view_report)
        row1.addStretch()
        v.addLayout(row1)

        self.lbl_save_status = QLabel("Load a series and models, then save results to build this view's report.")
        self.lbl_save_status.setProperty("role", "hint")
        self.lbl_save_status.setWordWrap(True)
        v.addWidget(self.lbl_save_status)

        row2 = QHBoxLayout()
        self.btn_final_report = QPushButton("Generate Final Clinical Report")
        self.btn_final_report.setEnabled(False)
        self.btn_final_report.clicked.connect(self._generate_final_report)
        row2.addWidget(self.btn_final_report)

        self.btn_open_final_report = QPushButton("Open Final Report")
        self.btn_open_final_report.setVisible(False)
        self.btn_open_final_report.clicked.connect(self._open_final_report)
        row2.addWidget(self.btn_open_final_report)
        row2.addStretch()
        v.addLayout(row2)

        self.lbl_final_status = QLabel("No views saved yet for this case.")
        self.lbl_final_status.setProperty("role", "hint")
        self.lbl_final_status.setWordWrap(True)
        v.addWidget(self.lbl_final_status)

        return card

    def _build_key_frames_card(self) -> QFrame:
        card = _card()
        v = QVBoxLayout(card)
        v.setContentsMargins(20, 16, 20, 16)
        v.setSpacing(10)

        header = QLabel("KEY FRAMES — EVIDENCE FOR EVERY FINDING")
        header.setProperty("role", "sectionHeader")
        v.addWidget(header)

        hint = QLabel(
            "Analyzed automatically once playback reaches the end of the series -- the smallest set "
            f"of frames (up to {KEY_FRAME_MAX_COUNT}) that shows every significant stenosis at least "
            "once, picked by vessel opacification quality, so every finding has real supporting "
            "evidence. Each picture shows only the original frame with a circle + short id (e.g. "
            "\"L1\") around its own stenosis -- no vessel mask or skeleton -- since these are this "
            "exact frame's own detections, always accurately placed."
        )
        hint.setProperty("role", "hint")
        hint.setWordWrap(True)
        v.addWidget(hint)

        self.lbl_key_frame_coverage = QLabel("Play through the series once to see key frames here.")
        self.lbl_key_frame_coverage.setProperty("role", "hint")
        self.lbl_key_frame_coverage.setWordWrap(True)
        v.addWidget(self.lbl_key_frame_coverage)

        nav_row = QHBoxLayout()
        self.btn_prev_key_frame = QPushButton("◀ Previous")
        self.btn_prev_key_frame.setFocusPolicy(Qt.NoFocus)
        self.btn_prev_key_frame.clicked.connect(self._go_prev_key_frame)
        self.btn_prev_key_frame.setEnabled(False)
        nav_row.addWidget(self.btn_prev_key_frame)

        self.lbl_key_frame_position = QLabel("No key frames yet")
        self.lbl_key_frame_position.setAlignment(Qt.AlignCenter)
        self.lbl_key_frame_position.setFont(QFont("Consolas", 10, QFont.Bold))
        nav_row.addWidget(self.lbl_key_frame_position, stretch=1)

        self.btn_next_key_frame = QPushButton("Next ▶")
        self.btn_next_key_frame.setFocusPolicy(Qt.NoFocus)
        self.btn_next_key_frame.clicked.connect(self._go_next_key_frame)
        self.btn_next_key_frame.setEnabled(False)
        nav_row.addWidget(self.btn_next_key_frame)
        v.addLayout(nav_row)

        box, unlabeled_image, labeled_image, caption_label = self._build_key_frame_panel_contents()
        self._kf_unlabeled_image = unlabeled_image
        self._kf_labeled_image = labeled_image
        self._kf_caption = caption_label
        v.addWidget(box)

        return card

    def _build_key_frame_panel_contents(self):
        box = _card()
        v = QVBoxLayout(box)
        v.setContentsMargins(14, 14, 14, 14)
        v.setSpacing(8)

        images_row = QHBoxLayout()
        images_row.setSpacing(14)

        unlabeled_col = QVBoxLayout()
        unlabeled_sub = QLabel("Original (unlabeled)")
        unlabeled_sub.setProperty("role", "hint")
        unlabeled_col.addWidget(unlabeled_sub)
        unlabeled_image = QLabel("No frame yet")
        unlabeled_image.setAlignment(Qt.AlignCenter)
        unlabeled_image.setFixedSize(460, 420)
        unlabeled_image.setStyleSheet("background-color: #0B0C0E; border-radius: 6px; color: #5C6067;")
        unlabeled_col.addWidget(unlabeled_image)
        images_row.addLayout(unlabeled_col)

        labeled_col = QVBoxLayout()
        labeled_sub = QLabel("Stenosis found")
        labeled_sub.setProperty("role", "hint")
        labeled_col.addWidget(labeled_sub)
        labeled_image = QLabel("No frame yet")
        labeled_image.setAlignment(Qt.AlignCenter)
        labeled_image.setFixedSize(460, 420)
        labeled_image.setStyleSheet("background-color: #0B0C0E; border-radius: 6px; color: #5C6067;")
        labeled_col.addWidget(labeled_image)
        images_row.addLayout(labeled_col)

        v.addLayout(images_row)

        caption_label = QLabel("")
        caption_label.setWordWrap(True)
        caption_label.setFont(QFont("Consolas", 10))
        caption_label.setProperty("role", "hint")
        v.addWidget(caption_label)

        return box, unlabeled_image, labeled_image, caption_label

    def _format_key_frame_caption(self, rec, angle_result) -> str:
        """Full detail (severity, DS%, MLD/RVD, confidence, location) for every
        significant lesion found on this exact frame -- more detail than a
        one-line summary since this is the only place a cardiologist sees a
        single lesion's full measurement next to its image."""
        track_of_lesion = {id(les): t for t in angle_result.tracks for les in t.detections}
        seen_ids = set()
        lines = [f"Frame {rec.frame_idx + 1} / {angle_result.n_frames_total}"]

        for les in rec.lesions:
            t = track_of_lesion.get(id(les))
            if t is None or t.track_id in seen_ids:
                continue
            rep = t.representative
            if rep["severity"] not in ("SEVERE", "SIGNIFICANT"):
                continue
            seen_ids.add(t.track_id)

            if rep.get("MLD_mm") is not None:
                mld, rvd, unit = rep["MLD_mm"], rep["RVD_mm"], "mm"
            else:
                mld, rvd, unit = rep["MLD_px"], rep["RVD_px"], "px"

            where = f"{t.label} ({t.artery})" if t.artery != "unknown" else "location unknown"
            occ = "  [TOTAL OCCLUSION]" if rep.get("total_occlusion") else ""
            conf = rep.get("confidence")
            conf_part = f"   Confidence: {conf:.2f}" if conf is not None else ""

            lines.append(
                f"\n{t.track_id} — {rep['severity']} — {rep['DS_percent']:.0f}% diameter stenosis{occ}\n"
                f"    Location: {where}\n"
                f"    MLD: {mld:.1f} {unit}   RVD: {rvd:.1f} {unit}{conf_part}"
            )

        if len(lines) == 1:
            lines.append("\nNo significant stenosis at this frame.")

        return "\n".join(lines)

    def _render_current_key_frame(self):
        result = self._current_angle_result
        if result is None or not result.key_frame_indices:
            self._kf_unlabeled_image.setPixmap(QPixmap())
            self._kf_unlabeled_image.setText("No frame yet")
            self._kf_labeled_image.setPixmap(QPixmap())
            self._kf_labeled_image.setText("No frame yet")
            self._kf_caption.setText("")
            self.lbl_key_frame_position.setText("No key frames yet")
            self.btn_prev_key_frame.setEnabled(False)
            self.btn_next_key_frame.setEnabled(False)
            return

        n = len(result.key_frame_indices)
        self._current_key_frame_page = max(0, min(self._current_key_frame_page, n - 1))
        idx = result.key_frame_indices[self._current_key_frame_page]
        rec = result.get_frame_record(idx)

        self.lbl_key_frame_position.setText(f"Key Frame {self._current_key_frame_page + 1} of {n}")
        self.btn_prev_key_frame.setEnabled(self._current_key_frame_page > 0)
        self.btn_next_key_frame.setEnabled(self._current_key_frame_page < n - 1)

        if rec is None:
            self._kf_unlabeled_image.setText("No frame data")
            self._kf_labeled_image.setText("No frame data")
            self._kf_caption.setText("")
            return

        self._kf_unlabeled_image.setPixmap(ndarray_to_qpixmap(
            rec.img_rgb, self._kf_unlabeled_image.width(), self._kf_unlabeled_image.height()
        ))

        vis_bgr = draw_frame_stenosis_only(rec, result.tracks)
        vis_rgb = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB)
        self._kf_labeled_image.setPixmap(ndarray_to_qpixmap(
            vis_rgb, self._kf_labeled_image.width(), self._kf_labeled_image.height()
        ))

        self._kf_caption.setText(self._format_key_frame_caption(rec, result))

    def _go_prev_key_frame(self):
        self._current_key_frame_page -= 1
        self._render_current_key_frame_keep_scroll()

    def _go_next_key_frame(self):
        self._current_key_frame_page += 1
        self._render_current_key_frame_keep_scroll()

    def _render_current_key_frame_keep_scroll(self):
        # Swapping in a new key frame's caption text can change its wrapped
        # height, which reflows everything below it -- pin the page's scroll
        # position across that reflow so Next/Previous doesn't visibly jump.
        scrollbar = self._page_scroll_area.verticalScrollBar()
        pos = scrollbar.value()
        self._render_current_key_frame()
        scrollbar.setValue(pos)

    def _update_key_frames(self, angle_result):
        self._current_angle_result = angle_result
        self._current_key_frame_page = 0
        self._render_current_key_frame()

        significant_tracks = [
            t for t in angle_result.tracks if t.representative["severity"] in ("SEVERE", "SIGNIFICANT")
        ]
        if not significant_tracks:
            self.lbl_key_frame_coverage.setText("No significant stenosis found across this series.")
            return

        track_of_lesion = {id(les): t for t in angle_result.tracks for les in t.detections}
        covered_ids = set()
        for idx in angle_result.key_frame_indices:
            rec = angle_result.get_frame_record(idx)
            if rec is None:
                continue
            for les in rec.lesions:
                t = track_of_lesion.get(id(les))
                if t is not None and t.representative["severity"] in ("SEVERE", "SIGNIFICANT"):
                    covered_ids.add(t.track_id)

        n_total = len(significant_tracks)
        n_covered = len({t.track_id for t in significant_tracks} & covered_ids)
        if n_covered >= n_total:
            self.lbl_key_frame_coverage.setText(
                f"{len(angle_result.key_frame_indices)} key frame(s) found, covering all {n_total} significant finding(s)."
            )
        else:
            self.lbl_key_frame_coverage.setText(
                f"{len(angle_result.key_frame_indices)} key frame(s) found, covering {n_covered}/{n_total} "
                f"significant finding(s) -- some findings never co-occurred with others within the "
                f"{KEY_FRAME_MAX_COUNT}-frame cap."
            )

    # ── Case / series selection ─────────────────────────────────────
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
            self.list_series.clear()
            self.lbl_series_status.setText(
                "No patient cases yet. Click \"+ New Patient\" to create one (with its DICOM data attached), "
                "then come back here."
            )
            self._refresh_final_report_status()

    def _on_case_selected(self, index: int):
        self.list_series.clear()
        self._series = []
        if index < 0 or index >= len(self._cases):
            return

        case_id = self._cases[index]["case_id"]
        dicom_dir = patient_store.get_case_dicom_dir(case_id)
        self._refresh_final_report_status()
        if not dicom_dir.exists():
            self.lbl_series_status.setText(f"No dicom/ folder found for case {case_id}.")
            return

        self._series = discover_series(dicom_dir)
        if not self._series:
            self.lbl_series_status.setText("No readable DICOM series found in this case.")
            return

        for s in self._series:
            rel = s.path.relative_to(dicom_dir)
            frame_word = "frame" if s.num_frames == 1 else "frames"
            self.list_series.addItem(QListWidgetItem(f"{rel} — {s.description} — {s.num_frames} {frame_word}"))
        self.lbl_series_status.setText(f"{len(self._series)} series found. Select one and press Load Series.")

    def _load_selected_series(self):
        row = self.list_series.currentRow()
        if row < 0 or row >= len(self._series):
            self.lbl_series_status.setText("Select a series from the list first.")
            return

        self.btn_load_series.setEnabled(False)
        self.lbl_series_status.setText("Loading series...")

        series_info = self._series[row]
        self._series_load_thread = _SeriesLoadThread(series_info.path)
        self._series_load_thread.loaded.connect(self._on_series_loaded)
        self._series_load_thread.error.connect(self._on_series_load_error)
        self._series_load_thread.start()

    def _on_series_loaded(self, frames):
        self.btn_load_series.setEnabled(True)
        self._series_frames = frames
        self.analysis_thread.set_frames(frames)

        self.slider_frame.blockSignals(True)
        self.slider_frame.setRange(0, len(frames) - 1)
        self.slider_frame.setValue(0)
        self.slider_frame.blockSignals(False)
        self.lbl_frame_count.setText(f"Frame 1 / {len(frames)}")

        self.lbl_series_status.setText(f"Loaded {len(frames)} frames. Ready to analyze.")
        self._set_playback_enabled(self.analysis_thread.seg_model is not None)
        if self.analysis_thread.seg_model is not None:
            self.analysis_thread.seek(0)

        self._current_angle_result = None
        self._current_key_frame_page = 0
        self._render_current_key_frame()
        self.lbl_key_frame_coverage.setText("Play through the series once to see key frames here.")

    def _on_series_load_error(self, message: str):
        self.btn_load_series.setEnabled(True)
        self.lbl_series_status.setText(f"Failed to load series: {message}")
        QMessageBox.warning(self, "Series Load Failed", message)

    # ── Models ───────────────────────────────────────────────────────
    def _browse_seg_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Segmentation Model", "", "Model Files (*.onnx *.pth)")
        if path:
            self.txt_seg_model.setText(path)

    def _browse_loc_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Localization Model", "", "Model Files (*.onnx *.pth)")
        if path:
            self.txt_loc_model.setText(path)

    def _load_models(self):
        seg_path = self.txt_seg_model.text().strip()
        if not seg_path:
            self.lbl_model_status.setText("Select a segmentation model first.")
            return

        try:
            self.analysis_thread.seg_model = SegmentationModel(seg_path)
        except Exception as e:
            self.lbl_model_status.setProperty("role", "statusError")
            self.lbl_model_status.setText(f"Segmentation model load failed: {e}")
            self._repolish(self.lbl_model_status)
            return

        loc_path = self.txt_loc_model.text().strip()
        self.analysis_thread.loc_model = None
        if loc_path:
            try:
                self.analysis_thread.loc_model = LocalizationModel(loc_path)
            except Exception as e:
                self.lbl_model_status.setProperty("role", "statusError")
                self.lbl_model_status.setText(f"Localization model load failed: {e}")
                self._repolish(self.lbl_model_status)
                return

        self.lbl_model_status.setProperty("role", "statusSuccess")
        msg = "Segmentation model loaded."
        msg += " Localization model loaded." if loc_path else " No localization model (location will be unavailable)."
        self.lbl_model_status.setText(msg)
        self._repolish(self.lbl_model_status)

        if self._series_frames:
            self._set_playback_enabled(True)
            self.analysis_thread.seek(self.slider_frame.value())

    # ── Viewer ───────────────────────────────────────────────────────
    def _on_toggle_mask(self, checked: bool):
        self.panel_mask.setVisible(checked)

    def _on_frame_ready(self, original_rgb, mask_overlay_rgb, qca_vis_rgb, frame_idx, total_frames, latency_ms, info):
        w = self.label_original.width()
        h = self.label_original.height()
        self.label_original.setPixmap(ndarray_to_qpixmap(original_rgb, w, h))
        self.label_qca.setPixmap(ndarray_to_qpixmap(qca_vis_rgb, w, h))
        self.label_mask.setPixmap(ndarray_to_qpixmap(mask_overlay_rgb, w, h))

        self.slider_frame.blockSignals(True)
        self.slider_frame.setValue(frame_idx)
        self.slider_frame.blockSignals(False)
        self.lbl_frame_count.setText(f"Frame {frame_idx + 1} / {total_frames}")
        self.lbl_latency.setText(f"Latency: {latency_ms:.0f} ms")

        if "SEVERE" in info:
            self.lbl_stenosis.setStyleSheet("color: #E5484D; font-weight: bold;")
        elif "SIGNIFICANT" in info:
            self.lbl_stenosis.setStyleSheet("color: #F5A623; font-weight: bold;")
        elif "MODERATE" in info:
            self.lbl_stenosis.setStyleSheet("color: #DBB40C; font-weight: bold;")
        elif "MILD" in info:
            self.lbl_stenosis.setStyleSheet("color: #BAED91; font-weight: bold;")
        else:
            self.lbl_stenosis.setStyleSheet("color: #8A8F98;")
        self.lbl_stenosis.setText(info)

    def _on_playback_finished(self):
        self.btn_play.setEnabled(True)
        self.btn_pause.setEnabled(False)
        self._auto_run_key_frame_preview()

    def _auto_run_key_frame_preview(self):
        """
        Runs the same whole-series analysis as "Save Results & Generate View
        Report" automatically once playback reaches the end, so Key Frames
        populates without an extra click. Unlike that button, this does NOT
        write anything to disk or prompt for a view label -- it's a live
        preview only; saving the PDF/results.json is still a deliberate,
        separate action via the Save & Report card.
        """
        if self._view_analysis_thread is not None and self._view_analysis_thread.isRunning():
            return
        if not self._series_frames or self.analysis_thread.seg_model is None:
            return

        row = self.list_series.currentRow()
        label = self._series[row].description if 0 <= row < len(self._series) else "View"

        self._view_analysis_is_preview_only = True
        self.lbl_key_frame_coverage.setText("Analyzing full series for key frames...")

        self._view_analysis_thread = _ViewAnalysisThread(
            list(self._series_frames), label,
            self.analysis_thread.seg_model, self.analysis_thread.loc_model,
            self.analysis_thread.qca_cfg, self.analysis_thread.threshold,
        )
        self._view_analysis_thread.progress.connect(self.lbl_key_frame_coverage.setText)
        self._view_analysis_thread.finished_ok.connect(self._on_view_analysis_finished)
        self._view_analysis_thread.error.connect(self._on_view_analysis_error)
        self._view_analysis_thread.start()

    def _on_analysis_error(self, message: str):
        self.lbl_stenosis.setStyleSheet("color: #E5484D; font-weight: bold;")
        self.lbl_stenosis.setText(f"Error: {message}")

    # ── Playback controls ────────────────────────────────────────────
    def _set_playback_enabled(self, enabled: bool):
        for w in (self.slider_frame, self.btn_prev_frame, self.btn_play, self.btn_pause, self.btn_next_frame):
            w.setEnabled(enabled)
        if enabled:
            self.btn_pause.setEnabled(False)

    def _play(self):
        self.btn_play.setEnabled(False)
        self.btn_pause.setEnabled(True)
        self.analysis_thread.play()

    def _pause(self):
        self.analysis_thread.pause()
        self.btn_play.setEnabled(True)
        self.btn_pause.setEnabled(False)

    def _step_prev(self):
        idx = max(0, self.slider_frame.value() - 1)
        self.analysis_thread.seek(idx)

    def _step_next(self):
        idx = min(self.slider_frame.value() + 1, self.slider_frame.maximum())
        self.analysis_thread.seek(idx)

    def _on_slider_value_changed(self, value: int):
        total = self.slider_frame.maximum() + 1
        self.lbl_frame_count.setText(f"Frame {value + 1} / {total}")

    def _on_slider_released(self):
        self.analysis_thread.seek(self.slider_frame.value())

    def _on_threshold_changed(self, value: int):
        t = value / 100.0
        self.analysis_thread.threshold = t
        self.lbl_threshold_val.setText(f"{t:.2f}")

    # ── Save & Report ────────────────────────────────────────────────
    def _current_case(self):
        row = self.combo_case.currentIndex()
        if row < 0 or row >= len(self._cases):
            return None
        return self._cases[row]

    def _save_view_results(self):
        if self._view_analysis_thread is not None and self._view_analysis_thread.isRunning():
            return
        if not self._series_frames:
            self.lbl_save_status.setText("Load a series first.")
            return
        if self.analysis_thread.seg_model is None:
            self.lbl_save_status.setText("Load AI models first.")
            return
        case = self._current_case()
        if case is None:
            self.lbl_save_status.setText("Select a patient case first.")
            return

        row = self.list_series.currentRow()
        default_label = self._series[row].description if 0 <= row < len(self._series) else "View"

        label, ok = QInputDialog.getText(
            self, "View Label", "Label for this angiographic view (e.g. RAO 30 / CRA 20):",
            text=default_label,
        )
        if not ok:
            return
        label = label.strip() or default_label

        self._pending_case_id = case["case_id"]
        self._view_analysis_is_preview_only = False
        self.btn_open_view_report.setVisible(False)
        self._set_save_busy(True)
        self.lbl_save_status.setText(f"Analyzing full series for '{label}'...")

        self._view_analysis_thread = _ViewAnalysisThread(
            list(self._series_frames), label,
            self.analysis_thread.seg_model, self.analysis_thread.loc_model,
            self.analysis_thread.qca_cfg, self.analysis_thread.threshold,
        )
        self._view_analysis_thread.progress.connect(self.lbl_save_status.setText)
        self._view_analysis_thread.finished_ok.connect(self._on_view_analysis_finished)
        self._view_analysis_thread.error.connect(self._on_view_analysis_error)
        self._view_analysis_thread.start()

    def _on_view_analysis_finished(self, angle_result):
        self._update_key_frames(angle_result)

        if self._view_analysis_is_preview_only:
            return  # auto preview after playback -- key frames only, no disk save/prompt

        case_id = self._pending_case_id
        analysis_dir = patient_store.get_case_analysis_dir(case_id)
        patient_info = patient_store.load_metadata(case_id)

        try:
            view_dir = analysis_results_store.save_view_results(
                analysis_dir, angle_result, patient_info, self.analysis_thread.qca_cfg
            )
        except Exception as e:
            self._set_save_busy(False)
            self.lbl_save_status.setText(f"Failed to save results: {e}")
            QMessageBox.warning(self, "Save Failed", str(e))
            return

        self._set_save_busy(False)
        self._last_view_report = view_dir / "view_report.pdf"
        n_lesions = len(angle_result.tracks)
        self.lbl_save_status.setText(
            f"Saved: {n_lesions} lesion(s) found across {angle_result.n_frames_analyzed} frames. "
            f"Results + view report saved to {view_dir}"
        )
        self.btn_open_view_report.setVisible(True)
        self._refresh_final_report_status()

    def _on_view_analysis_error(self, message: str):
        if self._view_analysis_is_preview_only:
            self.lbl_key_frame_coverage.setText(f"Key frame preview failed: {message}")
            return
        self._set_save_busy(False)
        self.lbl_save_status.setText(f"Analysis failed: {message}")
        QMessageBox.warning(self, "Analysis Failed", message)

    def _set_save_busy(self, busy: bool):
        self.btn_save_view.setEnabled(not busy)

    def _open_view_report(self):
        if self._last_view_report and hasattr(os, "startfile"):
            try:
                os.startfile(self._last_view_report)
            except OSError:
                pass

    def _refresh_final_report_status(self):
        case = self._current_case()
        if case is None:
            self.btn_final_report.setEnabled(False)
            self.lbl_final_status.setText("Select a patient case first.")
            return

        analysis_dir = patient_store.get_case_analysis_dir(case["case_id"])
        views = analysis_results_store.list_view_results(analysis_dir)
        self.btn_final_report.setEnabled(len(views) > 0)
        self.btn_open_final_report.setVisible(False)
        if views:
            names = ", ".join(v["view_label"] for v in views)
            self.lbl_final_status.setText(f"{len(views)} view(s) saved for this case: {names}")
        else:
            self.lbl_final_status.setText("No views saved yet for this case.")

    def _generate_final_report(self):
        case = self._current_case()
        if case is None:
            return
        case_id = case["case_id"]
        case_dir = patient_store.get_case_dir(case_id)
        analysis_dir = patient_store.get_case_analysis_dir(case_id)
        patient_info = patient_store.load_metadata(case_id)

        try:
            out_path = analysis_results_store.generate_final_clinical_report(
                case_dir, analysis_dir, patient_info, self.analysis_thread.qca_cfg
            )
        except Exception as e:
            self.lbl_final_status.setText(f"Failed to generate final report: {e}")
            QMessageBox.warning(self, "Report Generation Failed", str(e))
            return

        self._last_final_report = out_path
        self.lbl_final_status.setText(f"Final clinical diagnosis report saved to {out_path}")
        self.btn_open_final_report.setVisible(True)

    def _open_final_report(self):
        if self._last_final_report and hasattr(os, "startfile"):
            try:
                os.startfile(self._last_final_report)
            except OSError:
                pass

    # ── Cleanup ──────────────────────────────────────────────────────
    def shutdown(self):
        self.analysis_thread.stop()
        self.analysis_thread.wait()

    @staticmethod
    def _repolish(widget):
        widget.style().unpolish(widget)
        widget.style().polish(widget)
