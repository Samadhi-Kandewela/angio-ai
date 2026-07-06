"""
Real-Time Angiogram QCA Analysis Viewer
========================================
PyQt5 desktop application for real-time medical image segmentation
with integrated Quantitative Coronary Analysis (QCA).

Displays 3 panels:
  1. Original video frame
  2. Segmentation mask overlay
  3. Real-time QCA stenosis analysis

Usage:
    python src/desktop_app_qca.py
"""

import sys
import os
import time

# !! CRITICAL: Import onnxruntime BEFORE PyQt5 on Windows !!
try:
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QSlider, QComboBox, QFileDialog, QHBoxLayout, QVBoxLayout,
    QGroupBox, QStatusBar, QSizePolicy, QFrame, QLineEdit, QGridLayout,
    QListWidget, QListWidgetItem, QInputDialog, QMessageBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, pyqtSlot, QTimer
from PyQt5.QtGui import QImage, QPixmap, QFont, QColor, QPalette, QIcon

import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from qca import QCAConfig, draw_overlay
from frame_pipeline import (
    SegmentationModel, LocalizationModel,
    preprocess_frame, segment_frame, run_localization_frame, run_qca_frame,
)
from report_engine import analyze_angle_video
from pdf_report import render_clinical_report

DEFAULT_SEGMENTATION_MODEL_PATHS = [
    os.path.join(PROJECT_ROOT, "checkpoints", "mobileunetv3", "mobileunetv3_augmented_best.onnx"),
    os.path.join(PROJECT_ROOT, "checkpoints", "mobileunetv3", "mobileunetv3_augmented_best.pth"),
]
DEFAULT_LOCALIZATION_MODEL_PATHS = [
    os.path.join(PROJECT_ROOT, "checkpoints", "multitask_localization_v2", "multitask_latest.onnx"),
    os.path.join(PROJECT_ROOT, "checkpoints", "multitask_localization_v2", "multitask_latest.pth"),
    os.path.join(PROJECT_ROOT, "checkpoints", "multitask_localization_v2", "multitask_best.onnx"),
    os.path.join(PROJECT_ROOT, "checkpoints", "multitask_localization_v2", "multitask_best.pth"),
]


def first_existing_path(paths):
    for path in paths:
        if os.path.exists(path):
            return path
    return ""


DEFAULT_MODEL_PATH = first_existing_path(DEFAULT_SEGMENTATION_MODEL_PATHS)


# ─────────────────────────────────────────────────────────────────────────────
# Video + Inference + QCA Thread
# ─────────────────────────────────────────────────────────────────────────────
class VideoThread(QThread):
    """
    Background thread that reads video frames, runs segmentation inference,
    performs QCA analysis, and emits all three views to the GUI.
    """
    # Signals: original, mask_overlay, qca_overlay, fps, latency_ms, stenosis_info
    frame_ready = pyqtSignal(np.ndarray, np.ndarray, np.ndarray, float, float, str)
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self._running = False
        self._paused = False
        self.video_path = None
        self.model_path = None
        self.localization_model_path = None
        self.threshold = 0.5
        self.overlay_alpha = 0.4
        self.overlay_color = [255, 0, 0]  # Red (RGB)

        # Model handles (frame_pipeline.SegmentationModel / LocalizationModel)
        self._seg_model = None
        self._loc_model = None
        self._loc_class_map = None
        self._loc_confidence_map = None
        self._loc_frame_interval = 15
        self._frame_index = 0

        # Torch inference devices (set on load; CUDA when available)
        self._seg_device = None
        self._loc_device = None

        # QCA config
        self._qca_cfg = QCAConfig(severe_threshold=70.0)

    def load_model(self, model_path):
        """Load the segmentation model (ONNX or PyTorch) before starting the thread."""
        self.model_path = model_path
        try:
            self._seg_model = SegmentationModel(model_path)
            return True
        except Exception as e:
            self.error.emit(f"Model load error: {e}")
            return False

    def load_localization_model(self, model_path):
        """Load optional multitask localization model (ONNX or PyTorch)."""
        self.localization_model_path = model_path
        self._loc_model = None
        self._loc_class_map = None
        self._loc_confidence_map = None
        self._frame_index = 0

        if not model_path:
            return True

        try:
            self._loc_model = LocalizationModel(model_path)
            return True
        except Exception as e:
            self.error.emit(f"Localization model load error: {e}")
            return False

    def _run_qca(self, original_gray, mask_binary):
        """
        Run QCA analysis on a single frame.
        Returns (qca_overlay_bgr, stenosis_info_str) or (None, "") on failure.
        """
        try:
            branches, lesions, dt, bw = run_qca_frame(
                original_gray, mask_binary, self._qca_cfg,
                class_map=self._loc_class_map, confidence_map=self._loc_confidence_map,
            )

            if not branches:
                return None, ""

            # draw_overlay expects BGR input
            qca_vis = draw_overlay(original_gray, bw, branches, lesions)
            self._draw_localization_overlay(qca_vis, lesions)

            # Build stenosis info string
            if lesions:
                top = lesions[0]
                loc_text = "Location=unavailable"
                if "localization" in top:
                    loc = top["localization"]
                    loc_text = (
                        f"Location={loc['group']} ({loc['label']})  "
                        f"Artery={loc['artery']}  "
                        f"LocConf={loc['confidence']:.2f}"
                    )
                info = (
                    f"Top Lesion: DS={top['DS_percent']:.1f}%  "
                    f"Severity={top['severity']}  "
                    f"MLD={top['MLD_px']:.1f}px  "
                    f"RVD={top['RVD_px']:.1f}px  "
                    f"{loc_text}  "
                    f"| Total Lesions: {len(lesions)} across {len(branches)} branches"
                )
            else:
                info = f"No stenosis detected | {len(branches)} branches analyzed"

            return qca_vis, info

        except Exception:
            return None, ""

    @staticmethod
    def _draw_localization_overlay(qca_vis, lesions):
        """Draw anatomical localization near the top QCA lesion on the QCA overlay."""
        if qca_vis is None or not lesions:
            return

        top = lesions[0]
        loc = top.get("localization")
        if not loc:
            return

        y, x = top["min_pt"]
        conf = float(loc.get("confidence", 0.0))
        label = f"{loc.get('artery', 'unknown')} {loc.get('group', 'unknown')} {conf:.2f}"
        color = (0, 255, 0) if conf >= 0.70 else (0, 220, 255) if conf >= 0.45 else (0, 120, 255)

        h, w = qca_vis.shape[:2]
        x = int(np.clip(x, 0, w - 1))
        y = int(np.clip(y, 0, h - 1))

        cv2.circle(qca_vis, (x, y), 7, color, 2)
        cv2.line(qca_vis, (x, y), (min(w - 1, x + 22), max(0, y - 18)), color, 1)

        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.42
        thickness = 1
        (text_w, text_h), baseline = cv2.getTextSize(label, font, scale, thickness)
        text_x = int(np.clip(x + 26, 0, max(0, w - text_w - 6)))
        text_y = int(np.clip(y - 18, text_h + 6, h - baseline - 4))

        cv2.rectangle(
            qca_vis,
            (text_x - 3, text_y - text_h - 4),
            (text_x + text_w + 3, text_y + baseline + 3),
            (0, 0, 0),
            -1,
        )
        cv2.putText(qca_vis, label, (text_x, text_y), font, scale, color, thickness, cv2.LINE_AA)

    def run(self):
        """Main loop: read frames, infer, run QCA, and emit UI frames."""
        if not self.video_path or not os.path.exists(self.video_path):
            self.error.emit(f"Video not found: {self.video_path}")
            self.finished.emit()
            return

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            self.error.emit("Failed to open video.")
            self.finished.emit()
            return

        self._running = True

        while self._running and cap.isOpened():
            if self._paused:
                self.msleep(50)
                continue

            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            # ── Preprocessing ─────────────────────────────────────────
            img_rgb_original, img_rgb_enhanced, img_batch, img_gray = preprocess_frame(frame)

            # ── Inference ─────────────────────────────────────────────
            t_start = time.perf_counter()

            if self._seg_model is not None:
                mask_binary = segment_frame(self._seg_model, img_batch, self.threshold)
            else:
                mask_binary = np.zeros((512, 512), dtype=np.uint8)

            if self._loc_model is not None and (
                self._loc_class_map is None or self._frame_index % self._loc_frame_interval == 0
            ):
                try:
                    self._loc_class_map, self._loc_confidence_map = run_localization_frame(
                        self._loc_model, img_rgb_enhanced
                    )
                except Exception as e:
                    self.error.emit(f"Localization inference error: {e}")
                    self._loc_class_map = None
                    self._loc_confidence_map = None

            t_end = time.perf_counter()
            latency_ms = (t_end - t_start) * 1000.0
            fps = 1000.0 / max(latency_ms, 0.001)
            self._frame_index += 1

            # ── Panel 2: Mask Overlay ─────────────────────────────────
            overlay = img_rgb_original.copy()
            color = np.array(self.overlay_color, dtype=np.uint8)
            overlay[mask_binary == 1] = color
            mask_blended = cv2.addWeighted(
                overlay, self.overlay_alpha,
                img_rgb_original, 1.0 - self.overlay_alpha, 0
            )

            # ── Panel 3: QCA Analysis ─────────────────────────────────
            qca_vis_bgr, stenosis_info = self._run_qca(img_gray, mask_binary)

            if qca_vis_bgr is not None:
                # draw_overlay returns BGR; convert to RGB for Qt display
                qca_vis_rgb = cv2.cvtColor(qca_vis_bgr, cv2.COLOR_BGR2RGB)
            else:
                # Fallback: show mask overlay if QCA fails
                qca_vis_rgb = mask_blended.copy()
                if stenosis_info == "":
                    stenosis_info = "QCA: Insufficient mask data for analysis"

            # ── Emit all 3 panels ─────────────────────────────────────
            self.frame_ready.emit(
                img_rgb_original, mask_blended, qca_vis_rgb,
                fps, latency_ms, stenosis_info
            )

            self.msleep(1)

        cap.release()
        self.finished.emit()

    def pause(self):
        self._paused = True

    def resume(self):
        self._paused = False

    def stop(self):
        self._running = False
        self._paused = False


# ─────────────────────────────────────────────────────────────────────────────
# Clinical Report Thread
# ─────────────────────────────────────────────────────────────────────────────
class ReportThread(QThread):
    """
    Background thread that runs the offline whole-video QCA analysis for each
    queued angle/view and assembles the resulting clinical PDF report.
    """
    progress = pyqtSignal(str)
    finished_ok = pyqtSignal(str)   # emits the saved PDF path
    error = pyqtSignal(str)

    def __init__(self, angles, model_path, loc_model_path, patient_info, out_path, threshold=0.5):
        super().__init__()
        self.angles = angles  # list of (label, video_path)
        self.model_path = model_path
        self.loc_model_path = loc_model_path
        self.patient_info = patient_info
        self.out_path = out_path
        self.threshold = threshold

    def run(self):
        try:
            self.progress.emit("Loading segmentation model...")
            seg_model = SegmentationModel(self.model_path)

            loc_model = None
            if self.loc_model_path:
                self.progress.emit("Loading localization model...")
                loc_model = LocalizationModel(self.loc_model_path)

            cfg = QCAConfig(severe_threshold=70.0)
            angle_results = []

            for label, video_path in self.angles:
                self.progress.emit(f"Analyzing angle '{label}'...")

                def _cb(i, n, _label=label):
                    if n and (i % 10 == 0 or i == n):
                        self.progress.emit(f"Analyzing '{_label}': frame {i}/{n}")

                result = analyze_angle_video(
                    video_path, label, seg_model, loc_model, cfg,
                    threshold=self.threshold, progress_cb=_cb,
                )
                self.progress.emit(
                    f"'{label}': {len(result.tracks)} lesion(s) found across "
                    f"{result.n_frames_analyzed} frame(s)."
                )
                angle_results.append(result)

            self.progress.emit("Assembling PDF report...")
            out = render_clinical_report(self.out_path, self.patient_info, angle_results, cfg)

            self.finished_ok.emit(str(out))
        except Exception as e:
            self.error.emit(str(e))


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def ndarray_to_qpixmap(img_rgb: np.ndarray, target_w: int, target_h: int) -> QPixmap:
    """Convert an RGB numpy array to a QPixmap scaled to (target_w, target_h)."""
    h, w, ch = img_rgb.shape
    bytes_per_line = ch * w
    qimg = QImage(img_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg).scaled(
        target_w, target_h, Qt.KeepAspectRatio, Qt.SmoothTransformation
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main Window
# ─────────────────────────────────────────────────────────────────────────────
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Real-Time Angiogram QCA Analysis")
        self.setMinimumSize(1500, 750)
        self._apply_dark_theme()

        self.video_thread = VideoThread()
        self.video_thread.frame_ready.connect(self._on_frame)
        self.video_thread.finished.connect(self._on_finished)
        self.video_thread.error.connect(self._on_error)

        self._angles = []  # list of (label, video_path) queued for the clinical report
        self.report_thread = None

        self._build_ui()

    # ── Dark Theme ───────────────────────────────────────────────────────
    def _apply_dark_theme(self):
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(14, 17, 23))
        palette.setColor(QPalette.WindowText, QColor(250, 250, 250))
        palette.setColor(QPalette.Base, QColor(22, 27, 34))
        palette.setColor(QPalette.AlternateBase, QColor(30, 36, 44))
        palette.setColor(QPalette.ToolTipBase, Qt.white)
        palette.setColor(QPalette.ToolTipText, Qt.white)
        palette.setColor(QPalette.Text, QColor(250, 250, 250))
        palette.setColor(QPalette.Button, QColor(30, 36, 44))
        palette.setColor(QPalette.ButtonText, QColor(250, 250, 250))
        palette.setColor(QPalette.Highlight, QColor(0, 204, 102))
        palette.setColor(QPalette.HighlightedText, Qt.black)
        self.setPalette(palette)

        self.setStyleSheet("""
            QMainWindow { background-color: #0E1117; }
            QLabel { color: #FAFAFA; }
            QPushButton {
                background-color: #21262D;
                color: #FAFAFA;
                border: 1px solid #30363D;
                border-radius: 6px;
                padding: 8px 16px;
                font-weight: bold;
                font-size: 13px;
            }
            QPushButton:hover { background-color: #30363D; }
            QPushButton:pressed { background-color: #00CC66; color: #000; }
            QPushButton#playBtn { background-color: #238636; border-color: #2EA043; }
            QPushButton#playBtn:hover { background-color: #2EA043; }
            QPushButton#stopBtn { background-color: #DA3633; border-color: #F85149; }
            QPushButton#stopBtn:hover { background-color: #F85149; }
            QGroupBox {
                border: 1px solid #30363D;
                border-radius: 8px;
                margin-top: 12px;
                padding-top: 18px;
                font-weight: bold;
                color: #8B949E;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 6px;
            }
            QSlider::groove:horizontal {
                height: 6px;
                background: #30363D;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #00CC66;
                width: 16px;
                height: 16px;
                margin: -5px 0;
                border-radius: 8px;
            }
            QComboBox {
                background-color: #21262D;
                color: #FAFAFA;
                border: 1px solid #30363D;
                border-radius: 6px;
                padding: 6px 12px;
            }
            QLineEdit {
                background-color: #21262D;
                color: #FAFAFA;
                border: 1px solid #30363D;
                border-radius: 6px;
                padding: 6px 12px;
            }
            QStatusBar {
                background-color: #161B22;
                color: #8B949E;
                border-top: 1px solid #30363D;
            }
        """)

    # ── Build UI ─────────────────────────────────────────────────────────
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(16, 12, 16, 8)
        main_layout.setSpacing(10)

        # ── Title ────────────────────────────────────────────────────
        title = QLabel("AI Based Real-Time Analysis System For Coronary Angiography")
        title.setFont(QFont("Segoe UI", 22, QFont.Bold))
        title.setStyleSheet("color: #FFFFFF; padding-bottom: 4px;")
        main_layout.addWidget(title)

        # ── Video Panels (3 side-by-side) ────────────────────────────
        panels_layout = QHBoxLayout()
        panels_layout.setSpacing(10)

        # Panel 1: Original
        panel1_box = QGroupBox("Original Frame")
        panel1_layout = QVBoxLayout(panel1_box)
        self.label_original = QLabel("No video loaded")
        self.label_original.setAlignment(Qt.AlignCenter)
        self.label_original.setMinimumSize(380, 300)
        self.label_original.setStyleSheet(
            "background-color: #161B22; border-radius: 8px; "
            "font-size: 13px; color: #484F58;"
        )
        panel1_layout.addWidget(self.label_original)
        panels_layout.addWidget(panel1_box)

        # Panel 2: Segmentation Mask
        panel2_box = QGroupBox("Segmentation Mask")
        panel2_layout = QVBoxLayout(panel2_box)
        self.label_mask = QLabel("No video loaded")
        self.label_mask.setAlignment(Qt.AlignCenter)
        self.label_mask.setMinimumSize(380, 300)
        self.label_mask.setStyleSheet(
            "background-color: #161B22; border-radius: 8px; "
            "font-size: 13px; color: #484F58;"
        )
        panel2_layout.addWidget(self.label_mask)
        panels_layout.addWidget(panel2_box)

        # Panel 3: QCA Stenosis Analysis
        panel3_box = QGroupBox("QCA Stenosis Analysis")
        panel3_layout = QVBoxLayout(panel3_box)
        self.label_qca = QLabel("No video loaded")
        self.label_qca.setAlignment(Qt.AlignCenter)
        self.label_qca.setMinimumSize(380, 300)
        self.label_qca.setStyleSheet(
            "background-color: #161B22; border-radius: 8px; "
            "font-size: 13px; color: #484F58;"
        )
        panel3_layout.addWidget(self.label_qca)
        panels_layout.addWidget(panel3_box)

        main_layout.addLayout(panels_layout, stretch=1)

        # ── Stenosis Info Bar ────────────────────────────────────────
        stenosis_box = QGroupBox("Stenosis Report")
        stenosis_layout = QHBoxLayout(stenosis_box)
        self.lbl_stenosis = QLabel("No analysis yet - load a video and press Play")
        self.lbl_stenosis.setFont(QFont("Consolas", 12, QFont.Bold))
        self.lbl_stenosis.setStyleSheet("color: #FF6B6B; padding: 4px;")
        self.lbl_stenosis.setWordWrap(True)
        stenosis_layout.addWidget(self.lbl_stenosis)
        main_layout.addWidget(stenosis_box)

        # ── Metrics Bar ──────────────────────────────────────────────
        metrics_layout = QHBoxLayout()
        metrics_layout.setSpacing(24)

        self.lbl_fps = QLabel("FPS: --")
        self.lbl_fps.setFont(QFont("Consolas", 15, QFont.Bold))
        self.lbl_fps.setStyleSheet("color: #00CC66;")
        metrics_layout.addWidget(self.lbl_fps)

        self.lbl_latency = QLabel("Latency: -- ms")
        self.lbl_latency.setFont(QFont("Consolas", 15, QFont.Bold))
        self.lbl_latency.setStyleSheet("color: #FFA657;")
        metrics_layout.addWidget(self.lbl_latency)

        metrics_layout.addStretch()
        main_layout.addLayout(metrics_layout)

        # ── Controls ─────────────────────────────────────────────────
        ctrl_box = QGroupBox("Controls")
        ctrl_layout = QHBoxLayout(ctrl_box)
        ctrl_layout.setSpacing(10)

        # Video file
        self.btn_video = QPushButton("Open Video")
        self.btn_video.clicked.connect(self._browse_video)
        ctrl_layout.addWidget(self.btn_video)

        self.txt_video = QLineEdit()
        self.txt_video.setPlaceholderText("Select a video file...")
        self.txt_video.setReadOnly(True)
        self.txt_video.setMinimumWidth(160)
        ctrl_layout.addWidget(self.txt_video)

        # Model file (pre-filled with default)
        self.btn_model = QPushButton("Load Model")
        self.btn_model.clicked.connect(self._browse_model)
        ctrl_layout.addWidget(self.btn_model)

        self.txt_model = QLineEdit()
        self.txt_model.setPlaceholderText("Select .onnx or .pth model...")
        self.txt_model.setReadOnly(False)
        self.txt_model.setMinimumWidth(160)
        # Pre-fill with default model path
        if os.path.exists(DEFAULT_MODEL_PATH):
            self.txt_model.setText(DEFAULT_MODEL_PATH)
        ctrl_layout.addWidget(self.txt_model)

        self.btn_loc_model = QPushButton("Load Localization")
        self.btn_loc_model.clicked.connect(self._browse_localization_model)
        ctrl_layout.addWidget(self.btn_loc_model)

        self.txt_loc_model = QLineEdit()
        self.txt_loc_model.setPlaceholderText("Select multitask .onnx or .pth model...")
        self.txt_loc_model.setReadOnly(False)
        self.txt_loc_model.setMinimumWidth(160)
        default_loc = first_existing_path(DEFAULT_LOCALIZATION_MODEL_PATHS)
        if default_loc:
            self.txt_loc_model.setText(default_loc)
        ctrl_layout.addWidget(self.txt_loc_model)

        # Separator
        sep = QFrame()
        sep.setFrameShape(QFrame.VLine)
        sep.setStyleSheet("color: #30363D;")
        ctrl_layout.addWidget(sep)

        # Threshold slider
        ctrl_layout.addWidget(QLabel("Threshold:"))
        self.slider_threshold = QSlider(Qt.Horizontal)
        self.slider_threshold.setRange(0, 100)
        self.slider_threshold.setValue(50)
        self.slider_threshold.setFixedWidth(120)
        self.slider_threshold.valueChanged.connect(self._on_threshold_changed)
        ctrl_layout.addWidget(self.slider_threshold)
        self.lbl_threshold_val = QLabel("0.50")
        self.lbl_threshold_val.setFont(QFont("Consolas", 11))
        ctrl_layout.addWidget(self.lbl_threshold_val)

        # Overlay color
        ctrl_layout.addWidget(QLabel("Color:"))
        self.combo_color = QComboBox()
        self.combo_color.addItems(["Red", "Green", "Blue", "Yellow"])
        self.combo_color.currentTextChanged.connect(self._on_color_changed)
        ctrl_layout.addWidget(self.combo_color)

        main_layout.addWidget(ctrl_box)

        # ── Play / Pause / Stop ──────────────────────────────────────
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(10)

        self.btn_play = QPushButton("Play")
        self.btn_play.setObjectName("playBtn")
        self.btn_play.setFixedHeight(40)
        self.btn_play.clicked.connect(self._play)
        btn_layout.addWidget(self.btn_play)

        self.btn_pause = QPushButton("Pause")
        self.btn_pause.setFixedHeight(40)
        self.btn_pause.clicked.connect(self._pause)
        self.btn_pause.setEnabled(False)
        btn_layout.addWidget(self.btn_pause)

        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setObjectName("stopBtn")
        self.btn_stop.setFixedHeight(40)
        self.btn_stop.clicked.connect(self._stop)
        self.btn_stop.setEnabled(False)
        btn_layout.addWidget(self.btn_stop)

        btn_layout.addStretch()
        main_layout.addLayout(btn_layout)

        # ── Clinical Report ─────────────────────────────────────────
        report_box = QGroupBox("Clinical Report")
        report_layout = QVBoxLayout(report_box)
        report_layout.setSpacing(8)

        angles_row = QHBoxLayout()
        angles_row.setSpacing(10)

        self.list_angles = QListWidget()
        self.list_angles.setFixedHeight(80)
        angles_row.addWidget(self.list_angles, stretch=1)

        angle_btns = QVBoxLayout()
        self.btn_add_angle = QPushButton("Add Current Video as Angle")
        self.btn_add_angle.clicked.connect(self._add_angle)
        angle_btns.addWidget(self.btn_add_angle)

        self.btn_remove_angle = QPushButton("Remove Selected")
        self.btn_remove_angle.clicked.connect(self._remove_angle)
        angle_btns.addWidget(self.btn_remove_angle)
        angles_row.addLayout(angle_btns)

        report_layout.addLayout(angles_row)

        patient_row = QHBoxLayout()
        patient_row.setSpacing(10)
        patient_row.addWidget(QLabel("Patient ID:"))
        self.txt_patient_id = QLineEdit()
        self.txt_patient_id.setPlaceholderText("optional")
        patient_row.addWidget(self.txt_patient_id)

        patient_row.addWidget(QLabel("Study Date:"))
        self.txt_study_date = QLineEdit()
        self.txt_study_date.setPlaceholderText("optional")
        patient_row.addWidget(self.txt_study_date)

        patient_row.addWidget(QLabel("Operator:"))
        self.txt_operator = QLineEdit()
        self.txt_operator.setPlaceholderText("optional")
        patient_row.addWidget(self.txt_operator)
        report_layout.addLayout(patient_row)

        gen_row = QHBoxLayout()
        self.btn_generate_report = QPushButton("Generate Report")
        self.btn_generate_report.setFixedHeight(36)
        self.btn_generate_report.clicked.connect(self._generate_report)
        gen_row.addWidget(self.btn_generate_report)

        self.lbl_report_status = QLabel("No angles queued.")
        self.lbl_report_status.setStyleSheet("color: #8B949E; padding: 4px;")
        self.lbl_report_status.setWordWrap(True)
        gen_row.addWidget(self.lbl_report_status, stretch=1)
        report_layout.addLayout(gen_row)

        main_layout.addWidget(report_box)

        # ── Status Bar ───────────────────────────────────────────────
        self.statusBar().showMessage(
            "Ready. Load a video and model to begin. "
            "Model pre-loaded: mobileunetv3_augmented_best.pth"
        )

    # ── Slot: frame received ─────────────────────────────────────────
    @pyqtSlot(np.ndarray, np.ndarray, np.ndarray, float, float, str)
    def _on_frame(self, original, mask_overlay, qca_overlay, fps, latency_ms, stenosis_info):
        w = self.label_original.width()
        h = self.label_original.height()

        self.label_original.setPixmap(ndarray_to_qpixmap(original, w, h))
        self.label_mask.setPixmap(ndarray_to_qpixmap(mask_overlay, w, h))
        self.label_qca.setPixmap(ndarray_to_qpixmap(qca_overlay, w, h))

        self.lbl_fps.setText(f"FPS: {fps:.1f}")
        self.lbl_latency.setText(f"Latency: {latency_ms:.1f} ms")

        if stenosis_info:
            # Color-code the severity
            if "SEVERE" in stenosis_info:
                self.lbl_stenosis.setStyleSheet("color: #FF4444; padding: 4px; font-weight: bold;")
            elif "SIGNIFICANT" in stenosis_info:
                self.lbl_stenosis.setStyleSheet("color: #FFA657; padding: 4px; font-weight: bold;")
            elif "MODERATE" in stenosis_info:
                self.lbl_stenosis.setStyleSheet("color: #E8D44D; padding: 4px; font-weight: bold;")
            elif "MILD" in stenosis_info:
                self.lbl_stenosis.setStyleSheet("color: #00CC66; padding: 4px; font-weight: bold;")
            else:
                self.lbl_stenosis.setStyleSheet("color: #8B949E; padding: 4px;")
            self.lbl_stenosis.setText(stenosis_info)

    @pyqtSlot()
    def _on_finished(self):
        self.statusBar().showMessage("Video finished.")
        self.btn_play.setEnabled(True)
        self.btn_pause.setEnabled(False)
        self.btn_stop.setEnabled(False)

    @pyqtSlot(str)
    def _on_error(self, msg):
        self.statusBar().showMessage(f"Error: {msg}")

    # ── Browse files ─────────────────────────────────────────────────
    def _browse_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Angiogram Video", "",
            "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*)"
        )
        if path:
            self.txt_video.setText(path)
            self.statusBar().showMessage(f"Video loaded: {os.path.basename(path)}")

    def _browse_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Model", "",
            "Model Files (*.onnx *.pth);;All Files (*)"
        )
        if path:
            self.txt_model.setText(path)
            self.statusBar().showMessage(f"Model loaded: {os.path.basename(path)}")

    def _browse_localization_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Localization Model", "",
            "Model Files (*.onnx *.pth);;All Files (*)"
        )
        if path:
            self.txt_loc_model.setText(path)
            self.statusBar().showMessage(f"Localization model loaded: {os.path.basename(path)}")

    # ── Controls ─────────────────────────────────────────────────────
    def _play(self):
        video_path = self.txt_video.text()
        model_path = self.txt_model.text()
        loc_model_path = self.txt_loc_model.text().strip()

        if not video_path:
            self.statusBar().showMessage("Please select a video file first.")
            return
        if not model_path:
            self.statusBar().showMessage("Please select a model file first.")
            return

        # If thread is paused, just resume
        if self.video_thread.isRunning():
            self.video_thread.resume()
            self.btn_pause.setEnabled(True)
            self.btn_play.setEnabled(False)
            self.statusBar().showMessage("Resumed.")
            return

        # Load model
        ok = self.video_thread.load_model(model_path)
        if not ok:
            return
        ok = self.video_thread.load_localization_model(loc_model_path)
        if not ok:
            return

        self.video_thread.video_path = video_path
        self.video_thread.start()

        self.btn_play.setEnabled(False)
        self.btn_pause.setEnabled(True)
        self.btn_stop.setEnabled(True)
        if loc_model_path:
            self.statusBar().showMessage("Streaming with QCA and anatomical localization...")
        else:
            self.statusBar().showMessage("Streaming with QCA analysis...")

    def _pause(self):
        self.video_thread.pause()
        self.btn_play.setEnabled(True)
        self.btn_pause.setEnabled(False)
        self.statusBar().showMessage("Paused.")

    def _stop(self):
        self.video_thread.stop()
        self.video_thread.wait()
        self.btn_play.setEnabled(True)
        self.btn_pause.setEnabled(False)
        self.btn_stop.setEnabled(False)
        self.label_original.clear()
        self.label_original.setText("No video loaded")
        self.label_mask.clear()
        self.label_mask.setText("No video loaded")
        self.label_qca.clear()
        self.label_qca.setText("No video loaded")
        self.lbl_fps.setText("FPS: --")
        self.lbl_latency.setText("Latency: -- ms")
        self.lbl_stenosis.setText("No analysis yet - load a video and press Play")
        self.lbl_stenosis.setStyleSheet("color: #FF6B6B; padding: 4px;")
        self.statusBar().showMessage("Stopped.")

    def _on_threshold_changed(self, value):
        t = value / 100.0
        self.video_thread.threshold = t
        self.lbl_threshold_val.setText(f"{t:.2f}")

    def _on_color_changed(self, text):
        color_map = {
            "Green": [0, 255, 0],
            "Red":   [255, 0, 0],
            "Blue":  [0, 100, 255],
            "Yellow": [255, 255, 0],
        }
        self.video_thread.overlay_color = color_map.get(text, [0, 255, 0])

    # ── Clinical Report ──────────────────────────────────────────────
    def _add_angle(self):
        video_path = self.txt_video.text()
        if not video_path:
            self.statusBar().showMessage("Select a video first, then add it as an angle.")
            return

        suggested = f"View {len(self._angles) + 1}"
        label, ok = QInputDialog.getText(
            self, "Angle Label", "Short label for this angiographic view (e.g. RAO 30 / CRA 20):",
            text=suggested,
        )
        if not ok:
            return
        label = label.strip() or suggested

        self._angles.append((label, video_path))
        self.list_angles.addItem(QListWidgetItem(f"{label} — {os.path.basename(video_path)}"))
        self.lbl_report_status.setText(f"{len(self._angles)} angle(s) queued.")

    def _remove_angle(self):
        row = self.list_angles.currentRow()
        if row < 0:
            return
        self.list_angles.takeItem(row)
        del self._angles[row]
        self.lbl_report_status.setText(f"{len(self._angles)} angle(s) queued.")

    def _generate_report(self):
        if self.report_thread is not None and self.report_thread.isRunning():
            self.statusBar().showMessage("A report is already being generated.")
            return
        if not self._angles:
            self.statusBar().showMessage("Add at least one angle video before generating a report.")
            return

        model_path = self.txt_model.text()
        if not model_path:
            self.statusBar().showMessage("Please select a segmentation model first.")
            return
        loc_model_path = self.txt_loc_model.text().strip()

        default_name = f"qca_report_{time.strftime('%Y%m%d_%H%M%S')}.pdf"
        out_path, _ = QFileDialog.getSaveFileName(
            self, "Save Clinical Report", default_name, "PDF Files (*.pdf)"
        )
        if not out_path:
            return

        patient_info = {
            "patient_id": self.txt_patient_id.text().strip(),
            "study_date": self.txt_study_date.text().strip(),
            "operator": self.txt_operator.text().strip(),
        }

        self.btn_generate_report.setEnabled(False)
        self.lbl_report_status.setText("Starting report generation...")

        self.report_thread = ReportThread(
            list(self._angles), model_path, loc_model_path, patient_info, out_path,
            threshold=self.video_thread.threshold,
        )
        self.report_thread.progress.connect(self._on_report_progress)
        self.report_thread.finished_ok.connect(self._on_report_finished)
        self.report_thread.error.connect(self._on_report_error)
        self.report_thread.start()

    @pyqtSlot(str)
    def _on_report_progress(self, msg):
        self.lbl_report_status.setText(msg)

    @pyqtSlot(str)
    def _on_report_finished(self, out_path):
        self.btn_generate_report.setEnabled(True)
        self.lbl_report_status.setText(f"Report saved to {out_path}")
        self.statusBar().showMessage(f"Report saved to {out_path}")
        if hasattr(os, "startfile"):
            try:
                os.startfile(out_path)
            except OSError:
                pass

    @pyqtSlot(str)
    def _on_report_error(self, msg):
        self.btn_generate_report.setEnabled(True)
        self.lbl_report_status.setText(f"Report generation failed: {msg}")
        QMessageBox.warning(self, "Report Generation Failed", msg)

    # ── Cleanup ──────────────────────────────────────────────────────
    def closeEvent(self, event):
        self.video_thread.stop()
        self.video_thread.wait()
        if self.report_thread is not None:
            self.report_thread.wait()
        event.accept()



# ─────────────────────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setFont(QFont("Segoe UI", 10))
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
