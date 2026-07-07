"""
Live Stream Analysis page for the Angio-AI Clinical Dashboard.

Connects to a remote MJPEG (or any OpenCV-compatible) video stream URL —
typically the simulated stream server (simulate_stream.py) — and runs real-
time vessel segmentation and QCA analysis on every incoming frame, proving
the system's capability to analyze live video feeds, not just static DICOM
files.

Layout mirrors the Local DICOM Analysis page's visual language (cards,
section headers, accent buttons) but replaces the DICOM case/series picker
with a stream URL input and connect/disconnect controls, and replaces the
frame slider with live status telemetry (FPS, latency, connection state,
frame counter).
"""

import os
from pathlib import Path
from collections import deque
import time

import cv2
import numpy as np

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage, QPixmap, QFont
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QFrame,
    QFileDialog,
    QCheckBox,
    QSlider,
    QSizePolicy,
    QScrollArea,
)

from frame_pipeline import SegmentationModel, LocalizationModel
from live_stream_thread import LiveStreamThread


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

DEFAULT_STREAM_URL = "http://localhost:8080/stream"


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


class LiveStreamPage(QWidget):
    """Live stream analysis page: stream URL + AI models + real-time viewer."""

    go_to_new_patient = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)

        # FPS tracking
        self._fps_timestamps = deque(maxlen=30)
        self._latest_latency = 0.0

        # Stream thread
        self._stream_thread = LiveStreamThread()
        self._stream_thread.frame_ready.connect(self._on_frame_ready)
        self._stream_thread.connection_status.connect(self._on_connection_status)

        self._connected = False
        self._build_ui()

    # ── UI construction ────────────────────────────────────────────────

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

        title = QLabel("Live Stream Analysis")
        title.setProperty("role", "pageTitle")
        layout.addWidget(title)

        subtitle = QLabel(
            "Connect to a live MJPEG video stream (e.g. the simulate_stream.py server) and "
            "run real-time AI vessel segmentation and QCA stenosis analysis on every incoming "
            "frame. This demonstrates Angio-AI's ability to process live video feeds — not "
            "just stored DICOM files."
        )
        subtitle.setProperty("role", "pageSubtitle")
        subtitle.setWordWrap(True)
        layout.addWidget(subtitle)

        layout.addWidget(self._build_connection_card())
        layout.addWidget(self._build_model_card())
        layout.addWidget(self._build_viewer_card())
        layout.addWidget(self._build_telemetry_card())

    # ── Connection card ────────────────────────────────────────────────

    def _build_connection_card(self) -> QFrame:
        card = _card()
        v = QVBoxLayout(card)
        v.setContentsMargins(20, 16, 20, 16)
        v.setSpacing(10)

        header = QLabel("STREAM CONNECTION")
        header.setProperty("role", "sectionHeader")
        v.addWidget(header)

        row = QHBoxLayout()
        row.addWidget(_field_label("Stream URL"))
        self.txt_stream_url = QLineEdit(DEFAULT_STREAM_URL)
        self.txt_stream_url.setPlaceholderText("http://host:port/stream")
        row.addWidget(self.txt_stream_url, stretch=1)
        v.addLayout(row)

        btn_row = QHBoxLayout()
        self.btn_connect = QPushButton("Connect")
        self.btn_connect.setProperty("variant", "primary")
        self.btn_connect.clicked.connect(self._toggle_connection)
        btn_row.addWidget(self.btn_connect)

        self.lbl_connection_status = QLabel("Disconnected.")
        self.lbl_connection_status.setProperty("role", "hint")
        btn_row.addWidget(self.lbl_connection_status, stretch=1)
        v.addLayout(btn_row)

        hint = QLabel(
            "Start the simulator first:  python src/scripts/simulate_stream.py "
            "--video path/to/angiogram.mp4 --port 8080"
        )
        hint.setProperty("role", "hint")
        hint.setWordWrap(True)
        v.addWidget(hint)

        return card

    # ── Model card ─────────────────────────────────────────────────────

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
        self.txt_seg_model = QLineEdit(
            _first_existing(DEFAULT_SEGMENTATION_MODEL_PATHS)
        )
        grid.addWidget(self.txt_seg_model, 0, 1)
        self.btn_browse_seg = QPushButton("Browse...")
        self.btn_browse_seg.setProperty("variant", "ghost")
        self.btn_browse_seg.clicked.connect(self._browse_seg_model)
        grid.addWidget(self.btn_browse_seg, 0, 2)

        grid.addWidget(_field_label("Localization Model (optional)"), 1, 0)
        self.txt_loc_model = QLineEdit(
            _first_existing(DEFAULT_LOCALIZATION_MODEL_PATHS)
        )
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

        # Threshold slider
        thresh_row = QHBoxLayout()
        thresh_row.addWidget(_field_label("Threshold:"))
        self.slider_threshold = QSlider(Qt.Horizontal)
        self.slider_threshold.setRange(0, 100)
        self.slider_threshold.setValue(50)
        self.slider_threshold.setFixedWidth(150)
        self.slider_threshold.valueChanged.connect(self._on_threshold_changed)
        thresh_row.addWidget(self.slider_threshold)
        self.lbl_threshold_val = QLabel("0.50")
        self.lbl_threshold_val.setFont(QFont("Consolas", 10))
        thresh_row.addWidget(self.lbl_threshold_val)
        thresh_row.addStretch()
        v.addLayout(thresh_row)

        return card

    # ── Viewer card ────────────────────────────────────────────────────

    def _build_viewer_card(self) -> QFrame:
        card = _card()
        v = QVBoxLayout(card)
        v.setContentsMargins(20, 16, 20, 16)
        v.setSpacing(10)

        top_row = QHBoxLayout()
        header = QLabel("LIVE ANALYSIS VIEWER")
        header.setProperty("role", "sectionHeader")
        top_row.addWidget(header)
        top_row.addStretch()

        self.chk_show_mask = QCheckBox("Show Segmentation Mask")
        self.chk_show_mask.toggled.connect(self._on_toggle_mask)
        top_row.addWidget(self.chk_show_mask)
        v.addLayout(top_row)

        panels = QHBoxLayout()
        panels.setSpacing(10)

        self.panel_original, self.label_original = self._build_image_panel(
            "Original Frame"
        )
        panels.addWidget(self.panel_original, stretch=1)

        self.panel_qca, self.label_qca = self._build_image_panel(
            "QCA Stenosis Analysis"
        )
        panels.addWidget(self.panel_qca, stretch=1)

        self.panel_mask, self.label_mask = self._build_image_panel(
            "Segmentation Mask"
        )
        self.panel_mask.setVisible(False)
        panels.addWidget(self.panel_mask, stretch=1)

        v.addLayout(panels, stretch=1)

        self.lbl_stenosis = QLabel(
            "No analysis yet — load models and connect to a stream."
        )
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
        label.setStyleSheet(
            "background-color: #0B0C0E; border-radius: 6px; color: #5C6067;"
        )
        v.addWidget(label, stretch=1)

        return box, label

    # ── Telemetry card ─────────────────────────────────────────────────

    def _build_telemetry_card(self) -> QFrame:
        card = _card()
        v = QVBoxLayout(card)
        v.setContentsMargins(20, 16, 20, 16)
        v.setSpacing(10)

        header = QLabel("LIVE TELEMETRY")
        header.setProperty("role", "sectionHeader")
        v.addWidget(header)

        grid = QGridLayout()
        grid.setHorizontalSpacing(40)
        grid.setVerticalSpacing(8)

        self._telem_labels = {}
        metrics = [
            ("Frames Processed", "frames"),
            ("Inference Latency", "latency"),
            ("Effective FPS", "fps"),
            ("Connection", "connection"),
        ]
        for col, (name, key) in enumerate(metrics):
            name_lbl = QLabel(name)
            name_lbl.setProperty("role", "fieldLabel")
            grid.addWidget(name_lbl, 0, col)

            val_lbl = QLabel("—")
            val_lbl.setFont(QFont("Consolas", 14, QFont.Bold))
            val_lbl.setStyleSheet("color: #BAED91;")
            grid.addWidget(val_lbl, 1, col)
            self._telem_labels[key] = val_lbl

        v.addLayout(grid)
        return card

    # ── Model loading ──────────────────────────────────────────────────

    def _browse_seg_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Segmentation Model", "", "Model Files (*.onnx *.pth)"
        )
        if path:
            self.txt_seg_model.setText(path)

    def _browse_loc_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Localization Model", "", "Model Files (*.onnx *.pth)"
        )
        if path:
            self.txt_loc_model.setText(path)

    def _load_models(self):
        seg_path = self.txt_seg_model.text().strip()
        if not seg_path:
            self.lbl_model_status.setText("Select a segmentation model first.")
            return

        try:
            self._stream_thread.seg_model = SegmentationModel(seg_path)
        except Exception as e:
            self.lbl_model_status.setText(f"Seg model error: {e}")
            return

        loc_path = self.txt_loc_model.text().strip()
        if loc_path:
            try:
                self._stream_thread.loc_model = LocalizationModel(loc_path)
            except Exception as e:
                self._stream_thread.loc_model = None
                self.lbl_model_status.setText(
                    f"Seg model loaded. Loc model error: {e}"
                )
                return

        loc_status = (
            " + Localization" if self._stream_thread.loc_model else ""
        )
        self.lbl_model_status.setText(f"Segmentation{loc_status} model loaded ✓")
        self.lbl_model_status.setProperty("role", "statusSuccess")
        self.lbl_model_status.style().unpolish(self.lbl_model_status)
        self.lbl_model_status.style().polish(self.lbl_model_status)

    def _on_threshold_changed(self, value):
        t = value / 100.0
        self.lbl_threshold_val.setText(f"{t:.2f}")
        self._stream_thread.threshold = t

    # ── Connection toggle ──────────────────────────────────────────────

    def _toggle_connection(self):
        if self._connected:
            self._disconnect()
        else:
            self._connect()

    def _connect(self):
        url = self.txt_stream_url.text().strip()
        if not url:
            self.lbl_connection_status.setText("Enter a stream URL first.")
            return

        self._connected = True
        self.btn_connect.setText("Disconnect")
        self.btn_connect.setProperty("variant", "")
        self.btn_connect.style().unpolish(self.btn_connect)
        self.btn_connect.style().polish(self.btn_connect)
        self.txt_stream_url.setEnabled(False)

        self._fps_timestamps.clear()
        self._stream_thread.connect_to_stream(url)

    def _disconnect(self):
        self._connected = False
        self._stream_thread.disconnect_stream()

        self.btn_connect.setText("Connect")
        self.btn_connect.setProperty("variant", "primary")
        self.btn_connect.style().unpolish(self.btn_connect)
        self.btn_connect.style().polish(self.btn_connect)
        self.txt_stream_url.setEnabled(True)

    # ── Signal handlers ────────────────────────────────────────────────

    def _on_frame_ready(
        self,
        original_rgb: np.ndarray,
        mask_overlay_rgb: np.ndarray,
        qca_vis_rgb: np.ndarray,
        frame_number: int,
        latency_ms: float,
        stenosis_info: str,
    ):
        now = time.time()
        self._fps_timestamps.append(now)
        self._latest_latency = latency_ms

        # Calculate effective FPS from the timestamps deque
        if len(self._fps_timestamps) >= 2:
            elapsed = self._fps_timestamps[-1] - self._fps_timestamps[0]
            if elapsed > 0:
                fps = (len(self._fps_timestamps) - 1) / elapsed
            else:
                fps = 0.0
        else:
            fps = 0.0

        # Update image panels
        w_orig = self.label_original.width()
        h_orig = self.label_original.height()
        self.label_original.setPixmap(
            ndarray_to_qpixmap(original_rgb, w_orig, h_orig)
        )

        w_qca = self.label_qca.width()
        h_qca = self.label_qca.height()
        self.label_qca.setPixmap(
            ndarray_to_qpixmap(qca_vis_rgb, w_qca, h_qca)
        )

        if self.panel_mask.isVisible():
            w_mask = self.label_mask.width()
            h_mask = self.label_mask.height()
            self.label_mask.setPixmap(
                ndarray_to_qpixmap(mask_overlay_rgb, w_mask, h_mask)
            )

        # Update telemetry
        self._telem_labels["frames"].setText(str(frame_number))
        self._telem_labels["latency"].setText(f"{latency_ms:.1f} ms")
        self._telem_labels["fps"].setText(f"{fps:.1f}")

        # Stenosis info
        self.lbl_stenosis.setText(stenosis_info)

    def _on_connection_status(self, status: str):
        self.lbl_connection_status.setText(status)
        self._telem_labels["connection"].setText(status)

        # If the thread signals disconnection externally, reset the button
        if "Disconnected" in status and self._connected:
            self._connected = False
            self.btn_connect.setText("Connect")
            self.btn_connect.setProperty("variant", "primary")
            self.btn_connect.style().unpolish(self.btn_connect)
            self.btn_connect.style().polish(self.btn_connect)
            self.txt_stream_url.setEnabled(True)

    def _on_toggle_mask(self, show: bool):
        self.panel_mask.setVisible(show)

    # ── Shutdown ───────────────────────────────────────────────────────

    def shutdown(self):
        """Called by AppWindow.closeEvent to cleanly stop the stream thread."""
        self._stream_thread.disconnect_stream()
        self._stream_thread.wait(3000)
