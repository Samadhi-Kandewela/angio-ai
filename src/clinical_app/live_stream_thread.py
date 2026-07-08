"""
Background worker thread for the Live Stream Analysis page.

Connects to a remote MJPEG (or any OpenCV-compatible) video stream URL,
continuously captures incoming frames, runs the full segmentation + QCA
inference pipeline on each one, and emits Qt signals carrying the processed
results back to the UI thread.  This mirrors the design of
DicomAnalysisThread but operates on a live, unbounded stream instead of a
finite, pre-loaded series of DICOM frames.

Differences from DicomAnalysisThread:
    - No seek/play/pause over a fixed frame array -- it's always "playing"
      (pulling frames from the stream) once started.
    - Frames arrive at the stream's native pace; analysis runs as fast as it
      can.  If the model is slower than the stream, we skip stale frames
      rather than queuing up an ever-growing backlog.
    - Connection lifecycle (connect / disconnect / reconnect) is managed
      explicitly since the source can go away at any time.
"""

import time
from collections import deque

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal

from qca import QCAConfig
from frame_pipeline import (
    preprocess_frame,
    segment_frame,
    run_localization_frame,
    clean_vessel_mask,
    run_qca_from_clean_mask,
)
from report_engine import draw_live_stenosis_overlay

# How many of the most recently received raw frames to keep for "Save Results
# & Generate View Report" -- a live stream is unbounded, so we can't hold
# every frame ever seen; a sliding window of the most recent ones gives the
# same whole-run QCA analysis (analyze_frame_list) a cine-loop-like clip to
# work with, capturing whatever was just observed.
CAPTURE_MAXLEN = 300


class LiveStreamThread(QThread):
    """
    Signals
    -------
    frame_ready(original_rgb, mask_overlay_rgb, qca_vis_rgb, frame_number, latency_ms, stenosis_info)
        Emitted once per analyzed frame.
    connection_status(str)
        Emitted when the connection state changes (connecting / connected /
        disconnected / error message).
    """

    frame_ready = Signal(np.ndarray, np.ndarray, np.ndarray, int, float, str)
    connection_status = Signal(str)

    def __init__(self):
        super().__init__()
        self._running = False
        self._stream_url = ""

        self.seg_model = None
        self.loc_model = None
        self.qca_cfg = QCAConfig(severe_threshold=70.0)
        self.threshold = 0.5
        self.overlay_color = [255, 0, 0]
        self.overlay_alpha = 0.4

        self._loc_class_map = None
        self._loc_confidence_map = None

        self._captured_frames = deque(maxlen=CAPTURE_MAXLEN)

    # ── Captured frames (for Save Results & Generate View Report) ──────

    def get_captured_frames(self) -> list:
        """Returns a snapshot (plain list) of the most recently received raw frames."""
        return list(self._captured_frames)

    def clear_captured_frames(self):
        self._captured_frames.clear()

    # ── Control API (call from the UI thread) ──────────────────────────

    def connect_to_stream(self, url: str):
        """Start (or restart) streaming from `url`."""
        self._stream_url = url
        self._loc_class_map = None
        self._loc_confidence_map = None
        if not self.isRunning():
            self._running = True
            self.start()
        else:
            # Already running — the run-loop will pick up the new URL
            pass

    def disconnect_stream(self):
        """Gracefully stop the stream loop."""
        self._running = False

    # ── Thread loop ────────────────────────────────────────────────────

    def run(self):
        self._running = True
        frame_number = 0

        while self._running:
            url = self._stream_url
            if not url:
                self.connection_status.emit("No stream URL configured.")
                self.msleep(500)
                continue

            self.connection_status.emit(f"Connecting to {url} ...")
            cap = cv2.VideoCapture(url)

            if not cap.isOpened():
                self.connection_status.emit(f"Failed to connect to {url}")
                # Wait before retrying so we don't busy-spin on a bad URL
                for _ in range(20):  # ~2 s
                    if not self._running:
                        break
                    self.msleep(100)
                continue

            self.connection_status.emit("Connected — streaming live.")

            while self._running and cap.isOpened():
                ret, frame_bgr = cap.read()
                if not ret:
                    self.connection_status.emit("Stream ended or interrupted. Reconnecting...")
                    break

                frame_number += 1
                self._captured_frames.append(frame_bgr.copy())

                if self.seg_model is None:
                    # No model loaded — still show the raw frame so the user
                    # can see the connection is alive.
                    rgb = cv2.cvtColor(
                        cv2.resize(frame_bgr, (512, 512)), cv2.COLOR_BGR2RGB
                    )
                    self.frame_ready.emit(
                        rgb, rgb, rgb, frame_number, 0.0,
                        "Models not loaded — showing raw stream.",
                    )
                    continue

                try:
                    self._process_frame(frame_bgr, frame_number)
                except Exception as e:
                    self.connection_status.emit(f"Frame {frame_number} error: {e}")

            cap.release()

        self.connection_status.emit("Disconnected.")

    def _process_frame(self, frame_bgr: np.ndarray, frame_number: int):
        t0 = time.perf_counter()

        img_rgb_original, img_rgb_enhanced, img_batch, img_gray = preprocess_frame(
            frame_bgr
        )

        mask_binary = segment_frame(self.seg_model, img_batch, self.threshold)

        # Localization (optional)
        if self.loc_model is not None:
            try:
                self._loc_class_map, self._loc_confidence_map = (
                    run_localization_frame(
                        self.loc_model, img_rgb_enhanced, mask_binary
                    )
                )
            except Exception:
                self._loc_class_map, self._loc_confidence_map = None, None

        # Mask overlay
        overlay = img_rgb_original.copy()
        color = np.array(self.overlay_color, dtype=np.uint8)
        overlay[mask_binary.astype(bool)] = color
        mask_overlay_rgb = cv2.addWeighted(
            overlay,
            self.overlay_alpha,
            img_rgb_original,
            1.0 - self.overlay_alpha,
            0,
        )

        # QCA
        bw = clean_vessel_mask(mask_binary, self.qca_cfg)
        branches, lesions, dt = run_qca_from_clean_mask(
            bw,
            self.qca_cfg,
            class_map=self._loc_class_map,
            confidence_map=self._loc_confidence_map,
            use_merged_labels=(
                self.loc_model.use_merged_labels
                if self.loc_model is not None
                else False
            ),
        )

        if branches:
            qca_vis_bgr = draw_live_stenosis_overlay(
                img_gray, bw, branches, lesions
            )
            qca_vis_rgb = cv2.cvtColor(qca_vis_bgr, cv2.COLOR_BGR2RGB)
            if lesions:
                top = lesions[0]
                loc_text = ""
                if "localization" in top:
                    loc = top["localization"]
                    loc_text = (
                        f"  Location={loc['group']} ({loc['label']})"
                        f"  LocConf={loc['confidence']:.2f}"
                    )
                info = (
                    f"Top Lesion: DS={top['DS_percent']:.1f}%  "
                    f"Severity={top['severity']}  "
                    f"MLD={top['MLD_px']:.1f}px  "
                    f"RVD={top['RVD_px']:.1f}px{loc_text}  "
                    f"| Total Lesions: {len(lesions)} across "
                    f"{len(branches)} branches"
                )
            else:
                info = (
                    f"No stenosis detected | "
                    f"{len(branches)} branches analyzed"
                )
        else:
            qca_vis_rgb = mask_overlay_rgb.copy()
            info = "QCA: Insufficient mask data for analysis"

        latency_ms = (time.perf_counter() - t0) * 1000.0
        self.frame_ready.emit(
            img_rgb_original,
            mask_overlay_rgb,
            qca_vis_rgb,
            frame_number,
            latency_ms,
            info,
        )
