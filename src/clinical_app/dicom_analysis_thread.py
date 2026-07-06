"""
Background analysis thread for the Local DICOM Analysis page.

Holds a loaded DICOM series' frames in memory and runs segmentation + QCA on
individual frames on demand -- either advancing through Play, or jumping to
an arbitrary frame via Seek (e.g. dragging the frame slider) -- reusing the
exact same frame_pipeline path as the live preview (desktop_app_qca.py) and
the offline QCA report generator, so results are consistent everywhere.

Per-frame QCA (skeletonization, branch decomposition, sub-pixel lesion
refinement) takes on the order of a second, so this deliberately does not
try to play back at the source frame rate -- each frame is fully analyzed
before the next is shown, paced by that analysis time itself.
"""
import time

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal

from qca import QCAConfig, draw_overlay
from frame_pipeline import preprocess_frame, segment_frame, run_localization_frame, run_qca_frame


class DicomAnalysisThread(QThread):
    # original_rgb, mask_overlay_rgb, qca_vis_rgb, frame_index, total_frames, latency_ms, stenosis_info
    frame_ready = Signal(np.ndarray, np.ndarray, np.ndarray, int, int, float, str)
    playback_finished = Signal()
    error = Signal(str)

    def __init__(self):
        super().__init__()
        self._running = False
        self._playing = False
        self._seek_to = None

        self.frames = []
        self._index = 0

        self.seg_model = None
        self.loc_model = None
        self.qca_cfg = QCAConfig(severe_threshold=70.0)
        self.threshold = 0.5
        self.overlay_color = [255, 0, 0]
        self.overlay_alpha = 0.4

        self._loc_class_map = None
        self._loc_confidence_map = None

    # ── Control API (safe to call from the UI thread) ──────────────
    def set_frames(self, frames):
        self.frames = frames
        self._index = 0
        self._loc_class_map = None
        self._loc_confidence_map = None

    def play(self):
        self._playing = True

    def pause(self):
        self._playing = False

    def seek(self, index: int):
        self._playing = False
        self._seek_to = index

    def stop(self):
        self._running = False
        self._playing = False

    # ── Thread loop ─────────────────────────────────────────────────
    def run(self):
        self._running = True
        while self._running:
            if self._seek_to is not None:
                idx = self._seek_to
                self._seek_to = None
                if self.frames:
                    self._index = max(0, min(idx, len(self.frames) - 1))
                    self._process_frame(self._index)
                continue

            if self._playing and self.frames:
                self._process_frame(self._index)
                self._index += 1
                if self._index >= len(self.frames):
                    self._index = len(self.frames) - 1
                    self._playing = False
                    self.playback_finished.emit()
            else:
                self.msleep(30)

    def _process_frame(self, idx):
        if not self.frames or idx >= len(self.frames):
            return
        if self.seg_model is None:
            self.error.emit("No segmentation model loaded.")
            self._playing = False
            return

        try:
            t0 = time.perf_counter()
            frame_bgr = self.frames[idx]
            img_rgb_original, img_rgb_enhanced, img_batch, img_gray = preprocess_frame(frame_bgr)

            mask_binary = segment_frame(self.seg_model, img_batch, self.threshold)

            if self.loc_model is not None:
                try:
                    self._loc_class_map, self._loc_confidence_map = run_localization_frame(
                        self.loc_model, img_rgb_enhanced, mask_binary
                    )
                except Exception:
                    self._loc_class_map, self._loc_confidence_map = None, None

            branches, lesions, dt, bw = run_qca_frame(
                img_gray, mask_binary, self.qca_cfg,
                class_map=self._loc_class_map, confidence_map=self._loc_confidence_map,
                use_merged_labels=(self.loc_model.use_merged_labels if self.loc_model is not None else False),
            )

            overlay = img_rgb_original.copy()
            color = np.array(self.overlay_color, dtype=np.uint8)
            overlay[mask_binary.astype(bool)] = color
            mask_overlay_rgb = cv2.addWeighted(
                overlay, self.overlay_alpha, img_rgb_original, 1.0 - self.overlay_alpha, 0
            )

            if branches:
                qca_vis_bgr = draw_overlay(img_gray, bw, branches, lesions)
                qca_vis_rgb = cv2.cvtColor(qca_vis_bgr, cv2.COLOR_BGR2RGB)
                if lesions:
                    top = lesions[0]
                    loc_text = ""
                    if "localization" in top:
                        loc = top["localization"]
                        loc_text = f"  Location={loc['group']} ({loc['label']})  LocConf={loc['confidence']:.2f}"
                    info = (
                        f"Top Lesion: DS={top['DS_percent']:.1f}%  Severity={top['severity']}  "
                        f"MLD={top['MLD_px']:.1f}px  RVD={top['RVD_px']:.1f}px{loc_text}  "
                        f"| Total Lesions: {len(lesions)} across {len(branches)} branches"
                    )
                else:
                    info = f"No stenosis detected | {len(branches)} branches analyzed"
            else:
                qca_vis_rgb = mask_overlay_rgb.copy()
                info = "QCA: Insufficient mask data for analysis"

            latency_ms = (time.perf_counter() - t0) * 1000.0
            self.frame_ready.emit(
                img_rgb_original, mask_overlay_rgb, qca_vis_rgb, idx, len(self.frames), latency_ms, info
            )
        except Exception as e:
            self.error.emit(f"Frame {idx} analysis error: {e}")
