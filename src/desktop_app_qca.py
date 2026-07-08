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
import json
import subprocess
from datetime import datetime

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
    QTextEdit
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, pyqtSlot, QTimer
from PyQt5.QtGui import QImage, QPixmap, QFont, QColor, QPalette, QIcon

import cv2
import numpy as np

try:
    import pydicom
    HAS_PYDICOM = True
except ImportError:
    HAS_PYDICOM = False

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from qca import (
    QCAConfig, to_binary_mask, morph_cleanup, qca_from_mask, draw_overlay
)
from localization import anatomy_logits_to_map_and_confidence, localize_lesions

DEFAULT_SEGMENTATION_MODEL_PATHS = [
    os.path.join(PROJECT_ROOT, "checkpoints", "mobileunetv3", "mobileunetv3_augmented_best.onnx"),
    os.path.join(PROJECT_ROOT, "checkpoints", "mobileunetv3_augmented.onnx"),
    os.path.join(PROJECT_ROOT, "checkpoints", "mobileunetv3_best.onnx"),
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
DEFAULT_3D_MODEL_PATH = first_existing_path([
    os.path.join(PROJECT_ROOT, "checkpoints", "mobileunetv3", "mobileunetv3_augmented_best.onnx"),
    os.path.join(PROJECT_ROOT, "checkpoints", "mobileunetv3_augmented.onnx"),
    os.path.join(PROJECT_ROOT, "checkpoints", "mobileunetv3_best.onnx"),
    os.path.join(PROJECT_ROOT, "checkpoints", "model.onnx"),
])
DEFAULT_CASE_ROOT = os.path.join(PROJECT_ROOT, "dicom_files")
DEFAULT_RECON_ROOT = os.path.join(PROJECT_ROOT, "reconstructions")


def create_ort_session(model_path):
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(model_path, sess_options=options)


def normalize_dicom_frame(frame):
    if frame.dtype == np.uint8:
        out = frame
    else:
        f = frame.astype(np.float32)
        lo, hi = np.percentile(f, [1, 99])
        out = np.clip((f - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
    return cv2.resize(out, (512, 512), interpolation=cv2.INTER_AREA)


def scan_case_clips(case_dir):
    clips = []
    if not HAS_PYDICOM or not case_dir or not os.path.isdir(case_dir):
        return clips
    for root, _, files in os.walk(case_dir):
        for name in files:
            path = os.path.join(root, name)
            try:
                ds = pydicom.dcmread(path, stop_before_pixels=True, force=True)
            except Exception:
                continue
            if getattr(ds, "Modality", "") != "XA":
                continue
            if not hasattr(ds, "NumberOfFrames"):
                continue
            try:
                clips.append({
                    "path": path,
                    "frames": int(getattr(ds, "NumberOfFrames", 1)),
                    "primary": float(getattr(ds, "PositionerPrimaryAngle", 0.0)),
                    "secondary": float(getattr(ds, "PositionerSecondaryAngle", 0.0)),
                    "acquisition_time": str(getattr(ds, "AcquisitionTime", "")),
                })
            except Exception:
                continue
    clips = sorted(clips, key=lambda item: item["acquisition_time"])
    for index, clip in enumerate(clips):
        clip["index"] = index
        clip["label"] = (
            f"clip_{index:02d} | {clip['frames']} frames | "
            f"PA {clip['primary']:.1f} | SA {clip['secondary']:.1f}"
        )
    return clips


def load_pixmap_file(path, label):
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        label.setText("Image unavailable")
        return
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    label.setPixmap(ndarray_to_qpixmap(rgb, label.width(), label.height()))


class ReconstructionThread(QThread):
    progress = pyqtSignal(str)
    completed = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, case_dir, model_path, threshold, output_dir):
        super().__init__()
        self.case_dir = case_dir
        self.model_path = model_path
        self.threshold = threshold
        self.output_dir = output_dir

    def run(self):
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            command = [
                sys.executable,
                os.path.join(PROJECT_ROOT, "scripts", "run_full_3d_reconstruction.py"),
                "--dicom-dir",
                self.case_dir,
                "--output-dir",
                self.output_dir,
                "--model",
                self.model_path,
                "--threshold",
                f"{self.threshold:.2f}",
            ]
            self.progress.emit("Selecting views, segmenting frames, and reconstructing 3D tree...")
            result = subprocess.run(command, cwd=PROJECT_ROOT, text=True, capture_output=True)
            if result.returncode != 0:
                message = result.stderr.strip() or result.stdout.strip() or "3D reconstruction failed."
                self.error.emit(message[-1200:])
                return
            self.completed.emit(self.output_dir)
        except Exception as exc:
            self.error.emit(str(exc))


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

        # Segmentation model handles
        self._ort_session = None
        self._ort_input_name = None
        self._torch_model = None
        self._is_onnx = False

        # Localization model handles
        self._loc_ort_session = None
        self._loc_ort_input_name = None
        self._loc_output_names = None
        self._loc_torch_model = None
        self._loc_is_onnx = False
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
        """Load model (ONNX or PyTorch) before starting the thread."""
        self.model_path = model_path
        self._is_onnx = model_path.lower().endswith('.onnx')
        self._ort_input_name = None

        if self._is_onnx:
            if not HAS_ONNX:
                self.error.emit("onnxruntime not installed. Run: pip install onnxruntime")
                return False
            try:
                self._ort_session = create_ort_session(model_path)
                self._ort_input_name = self._ort_session.get_inputs()[0].name
                return True
            except Exception as e:
                self.error.emit(f"ONNX load error: {e}")
                return False
        else:
            try:
                import torch

                sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
                from model_lightweight import MobileUNetv3, MobileUNetv2, MobileUNet, DSCUNet, DeepLabV3Plus

                name = os.path.basename(model_path).lower()
                if 'mobileunetv3' in name:
                    model = MobileUNetv3(n_classes=1, pretrained=False)
                elif 'mobileunetv2' in name:
                    model = MobileUNetv2(n_classes=1, pretrained=False)
                elif 'mobileunet' in name:
                    model = MobileUNet(n_classes=1, pretrained=False)
                elif 'deeplab' in name:
                    model = DeepLabV3Plus(n_classes=1, pretrained=False)
                elif 'dscunet' in name:
                    model = DSCUNet(n_channels=3, n_classes=1)
                else:
                    model = MobileUNetv3(n_classes=1, pretrained=False)

                device = torch.device('cpu')
                model.load_state_dict(torch.load(model_path, map_location=device))
                model.eval()
                self._torch_model = model
                return True
            except ImportError:
                self.error.emit("PyTorch not installed. Use .onnx models instead.")
                return False
            except Exception as e:
                self.error.emit(f"PyTorch load error: {e}")
                return False

    def load_localization_model(self, model_path):
        """Load optional multitask localization model (ONNX or PyTorch)."""
        self.localization_model_path = model_path
        self._loc_ort_session = None
        self._loc_ort_input_name = None
        self._loc_output_names = None
        self._loc_torch_model = None
        self._loc_class_map = None
        self._loc_confidence_map = None
        self._frame_index = 0

        if not model_path:
            return True
        if not os.path.exists(model_path):
            self.error.emit(f"Localization model not found: {model_path}")
            return False

        self._loc_is_onnx = model_path.lower().endswith('.onnx')
        if self._loc_is_onnx:
            if not HAS_ONNX:
                self.error.emit("onnxruntime not installed. Run: pip install onnxruntime")
                return False
            try:
                self._loc_ort_session = create_ort_session(model_path)
                self._loc_ort_input_name = self._loc_ort_session.get_inputs()[0].name
                self._loc_output_names = [o.name.lower() for o in self._loc_ort_session.get_outputs()]
                return True
            except Exception as e:
                self.error.emit(f"Localization ONNX load error: {e}")
                return False

        try:
            import torch
            from model_multitask import MultiTaskMobileUNetv3

            device = torch.device('cpu')
            model = MultiTaskMobileUNetv3(pretrained=False)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()
            self._loc_torch_model = model
            return True
        except ImportError:
            self.error.emit("PyTorch not installed. Use .onnx localization models instead.")
            return False
        except Exception as e:
            self.error.emit(f"Localization PyTorch load error: {e}")
            return False

    def _run_localization(self, img_rgb_enhanced):
        """Refresh cached anatomical class and confidence maps."""
        try:
            if self._loc_ort_session is not None:
                loc_input = self._preprocess_localization_numpy(img_rgb_enhanced)
                outputs = self._loc_ort_session.run(None, {self._loc_ort_input_name: loc_input})
                if self._loc_output_names and "anatomy" in self._loc_output_names:
                    anatomy = outputs[self._loc_output_names.index("anatomy")]
                else:
                    anatomy = outputs[1] if len(outputs) > 1 else outputs[0]
            elif self._loc_torch_model is not None:
                import torch as _torch
                loc_input = self._preprocess_localization_numpy(img_rgb_enhanced)
                with _torch.no_grad():
                    tensor_in = _torch.from_numpy(loc_input).float()
                    outputs = self._loc_torch_model(tensor_in)
                    anatomy = outputs["anatomy"].cpu().numpy()
            else:
                return

            self._loc_class_map, self._loc_confidence_map = anatomy_logits_to_map_and_confidence(anatomy)
        except Exception as e:
            self.error.emit(f"Localization inference error: {e}")
            self._loc_class_map = None
            self._loc_confidence_map = None

    @staticmethod
    def _preprocess_localization_numpy(img_rgb_enhanced):
        """Localization model was trained with ImageNet normalization."""
        img_float = img_rgb_enhanced.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_float = (img_float - mean) / std
        img_chw = np.transpose(img_float, (2, 0, 1))
        return np.expand_dims(img_chw, axis=0).astype(np.float32)

    def _run_qca(self, original_gray, mask_binary_uint8):
        """
        Run QCA analysis on a single frame.
        Returns (qca_overlay_bgr, stenosis_info_str) or (None, "") on failure.
        """
        try:
            bw = to_binary_mask(mask_binary_uint8)
            bw = morph_cleanup(bw, self._qca_cfg)

            # Check if mask has enough content for QCA
            if np.sum(bw > 0) < self._qca_cfg.min_component_pixels:
                return None, ""

            branches, lesions, dt = qca_from_mask(bw, self._qca_cfg)

            if not branches:
                return None, ""

            if self._loc_class_map is not None and self._loc_confidence_map is not None and lesions:
                lesions = localize_lesions(lesions, self._loc_class_map, self._loc_confidence_map, radius=9)

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

        cap = None
        dicom_frames = None
        dicom_index = 0
        if self.video_path.lower().endswith(".dcm"):
            if not HAS_PYDICOM:
                self.error.emit("pydicom not installed. Cannot open DICOM cine.")
                self.finished.emit()
                return
            try:
                ds = pydicom.dcmread(self.video_path, force=True)
                arr = ds.pixel_array
                if arr.ndim == 2:
                    arr = arr[None, :, :]
                dicom_frames = arr
            except Exception as e:
                self.error.emit(f"Failed to open DICOM cine: {e}")
                self.finished.emit()
                return
        else:
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                self.error.emit("Failed to open video.")
                self.finished.emit()
                return

        self._running = True

        while self._running and (dicom_frames is not None or cap.isOpened()):
            if self._paused:
                self.msleep(50)
                continue

            if dicom_frames is not None:
                frame_gray = normalize_dicom_frame(dicom_frames[dicom_index])
                dicom_index = (dicom_index + 1) % len(dicom_frames)
                frame = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)
            else:
                ret, frame = cap.read()
                if not ret:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue

            # ── Preprocessing ─────────────────────────────────────────
            img_resized = cv2.resize(frame, (512, 512))
            img_rgb_original = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

            # CLAHE enhancement (matches training preprocessing)
            lab = cv2.cvtColor(img_rgb_original, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            limg = cv2.merge((cl, a, b))
            img_rgb_enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)

            img_float = img_rgb_enhanced.astype(np.float32) / 255.0
            img_chw = np.transpose(img_float, (2, 0, 1))
            img_batch = np.expand_dims(img_chw, axis=0)

            # ── Inference ─────────────────────────────────────────────
            t_start = time.perf_counter()

            if self._is_onnx and self._ort_session:
                pred = self._ort_session.run(None, {self._ort_input_name: img_batch})[0]
                pred = 1.0 / (1.0 + np.exp(-pred))  # sigmoid
                mask = pred.squeeze()
            elif self._torch_model is not None:
                import torch as _torch
                with _torch.no_grad():
                    tensor_in = _torch.from_numpy(img_batch).float()
                    pred = self._torch_model(tensor_in)
                    # Handle models returning dict (e.g. MobileUNetv3 returns {'out': ..., 'features': ...})
                    if isinstance(pred, dict):
                        pred = pred['out']
                    pred = _torch.sigmoid(pred)
                    mask = pred.squeeze().cpu().numpy()
            else:
                mask = np.zeros((512, 512), dtype=np.float32)

            if (
                self._loc_ort_session is not None or self._loc_torch_model is not None
            ) and (
                self._loc_class_map is None or self._frame_index % self._loc_frame_interval == 0
            ):
                self._run_localization(img_rgb_enhanced)

            t_end = time.perf_counter()
            latency_ms = (t_end - t_start) * 1000.0
            fps = 1000.0 / max(latency_ms, 0.001)
            self._frame_index += 1

            # ── Panel 2: Mask Overlay ─────────────────────────────────
            mask_binary = (mask > self.threshold).astype(np.uint8)

            overlay = img_rgb_original.copy()
            color = np.array(self.overlay_color, dtype=np.uint8)
            overlay[mask_binary == 1] = color
            mask_blended = cv2.addWeighted(
                overlay, self.overlay_alpha,
                img_rgb_original, 1.0 - self.overlay_alpha, 0
            )

            # ── Panel 3: QCA Analysis ─────────────────────────────────
            # QCA functions expect grayscale input and binary mask
            img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
            mask_for_qca = (mask_binary * 255).astype(np.uint8)

            qca_vis_bgr, stenosis_info = self._run_qca(img_gray, mask_for_qca)

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

        if cap is not None:
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
        self.reconstruction_thread = None
        self.case_dir = ""
        self.case_clips = []
        self.last_reconstruction_dir = ""
        self.last_reconstruction_obj = ""

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

        self.btn_case = QPushButton("Select Case")
        self.btn_case.clicked.connect(self._browse_case)
        ctrl_layout.addWidget(self.btn_case)

        self.txt_case = QLineEdit()
        self.txt_case.setPlaceholderText("Select DICOM case folder...")
        self.txt_case.setReadOnly(True)
        self.txt_case.setMinimumWidth(180)
        ctrl_layout.addWidget(self.txt_case)

        self.combo_case_clip = QComboBox()
        self.combo_case_clip.setMinimumWidth(260)
        self.combo_case_clip.addItem("No case selected")
        self.combo_case_clip.currentIndexChanged.connect(self._on_case_clip_changed)
        self.combo_case_clip.activated.connect(self._on_case_clip_changed)
        ctrl_layout.addWidget(self.combo_case_clip)

        # Video file
        self.btn_video = QPushButton("Open Selected Clip")
        self.btn_video.clicked.connect(self._browse_video)
        ctrl_layout.addWidget(self.btn_video)

        self.txt_video = QLineEdit()
        self.txt_video.setPlaceholderText("Selected case clip...")
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

        main_layout.insertWidget(1, ctrl_box)

        recon_box = QGroupBox("3D Reconstruction And Validation")
        recon_layout = QGridLayout(recon_box)

        self.btn_generate_3d = QPushButton("Generate 3D Reconstruction")
        self.btn_generate_3d.clicked.connect(self._generate_3d_reconstruction)
        recon_layout.addWidget(self.btn_generate_3d, 0, 0)

        self.btn_open_3d = QPushButton("Open 3D Viewer")
        self.btn_open_3d.clicked.connect(self._open_3d_viewer)
        self.btn_open_3d.setEnabled(False)
        recon_layout.addWidget(self.btn_open_3d, 0, 1)

        self.lbl_recon_status = QLabel("Select a case, then generate reconstruction.")
        self.lbl_recon_status.setWordWrap(True)
        self.lbl_recon_status.setStyleSheet("color: #8B949E; padding: 4px;")
        recon_layout.addWidget(self.lbl_recon_status, 0, 2, 1, 2)

        self.label_val_a = QLabel("View A validation")
        self.label_val_a.setAlignment(Qt.AlignCenter)
        self.label_val_a.setMinimumSize(280, 170)
        self.label_val_a.setStyleSheet("background-color: #161B22; border-radius: 8px; color: #484F58;")
        recon_layout.addWidget(self.label_val_a, 1, 0, 1, 2)

        self.label_val_b = QLabel("View B validation")
        self.label_val_b.setAlignment(Qt.AlignCenter)
        self.label_val_b.setMinimumSize(280, 170)
        self.label_val_b.setStyleSheet("background-color: #161B22; border-radius: 8px; color: #484F58;")
        recon_layout.addWidget(self.label_val_b, 1, 2, 1, 2)

        self.txt_recon_summary = QTextEdit()
        self.txt_recon_summary.setReadOnly(True)
        self.txt_recon_summary.setMaximumHeight(110)
        self.txt_recon_summary.setPlaceholderText("Reconstruction summary and validation metrics will appear here.")
        recon_layout.addWidget(self.txt_recon_summary, 2, 0, 1, 4)

        main_layout.insertWidget(2, recon_box)

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
    def _browse_case(self):
        start_dir = DEFAULT_CASE_ROOT if os.path.isdir(DEFAULT_CASE_ROOT) else PROJECT_ROOT
        path = QFileDialog.getExistingDirectory(self, "Select DICOM Case Folder", start_dir)
        if path:
            self._load_case(path)

    def _load_case(self, case_dir):
        self.case_dir = case_dir
        self.txt_case.setText(case_dir)
        self.case_clips = scan_case_clips(case_dir)
        self.combo_case_clip.blockSignals(True)
        self.combo_case_clip.clear()
        if not self.case_clips:
            self.combo_case_clip.addItem("No XA DICOM clips found")
            self.txt_video.clear()
            self.statusBar().showMessage("No XA DICOM clips found in selected case.")
        else:
            for clip in self.case_clips:
                self.combo_case_clip.addItem(clip["label"], clip["path"])
            self.combo_case_clip.setCurrentIndex(0)
            self.txt_video.setText(self.case_clips[0]["path"])
            self.statusBar().showMessage(
                f"Case loaded: {os.path.basename(case_dir)} ({len(self.case_clips)} XA clips)"
            )
        self.combo_case_clip.blockSignals(False)

    def _on_case_clip_changed(self, index):
        if 0 <= index < len(self.case_clips):
            self._set_selected_case_clip(index, load_for_playback=False)

    def _set_selected_case_clip(self, index, load_for_playback=True):
        if not (0 <= index < len(self.case_clips)):
            return
        if self.video_thread.isRunning():
            self.video_thread.stop()
            self.video_thread.wait()
            self.btn_play.setEnabled(True)
            self.btn_pause.setEnabled(False)
            self.btn_stop.setEnabled(False)
        clip = self.case_clips[index]
        self.txt_video.setText(clip["path"])
        action = "Loaded" if load_for_playback else "Selected"
        self.statusBar().showMessage(f"{action} {clip['label']}")

    def _browse_video(self):
        if self.case_clips:
            index = self.combo_case_clip.currentIndex()
            if 0 <= index < len(self.case_clips):
                self._set_selected_case_clip(index, load_for_playback=True)
            return
        self.statusBar().showMessage("Please select a DICOM case first.")

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
    def _generate_3d_reconstruction(self):
        if not self.case_dir or not os.path.isdir(self.case_dir):
            self.statusBar().showMessage("Please select a DICOM case first.")
            return
        model_path = self.txt_model.text().strip()
        if not model_path or not model_path.lower().endswith(".onnx") or not os.path.exists(model_path):
            model_path = DEFAULT_3D_MODEL_PATH
        if not model_path or not model_path.lower().endswith(".onnx") or not os.path.exists(model_path):
            self.statusBar().showMessage("3D reconstruction requires an ONNX segmentation model.")
            self.lbl_recon_status.setText("3D reconstruction requires an ONNX segmentation model.")
            self.txt_recon_summary.setPlainText(
                "The live viewer can use .pth, but the automatic 3D reconstruction pipeline uses ONNX Runtime.\n"
                "Place a valid .onnx model under checkpoints or load one in the Model field."
            )
            return
        if self.reconstruction_thread is not None and self.reconstruction_thread.isRunning():
            self.statusBar().showMessage("3D reconstruction is already running.")
            return

        case_name = os.path.basename(os.path.normpath(self.case_dir))
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(DEFAULT_RECON_ROOT, case_name, f"run_{stamp}")
        threshold = self.slider_threshold.value() / 100.0

        self.btn_generate_3d.setEnabled(False)
        self.btn_open_3d.setEnabled(False)
        self.lbl_recon_status.setText("Running automatic 3D reconstruction...")
        self.txt_recon_summary.setPlainText(
            "Working...\n"
            "Selecting views, segmenting frames, reconstructing, and validating.\n"
            f"3D model: {model_path}"
        )
        self.label_val_a.setText("View A validation pending")
        self.label_val_b.setText("View B validation pending")

        self.reconstruction_thread = ReconstructionThread(self.case_dir, model_path, threshold, output_dir)
        self.reconstruction_thread.progress.connect(self._on_reconstruction_progress)
        self.reconstruction_thread.completed.connect(self._on_reconstruction_completed)
        self.reconstruction_thread.error.connect(self._on_reconstruction_error)
        self.reconstruction_thread.start()

    @pyqtSlot(str)
    def _on_reconstruction_progress(self, message):
        self.statusBar().showMessage(message)
        self.lbl_recon_status.setText(message)

    @pyqtSlot(str)
    def _on_reconstruction_completed(self, output_dir):
        self.btn_generate_3d.setEnabled(True)
        self.last_reconstruction_dir = output_dir
        self.last_reconstruction_obj = os.path.join(output_dir, "hybrid_smoothed_confidence_colored.obj")
        if not os.path.exists(self.last_reconstruction_obj):
            self.last_reconstruction_obj = os.path.join(output_dir, "04_smoothed_confidence", "hybrid_smoothed_confidence_colored.obj")
        self.btn_open_3d.setEnabled(os.path.exists(self.last_reconstruction_obj))
        self._load_reconstruction_result(output_dir)
        self.statusBar().showMessage("3D reconstruction complete.")

    @pyqtSlot(str)
    def _on_reconstruction_error(self, message):
        self.btn_generate_3d.setEnabled(True)
        self.btn_open_3d.setEnabled(False)
        self.lbl_recon_status.setText("3D reconstruction failed.")
        self.txt_recon_summary.setPlainText(message)
        self.statusBar().showMessage("3D reconstruction failed.")

    def _load_reconstruction_result(self, output_dir):
        summary_path = os.path.join(output_dir, "full_3d_reconstruction_summary.json")
        validation_path = os.path.join(output_dir, "final_reprojection_validation_summary.json")
        lines = [f"Output: {output_dir}"]

        if os.path.exists(summary_path):
            with open(summary_path, "r", encoding="utf-8") as f:
                summary = json.load(f)
            pair = summary.get("selected_pair", {})
            lines.append(
                "Selected views: "
                f"A clip {pair.get('view_a_clip_index', '?')} frame {pair.get('view_a_frame', '?')} | "
                f"B clip {pair.get('view_b_clip_index', '?')} frame {pair.get('view_b_frame', '?')}"
            )

        if os.path.exists(validation_path):
            with open(validation_path, "r", encoding="utf-8") as f:
                validation = json.load(f)
            for view_key in ("view_a", "view_b"):
                view = validation.get("views", {}).get(view_key, {})
                if view:
                    lines.append(
                        f"{view_key}: median {float(view.get('median_branch_error_px', 0)):.2f}px, "
                        f"p90 {float(view.get('p90_branch_error_px', 0)):.2f}px, "
                        f"good/review/bad {view.get('good_branches', 0)}/"
                        f"{view.get('review_branches', 0)}/{view.get('bad_branches', 0)}"
                    )

        branch_report = os.path.join(output_dir, "branch_quality_report.csv")
        if os.path.exists(branch_report):
            lines.append(f"Branch report: {branch_report}")

        self.lbl_recon_status.setText("3D reconstruction complete. Review validation before clinical use.")
        self.txt_recon_summary.setPlainText("\n".join(lines))

        view_a_img = os.path.join(output_dir, "final_view_a_reprojection_validation.png")
        view_b_img = os.path.join(output_dir, "final_view_b_reprojection_validation.png")
        if os.path.exists(view_a_img):
            load_pixmap_file(view_a_img, self.label_val_a)
        if os.path.exists(view_b_img):
            load_pixmap_file(view_b_img, self.label_val_b)

    def _open_3d_viewer(self):
        if not self.last_reconstruction_obj or not os.path.exists(self.last_reconstruction_obj):
            self.statusBar().showMessage("No reconstructed OBJ is available yet.")
            return
        viewer = os.path.join(PROJECT_ROOT, "src", "artery_3d_vtk_viewer.py")
        subprocess.Popen([sys.executable, viewer, "--obj", self.last_reconstruction_obj], cwd=PROJECT_ROOT)
        self.statusBar().showMessage("3D viewer opened.")

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

    # ── Cleanup ──────────────────────────────────────────────────────
    def closeEvent(self, event):
        self.video_thread.stop()
        self.video_thread.wait()
        if self.reconstruction_thread is not None and self.reconstruction_thread.isRunning():
            self.reconstruction_thread.wait(1000)
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