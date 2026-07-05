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
    QGroupBox, QStatusBar, QSizePolicy, QFrame, QLineEdit, QGridLayout
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, pyqtSlot, QTimer
from PyQt5.QtGui import QImage, QPixmap, QFont, QColor, QPalette, QIcon

import cv2
import numpy as np

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
    os.path.join(PROJECT_ROOT, "checkpoints", "mobileunetv3", "mobileunetv3_augmented_best.pth"),
]
DEFAULT_LOCALIZATION_MODEL_PATHS = [
    # Preferred: two-stage mask-input model (15 merged classes, more accurate)
    os.path.join(PROJECT_ROOT, "checkpoints", "mask_localization_v2", "best.pth"),
    # Fallback: older single-network image-input model (26 raw classes)
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


def create_ort_session(model_path):
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(model_path, sess_options=options)


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

        # "mask" = new two-stage MaskLocalizationNet (mask input, 15 merged classes)
        # "multitask" = old MultiTaskMobileUNetv3 (image input, 26 raw classes)
        self._loc_model_type = None
        self._loc_label_fns = None  # (label_fn, group_fn, artery_fn) matching _loc_model_type

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

                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                model.load_state_dict(torch.load(model_path, map_location=device))
                model.to(device)
                model.eval()

                if device.type == 'cuda':
                    torch.backends.cudnn.benchmark = True
                    try:
                        # NOTE: mode="reduce-overhead" uses CUDA graphs, which
                        # bind to thread-local state on whichever thread first
                        # runs the model. Load happens on the main thread but
                        # inference runs inside VideoThread's own OS thread,
                        # so reduce-overhead crashes here (AssertionError in
                        # cudagraph_trees). Default mode compiles/fuses too
                        # but has no such thread affinity.
                        model = torch.compile(model)
                        # Warm up now (pays the one-time compile cost here, not on frame 1)
                        with torch.no_grad():
                            model(torch.zeros(1, 3, 512, 512, device=device))
                    except Exception:
                        pass  # fall back to eager mode if compile isn't supported here

                self._torch_model = model
                self._seg_device = device
                return True
            except ImportError:
                self.error.emit("PyTorch not installed. Use .onnx models instead.")
                return False
            except Exception as e:
                self.error.emit(f"PyTorch load error: {e}")
                return False

    def load_localization_model(self, model_path):
        """Load optional localization model (ONNX or PyTorch).

        Supports two architectures, auto-detected from the filename:
          - "mask_localization*"  -> new MaskLocalizationNet (mask input, 15 merged classes)
          - anything else         -> old MultiTaskMobileUNetv3 (image input, 26 raw classes)
        """
        self.localization_model_path = model_path
        self._loc_ort_session = None
        self._loc_ort_input_name = None
        self._loc_output_names = None
        self._loc_torch_model = None
        self._loc_class_map = None
        self._loc_confidence_map = None
        self._loc_model_type = None
        self._loc_label_fns = None
        self._frame_index = 0

        if not model_path:
            return True
        if not os.path.exists(model_path):
            self.error.emit(f"Localization model not found: {model_path}")
            return False

        name = os.path.basename(model_path).lower()
        is_mask_model = 'mask_localization' in model_path.lower() or name.startswith('mask')
        self._loc_model_type = 'mask' if is_mask_model else 'multitask'

        from localization_labels import (
            segment_label, segment_group, segment_artery,
            merged_segment_label, merged_segment_group, merged_segment_artery,
        )
        self._loc_label_fns = (
            (merged_segment_label, merged_segment_group, merged_segment_artery)
            if is_mask_model else
            (segment_label, segment_group, segment_artery)
        )

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

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            if is_mask_model:
                from model_mask_localization import MaskLocalizationNet
                from localization_labels import MERGED_NUM_ANATOMY_CLASSES

                model = MaskLocalizationNet(n_anatomy_classes=MERGED_NUM_ANATOMY_CLASSES, pretrained=False)
                model.load_state_dict(torch.load(model_path, map_location=device))
                warmup_input = torch.zeros(1, 1, 512, 512, device=device)  # 1-channel mask
            else:
                from model_multitask import MultiTaskMobileUNetv3

                model = MultiTaskMobileUNetv3(pretrained=False)
                # strict=False: older checkpoints may still contain a now-removed stenosis_head
                model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
                warmup_input = torch.zeros(1, 3, 512, 512, device=device)  # 3-channel image

            model.to(device)
            model.eval()

            if device.type == 'cuda':
                torch.backends.cudnn.benchmark = True
                try:
                    # See note above: default mode, not reduce-overhead (CUDA
                    # graphs from reduce-overhead crash when compile/warmup
                    # happens on a different thread than actual inference).
                    model = torch.compile(model)
                    with torch.no_grad():
                        model(warmup_input)
                except Exception:
                    pass

            self._loc_torch_model = model
            self._loc_device = device
            return True
        except ImportError:
            self.error.emit("PyTorch not installed. Use .onnx localization models instead.")
            return False
        except Exception as e:
            self.error.emit(f"Localization PyTorch load error: {e}")
            return False

    def _run_localization(self, img_rgb_enhanced, vessel_mask_binary):
        """Refresh cached anatomical class and confidence maps.

        vessel_mask_binary: (H, W) uint8/bool array, the segmentation model's
        already-thresholded prediction for this frame. Only used by the new
        "mask" model type, which takes the mask as input instead of the image.
        """
        try:
            is_mask_model = self._loc_model_type == 'mask'
            if is_mask_model:
                loc_input = vessel_mask_binary.astype(np.float32)[None, None, :, :]  # (1,1,H,W)
            else:
                loc_input = self._preprocess_localization_numpy(img_rgb_enhanced)

            if self._loc_ort_session is not None:
                outputs = self._loc_ort_session.run(None, {self._loc_ort_input_name: loc_input})
                if self._loc_output_names and "anatomy" in self._loc_output_names:
                    anatomy = outputs[self._loc_output_names.index("anatomy")]
                else:
                    anatomy = outputs[1] if len(outputs) > 1 else outputs[0]
            elif self._loc_torch_model is not None:
                import torch as _torch
                with _torch.no_grad():
                    tensor_in = _torch.from_numpy(loc_input).float().to(self._loc_device)
                    outputs = self._loc_torch_model(tensor_in)
                    anatomy = outputs["anatomy"].cpu().numpy()
            else:
                return

            class_map, confidence_map = anatomy_logits_to_map_and_confidence(anatomy)

            if is_mask_model:
                # This model is only ever supervised on ground-truth vessel
                # pixels (background is excluded from its training loss), so
                # its raw prediction is meaningless outside the vessel mask.
                # Constrain it to where the segmentation model actually found
                # vessel, same fix as predict_pipeline.py.
                class_map = class_map.copy()
                class_map[vessel_mask_binary == 0] = 0

            self._loc_class_map, self._loc_confidence_map = class_map, confidence_map
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
                label_fn, group_fn, artery_fn = self._loc_label_fns
                lesions = localize_lesions(
                    lesions, self._loc_class_map, self._loc_confidence_map, radius=9,
                    label_fn=label_fn, group_fn=group_fn, artery_fn=artery_fn,
                )

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

        # Pace playback to the video's actual recorded FPS. Inference is now
        # fast enough (GPU + compile) that without this, frames get read and
        # displayed as fast as the hardware allows rather than at the video's
        # real speed. Some files report bogus FPS metadata (0, or absurdly
        # high values from certain codecs/containers), so clamp to a sane
        # range and fall back to 30 otherwise.
        raw_reported_fps = cap.get(cv2.CAP_PROP_FPS)
        if raw_reported_fps and 1.0 <= raw_reported_fps <= 120.0:
            source_fps = raw_reported_fps
        else:
            source_fps = 30.0
        target_frame_time_ms = 1000.0 / source_fps

        self._running = True

        while self._running and cap.isOpened():
            if self._paused:
                self.msleep(50)
                continue

            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            frame_wall_start = time.perf_counter()

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
                    tensor_in = _torch.from_numpy(img_batch).float().to(self._seg_device)
                    pred = self._torch_model(tensor_in)
                    # Handle models returning dict (e.g. MobileUNetv3 returns {'out': ..., 'features': ...})
                    if isinstance(pred, dict):
                        pred = pred['out']
                    pred = _torch.sigmoid(pred)
                    mask = pred.squeeze().cpu().numpy()
            else:
                mask = np.zeros((512, 512), dtype=np.float32)

            # Needed for the mask overlay below, and as the localization
            # model's input if it's the new mask-input architecture.
            mask_binary = (mask > self.threshold).astype(np.uint8)

            if (
                self._loc_ort_session is not None or self._loc_torch_model is not None
            ) and (
                self._loc_class_map is None or self._frame_index % self._loc_frame_interval == 0
            ):
                self._run_localization(img_rgb_enhanced, mask_binary)

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

            # Sleep off whatever's left of this frame's real-time budget, so
            # playback matches the video's actual recorded speed instead of
            # running as fast as the hardware allows.
            elapsed_ms = (time.perf_counter() - frame_wall_start) * 1000.0
            remaining_ms = target_frame_time_ms - elapsed_ms
            self.msleep(max(1, int(remaining_ms)))

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
        # Fixed height (not just a size hint) so wrapping to 2-3 lines never
        # grows this box and steals space from the video panels above it,
        # which is what was causing the layout to visibly shift/jump.
        stenosis_box = QGroupBox("Stenosis Report")
        stenosis_box.setFixedHeight(92)
        stenosis_layout = QHBoxLayout(stenosis_box)
        self.lbl_stenosis = QLabel("No analysis yet - load a video and press Play")
        self.lbl_stenosis.setFont(QFont("Consolas", 12, QFont.Bold))
        self.lbl_stenosis.setStyleSheet("color: #FF6B6B; padding: 4px;")
        self.lbl_stenosis.setWordWrap(True)
        self.lbl_stenosis.setAlignment(Qt.AlignLeft | Qt.AlignTop)
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
            elif "MODERATE" in stenosis_info:
                self.lbl_stenosis.setStyleSheet("color: #FFA657; padding: 4px; font-weight: bold;")
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

    # ── Cleanup ──────────────────────────────────────────────────────
    def closeEvent(self, event):
        self.video_thread.stop()
        self.video_thread.wait()
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
