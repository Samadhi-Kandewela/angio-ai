"""
Shared frame-level inference pipeline for the angio-ai desktop app.

Both the live real-time preview (VideoThread in desktop_app_qca.py) and the
offline clinical report generator (report_engine.py) need to run the exact
same preprocessing -> segmentation -> localization -> QCA path on a frame.
This module is the single source of truth for that path so report numbers
never silently drift from what was shown live on screen.
"""

import os
import sys

import cv2
import numpy as np

try:
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from qca import QCAConfig, to_binary_mask, morph_cleanup, qca_from_mask
from localization import anatomy_logits_to_map_and_confidence, localize_lesions


def create_ort_session(model_path):
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(model_path, sess_options=options)


# ─────────────────────────────────────────────────────────────────────────────
# Model handles
# ─────────────────────────────────────────────────────────────────────────────
class SegmentationModel:
    """Wraps an ONNX or PyTorch vessel segmentation model behind one predict() call."""

    def __init__(self, model_path):
        self.model_path = model_path
        self.is_onnx = model_path.lower().endswith(".onnx")
        self._ort_session = None
        self._ort_input_name = None
        self._torch_model = None

        if self.is_onnx:
            if not HAS_ONNX:
                raise RuntimeError("onnxruntime not installed. Run: pip install onnxruntime")
            self._ort_session = create_ort_session(model_path)
            self._ort_input_name = self._ort_session.get_inputs()[0].name
        else:
            import torch
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

    def predict(self, img_batch: np.ndarray) -> np.ndarray:
        """img_batch: (1,3,H,W) float32 in [0,1]. Returns (H,W) sigmoid probability map."""
        if self._ort_session is not None:
            pred = self._ort_session.run(None, {self._ort_input_name: img_batch})[0]
            pred = 1.0 / (1.0 + np.exp(-pred))
            return pred.squeeze()

        import torch as _torch
        with _torch.no_grad():
            tensor_in = _torch.from_numpy(img_batch).float()
            pred = self._torch_model(tensor_in)
            if isinstance(pred, dict):
                pred = pred['out']
            pred = _torch.sigmoid(pred)
            return pred.squeeze().cpu().numpy()


class LocalizationModel:
    """Wraps the optional multitask anatomical-localization model."""

    def __init__(self, model_path):
        self.model_path = model_path
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Localization model not found: {model_path}")

        self.is_onnx = model_path.lower().endswith(".onnx")
        self._ort_session = None
        self._ort_input_name = None
        self._output_names = None
        self._torch_model = None

        if self.is_onnx:
            if not HAS_ONNX:
                raise RuntimeError("onnxruntime not installed. Run: pip install onnxruntime")
            self._ort_session = create_ort_session(model_path)
            self._ort_input_name = self._ort_session.get_inputs()[0].name
            self._output_names = [o.name.lower() for o in self._ort_session.get_outputs()]
        else:
            import torch
            from model_multitask import MultiTaskMobileUNetv3

            device = torch.device('cpu')
            model = MultiTaskMobileUNetv3(pretrained=False)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()
            self._torch_model = model

    def predict(self, img_rgb_enhanced: np.ndarray):
        """Returns (class_map, confidence_map) for one CLAHE-enhanced RGB frame."""
        loc_input = preprocess_localization_numpy(img_rgb_enhanced)

        if self._ort_session is not None:
            outputs = self._ort_session.run(None, {self._ort_input_name: loc_input})
            if self._output_names and "anatomy" in self._output_names:
                anatomy = outputs[self._output_names.index("anatomy")]
            else:
                anatomy = outputs[1] if len(outputs) > 1 else outputs[0]
        else:
            import torch as _torch
            with _torch.no_grad():
                tensor_in = _torch.from_numpy(loc_input).float()
                outputs = self._torch_model(tensor_in)
                anatomy = outputs["anatomy"].cpu().numpy()

        return anatomy_logits_to_map_and_confidence(anatomy)


def preprocess_localization_numpy(img_rgb_enhanced):
    """Localization model was trained with ImageNet normalization."""
    img_float = img_rgb_enhanced.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_float = (img_float - mean) / std
    img_chw = np.transpose(img_float, (2, 0, 1))
    return np.expand_dims(img_chw, axis=0).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Frame pipeline
# ─────────────────────────────────────────────────────────────────────────────
def preprocess_frame(frame_bgr: np.ndarray, size=(512, 512)):
    """
    CLAHE-enhanced preprocessing matching the training pipeline.
    Returns (img_rgb_original, img_rgb_enhanced, img_batch, img_gray).
    """
    img_resized = cv2.resize(frame_bgr, size)
    img_rgb_original = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

    lab = cv2.cvtColor(img_rgb_original, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    img_rgb_enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)

    img_float = img_rgb_enhanced.astype(np.float32) / 255.0
    img_chw = np.transpose(img_float, (2, 0, 1))
    img_batch = np.expand_dims(img_chw, axis=0)

    img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    return img_rgb_original, img_rgb_enhanced, img_batch, img_gray


def segment_frame(model: SegmentationModel, img_batch: np.ndarray, threshold: float) -> np.ndarray:
    """Returns a uint8 {0,1} binary vessel mask at model resolution."""
    prob = model.predict(img_batch)
    return (prob > threshold).astype(np.uint8)


def run_localization_frame(model: LocalizationModel, img_rgb_enhanced: np.ndarray):
    """Returns (class_map, confidence_map), or (None, None) if model is None."""
    if model is None:
        return None, None
    return model.predict(img_rgb_enhanced)


def run_qca_frame(img_gray: np.ndarray, mask_binary: np.ndarray, cfg: QCAConfig,
                  class_map=None, confidence_map=None):
    """
    Runs QCA analysis on a single frame's segmentation mask.

    `mask_binary` may be {0,1} or {0,255} — it is normalized internally.
    Returns (branches, lesions, dt, bw) where bw is the cleaned {0,255} mask
    actually analyzed (needed downstream for overlay drawing / crops).
    """
    mask_u8 = (mask_binary.astype(np.uint8) * 255) if mask_binary.max() <= 1 else mask_binary.astype(np.uint8)
    bw = to_binary_mask(mask_u8)
    bw = morph_cleanup(bw, cfg)

    if np.sum(bw > 0) < cfg.min_component_pixels:
        return [], [], np.zeros_like(bw, dtype=np.float32), bw

    branches, lesions, dt = qca_from_mask(bw, cfg)

    if class_map is not None and confidence_map is not None and lesions:
        lesions = localize_lesions(lesions, class_map, confidence_map, radius=9)

    return branches, lesions, dt, bw
