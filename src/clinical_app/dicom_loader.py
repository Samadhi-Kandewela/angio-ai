"""
DICOM discovery and frame decoding for local case analysis.

Converts DICOM pixel data (any bit depth, mono or color, single- or
multi-frame) into a list of 8-bit BGR frames ready for frame_pipeline's
preprocess_frame() -- the same entry point the QCA report generator and the
live preview use, so DICOM-sourced frames go through the identical
segmentation/QCA path as video-sourced frames.

No Qt dependency -- reusable from any UI layer, a CLI, or a batch tool.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import cv2
import numpy as np
import pydicom

try:
    from pydicom.pixels import apply_voi_lut
except ImportError:  # older pydicom
    from pydicom.pixel_data_handlers.util import apply_voi_lut


@dataclass
class DicomSeriesInfo:
    path: Path
    num_frames: int
    rows: int
    cols: int
    modality: str
    description: str


def discover_series(root_dir) -> List[DicomSeriesInfo]:
    """Recursively finds readable DICOM image files under root_dir (metadata only, fast)."""
    root = Path(root_dir)
    results = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        try:
            ds = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
        except Exception:
            continue
        if not hasattr(ds, "Rows") or not hasattr(ds, "Columns"):
            continue

        num_frames = int(getattr(ds, "NumberOfFrames", 1))
        description = (
            getattr(ds, "SeriesDescription", "") or getattr(ds, "StudyDescription", "") or path.name
        )
        results.append(DicomSeriesInfo(
            path=path, num_frames=num_frames, rows=int(ds.Rows), cols=int(ds.Columns),
            modality=str(getattr(ds, "Modality", "")), description=str(description),
        ))
    return results


def _normalize_to_uint8(arr: np.ndarray, ds) -> np.ndarray:
    """Applies VOI LUT / windowing then rescales to 0-255 uint8, honoring MONOCHROME1 inversion."""
    try:
        arr = apply_voi_lut(arr, ds).astype(np.float32)
    except Exception:
        arr = arr.astype(np.float32)

    lo, hi = float(arr.min()), float(arr.max())
    if hi <= lo:
        hi = lo + 1.0
    arr = np.clip((arr - lo) / (hi - lo), 0.0, 1.0) * 255.0
    arr = arr.astype(np.uint8)

    if str(getattr(ds, "PhotometricInterpretation", "")) == "MONOCHROME1":
        arr = 255 - arr
    return arr


def load_series_frames(path) -> List[np.ndarray]:
    """
    Reads a DICOM file and returns its frame(s) as a list of 8-bit BGR numpy
    arrays (even single-frame / grayscale sources), ready for
    frame_pipeline.preprocess_frame().
    """
    ds = pydicom.dcmread(str(path), force=True)
    arr = _normalize_to_uint8(ds.pixel_array, ds)

    samples_per_pixel = int(getattr(ds, "SamplesPerPixel", 1))
    is_color = samples_per_pixel == 3

    if arr.ndim == 2:
        frames = [arr]
    elif arr.ndim == 3:
        frames = [arr] if is_color else [arr[i] for i in range(arr.shape[0])]
    elif arr.ndim == 4:
        frames = [arr[i] for i in range(arr.shape[0])]
    else:
        raise ValueError(f"Unsupported DICOM pixel array shape: {arr.shape}")

    bgr_frames = []
    for f in frames:
        if f.ndim == 2:
            bgr_frames.append(cv2.cvtColor(f, cv2.COLOR_GRAY2BGR))
        else:
            bgr_frames.append(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    return bgr_frames
