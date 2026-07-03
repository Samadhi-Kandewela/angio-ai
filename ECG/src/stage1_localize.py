"""
stage1_localize.py
------------------
Stage I: Baseline detection and vertical anchor point extraction for a
single-lead ECG rhythm strip.

For a single-lead strip there is only ONE baseline to detect.
We use the Hough Transform approach from Wu et al. (2022) to find the
dominant near-horizontal line that represents the isoelectric baseline.

Outputs:
  - baseline_y : the y-coordinate (row) of the ECG baseline
  - crop_top   : upper boundary for waveform extraction
  - crop_bottom: lower boundary for waveform extraction
"""

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for servers
import matplotlib.pyplot as plt
from utils import preprocess_image, save_visualization


def detect_baseline(binary: np.ndarray, original_bgr: np.ndarray = None) -> int:
    """
    Detect the dominant horizontal baseline in a single-lead ECG binary image.

    For GRAYSCALE ECG images: use the argmin trace (darkest pixel per column)
    and take its mode — the y-coordinate the trace visits most often is the
    isoelectric baseline.

    For COLOR ECG images (red grid): Wu et al. Hough-line approach on the
    pre-processed binary.

    Returns the y-coordinate of the detected baseline.
    """
    h, w = binary.shape

    # ── Argmin path (darkest pixel) ─────────────────────────────────────────
    if original_bgr is not None:
        gray = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2GRAY)
        trace_y = np.argmin(gray, axis=0).astype(np.int32)
        # Mode = the y the trace visits most often = isoelectric baseline
        counts = np.bincount(trace_y, minlength=h)
        # Restrict to middle 60% of image to avoid border artefacts
        top_limit = int(h * 0.20)
        bot_limit = int(h * 0.80)
        counts[:top_limit] = 0
        counts[bot_limit:] = 0
        return int(np.argmax(counts))

    # ── Color / Hough path (Wu et al.) ──────────────────────────────────────
    row_sums = binary.sum(axis=1).astype(np.float32)
    kernel_size = max(3, h // 40)
    if kernel_size % 2 == 0:
        kernel_size += 1
    row_sums_smooth = cv2.GaussianBlur(
        row_sums.reshape(-1, 1), (1, kernel_size), 0
    ).flatten()

    top_limit = int(h * 0.20)
    bot_limit = int(h * 0.80)
    search_region = row_sums_smooth[top_limit:bot_limit]
    baseline_y = int(np.argmax(search_region) + top_limit)

    edges = cv2.Canny(binary, 50, 150)
    min_line_length = int(w * 0.80)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=int(w * 0.3),
        minLineLength=min_line_length,
        maxLineGap=int(w * 0.10),
    )
    if lines is not None:
        horizontal_lines = []
        for x1, y1, x2, y2 in lines[:, 0]:
            angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
            if angle <= 2.5 or angle >= 177.5:
                horizontal_lines.append((y1 + y2) // 2)
        if horizontal_lines:
            baseline_y = int(np.median(horizontal_lines))

    return baseline_y


def compute_crop_boundaries(
    binary: np.ndarray,
    baseline_y: int,
    margin_factor: float = 0.7
) -> tuple[int, int]:
    """
    Compute vertical crop boundaries around the baseline.

    Wu et al.: upper/lower boundaries = baseline +/- 0.7 x inter-lead spacing.
    For a single strip we use 0.7 x half_height as the margin.

    Returns (crop_top, crop_bottom) pixel rows.
    """
    h = binary.shape[0]
    # For a single strip, use a generous margin (40% of total height)
    margin = int(h * 0.40)
    crop_top = max(0, baseline_y - margin)
    crop_bottom = min(h, baseline_y + margin)
    return crop_top, crop_bottom


def localize(image_path: str, visualize: bool = True) -> dict:
    """
    Run Stage I localization on a single-lead ECG image.

    Args:
        image_path : path to the scanned ECG image
        visualize  : if True, save an annotated PNG showing detected baseline

    Returns a dict with keys:
        binary       : binarized image
        original_bgr : original colour image
        baseline_y   : detected baseline row
        crop_top     : upper crop boundary
        crop_bottom  : lower crop boundary
        cropped_binary: binary image cropped to the lead region
    """
    print(f"[Stage I] Loading image: {image_path}")
    original_bgr, binary = preprocess_image(image_path)

    print("[Stage I] Detecting ECG baseline...")
    baseline_y = detect_baseline(binary, original_bgr=original_bgr)

    print(f"[Stage I] Baseline detected at y={baseline_y}")
    crop_top, crop_bottom = compute_crop_boundaries(binary, baseline_y)
    print(f"[Stage I] Crop region: rows {crop_top}–{crop_bottom}")

    cropped_binary = binary[crop_top:crop_bottom, :]
    cropped_original = original_bgr[crop_top:crop_bottom, :]

    if visualize:
        vis = original_bgr.copy()
        h, w = vis.shape[:2]
        # Draw baseline line in green
        cv2.line(vis, (0, baseline_y), (w, baseline_y), (0, 200, 0), 2)
        # Draw crop boundaries in blue
        cv2.line(vis, (0, crop_top), (w, crop_top), (255, 100, 0), 1)
        cv2.line(vis, (0, crop_bottom), (w, crop_bottom), (255, 100, 0), 1)
        out_path = str(image_path).replace(".png", "_stage1.png").replace(".jpg", "_stage1.jpg")
        save_visualization(vis, out_path)
        print(f"[Stage I] Visualization saved -> {out_path}")

    return {
        "binary": binary,
        "original_bgr": original_bgr,
        "baseline_y": baseline_y,
        "crop_top": crop_top,
        "crop_bottom": crop_bottom,
        "cropped_binary": cropped_binary,
        "cropped_original": cropped_original,
    }


if __name__ == "__main__":
    import sys
    img_path = sys.argv[1] if len(sys.argv) > 1 else "../sample.png"
    result = localize(img_path, visualize=True)
    print(f"\nStage I complete. Baseline at y={result['baseline_y']}, "
          f"crop rows: {result['crop_top']}–{result['crop_bottom']}")
