"""
utils.py
--------
Shared helper functions for the single-lead ECG pipeline.
Handles image loading, red-grid removal, and binarization,
following the method described in Wu et al. (2022) Scientific Reports.
"""

import cv2
import numpy as np
from pathlib import Path


def load_image(image_path: str) -> np.ndarray:
    """Load an image from disk and return a BGR numpy array."""
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    return img


def remove_redacted_header(img_bgr: np.ndarray) -> np.ndarray:
    """
    Remove any fully-black redacted header region from the top of the image.
    Wu et al.: "average pixel intensity of each row of the redacted region is zero"
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    row_means = gray.mean(axis=1)
    # Find first row where mean intensity is above a low threshold (not black)
    first_content_row = 0
    for i, mean_val in enumerate(row_means):
        if mean_val > 10:
            first_content_row = i
            break
    return img_bgr[first_content_row:, :]


def remove_red_grid(img_bgr: np.ndarray, threshold: float = 0.94) -> np.ndarray:
    """
    Remove the ECG grid and return a binary image containing only the signal.

    Handles two image types automatically:

    COLOR ECG (red grid on white paper) — Wu et al. (2022) two-step method:
      1. Set R=1, convert to grayscale, discard pixels > threshold.
         This removes the white/near-white background (high G+B keeps gray ~1.0).
      2. Additionally exclude red-dominant pixels (grid lines): R >> G and R >> B.
         The black ECG trace has R ≈ G ≈ B (all near zero), so it is kept.

    GRAYSCALE ECG (gray or black grid on light paper) — direct threshold:
      The Wu et al. R=1 method fails for grayscale images because background
      pixels (gray ≈ 226) also fall below 0.94 after the R-clamp, flooding the
      binary with false signal. Instead we use a low absolute brightness cutoff:
      only pixels darker than ~30% brightness are kept as signal.
      ECG trace ≈ gray 14–76 (very dark ink); grid ≈ gray 81–180; paper ≈ 200+.

    Returns a binary uint8 image (0=background, 255=signal).
    """
    img_float = img_bgr.astype(np.float32) / 255.0
    r = img_float[:, :, 2]  # BGR: index 2 = Red
    g = img_float[:, :, 1]
    b = img_float[:, :, 0]

    # Decide image type: grayscale images have R ≈ G ≈ B (mean |R-G| < 0.03)
    mean_rg_diff = float(np.mean(np.abs(r - g)))

    if mean_rg_diff > 0.03:
        # ── COLOR ECG with red grid ────────────────────────────────────────────
        # Step 1: background removal via Wu et al.
        img_mod = img_float.copy()
        img_mod[:, :, 2] = 1.0
        gray_mod = cv2.cvtColor((img_mod * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
        gray_mod_norm = gray_mod.astype(np.float32) / 255.0
        candidate_signal = gray_mod_norm <= threshold

        # Step 2: exclude red-dominant grid pixels (R >> G and R >> B)
        is_red = (r - g > 0.15) & (r - b > 0.15) & (r > 0.3)
        return (candidate_signal & ~is_red).astype(np.uint8) * 255

    else:
        # ── GRAYSCALE ECG ──────────────────────────────────────────────────────
        # Use a low brightness threshold to isolate the dark ECG trace only.
        # The grid (medium gray) and background (near-white) are both excluded.
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        gray_norm = gray.astype(np.float32) / 255.0
        return (gray_norm <= 0.30).astype(np.uint8) * 255


def preprocess_image(image_path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Full preprocessing pipeline: load -> remove header -> remove grid -> binarize.

    Returns:
        original_bgr  : original colour image (for visualization)
        binary        : binary image (255=signal, 0=background)
    """
    img_bgr = load_image(image_path)
    img_bgr = remove_redacted_header(img_bgr)
    binary = remove_red_grid(img_bgr)
    return img_bgr, binary


def save_visualization(img_bgr: np.ndarray, output_path: str):
    """Save a BGR image to disk."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output_path, img_bgr)
    print(f"Saved visualization -> {output_path}")
