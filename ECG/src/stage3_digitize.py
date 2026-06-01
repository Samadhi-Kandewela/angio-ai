"""
stage3_digitize.py
------------------
Stage III: Waveform digitization for a single-lead ECG strip.

Follows the exact method from Wu et al. (2022) Scientific Reports:
  1. Morphological dilation  -> connect signal discontinuities
  2. Largest-component labelling -> isolate ECG signal from artefacts
  3. Median-pixel extraction -> one amplitude value per time column
  4. Calibration             -> convert pixels to mV and milliseconds

Standard ECG calibration:
  - Paper speed : 25 mm/s  (1 mm = 40 ms)
  - Amplitude   : 10 mm/mV (1 mm = 0.1 mV)
  - At 200 dpi  : 1 mm ≈ 7.87 pixels
  We compute the time/voltage resolutions dynamically from image dimensions
  assuming a 10-second rhythm strip.
"""

import cv2
import numpy as np
from skimage.measure import label, regionprops
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path


# Standard ECG calibration constants
PAPER_SPEED_MM_PER_S = 25.0       # mm per second
AMPLITUDE_MM_PER_MV = 10.0        # mm per millivolt
STRIP_DURATION_S = 10.0            # standard rhythm strip duration


def dilate_signal(binary: np.ndarray, kernel_width: int = 5, kernel_height: int = 3) -> np.ndarray:
    """
    Dilation to connect discontinuities in the ECG trace.

    Wu et al.: "image dilation to connect any discontinuities in the ECG signal"
    A kernel with height > 1 is required so that steep QRS slopes (where
    adjacent columns have trace pixels several rows apart) are connected
    in addition to purely horizontal gaps.
    """
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (kernel_width, kernel_height)
    )
    dilated = cv2.dilate(binary, kernel, iterations=2)
    return dilated


def extract_largest_component(binary: np.ndarray) -> np.ndarray:
    """
    Keep only the largest connected component in the binary image.

    Wu et al.: "we considered the largest detectable object in the image as
    the ECG signal of interest and all other objects as artefacts."

    Returns a binary image containing only the largest component.
    """
    labeled = label(binary > 0)
    if labeled.max() == 0:
        return binary  # No components found, return as-is

    regions = regionprops(labeled)
    # Sort by area descending, pick the largest
    largest = max(regions, key=lambda r: r.area)
    mask = (labeled == largest.label).astype(np.uint8) * 255
    return mask


def median_pixel_extraction(binary: np.ndarray) -> np.ndarray:
    """
    For each x-column (time point), compute the median y-coordinate of signal pixels.

    Wu et al.: "we used the median amplitude pixel (y-axis) in the binary image
    to reconstruct the digital ECG signal."

    Returns a 1D array of y-pixel values, length = image width.
    NaN is stored for columns with no signal pixels.
    """
    h, w = binary.shape
    signal_y = np.full(w, np.nan)

    for x in range(w):
        col = binary[:, x]
        signal_rows = np.where(col > 0)[0]
        if len(signal_rows) > 0:
            signal_y[x] = np.median(signal_rows)

    # Interpolate over any NaN gaps (missing columns)
    nan_mask = np.isnan(signal_y)
    if nan_mask.any():
        x_indices = np.arange(w)
        valid = ~nan_mask
        if valid.sum() >= 2:
            signal_y = np.interp(x_indices, x_indices[valid], signal_y[valid])

    return signal_y


def calibrate_signal(
    signal_y_pixels: np.ndarray,
    image_height: int,
    image_width: int,
    baseline_y_local: int,
    dpi: int = 200
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert pixel coordinates to calibrated mV and milliseconds.

    Calibration approach:
      - Time axis: assume image width represents STRIP_DURATION_S seconds.
        time_resolution = STRIP_DURATION_S / image_width  (seconds/pixel)
      - Voltage axis: use DPI to determine pixels-per-mm, then apply
        AMPLITUDE_MM_PER_MV to get mV/pixel.
        pixels_per_mm = dpi / 25.4
        voltage_resolution = 1 / (pixels_per_mm * AMPLITUDE_MM_PER_MV)  (mV/pixel)

    The baseline_y_local (the isoelectric line in pixel coords within the
    cropped image) defines 0 mV. Pixels above baseline = positive voltage.

    Returns:
        time_ms   : 1D array of timestamps in milliseconds
        voltage_mv: 1D array of voltages in millivolts
    """
    w = image_width

    # --- Time calibration ---
    time_resolution_s = STRIP_DURATION_S / w          # seconds per pixel
    time_ms = np.arange(w) * time_resolution_s * 1000  # convert to ms

    # --- Voltage calibration ---
    pixels_per_mm = dpi / 25.4                                    # ~7.87 px/mm at 200dpi
    voltage_resolution_mv = 1.0 / (pixels_per_mm * AMPLITUDE_MM_PER_MV)  # mV/pixel

    # Flip y-axis: in image coords, y increases downward.
    # Signal above baseline (smaller y) = positive voltage.
    voltage_mv = (baseline_y_local - signal_y_pixels) * voltage_resolution_mv

    return time_ms, voltage_mv


def _is_grayscale(img_bgr: np.ndarray) -> bool:
    r = img_bgr[:, :, 2].astype(np.float32) / 255.0
    g = img_bgr[:, :, 1].astype(np.float32) / 255.0
    return float(np.mean(np.abs(r - g))) < 0.03


def _digitize_grayscale(
    cropped_bgr: np.ndarray,
    baseline_y_local: int,
    dpi: int,
    visualize: bool,
    output_prefix: str,
) -> dict:
    """
    Trace extraction for grayscale ECG images via darkest-pixel-per-column.

    The ECG trace is the darkest structure in a grayscale image. For every
    column, argmin(gray) gives the y-coordinate of the trace pixel. This
    completely sidesteps the binary-threshold / morphological pipeline, which
    fails when the trace and grid have overlapping grayscale values.

    A 7-sample median filter removes isolated dark artefacts (dust, JPEG
    compression speckle) without smearing the QRS morphology.
    """
    from scipy.ndimage import median_filter

    gray = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    h, w = gray.shape

    signal_y_raw = np.argmin(gray, axis=0).astype(np.float64)
    signal_y = median_filter(signal_y_raw, size=7)

    time_ms, voltage_mv = calibrate_signal(
        signal_y, h, w, baseline_y_local, dpi
    )

    if visualize:
        _save_digitization_plot(
            # Pass a dummy cleaned binary (white trace on black) for display
            (gray < np.percentile(gray, 5)).astype(np.uint8) * 255,
            (gray < np.percentile(gray, 5)).astype(np.uint8) * 255,
            signal_y, time_ms, voltage_mv, output_prefix,
        )

    print(f"[Stage III] Grayscale argmin extraction complete. "
          f"Signal length: {len(time_ms)} samples")
    print(f"[Stage III] Duration: {time_ms[-1]:.0f} ms | "
          f"Voltage range: [{voltage_mv.min():.3f}, {voltage_mv.max():.3f}] mV")

    return {
        "time_ms": time_ms,
        "voltage_mv": voltage_mv,
        "signal_y": signal_y,
        "cleaned_binary": np.zeros((h, w), dtype=np.uint8),
    }


def digitize(
    cropped_binary: np.ndarray,
    baseline_y_local: int,
    dpi: int = 200,
    visualize: bool = True,
    output_prefix: str = "ecg",
    cropped_original_bgr: np.ndarray = None,
) -> dict:
    """
    Run the full Stage III digitization pipeline on a cropped single-lead binary image.

    Args:
        cropped_binary  : binary uint8 image of the lead region (from Stage I)
        baseline_y_local: baseline row within the cropped image
        dpi             : scanning DPI of the image (default 200)
        visualize       : save diagnostic plots if True
        output_prefix   : prefix for output files

    Returns a dict with:
        time_ms    : calibrated time array (ms)
        voltage_mv : calibrated voltage array (mV)
        signal_y   : raw pixel y-coordinates before calibration
    """
    # For grayscale ECGs, binary thresholding cannot cleanly separate the
    # trace from the grid; use argmin-based extraction instead.
    if cropped_original_bgr is not None and _is_grayscale(cropped_original_bgr):
        return _digitize_grayscale(
            cropped_original_bgr, baseline_y_local, dpi, visualize, output_prefix
        )

    print("[Stage III] Dilating signal to connect discontinuities...")
    dilated = dilate_signal(cropped_binary)

    print("[Stage III] Isolating largest connected component (ECG signal)...")
    cleaned = extract_largest_component(dilated)

    # Erode back after isolation to recover original thickness
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cleaned = cv2.erode(cleaned, kernel, iterations=1)

    print("[Stage III] Extracting 1D signal via median-pixel method...")
    h, w = cleaned.shape
    signal_y = median_pixel_extraction(cleaned)

    print("[Stage III] Calibrating to mV and milliseconds...")
    time_ms, voltage_mv = calibrate_signal(
        signal_y, h, w, baseline_y_local, dpi
    )

    if visualize:
        _save_digitization_plot(
            cropped_binary, cleaned, signal_y,
            time_ms, voltage_mv, output_prefix
        )

    print(f"[Stage III] Digitization complete. Signal length: {len(time_ms)} samples")
    print(f"[Stage III] Duration: {time_ms[-1]:.0f} ms | "
          f"Voltage range: [{voltage_mv.min():.3f}, {voltage_mv.max():.3f}] mV")

    return {
        "time_ms": time_ms,
        "voltage_mv": voltage_mv,
        "signal_y": signal_y,
        "cleaned_binary": cleaned,
    }


def _save_digitization_plot(
    binary_input: np.ndarray,
    binary_cleaned: np.ndarray,
    signal_y: np.ndarray,
    time_ms: np.ndarray,
    voltage_mv: np.ndarray,
    output_prefix: str
):
    """Save a 3-panel diagnostic plot."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 9))
    fig.patch.set_facecolor("#1a1a2e")

    # Panel 1: Input binary
    axes[0].imshow(binary_input, cmap="gray", aspect="auto")
    axes[0].set_title("Binary Input (Grid Removed)", color="white", fontsize=11)
    axes[0].axis("off")

    # Panel 2: Cleaned binary with signal overlay
    axes[1].imshow(binary_cleaned, cmap="gray", aspect="auto")
    x_coords = np.arange(len(signal_y))
    axes[1].plot(x_coords, signal_y, color="#00f5ff", linewidth=1.0, label="Extracted median")
    axes[1].set_title("Cleaned Signal + Median Trace", color="white", fontsize=11)
    axes[1].legend(facecolor="#1a1a2e", labelcolor="white", fontsize=9)
    axes[1].axis("off")

    # Panel 3: Calibrated 1D signal
    axes[2].set_facecolor("#0d1b2a")
    axes[2].plot(time_ms, voltage_mv, color="#00e676", linewidth=1.2)
    axes[2].axhline(0, color="#ff5252", linewidth=0.8, linestyle="--", label="Baseline (0 mV)")
    axes[2].set_xlabel("Time (ms)", color="white")
    axes[2].set_ylabel("Voltage (mV)", color="white")
    axes[2].set_title("Digitized ECG Signal (Calibrated)", color="white", fontsize=11)
    axes[2].tick_params(colors="white")
    axes[2].spines[:].set_color("#555")
    axes[2].legend(facecolor="#0d1b2a", labelcolor="white", fontsize=9)

    plt.tight_layout()
    out_path = f"{output_prefix}_stage3.png"
    plt.savefig(out_path, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"[Stage III] Plot saved -> {out_path}")


if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from stage1_localize import localize

    img_path = sys.argv[1] if len(sys.argv) > 1 else "../sample.png"
    stage1 = localize(img_path, visualize=True)

    h_crop = stage1["cropped_binary"].shape[0]
    baseline_local = stage1["baseline_y"] - stage1["crop_top"]

    result = digitize(
        cropped_binary=stage1["cropped_binary"],
        baseline_y_local=baseline_local,
        visualize=True,
        output_prefix=img_path.replace(".png", "").replace(".jpg", "")
    )

    # Save CSV
    csv_path = img_path.replace(".png", "_signal.csv").replace(".jpg", "_signal.csv")
    data = np.column_stack([result["time_ms"], result["voltage_mv"]])
    np.savetxt(csv_path, data, delimiter=",", header="time_ms,voltage_mv", comments="")
    print(f"Signal saved to CSV -> {csv_path}")
