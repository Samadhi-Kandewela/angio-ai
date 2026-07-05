"""
ecg_segment.py
==============
Fully automatic 12-lead ECG segmentation, signal extraction and clinical
diagnosis from a scanned ECG paper image.

What it does
------------
1. Opens a file-picker dialog to select the ECG image.
2. Auto-detects lead positions by OCR (pytesseract) — falls back to the
   standard 4×3 + 1 rhythm-strip layout if OCR is unavailable or fails.
3. Crops each lead into an individual image.
4. Extracts the ECG waveform from each lead image.
5. Applies clinical rules from the LITFL ECG library to produce a diagnosis:
     • Heart rate & rhythm
     • P-wave presence (AF screen)
     • QRS duration (bundle branch block)
     • Axis (limb leads I and aVF)
     • ST elevation / depression (ischaemia screen)
6. Saves lead images, signal plots, a segmentation preview and a formatted
   clinical report.

Usage
-----
    python ecg_segment.py          # file picker opens
    python ecg_segment.py ecg.png  # direct path
    python ecg_segment.py ecg.png --dpi 300 --out results/

Clinical rules source
---------------------
LITFL ECG Library  https://litfl.com/ecg-library/
  Heart rate      : <60 bradycardia | 60–100 normal | >100 tachycardia
  Rhythm          : CV(R-R) <10% = regular; >10% = irregular
  P-wave          : absent / chaotic before QRS → AF screen
  QRS duration    : <120 ms normal | ≥120 ms = BBB / ventricular origin
  Axis            : lead I + aVF → normal / LAD / RAD / extreme
  ST elevation    : ≥1 mm in ≥2 contiguous leads → ischaemia
  ST depression   : ≥0.5 mm in ≥2 leads → subendocardial ischaemia
  QTc             : >440 ms (men) / >460 ms (women) = prolonged
"""

import argparse
import sys
import math
from pathlib import Path
from collections import OrderedDict

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter, medfilt, butter, filtfilt

# Standard 12-lead layout (most common clinical format).
# The rhythm strip is labelled "II_rhythm" to avoid overwriting lead II.
STANDARD_LAYOUT = [
    ["I",        "aVR", "V1", "V4"],
    ["II",       "aVL", "V2", "V5"],
    ["III",      "aVF", "V3", "V6"],
    ["II_rhythm"],                    # full-width rhythm strip
]

# Map rhythm-strip name back to its base lead for analysis
RHYTHM_STRIP_MAP = {"II_rhythm": "II"}

LEAD_NAMES = {"I", "II", "III", "aVR", "aVL", "aVF",
              "V1", "V2", "V3", "V4", "V5", "V6"}


# =================================================================
# 1.  PREPROCESSING
# =================================================================

def remove_grid(img_bgr: np.ndarray) -> np.ndarray:
    """
    Replace the ECG paper grid with white, keeping the trace.

    Two-stage approach:
    1. HSV colour removal  — catches red/pink grids (standard ECG paper).
    2. Morphological removal — catches grey/black grids in B&W images or
       photocopies where colour detection finds nothing useful.
       Uses a horizontal structuring element of width W//3: this matches
       lines that span at least 33% of the image width (true grid lines
       span 100%), but never matches an isoelectric ECG segment which is
       at most one lead-column wide (≤ 25%).
    """
    hsv   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    red1  = cv2.inRange(hsv, (0,   15, 140), (12,  255, 255))
    red2  = cv2.inRange(hsv, (165, 15, 140), (180, 255, 255))
    light = cv2.inRange(hsv, (0,    0, 210), (180,  40, 255))
    colour_mask = cv2.dilate(cv2.bitwise_or(cv2.bitwise_or(red1, red2), light),
                             np.ones((2, 2), np.uint8))
    out = img_bgr.copy()
    out[colour_mask > 0] = 255

    # ── Morphological horizontal-line removal (B&W / photocopy support) ──
    # After colour removal the only remaining dark pixels should be the trace.
    # If black grid lines are still present (B&W scan), detect and remove them.
    H, W   = img_bgr.shape[:2]
    gray   = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
    dark   = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)[1]
    h_len  = max(15, W // 3)           # must span ≥ 33% of image width
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_len, 1))
    h_lines = cv2.morphologyEx(dark, cv2.MORPH_OPEN, kernel)
    h_lines = cv2.dilate(h_lines, np.ones((2, 2), np.uint8))
    out[h_lines > 0] = 255
    return out


def to_binary(gray: np.ndarray, thr: int = 127) -> np.ndarray:
    """Dark trace → 255, light background → 0."""
    _, b = cv2.threshold(gray, thr, 255, cv2.THRESH_BINARY_INV)
    return b


# =================================================================
# 2.  AUTO LEAD DETECTION
# =================================================================

def _detect_rows(binary: np.ndarray) -> list:
    H, W   = binary.shape
    # Skip left 10% (lead labels) and right 10% (edge noise) so that label
    # text does not fill in the gaps between ECG rows and prevent detection.
    x0     = max(1, W // 10)
    x1     = min(W, W - W // 10)
    proj   = np.convolve(np.sum(binary[:, x0:x1], axis=1).astype(float),
                         np.ones(max(3, H // 80)) / max(3, H // 80), mode="same")
    active = proj > 0.01 * proj.max()
    min_h  = max(10, int(H * 0.04))
    rows, start, prev = [], 0, False
    for y, flag in enumerate(active):
        if flag and not prev:
            start, prev = y, True
        elif not flag and prev:
            if y - start >= min_h:
                rows.append((start, y))
            prev = False
    if prev and H - start >= min_h:
        rows.append((start, H))
    return rows


def _detect_cols(binary_row: np.ndarray, n: int) -> list:
    if n <= 1:
        return [(0, binary_row.shape[1])]
    W      = binary_row.shape[1]
    proj   = np.convolve(np.sum(binary_row, axis=0).astype(float),
                         np.ones(max(3, W // 60)) / max(3, W // 60), mode="same")
    half   = max(5, W // (n * 4))
    splits = [0]
    for i in range(1, n):
        exp  = W * i // n
        lo, hi = max(0, exp - half), min(W, exp + half)
        splits.append(lo + int(np.argmin(proj[lo:hi])))
    splits.append(W)
    return [(splits[i], splits[i + 1]) for i in range(n)]


def _try_ocr(img_bgr: np.ndarray) -> dict:
    """
    Attempt pytesseract OCR to locate lead labels.
    Returns {lead_name: (y, x)} or {} if OCR unavailable / fails.
    """
    try:
        import pytesseract
        from pytesseract import Output

        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        _, thr = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)
        data = pytesseract.image_to_data(thr, output_type=Output.DICT,
                                         config="--psm 11 --oem 3")
        found = {}
        for i, raw in enumerate(data["text"]):
            txt = raw.strip()
            if not txt:
                continue
            # Normalise: V1..V6 may be read as "vi", "v2" etc.
            norm = txt.upper().replace("VI", "V1").replace("VZ", "V2")
            for lead in LEAD_NAMES:
                if norm == lead.upper() and int(data["conf"][i]) > 25:
                    cy = data["top"][i]  + data["height"][i] // 2
                    cx = data["left"][i] + data["width"][i]  // 2
                    if lead not in found:        # keep first (highest) occurrence
                        found[lead] = (cy, cx)
        return found
    except Exception:
        return {}


def _layout_from_ocr(found: dict, img_shape: tuple) -> list:
    """
    Convert OCR label positions to a lead_layout list of lists.
    Groups labels into rows by proximity in y, then sorts each row by x.
    Falls back to STANDARD_LAYOUT if fewer than 10 labels are found.
    """
    if len(found) < 10:
        return None

    # Cluster by y-coordinate: sort all found labels by y, then group
    items  = sorted(found.items(), key=lambda kv: kv[1][0])   # sort by y
    rows   = []
    cur    = [items[0]]
    row_y  = items[0][1][0]

    for name, (y, x) in items[1:]:
        if abs(y - row_y) < img_shape[0] * 0.08:
            cur.append((name, (y, x)))
        else:
            rows.append(sorted(cur, key=lambda kv: kv[1][1]))  # sort by x
            cur = [(name, (y, x))]
            row_y = y
    rows.append(sorted(cur, key=lambda kv: kv[1][1]))

    layout = [[name for name, _ in row] for row in rows]
    return layout


def auto_segment_leads(img_bgr: np.ndarray,
                        row_pad: int = 6,
                        col_pad: int = 4) -> dict:
    """
    Fully automatic lead segmentation.

    Strategy
    --------
    1. Try OCR to locate lead-name labels printed on the image.
       If ≥10 lead names are detected, use their positions to determine
       the layout.
    2. Fall back to the standard 4×3+1 layout, splitting detected rows
       into equal columns.

    Returns OrderedDict  lead_name → {'gray', 'color', 'bbox'}
    """
    clean  = remove_grid(img_bgr)
    gray   = cv2.cvtColor(clean, cv2.COLOR_BGR2GRAY)
    binary = to_binary(gray)
    H, W   = img_bgr.shape[:2]

    # ── Try OCR ───────────────────────────────────────────────────
    ocr_labels = _try_ocr(img_bgr)
    layout     = _layout_from_ocr(ocr_labels, img_bgr.shape)

    if layout:
        print(f"  [OCR] Detected leads: "
              f"{', '.join(n for row in layout for n in row)}")
    else:
        layout = STANDARD_LAYOUT
        print("  [layout] Using standard 4x3+1 layout (I,aVR,V1,V4 / …)")

    # ── Row detection ─────────────────────────────────────────────
    rows       = _detect_rows(binary)
    n_expected = len(layout)
    if len(rows) != n_expected:
        if len(rows) > n_expected:
            # Extra rows (e.g. a patient-info header): keep the N tallest,
            # then re-sort by y-position so row order is preserved.
            rows = sorted(
                sorted(rows, key=lambda r: r[1] - r[0], reverse=True)[:n_expected]
            )
        else:
            rows = [(H * i // n_expected, H * (i + 1) // n_expected)
                    for i in range(n_expected)]

    leads = OrderedDict()
    for row_idx, (y0, y1) in enumerate(rows):
        row_leads = layout[row_idx]
        n_cols    = len(row_leads)
        col_segs  = _detect_cols(binary[y0:y1, :], n_cols)

        for col_idx, (x0, x1) in enumerate(col_segs):
            name = row_leads[col_idx]
            ry0  = max(0, y0 - row_pad);  ry1 = min(H, y1 + row_pad)
            rx0  = max(0, x0 - col_pad);  rx1 = min(W, x1 + col_pad)
            leads[name] = {
                "gray":  gray[ry0:ry1, rx0:rx1].copy(),
                "color": img_bgr[ry0:ry1, rx0:rx1].copy(),
                "bbox":  (ry0, ry1, rx0, rx1),
            }
    return leads


# =================================================================
# 3.  SIGNAL EXTRACTION
# =================================================================

def extract_signal(gray_lead: np.ndarray,
                   smooth_win: int = 7,
                   label_skip_frac: float = 0.06) -> np.ndarray:
    """
    Convert a grayscale lead-strip image to a 1-D ECG signal (mV-ready).

    Digitization pipeline (incorporates ECG/src/stage1_localize.py and
    ECG/src/stage3_digitize.py methods adapted for 12-lead crops):

    1. Skip the leftmost label-text region (lead name printed on image).
    2. Constrained trace tracker: for rhythm strips (W>800) track L→R from
       H/2; for narrow leads track bidirectionally from the darkest anchor
       in the middle 60% of lead height.  max_jump = H×0.20 per column
       prevents locking onto horizontal grid artifacts.
    3. Detect "no-trace" columns (tracked pixel still near background) → NaN.
    4. Linearly interpolate through no-trace gaps (isoelectric baseline).
    5. Stage-I baseline: mode of trace y-values = isoelectric reference.
       Replaces the H/2 assumption; eliminates DC-offset introduced by
       off-centre traces (e.g. hyperacute T-waves in anterior STEMI).
    6. Residual baseline wander removal (global-mean window, DC only).
    7. Outlier spike rejection (>3.5 SD from median).
    8. Median pre-filter (kernel=3) + Savitzky-Golay smooth.
    """
    from scipy.ndimage import uniform_filter1d

    H, W = gray_lead.shape

    # ── 1. Skip lead-label area (left margin) ────────────────────
    if W > 800:
        label_skip_frac = max(label_skip_frac, 0.15)
    x_start = max(1, int(W * label_skip_frac))
    crop    = gray_lead[:, x_start:].astype(float)
    cW      = crop.shape[1]

    # ── 2. Constrained trace tracker ─────────────────────────────
    max_jump = max(4, int(H * 0.20))

    v_lo = int(H * 0.20)
    v_hi = int(H * 0.80)
    mid_crop     = crop[v_lo:v_hi, :]
    mid_min_vals = mid_crop.min(axis=0)

    background = float(np.percentile(crop, 95))

    start_x = int(np.argmin(mid_min_vals))

    signal_y   = np.empty(cW)
    min_at_pos = np.empty(cW)

    def _step(x, center):
        lo  = max(0, int(center) - max_jump)
        hi  = min(H, int(center) + max_jump + 1)
        idx = int(np.argmin(crop[lo:hi, x]))
        return float(lo + idx), float(crop[lo + idx, x])

    if W > 800:
        # Rhythm strip: track L→R from H/2.
        # Bidirectional anchor tracking fails here because the anchor is a
        # label pixel near the top (y≈17-18); backward tracking from there
        # cannot reach the ECG baseline (y≈38, gap > max_jump).
        prev = H / 2.0
        for x in range(cW):
            signal_y[x], min_at_pos[x] = _step(x, prev)
            prev = signal_y[x]
    else:
        # Narrow lead: bidirectional from anchor at darkest middle-zone column.
        anchor_in_mid       = int(np.argmin(mid_crop[:, start_x]))
        signal_y[start_x]   = float(v_lo + anchor_in_mid)
        min_at_pos[start_x] = float(mid_crop[anchor_in_mid, start_x])

        prev = signal_y[start_x]
        for x in range(start_x + 1, cW):
            signal_y[x], min_at_pos[x] = _step(x, prev)
            prev = signal_y[x]

        prev = signal_y[start_x]
        for x in range(start_x - 1, -1, -1):
            signal_y[x], min_at_pos[x] = _step(x, prev)
            prev = signal_y[x]

    # ── 3. Mark no-trace columns as NaN ──────────────────────────
    margin       = max(20.0, background * 0.12)
    no_trace     = min_at_pos > (background - margin)
    signal_y_nan = signal_y.copy()
    signal_y_nan[no_trace] = np.nan

    # ── 4. Interpolate over NaN gaps ─────────────────────────────
    valid = ~np.isnan(signal_y_nan)
    if valid.sum() >= 2:
        x_all    = np.arange(cW, dtype=float)
        signal_y = np.interp(x_all, x_all[valid], signal_y_nan[valid])
    else:
        signal_y = np.full(cW, H / 2.0)

    # ── 5. Stage-I baseline: mode of trace y-values ──────────────
    # The y-coordinate visited most often by the trace = isoelectric line
    # (Stage I from ECG/src/stage1_localize.py, adapted for lead crops).
    # Replaces H/2 as the voltage zero-reference; correct even when the
    # trace is offset from centre due to lead polarity or image layout.
    y_ints     = np.clip(np.round(signal_y).astype(int), v_lo, v_hi - 1)
    counts     = np.bincount(y_ints, minlength=H)
    counts[:v_lo] = 0
    counts[v_hi:] = 0
    baseline_y = float(np.argmax(counts))

    signal = baseline_y - signal_y   # up (smaller y) = positive voltage

    pad    = np.full(x_start, signal[0])
    signal = np.concatenate([pad, signal])

    # ── 6. Residual baseline wander removal (global mean only) ────
    win_bl = min(len(signal), 1000)
    if win_bl > 3:
        baseline = uniform_filter1d(signal.astype(float), size=win_bl, mode="nearest")
        signal   = signal - baseline

    # ── 7. Spike / outlier rejection ─────────────────────────────
    med = float(np.median(signal))
    std = float(np.std(signal)) or 1.0
    signal = np.clip(signal, med - 3.5 * std, med + 3.5 * std)

    # ── 8. Median pre-filter + Savitzky-Golay smooth ─────────────
    signal = medfilt(signal.astype(float), kernel_size=3)
    win = min(smooth_win, len(signal) if len(signal) % 2 == 1 else len(signal) - 1)
    if win >= 3 and len(signal) > win * 2:
        signal = savgol_filter(signal, window_length=win, polyorder=2)

    return signal


def px_to_mv(sig_px, px_per_mm: float, gain: float = 10.0) -> np.ndarray:
    return sig_px / (px_per_mm * gain)


def time_axis(n: int, px_per_mm: float, speed: float = 25.0) -> np.ndarray:
    return np.arange(n) / (px_per_mm * speed)


def estimate_px_per_mm(img_bgr: np.ndarray, known_dpi: float = None) -> float:
    if known_dpi:
        return known_dpi / 25.4
    try:
        gray  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(float)
        H, W  = gray.shape
        proj  = (255.0 - gray[H // 4: 3 * H // 4, :]).mean(axis=0)
        proj -= proj.mean()
        fft   = np.abs(np.fft.rfft(proj)); fft[0] = 0
        idx   = int(np.argmax(fft[1: W // 2])) + 1
        spacing = W / idx
        if 2.0 <= spacing <= 30.0:
            return spacing
    except Exception:
        pass
    return 96.0 / 25.4


# =================================================================
# 4.  PER-LEAD ANALYSIS
# =================================================================

def _highpass_qrs(sig_mv: np.ndarray, fs: float, cutoff: float = 5.0) -> np.ndarray:
    """
    High-pass filter at 5 Hz before R-peak detection.
    QRS complexes dominate 5–40 Hz; P-waves and T-waves sit below 5 Hz.
    This prevents hyperacute or giant T-waves from being mis-detected as
    R-peaks (which would invert all subsequent ST/PR timing measurements).
    """
    nyq = fs / 2.0
    if cutoff >= nyq or len(sig_mv) < 10:
        return sig_mv
    try:
        b, a = butter(3, cutoff / nyq, btype='high')
        return filtfilt(b, a, sig_mv)
    except Exception:
        return sig_mv


def detect_rpeaks(sig_mv: np.ndarray, fs: float) -> np.ndarray:
    """Detect R-peaks using a HPF copy of the signal (T-wave suppression)
    and handle both upright and inverted QRS complexes (e.g. aVR)."""
    filtered  = _highpass_qrs(sig_mv, fs)
    min_dist  = max(1, int(0.35 * fs))

    # Use the 85th-percentile (not 95th) so that a few large-amplitude
    # artefact spikes (calibration pulse, PVCs) do not inflate the threshold
    # above the true QRS amplitude.  Floor at 0.05 mV to handle leads with
    # genuinely small R-waves (loss of R-wave height in anterior STEMI).
    pos_thr = max(0.05, float(np.percentile(filtered, 85)) * 0.30)
    min_width = max(2, int(0.025 * fs))   # reject HPF ringing spikes < 25 ms
    pos_peaks, _ = find_peaks(filtered, height=pos_thr, distance=min_dist,
                              width=min_width)

    neg_thr = max(0.05, abs(float(np.percentile(filtered, 15))) * 0.30)
    neg_peaks, _ = find_peaks(-filtered, height=neg_thr, distance=min_dist,
                              width=min_width)

    if not len(pos_peaks) and not len(neg_peaks):
        return pos_peaks
    if not len(pos_peaks):
        return neg_peaks
    if not len(neg_peaks):
        return pos_peaks

    pos_mean = float(np.mean(filtered[pos_peaks]))
    neg_mean = float(np.mean(-filtered[neg_peaks]))
    return pos_peaks if pos_mean >= neg_mean else neg_peaks


def analyze_lead(sig_px: np.ndarray, name: str,
                 px_per_mm: float, speed: float = 25.0,
                 gain: float = 10.0) -> dict:
    fs       = px_per_mm * speed
    sig_mv   = px_to_mv(sig_px, px_per_mm, gain)
    t_s      = time_axis(len(sig_px), px_per_mm, speed)
    peaks    = detect_rpeaks(sig_mv, fs)
    hr       = (round(60.0 / float(np.mean(np.diff(peaks) / fs)), 1)
                if len(peaks) >= 2 else float("nan"))
    amp      = float(np.percentile(sig_mv, 95) - np.percentile(sig_mv, 5))
    qrs_dur  = _estimate_qrs_duration(sig_mv, peaks, fs)
    st_dev   = _measure_st_deviation(sig_mv, peaks, fs)
    p_present = _detect_p_waves(sig_mv, peaks, fs)
    net_qrs  = _net_qrs_deflection(sig_mv, peaks, fs)

    return {
        "lead":         name,
        "signal_mv":    sig_mv,
        "time_s":       t_s,
        "r_peaks":      peaks,
        "hr_bpm":       hr,
        "amplitude_mv": round(amp, 3),
        "duration_s":   round(float(t_s[-1]), 2) if len(t_s) > 1 else 0.0,
        "fs_hz":        round(fs, 1),
        "n_beats":      len(peaks),
        "qrs_dur_ms":   qrs_dur,
        "st_dev_mv":    st_dev,
        "p_present":    p_present,
        "net_qrs_mv":   net_qrs,
    }


# ── Signal measurement helpers ─────────────────────────────────

def _estimate_qrs_duration(sig_mv, peaks, fs):
    """
    Estimate mean QRS duration in ms.

    Uses a fixed ±80 ms window around each R-peak and finds where the
    signal first crosses 20% of the R-peak amplitude on either side.
    This handles both upright and inverted QRS complexes and is robust
    to low-amplitude leads where the 50%-height method fails.
    """
    if len(peaks) == 0:
        return float("nan")

    hw  = int(0.08 * fs)   # 80 ms half-window
    durs = []
    for p in peaks:
        seg_lo = max(0, p - hw)
        seg_hi = min(len(sig_mv) - 1, p + hw)
        h = sig_mv[p]
        if abs(h) < 0.05:          # too small to measure reliably
            continue
        thr = h * 0.20             # 20% of peak amplitude

        # Walk left from R until signal falls below threshold
        lo = p
        while lo > seg_lo and sig_mv[lo] * np.sign(h) > thr * np.sign(h):
            lo -= 1

        # Walk right from R until signal falls below threshold
        hi = p
        while hi < seg_hi and sig_mv[hi] * np.sign(h) > thr * np.sign(h):
            hi += 1

        dur_ms = (hi - lo) / fs * 1000.0
        if 20 <= dur_ms <= 200:    # physiological range: 20–200 ms
            durs.append(dur_ms)

    return round(float(np.median(durs)), 1) if durs else float("nan")


def _measure_st_deviation(sig_mv, peaks, fs):
    """
    Measure mean ST deviation (mV) at J+40 ms relative to baseline.
    Baseline = mean of signal 80–40 ms before each R-peak (PR segment).
    """
    if len(peaks) == 0:
        return float("nan")
    j_offset  = int(0.04 * fs)   # 40 ms after approximate J-point
    qrs_half  = int(0.05 * fs)   # approximate half-QRS width
    pr_start  = int(0.08 * fs)   # 80 ms before R
    pr_end    = int(0.04 * fs)   # 40 ms before R
    devs = []
    for p in peaks:
        j   = p + qrs_half
        st  = j + j_offset
        bs_lo = p - pr_start
        bs_hi = p - pr_end
        if bs_lo < 0 or st >= len(sig_mv):
            continue
        baseline = float(np.mean(sig_mv[bs_lo:bs_hi]))
        devs.append(float(sig_mv[st]) - baseline)
    return round(float(np.mean(devs)), 3) if devs else float("nan")


def _detect_p_waves(sig_mv, peaks, fs):
    """
    Return True if P-waves are likely present (small peak before R).

    Search window: 100–350 ms before R-peak.
    The upper bound of 350 ms covers 1st-degree AV block (PR up to ~400 ms)
    without overlapping the previous beat's T-wave at typical heart rates.
    The original 120–220 ms window missed P-waves in 1st-degree AV block,
    causing false AF diagnoses.
    """
    if len(peaks) < 2:
        return None
    pr_lo = int(0.10 * fs)   # 100 ms  (was 120 ms)
    pr_hi = int(0.35 * fs)   # 350 ms  (was 220 ms)
    found = 0
    for p in peaks:
        seg_start = p - pr_hi
        seg_end   = p - pr_lo
        if seg_start < 0:
            continue
        seg = sig_mv[seg_start:seg_end]
        if len(seg) == 0:
            continue
        p_peaks, _ = find_peaks(seg, height=0.05,
                                 distance=max(1, len(seg) // 3))
        if len(p_peaks) > 0:
            found += 1
    return found >= len(peaks) * 0.5


def _net_qrs_deflection(sig_mv, peaks, fs):
    """Net QRS area (integral) — positive = upright QRS."""
    if len(peaks) == 0:
        return float("nan")
    hw    = int(0.06 * fs)     # ±60 ms around R-peak
    areas = []
    for p in peaks:
        lo = max(0, p - hw)
        hi = min(len(sig_mv), p + hw)
        areas.append(float(np.trapezoid(sig_mv[lo:hi])))
    return round(float(np.mean(areas)), 3) if areas else float("nan")


# =================================================================
# 5.  CLINICAL DIAGNOSIS  (LITFL rules)
# =================================================================

class ECGDiagnoser:
    """
    Applies LITFL-based clinical rules to 12-lead ECG analysis results.

    Rules reference: https://litfl.com/ecg-library/
    """

    # Contiguous lead groups for ischaemia localisation
    CONTIGUOUS = {
        "inferior":   ["II", "III", "aVF"],
        "lateral":    ["I", "aVL", "V5", "V6"],
        "anterior":   ["V1", "V2", "V3", "V4"],
        "septal":     ["V1", "V2"],
        "high_lat":   ["I", "aVL"],
    }

    def diagnose(self, lead_results: list, sex: str = "M") -> dict:
        L = {r["lead"]: r for r in lead_results}

        hr_bpm     = self._mean_hr(L)
        rhythm     = self._rhythm(L)
        p_wave     = self._p_wave_status(L)
        qrs_ms     = self._mean_qrs(L)
        axis_name  = self._axis(L)
        st_up      = self._st_changes(L, direction="elevation")
        st_dn      = self._st_changes(L, direction="depression")
        qtc        = self._qtc(L)

        findings = []
        severity = "NORMAL"

        # ── Heart rate ───────────────────────────────────────────
        if not math.isnan(hr_bpm):
            if hr_bpm < 60:
                findings.append(f"Bradycardia  ({hr_bpm:.0f} bpm)")
                severity = "ABNORMAL"
            elif hr_bpm > 100:
                findings.append(f"Tachycardia  ({hr_bpm:.0f} bpm)")
                severity = "ABNORMAL"
            else:
                findings.append(f"Normal heart rate  ({hr_bpm:.0f} bpm)")

        # ── Rhythm ───────────────────────────────────────────────
        if rhythm == "irregular":
            if not p_wave:
                findings.append("Irregular rhythm with absent P-waves "
                                 "— likely Atrial Fibrillation")
                severity = "CRITICAL"
            else:
                findings.append("Irregular rhythm (P-waves present) "
                                 "— consider sinus arrhythmia / ectopics")
                severity = "ABNORMAL"
        else:
            findings.append("Regular rhythm")

        # ── P-waves ──────────────────────────────────────────────
        if p_wave is False:
            findings.append("P-waves absent or disorganised")

        # ── Axis ─────────────────────────────────────────────────
        if axis_name != "Normal":
            findings.append(f"{axis_name}")
            if severity == "NORMAL":
                severity = "ABNORMAL"

        # ── QRS duration ─────────────────────────────────────────
        if not math.isnan(qrs_ms):
            if qrs_ms >= 120:
                findings.append(f"Broad QRS ({qrs_ms:.0f} ms) "
                                 "— Bundle Branch Block or ventricular origin")
                severity = "ABNORMAL"
            else:
                findings.append(f"Normal QRS duration ({qrs_ms:.0f} ms)")

        # ── ST elevation ─────────────────────────────────────────
        for territory, leads_affected in st_up.items():
            if leads_affected:
                joined = ", ".join(leads_affected)
                findings.append(
                    f"ST ELEVATION in {territory} leads ({joined}) "
                    "— consider STEMI / acute ischaemia")
                severity = "CRITICAL"

        # ── ST depression ────────────────────────────────────────
        for territory, leads_affected in st_dn.items():
            if leads_affected:
                joined = ", ".join(leads_affected)
                findings.append(
                    f"ST depression in {territory} leads ({joined}) "
                    "— consider subendocardial ischaemia")
                if severity == "NORMAL":
                    severity = "ABNORMAL"

        # ── QTc ──────────────────────────────────────────────────
        qtc_limit = 440 if sex.upper() == "M" else 460
        if not math.isnan(qtc):
            if qtc > 500:
                findings.append(f"Severely prolonged QTc ({qtc:.0f} ms) "
                                 "— high Torsades risk")
                severity = "CRITICAL"
            elif qtc > qtc_limit:
                findings.append(f"Prolonged QTc ({qtc:.0f} ms > {qtc_limit} ms)")
                if severity == "NORMAL":
                    severity = "ABNORMAL"

        if severity == "NORMAL" and not findings:
            findings.append("No acute abnormalities detected")

        return {
            "heart_rate_bpm": round(hr_bpm, 1) if not math.isnan(hr_bpm) else None,
            "rhythm":         rhythm,
            "p_waves":        p_wave,
            "qrs_dur_ms":     round(qrs_ms, 1) if not math.isnan(qrs_ms) else None,
            "axis":           axis_name,
            "st_elevation":   {k: v for k, v in st_up.items() if v},
            "st_depression":  {k: v for k, v in st_dn.items() if v},
            "qtc_ms":         round(qtc, 1) if not math.isnan(qtc) else None,
            "findings":       findings,
            "severity":       severity,
        }

    # ── Helpers ────────────────────────────────────────────────

    def _mean_hr(self, L):
        # Prefer the rhythm strip when it detected enough beats.
        # Require ≥ 6 beats (5 R-R intervals) so that a strip with only
        # 4 noisy detections (38 bpm) does not override 12 cleaner leads.
        for strip in ("II_rhythm", "II"):
            if strip in L and not math.isnan(L[strip]["hr_bpm"]) \
                    and L[strip]["n_beats"] >= 6:
                return L[strip]["hr_bpm"]
        # Fallback: median of all leads with ≥ 2 beats (median is more
        # robust than mean against outlier leads with wrong peak counts).
        hrs = [r["hr_bpm"] for r in L.values()
               if not math.isnan(r["hr_bpm"]) and r["n_beats"] >= 2]
        return float(np.median(hrs)) if hrs else float("nan")

    def _rhythm(self, L):
        # Use rhythm strip if available, else lead with most beats
        best = (L.get("II_rhythm") or
                max(L.values(), key=lambda r: r["n_beats"]))
        peaks = best["r_peaks"]
        if len(peaks) < 3:
            return "undetermined"
        rr  = np.diff(peaks).astype(float)
        cv  = np.std(rr) / np.mean(rr) if np.mean(rr) > 0 else 0
        return "irregular" if cv > 0.10 else "regular"

    def _p_wave_status(self, L):
        """Aggregate P-wave detection across limb leads."""
        results = [r["p_present"] for r in L.values()
                   if r["p_present"] is not None]
        if not results:
            return None
        return sum(results) / len(results) >= 0.5

    def _mean_qrs(self, L):
        # Use only limb leads and V1-V3 where QRS is clearest;
        # exclude leads where the amplitude is very low (unreliable measurement).
        reliable = ["II", "II_rhythm", "I", "V1", "V2", "V3", "aVF"]
        vals = [L[n]["qrs_dur_ms"] for n in reliable
                if n in L
                and not math.isnan(L[n]["qrs_dur_ms"])
                and L[n]["amplitude_mv"] > 0.3]
        if not vals:   # fallback: use all leads
            vals = [r["qrs_dur_ms"] for r in L.values()
                    if not math.isnan(r["qrs_dur_ms"])]
        return float(np.median(vals)) if vals else float("nan")

    def _axis(self, L):
        """
        Determine electrical axis from leads I and aVF net QRS deflection.
        Normal:    I +,  aVF +  → -30° to +90°
        LAD:       I +,  aVF -  → < -30°
        RAD:       I -,  aVF +  → > +90°
        Extreme:   I -,  aVF -  → northwest axis
        """
        i_net   = L.get("I",   {}).get("net_qrs_mv", float("nan"))
        avf_net = L.get("aVF", {}).get("net_qrs_mv", float("nan"))
        if math.isnan(i_net) or math.isnan(avf_net):
            return "Axis undetermined"
        if i_net >= 0 and avf_net >= 0:
            return "Normal axis"
        if i_net >= 0 and avf_net < 0:
            return "Left Axis Deviation (LAD)"
        if i_net < 0  and avf_net >= 0:
            return "Right Axis Deviation (RAD)"
        return "Extreme (Northwest) Axis Deviation"

    def _st_changes(self, L, direction: str) -> dict:
        """
        Return dict of territory → list of affected leads.
        Elevation threshold : ≥0.10 mV (1 mm at standard gain).
        Depression threshold: ≥0.05 mV (0.5 mm).
        Requires ≥2 contiguous leads in a territory.
        """
        thr = 0.10 if direction == "elevation" else -0.05
        positive = (lambda v: v >= thr) if direction == "elevation" \
                   else (lambda v: v <= thr)

        # For ST elevation only: a single lead with large reliable elevation
        # (≥ 0.25 mV with ≥ 2 detected beats) may flag STEMI even without a
        # second contiguous lead.  "Reliable" means n_beats ≥ 2 so that the
        # ST measurement is averaged over multiple beats.  This handles B&W
        # images where adjacent leads fail to detect any peaks.
        # Single-peak leads (HR n/a, n_beats < 2) are excluded to prevent
        # artefact T-wave detections from triggering false CRITICAL findings.
        large_elev_thr = 0.25 if direction == "elevation" else None

        # Median HR from leads with ≥ 3 beats — used to flag concordance.
        # A lead whose HR is < 40% of the median is likely under-detecting
        # R-peaks; its averaged ST measurement is unreliable.
        _reliable_hrs = [L[k]["hr_bpm"] for k in L
                         if not math.isnan(L[k]["hr_bpm"])
                         and L[k]["n_beats"] >= 3]
        _median_hr = float(np.median(_reliable_hrs)) if _reliable_hrs \
                     else float("nan")

        results = {}
        for territory, members in self.CONTIGUOUS.items():
            # Only leads with ≥ 2 detected beats have a reliable averaged
            # ST measurement.  Single-peak (HR n/a) leads are excluded to
            # prevent two artefact T-wave detections from triggering a false
            # STEMI alarm via the ≥ 2 contiguous leads criterion.
            # HR concordance: exclude leads whose HR is < 40% of the overall
            # median (strongly discordant HR → suspect poor detection).
            affected = [m for m in members
                        if m in L
                        and not math.isnan(L[m]["st_dev_mv"])
                        and L[m]["n_beats"] >= 2
                        and positive(L[m]["st_dev_mv"])
                        and (math.isnan(_median_hr)
                             or L[m]["hr_bpm"] >= 0.40 * _median_hr)]
            if len(affected) >= 2:
                results[territory] = affected
            elif (large_elev_thr is not None
                  and any(L[m]["st_dev_mv"] >= large_elev_thr
                          and L[m]["n_beats"] >= 2
                          for m in affected)):
                results[territory] = affected
            elif direction == "elevation":
                # Single-beat very-large elevation (≥ 0.50 mV with valid QRS):
                # catches genuine STEMI when R-peak detection is limited by
                # small-amplitude signals (anterior STEMI, loss of R-wave height).
                # Threshold is 2× the multi-beat exception to compensate for
                # single-measurement uncertainty.
                extra = [m for m in members
                         if m in L
                         and not math.isnan(L[m]["st_dev_mv"])
                         and not math.isnan(L[m]["qrs_dur_ms"])
                         and L[m]["n_beats"] == 1
                         and L[m]["st_dev_mv"] >= 0.50]
                results[territory] = extra
            else:
                results[territory] = []
        return results

    def _qtc(self, L):
        """
        Estimate QTc using Bazett's formula:  QTc = QT / sqrt(RR).
        Approximate QT from QRS duration + fixed ST+T estimate when
        T-wave end is not reliably detected from image signals.
        """
        best = (L.get("II_rhythm") or
                max(L.values(), key=lambda r: r["n_beats"]))
        peaks = best["r_peaks"]
        if len(peaks) < 2:
            return float("nan")
        fs    = best["fs_hz"]
        rr_s  = float(np.mean(np.diff(peaks))) / fs
        qrs_s = best["qrs_dur_ms"] / 1000.0
        if math.isnan(qrs_s) or qrs_s <= 0:
            return float("nan")
        # Approximate QT ≈ QRS + estimated ST+T duration
        qt_s  = qrs_s + 0.20    # rough fixed ST+T estimate
        qtc_s = qt_s / math.sqrt(rr_s) if rr_s > 0 else float("nan")
        return qtc_s * 1000.0   # convert to ms


# =================================================================
# 6.  SAVE OUTPUTS
# =================================================================

def save_lead_images(leads: dict, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, d in leads.items():
        cv2.imwrite(str(out_dir / f"lead_{name.replace('/', '-')}.png"),
                    d["color"])


def save_segmentation_preview(img_bgr: np.ndarray, leads: dict, out_dir: Path):
    """Draw bounding boxes on the original image."""
    vis    = img_bgr.copy()
    colors = [(220,0,0),(0,180,0),(0,0,220),(180,140,0),
              (0,180,180),(180,0,180),(100,220,0),(0,100,220)]
    for i, (name, d) in enumerate(leads.items()):
        y0, y1, x0, x1 = d["bbox"]
        col = colors[i % len(colors)]
        cv2.rectangle(vis, (x0, y0), (x1, y1), col, 2)
        cv2.putText(vis, name, (x0 + 5, y0 + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, col, 2, cv2.LINE_AA)
    cv2.imwrite(str(out_dir / "segmentation_preview.png"), vis)
    print(f"  Preview        -> {out_dir / 'segmentation_preview.png'}")


def save_signal_plots(results: list, out_dir: Path):
    n    = len(results)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols,
                              figsize=(cols * 5, rows * 2.5), squeeze=False)
    flat = axes.flatten()

    for i, r in enumerate(results):
        ax  = flat[i]
        t, s = r["time_s"], r["signal_mv"]
        ax.plot(t, s, color="black", lw=0.8)
        rp = r["r_peaks"]
        if len(rp):
            ax.plot(t[rp], s[rp], "rv", markersize=5)
        hr_str = f"{r['hr_bpm']:.0f} bpm" if not math.isnan(r["hr_bpm"]) else "n/a"
        ax.set_title(f"{r['lead']}   {hr_str}   QRS {r['qrs_dur_ms']:.0f} ms"
                     if not math.isnan(r["qrs_dur_ms"]) else
                     f"{r['lead']}   {hr_str}", fontsize=9)
        ax.axhline(0, color="gray", lw=0.4, ls="--")
        ax.set_xlabel("s", fontsize=7); ax.set_ylabel("mV", fontsize=7)
        ax.tick_params(labelsize=7)

    for i in range(n, len(flat)):
        flat[i].set_visible(False)

    plt.suptitle("ECG — Lead Signals", fontweight="bold", fontsize=12)
    plt.tight_layout()
    path = out_dir / "all_leads_signals.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Signal plot    -> {path}")


def save_diagnosis_report(diag: dict, out_dir: Path, img_name: str = ""):
    """Save a formatted plain-text clinical report and a colour-coded figure."""

    sev_color = {"NORMAL":   "#2ecc71",
                 "ABNORMAL": "#e67e22",
                 "CRITICAL": "#e74c3c"}.get(diag["severity"], "#95a5a6")

    # ── Text report ──────────────────────────────────────────────
    lines = [
        "=" * 60,
        "  AUTOMATED ECG INTERPRETATION REPORT",
        f"  Image  : {img_name}",
        "=" * 60,
        "",
        f"  Overall severity   : {diag['severity']}",
        f"  Heart rate         : {diag['heart_rate_bpm']} bpm"
          if diag['heart_rate_bpm'] else "  Heart rate         : undetermined",
        f"  Rhythm             : {diag['rhythm']}",
        f"  P-waves            : {'present' if diag['p_waves'] else 'absent/unclear'}",
        f"  QRS duration       : {diag['qrs_dur_ms']} ms"
          if diag['qrs_dur_ms'] else "  QRS duration       : undetermined",
        f"  Electrical axis    : {diag['axis']}",
        f"  QTc (estimated)    : {diag['qtc_ms']} ms"
          if diag['qtc_ms'] else "  QTc                : undetermined",
        "",
        "  FINDINGS",
        "  " + "-" * 50,
    ]
    for f in diag["findings"]:
        lines.append(f"    * {f}")

    lines += [
        "",
        "  NOTE: This is an automated screen. Clinical correlation",
        "        and expert review are required before any decision.",
        "=" * 60,
    ]

    report_txt = out_dir / "diagnosis_report.txt"
    report_txt.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Diagnosis report -> {report_txt}")
    for line in lines:
        print(line)

    # ── Visual summary figure ─────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, max(3, len(diag["findings"]) * 0.45 + 2.5)))
    ax.set_facecolor(sev_color + "22")
    fig.patch.set_facecolor("white")
    ax.axis("off")

    title = f"ECG REPORT   —   {diag['severity']}"
    ax.text(0.5, 0.97, title, ha="center", va="top",
            fontsize=14, fontweight="bold", color=sev_color,
            transform=ax.transAxes)

    body  = "\n".join(f"  • {f}" for f in diag["findings"])
    meta  = (f"\n  HR: {diag['heart_rate_bpm']} bpm"
             f"   Rhythm: {diag['rhythm']}"
             f"   QRS: {diag['qrs_dur_ms']} ms"
             f"   Axis: {diag['axis']}")

    ax.text(0.05, 0.85, meta, ha="left", va="top",
            fontsize=10, color="#2c3e50", transform=ax.transAxes,
            fontfamily="monospace")
    ax.text(0.05, 0.65, body, ha="left", va="top",
            fontsize=10, color="#2c3e50", transform=ax.transAxes)
    ax.text(0.5, 0.02,
            "Automated screen only — expert review required",
            ha="center", va="bottom", fontsize=8, color="gray",
            transform=ax.transAxes)

    plt.tight_layout()
    fig_path = out_dir / "diagnosis_summary.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Diagnosis figure -> {fig_path}")


# =================================================================
# 7.  MAIN PIPELINE
# =================================================================

def pick_image_file() -> str:
    """Open a file-picker dialog; fall back to console prompt."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.attributes("-topmost", True)
        root.withdraw()
        path = filedialog.askopenfilename(
            title="Select ECG Image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"),
                       ("All files", "*.*")])
        root.destroy()
        if not path:
            sys.exit("No file selected.")
        return path
    except Exception:
        path = input("Path to ECG image: ").strip().strip('"\'')
        if not path:
            sys.exit("No path entered.")
        return path


def run(img_path:    str,
        out_dir:     str   = "ecg_output",
        paper_speed: float = 25.0,
        paper_gain:  float = 10.0,
        dpi:         float = None,
        sex:         str   = "M",
        model               = None) -> dict:
    """
    Full pipeline — no user prompts required.

    Each image gets its own sub-folder inside out_dir:
        ecg_output/
            ecg00051/
                original.png          <- copy of the input image
                segmentation_preview.png
                all_leads_signals.png
                diagnosis_report.txt
                diagnosis_summary.png
                lead_images/
                    lead_I.png  ...

    Returns the diagnosis dict.
    """
    img_path = str(img_path)
    stem     = Path(img_path).stem          # e.g. "ecg00051"
    out      = Path(out_dir) / stem         # unique sub-folder per image
    out.mkdir(parents=True, exist_ok=True)

    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(img_path)

    # ── Save a copy of the original image for easy debugging ──────
    orig_copy = out / f"original{Path(img_path).suffix}"
    cv2.imwrite(str(orig_copy), img)

    print(f"\n[1] Image      {Path(img_path).name}  "
          f"({img.shape[1]} x {img.shape[0]} px)")
    print(f"    Output     -> {out}")

    px_per_mm = estimate_px_per_mm(img, known_dpi=dpi)
    print(f"[2] Scale      {px_per_mm:.2f} px/mm  "
          f"(fs = {px_per_mm * paper_speed:.1f} Hz)")

    print("[3] Segmenting leads (auto) ...")
    leads = auto_segment_leads(img)
    print(f"    Extracted: {', '.join(leads)}")
    save_lead_images(leads, out / "lead_images")
    save_segmentation_preview(img, leads, out)

    print("[4] Extracting signals ...")
    results = []
    for name, data in leads.items():
        sig_px = extract_signal(data["gray"])
        res    = analyze_lead(sig_px, name, px_per_mm, paper_speed, paper_gain)
        res["model_result"] = (model(res["signal_mv"], name)
                               if model else {"result": "no model loaded"})
        results.append(res)
        hr_s = f"{res['hr_bpm']:.0f} bpm" if not math.isnan(res["hr_bpm"]) else "n/a"
        print(f"    {name:<6}  HR {hr_s:<10}  "
              f"QRS {res['qrs_dur_ms']:.0f} ms  "
              f"ST {res['st_dev_mv']:+.3f} mV")

    print("[5] Clinical diagnosis ...")
    diagnoser = ECGDiagnoser()
    diag      = diagnoser.diagnose(results, sex=sex)

    print("[6] Saving outputs ...")
    save_signal_plots(results, out)
    save_diagnosis_report(diag, out, img_name=Path(img_path).name)

    return diag


def main():
    parser = argparse.ArgumentParser(
        description="Automatic 12-lead ECG segmentation and clinical diagnosis.")
    parser.add_argument("image", nargs="?", default=None,
                        help="ECG image path. Omit to open file picker.")
    parser.add_argument("--out",   default="ecg_output")
    parser.add_argument("--speed", type=float, default=25.0)
    parser.add_argument("--gain",  type=float, default=10.0)
    parser.add_argument("--dpi",   type=float, default=None)
    parser.add_argument("--sex",   default="M",
                        help="Patient sex M/F for QTc threshold (default M).")
    args = parser.parse_args()

    img_path = args.image or pick_image_file()
    if not Path(img_path).exists():
        sys.exit(f"File not found: {img_path}")

    run(img_path,
        out_dir=args.out,
        paper_speed=args.speed,
        paper_gain=args.gain,
        dpi=args.dpi,
        sex=args.sex)


if __name__ == "__main__":
    main()
