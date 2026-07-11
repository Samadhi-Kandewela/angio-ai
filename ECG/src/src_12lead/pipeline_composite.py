"""
pipeline_composite.py
---------------------
End-to-end pipeline for composite 12-lead ECG images (single scan).

Handles the standard 4×3 lead grid on both colour and black-and-white
ECG paper:
    Row 0 :  I    aVR  V1   V4
    Row 1 :  II   aVL  V2   V5
    Row 2 :  III  aVF  V3   V6

Run with no arguments to open a file-browser dialog.
Outputs saved to:  ECG/ecg-12-lead-model/pipeline_outputs/<stem>_<timestamp>/

Optional CLI:
    python pipeline_composite.py --img <path> [--model <pth>] [--output_dir <dir>]
"""

import os
import sys
import json
import math
import argparse
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from scipy.signal import butter, filtfilt, find_peaks, savgol_filter, medfilt, resample
from scipy.ndimage import uniform_filter1d

# ── Path setup ─────────────────────────────────────────────────────────────────
_HERE          = Path(__file__).resolve().parent
_ECG_ROOT      = _HERE.parent.parent
_MODEL_DIR     = _ECG_ROOT / "ecg-12-lead-model"
_OUTPUT_ROOT   = _MODEL_DIR / "pipeline_outputs"
_DEFAULT_MODEL = (_MODEL_DIR / "run_20260603_141028" / "ecg_resnet1d_best.pth"
                  if (_MODEL_DIR / "run_20260603_141028" / "ecg_resnet1d_best.pth").exists()
                  else _MODEL_DIR / "ecg_resnet1d_best.pth")
_ANGIO_SRC     = _ECG_ROOT.parent / "src"

sys.path.insert(0, str(_ANGIO_SRC))

# Import ecg_segment helpers (graceful fallback if not present)
try:
    from ecg_segment import (
        estimate_px_per_mm, px_to_mv, time_axis,
        ECGDiagnoser, save_diagnosis_report,
        _estimate_qrs_duration, _measure_st_deviation,
        _detect_p_waves, _net_qrs_deflection,
    )
    # auto_segment_leads imported separately (used only for coloured images)
    try:
        from ecg_segment import auto_segment_leads as _auto_seg
        _HAS_AUTO_SEG = True
    except ImportError:
        _HAS_AUTO_SEG = False
    _HAS_SEG = True
except ImportError:
    _HAS_SEG = False
    _HAS_AUTO_SEG = False

from stage4_classify import LEAD_NAMES, SIGNAL_LENGTH, CLASS_NAMES

# ── ML Classifier: load the training-compatible version via importlib ──────────
import importlib.util as _ilu
import torch as _torch

_S4_12L_PATH = _ECG_ROOT / "src" / "12-lead-classification" / "stage4_classify.py"
if _S4_12L_PATH.exists():
    _spec   = _ilu.spec_from_file_location("_s4_12l", str(_S4_12L_PATH))
    _s4_12l = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_s4_12l)
    _ML_CLASSIFY    = _s4_12l.classify          # classify(lead_signals_dict, model)
    _ML_ECGResNet1D = _s4_12l.ECGResNet1D
    _ML_AVAIL = True
else:
    _ML_AVAIL = False


def _load_model_robust(path: str):
    """Load ECGResNet1D; handles both bare state-dict and wrapped checkpoints."""
    if _ML_AVAIL:
        model = _ML_ECGResNet1D(in_channels=12)
    else:
        from stage4_classify import ECGResNet1D
        model = ECGResNet1D()
    try:
        try:
            ckpt = _torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:                        # older PyTorch without weights_only
            ckpt = _torch.load(path, map_location="cpu")
    except Exception:
        raise
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        ckpt = ckpt["model_state_dict"]
    model.load_state_dict(ckpt)
    model.eval()
    return model

# ── Lead layout ────────────────────────────────────────────────────────────────
LAYOUT_4x3 = [
    ["I",   "aVR", "V1", "V4"],
    ["II",  "aVL", "V2", "V5"],
    ["III", "aVF", "V3", "V6"],
]
_LEAD_IDX = {name: i for i, name in enumerate(LEAD_NAMES)}


# ══════════════════════════════════════════════════════════════════════════════
# File browser
# ══════════════════════════════════════════════════════════════════════════════
def _browse_for_image() -> str:
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError:
        sys.exit("tkinter is not available.  Pass --img on the command line.")
    root = tk.Tk()
    root.withdraw()
    root.lift()
    root.attributes("-topmost", True)
    path = filedialog.askopenfilename(
        title="Select composite ECG image (single 12-lead scan)",
        filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"),
                   ("All files",   "*.*")],
        initialdir=str(_ECG_ROOT),
    )
    root.destroy()
    if not path:
        sys.exit("No file selected.  Exiting.")
    return path


# ══════════════════════════════════════════════════════════════════════════════
# Image type detection
# ══════════════════════════════════════════════════════════════════════════════
def _is_greyscale(img_bgr: np.ndarray) -> bool:
    """Return True if the image has no meaningful colour (B&W / greyscale scan)."""
    if img_bgr.ndim < 3 or img_bgr.shape[2] < 3:
        return True
    diff = (np.abs(img_bgr[:, :, 0].astype(np.int16) -
                   img_bgr[:, :, 2].astype(np.int16)).mean())
    return diff < 12.0


# ══════════════════════════════════════════════════════════════════════════════
# Grid-based segmentation  (primary for B&W; fallback for colour)
# ══════════════════════════════════════════════════════════════════════════════
def _find_cuts(proj: np.ndarray, n: int) -> list:
    """
    Find n+1 boundary indices that split proj into n segments.

    For B&W ECG images the 'separator' may just be slightly brighter white
    space or may not exist at all — in that case equal division is the correct
    answer and the function returns it directly.
    """
    total = len(proj)
    step  = total // n
    equal = [i * step for i in range(n)] + [total]

    sm = np.convolve(proj, np.ones(max(1, step // 12)) / max(1, step // 12),
                     mode="same")
    if sm.max() < 200:           # too dark — skip brightness detection
        return equal

    threshold = sm.max() * 0.90
    cuts = [0]
    for i in range(1, n):
        anchor  = equal[i]
        halfwin = max(int(step * 0.30), 20)
        window  = [j for j in range(max(0, anchor - halfwin),
                                    min(total, anchor + halfwin))
                   if sm[j] >= threshold]
        cuts.append(int(np.median(window)) if window else anchor)
    cuts.append(total)
    return cuts


def _segment_grid(img_bgr: np.ndarray):
    """
    Segment a composite ECG image into 12 lead strips using a 4×3 grid.

    Returns:
        leads  : dict  {lead_name: gray_uint8_ndarray}
        bounds : dict  {lead_name: (x1, y1, x2, y2)}  — actual pixel bounds
    """
    gray = (cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            if img_bgr.ndim == 3 else img_bgr.copy())

    # Normalise: ensure background = bright (white)
    if float(np.median(gray)) < 128:
        gray = 255 - gray

    row_cuts = _find_cuts(gray.mean(axis=1), n=3)   # horizontal boundaries
    col_cuts = _find_cuts(gray.mean(axis=0), n=4)   # vertical boundaries

    leads  = {}
    bounds = {}
    for r, row_names in enumerate(LAYOUT_4x3):
        y1, y2 = row_cuts[r], row_cuts[r + 1]
        for c, name in enumerate(row_names):
            x1, x2 = col_cuts[c], col_cuts[c + 1]
            leads[name]  = gray[y1:y2, x1:x2]
            bounds[name] = (x1, y1, x2, y2)

    return leads, bounds


def _validate_crops(leads: dict, bounds: dict, img_w: int, img_h: int) -> bool:
    """
    Return True if all crops are plausibly sized.
    A crop is suspicious if it is < 25% of the expected column width.
    """
    min_w = img_w // (4 * 4)    # 25% of W/4
    min_h = img_h // (3 * 4)    # 25% of H/3
    for name, gray in leads.items():
        h, w = gray.shape[:2]
        if w < min_w or h < min_h:
            return False
    return True


# ══════════════════════════════════════════════════════════════════════════════
# Composite image segmentation  (orchestrator)
# ══════════════════════════════════════════════════════════════════════════════
def _segment_composite(img_bgr: np.ndarray):
    """
    Split a composite 12-lead ECG image into individual lead strips.

    Strategy:
    • B&W images  → _segment_grid() always  (auto_segment_leads is designed
      for coloured ECG paper and mis-detects column widths on B&W scans)
    • Colour images → try auto_segment_leads() first; fall back to _segment_grid()
      if it returns fewer than 10 leads or any crop is abnormally narrow.

    Returns:
        leads  : dict  {name: gray_ndarray}
        bounds : dict  {name: (x1, y1, x2, y2)}
    """
    H, W = img_bgr.shape[:2]

    if not _is_greyscale(img_bgr) and _HAS_AUTO_SEG:
        # Coloured image: try the full auto-segmenter
        try:
            segs   = _auto_seg(img_bgr)
            leads  = {k: v["gray"] for k, v in segs.items() if k in _LEAD_IDX}
            if (len(leads) >= 10 and
                    _validate_crops(leads, {}, W, H)):
                # Build approximate bounds from equal-division (auto_seg
                # doesn't return explicit pixel bounds)
                bounds = {}
                for name, gray in leads.items():
                    for r, row in enumerate(LAYOUT_4x3):
                        if name in row:
                            c = row.index(name)
                            x1 = c * (W // 4)
                            y1 = r * (H // 3)
                            bounds[name] = (x1, y1, x1 + gray.shape[1],
                                            y1 + gray.shape[0])
                return leads, bounds
            print("  [segment] auto_segment_leads returned narrow crops — "
                  "switching to grid")
        except Exception as exc:
            print(f"  [segment] auto_segment_leads error ({exc}) — "
                  f"using grid fallback")

    # B&W image  OR  colour fallback
    return _segment_grid(img_bgr)


# ══════════════════════════════════════════════════════════════════════════════
# Signal extraction  (probe-based L→R tracker)
# ══════════════════════════════════════════════════════════════════════════════
def _extract_signal(gray_lead: np.ndarray, smooth_win: int = 7) -> np.ndarray:
    H, W    = gray_lead.shape
    l_skip  = 0.15 if W > 800 else 0.13
    x_start = max(1, int(W * l_skip))
    crop    = gray_lead[:, x_start:].astype(float)
    cW      = crop.shape[1]

    background = float(np.percentile(crop, 95))
    v_lo = int(H * 0.10)
    v_hi = int(H * 0.90)
    max_jump = max(4, int(H * 0.35))

    def _step(x, center):
        lo  = max(0, int(center) - max_jump)
        hi  = min(H, int(center) + max_jump + 1)
        idx = int(np.argmin(crop[lo:hi, x]))
        return float(lo + idx), float(crop[lo + idx, x])

    n_probe  = max(10, int(cW * 0.10))
    probe_ys = []
    for px in range(min(n_probe, cW)):
        if float(crop[v_lo:v_hi, px].min()) < background * 0.75:
            probe_ys.append(v_lo + int(np.argmin(crop[v_lo:v_hi, px])))

    if len(probe_ys) >= 3:
        p_arr  = np.clip(np.array(probe_ys), v_lo, v_hi - 1)
        _cnt   = np.bincount(p_arr, minlength=H)
        _cnt[:v_lo] = 0
        _cnt[v_hi:] = 0
        start_y = float(np.argmax(_cnt))
    else:
        start_y = H / 2.0

    signal_y   = np.empty(cW)
    min_at_pos = np.empty(cW)
    prev = start_y
    for x in range(cW):
        signal_y[x], min_at_pos[x] = _step(x, prev)
        prev = signal_y[x]

    margin       = max(20.0, background * 0.12)
    no_trace     = min_at_pos > (background - margin)
    sig_nan      = signal_y.copy()
    sig_nan[no_trace] = np.nan
    valid = ~np.isnan(sig_nan)
    if valid.sum() >= 2:
        x_all    = np.arange(cW, dtype=float)
        signal_y = np.interp(x_all, x_all[valid], sig_nan[valid])
    else:
        signal_y = np.full(cW, H / 2.0)

    y_ints     = np.clip(np.round(signal_y).astype(int), v_lo, v_hi - 1)
    counts     = np.bincount(y_ints, minlength=H)
    counts[:v_lo] = 0
    counts[v_hi:] = 0
    baseline_y = float(np.argmax(counts))
    signal     = baseline_y - signal_y

    pad    = np.full(x_start, signal[0])
    signal = np.concatenate([pad, signal])

    win_bl = min(len(signal), 1000)
    if win_bl > 3:
        drift  = uniform_filter1d(signal.astype(float), size=win_bl, mode="nearest")
        signal = signal - drift

    med    = float(np.median(signal))
    std    = float(np.std(signal)) or 1.0
    signal = np.clip(signal, med - 3.5 * std, med + 3.5 * std)
    signal = medfilt(signal.astype(float), kernel_size=3)
    win = min(smooth_win, len(signal) if len(signal) % 2 == 1 else len(signal) - 1)
    if win >= 3 and len(signal) > win * 2:
        signal = savgol_filter(signal, window_length=win, polyorder=2)

    return signal


# ══════════════════════════════════════════════════════════════════════════════
# R-peak detector
# ══════════════════════════════════════════════════════════════════════════════
def _detect_rpeaks(sig_mv: np.ndarray, fs: float) -> np.ndarray:
    nyq = fs / 2.0
    try:
        b, a     = butter(3, 5.0 / nyq, btype="high")
        filtered = filtfilt(b, a, sig_mv)
    except Exception:
        filtered = sig_mv.copy()

    n        = len(filtered)
    mid      = filtered[int(0.12 * n): int(0.88 * n)]
    pos_thr  = max(0.05, float(np.percentile(mid, 85))  * 0.30)
    neg_thr  = max(0.05, abs(float(np.percentile(mid, 15))) * 0.30)
    min_dist = max(1, int(0.43 * fs))
    lead_in  = int(0.20 * n)

    pos_p, _ = find_peaks( filtered, height=pos_thr, distance=min_dist)
    neg_p, _ = find_peaks(-filtered, height=neg_thr, distance=min_dist)
    pos_p = pos_p[pos_p >= lead_in]
    neg_p = neg_p[neg_p >= lead_in]

    if not len(pos_p) and not len(neg_p):
        return pos_p
    if not len(pos_p):
        peaks = neg_p
    elif not len(neg_p):
        peaks = pos_p
    else:
        pm    = float(np.mean(filtered[pos_p]))
        nm    = float(np.mean(-filtered[neg_p]))
        peaks = pos_p if pm >= nm else neg_p

    # Short-RR filter (T-wave suppression)
    if len(peaks) >= 3:
        rr  = np.diff(peaks)
        med = float(np.median(rr))
        keep = np.ones(len(peaks), dtype=bool)
        for i in range(len(rr)):
            if rr[i] < 0.65 * med:
                keep[i + 1] = False
        peaks = peaks[keep]

    # Even-index check (interleaved T-wave between every QRS pair)
    if len(peaks) >= 3:
        rr     = np.diff(peaks)
        hr_est = 60.0 / (float(np.mean(rr)) / fs)
        if hr_est > 90:
            alt = peaks[::2]
            if len(alt) >= 2:
                alt_hr = 60.0 / (float(np.mean(np.diff(alt))) / fs)
                if 40 <= alt_hr <= 90 and alt_hr < hr_est * 0.65:
                    peaks = alt

    return peaks


# ══════════════════════════════════════════════════════════════════════════════
# PR interval measurement
# ══════════════════════════════════════════════════════════════════════════════
def _measure_pr_interval(sig_mv: np.ndarray, peaks: np.ndarray, fs: float) -> float:
    """
    Estimate the mean PR interval (ms) across all beats.

    Strategy:
      For each R-peak, search the window 300 ms → 50 ms BEFORE the peak for
      a positive local maximum (the P-wave peak).  Then walk backward from
      that peak until the signal drops below 20% of its amplitude — that is
      the P-wave onset.  PR interval = P-wave onset → R-peak.

    Returns NaN if no reliable P-waves are found.
    """
    if len(peaks) < 2:
        return float("nan")

    p_search_start = max(1, int(0.05 * fs))   # 50 ms before R-peak
    p_search_end   = max(1, int(0.30 * fs))   # 300 ms before R-peak

    pr_vals = []
    for pk in peaks:
        lo = pk - p_search_end
        hi = pk - p_search_start
        if lo < 0 or hi <= lo + 3:
            continue
        window = sig_mv[lo:hi]
        p_idx  = int(np.argmax(window))
        p_amp  = float(window[p_idx])
        if p_amp < 0.02:          # no visible P-wave
            continue
        # Walk back from P-peak to find onset (signal < 20% of P-amplitude)
        onset_idx = p_idx
        for j in range(p_idx, -1, -1):
            if window[j] < p_amp * 0.20:
                onset_idx = j
                break
        pr_samples = pk - (lo + onset_idx)
        pr_ms = (pr_samples / fs) * 1000.0
        if 80.0 <= pr_ms <= 400.0:   # physiologically plausible range
            pr_vals.append(pr_ms)

    return round(float(np.mean(pr_vals)), 1) if pr_vals else float("nan")


# ══════════════════════════════════════════════════════════════════════════════
# T-wave analysis
# ══════════════════════════════════════════════════════════════════════════════
def _analyze_twave(sig_mv: np.ndarray, peaks: np.ndarray, fs: float,
                   lead_name: str = "") -> dict:
    """
    Measure T-wave amplitude and detect hyperacute T waves.

    The T-wave window starts 80 ms after the detected QRS peak (whether that
    peak is an R-wave or the S-wave nadir in a QS lead).  The maximum positive
    value in the 80–400 ms window is taken as the T-wave amplitude.

    Hyperacute criteria (upright, disproportionately tall T waves):
      • Anterior leads (V1–V4) : T > 0.40 mV
      • All other leads         : T > 0.50 mV AND T/QRS-magnitude > 0.75
    """
    if len(peaks) < 1:
        return {"t_peak_mv": 0.0, "hyperacute": False, "t_r_ratio": 0.0}

    qrs_end  = max(1, int(0.08 * fs))   # 80 ms after QRS peak
    t_search = max(2, int(0.40 * fs))   # search window ends at 400 ms

    t_peaks  = []
    qrs_amps = []
    for pk in peaks:
        t_lo = min(pk + qrs_end, len(sig_mv) - 1)
        t_hi = min(pk + t_search, len(sig_mv))
        if t_hi <= t_lo + 5:
            continue
        t_peaks.append(float(sig_mv[t_lo:t_hi].max()))
        qrs_amps.append(max(abs(float(sig_mv[pk])), 0.01))

    if not t_peaks:
        return {"t_peak_mv": 0.0, "hyperacute": False, "t_r_ratio": 0.0}

    mean_t    = float(np.mean(t_peaks))
    mean_qrs  = float(np.mean(qrs_amps))
    t_r_ratio = mean_t / mean_qrs

    is_anterior = lead_name in ("V1", "V2", "V3", "V4")
    hyperacute  = (
        (is_anterior and mean_t > 0.40) or
        (not is_anterior and mean_t > 0.50 and t_r_ratio > 0.75)
    )
    return {
        "t_peak_mv": round(mean_t, 3),
        "hyperacute": hyperacute,
        "t_r_ratio":  round(t_r_ratio, 3),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Diagnosis post-processing
# ══════════════════════════════════════════════════════════════════════════════
def _augment_diagnosis(diag: dict, results: list) -> dict:
    """
    Extend ECGDiagnoser output with findings it misses:
      1. Lateral ST elevation at 0.08 mV threshold (vs engine's 0.10 mV)
      2. Anteroseptal/Anterior STEMI pattern synthesis
      3. Hyperacute T-wave findings from per-lead twave analysis
      4. QRS duration fix (compute mean of per-lead values when header is 0)
      5. P-wave status from per-lead p_present counts
    """
    if not isinstance(diag, dict):
        return diag

    findings = list(diag.get("findings", []))
    res_dict = {r["lead"]: r for r in results}

    # 1. Fix QRS duration header (ECGDiagnoser often returns 0 aggregate)
    if diag.get("qrs_duration_ms", 0) < 5:
        qrs_vals = [r["qrs_dur_ms"] for r in results if r.get("qrs_dur_ms", 0) > 5]
        if qrs_vals:
            diag["qrs_duration_ms"] = round(float(np.mean(qrs_vals)), 1)

    # 1b. Aggregate mean PR interval across all leads
    pr_vals = [
        r["pr_interval_ms"] for r in results
        if not math.isnan(r.get("pr_interval_ms", float("nan")))
    ]
    if pr_vals:
        diag["pr_interval_ms"] = round(float(np.mean(pr_vals)), 1)

    # 2. Fix P-wave status (ECGDiagnoser often returns 'unknown')
    if diag.get("p_wave_status", "unknown") == "unknown":
        n_p = sum(1 for r in results if r.get("p_present", False))
        if n_p >= 6:
            diag["p_wave_status"] = "present"
        elif n_p <= 2:
            diag["p_wave_status"] = "absent or disorganised"
        else:
            diag["p_wave_status"] = f"inconsistent ({n_p}/12 leads)"

    # 3. Lateral ST elevation at lower threshold (I, aVL, V5, V6)
    LATERAL = ["I", "aVL", "V5", "V6"]
    existing_text = " ".join(findings)
    lat_elevated  = [ln for ln in LATERAL
                     if res_dict.get(ln, {}).get("st_dev_mv", 0) >= 0.08]
    new_lat = [ln for ln in lat_elevated if ln not in existing_text]
    if new_lat:
        findings.append(
            f"ST elevation in lateral leads ({', '.join(new_lat)}) "
            f"— consider STEMI / acute ischaemia"
        )

    # 4. Anteroseptal/Anterior STEMI synthesis
    ant_leads = [ln for ln in ["V1", "V2", "V3", "V4"]
                 if res_dict.get(ln, {}).get("st_dev_mv", 0) >= 0.10]
    lat_leads = [ln for ln in ["I", "aVL"]
                 if res_dict.get(ln, {}).get("st_dev_mv", 0) >= 0.08]
    recip_inf = any(res_dict.get(ln, {}).get("st_dev_mv", 0) <= -0.08
                    for ln in ["III", "aVF"])

    if ant_leads:
        qualifier = ("Anteroseptal" if any(ln in ant_leads for ln in ["V1","V2"])
                     else "Anterior")
        all_elev  = ant_leads + [ln for ln in lat_leads if ln not in ant_leads]
        msg = (f"{qualifier} STEMI pattern: ST elevation in "
               f"{', '.join(all_elev)}" +
               (" with reciprocal inferior ST depression" if recip_inf else ""))
        if not any("STEMI pattern" in f for f in findings):
            findings.insert(0, msg)

    # 5. Hyperacute T waves
    hyper_leads = [r["lead"] for r in results
                   if r.get("twave", {}).get("hyperacute", False)]
    if hyper_leads:
        findings.append(
            f"Hyperacute T waves in {', '.join(hyper_leads)} "
            f"— earliest STEMI sign (tall upright T waves)"
        )

    diag["findings"] = findings
    return diag


# ══════════════════════════════════════════════════════════════════════════════
# Full per-lead analysis (needed for ECGDiagnoser)
# ══════════════════════════════════════════════════════════════════════════════
def _analyze_lead(sig_px: np.ndarray, name: str,
                  px_per_mm: float, paper_speed: float = 25.0,
                  paper_gain: float = 10.0) -> dict:
    """
    Convert pixel signal → mV, detect R-peaks, and compute clinical metrics.
    Returns the full result dict expected by ECGDiagnoser.
    """
    fs     = px_per_mm * paper_speed
    if _HAS_SEG:
        sig_mv = px_to_mv(sig_px, px_per_mm, paper_gain)
        t_s    = time_axis(len(sig_px), px_per_mm, paper_speed)
    else:
        sig_mv = sig_px / (paper_gain * px_per_mm)
        t_s    = np.arange(len(sig_px)) / fs

    peaks  = _detect_rpeaks(sig_mv, fs)
    n_beats = len(peaks)
    hr = (round(60.0 / float(np.mean(np.diff(peaks)) / fs), 1)
          if n_beats >= 2 else float("nan"))
    amp = float(sig_mv.max() - sig_mv.min())

    if _HAS_SEG:
        qrs_dur   = _estimate_qrs_duration(sig_mv, peaks, fs)
        st_dev    = _measure_st_deviation(sig_mv, peaks, fs)
        p_present = _detect_p_waves(sig_mv, peaks, fs)
        net_qrs   = _net_qrs_deflection(sig_mv, peaks, fs)
    else:
        qrs_dur, st_dev, p_present, net_qrs = 0.0, 0.0, False, 0.0

    twave  = _analyze_twave(sig_mv, peaks, fs, lead_name=name)
    pr_ms  = _measure_pr_interval(sig_mv, peaks, fs)

    # Resample to SIGNAL_LENGTH for the classifier
    sig_ml = sig_mv.astype(np.float32)
    if len(sig_ml) != SIGNAL_LENGTH:
        sig_ml = resample(sig_ml, SIGNAL_LENGTH).astype(np.float32)

    return {
        "lead":           name,
        "signal_mv":      sig_mv,
        "signal_ml":      sig_ml,       # resampled for ML
        "time_s":         t_s,
        "r_peaks":        peaks,
        "hr_bpm":         hr,
        "amplitude_mv":   round(amp, 3),
        "duration_s":     round(float(t_s[-1]), 2) if len(t_s) > 1 else 0.0,
        "fs_hz":          round(fs, 1),
        "n_beats":        n_beats,
        "qrs_dur_ms":     qrs_dur,
        "st_dev_mv":      st_dev,
        "p_present":      p_present,
        "net_qrs_mv":     net_qrs,
        "twave":          twave,
        "pr_interval_ms": pr_ms,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Output helpers
# ══════════════════════════════════════════════════════════════════════════════
def _save_seg_preview(img_bgr: np.ndarray, bounds: dict, out_dir: str):
    """Annotate and save segmentation preview using actual crop boundaries."""
    preview = (img_bgr.copy() if img_bgr.ndim == 3
               else cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR))

    for name, (x1, y1, x2, y2) in bounds.items():
        cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 200, 80), 2)
        # Label background chip
        lbl_w = max(60, len(name) * 12 + 10)
        cv2.rectangle(preview, (x1, y1), (x1 + lbl_w, y1 + 26), (0, 200, 80), -1)
        cv2.putText(preview, name, (x1 + 5, y1 + 19),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    out = os.path.join(out_dir, "segmentation_preview.png")
    cv2.imwrite(out, preview)
    print(f"  Preview  -> {out}")


def _save_signal_plot(results: list, out_dir: str):
    """Dark-theme 12-lead signal overview with R-peak markers."""
    fig, axes = plt.subplots(12, 1, figsize=(16, 22), facecolor="#0d1b2a")
    fig.suptitle("12-Lead ECG — Digitised Signals", color="white",
                 fontsize=13, y=0.996)

    res_by_lead = {r["lead"]: r for r in results}

    for i, name in enumerate(LEAD_NAMES):
        ax  = axes[i]
        ax.set_facecolor("#111827")
        res = res_by_lead.get(name)

        if res is not None:
            t   = res["time_s"]
            sig = res["signal_mv"]
            ax.plot(t, sig, color="#00e5ff", linewidth=0.85, zorder=2)
            ax.axhline(0, color="#444", linewidth=0.5, linestyle="--", zorder=1)

            # Mark R-peaks
            pks = res["r_peaks"]
            if len(pks) > 0 and len(pks) < len(t):
                ax.scatter(t[pks], sig[pks], color="#ff6b35", s=18, zorder=3,
                           marker="v")

            hr_s = (f"{res['hr_bpm']:.0f} bpm"
                    if not math.isnan(res["hr_bpm"]) else "n/a")
            st_s = (f"ST {res['st_dev_mv']:+.2f} mV" if _HAS_SEG else "")
            ax.set_title(f"{name}   HR {hr_s}   {st_s}",
                         color="#ccc", fontsize=7.5, loc="left", pad=2)
        else:
            ax.text(0.5, 0.5, "missing", transform=ax.transAxes,
                    color="#555", ha="center", va="center")

        ax.set_ylabel("mV", color="#555", fontsize=7, rotation=0, labelpad=26)
        ax.tick_params(colors="#555", labelsize=6)
        for sp in ax.spines.values():
            sp.set_color("#2a2a2a")
        if i < 11:
            ax.set_xticks([])

    axes[-1].set_xlabel("Time (s)", color="#aaa", fontsize=8)
    plt.tight_layout(rect=[0, 0, 1, 0.996])
    out = os.path.join(out_dir, "signals_12lead.png")
    plt.savefig(out, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Signals  -> {out}")


def _save_result_card(ml_result: dict, diag: dict, results: list, out_dir: str):
    """Summary card: ML probabilities + key clinical findings."""
    probs = ml_result.get("probabilities", {})

    hr_vals = [r["hr_bpm"] for r in results
               if not math.isnan(r.get("hr_bpm", float("nan")))]
    mean_hr = round(float(np.mean(hr_vals)), 1) if hr_vals else float("nan")

    fig = plt.figure(figsize=(13, 5), facecolor="#0d1b2a")
    gs  = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.4, 0.9], wspace=0.35)

    # ── ML probability bars ──────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor("#111827")
    for sp in ax1.spines.values():
        sp.set_color("#333")
    ax1.tick_params(colors="#aaa", labelsize=9)

    if probs:
        names  = list(probs.keys())
        vals   = [probs[n] for n in names]
        pred   = ml_result.get("label", "")
        colours = ["#ef5350" if n == pred else "#42a5f5" for n in names]
        ax1.barh(names, vals, color=colours, alpha=0.9)
        ax1.set_xlim(0, 1)
        ax1.set_xlabel("Probability", color="#aaa", fontsize=9)
        ax1.set_title("ML Classification", color="white", fontsize=11)
        for j, v in enumerate(vals):
            ax1.text(v + 0.01, j, f"{v*100:.1f}%",
                     va="center", color="white", fontsize=9)

    # ── Clinical findings ────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    ax2.set_facecolor("#111827")
    for sp in ax2.spines.values():
        sp.set_color("#333")
    ax2.axis("off")
    ax2.set_title("Clinical Analysis", color="white", fontsize=11)

    import textwrap
    findings_text = ""
    if diag and "findings" in diag:
        for f in diag["findings"][:8]:
            wrapped_lines = textwrap.wrap(f, width=42)
            if wrapped_lines:
                findings_text += f"  • {wrapped_lines[0]}\n"
                for wl in wrapped_lines[1:]:
                    findings_text += f"    {wl}\n"
    else:
        findings_text = "  (ECGDiagnoser not available)"

    severity = diag.get("severity", "UNKNOWN") if diag else "UNKNOWN"
    sev_colour = {"CRITICAL": "#ef5350", "HIGH": "#ff9800",
                  "MODERATE": "#ffc107", "LOW": "#66bb6a",
                  "NORMAL": "#42a5f5"}.get(severity, "#aaa")

    info = (
        f"Severity   : {severity}\n"
        f"Heart Rate : {mean_hr:.0f} bpm\n"
        f"Leads OK   : {len(hr_vals)}/12\n\n"
        f"ML Result  : {ml_result.get('label','?')}\n"
        f"Confidence : {ml_result.get('confidence',0)*100:.1f}%\n"
        f"\nFindings:\n{findings_text}"
    )
    ax2.text(0.05, 0.95, info, transform=ax2.transAxes,
             color="white", fontsize=9, va="top",
             fontfamily="monospace")
    ax2.add_patch(plt.Rectangle((0.0, 0.0), 1.0, 0.05,
                                  transform=ax2.transAxes,
                                  color=sev_colour, alpha=0.6, zorder=0))

    # ── Per-lead HR table ────────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[2])
    ax3.set_facecolor("#111827")
    for sp in ax3.spines.values():
        sp.set_color("#333")
    ax3.axis("off")
    ax3.set_title("Per-Lead HR / ST", color="white", fontsize=11)

    res_dict = {r["lead"]: r for r in results}
    rows = []
    for name in LEAD_NAMES:
        r = res_dict.get(name)
        if r:
            hr_s = f"{r['hr_bpm']:.0f}" if not math.isnan(r["hr_bpm"]) else "n/a"
            st_s = f"{r['st_dev_mv']:+.2f}" if _HAS_SEG else "-"
        else:
            hr_s, st_s = "-", "-"
        rows.append([name, hr_s, st_s])

    tbl = ax3.table(
        cellText=rows,
        colLabels=["Lead", "HR (bpm)", "ST (mV)"],
        loc="center", cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    for (row, col), cell in tbl.get_celld().items():
        cell.set_facecolor("#1e293b" if row % 2 == 0 else "#111827")
        cell.set_text_props(color="white")
        cell.set_edgecolor("#333")

    plt.tight_layout()
    out = os.path.join(out_dir, "result_card.png")
    plt.savefig(out, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Card     -> {out}")


def _generate_impression(diag: dict, ml_result: dict) -> list:
    """
    Generate a structured clinical impression (1-3 sentence summary).
    Maps rhythm + severity + findings + ML label → a cardiologist-style statement.
    """
    sentences = []

    if not diag:
        label = ml_result.get("label", "?")
        conf  = ml_result.get("confidence", 0) * 100
        sentences.append(f"ML classifier predicts {label} ({conf:.1f}% confidence).")
        sentences.append("Clinical correlation required.")
        return sentences

    severity = diag.get("severity", "UNKNOWN")
    rhythm   = diag.get("rhythm", "unknown")
    p_status = diag.get("p_wave_status", "unknown")
    axis     = diag.get("axis", "unknown")
    ml_label = ml_result.get("label", "?")
    ml_conf  = ml_result.get("confidence", 0) * 100
    findings = diag.get("findings", [])
    f_text   = " ".join(findings).lower()

    # --- Sentence 1: Rhythm ---
    if "regular" in rhythm and "present" in p_status:
        sentences.append("Normal sinus rhythm.")
    elif "regular" in rhythm:
        sentences.append("Regular rhythm.")
    else:
        sentences.append("Irregular rhythm — sinus arrhythmia or ectopic activity suspected.")

    # --- Sentence 2: Primary finding ---
    stemi_findings = [f for f in findings if "stemi" in f.lower()]
    if stemi_findings:
        if any(t in f_text for t in ["anteroseptal", "anterior"]):
            territory = "Anteroseptal"
        elif "inferior" in f_text:
            territory = "Inferior"
        elif "lateral" in f_text:
            territory = "Lateral"
        else:
            territory = "Acute"
        hyper_str = (
            " with hyperacute T-waves"
            if any("hyperacute" in f.lower() for f in findings)
            else ""
        )
        sentences.append(
            f"{territory} STEMI pattern{hyper_str} identified. "
            f"Urgent cardiology evaluation recommended."
        )
    elif severity in ("CRITICAL", "HIGH"):
        sentences.append(
            f"Significant ST-T changes (severity: {severity}). "
            f"ML: {ml_label} ({ml_conf:.1f}%). Cardiology review required."
        )
    elif ml_label == "NORM" and severity in ("LOW", "NORMAL", "UNKNOWN"):
        sentences.append("No acute ischemic changes identified.")
    elif ml_label == "STTC":
        sentences.append(
            f"Non-specific ST-T wave changes (ML: STTC, {ml_conf:.1f}%). "
            f"Clinical correlation advised."
        )
    elif ml_label == "CD":
        sentences.append(
            f"Conduction disturbance pattern (ML: CD, {ml_conf:.1f}%). "
            f"Further evaluation recommended."
        )
    elif ml_label == "HYP":
        sentences.append(
            f"Hypertrophy pattern suspected (ML: HYP, {ml_conf:.1f}%). "
            f"Clinical correlation advised."
        )
    else:
        sentences.append(
            f"Automated analysis: {ml_label} ({ml_conf:.1f}%). "
            f"Clinical correlation required."
        )

    # --- Sentence 3: Axis note (if abnormal) ---
    if axis and "normal" not in axis.lower() and "unknown" not in axis.lower():
        sentences.append(f"{axis} noted.")

    return sentences


def _save_text_report(ml_result: dict, diag: dict, results: list,
                      image_path: str, out_dir: str):
    """Write a plain-text clinical diagnosis report."""
    res_dict = {r["lead"]: r for r in results}
    hr_vals  = [r["hr_bpm"] for r in results
                if not math.isnan(r.get("hr_bpm", float("nan")))]
    mean_hr  = round(float(np.mean(hr_vals)), 1) if hr_vals else float("nan")

    lines = []
    lines.append("=" * 62)
    lines.append("  AUTOMATED 12-LEAD ECG INTERPRETATION REPORT")
    lines.append(f"  Image   : {Path(image_path).name}")
    lines.append(f"  Date    : {datetime.now().strftime('%Y-%m-%d  %H:%M:%S')}")
    lines.append("=" * 62)
    lines.append("")

    if diag:
        lines.append(f"  Overall severity   : {diag.get('severity', 'UNKNOWN')}")
        lines.append(f"  Heart rate         : {mean_hr:.0f} bpm")
        lines.append(f"  Rhythm             : {diag.get('rhythm', 'unknown')}")
        lines.append(f"  P-waves            : {diag.get('p_wave_status', 'unknown')}")
        pr_ms = diag.get("pr_interval_ms", float("nan"))
        pr_str = f"{pr_ms:.0f} ms" if not math.isnan(pr_ms) else "not measurable"
        lines.append(f"  PR interval        : {pr_str}")
        lines.append(f"  QRS duration       : {diag.get('qrs_duration_ms', 0):.1f} ms")
        lines.append(f"  Electrical axis    : {diag.get('axis', 'unknown')}")
        lines.append(f"  QTc (estimated)    : {diag.get('qtc_ms', 0):.1f} ms")
        lines.append("")
        lines.append("  FINDINGS")
        lines.append("  " + "-" * 50)
        for finding in diag.get("findings", []):
            lines.append(f"    * {finding}")
        lines.append("")
        lines.append("  IMPRESSION")
        lines.append("  " + "-" * 50)
        for sentence in _generate_impression(diag, ml_result):
            lines.append(f"    {sentence}")
    else:
        lines.append(f"  Mean Heart Rate    : {mean_hr:.0f} bpm")
        lines.append("  (Full clinical analysis unavailable — ECGDiagnoser not loaded)")
        lines.append("")
        lines.append("  IMPRESSION")
        lines.append("  " + "-" * 50)
        for sentence in _generate_impression(None, ml_result):
            lines.append(f"    {sentence}")

    lines.append("")
    lines.append("  ML CLASSIFICATION  (ECGResNet1D trained on PTB-XL)")
    lines.append("  " + "-" * 50)
    lines.append(f"    Prediction : {ml_result.get('label', '?')}")
    lines.append(f"    Confidence : {ml_result.get('confidence', 0)*100:.1f}%")
    if "probabilities" in ml_result:
        for cls, p in ml_result["probabilities"].items():
            bar  = "#" * int(p * 28)
            mark = " <--" if cls == ml_result.get("label") else ""
            lines.append(f"    {cls:<6} {p*100:5.1f}%  {bar}{mark}")

    lines.append("")
    lines.append("  PER-LEAD SUMMARY")
    lines.append("  " + "-" * 50)
    lines.append(f"  {'Lead':<6}  {'HR (bpm)':<10}  {'ST (mV)':<9}  "
                 f"{'QRS (ms)':<9}  {'Amp (mV)':<9}  {'T-pk (mV)'}")
    for name in LEAD_NAMES:
        r = res_dict.get(name)
        if r:
            hr_s  = f"{r['hr_bpm']:>6.0f}" if not math.isnan(r["hr_bpm"]) else "   n/a"
            st_s  = f"{r['st_dev_mv']:+.3f}" if _HAS_SEG else "  -  "
            qrs_s = f"{r['qrs_dur_ms']:>6.1f}" if _HAS_SEG else "  -  "
            amp_s = f"{r['amplitude_mv']:.3f}"
            tw    = r.get("twave", {})
            tp_s  = f"{tw.get('t_peak_mv', 0):+.3f}"
            if tw.get("hyperacute", False):
                tp_s += " [H]"
        else:
            hr_s, st_s, qrs_s, amp_s, tp_s = "   n/a", "  -  ", "  -  ", "  -  ", "  -  "
        lines.append(f"  {name:<6}  {hr_s:<10}  {st_s:<9}  {qrs_s:<9}  {amp_s:<9}  {tp_s}")

    lines.append("")
    lines.append("  [H] = Hyperacute T wave detected (tall/broad upright T)")
    lines.append("  NOTE: Automated screen only.  Clinical correlation and")
    lines.append("        expert review required before any clinical decision.")
    lines.append("=" * 62)

    out = os.path.join(out_dir, "diagnosis_report.txt")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Report   -> {out}")


# ══════════════════════════════════════════════════════════════════════════════
# Main pipeline
# ══════════════════════════════════════════════════════════════════════════════
def run_pipeline(image_path:  str,
                 model_path:  str   = None,
                 output_dir:  str   = None,
                 paper_speed: float = 25.0,
                 paper_gain:  float = 10.0) -> dict:

    image_path = str(Path(image_path).resolve())
    if model_path is None:
        model_path = str(_DEFAULT_MODEL)
    if not os.path.exists(model_path):
        sys.exit(f"Model checkpoint not found: {model_path}")

    stem = Path(image_path).stem
    if output_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = str(_OUTPUT_ROOT / f"{stem}_{ts}")
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*64}")
    print(f"  12-Lead Composite ECG Pipeline")
    print(f"  Image  : {image_path}")
    print(f"  Model  : {model_path}")
    print(f"  Output : {output_dir}")
    print(f"{'='*64}\n")

    # ── Load image ─────────────────────────────────────────────────────────────
    img = cv2.imread(image_path)
    if img is None:
        sys.exit(f"Cannot read image: {image_path}")
    img_type = "B&W" if _is_greyscale(img) else "colour"
    print(f"  Loaded  {img.shape[1]}x{img.shape[0]} px  [{img_type}]")

    # ── Pixel scale ────────────────────────────────────────────────────────────
    if _HAS_SEG:
        try:
            px_per_mm = estimate_px_per_mm(img)
        except Exception:
            px_per_mm = img.shape[1] / (paper_speed * 4 * 10)
    else:
        px_per_mm = img.shape[1] / (paper_speed * 4 * 10)
    fs = px_per_mm * paper_speed
    print(f"  Scale   {px_per_mm:.2f} px/mm  ->  fs {fs:.1f} Hz\n")

    # ── [1] Segment ────────────────────────────────────────────────────────────
    print("[1] Segmenting composite image ...")
    leads, bounds = _segment_composite(img)
    print(f"    Found : {', '.join(sorted(leads.keys()))}")
    _save_seg_preview(img, bounds, output_dir)

    crops_dir = os.path.join(output_dir, "lead_crops")
    os.makedirs(crops_dir, exist_ok=True)
    for name, gray in leads.items():
        cv2.imwrite(os.path.join(crops_dir, f"{name}.png"), gray)

    # ── [2] Digitise & analyse each lead ──────────────────────────────────────
    print("\n[2] Digitising and analysing signals ...")
    results = []
    signal_12lead_mv = np.zeros((12, SIGNAL_LENGTH), dtype=np.float32)

    for lead_name in LEAD_NAMES:
        gray = leads.get(lead_name)
        if gray is None:
            print(f"    {lead_name:<5}  missing — zero-padded")
            continue
        try:
            sig_px = _extract_signal(gray)
            res    = _analyze_lead(sig_px, lead_name, px_per_mm,
                                   paper_speed, paper_gain)
            results.append(res)
            signal_12lead_mv[_LEAD_IDX[lead_name]] = res["signal_ml"]

            hr_s = (f"{res['hr_bpm']:.0f} bpm"
                    if not math.isnan(res["hr_bpm"]) else "n/a")
            st_s = (f"ST {res['st_dev_mv']:+.3f} mV" if _HAS_SEG else "")
            print(f"    {lead_name:<5}  HR {hr_s:<10}  {st_s}")
        except Exception as exc:
            print(f"    {lead_name:<5}  ERROR: {exc}  (zero-padded)")

    missing = [n for n in LEAD_NAMES if n not in leads]
    if missing:
        print(f"\n    WARNING: leads not found -> zero-padded: {missing}")

    # ── [3] Signal plot ────────────────────────────────────────────────────────
    print()
    _save_signal_plot(results, output_dir)

    # ── [4] Clinical diagnosis ────────────────────────────────────────────────
    print("\n[4] Running clinical diagnosis ...")
    diag = None
    if _HAS_SEG and results:
        try:
            diag = ECGDiagnoser().diagnose(results)
            diag = _augment_diagnosis(diag, results)    # Fix thresholds + add T-wave
            # Print key findings
            print(f"    Severity   : {diag.get('severity', '?')}")
            print(f"    Rhythm     : {diag.get('rhythm', '?')}")
            print(f"    Axis       : {diag.get('axis', '?')}")
            print(f"    Findings:")
            for f in diag.get("findings", []):
                print(f"      * {f}")
            # Save the built-in LITFL report if available
            try:
                save_diagnosis_report(
                    diag,
                    Path(image_path).name,
                    os.path.join(output_dir, "diagnosis_litfl.txt"),
                )
            except Exception:
                pass
        except Exception as exc:
            print(f"    Diagnosis engine error: {exc}")

    # ── [5] ML classification ─────────────────────────────────────────────────
    print("\n[5] Classifying ...")
    try:
        model = _load_model_robust(model_path)
        lead_signals = {r["lead"]: r["signal_ml"] for r in results}
        if _ML_AVAIL:
            ml_result = _ML_CLASSIFY(lead_signals, model)
        else:
            from stage4_classify import predict as _predict
            ml_result = _predict(model, signal_12lead_mv)
        label = ml_result["label"]
        conf  = ml_result["confidence"] * 100
        print(f"\n  {'-'*44}")
        print(f"  ML Result  : {label}  ({conf:.1f}%)")
        print(f"  {'-'*44}")
        for cls, p in ml_result["probabilities"].items():
            bar  = "#" * int(p * 30)
            mark = " <--" if cls == label else ""
            print(f"  {cls:<6} {p*100:5.1f}%  {bar}{mark}")
        print(f"  {'-'*44}\n")
    except Exception as exc:
        print(f"  Classification failed: {exc}")
        ml_result = {"error": str(exc), "label": "ERROR",
                     "confidence": 0.0, "probabilities": {}}

    # ── [6] Save outputs ───────────────────────────────────────────────────────
    _save_result_card(ml_result, diag or {}, results, output_dir)
    _save_text_report(ml_result, diag or {}, results, image_path, output_dir)

    csv_path = os.path.join(output_dir, "signal_12lead.csv")
    np.savetxt(csv_path, signal_12lead_mv.T, delimiter=",",
               header=",".join(LEAD_NAMES), comments="")

    report = {
        "timestamp":    datetime.now().isoformat(),
        "image_path":   image_path,
        "model_path":   model_path,
        "scale_px_mm":  round(px_per_mm, 3),
        "fs_hz":        round(fs, 1),
        "image_type":   img_type,
        "leads": {
            r["lead"]: {
                "hr_bpm":      r["hr_bpm"],
                "amp_mv":      r["amplitude_mv"],
                "st_dev_mv":   r["st_dev_mv"],
                "qrs_dur_ms":  r["qrs_dur_ms"],
                "p_present":   r["p_present"],
            } for r in results
        },
        "diagnosis":    diag,
        "classification": ml_result,
    }
    report_path = os.path.join(output_dir, "report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"  CSV      -> {csv_path}")
    print(f"  JSON     -> {report_path}")
    print(f"\n{'='*64}")
    print(f"  All outputs -> {output_dir}")
    print(f"{'='*64}\n")
    return report


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="12-Lead Composite ECG Pipeline")
    parser.add_argument("--img",        default=None,
                        help="Composite ECG image path (omit to open file browser)")
    parser.add_argument("--model",      default=None,
                        help=f"Model checkpoint (default: {_DEFAULT_MODEL})")
    parser.add_argument("--output_dir", default=None,
                        help=f"Output dir (default: {_OUTPUT_ROOT}/<stem>_<ts>/)")
    parser.add_argument("--paper_speed", type=float, default=25.0,
                        help="Paper speed mm/s (default: 25)")
    parser.add_argument("--paper_gain",  type=float, default=10.0,
                        help="Gain mm/mV (default: 10)")
    args = parser.parse_args()

    img_path = args.img if args.img else _browse_for_image()
    run_pipeline(img_path, args.model, args.output_dir,
                 args.paper_speed, args.paper_gain)
