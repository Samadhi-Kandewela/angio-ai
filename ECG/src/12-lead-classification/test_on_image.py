r"""
test_on_image.py
----------------
Standalone test: run the 12-lead ResNet classifier on a scanned ECG image.

Steps
-----
1. Segment the image into 12 individual lead crops (reusing ecg_segment.py).
2. Extract the digitised signal from each lead crop using an IMPROVED local
   digitizer (_improved_extract_signal) that fixes tracking of inverted,
   small-amplitude, and tall-amplitude leads.
3. Stack the 12 signals -> (12, 1000) tensor.
4. Run ECGResNet1D and print the classification result.

Digitizer improvements vs ecg_segment.extract_signal
-----------------------------------------------------
* Search zone widened to 10-90% of strip height (was 20-80%)
  so tall R-waves (V4) and deep S-waves (III, aVF) stay inside the window.
* max_jump raised to H*0.35 (was H*0.20) so the tracker can return from
  a deep S-wave trough or a tall R-peak to the isoelectric baseline.
* Probe-based starting y: scans the first 10% of columns unconstrained to
  find the mode baseline, replacing the single-column bidirectional anchor
  which selected S-wave nadirs and got permanently stuck there.
* Narrow-lead label skip raised to 13% (was 6%) to clear calibration boxes.
* R-peak thresholds computed from the middle 80% of the signal so the
  calibration-pulse transient at the start of column-1 leads (I, II, III)
  does not inflate the threshold above the true QRS amplitude.
* min_dist for R-peak detection set to 0.43*fs (was 0.35*fs): the 5 Hz
  Butterworth HPF ringing creates a second lobe at ~2/5 = 0.40 s; raising
  min_dist to 0.43 s keeps that lobe below the exclusion window and prevents
  double-counting in inverted leads (III, aVF).

Usage
-----
    cd D:/UNI/FYP/Pipeline/ECG
    venv/Scripts/activate
    python src/12-lead-classification/test_on_image.py <image.png> --model <checkpoint.pth>

Example
-------
    python src/12-lead-classification/test_on_image.py
        ecg-image-kit/sample-data/ecg-images/ecg00001.png
        --model ecg-12-lead-model/run_20260603_141028/ecg_resnet1d_best.pth
"""

import argparse
import sys
import math
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
_HERE      = Path(__file__).resolve().parent                          # .../ECG/src/12-lead-classification
_SEGMENT   = _HERE.parent.parent.parent / "angio-ai" / "src"         # .../angio-ai/src
_ECG_ROOT  = _HERE.parent.parent                                      # .../ECG

sys.path.insert(0, str(_HERE))       # stage4_classify (12-lead version)
sys.path.insert(0, str(_SEGMENT))    # ecg_segment  (segmentation + digitizer)

# ── Imports ───────────────────────────────────────────────────────────────────
import cv2
import numpy as np
from scipy.signal import find_peaks, savgol_filter, medfilt, butter, filtfilt

from ecg_segment import (
    auto_segment_leads,
    analyze_lead,
    estimate_px_per_mm,
    save_lead_images,
    save_segmentation_preview,
    save_signal_plots,
    save_diagnosis_report,
    ECGDiagnoser,
    px_to_mv,
    time_axis,
    _estimate_qrs_duration,
    _measure_st_deviation,
    _detect_p_waves,
    _net_qrs_deflection,
)
from stage4_classify import load_model, classify, CLASS_NAMES, LEAD_ORDER

LEAD_12 = set(LEAD_ORDER)   # {"I","II","III","aVR","aVL","aVF","V1"..."V6"}


# =============================================================================
# Improved local digitizer
# =============================================================================

def _improved_extract_signal(gray_lead: np.ndarray,
                              label_skip_frac: float = 0.06,
                              smooth_win: int = 7) -> np.ndarray:
    """
    Probe-based L-R trace tracker with wider search bounds.

    Key differences from ecg_segment.extract_signal:
    - v_lo/v_hi: 10/90% (was 20/80%) — keeps tall R-waves and deep S-waves in range
    - max_jump: H*0.35 (was H*0.20) — tracker can escape S-wave troughs
    - Starting y estimated from probe (mode of first 10% unconstrained) instead
      of a single bidirectional anchor column that selects S-wave nadirs
    - label_skip: 13% for narrow leads (was 6%) — clears calibration boxes
    """
    from scipy.ndimage import uniform_filter1d

    H, W = gray_lead.shape

    # ── 1. Label / calibration-box skip ───────────────────────────
    if W > 800:
        label_skip_frac = max(label_skip_frac, 0.15)
    else:
        # Calibration box spans ~10-12% of a narrow lead; 13% gives margin.
        # Note: any residual calibration-box artifact peak is handled by the
        # lead-in exclusion in _improved_detect_rpeaks (first 20% discarded).
        label_skip_frac = max(label_skip_frac, 0.13)
    x_start = max(1, int(W * label_skip_frac))
    crop    = gray_lead[:, x_start:].astype(float)
    cW      = crop.shape[1]

    background = float(np.percentile(crop, 95))

    # ── 2. Constrained trace tracker ──────────────────────────────
    # Widened zone: [10%, 90%] of strip height instead of [20%, 80%].
    # This keeps leads whose trace reaches near the strip edges (V4 tall
    # R-wave, III/aVF deep S-wave) inside the search window.
    v_lo = int(H * 0.10)
    v_hi = int(H * 0.90)

    # Larger jump window so the tracker can return from a deep trough
    # (inverted leads: III, aVL, aVF) to the isoelectric baseline in a
    # single column step. The old H*0.20 caused it to get permanently stuck.
    max_jump = max(4, int(H * 0.35))

    def _step(x, center):
        lo  = max(0, int(center) - max_jump)
        hi  = min(H, int(center) + max_jump + 1)
        idx = int(np.argmin(crop[lo:hi, x]))
        return float(lo + idx), float(crop[lo + idx, x])

    # Probe the first 10% of columns (unconstrained within the search zone)
    # to find where the isoelectric baseline sits in this lead.
    # The mode of these probe y-values = best starting estimate.
    # This replaces the single-column bidirectional anchor which started at
    # an S-wave nadir and could never reach the baseline (gap > max_jump).
    n_probe  = max(10, int(cW * 0.10))
    probe_ys = []
    for px in range(min(n_probe, cW)):
        col_min = float(crop[v_lo:v_hi, px].min())
        if col_min < background * 0.75:          # only clear-trace columns
            probe_ys.append(v_lo + int(np.argmin(crop[v_lo:v_hi, px])))

    if len(probe_ys) >= 3:
        p_arr = np.clip(np.array(probe_ys), v_lo, v_hi - 1)
        _cnt  = np.bincount(p_arr, minlength=H)
        _cnt[:v_lo] = 0;  _cnt[v_hi:] = 0
        start_y = float(np.argmax(_cnt))
    else:
        start_y = H / 2.0

    signal_y   = np.empty(cW)
    min_at_pos = np.empty(cW)

    # Single L-R pass for all lead widths (no bifurcation by width).
    prev = start_y
    for x in range(cW):
        signal_y[x], min_at_pos[x] = _step(x, prev)
        prev = signal_y[x]

    # ── 3. No-trace columns -> NaN ────────────────────────────────
    margin       = max(20.0, background * 0.12)
    no_trace     = min_at_pos > (background - margin)
    signal_y_nan = signal_y.copy()
    signal_y_nan[no_trace] = np.nan

    # ── 4. Interpolate NaN gaps ───────────────────────────────────
    valid = ~np.isnan(signal_y_nan)
    if valid.sum() >= 2:
        x_all    = np.arange(cW, dtype=float)
        signal_y = np.interp(x_all, x_all[valid], signal_y_nan[valid])
    else:
        signal_y = np.full(cW, H / 2.0)

    # ── 5. Mode-based baseline correction ────────────────────────
    y_ints     = np.clip(np.round(signal_y).astype(int), v_lo, v_hi - 1)
    counts     = np.bincount(y_ints, minlength=H)
    counts[:v_lo] = 0;  counts[v_hi:] = 0
    baseline_y = float(np.argmax(counts))
    signal     = baseline_y - signal_y          # positive = upward deflection

    pad    = np.full(x_start, signal[0])
    signal = np.concatenate([pad, signal])

    # ── 6. Residual baseline wander removal ───────────────────────
    win_bl = min(len(signal), 1000)
    if win_bl > 3:
        drift  = uniform_filter1d(signal.astype(float), size=win_bl, mode="nearest")
        signal = signal - drift

    # ── 7. Outlier / spike rejection ──────────────────────────────
    med    = float(np.median(signal))
    std    = float(np.std(signal)) or 1.0
    signal = np.clip(signal, med - 3.5 * std, med + 3.5 * std)

    # ── 8. Median pre-filter + Savitzky-Golay smoothing ──────────
    signal = medfilt(signal.astype(float), kernel_size=3)
    win    = min(smooth_win, len(signal) if len(signal) % 2 == 1 else len(signal) - 1)
    if win >= 3 and len(signal) > win * 2:
        signal = savgol_filter(signal, window_length=win, polyorder=2)

    return signal


# =============================================================================
# Improved local R-peak detector
# =============================================================================

def _improved_detect_rpeaks(sig_mv: np.ndarray, fs: float) -> np.ndarray:
    """
    R-peak detector with three targeted fixes over ecg_segment.detect_rpeaks:

    1. Threshold from middle 80% of signal — excludes the calibration-pulse
       transient at the start of column-1 leads (I, II, III) which inflates
       the 85th-percentile threshold above the true QRS amplitude.

    2. No width constraint (removed width=min_width).  The minimum-distance
       and amplitude threshold are sufficient; width=7 incorrectly filtered
       the second QRS peak in lead II (rS pattern, narrow HPF transient).

    3. Two post-processing passes to suppress T-waves / f-waves that survive
       the 5 Hz HPF and appear > 0.43 s after the QRS:
       a) Short-RR filter: removes any peak preceded by a gap < 65% of the
          median RR (catches T-waves that just barely exceed min_dist=0.43 s,
          e.g. the T-wave at sample 945 in aVF that is 136 samples after
          the QRS at 809, while min_dist=132).
       b) Even-index check: if HR > 90 bpm after step (a), test whether
          taking every other peak gives a more plausible (40-90 bpm) rate;
          handles the case where a T-wave interleaves between every real QRS
          (lead III: QRS@147, T-wave@318, QRS@535 → use [147, 535] = 48 bpm).
    """
    nyq = fs / 2.0
    try:
        b, a    = butter(3, 5.0 / nyq, btype='high')
        filtered = filtfilt(b, a, sig_mv)
    except Exception:
        filtered = sig_mv

    # Thresholds from middle 80% to exclude calibration-pulse artefacts
    n   = len(filtered)
    mid = filtered[int(0.12 * n): int(0.88 * n)]
    pos_thr = max(0.05, float(np.percentile(mid, 85)) * 0.30)
    neg_thr = max(0.05, abs(float(np.percentile(mid, 15))) * 0.30)

    min_dist = max(1, int(0.43 * fs))   # 0.43 s -> max ~140 bpm

    # No width= constraint: distance + threshold are sufficient; the width
    # filter incorrectly rejected the 2nd QRS peak in lead II (HR n/a).
    pos_peaks, _ = find_peaks(filtered,  height=pos_thr, distance=min_dist)
    neg_peaks, _ = find_peaks(-filtered, height=neg_thr, distance=min_dist)

    # Discard peaks in the first 20% — calibration-box trailing-edge
    # transients and early AFib f-wave bursts live here.
    lead_in = int(0.20 * n)
    pos_peaks = pos_peaks[pos_peaks >= lead_in]
    neg_peaks = neg_peaks[neg_peaks >= lead_in]

    if not len(pos_peaks) and not len(neg_peaks):
        return pos_peaks
    if not len(pos_peaks):
        return neg_peaks
    if not len(neg_peaks):
        return pos_peaks

    pos_mean = float(np.mean(filtered[pos_peaks]))
    neg_mean = float(np.mean(-filtered[neg_peaks]))
    peaks = pos_peaks if pos_mean >= neg_mean else neg_peaks

    # ── Post-processing: suppress late T-waves / f-waves ─────────────────────

    # Pass (a): remove any peak that follows an unusually short RR interval
    # (< 65% of the median).  These are T-waves or f-wave clusters that just
    # barely exceed min_dist=0.43 s and would otherwise double-count a beat.
    if len(peaks) >= 3:
        rr = np.diff(peaks)
        median_rr = float(np.median(rr))
        keep = np.ones(len(peaks), dtype=bool)
        for i in range(len(rr)):
            if rr[i] < 0.65 * median_rr:
                keep[i + 1] = False           # short-RR peak is the T-wave
        peaks = peaks[keep]

    # Pass (b): if HR is still implausibly high (> 90 bpm) with ≥ 3 peaks,
    # test even-index subsampling.  This handles leads where a T-wave lands
    # exactly between every pair of QRS peaks (lead III: T-wave at 318
    # between QRS@147 and QRS@535 — subsampling [147,318,535]→[147,535]).
    if len(peaks) >= 3:
        rr = np.diff(peaks)
        hr_est = 60.0 / (float(np.mean(rr)) / fs)
        if hr_est > 90:
            alt_peaks = peaks[::2]
            if len(alt_peaks) >= 2:
                alt_rr  = np.diff(alt_peaks)
                alt_hr  = 60.0 / (float(np.mean(alt_rr)) / fs)
                if 40 <= alt_hr <= 90 and alt_hr < hr_est * 0.65:
                    peaks = alt_peaks

    return peaks


def _improved_analyze_lead(sig_px: np.ndarray, name: str,
                            px_per_mm: float, speed: float = 25.0,
                            gain: float = 10.0) -> dict:
    """analyze_lead wrapper that uses _improved_detect_rpeaks."""
    fs       = px_per_mm * speed
    sig_mv   = px_to_mv(sig_px, px_per_mm, gain)
    t_s      = time_axis(len(sig_px), px_per_mm, speed)
    peaks    = _improved_detect_rpeaks(sig_mv, fs)
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


# =============================================================================
# Main test runner
# =============================================================================

def run_test(image_path: str, model_path: str,
             out_dir: str = "ecg_output",
             dpi: float = None, paper_speed: float = 25.0,
             paper_gain: float = 10.0):

    img_path = Path(image_path)
    if not img_path.exists():
        sys.exit(f"Image not found: {img_path}")

    model_path = Path(model_path)
    if not model_path.exists():
        sys.exit(f"Checkpoint not found: {model_path}")

    # ── Output folder ────────────────────────────────────────────────────────
    out = Path(out_dir) / img_path.stem
    out.mkdir(parents=True, exist_ok=True)

    # ── 1. Load image ────────────────────────────────────────────────────────
    img = cv2.imread(str(img_path))
    if img is None:
        sys.exit(f"Could not read image: {img_path}")
    cv2.imwrite(str(out / f"original{img_path.suffix}"), img)
    print(f"\n[1] Image      : {img_path.name}  ({img.shape[1]}x{img.shape[0]} px)")
    print(f"    Output     -> {out}")

    # ── 2. Estimate pixel scale ──────────────────────────────────────────────
    px_per_mm = estimate_px_per_mm(img, known_dpi=dpi)
    print(f"[2] Scale      : {px_per_mm:.2f} px/mm  (fs ~{px_per_mm*paper_speed:.1f} Hz)")

    # ── 3. Segment leads ─────────────────────────────────────────────────────
    print("[3] Segmenting leads ...")
    leads = auto_segment_leads(img)
    print(f"    Found: {', '.join(leads)}")
    save_lead_images(leads, out / "lead_images")
    save_segmentation_preview(img, leads, out)

    # ── 4. Extract signals & per-lead analysis ───────────────────────────────
    print("[4] Extracting signals (improved digitizer) ...")
    results = []
    lead_signals = {}
    for name, data in leads.items():
        # Use improved local digitizer instead of ecg_segment.extract_signal
        sig_px = _improved_extract_signal(data["gray"])
        # Use improved local analyzer for better R-peak detection
        res    = _improved_analyze_lead(sig_px, name, px_per_mm,
                                        paper_speed, paper_gain)
        results.append(res)
        if name in LEAD_12:
            lead_signals[name] = res["signal_mv"]
        hr_s = f"{res['hr_bpm']:.0f} bpm" if not math.isnan(res["hr_bpm"]) else "n/a"
        st_s = f"{res['st_dev_mv']:+.3f} mV"
        print(f"    {name:<10}  HR {hr_s:<10}  ST {st_s}")

    missing = [n for n in LEAD_ORDER if n not in lead_signals]
    if missing:
        print(f"    WARNING: missing leads {missing} -- zero-padded for model input")

    # ── 5. Rule-based clinical diagnosis (LITFL) ─────────────────────────────
    print("[5] Rule-based diagnosis (LITFL) ...")
    diag = ECGDiagnoser().diagnose(results)

    # ── 6. Load model & classify ─────────────────────────────────────────────
    print(f"\n[6] Loading model: {model_path.name}")
    model = load_model(str(model_path))
    print("[7] Running 12-lead ResNet ...")
    ml = classify(lead_signals, model)
    diag["ml_classification"] = ml

    # ── 7. Print combined results ─────────────────────────────────────────────
    print("\n" + "=" * 55)
    print(f"  RULE ENGINE  : {diag['severity']}")
    print(f"  ResNet-12L   : {ml['label']}  ({ml['confidence']:.1%} confidence)")
    print()
    print("  CLASS PROBABILITIES")
    for cls, prob in ml["probabilities"].items():
        bar = "#" * int(prob * 30)
        print(f"    {cls:<6} {prob:5.1%}  {bar}")
    print("=" * 55)

    # ── 8. Save visualizations ───────────────────────────────────────────────
    print("\n[8] Saving outputs ...")
    save_signal_plots(results, out)
    save_diagnosis_report(diag, out, img_name=img_path.name)
    _save_ml_summary(ml, diag, out)

    print(f"\n  All outputs -> {out}")
    return diag


def _save_ml_summary(ml: dict, diag: dict, out_dir: Path):
    """Side-by-side ML probability bars + rule-engine findings figure."""
    import matplotlib.pyplot as plt

    classes = list(ml["probabilities"].keys())
    probs   = list(ml["probabilities"].values())
    colors  = ["#2ecc71" if c == ml["label"] else "#5dade2" for c in classes]

    sev_color = {"NORMAL": "#2ecc71", "ABNORMAL": "#e67e22",
                 "CRITICAL": "#e74c3c"}.get(diag["severity"], "#95a5a6")

    fig, (ax_bar, ax_text) = plt.subplots(1, 2, figsize=(12, 4),
                                           gridspec_kw={"width_ratios": [1, 1.4]})
    fig.patch.set_facecolor("#0d1b2a")

    ax_bar.set_facecolor("#111827")
    bars = ax_bar.barh(classes, probs, color=colors, edgecolor="none")
    for bar, p in zip(bars, probs):
        ax_bar.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{p:.1%}", va="center", color="white", fontsize=10)
    ax_bar.set_xlim(0, 1.15)
    ax_bar.set_xlabel("Probability", color="#aaa")
    ax_bar.set_title(f"ResNet-12L: {ml['label']}  ({ml['confidence']:.1%})",
                     color="white", fontsize=11, pad=8)
    ax_bar.tick_params(colors="white")
    for spine in ax_bar.spines.values():
        spine.set_color("#333")

    ax_text.set_facecolor(sev_color + "22")
    ax_text.axis("off")
    ax_text.text(0.5, 0.97, f"Rule Engine: {diag['severity']}",
                 ha="center", va="top", fontsize=12, fontweight="bold",
                 color=sev_color, transform=ax_text.transAxes)
    findings_str = "\n".join(f"  * {f}" for f in diag["findings"])
    ax_text.text(0.05, 0.80, findings_str, ha="left", va="top",
                 fontsize=9, color="#ecf0f1", transform=ax_text.transAxes,
                 linespacing=1.6)

    plt.tight_layout()
    path = out_dir / "ml_diagnosis_summary.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  ML summary     -> {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Test 12-lead ResNet on a scanned ECG image.")
    parser.add_argument("image",
        help="Path to the ECG image (PNG/JPG).")
    parser.add_argument("--model",
        default=str(_ECG_ROOT / "ecg-12-lead-model"
                              / "run_20260603_141028"
                              / "ecg_resnet1d_best.pth"),
        help="Path to the 12-lead ResNet checkpoint (.pth). "
             "Defaults to run_20260603_141028/ecg_resnet1d_best.pth")
    parser.add_argument("--out",   default="ecg_output",
        help="Output folder (default: ecg_output/<image_stem>/).")
    parser.add_argument("--dpi",   type=float, default=None,
        help="Image DPI (leave blank to auto-detect).")
    parser.add_argument("--speed", type=float, default=25.0,
        help="Paper speed mm/s (default 25).")
    parser.add_argument("--gain",  type=float, default=10.0,
        help="Amplitude gain mm/mV (default 10).")
    args = parser.parse_args()

    run_test(args.image, args.model, out_dir=args.out,
             dpi=args.dpi, paper_speed=args.speed, paper_gain=args.gain)


if __name__ == "__main__":
    main()
