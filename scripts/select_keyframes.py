"""
Select well-opacified, de-identified keyframes from raw angiogram frame dumps.

Each `Study-*` folder contains one or more `Series-*` folders, each holding every
extracted frame of one run (mostly non-contrast fluoro + a contrast bolus window).
This script, per series:

  1. Auto-detects the burned-in text overlay bands (top/bottom) from the first
     frame and blacks them out on every frame it writes out (de-identification —
     these bands contain patient name/MRN/DOB-derived age).
  2. Scores every frame with a Frangi vesselness filter restricted to the
     non-overlay ROI, as a proxy for "how much of the artery tree is opacified".
  3. Finds the peak/plateau of that score curve and picks up to --top-n frames
     spread across it (skips series with no clear contrast rise).
  4. Writes the selected, de-identified frames + a manifest CSV for the next
     (labeling) stage. Never modifies the source study folder.

Usage:
  python select_keyframes.py --study-dir "D:/UNI/FYP/Pipeline/Study-125-XA-Cardiac[20260102]" \
      --out D:/UNI/FYP/Pipeline/keyframes_out --plot

  python select_keyframes.py --studies-root D:/UNI/FYP/Pipeline/raw_studies \
      --out D:/UNI/FYP/Pipeline/keyframes_out --workers 8
"""
import argparse
import csv
import os
import sys
from dataclasses import dataclass, field
from multiprocessing import Pool, cpu_count

import cv2
import numpy as np
from skimage.filters import frangi

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


@dataclass
class SeriesResult:
    study: str
    series: str
    frames: list = field(default_factory=list)      # file names, sorted
    scores: list = field(default_factory=list)       # float per frame
    top_band: tuple = (0, 0)                         # (start, end) rows, blacked out
    bottom_band: tuple = (0, 0)
    has_contrast: bool = False
    selected: list = field(default_factory=list)     # indices into frames/scores


def find_overlay_bands(gray, margin_frac=0.22, bright_thresh=235, min_bright_px=15):
    """Detect burned-in text rows near the top/bottom margins of a frame.

    Returns ((top_start, top_end), (bottom_start, bottom_end)) as half-open row
    ranges to black out. Empty range (0, 0) if nothing found on that side.
    """
    h, w = gray.shape
    margin = int(h * margin_frac)

    bright_count = (gray > bright_thresh).sum(axis=1)
    rows_with_text = bright_count > min_bright_px

    top_rows = np.where(rows_with_text[:margin])[0]
    top_band = (0, int(top_rows.max()) + 6) if len(top_rows) else (0, 0)

    bottom_rows = np.where(rows_with_text[h - margin:])[0]
    bottom_band = (h - margin + int(bottom_rows.min()) - 6, h) if len(bottom_rows) else (0, 0)

    return top_band, bottom_band


def deidentify(gray_or_bgr, top_band, bottom_band):
    out = gray_or_bgr.copy()
    if top_band[1] > top_band[0]:
        out[top_band[0]:top_band[1], ...] = 0
    if bottom_band[1] > bottom_band[0]:
        out[bottom_band[0]:bottom_band[1], ...] = 0
    return out


# skimage's frangi() defaults to gamma=None, which auto-derives the vesselness
# sensitivity from *each image's own* Hessian-norm distribution. That makes the
# response scale different per frame, so absolute/area-based thresholds compare
# apples to oranges across a series (noisy non-contrast frames with a catheter +
# rib/diaphragm edges can outscore a frame with a genuinely opacified vessel
# tree). Pinning gamma to a fixed constant makes scores comparable frame-to-frame
# within a series -- verified against a hand-labeled contrast window on sample
# data, where auto-gamma ranked the clearest frame lowest and a noise-only frame
# highest; fixed gamma correctly ranked the clear frame highest.
FRANGI_GAMMA = 0.03
# Response magnitude still shifts somewhat with each series' own exposure
# technique (mA/kV differs series to series, e.g. 571mA/125kV vs 913mA/78kV in
# our sample study). 0.15 was tuned only against one series and silently cut off
# most of the real vessel tree in a lower-magnitude-response series, nearly
# erasing the peak/baseline gap (ratio 1.3x) even though the frame visually had
# an obvious, clean opacified tree. 0.02 preserves a consistent ~5x peak/baseline
# separation across both series tested.
FRANGI_AREA_THRESH = 0.02
# frangi() pads with mode='reflect' at the array border, which can synthesize
# fake ridge-like responses right at the edge of whatever ROI crop we pass in --
# verified this made the score sensitive to +/-5px changes in the (auto-detected,
# so not pixel-exact) overlay-band crop, enough to flip a non-contrast frame's
# score from 0 to 45. Trimming a border margin off the response before scoring
# makes the score stable regardless of exact crop placement.
FRANGI_BORDER_TRIM = 12


def vesselness_score(gray_roi, scale=1.0, sigmas=range(2, 6)):
    """Frangi-filter based score: higher = more visible tubular (vessel) structure."""
    if scale != 1.0:
        gray_roi = cv2.resize(
            gray_roi, (max(1, int(gray_roi.shape[1] * scale)), max(1, int(gray_roi.shape[0] * scale)))
        )
    norm = gray_roi.astype(np.float64) / 255.0
    resp = frangi(norm, sigmas=sigmas, black_ridges=True, gamma=FRANGI_GAMMA)
    m = FRANGI_BORDER_TRIM
    resp = resp[m:-m, m:-m] if resp.shape[0] > 2 * m and resp.shape[1] > 2 * m else resp
    # Area of strong-response pixels == proxy for "how much of the tree is
    # opacified", not just "is there a thin catheter/rib edge".
    return float((resp > FRANGI_AREA_THRESH).sum())


def process_series(args):
    study_dir, series_name, scale, stride = args
    series_dir = os.path.join(study_dir, series_name)
    frames = sorted(f for f in os.listdir(series_dir) if f.lower().endswith(IMG_EXTS))
    if not frames:
        return None

    first = cv2.imread(os.path.join(series_dir, frames[0]), cv2.IMREAD_GRAYSCALE)
    if first is None:
        return None
    top_band, bottom_band = find_overlay_bands(first)

    result = SeriesResult(study=os.path.basename(study_dir), series=series_name,
                           frames=frames, top_band=top_band, bottom_band=bottom_band)

    roi_top = top_band[1]
    roi_bottom = bottom_band[0] if bottom_band[1] > bottom_band[0] else first.shape[0]

    scores = []
    for i, fname in enumerate(frames):
        if i % stride != 0:
            scores.append(None)
            continue
        gray = cv2.imread(os.path.join(series_dir, fname), cv2.IMREAD_GRAYSCALE)
        roi = gray[roi_top:roi_bottom, :]
        scores.append(vesselness_score(roi, scale=scale))

    # fill skipped frames (stride > 1) by linear interpolation for a smooth curve
    idxs = [i for i, s in enumerate(scores) if s is not None]
    for i in range(len(scores)):
        if scores[i] is None:
            lo = max([j for j in idxs if j <= i], default=idxs[0])
            hi = min([j for j in idxs if j >= i], default=idxs[-1])
            scores[i] = scores[lo] if lo == hi else np.interp(i, [lo, hi], [scores[lo], scores[hi]])
    result.scores = scores
    return result


def smooth(values, window=3):
    values = np.asarray(values, dtype=np.float64)
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="same")


def select_from_curve(scores, top_n, plateau_ratio, min_relative_rise, min_peak_area):
    smoothed = smooth(scores)
    baseline = float(np.percentile(smoothed, 25))
    peak = float(smoothed.max())
    peak_idx = int(smoothed.argmax())

    # Fixed-gamma Frangi area legitimately sits near 0 on non-contrast frames, so
    # guard the relative-rise ratio against a near-zero baseline with a floor, and
    # additionally require an absolute minimum peak so a noisy-but-flat curve
    # (e.g. catheter-only, no dye at all) doesn't get accepted just because its
    # tiny baseline makes the ratio look big.
    rise = (peak - baseline) / max(baseline, 0.5)
    if peak < min_peak_area or rise < min_relative_rise:
        return False, []

    plateau_thresh = baseline + plateau_ratio * (peak - baseline)
    plateau_idxs = [i for i, v in enumerate(smoothed) if v >= plateau_thresh]

    if len(plateau_idxs) <= top_n:
        chosen = plateau_idxs
    else:
        # spread evenly across the plateau instead of clustering near the peak
        positions = np.linspace(0, len(plateau_idxs) - 1, top_n).round().astype(int)
        chosen = sorted({plateau_idxs[p] for p in positions})
        if peak_idx not in chosen:
            chosen[-1] = peak_idx if len(chosen) == top_n else chosen.append(peak_idx) or chosen
    return True, sorted(set(chosen))


def save_contact_sheet(result, series_dir, out_path, thumb_w=160, cols=8):
    """De-identified thumbnail grid of every frame, selected ones outlined in red.

    Lets a human confirm/override the automatic picks in a few seconds by eye
    instead of trusting the score blindly or scrubbing frame-by-frame.
    """
    first = cv2.imread(os.path.join(series_dir, result.frames[0]))
    h, w = first.shape[:2]
    thumb_h = int(thumb_w * h / w)
    rows = (len(result.frames) + cols - 1) // cols
    canvas = np.zeros((rows * (thumb_h + 18), cols * thumb_w, 3), dtype=np.uint8)

    for i, fname in enumerate(result.frames):
        img = cv2.imread(os.path.join(series_dir, fname))
        img = deidentify(img, result.top_band, result.bottom_band)
        thumb = cv2.resize(img, (thumb_w, thumb_h))
        r, c = divmod(i, cols)
        y0, x0 = r * (thumb_h + 18), c * thumb_w
        if i in result.selected:
            cv2.rectangle(thumb, (0, 0), (thumb_w - 1, thumb_h - 1), (0, 0, 255), 3)
        canvas[y0:y0 + thumb_h, x0:x0 + thumb_w] = thumb
        cv2.putText(canvas, str(i), (x0 + 4, y0 + thumb_h + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imwrite(out_path, canvas)


def save_outputs(result, study_dir, out_root, plot=False):
    series_out = os.path.join(out_root, result.study, result.series)
    os.makedirs(series_out, exist_ok=True)
    series_dir = os.path.join(study_dir, result.series)

    for idx in result.selected:
        fname = result.frames[idx]
        img = cv2.imread(os.path.join(series_dir, fname))
        img = deidentify(img, result.top_band, result.bottom_band)
        cv2.imwrite(os.path.join(series_out, fname), img)

    if plot:
        save_contact_sheet(result, series_dir, os.path.join(series_out, "_contact_sheet.png"))
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 3))
        plt.plot(result.scores, label="vesselness score")
        plt.plot(smooth(result.scores), label="smoothed")
        for idx in result.selected:
            plt.axvline(idx, color="red", alpha=0.6)
        plt.title(f"{result.study} / {result.series}  (contrast={result.has_contrast})")
        plt.xlabel("frame index")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(series_out, "_score_curve.png"))
        plt.close()


def find_study_dirs(studies_root=None, study_dir=None):
    if study_dir:
        return [study_dir]
    return [os.path.join(studies_root, d) for d in sorted(os.listdir(studies_root))
            if d.startswith("Study-") and os.path.isdir(os.path.join(studies_root, d))]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--study-dir", help="Path to a single Study-* folder")
    ap.add_argument("--studies-root", help="Path containing multiple Study-* folders")
    ap.add_argument("--out", required=True, help="Output directory for de-identified keyframes + manifest")
    ap.add_argument("--top-n", type=int, default=3, help="Max frames to keep per series")
    ap.add_argument("--plateau-ratio", type=float, default=0.6)
    ap.add_argument("--min-relative-rise", type=float, default=0.8,
                     help="Min (peak-baseline)/max(baseline,0.5) to consider a series contrast-injected")
    ap.add_argument("--min-peak-area", type=float, default=4.0,
                     help="Min absolute smoothed peak score (vessel-pixel area) to consider a series contrast-injected")
    ap.add_argument("--scale", type=float, default=1.0,
                     help="Downscale factor for vesselness scoring (1.0=full res, validated setting; lower trades accuracy for speed)")
    ap.add_argument("--stride", type=int, default=1, help="Score every Nth frame, interpolate the rest")
    ap.add_argument("--workers", type=int, default=max(1, cpu_count() - 1))
    ap.add_argument("--plot", action="store_true", help="Save a score-curve PNG per series (QA)")
    args = ap.parse_args()

    if not args.study_dir and not args.studies_root:
        ap.error("pass --study-dir or --studies-root")

    study_dirs = find_study_dirs(args.studies_root, args.study_dir)
    os.makedirs(args.out, exist_ok=True)

    tasks = []
    for sd in study_dirs:
        for series_name in sorted(os.listdir(sd)):
            if series_name.startswith("Series-") and os.path.isdir(os.path.join(sd, series_name)):
                tasks.append((sd, series_name, args.scale, args.stride))

    print(f"{len(study_dirs)} study folder(s), {len(tasks)} series to process, {args.workers} workers")

    manifest_rows = []
    with Pool(args.workers) as pool:
        for i, result in enumerate(pool.imap_unordered(process_series, tasks), 1):
            if result is None:
                continue
            study_dir = next(sd for sd, sn, _, _ in tasks if sd.endswith(result.study) and sn == result.series)
            result.has_contrast, result.selected = select_from_curve(
                result.scores, args.top_n, args.plateau_ratio, args.min_relative_rise, args.min_peak_area
            )
            status = f"{len(result.selected)} frame(s)" if result.has_contrast else "NO CONTRAST DETECTED"
            print(f"[{i}/{len(tasks)}] {result.study}/{result.series}: {status}")

            if result.has_contrast:
                save_outputs(result, study_dir, args.out, plot=args.plot)

            for idx in range(len(result.frames)):
                manifest_rows.append({
                    "study": result.study,
                    "series": result.series,
                    "frame": result.frames[idx],
                    "frame_idx": idx,
                    "score": round(result.scores[idx], 4),
                    "has_contrast_series": result.has_contrast,
                    "selected": idx in result.selected,
                })

    manifest_path = os.path.join(args.out, "manifest.csv")
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)

    n_series = len({(r["study"], r["series"]) for r in manifest_rows})
    n_selected = sum(1 for r in manifest_rows if r["selected"])
    n_no_contrast = len({(r["study"], r["series"]) for r in manifest_rows if not r["has_contrast_series"]})
    print(f"\nDone. {n_series} series, {n_selected} keyframes selected, "
          f"{n_no_contrast} series flagged as no-contrast. Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
