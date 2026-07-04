"""
qca2.py  —  Shape-aware Quantitative Coronary Angiography
==========================================================

Key addition over qca.py:
  Before any lesion detection, each branch diameter profile is classified by shape:

    UNIFORM   — smoothed profile is roughly horizontal or pure linear taper.
                Residual range < uniform_shape_thresh × mean_d.  No pathology.

    STENOSIS  — V-shape: diameter decreases then recovers.
                Residual dips below −stenosis_shape_thresh × mean_d.
                → Standard QCA (MLD, RVD, DS%) with trend-based RVD.

    ECTASIA   — ∧-shape (inverted V): vessel widens then returns to normal.
                Residual rises above +ectasia_shape_thresh × mean_d.
                → Ectasia ratio = (peak_d / trend_at_peak − 1) × 100 %.

    COMPLEX   — both a V-dip and a ∧-bulge present on the same branch.
                Both stenosis and ectasia findings are reported.

    TAPERED   — monotonic decline within physiological range; no V or ∧.
                QCA is still run (using trend-based RVD) but with awareness
                that slight DS values are likely physiological taper, not lesion.

Reference line:
  A linear least-squares fit to d_smooth is used as the "expected" diameter
  at every position.  This is mathematically equivalent to the interpolated
  RVD of Suzuki et al. 2020 (JACIT) but derived from the full branch, making
  it more robust for diffuse / long-segment lesions.
"""

import sys
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Import infrastructure from qca.py
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).parent
sys.path.insert(0, str(_THIS_DIR))

from qca import (                                           # noqa: E402
    QCAConfig,
    to_binary_mask, morph_cleanup,
    get_skeleton_from_mask, extract_all_branches,
    ordered_centerline_from_mask,
    smooth_1d, smooth_branch_coords,
    _perpendicular_diameter, _find_vessel_shoulders,
    local_minima_indices, arc_length,
    _separate_touching_vessels,
    draw_overlay,
    _BRANCH_COLORS,
)
import qca as _qca  # fallback for UNIFORM/TAPERED branches
from scipy.ndimage import distance_transform_edt


# ==========================================================================
# Extended config
# ==========================================================================
@dataclass
class QCA2Config(QCAConfig):
    # Shape classification thresholds (fraction of mean branch diameter)
    stenosis_shape_thresh: float = 0.12  # residual < -12 % mean → STENOSIS
    ectasia_shape_thresh:  float = 0.15  # residual > +15 % mean → ECTASIA
    uniform_shape_thresh:  float = 0.08  # total residual range < 8 % → UNIFORM

    # Ectasia severity (ectasia ratio thresholds)
    ectasia_mild_thresh:   float = 0.20  # ratio < 20 % → MILD
    ectasia_severe_thresh: float = 0.50  # ratio ≥ 50 % → SEVERE (aneurysm)

    # Minimum DS% to report a stenosis finding
    min_ds_report: float = 8.0


# ==========================================================================
# Profile shape analysis
# ==========================================================================
@dataclass
class ProfileShape:
    kind:        str            # UNIFORM / STENOSIS / ECTASIA / COMPLEX / TAPERED
    slope:       float          # linear trend slope (px per step)
    intercept:   float          # linear trend y-intercept
    d_trend:     np.ndarray     # trend values at each index
    d_residual:  np.ndarray     # d_smooth − d_trend
    min_residual: float         # most negative residual value (px)
    max_residual: float         # most positive residual value (px)
    mean_d:      float          # mean smoothed diameter of the branch


def analyze_profile_shape(d_s: np.ndarray, cfg: QCA2Config) -> ProfileShape:
    """Fit a linear trend and classify shape by residual deviation.

    The trend represents the "expected" diameter progression (natural taper).
    Deviations from it reveal pathology:
      negative V-dip  → stenosis
      positive ∧-bulge → ectasia / aneurysm
    """
    N = len(d_s)
    mean_d = float(np.mean(d_s)) if N > 0 else 0.0

    if N < 4 or mean_d < 1e-6:
        trend = np.full(N, mean_d)
        res = d_s.astype(float) - trend
        return ProfileShape("UNIFORM", 0.0, mean_d, trend, res,
                            float(res.min()) if N else 0.0,
                            float(res.max()) if N else 0.0, mean_d)

    x = np.arange(N, dtype=float)
    slope, intercept = np.polyfit(x, d_s.astype(float), 1)
    d_trend = slope * x + intercept
    d_residual = d_s.astype(float) - d_trend

    min_res = float(d_residual.min())
    max_res = float(d_residual.max())

    st = cfg.stenosis_shape_thresh * mean_d
    et = cfg.ectasia_shape_thresh  * mean_d
    ut = cfg.uniform_shape_thresh  * mean_d

    has_stenosis = min_res < -st
    has_ectasia  = max_res >  et

    if (max_res - min_res) < ut:
        kind = "UNIFORM"
    elif has_stenosis and has_ectasia:
        kind = "COMPLEX"
    elif has_stenosis:
        kind = "STENOSIS"
    elif has_ectasia:
        kind = "ECTASIA"
    else:
        kind = "TAPERED"

    return ProfileShape(kind=kind, slope=slope, intercept=intercept,
                        d_trend=d_trend, d_residual=d_residual,
                        min_residual=min_res, max_residual=max_res,
                        mean_d=mean_d)


# ==========================================================================
# Helpers
# ==========================================================================
def _local_maxima_indices(x: np.ndarray) -> List[int]:
    """Return indices of local maxima (including endpoints)."""
    N = len(x)
    maxs = []
    for i in range(1, N - 1):
        if x[i] >= x[i - 1] and x[i] > x[i + 1]:
            maxs.append(i)
    if N >= 2 and x[0] > x[1]:
        maxs.append(0)
    if N >= 2 and x[N - 1] > x[N - 2]:
        maxs.append(N - 1)
    return maxs


def _merge_findings(findings: list, sort_key: str) -> list:
    """Merge overlapping findings on the same branch, keeping the most severe."""
    findings = sorted(findings, key=lambda z: (z["L_idx"], z["R_idx"]))
    merged = []
    for f in findings:
        if not merged:
            merged.append(f)
            continue
        prev = merged[-1]
        if f["L_idx"] <= prev["R_idx"]:
            best = f if f.get(sort_key, 0) > prev.get(sort_key, 0) else prev
            best = dict(best)
            best["L_idx"] = min(prev["L_idx"], f["L_idx"])
            best["R_idx"] = max(prev["R_idx"], f["R_idx"])
            merged[-1] = best
        else:
            merged.append(f)
    return merged


def _gradient_stats(d_s: np.ndarray, L: int, m: int, R: int) -> Tuple[float, float]:
    """Return (entry_grad, exit_grad) in px/step."""
    span = R - L + 1
    grad_win = max(3, min(8, span // 4))
    entry_seg = d_s[L : min(L + grad_win + 1, m + 1)]
    exit_seg  = d_s[max(m, R - grad_win) : R + 1]
    entry_grad = float((entry_seg[-1] - entry_seg[0]) / max(1, len(entry_seg) - 1)) \
        if len(entry_seg) > 1 else 0.0
    exit_grad  = float((exit_seg[-1] - exit_seg[0]) / max(1, len(exit_seg) - 1)) \
        if len(exit_seg) > 1 else 0.0
    return entry_grad, exit_grad


# ==========================================================================
# Shape-aware finding detection
# ==========================================================================
def detect_findings_on_branch(branch: List[Tuple[int, int]],
                               dt: np.ndarray,
                               cfg: QCA2Config,
                               bw_mask: Optional[np.ndarray] = None) -> list:
    """Shape-aware stenosis + ectasia detection for a single branch.

    Returns a list of finding dicts.  Each dict carries:
      "type"         : "STENOSIS" or "ECTASIA"
      "severity"     : MILD / MODERATE / SEVERE
      "DS_percent"   : diameter stenosis % (negative for ectasia)
      "ectasia_ratio": ectasia ratio % (0 for stenosis findings)
      "shape_kind"   : UNIFORM / STENOSIS / ECTASIA / COMPLEX / TAPERED
      ... plus all standard qca fields (MLD_px, RVD_px, L_idx, R_idx, …)
    """
    N = len(branch)

    # ----- Diameter profile -----
    d_raw = np.zeros(N, dtype=np.float32)
    for i, (y, x) in enumerate(branch):
        if bw_mask is not None:
            pd = _perpendicular_diameter(branch, bw_mask, i)
            d_raw[i] = pd if pd > 0.5 else 2.0 * dt[y, x]
        else:
            d_raw[i] = 2.0 * dt[y, x]

    d_s    = smooth_1d(d_raw, cfg.smooth_win)
    d_fine = smooth_1d(d_raw, max(3, cfg.smooth_win // 3))

    # ----- Shape classification -----
    shape = analyze_profile_shape(d_s, cfg)
    mean_d  = shape.mean_d
    d_trend = shape.d_trend
    d_res   = shape.d_residual

    if mean_d < 1e-6:
        return []

    branch_ref = float(np.percentile(d_s, 80)) if N > 0 else mean_d
    findings: list = []

    # ══════════════════════════════════════════════════════════════════════
    # STENOSIS path  (STENOSIS, COMPLEX, or TAPERED branches)
    # ══════════════════════════════════════════════════════════════════════
    if shape.kind in ("STENOSIS", "COMPLEX", "TAPERED", "UNIFORM"):
        st = cfg.stenosis_shape_thresh * mean_d

        # Candidates: local minima in the *residual* reveal shape-relative dips.
        # Also include d_s minima to catch lesions on UNIFORM/TAPERED branches.
        res_mins = local_minima_indices(d_res)
        ds_mins  = local_minima_indices(d_s)
        all_mins = sorted(set(res_mins + ds_mins))

        for m in all_mins:
            # For STENOSIS/COMPLEX, require the residual to be a meaningful dip.
            # For TAPERED/UNIFORM, fall back to absolute threshold against branch_ref.
            if shape.kind in ("STENOSIS", "COMPLEX"):
                if d_res[m] >= -st:
                    continue
            else:
                if d_s[m] >= cfg.lesion_alpha * branch_ref:
                    continue

            # Trend value at m is the expected (reference) diameter.
            RVD_at_m = max(float(d_trend[m]), branch_ref * 0.5)
            thr = cfg.lesion_alpha * RVD_at_m

            # Walk outward while d_s < threshold (vessel still narrowed)
            L = m
            while L > 0 and d_s[L] < thr:
                L -= 1
            R = m
            while R < N - 1 and d_s[R] < thr:
                R += 1

            if (R - L + 1) < cfg.min_lesion_points:
                continue

            # MLD from the fine-smooth profile within the span
            m_fine = L + int(np.argmin(d_fine[L : R + 1]))
            MLD    = float(d_fine[m_fine])

            # RVD = trend at MLD position (linear interpolation accounts for taper)
            RVD = max(float(d_trend[m_fine]), branch_ref * 0.5)
            if RVD <= 1e-6:
                continue

            percent_DS = (1.0 - MLD / RVD) * 100.0
            if percent_DS < cfg.min_ds_report:
                continue

            length_px = arc_length(branch, L, R)
            entry_grad, exit_grad = _gradient_stats(d_s, L, m_fine, R)

            severity = "MILD"
            if percent_DS >= cfg.severe_threshold:
                severity = "SEVERE"
            elif percent_DS >= cfg.moderate_threshold:
                severity = "MODERATE"

            findings.append({
                "type":         "STENOSIS",
                "L_idx":        int(L),
                "R_idx":        int(R),
                "min_idx":      int(m_fine),
                "MLD_px":       MLD,
                "RVD_px":       RVD,
                "DS_percent":   percent_DS,
                "ectasia_ratio": 0.0,
                "length_px":    length_px,
                "MLD_mm":       None,
                "RVD_mm":       None,
                "length_mm":    None,
                "entry_grad":   entry_grad,
                "exit_grad":    exit_grad,
                "severity":     severity,
                "min_pt":       branch[m_fine],
                "branch":       branch,
                "d_smooth":     d_s,
                "d_fine":       d_fine,
                "d_trend":      d_trend,
                "d_residual":   d_res,
                "shape_kind":   shape.kind,
                "branch_shape": shape,
            })

    # ══════════════════════════════════════════════════════════════════════
    # ECTASIA path  (ECTASIA or COMPLEX branches)
    # ══════════════════════════════════════════════════════════════════════
    if shape.kind in ("ECTASIA", "COMPLEX"):
        et = cfg.ectasia_shape_thresh * mean_d
        res_maxs = _local_maxima_indices(d_res)

        for m in res_maxs:
            if d_res[m] <= et:
                continue

            RVD_at_m = max(float(d_trend[m]), branch_ref * 0.5)
            # Walk outward while vessel is still elevated
            thr = RVD_at_m + cfg.lesion_alpha * et
            L = m
            while L > 0 and d_s[L] > thr:
                L -= 1
            R = m
            while R < N - 1 and d_s[R] > thr:
                R += 1

            if (R - L + 1) < cfg.min_lesion_points:
                continue

            m_peak = L + int(np.argmax(d_s[L : R + 1]))
            peak_d = float(d_s[m_peak])
            ref_d  = max(float(d_trend[m_peak]), branch_ref * 0.5)
            if ref_d <= 1e-6:
                continue

            ectasia_ratio = (peak_d / ref_d - 1.0) * 100.0
            if ectasia_ratio < 5.0:
                continue

            length_px = arc_length(branch, L, R)
            entry_grad, exit_grad = _gradient_stats(d_s, L, m_peak, R)

            ect_sev = "MILD"
            if ectasia_ratio >= cfg.ectasia_severe_thresh * 100:
                ect_sev = "SEVERE"
            elif ectasia_ratio >= cfg.ectasia_mild_thresh * 100:
                ect_sev = "MODERATE"

            findings.append({
                "type":          "ECTASIA",
                "L_idx":         int(L),
                "R_idx":         int(R),
                "min_idx":       int(m_peak),
                "peak_idx":      int(m_peak),
                "MLD_px":        peak_d,     # semantically "peak" for ectasia
                "RVD_px":        ref_d,
                "DS_percent":    -ectasia_ratio,  # negative = widening
                "ectasia_ratio": ectasia_ratio,
                "length_px":     length_px,
                "MLD_mm":        None,
                "RVD_mm":        None,
                "length_mm":     None,
                "entry_grad":    entry_grad,
                "exit_grad":     exit_grad,
                "severity":      ect_sev,
                "min_pt":        branch[m_peak],
                "branch":        branch,
                "d_smooth":      d_s,
                "d_fine":        d_fine,
                "d_trend":       d_trend,
                "d_residual":    d_res,
                "shape_kind":    shape.kind,
                "branch_shape":  shape,
            })

    # ── Merge overlapping findings of each type separately ─────────────────
    stenoses = _merge_findings(
        [f for f in findings if f["type"] == "STENOSIS"], "DS_percent")
    ectasias = _merge_findings(
        [f for f in findings if f["type"] == "ECTASIA"], "ectasia_ratio")

    return stenoses + ectasias


# ==========================================================================
# Main pipeline
# ==========================================================================
def qca2_from_mask(bw_mask: np.ndarray, cfg: QCA2Config):
    """Shape-aware multi-branch QCA.

    Returns (branches, all_findings, dt, branch_shapes_map)
    where branch_shapes_map maps id(branch) → ProfileShape.
    """
    ys, xs = np.where(bw_mask > 0)
    if len(ys) == 0:
        return [], [], np.zeros_like(bw_mask, dtype=np.float32), {}

    y_min, y_max = int(ys.min()), int(ys.max())
    x_min, x_max = int(xs.min()), int(xs.max())
    pad = 10
    H, W = bw_mask.shape
    y1 = max(0, y_min - pad); y2 = min(H, y_max + pad + 1)
    x1 = max(0, x_min - pad); x2 = min(W, x_max + pad + 1)

    cropped = bw_mask[y1:y2, x1:x2]
    cr_skel = get_skeleton_from_mask(cropped, cfg)

    vessel_labels = _separate_touching_vessels(cropped)
    n_labels = int(vessel_labels.max())
    cr_dt = np.zeros(cropped.shape, dtype=np.float32)
    for vi in range(1, n_labels + 1):
        region = (vessel_labels == vi).astype(np.uint8)
        cr_dt = np.maximum(cr_dt, distance_transform_edt(region).astype(np.float32))

    skel = np.zeros_like(bw_mask)
    skel[y1:y2, x1:x2] = cr_skel
    dt = np.zeros_like(bw_mask, dtype=np.float32)
    dt[y1:y2, x1:x2] = cr_dt

    branches = extract_all_branches(skel, cfg.min_branch_len)
    if cfg.centerline_smooth_win > 1:
        branches = [smooth_branch_coords(b, cfg.centerline_smooth_win, H, W)
                    for b in branches]

    if not branches:
        try:
            cl = ordered_centerline_from_mask(bw_mask, cfg)
            branches = [cl]
        except RuntimeError:
            return [], [], dt, {}

    all_findings: list = []
    branch_shapes: Dict[int, ProfileShape] = {}

    for branch in branches:
        # Classify shape (needed for the plot even if no finding)
        N = len(branch)
        d_raw = np.array([2.0 * dt[y, x] for (y, x) in branch], dtype=np.float32)
        d_s   = smooth_1d(d_raw, cfg.smooth_win)
        branch_shapes[id(branch)] = analyze_profile_shape(d_s, cfg)

        findings = detect_findings_on_branch(branch, dt, cfg, bw_mask=bw_mask)
        all_findings.extend(findings)

    # Sort: STENOSIS by DS%, then ECTASIA by ectasia_ratio
    stenoses = sorted([f for f in all_findings if f["type"] == "STENOSIS"],
                      key=lambda z: z["DS_percent"], reverse=True)
    ectasias = sorted([f for f in all_findings if f["type"] == "ECTASIA"],
                      key=lambda z: z["ectasia_ratio"], reverse=True)
    all_findings = stenoses + ectasias

    return branches, all_findings, dt, branch_shapes


# ==========================================================================
# Visualization
# ==========================================================================
def draw_overlay_v2(angio: np.ndarray, mask: np.ndarray,
                    branches: List[List[Tuple[int, int]]],
                    findings: list,
                    branch_shapes: Dict[int, ProfileShape]) -> np.ndarray:
    """Overlay with shape labels, stenosis (red) and ectasia (cyan) markers."""
    if angio.ndim == 2:
        vis = cv2.cvtColor(angio, cv2.COLOR_GRAY2BGR)
    else:
        vis = angio.copy()

    # Vessel contour
    mask_f = (mask > 0).astype(np.float32)
    mask_smooth = cv2.GaussianBlur(mask_f, (0, 0), 1.0)
    contours, _ = cv2.findContours((mask_smooth > 0.5).astype(np.uint8),
                                   cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        cv2.polylines(vis, [cnt], True, (0, 255, 0), 1, cv2.LINE_AA)

    # Branch centerlines + shape label
    for bi, branch in enumerate(branches):
        if len(branch) < 2:
            continue
        color = _BRANCH_COLORS[bi % len(_BRANCH_COLORS)]
        pts = np.array([(x, y) for (y, x) in branch], dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(vis, [pts], False, color, 1, cv2.LINE_AA)
        by, bx = branch[len(branch) // 2]
        sh = branch_shapes.get(id(branch))
        label = f"B{bi+1}" + (f" [{sh.kind[0]}]" if sh else "")
        cv2.putText(vis, label, (bx + 5, by),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)

    # Finding markers
    for k, f in enumerate(findings[:8]):
        branch = f["branch"]
        L, R   = f["L_idx"], f["R_idx"]
        ftype  = f["type"]
        seg    = branch[L : min(R + 1, len(branch))]

        # Stenosis = red/orange; Ectasia = cyan/teal
        if ftype == "STENOSIS":
            seg_color = (0, 0, 255) if f["severity"] == "SEVERE" else (0, 140, 255)
            lbl_color = seg_color
            lbl = f"#{k+1} {f['DS_percent']:.1f}% DS ({f['severity'][0]})"
        else:
            seg_color = (200, 200, 0)   # teal
            lbl_color = (200, 200, 0)
            lbl = f"#{k+1} ECT +{f['ectasia_ratio']:.1f}% ({f['severity'][0]})"

        if len(seg) >= 2:
            pts = np.array([(x, y) for (y, x) in seg], dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(vis, [pts], False, seg_color, 3, cv2.LINE_AA)
        elif seg:
            y, x = seg[0]
            cv2.circle(vis, (x, y), 3, seg_color, -1, cv2.LINE_AA)

        y0, x0 = f["min_pt"]
        cv2.putText(vis, lbl, (x0 + 8, y0 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, lbl_color, 1, cv2.LINE_AA)

    return vis


def save_diameter_plot_v2(out_path: Path,
                          branches: List[List[Tuple[int, int]]],
                          dt: np.ndarray,
                          findings: list,
                          branch_shapes: Dict[int, ProfileShape],
                          cfg: QCA2Config):
    """Enhanced diameter plot.

    Each branch gets two panels:
      Top:    raw + smooth + linear trend + stenosis/ectasia spans
      Bottom: detrended residual + shape classification + zero line
    """
    n = len(branches)
    if n == 0:
        return

    fig, axes = plt.subplots(n, 2,
                             figsize=(12, 3.5 * n),
                             gridspec_kw={"width_ratios": [3, 1]},
                             squeeze=False)

    for bi, branch in enumerate(branches):
        ax_main = axes[bi, 0]
        ax_res  = axes[bi, 1]

        d_raw = np.array([2.0 * dt[y, x] for (y, x) in branch], dtype=np.float32)
        d_s   = smooth_1d(d_raw, cfg.smooth_win)
        shape = branch_shapes.get(id(branch))

        N = len(d_s)
        x_ax = np.arange(N)

        # ── Top panel ──────────────────────────────────────────────────────
        ax_main.plot(d_raw, color="silver", alpha=0.6, linewidth=0.8, label="raw")
        ax_main.plot(d_s,   color="steelblue", linewidth=1.3, label="smooth")

        if shape is not None:
            ax_main.plot(x_ax, shape.d_trend, color="dimgray", linewidth=1.1,
                         linestyle="--", label=f"trend (slope={shape.slope:+.3f})")

        # Findings on this branch
        branch_findings = [f for f in findings if f["branch"] is branch]
        for f in branch_findings:
            L, R, m = f["L_idx"], f["R_idx"], f["min_idx"]
            ftype   = f["type"]
            sev     = f["severity"]

            if ftype == "STENOSIS":
                span_c = {"SEVERE": "red", "MODERATE": "orange", "MILD": "gold"}.get(sev, "red")
                marker = "v"
                y_mark = d_s[m]
                entry_g = f.get("entry_grad", 0.0)
                exit_g  = f.get("exit_grad",  0.0)
                ann = (f"{f['DS_percent']:.1f}% DS {sev[0]}\n"
                       f"↓{abs(entry_g):.2f} ↑{exit_g:.2f} px/pt")
            else:
                span_c = "teal"
                marker = "^"
                y_mark = d_s[m]
                ann = (f"+{f['ectasia_ratio']:.1f}% ECT {sev[0]}\n"
                       f"peak {f['MLD_px']:.1f} px")

            ax_main.axvspan(L, R, alpha=0.18, color=span_c)
            ax_main.scatter([m], [y_mark], marker=marker, color=span_c, zorder=5, s=40)
            ax_main.annotate(ann, (m, y_mark),
                             textcoords="offset points", xytext=(5, 6),
                             fontsize=7, color=span_c)

            # Trend reference line across the finding span
            if shape is not None:
                ax_main.plot(x_ax[L:R+1], shape.d_trend[L:R+1],
                             color="green", linestyle=":", linewidth=1.0)

        sk = f"  [{shape.kind}]" if shape else ""
        ax_main.set_title(f"Branch {bi+1} ({N} pts){sk}", fontsize=9)
        ax_main.set_xlabel("Centerline index", fontsize=8)
        ax_main.set_ylabel("Diameter (px)", fontsize=8)
        ax_main.legend(fontsize=6, loc="upper right")

        # ── Right panel: detrended residual ────────────────────────────────
        if shape is not None:
            res = shape.d_residual
            res_color = "steelblue"
            ax_res.fill_between(x_ax, 0, res,
                                where=(res < 0), alpha=0.35, color="tomato",
                                label="narrowing")
            ax_res.fill_between(x_ax, 0, res,
                                where=(res > 0), alpha=0.35, color="teal",
                                label="widening")
            ax_res.plot(x_ax, res, color=res_color, linewidth=0.9)
            ax_res.axhline(0, color="black", linewidth=0.7, linestyle="--")

            # Threshold lines
            st = cfg.stenosis_shape_thresh * shape.mean_d
            et = cfg.ectasia_shape_thresh  * shape.mean_d
            ax_res.axhline(-st, color="red",  linewidth=0.6, linestyle=":",
                           label=f"−{cfg.stenosis_shape_thresh*100:.0f}%")
            ax_res.axhline(+et, color="teal", linewidth=0.6, linestyle=":",
                           label=f"+{cfg.ectasia_shape_thresh*100:.0f}%")

            ax_res.set_title("Residual", fontsize=8)
            ax_res.set_xlabel("idx", fontsize=7)
            ax_res.set_ylabel("px", fontsize=7)
            ax_res.legend(fontsize=6, loc="upper right")
            ax_res.tick_params(labelsize=6)

            # Colour the shape label background
            kind_c = {"STENOSIS": "#ffe0e0", "ECTASIA": "#e0f7f7",
                      "COMPLEX": "#fff0d0", "UNIFORM": "#f0f0f0",
                      "TAPERED": "#f0f0f0"}
            ax_res.set_facecolor(kind_c.get(shape.kind, "#f8f8f8"))
        else:
            ax_res.axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# ==========================================================================
# Batch + single runners
# ==========================================================================
def run_qca2_single(angio_path: str, mask_path: str, out_dir: str, cfg: QCA2Config):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(angio_path).stem

    angio    = cv2.imread(angio_path, cv2.IMREAD_GRAYSCALE)
    mask_img = cv2.imread(mask_path,  cv2.IMREAD_GRAYSCALE)
    if angio is None or mask_img is None:
        print("[FAIL] Could not read image or mask.")
        return

    bw = to_binary_mask(mask_img)
    bw = morph_cleanup(bw, cfg)

    try:
        branches, findings, dt, bshapes = qca2_from_mask(bw, cfg)
    except Exception as e:
        print(f"[FAIL] {stem}: {e}")
        return

    overlay = draw_overlay_v2(angio, bw, branches, findings, bshapes)
    cv2.imwrite(str(out / f"{stem}_overlay2.png"), overlay)
    save_diameter_plot_v2(out / f"{stem}_diameter2.png",
                          branches, dt, findings, bshapes, cfg)

    print(f"\n[OK] {stem}")
    print(f"  Branches: {len(branches)}")
    stenoses = [f for f in findings if f['type'] == 'STENOSIS']
    ectasias = [f for f in findings if f['type'] == 'ECTASIA']
    print(f"  Stenoses: {len(stenoses)}")
    for f in stenoses:
        bi = next((i for i, b in enumerate(branches) if b is f['branch']), -1)
        shape_lbl = bshapes.get(id(f['branch']))
        sk = shape_lbl.kind if shape_lbl else "?"
        print(f"    B{bi+1} [{sk}]  DS={f['DS_percent']:.1f}%  "
              f"({f['severity']})  "
              f"MLD={f['MLD_px']:.1f}px  RVD={f['RVD_px']:.1f}px  "
              f"entry↓{abs(f['entry_grad']):.3f} exit↑{f['exit_grad']:.3f}")
    print(f"  Ectasias: {len(ectasias)}")
    for f in ectasias:
        bi = next((i for i, b in enumerate(branches) if b is f['branch']), -1)
        shape_lbl = bshapes.get(id(f['branch']))
        sk = shape_lbl.kind if shape_lbl else "?"
        print(f"    B{bi+1} [{sk}]  ECT=+{f['ectasia_ratio']:.1f}%  "
              f"({f['severity']})  "
              f"peak={f['MLD_px']:.1f}px  ref={f['RVD_px']:.1f}px")

    print(f"  Saved: {out / f'{stem}_overlay2.png'}")
    print(f"  Saved: {out / f'{stem}_diameter2.png'}")


def run_qca2_batch(angiogram_dir: str, mask_dir: str, out_dir: str,
                   cfg: QCA2Config,
                   angio_exts=(".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
    angio_dirp = Path(angiogram_dir)
    mask_dirp  = Path(mask_dir)
    out        = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    angio_files: list = []
    for ext in angio_exts:
        angio_files += list(angio_dirp.glob(f"*{ext}"))
    if not angio_files:
        raise RuntimeError(f"No images found in: {angiogram_dir}")

    report_csv = out / "qca2_report.csv"
    fieldnames = [
        "file", "branch", "shape", "finding_type", "rank",
        "severity", "DS_percent", "ectasia_ratio",
        "MLD_px", "RVD_px", "length_px",
        "entry_grad", "exit_grad",
        "L_idx", "R_idx", "min_idx",
    ]
    with open(report_csv, "w", newline="", encoding="utf-8") as csvf:
        writer = csv.DictWriter(csvf, fieldnames=fieldnames)
        writer.writeheader()

        for img_path in sorted(angio_files):
            stem = img_path.stem
            mask_path = None
            for ext in angio_exts:
                cand = mask_dirp / f"{stem}{ext}"
                if cand.exists():
                    mask_path = cand
                    break
            if mask_path is None:
                for ext in angio_exts:
                    cand = mask_dirp / f"{stem}_mask{ext}"
                    if cand.exists():
                        mask_path = cand
                        break
            if mask_path is None:
                print(f"[SKIP] No mask for {img_path.name}")
                continue

            angio    = cv2.imread(str(img_path),   cv2.IMREAD_GRAYSCALE)
            mask_img = cv2.imread(str(mask_path),  cv2.IMREAD_GRAYSCALE)
            if angio is None or mask_img is None:
                print(f"[SKIP] Cannot read {img_path.name}")
                continue

            bw = to_binary_mask(mask_img)
            bw = morph_cleanup(bw, cfg)

            try:
                branches, findings, dt, bshapes = qca2_from_mask(bw, cfg)
            except Exception as e:
                print(f"[FAIL] {img_path.name}: {e}")
                continue

            overlay = draw_overlay_v2(angio, bw, branches, findings, bshapes)
            cv2.imwrite(str(out / f"{stem}_overlay2.png"), overlay)
            save_diameter_plot_v2(out / f"{stem}_diameter2.png",
                                  branches, dt, findings, bshapes, cfg)

            for rank, f in enumerate(findings[:8], start=1):
                bi = next((i for i, b in enumerate(branches) if b is f['branch']), -1)
                sh = bshapes.get(id(f['branch']))
                writer.writerow({
                    "file":          img_path.name,
                    "branch":        f"B{bi+1}",
                    "shape":         sh.kind if sh else "?",
                    "finding_type":  f["type"],
                    "rank":          rank,
                    "severity":      f["severity"],
                    "DS_percent":    f"{f['DS_percent']:.3f}",
                    "ectasia_ratio": f"{f['ectasia_ratio']:.3f}",
                    "MLD_px":        f"{f['MLD_px']:.3f}",
                    "RVD_px":        f"{f['RVD_px']:.3f}",
                    "length_px":     f"{f['length_px']:.3f}",
                    "entry_grad":    f"{f['entry_grad']:.4f}",
                    "exit_grad":     f"{f['exit_grad']:.4f}",
                    "L_idx":         f["L_idx"],
                    "R_idx":         f["R_idx"],
                    "min_idx":       f["min_idx"],
                })

            print(f"[OK] {img_path.name}: "
                  f"{len([f for f in findings if f['type']=='STENOSIS'])} stenosis, "
                  f"{len([f for f in findings if f['type']=='ECTASIA'])} ectasia")

    print(f"\nReport: {report_csv}")
    print(f"Output: {out}")


# ==========================================================================
# CLI entry point
# ==========================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="qca2: shape-aware QCA (stenosis + ectasia detection)")
    sub = parser.add_subparsers(dest="cmd")

    sp = sub.add_parser("single", help="Process one image")
    sp.add_argument("angio")
    sp.add_argument("mask")
    sp.add_argument("--out", default="qca2_out")

    bp = sub.add_parser("batch", help="Process a directory of images")
    bp.add_argument("angio_dir")
    bp.add_argument("mask_dir")
    bp.add_argument("--out", default="qca2_out")

    args = parser.parse_args()
    cfg  = QCA2Config()

    if args.cmd == "single":
        run_qca2_single(args.angio, args.mask, args.out, cfg)
    elif args.cmd == "batch":
        run_qca2_batch(args.angio_dir, args.mask_dir, args.out, cfg)
    else:
        parser.print_help()
