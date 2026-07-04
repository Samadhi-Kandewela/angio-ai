"""
Assembles a clinical QCA report PDF from one or more analyzed AngleResults.

Uses matplotlib's PdfPages so no new dependency (reportlab/fpdf) is needed —
matplotlib is already used throughout qca.py for the existing explainable
per-lesion figures, which this module reuses directly via build_lesion_figure.
"""

import datetime
import textwrap
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from qca import QCAConfig, build_lesion_figure, _BRANCH_COLORS, _SEVERITY_BGR, _SEVERITY_RADIUS
from report_engine import AngleResult, FrameRecord, LesionTrack, generate_reasoning

_SEVERITY_RGB_MPL = {
    "SEVERE": "#D62728",
    "MODERATE": "#FF7F0E",
    "MILD": "#BCBD22",
}


def render_clinical_report(out_path, patient_info: Dict[str, str],
                           angle_results: List[AngleResult], cfg: QCAConfig) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    all_tracks = [(ar, t) for ar in angle_results for t in ar.tracks]
    all_tracks.sort(key=lambda pair: pair[1].representative["DS_percent"], reverse=True)

    any_localization = any(ar.has_localization for ar in angle_results)
    all_localization = all(ar.has_localization for ar in angle_results) if angle_results else True

    with PdfPages(out_path) as pdf:
        _add_title_page(pdf, patient_info, angle_results, all_tracks)

        for ar in angle_results:
            if ar.tracks and ar.summary_frame_idx is not None:
                _add_angle_summary_page(pdf, ar)

        for ar, t in all_tracks:
            _add_lesion_page(pdf, ar, t, cfg)

        _add_methodology_page(pdf, cfg, any_localization, all_localization)

    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# Summary-frame overlay (uses each track's representative DS%/severity, not
# this single frame's own noisy value, so numbers match the per-lesion pages)
# ─────────────────────────────────────────────────────────────────────────────
def _draw_angle_summary_bgr(rec: FrameRecord, tracks: List[LesionTrack]) -> np.ndarray:
    vis = cv2.cvtColor(rec.img_gray, cv2.COLOR_GRAY2BGR)

    contours, _ = cv2.findContours((rec.bw_mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, contours, -1, (0, 255, 0), 1)

    for bi, branch in enumerate(rec.branches):
        color = _BRANCH_COLORS[bi % len(_BRANCH_COLORS)]
        for (y, x) in branch:
            vis[y, x] = color

    track_of_lesion = {id(les): t for t in tracks for les in t.detections}

    for les in rec.lesions:
        t = track_of_lesion.get(id(les))
        if t is None:
            continue
        rep = t.representative
        sev = rep["severity"]
        if sev not in ("SEVERE", "MODERATE"):
            continue

        color = _SEVERITY_BGR.get(sev, _SEVERITY_BGR["MILD"])
        radius = _SEVERITY_RADIUS.get(sev, 1)
        branch = les["branch"]
        L, R = les["L_idx"], les["R_idx"]
        for i in range(L, min(R + 1, len(branch))):
            y, x = branch[i]
            cv2.circle(vis, (x, y), radius, color, -1)

        y0, x0 = les["min_pt"]
        occ = " [OCC]" if rep.get("total_occlusion") else ""
        label = f"{t.track_id} {rep['DS_percent']:.1f}% {sev[:3]}{occ}"
        font_scale = 0.45 if sev == "SEVERE" else 0.38
        cv2.putText(vis, label, (x0 + 6, y0 - 4), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1, cv2.LINE_AA)

    return vis


# ─────────────────────────────────────────────────────────────────────────────
# Pages
# ─────────────────────────────────────────────────────────────────────────────
def _add_title_page(pdf, patient_info, angle_results: List[AngleResult], all_tracks):
    fig = plt.figure(figsize=(8.5, 11))
    fig.suptitle("AI-Assisted Coronary Angiography QCA Report", fontsize=18, fontweight="bold", y=0.97)

    ax = fig.add_axes((0.08, 0.08, 0.84, 0.82))
    ax.axis("off")

    view_names = ", ".join(ar.angle_label for ar in angle_results) if angle_results else "none"
    lines = [
        f"Patient ID: {patient_info.get('patient_id') or 'Not specified'}",
        f"Study Date: {patient_info.get('study_date') or 'Not specified'}",
        f"Operator: {patient_info.get('operator') or 'Not specified'}",
        f"Report generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Angiographic views analyzed: {len(angle_results)} ({view_names})",
        "",
    ]

    if not all_tracks:
        lines.append("Impression: No significant stenosis detected across analyzed views.")
    else:
        worst_ar, worst_t = all_tracks[0]
        worst = worst_t.representative
        impression = (
            f"Impression: {len(all_tracks)} lesion(s) detected across {len(angle_results)} view(s). "
            f"Most severe: {worst['DS_percent']:.1f}% DS ({worst['severity']}) in {worst_t.label} "
            f"({worst_ar.angle_label})."
        )
        lines.extend(textwrap.wrap(impression, width=92))

    y = 0.98
    for line in lines:
        ax.text(0, y, line, transform=ax.transAxes, fontsize=11, va="top")
        y -= 0.045

    if all_tracks:
        y -= 0.03
        ax.text(0, y, "Lesion Summary (most to least severe):", transform=ax.transAxes,
                fontsize=12, fontweight="bold", va="top")
        y -= 0.045

        col_x = [0.0, 0.07, 0.24, 0.58, 0.70, 0.80, 0.90]
        headers = ["#", "View", "Location", "Severity", "DS%", "MLD", "Conf."]
        for cx, h in zip(col_x, headers):
            ax.text(cx, y, h, transform=ax.transAxes, fontsize=9, fontweight="bold", va="top")
        y -= 0.032

        def _trunc(s, n):
            return s if len(s) <= n else s[:n - 1] + "…"

        for ar, t in all_tracks:
            if y < 0.05:
                ax.text(0, y, "... see individual lesion pages for the remainder.",
                        transform=ax.transAxes, fontsize=8.5, style="italic", va="top")
                break
            rep = t.representative
            mld = f"{rep['MLD_mm']:.2f}mm" if rep.get("MLD_mm") is not None else f"{rep['MLD_px']:.1f}px"
            row = [
                t.track_id, _trunc(ar.angle_label, 12), _trunc(t.label, 22), rep["severity"],
                f"{rep['DS_percent']:.1f}%", mld, f"{rep.get('confidence', 0):.2f}",
            ]
            color = _SEVERITY_RGB_MPL.get(rep["severity"], "black")
            for cx, val in zip(col_x, row):
                ax.text(cx, y, val, transform=ax.transAxes, fontsize=8.5, va="top", color=color)
            y -= 0.028

    pdf.savefig(fig)
    plt.close(fig)


def _add_angle_summary_page(pdf, ar: AngleResult):
    rec = ar.get_frame_record(ar.summary_frame_idx)
    if rec is None:
        return

    vis_rgb = cv2.cvtColor(_draw_angle_summary_bgr(rec, ar.tracks), cv2.COLOR_BGR2RGB)

    fig = plt.figure(figsize=(8.5, 11))
    ax_img = fig.add_axes((0.06, 0.20, 0.88, 0.68))
    ax_img.imshow(vis_rgb)
    ax_img.axis("off")
    ax_img.set_title(f"{ar.angle_label} — Stenosis Overview", fontsize=14, fontweight="bold")

    shown = sum(
        1 for t in ar.tracks
        if t.representative["severity"] in ("SEVERE", "MODERATE") and rec.frame_idx in t.frame_indices
    )
    caption = (
        f"{shown} of {len(ar.tracks)} detected lesion(s) in this view are co-visible in this frame; "
        "any remaining lesions are documented on their own pages."
    )
    fig.text(0.5, 0.155, caption, ha="center", fontsize=8.5, style="italic")

    ax_legend = fig.add_axes((0.06, 0.03, 0.88, 0.10))
    ax_legend.set_xlim(0, 1)
    ax_legend.set_ylim(0, 1)
    ax_legend.axis("off")
    for i, (label, color) in enumerate([("SEVERE (≥70% DS)", "#D62728"), ("MODERATE (50–69% DS)", "#FF7F0E")]):
        ax_legend.scatter([0.05 + i * 0.4], [0.5], color=color, s=80)
        ax_legend.text(0.09 + i * 0.4, 0.5, label, fontsize=9, va="center")

    pdf.savefig(fig)
    plt.close(fig)


def _add_lesion_page(pdf, ar: AngleResult, t: LesionTrack, cfg: QCAConfig):
    les = t.representative
    rec = ar.get_frame_record(les["frame_idx"])
    if rec is None:
        return

    title = f"{t.track_id} — {ar.angle_label} — {t.label}"
    fig = build_lesion_figure(rec.img_gray, rec.bw_mask, rec.dt, les, cfg, title)

    reasoning = generate_reasoning(t, cfg)
    fig.subplots_adjust(bottom=0.24)
    fig.text(0.02, 0.02, "Reasoning: " + reasoning, wrap=True, fontsize=9, va="bottom")

    pdf.savefig(fig)
    plt.close(fig)


def _add_methodology_page(pdf, cfg: QCAConfig, any_localization: bool, all_localization: bool):
    fig = plt.figure(figsize=(8.5, 11))
    fig.suptitle("Methodology & Limitations", fontsize=16, fontweight="bold", y=0.96)
    ax = fig.add_axes((0.08, 0.08, 0.84, 0.82))
    ax.axis("off")

    lines = [
        "This report was generated by an AI-assisted Quantitative Coronary Analysis",
        "(QCA) pipeline: deep-learning vessel segmentation, skeleton-based centerline",
        "extraction, per-branch lesion detection, and AHA/SYNTAX anatomical localization.",
        "",
        f"Severity thresholds follow the ACC/AHA/ESC consensus: Severe ≥{cfg.severe_threshold:.0f}%",
        f"diameter stenosis (DS), Moderate ≥{cfg.moderate_threshold:.0f}% DS, below that classified",
        "Mild / non-obstructive.",
        "",
        "Each lesion's reported measurement is taken from the single video frame where",
        "that lesion's detection confidence (edge sharpness + reference-segment quality +",
        "lesion-span stability) was highest across the whole cine run, not an arbitrary frame.",
        "",
        "Per-view overview diagrams show only lesions that are simultaneously visible in one",
        "representative frame; lesions not co-visible with others in that frame are still fully",
        "documented on their own page but may be absent from the overview image.",
        "",
    ]

    if not all_localization:
        if any_localization:
            lines += [
                "Note: anatomical localization (AHA/SYNTAX segment naming) was only available",
                "for some views; lesions in the remaining views are grouped by image position",
                "only, which is less robust to frame-to-frame motion.",
            ]
        else:
            lines += [
                "Note: no anatomical localization model was supplied. Lesions are grouped by",
                "image position only, which is less robust to frame-to-frame motion.",
            ]
        lines.append("")

    lines += [
        "This report is AI-assisted and intended to support, not replace, clinical",
        "judgement. All findings should be corroborated by a qualified clinician.",
    ]

    y = 0.98
    for line in lines:
        ax.text(0, y, line, transform=ax.transAxes, fontsize=10.5, va="top")
        y -= 0.032

    pdf.savefig(fig)
    plt.close(fig)
