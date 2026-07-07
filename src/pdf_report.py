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

from qca import QCAConfig, build_lesion_figure
from report_engine import AngleResult, FrameRecord, LesionTrack, draw_angle_summary_bgr, generate_reasoning

_SEVERITY_RGB_MPL = {
    "SEVERE": "#D62728",
    "SIGNIFICANT": "#FF7F0E",
    "MODERATE": "#DBB40C",
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


def render_clinical_diagnosis_report(out_path, patient_info: Dict[str, str],
                                     view_summaries: List[dict], cfg: QCAConfig) -> Path:
    """
    Final cross-view clinical diagnosis report, assembled from saved
    (JSON + PNG) per-view results (see analysis_results_store.py) rather than
    live AngleResults -- so it can be (re)generated at any time after each
    view has been analyzed and saved, independent of the analysis session
    that produced them.

    Clinical-only presentation: per-view stenosis location/severity diagrams
    and a combined summary table -- no explainable-AI lesion detail (crops,
    heatmaps, reasoning) -- with an explicit AI-generated disclaimer.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    all_lesions = [(vs, les) for vs in view_summaries for les in vs.get("lesions", [])]
    all_lesions.sort(key=lambda pair: pair[1]["DS_percent"], reverse=True)

    any_localization = any(vs.get("has_localization") for vs in view_summaries)
    all_localization = all(vs.get("has_localization") for vs in view_summaries) if view_summaries else True

    with PdfPages(out_path) as pdf:
        _add_diagnosis_title_page(pdf, patient_info, view_summaries, all_lesions)

        for vs in view_summaries:
            if vs.get("lesions") and vs.get("summary_image"):
                _add_diagnosis_view_page(pdf, vs)
            _add_diagnosis_key_frames_pages(pdf, vs)

        _add_methodology_page(pdf, cfg, any_localization, all_localization)

    return out_path


def _add_diagnosis_title_page(pdf, patient_info: Dict[str, str], view_summaries: List[dict], all_lesions):
    fig = plt.figure(figsize=(8.5, 11))
    fig.suptitle("Coronary Angiography Clinical Diagnosis Report", fontsize=18, fontweight="bold", y=0.97)

    ax = fig.add_axes((0.08, 0.08, 0.84, 0.82))
    ax.axis("off")

    ax.text(0, 0.995, "⚠ AI-GENERATED ANALYSIS — for clinical correlation, not a standalone diagnosis",
            transform=ax.transAxes, fontsize=10.5, fontweight="bold", color="#D62728", va="top")

    view_names = ", ".join(vs["view_label"] for vs in view_summaries) if view_summaries else "none"
    lines = [
        f"Patient ID: {patient_info.get('patient_id') or 'Not specified'}",
        f"Patient Name: {patient_info.get('full_name') or 'Not specified'}",
        f"Study Date: {patient_info.get('study_date') or 'Not specified'}",
        f"Operator: {patient_info.get('operator') or 'Not specified'}",
        f"Report generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Angiographic views analyzed: {len(view_summaries)} ({view_names})",
        "",
    ]

    y = 0.93
    for line in lines:
        ax.text(0, y, line, transform=ax.transAxes, fontsize=11, va="top")
        y -= 0.045

    if not all_lesions:
        ax.text(0, y, "Impression: No significant stenosis detected across analyzed views.",
                transform=ax.transAxes, fontsize=11, va="top")
        y -= 0.045
    else:
        worst_vs, worst_les = all_lesions[0]
        impression = (
            f"Impression: {len(all_lesions)} lesion(s) detected across {len(view_summaries)} view(s). "
            f"Most severe: {worst_les['DS_percent']:.1f}% DS ({worst_les['severity']}) in "
            f"{worst_les['label']} ({worst_vs['view_label']})."
        )
        for wrapped in textwrap.wrap(impression, width=92):
            ax.text(0, y, wrapped, transform=ax.transAxes, fontsize=11, va="top")
            y -= 0.045

    if all_lesions:
        y -= 0.03
        ax.text(0, y, "Stenosis Summary (most to least severe):", transform=ax.transAxes,
                fontsize=12, fontweight="bold", va="top")
        y -= 0.045

        col_x = [0.0, 0.20, 0.55, 0.68, 0.80]
        headers = ["View", "Location", "Severity", "DS%", "Confidence"]
        for cx, h in zip(col_x, headers):
            ax.text(cx, y, h, transform=ax.transAxes, fontsize=9, fontweight="bold", va="top")
        y -= 0.032

        def _trunc(s, n):
            return s if len(s) <= n else s[:n - 1] + "…"

        for vs, les in all_lesions:
            if y < 0.05:
                ax.text(0, y, "... see per-view pages for the remainder.",
                        transform=ax.transAxes, fontsize=8.5, style="italic", va="top")
                break
            row = [
                _trunc(vs["view_label"], 16), _trunc(les["label"], 26), les["severity"],
                f"{les['DS_percent']:.1f}%", f"{les.get('confidence', 0):.2f}",
            ]
            color = _SEVERITY_RGB_MPL.get(les["severity"], "black")
            for cx, val in zip(col_x, row):
                ax.text(cx, y, val, transform=ax.transAxes, fontsize=8.5, va="top", color=color)
            y -= 0.028

    pdf.savefig(fig)
    plt.close(fig)


def _add_diagnosis_view_page(pdf, view_summary: dict):
    view_dir = view_summary.get("_view_dir")
    image_name = view_summary.get("summary_image")
    if not view_dir or not image_name:
        return
    image_path = Path(view_dir) / image_name
    if not image_path.exists():
        return

    img = plt.imread(str(image_path))

    fig = plt.figure(figsize=(8.5, 11))
    ax_img = fig.add_axes((0.06, 0.20, 0.88, 0.68))
    ax_img.imshow(img)
    ax_img.axis("off")
    ax_img.set_title(f"{view_summary['view_label']} — Stenosis Location & Severity",
                     fontsize=14, fontweight="bold")

    lesions = view_summary.get("lesions", [])
    shown = sum(1 for les in lesions if les.get("co_visible_in_summary_frame"))
    has_key_frames = bool(view_summary.get("key_frame_images"))
    caption = (
        f"{shown} of {len(lesions)} detected lesion(s) in this view are co-visible in this diagram; "
        + ("every lesion is also pictured individually in the key frames that follow."
           if has_key_frames else "all are listed in the summary table.")
    )
    fig.text(0.5, 0.155, caption, ha="center", fontsize=8.5, style="italic")

    ax_legend = fig.add_axes((0.06, 0.03, 0.88, 0.10))
    ax_legend.set_xlim(0, 1)
    ax_legend.set_ylim(0, 1)
    ax_legend.axis("off")
    for i, (label, color) in enumerate([("SEVERE (≥70% DS)", "#D62728"), ("SIGNIFICANT (50–69% DS)", "#FF7F0E")]):
        ax_legend.scatter([0.05 + i * 0.4], [0.5], color=color, s=80)
        ax_legend.text(0.09 + i * 0.4, 0.5, label, fontsize=9, va="center")

    pdf.savefig(fig)
    plt.close(fig)


def _add_diagnosis_key_frames_pages(pdf, view_summary: dict):
    """
    Embeds every saved key-frame image for this view (see
    report_engine.select_key_frames / analysis_results_store.save_view_results)
    -- one circle+id marker per lesion, already burned into each PNG -- so the
    final combined report gives visual evidence for every lesion in the
    summary table, not just whichever ones happened to be co-visible in the
    single overview diagram from _add_diagnosis_view_page. Tiles up to 6
    images per page (2 columns x 3 rows), spilling onto extra pages if a view
    has more key frames than that.
    """
    view_dir = view_summary.get("_view_dir")
    key_frame_images = view_summary.get("key_frame_images") or []
    if not view_dir or not key_frame_images:
        return

    cols, rows = 2, 3
    per_page = cols * rows
    for page_start in range(0, len(key_frame_images), per_page):
        page_items = key_frame_images[page_start:page_start + per_page]

        fig = plt.figure(figsize=(8.5, 11))
        fig.suptitle(f"{view_summary['view_label']} — Key Frames (evidence for each finding)",
                    fontsize=13, fontweight="bold", y=0.97)

        cell_w, cell_h = 0.90 / cols, 0.84 / rows
        for i, item in enumerate(page_items):
            image_path = Path(view_dir) / item["image"]
            if not image_path.exists():
                continue
            img = plt.imread(str(image_path))

            r, c = divmod(i, cols)
            ax = fig.add_axes((
                0.05 + c * (0.94 / cols), 0.06 + (rows - 1 - r) * (0.88 / rows),
                cell_w, cell_h,
            ))
            ax.imshow(img)
            ax.axis("off")
            ax.set_title(f"frame {item['frame_idx']}", fontsize=9)

        pdf.savefig(fig)
        plt.close(fig)


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

    vis_rgb = cv2.cvtColor(draw_angle_summary_bgr(rec, ar.tracks), cv2.COLOR_BGR2RGB)

    fig = plt.figure(figsize=(8.5, 11))
    ax_img = fig.add_axes((0.06, 0.20, 0.88, 0.68))
    ax_img.imshow(vis_rgb)
    ax_img.axis("off")
    ax_img.set_title(f"{ar.angle_label} — Stenosis Overview", fontsize=14, fontweight="bold")

    shown = sum(
        1 for t in ar.tracks
        if t.representative["severity"] in ("SEVERE", "SIGNIFICANT") and rec.frame_idx in t.frame_indices
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
    for i, (label, color) in enumerate([("SEVERE (≥70% DS)", "#D62728"), ("SIGNIFICANT (50–69% DS)", "#FF7F0E")]):
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
        f"Severity follows the JACIT/ARC-2 hierarchical consensus: Severe ≥{cfg.severe_threshold:.0f}%",
        f"diameter stenosis (DS) regardless of symptoms; Significant ≥{cfg.significant_threshold:.0f}% DS,",
        f"actionable if symptomatic or a positive functional test; Moderate ≥{cfg.moderate_threshold:.0f}% DS;",
        "below that classified Mild / non-obstructive.",
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
