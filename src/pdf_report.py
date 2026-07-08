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
from report_engine import (
    AngleResult, FrameRecord, LesionTrack, draw_angle_summary_bgr, generate_reasoning,
    merge_cross_view_lesions,
)

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

    Clinical-only presentation: for each distinct, cross-view-deduplicated
    lesion, a dedicated 4-panel explainable detail page (raw crop, centerline
    + diameter crop, width heatmap, local diameter profile in mm) sourced
    from that lesion's single highest-score/confidence detection, plus each
    view's key frames for vessel-tree context -- with an explicit
    AI-generated disclaimer.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Each view is analyzed completely independently, so the same real lesion
    # commonly gets (re)detected across several views/runs at slightly
    # different apparent severity -- merge those into one finding per real
    # lesion (highest reading wins) before it ever reaches the summary table,
    # rather than listing every independent view's raw detection as if it
    # were a separate finding.
    merged_lesions = merge_cross_view_lesions(view_summaries)

    any_localization = any(vs.get("has_localization") for vs in view_summaries)
    all_localization = all(vs.get("has_localization") for vs in view_summaries) if view_summaries else True

    with PdfPages(out_path) as pdf:
        _add_diagnosis_title_page(pdf, patient_info, view_summaries, merged_lesions)

        _add_lesion_detail_pages(pdf, merged_lesions)

        for vs in view_summaries:
            _add_diagnosis_key_frame_pages(pdf, vs)

        _add_methodology_page(pdf, cfg, any_localization, all_localization, include_cross_view_note=True)

    return out_path


def _add_diagnosis_title_page(pdf, patient_info: Dict[str, str], view_summaries: List[dict],
                              merged_lesions: List[dict]):
    """
    merged_lesions: one entry per real, cross-view-deduplicated finding (see
    report_engine.merge_cross_view_lesions) -- the same anatomical location
    detected independently by more than one view is listed once, at its
    highest-severity reading, with "n_views" recording how many views
    corroborated it.
    """
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

    if not merged_lesions:
        ax.text(0, y, "Impression: No significant stenosis detected across analyzed views.",
                transform=ax.transAxes, fontsize=11, va="top")
        y -= 0.045
    else:
        worst = merged_lesions[0]
        n_raw = sum(len(vs.get("lesions", [])) for vs in view_summaries)
        impression = (
            f"Impression: {len(merged_lesions)} distinct lesion(s) identified across "
            f"{len(view_summaries)} view(s) ({n_raw} raw detections; the same lesion independently "
            f"caught by more than one view is merged, highest reading kept). Most severe: "
            f"{worst['DS_percent']:.1f}% DS ({worst['severity']}) in {worst['label']} "
            f"({worst['view_label']})."
        )
        for wrapped in textwrap.wrap(impression, width=92):
            ax.text(0, y, wrapped, transform=ax.transAxes, fontsize=11, va="top")
            y -= 0.045

    if merged_lesions:
        y -= 0.03
        ax.text(0, y, "Stenosis Summary (most to least severe):", transform=ax.transAxes,
                fontsize=12, fontweight="bold", va="top")
        y -= 0.045

        col_x = [0.0, 0.20, 0.55, 0.68, 0.80, 0.90]
        headers = ["Best View", "Location", "Severity", "DS%", "Confidence", "Views"]
        for cx, h in zip(col_x, headers):
            ax.text(cx, y, h, transform=ax.transAxes, fontsize=9, fontweight="bold", va="top")
        y -= 0.032

        def _trunc(s, n):
            return s if len(s) <= n else s[:n - 1] + "…"

        for m in merged_lesions:
            if y < 0.05:
                ax.text(0, y, "... see per-view pages for the remainder.",
                        transform=ax.transAxes, fontsize=8.5, style="italic", va="top")
                break
            row = [
                _trunc(m["view_label"], 16), _trunc(m["label"], 26), m["severity"],
                f"{m['DS_percent']:.1f}%", f"{m.get('confidence', 0):.2f}", f"×{m['n_views']}",
            ]
            color = _SEVERITY_RGB_MPL.get(m["severity"], "black")
            for cx, val in zip(col_x, row):
                ax.text(cx, y, val, transform=ax.transAxes, fontsize=8.5, va="top", color=color)
            y -= 0.028

    pdf.savefig(fig)
    plt.close(fig)


def _add_lesion_detail_pages(pdf, merged_lesions: List[dict]):
    """
    One page per distinct, cross-view-deduplicated lesion (most to least
    severe), embedding the pre-rendered 4-panel explainable figure (raw crop,
    centerline + diameter crop, width heatmap, local diameter profile in mm)
    saved at analysis time (analysis_results_store.save_view_results) from
    that lesion's single highest-score/confidence detection. Falls back to an
    explanatory note for lesions saved before this feature existed, whose
    view has no saved detail image on disk.
    """
    for m in merged_lesions:
        fig = plt.figure(figsize=(8.5, 11))
        corroboration = f" — corroborated by {m['n_views']} view(s)" if m["n_views"] > 1 else ""
        title = f"{m['label']} — {m['severity']} ({m['DS_percent']:.1f}% DS) — {m['view_label']}{corroboration}"
        wrapped_title = "\n".join(textwrap.wrap(title, width=62))
        fig.suptitle(wrapped_title, fontsize=12.5, fontweight="bold", y=0.98)

        view_dir = m.get("_view_dir")
        detail_image = m.get("detail_image")
        image_path = Path(view_dir) / detail_image if view_dir and detail_image else None

        if image_path is not None and image_path.exists():
            # The saved figure (full-frame context row + cropped detail row)
            # is roughly a fixed aspect -- size the axes box to it so it fills
            # the page width without leaving a large dead band above/below it.
            ax = fig.add_axes((0.05, 0.30, 0.90, 0.40))
            ax.imshow(plt.imread(str(image_path)))
            ax.axis("off")
        else:
            ax = fig.add_axes((0.1, 0.30, 0.8, 0.60))
            ax.axis("off")
            ax.text(
                0.5, 0.5,
                "Detail imagery unavailable for this lesion (saved before this feature was "
                "added) — re-run and re-save this view's analysis to generate it.",
                ha="center", va="center", fontsize=11, wrap=True,
            )

        pdf.savefig(fig)
        plt.close(fig)


def _add_diagnosis_key_frame_pages(pdf, view_summary: dict):
    """
    Adds pages showing every key frame saved for this view (see
    analysis_results_store.save_view_results / report_engine.select_key_frames)
    -- the smallest set of frames that together guarantee every SEVERE/
    SIGNIFICANT lesion in this view is visually depicted at least once, for
    vessel-tree context alongside each lesion's own detail page
    (_add_lesion_detail_pages).

    Two key frames per page (raw | AI-labeled pairs, stacked), sized down
    from the overview page's single full-page image so both fit legibly.
    """
    view_dir = view_summary.get("_view_dir")
    key_frames = view_summary.get("key_frame_images") or []
    if not view_dir or not key_frames:
        return

    per_page = 2
    row_tops = [0.53, 0.08]
    row_height = 0.40

    for page_start in range(0, len(key_frames), per_page):
        page_frames = key_frames[page_start:page_start + per_page]
        fig = plt.figure(figsize=(8.5, 11))
        fig.suptitle(
            f"{view_summary['view_label']} — Key Frames "
            f"({page_start + 1}-{page_start + len(page_frames)} of {len(key_frames)})",
            fontsize=13, fontweight="bold", y=0.975,
        )

        for row_i, kf in enumerate(page_frames):
            top = row_tops[row_i]
            annotated_path = Path(view_dir) / kf["image"]
            if not annotated_path.exists():
                continue
            raw_name = kf.get("raw_image")
            raw_path = Path(view_dir) / raw_name if raw_name else None
            has_raw = raw_path is not None and raw_path.exists()

            fig.text(0.5, top + row_height + 0.02, f"Frame {kf['frame_idx'] + 1}",
                     ha="center", fontsize=10, fontweight="bold")

            if has_raw:
                ax_raw = fig.add_axes((0.06, top, 0.42, row_height))
                ax_raw.imshow(plt.imread(str(raw_path)))
                ax_raw.axis("off")
                ax_raw.set_title("Raw", fontsize=9)

                ax_ann = fig.add_axes((0.52, top, 0.42, row_height))
                ax_ann.imshow(plt.imread(str(annotated_path)))
                ax_ann.axis("off")
                ax_ann.set_title("AI-Labeled", fontsize=9)
            else:
                ax_ann = fig.add_axes((0.22, top, 0.56, row_height))
                ax_ann.imshow(plt.imread(str(annotated_path)))
                ax_ann.axis("off")
                ax_ann.set_title("AI-Labeled (raw comparison unavailable for this saved result)",
                                 fontsize=8.5)

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


def _add_methodology_page(pdf, cfg: QCAConfig, any_localization: bool, all_localization: bool,
                          include_cross_view_note: bool = False):
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
        "Each lesion's reported measurement is the most severe reading among its reliably-",
        "measured detections (not simply its single highest-confidence one) across the whole",
        "cine run, so a lesion's true worst narrowing is reported even if it wasn't captured",
        "in the run's single cleanest-measured frame.",
        "",
        "Each distinct lesion is shown on its own detail page: the full frame (raw, and with",
        "the vessel mask and this lesion's location marked) for whole-vessel context, plus a",
        "tight crop around the lesion -- raw, with centerline and diameter markers, a width",
        "heatmap, and the local diameter profile -- all drawn from that lesion's single",
        "highest-score/confidence detection. Diameters (MLD, RVD) are reported in millimeters",
        "using a 0.20 mm/pixel calibration factor (JACIT consensus default for the standard",
        "catheter/isocenter technique) unless a case-specific calibration was supplied.",
        "",
        "Each view's own Key Frames page(s) additionally show that view's vessel tree at a",
        "small set of frames chosen to guarantee every SEVERE/SIGNIFICANT finding in the",
        "view is visually captured at least once.",
        "",
    ]

    if include_cross_view_note:
        lines += [
            "Each angiographic view is analyzed independently, so the same real lesion is",
            "often (re)detected by more than one view at a slightly different apparent",
            "severity. The Stenosis Summary reports each anatomical location once, at its",
            "highest-severity reading across all views that detected it (\"Views\" column =",
            "how many views corroborated that finding) -- see each view's own pages for its",
            "individual, un-merged reading.",
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
