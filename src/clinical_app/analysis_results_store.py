"""
Persists per-view QCA analysis results into a patient case's
analysis_results/ folder, and assembles the final cross-view clinical
diagnosis report from those saved results -- so the final report can be
(re)generated at any time, independent of the in-memory analysis session
that produced the individual view results.

No Qt dependency -- reusable from any UI layer, a CLI, or a batch tool.
"""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import cv2

import pdf_report
from qca import QCAConfig
from report_engine import AngleResult, draw_angle_summary_bgr, draw_frame_stenosis_only, generate_reasoning


def _sanitize(name: str) -> str:
    name = name.strip().replace(" ", "_")
    return re.sub(r"[^A-Za-z0-9_.-]", "", name) or "view"


def save_view_results(analysis_dir: Path, angle_result: AngleResult,
                      patient_info: dict, cfg: QCAConfig) -> Path:
    """
    Saves one analyzed view's results into analysis_dir/<sanitized view label>/:
      - results.json       -- lightweight per-lesion summary (location,
                               severity, DS%, MLD/RVD, confidence, reasoning)
                               and view metadata. No image/array data --
                               this is the durable "where are the stenoses"
                               record.
      - summary_overlay.png -- the view's 2D stenosis-location diagram, saved
                               so the final combined report can embed it later
                               without needing the original frames/masks kept
                               around.
      - key_frame_N.png     -- the smallest set of frames that shows every
                               significant stenosis at least once (see
                               select_key_frames), each as a plain frame with
                               a circle + short id per lesion -- no vessel
                               mask/skeleton, since these are that exact
                               frame's own detections and always accurately
                               placed -- so the analyst can see the
                               strongest evidence for each detection without
                               re-running analysis.
      - view_report.pdf     -- the per-view explainable report (crops,
                               heatmaps, diameter profiles, reasoning text),
                               via the existing multi-view PDF renderer
                               called with just this one view.

    Returns the view's results directory.
    """
    view_dir = Path(analysis_dir) / _sanitize(angle_result.angle_label)
    view_dir.mkdir(parents=True, exist_ok=True)

    pdf_report.render_clinical_report(view_dir / "view_report.pdf", patient_info, [angle_result], cfg)

    image_name = None
    co_visible_ids = set()
    if angle_result.tracks and angle_result.summary_frame_idx is not None:
        rec = angle_result.get_frame_record(angle_result.summary_frame_idx)
        if rec is not None:
            vis_bgr = draw_angle_summary_bgr(rec, angle_result.tracks)
            image_name = "summary_overlay.png"
            cv2.imwrite(str(view_dir / image_name), vis_bgr)

            track_of_lesion = {id(les): t for t in angle_result.tracks for les in t.detections}
            for les in rec.lesions:
                t = track_of_lesion.get(id(les))
                if t is not None:
                    co_visible_ids.add(t.track_id)

    key_frame_images = []
    for idx in angle_result.key_frame_indices:
        rec = angle_result.get_frame_record(idx)
        if rec is None:
            continue
        vis_bgr = draw_frame_stenosis_only(rec, angle_result.tracks)
        name = f"key_frame_{idx}.png"
        cv2.imwrite(str(view_dir / name), vis_bgr)
        key_frame_images.append({"frame_idx": idx, "image": name})

    lesions_json = []
    for t in angle_result.tracks:
        rep = t.representative
        lesions_json.append({
            "track_id": t.track_id,
            "label": t.label,
            "artery": t.artery,
            "group": t.group,
            "severity": rep["severity"],
            "DS_percent": rep["DS_percent"],
            "MLD_px": rep["MLD_px"],
            "RVD_px": rep["RVD_px"],
            "MLD_mm": rep.get("MLD_mm"),
            "RVD_mm": rep.get("RVD_mm"),
            "confidence": rep.get("confidence"),
            "total_occlusion": bool(rep.get("total_occlusion")),
            "co_visible_in_summary_frame": t.track_id in co_visible_ids,
            "reasoning": generate_reasoning(t, cfg),
        })

    summary = {
        "view_label": angle_result.angle_label,
        "source": angle_result.video_path,
        "analyzed_at": datetime.now().isoformat(timespec="seconds"),
        "n_frames_total": angle_result.n_frames_total,
        "n_frames_analyzed": angle_result.n_frames_analyzed,
        "has_localization": angle_result.has_localization,
        "summary_image": image_name,
        "key_frame_images": key_frame_images,
        "lesions": lesions_json,
    }
    (view_dir / "results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return view_dir


def list_view_results(analysis_dir: Path) -> List[dict]:
    """
    Loads every saved per-view results.json under a case's analysis_results/
    folder. Each returned dict carries a "_view_dir" Path so callers (e.g.
    the final report renderer) can locate that view's saved summary_overlay.png.
    """
    analysis_dir = Path(analysis_dir)
    results = []
    if not analysis_dir.exists():
        return results
    for results_json in sorted(analysis_dir.glob("*/results.json")):
        try:
            data = json.loads(results_json.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        data["_view_dir"] = results_json.parent
        results.append(data)
    return results


def generate_final_clinical_report(case_dir: Path, analysis_dir: Path,
                                   patient_info: dict, cfg: QCAConfig) -> Path:
    """
    Assembles the final cross-view clinical diagnosis report from every saved
    per-view results.json + summary_overlay.png under analysis_dir --
    reusable at any time, independent of which analysis session produced them.
    Saves to case_dir/clinical_diagnosis_report.pdf.
    """
    view_summaries = list_view_results(analysis_dir)
    out_path = Path(case_dir) / "clinical_diagnosis_report.pdf"
    return pdf_report.render_clinical_diagnosis_report(out_path, patient_info, view_summaries, cfg)
