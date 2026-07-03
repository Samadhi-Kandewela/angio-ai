"""
Use a third angiography view to refine confidence labels on the current best
smoothed confidence-colored OBJ.

This first version is intentionally conservative:
- It does not move geometry.
- It keeps existing reliable/usable labels.
- It upgrades estimated branches to usable only if their projection into IM1
  lands close to the IM1 vessel mask.

Example:
    python scripts/im1_third_view_confidence_refine.py --dicom-dir dicom_sample --clip-index 1 --frame 7
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import onnxruntime as ort


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from dicom_3d_pipeline import DEFAULT_MODEL, extract_graph, find_xa_clips, read_frame, segment_frame, view_from_clip  # noqa: E402
from reprojection_validation_report import parse_obj_centerlines, project_points  # noqa: E402


MATERIALS = {
    "reliable": (0.82, 0.82, 0.82),
    "usable": (0.18, 0.42, 1.0),
    "estimated": (1.0, 0.58, 0.06),
    "junction": (0.72, 0.72, 0.72),
}


def write_mtl(path: Path):
    with open(path, "w", encoding="utf-8") as f:
        for name, color in MATERIALS.items():
            f.write(f"newmtl {name}\n")
            f.write(f"Kd {color[0]} {color[1]} {color[2]}\n")


def load_branch_report(path: Path) -> Dict[int, Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return {int(row["branch_id"]): row for row in csv.DictReader(f)}


def distance_stats(points_xy: np.ndarray, distance_map: np.ndarray) -> Dict[str, float]:
    h, w = distance_map.shape[:2]
    finite = np.isfinite(points_xy).all(axis=1)
    inside = finite & (points_xy[:, 0] >= 0) & (points_xy[:, 0] < w) & (points_xy[:, 1] >= 0) & (points_xy[:, 1] < h)
    if not np.any(inside):
        return {
            "inside_pct": 0.0,
            "median_px": float("inf"),
            "mean_px": float("inf"),
            "p90_px": float("inf"),
            "max_px": float("inf"),
        }
    pixels = np.rint(points_xy[inside]).astype(int)
    distances = distance_map[pixels[:, 1], pixels[:, 0]].astype(np.float64)
    return {
        "inside_pct": float(np.count_nonzero(inside) / max(len(points_xy), 1) * 100.0),
        "median_px": float(np.median(distances)),
        "mean_px": float(np.mean(distances)),
        "p90_px": float(np.percentile(distances, 90)),
        "max_px": float(np.max(distances)),
    }


def validation_status(stats: Dict[str, float]) -> str:
    if stats["inside_pct"] >= 80.0 and stats["median_px"] <= 3.0 and stats["p90_px"] <= 10.0:
        return "supported_by_im1"
    if stats["inside_pct"] >= 70.0 and stats["median_px"] <= 6.0 and stats["p90_px"] <= 18.0:
        return "weakly_supported_by_im1"
    return "not_supported_by_im1"


def draw_overlay(original: np.ndarray, mask: np.ndarray, projected: Dict[int, np.ndarray], report_rows: List[Dict[str, object]], path: Path):
    canvas = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
    red = canvas.copy()
    red[mask > 0] = (35, 35, 255)
    canvas = cv2.addWeighted(red, 0.30, canvas, 0.70, 0)

    rows = {int(row["branch_id"]): row for row in report_rows}
    colors = {
        "reliable": (225, 225, 225),
        "usable": (255, 90, 35),
        "upgraded_by_im1": (255, 90, 35),
        "estimated": (0, 165, 255),
    }
    for branch_id, points in projected.items():
        row = rows.get(branch_id, {})
        status = str(row.get("final_status", "estimated"))
        color = colors.get(status, (0, 165, 255))
        finite = np.isfinite(points).all(axis=1)
        pixels = np.rint(points[finite]).astype(int)
        if len(pixels) < 2:
            continue
        for a, b in zip(pixels[:-1], pixels[1:]):
            cv2.line(canvas, tuple(a), tuple(b), color, 1, cv2.LINE_AA)
        mid = tuple(pixels[len(pixels) // 2])
        cv2.putText(canvas, str(branch_id), mid, cv2.FONT_HERSHEY_SIMPLEX, 0.36, color, 1, cv2.LINE_AA)
    cv2.imwrite(str(path), canvas)


def refine_statuses(branch_report: Dict[int, Dict[str, str]], centerlines, view_c, distance_map):
    projected = {}
    rows = []
    for branch in centerlines:
        branch_id = branch.branch_id
        row = branch_report.get(branch_id, {})
        current_status = row.get("status", branch.status)
        points_xy = project_points(branch.points, view_c)
        projected[branch_id] = points_xy
        stats = distance_stats(points_xy, distance_map)
        im1_status = validation_status(stats)

        final_status = current_status
        confidence_source = row.get("confidence_source", "")
        if current_status == "estimated" and im1_status == "supported_by_im1":
            final_status = "upgraded_by_im1"
            confidence_source = "im1_third_view_support"

        rows.append(
            {
                "branch_id": branch_id,
                "previous_status": current_status,
                "final_status": final_status,
                "im1_validation_status": im1_status,
                "inside_pct": stats["inside_pct"],
                "median_error_px": stats["median_px"],
                "mean_error_px": stats["mean_px"],
                "p90_error_px": stats["p90_px"],
                "max_error_px": stats["max_px"],
                "confidence_source": confidence_source,
            }
        )
    return rows, projected


def material_for_status(status: str) -> str:
    if status == "upgraded_by_im1":
        return "usable"
    if status in {"reliable", "usable", "estimated"}:
        return status
    return "estimated"


def rewrite_obj_with_status(source_obj: Path, output_obj: Path, final_status_by_branch: Dict[int, str]):
    output_obj.parent.mkdir(parents=True, exist_ok=True)
    write_mtl(output_obj.with_suffix(".mtl"))
    current_material = "estimated"
    with open(source_obj, "r", encoding="utf-8", errors="ignore") as src, open(output_obj, "w", encoding="utf-8") as dst:
        dst.write(f"mtllib {output_obj.with_suffix('.mtl').name}\n")
        dst.write("# Smoothed confidence model refined by IM1 third-view validation.\n")
        for raw in src:
            line = raw.rstrip("\n")
            if line.startswith("mtllib ") or line.startswith("#"):
                continue
            if line.startswith("o "):
                name = line[2:].strip()
                match = re.search(r"branch[_-](\d+)", name)
                if match:
                    branch_id = int(match.group(1))
                    status = final_status_by_branch.get(branch_id, "estimated")
                    current_material = material_for_status(status)
                    dst.write(f"o branch_{branch_id:02d}_{current_material}\n")
                    continue
                if name.startswith("junction"):
                    current_material = "junction"
            if line.startswith("usemtl "):
                dst.write(f"usemtl {current_material}\n")
                continue
            dst.write(raw)


def write_branch_quality_report(path: Path, original_report: Dict[int, Dict[str, str]], validation_rows: List[Dict[str, object]]):
    rows = []
    validation_by_branch = {int(row["branch_id"]): row for row in validation_rows}
    for branch_id in sorted(validation_by_branch):
        base = original_report.get(branch_id, {})
        validation = validation_by_branch[branch_id]
        final_status = str(validation["final_status"])
        rows.append(
            {
                "branch_id": branch_id,
                "status": material_for_status(final_status),
                "final_status_detail": final_status,
                "estimated_diameter_stenosis_pct": base.get("estimated_diameter_stenosis_pct", ""),
                "min_radius_mm": base.get("min_radius_mm", ""),
                "ref_radius_mm_p80": base.get("ref_radius_mm_p80", ""),
                "median_residual_mm": base.get("median_residual_mm", ""),
                "p90_residual_mm": base.get("p90_residual_mm", ""),
                "confidence_source": validation["confidence_source"],
                "im1_validation_status": validation["im1_validation_status"],
                "im1_median_error_px": validation["median_error_px"],
                "im1_p90_error_px": validation["p90_error_px"],
            }
        )
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def copy_context(source_dir: Path, output_dir: Path):
    for name in (
        "pipeline_summary.json",
        "view_a_original.png",
        "view_b_original.png",
        "view_a_overlay.png",
        "view_b_overlay.png",
        "view_a_mask.png",
        "view_b_mask.png",
        "view_a_skeleton.png",
        "view_b_skeleton.png",
    ):
        source = source_dir / name
        if source.exists():
            shutil.copyfile(source, output_dir / name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dicom-dir", type=Path, default=ROOT / "dicom_sample")
    parser.add_argument("--source-dir", type=Path, default=ROOT / "dicom_smoothed_confidence")
    parser.add_argument("--source-obj", type=Path, default=ROOT / "dicom_smoothed_confidence" / "hybrid_smoothed_confidence_colored.obj")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "dicom_smoothed_confidence_im1_refined")
    parser.add_argument("--clip-index", type=int, default=1)
    parser.add_argument("--frame", type=int, default=7)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    clips = find_xa_clips(args.dicom_dir)
    clip = next((item for item in clips if item.index == args.clip_index), None)
    if clip is None:
        raise RuntimeError(f"Could not find XA clip index {args.clip_index}")

    session = ort.InferenceSession(str(args.model))
    input_name = session.get_inputs()[0].name
    gray = read_frame(clip, args.frame)
    mask, overlay = segment_frame(session, input_name, gray, args.threshold)
    clean, skeleton, _, _ = extract_graph(mask)
    view_c = {**view_from_clip(clip, "view_c"), "clip_index": clip.index, "dicom_path": str(clip.path), "frame": args.frame}
    distance_map = cv2.distanceTransform((clean == 0).astype(np.uint8), cv2.DIST_L2, 5)

    cv2.imwrite(str(out / "view_c_original.png"), gray)
    cv2.imwrite(str(out / "view_c_overlay.png"), overlay)
    cv2.imwrite(str(out / "view_c_mask.png"), mask)
    cv2.imwrite(str(out / "view_c_clean_mask.png"), clean)
    cv2.imwrite(str(out / "view_c_skeleton.png"), skeleton)

    centerlines = parse_obj_centerlines(args.source_obj)
    branch_report = load_branch_report(args.source_dir / "branch_quality_report.csv")
    validation_rows, projected = refine_statuses(branch_report, centerlines, view_c, distance_map)
    final_status_by_branch = {int(row["branch_id"]): str(row["final_status"]) for row in validation_rows}

    output_obj = out / "hybrid_smoothed_confidence_im1_refined.obj"
    rewrite_obj_with_status(args.source_obj, output_obj, final_status_by_branch)
    with open(out / "im1_branch_validation_report.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(validation_rows[0].keys()))
        writer.writeheader()
        writer.writerows(validation_rows)
    write_branch_quality_report(out / "branch_quality_report.csv", branch_report, validation_rows)
    draw_overlay(gray, clean, projected, validation_rows, out / "view_c_im1_validation_overlay.png")
    copy_context(args.source_dir, out)

    if (out / "pipeline_summary.json").exists():
        with open(out / "pipeline_summary.json", "r", encoding="utf-8") as f:
            summary = json.load(f)
    else:
        summary = {}
    summary["view_c"] = view_c
    with open(out / "pipeline_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    counts = {}
    for row in validation_rows:
        counts[str(row["final_status"])] = counts.get(str(row["final_status"]), 0) + 1
    output_summary = {
        "source_obj": str(args.source_obj),
        "output_obj": str(output_obj),
        "third_view": view_c,
        "status_counts": counts,
        "upgraded_branches": [int(row["branch_id"]) for row in validation_rows if row["final_status"] == "upgraded_by_im1"],
        "outputs": {
            "obj": str(output_obj),
            "branch_quality_report": str(out / "branch_quality_report.csv"),
            "im1_validation_report": str(out / "im1_branch_validation_report.csv"),
            "im1_overlay": str(out / "view_c_im1_validation_overlay.png"),
        },
    }
    with open(out / "im1_refinement_summary.json", "w", encoding="utf-8") as f:
        json.dump(output_summary, f, indent=2)
    print(json.dumps(output_summary, indent=2))


if __name__ == "__main__":
    main()
