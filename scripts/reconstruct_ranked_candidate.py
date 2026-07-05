"""
Reconstruct a conservative 3D OBJ from one ranked branch-candidate folder.

This script consumes a folder produced by scripts/build_batch_branch_graphs.py.
It recomputes strict one-to-one epipolar matches, keeps only reliable/usable
branches, triangulates their centerlines, optimizes reprojection, applies QCA-like
radius from the view-A mask, and exports:

- confidence-colored OBJ/MTL
- 3D centerline JSON
- branch match/reprojection report CSV
- view-A/view-B reprojection overlays

Example:
    python scripts/reconstruct_ranked_candidate.py ^
        --candidate-dir dicom_batch_branch_graphs_expanded/case_030/group_a_sequence_block/C09_F019__C10_F025 ^
        --output-dir dicom_ranked_reconstructions/case_030_group_a
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from dicom_3d_pipeline import extract_graph, variable_tube, write_obj  # noqa: E402
from epipolar_optimized_centerline import (  # noqa: E402
    one_to_one_matches,
    optimize_branch,
    project_points,
    sample_radius,
)


def load_gray(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(path)
    return image


def read_json(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def write_csv(path: Path, rows: List[Dict[str, object]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def status_rank(status: str) -> int:
    return {"reliable": 2, "usable": 1}.get(status, 0)


def accepted_matches(matches: List[Dict[str, object]], min_status: str) -> List[Dict[str, object]]:
    minimum = status_rank(min_status)
    return [match for match in matches if status_rank(str(match["status"])) >= minimum]


def draw_reprojection_overlay(original: np.ndarray, mask: np.ndarray, accepted: List[Dict[str, object]], view: Dict[str, object], output_path: Path):
    canvas = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
    red = np.zeros_like(canvas)
    red[mask > 0] = (0, 0, 255)
    canvas = cv2.addWeighted(canvas, 0.78, red, 0.32, 0)
    colors = {"reliable": (220, 220, 220), "usable": (255, 140, 20)}
    h, w = canvas.shape[:2]
    for item in accepted:
        pts = project_points(item["optimized_points"], view)
        finite = np.isfinite(pts).all(axis=1)
        pixels = np.rint(pts[finite]).astype(int)
        if len(pixels) < 2:
            continue
        color = colors.get(str(item["status"]), (0, 255, 255))
        for a, b in zip(pixels[:-1], pixels[1:]):
            if (
                -w <= a[0] <= 2 * w
                and -h <= a[1] <= 2 * h
                and -w <= b[0] <= 2 * w
                and -h <= b[1] <= 2 * h
            ):
                cv2.line(canvas, tuple(a), tuple(b), color, 2, cv2.LINE_AA)
        mid = pixels[len(pixels) // 2]
        if 0 <= mid[0] < w and 0 <= mid[1] < h:
            cv2.putText(canvas, str(item["branch_a"]["branch_id"]), tuple(mid), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    cv2.imwrite(str(output_path), canvas)


def reconstruct_candidate(candidate_dir: Path, output_dir: Path, min_status: str = "usable", tube_segments: int = 16) -> Dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)

    graph_a = read_json(candidate_dir / "branch_graph_view_a.json")
    graph_b = read_json(candidate_dir / "branch_graph_view_b.json")
    view_a = graph_a["view_geometry"]
    view_b = graph_b["view_geometry"]

    original_a = load_gray(candidate_dir / "view_a_original.png")
    original_b = load_gray(candidate_dir / "view_b_original.png")
    mask_a = load_gray(candidate_dir / "view_a_mask.png")
    mask_b = load_gray(candidate_dir / "view_b_mask.png")

    clean_a, _, radius_dt_a, branches_a = extract_graph(mask_a)
    clean_b, _, _, branches_b = extract_graph(mask_b)
    dist_to_mask_a = cv2.distanceTransform((clean_a == 0).astype(np.uint8), cv2.DIST_L2, 5)
    dist_to_mask_b = cv2.distanceTransform((clean_b == 0).astype(np.uint8), cv2.DIST_L2, 5)

    matches = one_to_one_matches(branches_a, branches_b, view_a, view_b, dist_to_mask_a, dist_to_mask_b)
    accepted = accepted_matches(matches, min_status=min_status)

    mag_factor = float(view_a["source_detector_distance_mm"]) / max(float(view_a["source_patient_distance_mm"]), 1e-6)
    px_to_object_mm = float(view_a["imager_pixel_spacing_mm"][0]) / max(mag_factor, 1e-6)

    parts = []
    report = []
    centerlines = []
    for match in accepted:
        score = match["score"]
        optimized = optimize_branch(score["points_3d"], score["samples_a"], score["samples_b"], view_a, view_b)
        match["optimized_points"] = optimized
        radii = sample_radius(radius_dt_a, score["samples_a"], px_to_object_mm)
        vertices, faces = variable_tube(optimized, radii, segments=tube_segments)
        parts.append((f"branch_{int(match['branch_a']['branch_id']):02d}_{match['status']}", match["status"], vertices, faces))

        min_radius = float(np.min(radii)) if len(radii) else 0.0
        ref_radius = float(np.percentile(radii, 80)) if len(radii) else 0.0
        report.append(
            {
                "view_a_branch_id": int(match["branch_a"]["branch_id"]),
                "view_b_branch_id": int(match["branch_b"]["branch_id"]),
                "status": match["status"],
                "objective": round(float(score["objective"]), 4),
                "ray_median_mm": round(float(score["ray_median_mm"]), 4),
                "ray_p90_mm": round(float(score["ray_p90_mm"]), 4),
                "reproj_median_px": round(float(score["reproj_median_px"]), 4),
                "mask_a_median_px": round(float(score["mask_a_median_px"]), 4),
                "mask_b_median_px": round(float(score["mask_b_median_px"]), 4),
                "inside_a_pct": round(float(score["inside_a_pct"]), 2),
                "inside_b_pct": round(float(score["inside_b_pct"]), 2),
                "min_radius_mm": round(min_radius, 4),
                "ref_radius_mm_p80": round(ref_radius, 4),
                "estimated_diameter_stenosis_pct": round(max(0.0, (1.0 - min_radius / max(ref_radius, 1e-6)) * 100.0), 2),
                "reverse_a": bool(score["reverse_a"]),
                "reverse_b": bool(score["reverse_b"]),
            }
        )
        centerlines.append(
            {
                "view_a_branch_id": int(match["branch_a"]["branch_id"]),
                "view_b_branch_id": int(match["branch_b"]["branch_id"]),
                "status": match["status"],
                "points_xyz": optimized.tolist(),
                "radii_mm": radii.tolist(),
                "samples_view_a_yx": score["samples_a"].tolist(),
                "samples_view_b_yx": score["samples_b"].tolist(),
            }
        )

    if parts:
        write_obj(output_dir / "ranked_candidate_reconstruction.obj", parts)
    write_csv(output_dir / "ranked_candidate_branch_report.csv", report)
    write_json(output_dir / "ranked_candidate_centerlines.json", {"branches": centerlines})
    draw_reprojection_overlay(original_a, clean_a, accepted, view_a, output_dir / "view_a_reprojection_overlay.png")
    draw_reprojection_overlay(original_b, clean_b, accepted, view_b, output_dir / "view_b_reprojection_overlay.png")

    status_counts: Dict[str, int] = {}
    for match in matches:
        status_counts[str(match["status"])] = status_counts.get(str(match["status"]), 0) + 1

    summary = {
        "candidate_dir": str(candidate_dir),
        "output_dir": str(output_dir),
        "view_a": {
            "clip_index": graph_a["clip_index"],
            "frame": graph_a["frame"],
            "primary_angle_deg": view_a["primary_angle_deg"],
            "secondary_angle_deg": view_a["secondary_angle_deg"],
        },
        "view_b": {
            "clip_index": graph_b["clip_index"],
            "frame": graph_b["frame"],
            "primary_angle_deg": view_b["primary_angle_deg"],
            "secondary_angle_deg": view_b["secondary_angle_deg"],
        },
        "num_view_a_branches": len(branches_a),
        "num_view_b_branches": len(branches_b),
        "num_matches": len(matches),
        "match_status_counts": status_counts,
        "num_accepted": len(accepted),
        "min_status": min_status,
        "outputs": {
            "obj": str(output_dir / "ranked_candidate_reconstruction.obj"),
            "branch_report": str(output_dir / "ranked_candidate_branch_report.csv"),
            "centerlines": str(output_dir / "ranked_candidate_centerlines.json"),
            "view_a_overlay": str(output_dir / "view_a_reprojection_overlay.png"),
            "view_b_overlay": str(output_dir / "view_b_reprojection_overlay.png"),
        },
    }
    write_json(output_dir / "ranked_candidate_reconstruction_summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-status", choices=["usable", "reliable"], default="usable")
    parser.add_argument("--tube-segments", type=int, default=16)
    args = parser.parse_args()

    summary = reconstruct_candidate(args.candidate_dir, args.output_dir, args.min_status, args.tube_segments)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
