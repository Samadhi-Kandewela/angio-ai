"""
Epipolar-guided two-view coronary centerline reconstruction.

This is a stricter research reconstruction than the visual hybrid pipeline:

1. Extract centerline branches from the two saved segmentation masks.
2. Match branches one-to-one using epipolar/ray residual, length, radius, and
   local reprojection checks.
3. Triangulate matched branch centerlines from DICOM geometry.
4. Refine each 3D branch with a small reprojection + smoothness optimization.
5. Export an OBJ, centerline JSON, match report, and reprojection overlays.

Example:
    python scripts/epipolar_optimized_centerline.py --input-dir dicom_pipeline_output_v2 --output-dir dicom_epipolar_optimized
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from scipy.optimize import least_squares, linear_sum_assignment


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from dicom_3d_pipeline import (  # noqa: E402
    closest_point_between_rays,
    extract_graph,
    ray_from_pixel,
    resample_polyline,
    rotation_from_angles,
    smooth_1d,
    variable_tube,
    write_obj,
)


def load_gray(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(path)
    return image


def project_points(points_xyz: np.ndarray, view: Dict[str, object]) -> np.ndarray:
    rows, cols = [int(v) for v in view["image_size_px"]]
    spacing_y, spacing_x = [float(v) for v in view["imager_pixel_spacing_mm"]]
    sdd = float(view["source_detector_distance_mm"])
    spd = float(view["source_patient_distance_mm"])
    detector_z = sdd - spd
    source = np.array([0.0, 0.0, -spd], dtype=np.float64)
    rotation = rotation_from_angles(float(view["primary_angle_deg"]), float(view["secondary_angle_deg"]))

    out = []
    for point in points_xyz:
        local = rotation.T @ point
        denom = local[2] - source[2]
        if abs(float(denom)) < 1e-8:
            out.append((np.nan, np.nan))
            continue
        t = (detector_z - source[2]) / denom
        detector = source + t * (local - source)
        x = detector[0] / spacing_x + (cols - 1) * 0.5
        y = detector[1] / spacing_y + (rows - 1) * 0.5
        out.append((x, y))
    return np.asarray(out, dtype=np.float64)


def distance_to_mask_stats(points_xy: np.ndarray, distance_map: np.ndarray) -> Dict[str, float]:
    h, w = distance_map.shape[:2]
    finite = np.isfinite(points_xy).all(axis=1)
    inside = finite & (points_xy[:, 0] >= 0) & (points_xy[:, 0] < w) & (points_xy[:, 1] >= 0) & (points_xy[:, 1] < h)
    if not np.any(inside):
        return {"inside_pct": 0.0, "median_px": float("inf"), "p90_px": float("inf")}
    pix = np.rint(points_xy[inside]).astype(int)
    pix[:, 0] = np.clip(pix[:, 0], 0, w - 1)
    pix[:, 1] = np.clip(pix[:, 1], 0, h - 1)
    d = distance_map[pix[:, 1], pix[:, 0]].astype(np.float64)
    return {
        "inside_pct": float(np.count_nonzero(inside) / max(len(points_xy), 1) * 100.0),
        "median_px": float(np.median(d)),
        "p90_px": float(np.percentile(d, 90)),
    }


def sample_radius(dt: np.ndarray, coords_yx: np.ndarray, pixel_to_mm: float) -> np.ndarray:
    y = np.clip(np.rint(coords_yx[:, 0]).astype(int), 0, dt.shape[0] - 1)
    x = np.clip(np.rint(coords_yx[:, 1]).astype(int), 0, dt.shape[1] - 1)
    return np.clip(smooth_1d(dt[y, x] * pixel_to_mm, 9), 0.14, 2.4)


def triangulate_samples(samples_a: np.ndarray, samples_b: np.ndarray, view_a: Dict[str, object], view_b: Dict[str, object]):
    points = []
    ray_gaps = []
    for pa, pb in zip(samples_a, samples_b):
        source_a, dir_a = ray_from_pixel(pa, view_a)
        source_b, dir_b = ray_from_pixel(pb, view_b)
        point, gap = closest_point_between_rays(source_a, dir_a, source_b, dir_b)
        points.append(point)
        ray_gaps.append(gap)
    return np.asarray(points, dtype=np.float64), np.asarray(ray_gaps, dtype=np.float64)


def score_branch_pair(
    branch_a: Dict[str, object],
    branch_b: Dict[str, object],
    view_a: Dict[str, object],
    view_b: Dict[str, object],
    dist_a: np.ndarray,
    dist_b: np.ndarray,
    reverse_a: bool,
    reverse_b: bool,
    samples: int = 56,
) -> Dict[str, object]:
    pts_a = branch_a["centerline_yx"][::-1] if reverse_a else branch_a["centerline_yx"]
    pts_b = branch_b["centerline_yx"][::-1] if reverse_b else branch_b["centerline_yx"]
    n = min(samples, max(18, min(len(pts_a), len(pts_b))))
    samples_a = resample_polyline(pts_a, n)
    samples_b = resample_polyline(pts_b, n)
    points_3d, ray_gaps = triangulate_samples(samples_a, samples_b, view_a, view_b)

    proj_a = project_points(points_3d, view_a)
    proj_b = project_points(points_3d, view_b)
    reproj_a = np.linalg.norm(proj_a - np.column_stack([samples_a[:, 1], samples_a[:, 0]]), axis=1)
    reproj_b = np.linalg.norm(proj_b - np.column_stack([samples_b[:, 1], samples_b[:, 0]]), axis=1)
    mask_a = distance_to_mask_stats(proj_a, dist_a)
    mask_b = distance_to_mask_stats(proj_b, dist_b)

    length_ratio = max(float(branch_a["length_px"]), 1.0) / max(float(branch_b["length_px"]), 1.0)
    diameter_ratio = max(float(branch_a["mean_diameter_px"]), 0.2) / max(float(branch_b["mean_diameter_px"]), 0.2)
    ray_median = float(np.median(ray_gaps))
    ray_p90 = float(np.percentile(ray_gaps, 90))
    reproj_median = float(np.median(np.r_[reproj_a, reproj_b]))
    mask_median = float((mask_a["median_px"] + mask_b["median_px"]) * 0.5)
    objective = (
        ray_median
        + 0.18 * ray_p90
        + 0.65 * reproj_median
        + 0.35 * mask_median
        + 3.2 * abs(math.log(length_ratio))
        + 1.4 * abs(math.log(diameter_ratio))
    )

    return {
        "objective": float(objective),
        "samples_a": samples_a,
        "samples_b": samples_b,
        "points_3d": points_3d,
        "ray_median_mm": ray_median,
        "ray_p90_mm": ray_p90,
        "reproj_median_px": reproj_median,
        "mask_a_median_px": mask_a["median_px"],
        "mask_b_median_px": mask_b["median_px"],
        "inside_a_pct": mask_a["inside_pct"],
        "inside_b_pct": mask_b["inside_pct"],
        "reverse_a": reverse_a,
        "reverse_b": reverse_b,
    }


def best_pair_score(branch_a, branch_b, view_a, view_b, dist_a, dist_b) -> Dict[str, object]:
    best = None
    for reverse_a in (False, True):
        for reverse_b in (False, True):
            score = score_branch_pair(branch_a, branch_b, view_a, view_b, dist_a, dist_b, reverse_a, reverse_b)
            if best is None or score["objective"] < best["objective"]:
                best = score
    return best


def classify_match(score: Dict[str, object], branch_a: Dict[str, object]) -> str:
    if (
        score["ray_median_mm"] <= 2.5
        and score["ray_p90_mm"] <= 6.0
        and score["reproj_median_px"] <= 2.5
        and score["inside_a_pct"] >= 80.0
        and score["inside_b_pct"] >= 70.0
    ):
        return "reliable"
    if (
        score["ray_median_mm"] <= 4.5
        and score["ray_p90_mm"] <= 10.0
        and score["reproj_median_px"] <= 4.0
        and float(branch_a["length_px"]) >= 45.0
        and score["inside_a_pct"] >= 65.0
        and score["inside_b_pct"] >= 55.0
    ):
        return "usable"
    if (
        score["ray_median_mm"] <= 2.6
        and score["ray_p90_mm"] <= 4.5
        and score["reproj_median_px"] <= 7.5
        and score["mask_a_median_px"] <= 1.0
        and score["mask_b_median_px"] <= 1.0
        and score["inside_a_pct"] >= 90.0
        and score["inside_b_pct"] >= 90.0
    ):
        return "usable"
    return "rejected"


def one_to_one_matches(branches_a, branches_b, view_a, view_b, dist_a, dist_b):
    score_table: List[List[Dict[str, object]]] = []
    cost = np.zeros((len(branches_a), len(branches_b)), dtype=np.float64)
    for i, branch_a in enumerate(branches_a):
        row = []
        for j, branch_b in enumerate(branches_b):
            score = best_pair_score(branch_a, branch_b, view_a, view_b, dist_a, dist_b)
            row.append(score)
            cost[i, j] = score["objective"]
        score_table.append(row)

    row_idx, col_idx = linear_sum_assignment(cost)
    matches = []
    for i, j in zip(row_idx, col_idx):
        score = score_table[i][j]
        status = classify_match(score, branches_a[i])
        matches.append({"branch_a": branches_a[i], "branch_b": branches_b[j], "score": score, "status": status})
    return matches


def optimize_branch(points_3d: np.ndarray, samples_a: np.ndarray, samples_b: np.ndarray, view_a, view_b) -> np.ndarray:
    if len(points_3d) < 4:
        return points_3d
    target_a = np.column_stack([samples_a[:, 1], samples_a[:, 0]])
    target_b = np.column_stack([samples_b[:, 1], samples_b[:, 0]])
    initial = points_3d.copy()
    scale = max(float(np.ptp(initial[:, 0])), float(np.ptp(initial[:, 1])), float(np.ptp(initial[:, 2])), 1.0)

    def residual(flat):
        pts = flat.reshape((-1, 3))
        proj_a = project_points(pts, view_a)
        proj_b = project_points(pts, view_b)
        data = np.r_[
            (proj_a - target_a).ravel() * 0.055,
            (proj_b - target_b).ravel() * 0.055,
        ]
        smooth = []
        if len(pts) >= 3:
            smooth = ((pts[:-2] - 2.0 * pts[1:-1] + pts[2:]) / scale * 0.9).ravel()
        anchor = ((pts - initial) / scale * 0.12).ravel()
        return np.r_[data, smooth, anchor]

    result = least_squares(residual, initial.ravel(), max_nfev=80, ftol=1e-4, xtol=1e-4, verbose=0)
    optimized = result.x.reshape((-1, 3))
    for dim in range(3):
        optimized[:, dim] = smooth_1d(optimized[:, dim], min(7, max(3, len(optimized) // 2 * 2 - 1)))
    optimized[0] = points_3d[0]
    optimized[-1] = points_3d[-1]
    return optimized


def write_centerline_json(path: Path, accepted):
    payload = []
    for item in accepted:
        branch = item["branch_a"]
        score = item["score"]
        payload.append(
            {
                "branch_id": int(branch["branch_id"]),
                "matched_view_b_branch_id": int(item["branch_b"]["branch_id"]),
                "status": item["status"],
                "points_xyz": item["optimized_points"].tolist(),
                "samples_view_a_yx": score["samples_a"].tolist(),
                "samples_view_b_yx": score["samples_b"].tolist(),
            }
        )
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"branches": payload}, f, indent=2)


def draw_match_overlay(original, mask, accepted, view, output_path: Path):
    canvas = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR) if original.ndim == 2 else original.copy()
    red = canvas.copy()
    red[mask > 0] = (35, 35, 255)
    canvas = cv2.addWeighted(red, 0.32, canvas, 0.68, 0)
    colors = {"reliable": (0, 255, 80), "usable": (0, 220, 255)}
    for item in accepted:
        pts = project_points(item["optimized_points"], view)
        finite = np.isfinite(pts).all(axis=1)
        pixels = np.rint(pts[finite]).astype(int)
        if len(pixels) < 2:
            continue
        color = colors.get(item["status"], (0, 0, 255))
        for a, b in zip(pixels[:-1], pixels[1:]):
            cv2.line(canvas, tuple(a), tuple(b), color, 1, cv2.LINE_AA)
        cv2.putText(canvas, str(item["branch_a"]["branch_id"]), tuple(pixels[len(pixels) // 2]), cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)
    cv2.imwrite(str(output_path), canvas)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=ROOT / "dicom_pipeline_output_v2")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "dicom_epipolar_optimized")
    args = parser.parse_args()

    input_dir = args.input_dir
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    with open(input_dir / "pipeline_summary.json", "r", encoding="utf-8") as f:
        summary = json.load(f)
    view_a = summary["view_a"]
    view_b = summary["view_b"]

    mask_a = load_gray(input_dir / "view_a_clean_mask.png")
    mask_b = load_gray(input_dir / "view_b_clean_mask.png")
    original_a = load_gray(input_dir / "view_a_original.png")
    original_b = load_gray(input_dir / "view_b_original.png")
    clean_a, _, dt_a, branches_a = extract_graph(mask_a)
    clean_b, _, dt_b, branches_b = extract_graph(mask_b)
    dist_a = cv2.distanceTransform((clean_a == 0).astype(np.uint8), cv2.DIST_L2, 5)
    dist_b = cv2.distanceTransform((clean_b == 0).astype(np.uint8), cv2.DIST_L2, 5)

    matches = one_to_one_matches(branches_a, branches_b, view_a, view_b, dist_a, dist_b)
    mag_factor = float(view_a["source_detector_distance_mm"]) / max(float(view_a["source_patient_distance_mm"]), 1e-6)
    px_to_object_mm = float(view_a["imager_pixel_spacing_mm"][0]) / max(mag_factor, 1e-6)

    accepted = []
    report = []
    parts = []
    for match in matches:
        score = match["score"]
        row = {
            "branch_id": match["branch_a"]["branch_id"],
            "matched_view_b_branch_id": match["branch_b"]["branch_id"],
            "status": match["status"],
            "objective": score["objective"],
            "ray_median_mm": score["ray_median_mm"],
            "ray_p90_mm": score["ray_p90_mm"],
            "reproj_median_px": score["reproj_median_px"],
            "mask_a_median_px": score["mask_a_median_px"],
            "mask_b_median_px": score["mask_b_median_px"],
            "inside_a_pct": score["inside_a_pct"],
            "inside_b_pct": score["inside_b_pct"],
            "reverse_a": score["reverse_a"],
            "reverse_b": score["reverse_b"],
        }
        report.append(row)
        if match["status"] == "rejected":
            continue
        optimized = optimize_branch(score["points_3d"], score["samples_a"], score["samples_b"], view_a, view_b)
        match["optimized_points"] = optimized
        accepted.append(match)
        radii = sample_radius(dt_a, score["samples_a"], px_to_object_mm)
        vertices, faces = variable_tube(optimized, radii, segments=16)
        parts.append((f"branch_{match['branch_a']['branch_id']:02d}_{match['status']}", match["status"], vertices, faces))

    if parts:
        write_obj(out / "epipolar_optimized_centerline_tree.obj", parts)
    write_centerline_json(out / "epipolar_optimized_centerlines.json", accepted)
    with open(out / "epipolar_branch_match_report.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(report[0].keys()))
        writer.writeheader()
        writer.writerows(report)

    draw_match_overlay(original_a, clean_a, accepted, view_a, out / "view_a_epipolar_reprojection.png")
    draw_match_overlay(original_b, clean_b, accepted, view_b, out / "view_b_epipolar_reprojection.png")

    status_counts = {}
    for row in report:
        status_counts[row["status"]] = status_counts.get(row["status"], 0) + 1
    output_summary = {
        "input_dir": str(input_dir),
        "output_dir": str(out),
        "num_view_a_branches": len(branches_a),
        "num_view_b_branches": len(branches_b),
        "num_matches": len(matches),
        "num_accepted": len(accepted),
        "status_counts": status_counts,
        "outputs": {
            "obj": str(out / "epipolar_optimized_centerline_tree.obj"),
            "centerlines": str(out / "epipolar_optimized_centerlines.json"),
            "match_report": str(out / "epipolar_branch_match_report.csv"),
            "view_a_overlay": str(out / "view_a_epipolar_reprojection.png"),
            "view_b_overlay": str(out / "view_b_epipolar_reprojection.png"),
        },
    }
    with open(out / "epipolar_optimized_summary.json", "w", encoding="utf-8") as f:
        json.dump(output_summary, f, indent=2)
    print(json.dumps(output_summary, indent=2))


if __name__ == "__main__":
    main()
