"""
Multi-view coronary 3D reconstruction prototype.

This extends the two-view research pipeline by using one reference view for the
full visible tree and any number of supporting views for depth constraints.

Example:
    python scripts/dicom_multiview_3d_pipeline.py ^
        --dicom-dir dicom_files/case_003 ^
        --views 0:23,2:17,3:24 ^
        --output-dir dicom_case003_lca_multiview
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import pydicom
from scipy.interpolate import RBFInterpolator

from dicom_3d_pipeline import (
    DEFAULT_MODEL,
    clean_mask,
    extract_graph,
    find_branch_matches,
    find_xa_clips,
    read_frame,
    ray_from_pixel,
    resample_polyline,
    segment_frame,
    smooth_1d,
    variable_tube,
    view_from_clip,
)
from reprojection_validation_report import ORIENTATION_MODES, distance_stats, orient_pixels, project_points, status_from_error


ROOT = Path(__file__).resolve().parents[1]

MATERIALS = {
    "multi_view_reliable": (0.84, 0.84, 0.84),
    "two_view_supported": (0.20, 0.48, 1.00),
    "single_view_estimated": (1.00, 0.58, 0.10),
}


def parse_views(value: str) -> list[tuple[int, int]]:
    views = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        clip_text, frame_text = part.split(":", 1)
        views.append((int(clip_text), int(frame_text)))
    if len(views) < 2:
        raise argparse.ArgumentTypeError("--views must contain at least two clip:frame entries")
    return views


def write_obj(path: Path, parts):
    with open(path.with_suffix(".mtl"), "w", encoding="utf-8") as f:
        for name, color in MATERIALS.items():
            f.write(f"newmtl {name}\n")
            f.write(f"Kd {color[0]} {color[1]} {color[2]}\n")
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"mtllib {path.with_suffix('.mtl').name}\n")
        f.write("# Multi-view DICOM-to-3D pipeline output.\n")
        offset = 1
        for name, material, vertices, faces in parts:
            f.write(f"o {name}\nusemtl {material}\n")
            for vertex in vertices:
                f.write(f"v {vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f}\n")
            for a, b, c in faces:
                f.write(f"f {a + offset} {b + offset} {c + offset}\n")
            offset += len(vertices)


def branch_status(accepted_count: int, reliable_count: int) -> str:
    if accepted_count >= 2 or reliable_count >= 1:
        return "multi_view_reliable"
    if accepted_count == 1:
        return "two_view_supported"
    return "single_view_estimated"


def points_on_reference_rays(coords_yx: np.ndarray, z_values: np.ndarray, reference_view: dict) -> np.ndarray:
    points = []
    for coord, z_value in zip(coords_yx, z_values):
        source, direction = ray_from_pixel(coord, reference_view)
        if abs(float(direction[2])) < 1e-6:
            points.append(source + direction * 500.0)
            continue
        t = (float(z_value) - source[2]) / direction[2]
        points.append(source + t * direction)
    return np.asarray(points, dtype=np.float64)


def smooth_polyline_array(values: np.ndarray, window: int) -> np.ndarray:
    if len(values) < 5:
        return values
    window = min(window, len(values) if len(values) % 2 == 1 else len(values) - 1)
    if window < 3:
        return values
    out = values.copy()
    for col in range(values.shape[1]):
        out[:, col] = smooth_1d(values[:, col], window)
    out[0] = values[0]
    out[-1] = values[-1]
    return out


def smooth_depth(z_values: np.ndarray, status: str) -> np.ndarray:
    z = np.asarray(z_values, dtype=np.float64).copy()
    if len(z) < 3:
        return z
    if status == "single_view_estimated":
        # Unsupported branches should not inherit high-frequency RBF noise.
        # A gentle linear depth trend keeps them anatomical while clearly
        # marking them as estimates in the report/material.
        trend = np.linspace(float(np.percentile(z, 35)), float(np.percentile(z, 65)), len(z))
        z = 0.25 * z + 0.75 * trend
        window = 21
        max_step = 1.1
    else:
        window = 13
        max_step = 2.0
    z = smooth_1d(z, min(window, len(z) if len(z) % 2 == 1 else len(z) - 1))
    for idx in range(1, len(z)):
        delta = z[idx] - z[idx - 1]
        if abs(float(delta)) > max_step:
            z[idx] = z[idx - 1] + math.copysign(max_step, float(delta))
    return z


def smooth_centerline(points: np.ndarray, status: str) -> np.ndarray:
    if len(points) < 5:
        return points
    window = 15 if status == "single_view_estimated" else 9
    return smooth_polyline_array(points, window)


def build_multiview_mesh(reference, support_results, output_dir: Path, mag_factor: float):
    ref_branches = reference["branches"]
    ref_dt = reference["distance_transform"]

    anchors_yx = []
    anchors_xyz = []
    support_by_branch = {int(branch["branch_id"]): [] for branch in ref_branches}

    for support in support_results:
        for match in support["matches"]:
            if match["status"] not in {"reliable", "usable"}:
                continue
            branch_id = int(match["branch_a"]["branch_id"])
            anchors_yx.extend(match["score"]["samples_a"])
            anchors_xyz.extend(match["score"]["points_3d"])
            support_by_branch[branch_id].append(
                {
                    "view_name": support["name"],
                    "support_branch_id": int(match["branch_b"]["branch_id"]),
                    "status": match["status"],
                    "median_residual_mm": float(match["score"]["median"]),
                    "p90_residual_mm": float(match["score"]["p90"]),
                }
            )

    if len(anchors_yx) < 8:
        raise RuntimeError("Not enough multi-view-supported points to build a depth model.")

    depth_model = RBFInterpolator(np.asarray(anchors_yx), np.asarray(anchors_xyz), neighbors=45, smoothing=6.0)
    object_px_mm = 0.2375 / max(mag_factor, 1e-6)
    anchor_z = np.asarray(anchors_xyz, dtype=np.float64)[:, 2]
    z_low = float(np.percentile(anchor_z, 5) - 8.0)
    z_high = float(np.percentile(anchor_z, 95) + 8.0)

    parts = []
    centerlines = {}
    report_rows = []
    for branch in ref_branches:
        branch_id = int(branch["branch_id"])
        supports = support_by_branch[branch_id]
        reliable_count = sum(1 for item in supports if item["status"] == "reliable")
        status = branch_status(len(supports), reliable_count)

        coords = resample_polyline(branch["centerline_yx"], min(96, max(32, branch["points"])))
        coords = smooth_polyline_array(coords, 9)
        estimated_points = depth_model(coords)
        z = estimated_points[:, 2]
        if status == "single_view_estimated":
            z = np.clip(z, z_low, z_high)
        else:
            z = np.clip(z, z_low - 6.0, z_high + 6.0)
        z = smooth_depth(z, status)
        points = points_on_reference_rays(coords, z, reference["view"])
        points = smooth_centerline(points, status)

        y = np.clip(np.rint(coords[:, 0]).astype(int), 0, ref_dt.shape[0] - 1)
        x = np.clip(np.rint(coords[:, 1]).astype(int), 0, ref_dt.shape[1] - 1)
        radii = np.clip(smooth_1d(ref_dt[y, x] * object_px_mm, 9), 0.16, 2.2)
        if status == "single_view_estimated":
            radii *= 0.85

        vertices, faces = variable_tube(points, radii)
        parts.append((f"branch_{branch_id:02d}_{status}", status, vertices, faces))
        centerlines[str(branch_id)] = points.tolist()

        residuals = [item["median_residual_mm"] for item in supports]
        p90s = [item["p90_residual_mm"] for item in supports]
        min_r = float(np.min(radii))
        ref_r = float(np.percentile(radii, 80))
        report_rows.append(
            {
                "branch_id": branch_id,
                "status": status,
                "support_count": len(supports),
                "reliable_support_count": reliable_count,
                "support_views": ";".join(item["view_name"] for item in supports),
                "median_residual_mm": float(np.median(residuals)) if residuals else "",
                "p90_residual_mm": float(np.median(p90s)) if p90s else "",
                "min_radius_mm": min_r,
                "ref_radius_mm_p80": ref_r,
                "estimated_diameter_stenosis_pct": max(0.0, (1.0 - min_r / max(ref_r, 1e-6)) * 100.0),
            }
        )

    obj_path = output_dir / "multiview_lca_tree.obj"
    write_obj(obj_path, parts)
    with open(output_dir / "multiview_centerlines.json", "w", encoding="utf-8") as f:
        json.dump(centerlines, f)
    with open(output_dir / "multiview_branch_quality_report.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(report_rows[0].keys()))
        writer.writeheader()
        writer.writerows(report_rows)
    return obj_path, report_rows, centerlines


def best_orientation_for_view(projected_by_branch, branch_ids, distance_map):
    height, width = distance_map.shape[:2]
    if not branch_ids:
        branch_ids = list(projected_by_branch)
    all_points = np.vstack([projected_by_branch[branch_id] for branch_id in branch_ids])
    scores = {}
    for mode in ORIENTATION_MODES:
        stats = distance_stats(orient_pixels(all_points, mode, width, height), distance_map)
        objective = stats["median_px"]
        if not math.isfinite(objective):
            objective = 1e9
        objective += (100.0 - stats["inside_pct"]) * 0.05
        scores[mode] = objective
    return min(scores, key=scores.get)


def draw_multiview_validation(view_data, centerlines, report_by_branch, output_path: Path):
    original = view_data["gray"]
    mask = view_data["clean_mask"]
    canvas = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
    red = canvas.copy()
    red[mask > 0] = (35, 35, 255)
    canvas = cv2.addWeighted(red, 0.35, canvas, 0.65, 0)

    vessel = (mask > 0).astype(np.uint8)
    distance_map = cv2.distanceTransform((vessel == 0).astype(np.uint8), cv2.DIST_L2, 5)
    projected_raw = {
        int(branch_id_text): project_points(np.asarray(points, dtype=np.float64), view_data["view"])
        for branch_id_text, points in centerlines.items()
    }
    supported_ids = [
        branch_id
        for branch_id in projected_raw
        if report_by_branch[branch_id]["status"] != "single_view_estimated"
    ]
    orientation = best_orientation_for_view(projected_raw, supported_ids, distance_map)
    height, width = distance_map.shape[:2]

    rows = []
    overlay_items = []
    for branch_id_text, points in centerlines.items():
        branch_id = int(branch_id_text)
        projected = orient_pixels(projected_raw[branch_id], orientation, width, height)
        stats = distance_stats(projected, distance_map)
        validation_status = status_from_error(stats["median_px"], stats["p90_px"], stats["inside_pct"])
        branch_status_value = report_by_branch[branch_id]["status"]
        rows.append(
            {
                "branch_id": branch_id,
                "branch_status": branch_status_value,
                "validation_status": validation_status,
                "median_error_px": stats["median_px"],
                "p90_error_px": stats["p90_px"],
                "inside_pct": stats["inside_pct"],
            }
        )

        if branch_status_value == "single_view_estimated":
            color = (0, 165, 255)
        elif validation_status == "good":
            color = (0, 255, 80)
        elif validation_status == "review":
            color = (0, 210, 255)
        else:
            color = (0, 0, 255)
        overlay_items.append((branch_id, branch_status_value, projected, color))

    def paint(path: Path, supported_only: bool):
        out = canvas.copy()
        for branch_id, branch_status_value, projected, color in overlay_items:
            if supported_only and branch_status_value == "single_view_estimated":
                continue
            finite = np.isfinite(projected).all(axis=1)
            pts = np.rint(projected[finite]).astype(int)
            pts[:, 0] = np.clip(pts[:, 0], 0, out.shape[1] - 1)
            pts[:, 1] = np.clip(pts[:, 1], 0, out.shape[0] - 1)
            for a, b in zip(pts[:-1], pts[1:]):
                cv2.line(out, tuple(a), tuple(b), color, 1, cv2.LINE_AA)
            if len(pts):
                cv2.circle(out, tuple(pts[0]), 3, color, -1, cv2.LINE_AA)
                cv2.putText(out, str(branch_id), tuple(pts[len(pts) // 2]), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)
        cv2.imwrite(str(path), out)

    paint(output_path, supported_only=False)
    paint(output_path.with_name(output_path.stem.replace("_multiview_validation", "_supported_multiview_validation") + output_path.suffix), supported_only=True)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dicom-dir", type=Path, required=True)
    parser.add_argument("--views", type=parse_views, required=True, help="Comma-separated clip:frame list, first entry is reference.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "dicom_multiview_3d_run")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    clips = find_xa_clips(args.dicom_dir)
    session = ort.InferenceSession(str(args.model))
    input_name = session.get_inputs()[0].name

    view_data = []
    for view_number, (clip_index, frame) in enumerate(args.views):
        clip = clips[clip_index]
        name = f"view_{view_number:02d}_clip{clip_index}_frame{frame}"
        gray = read_frame(clip, frame)
        mask, overlay = segment_frame(session, input_name, gray, args.threshold)
        clean, skel, dt, branches = extract_graph(mask)
        view = view_from_clip(clip, name)
        view.update({"clip_index": clip_index, "frame": frame, "dicom_path": str(clip.path)})

        cv2.imwrite(str(output_dir / f"{name}_original.png"), gray)
        cv2.imwrite(str(output_dir / f"{name}_overlay.png"), overlay)
        cv2.imwrite(str(output_dir / f"{name}_mask.png"), mask)
        cv2.imwrite(str(output_dir / f"{name}_clean_mask.png"), clean)
        cv2.imwrite(str(output_dir / f"{name}_skeleton.png"), skel)

        view_data.append(
            {
                "name": name,
                "view": view,
                "gray": gray,
                "clean_mask": clean,
                "distance_transform": dt,
                "branches": branches,
            }
        )

    reference = view_data[0]
    support_results = []
    for support in view_data[1:]:
        matches = find_branch_matches(reference["branches"], support["branches"], reference["view"], support["view"])
        support_results.append({"name": support["name"], "view": support["view"], "matches": matches})

    mag_factor = getattr(
        pydicom.dcmread(str(clips[args.views[0][0]].path), stop_before_pixels=True, force=True),
        "EstimatedRadiographicMagnificationFactor",
        1.0,
    )
    obj_path, report_rows, centerlines = build_multiview_mesh(reference, support_results, output_dir, mag_factor)
    report_by_branch = {int(row["branch_id"]): row for row in report_rows}

    validation_rows = []
    for data in view_data:
        rows = draw_multiview_validation(
            data,
            centerlines,
            report_by_branch,
            output_dir / f"{data['name']}_multiview_validation.png",
        )
        for row in rows:
            row["view"] = data["name"]
        validation_rows.extend(rows)

    with open(output_dir / "multiview_validation_report.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(validation_rows[0].keys()))
        writer.writeheader()
        writer.writerows(validation_rows)

    counts = {}
    for row in report_rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    summary = {
        "dicom_dir": str(args.dicom_dir),
        "selected_views": [data["view"] for data in view_data],
        "reference_view": view_data[0]["name"],
        "num_reference_branches": len(reference["branches"]),
        "branch_status_counts": counts,
        "outputs": {
            "obj": str(obj_path),
            "branch_report": str(output_dir / "multiview_branch_quality_report.csv"),
            "centerlines": str(output_dir / "multiview_centerlines.json"),
            "validation_report": str(output_dir / "multiview_validation_report.csv"),
        },
        "viewer_command": f"python src\\artery_3d_vtk_viewer.py --obj {obj_path}",
    }
    with open(output_dir / "multiview_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
