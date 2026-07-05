"""
Build a full-tree hybrid 3D reconstruction from one candidate pair folder.

This is the doctor-facing reconstruction style:

- preserve every visible reference-view branch
- use two-view epipolar matches to estimate the depth field
- color branches by confidence
    reliable: gray/white
    usable: blue
    estimated: orange

It is intentionally different from reconstruct_ranked_candidate.py, which exports
only validated branches and therefore can look like disconnected pipes.

Example:
    python scripts/reconstruct_full_tree_hybrid.py ^
        --candidate-dir dicom_batch_branch_graphs_expanded/case_030/group_a_sequence_block/C09_F019__C10_F025 ^
        --output-dir dicom_full_tree_hybrid/case_030_group_a
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
from scipy.interpolate import RBFInterpolator


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from dicom_3d_pipeline import extract_graph, resample_polyline, variable_tube  # noqa: E402
from epipolar_optimized_centerline import one_to_one_matches, optimize_branch, sample_radius, project_points  # noqa: E402


MATERIAL_COLORS = {
    "reliable": (0.82, 0.82, 0.82),
    "usable": (0.18, 0.42, 1.0),
    "estimated": (1.0, 0.56, 0.06),
    "junction": (0.72, 0.72, 0.72),
}


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


def smooth_1d(values: np.ndarray, window: int = 9) -> np.ndarray:
    if len(values) < window:
        return values
    if window % 2 == 0:
        window += 1
    pad = window // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    return np.convolve(padded, np.ones(window, dtype=np.float64) / window, mode="valid")


def smooth_centerline(points: np.ndarray, window: int = 9, iterations: int = 2) -> np.ndarray:
    out = points.copy()
    for _ in range(iterations):
        for dim in range(3):
            out[:, dim] = smooth_1d(out[:, dim], min(window, max(3, len(out) // 2 * 2 - 1)))
        out[0] = points[0]
        out[-1] = points[-1]
    return out


def object_xy_from_yx(coords_yx: np.ndarray, view: Dict[str, object]) -> np.ndarray:
    rows, cols = view["image_size_px"]
    spacing_y, spacing_x = view["imager_pixel_spacing_mm"]
    mag = float(view["source_detector_distance_mm"]) / max(float(view["source_patient_distance_mm"]), 1e-6)
    px_y = float(spacing_y) / max(mag, 1e-6)
    px_x = float(spacing_x) / max(mag, 1e-6)
    y = coords_yx[:, 0]
    x = coords_yx[:, 1]
    out_x = (x - (cols - 1) * 0.5) * px_x
    out_y = -(y - (rows - 1) * 0.5) * px_y
    return np.column_stack([out_x, out_y])


def build_depth_model(accepted: List[Dict[str, object]], fallback_z: float = 0.0):
    anchors_yx = []
    anchors_z = []
    for item in accepted:
        score = item["score"]
        anchors_yx.extend(score["samples_a"].tolist())
        anchors_z.extend(item["optimized_points"][:, 2].tolist())
    if len(anchors_yx) < 8:
        return None, fallback_z, fallback_z
    anchors_yx_arr = np.asarray(anchors_yx, dtype=np.float64)
    anchors_z_arr = np.asarray(anchors_z, dtype=np.float64)[:, None]
    model = RBFInterpolator(
        anchors_yx_arr,
        anchors_z_arr,
        neighbors=min(45, len(anchors_yx_arr)),
        smoothing=8.0,
    )
    z_low = float(np.percentile(anchors_z_arr, 5) - 8.0)
    z_high = float(np.percentile(anchors_z_arr, 95) + 8.0)
    return model, z_low, z_high


def make_sphere(center: np.ndarray, radius: float, rings: int = 8, segments: int = 14):
    vertices, faces = [], []
    for ring in range(rings + 1):
        phi = math.pi * ring / rings
        for seg in range(segments):
            theta = 2.0 * math.pi * seg / segments
            vertices.append(
                center
                + radius
                * np.array(
                    [
                        math.sin(phi) * math.cos(theta),
                        math.sin(phi) * math.sin(theta),
                        math.cos(phi),
                    ],
                    dtype=np.float64,
                )
            )
    for ring in range(rings):
        for seg in range(segments):
            a = ring * segments + seg
            b = ring * segments + ((seg + 1) % segments)
            c = (ring + 1) * segments + seg
            d = (ring + 1) * segments + ((seg + 1) % segments)
            faces.append((a, c, d))
            faces.append((a, d, b))
    return vertices, faces


def blend_junctions(branches: List[Dict[str, object]], radius_mm: float = 4.0):
    endpoints = []
    for branch in branches:
        centerline = branch["centerline"]
        endpoints.append((branch, "start", centerline[0].copy()))
        endpoints.append((branch, "end", centerline[-1].copy()))

    used = set()
    junctions = []
    for idx, item in enumerate(endpoints):
        if idx in used:
            continue
        used.add(idx)
        members = [item]
        changed = True
        while changed:
            changed = False
            center = np.mean([member[2] for member in members], axis=0)
            for j, other in enumerate(endpoints):
                if j in used:
                    continue
                if np.linalg.norm(other[2] - center) <= radius_mm:
                    used.add(j)
                    members.append(other)
                    changed = True
        if len(members) < 2:
            continue

        weights = []
        points = []
        radii = []
        for branch, side, point in members:
            status = branch["status"]
            weights.append(3.0 if status == "reliable" else 2.0 if status == "usable" else 0.8)
            points.append(point)
            radii.append(branch["radii"][0] if side == "start" else branch["radii"][-1])
        node = np.average(np.asarray(points), axis=0, weights=np.asarray(weights))
        node_radius = float(np.average(np.asarray(radii), weights=np.asarray(weights)))
        junctions.append((node, node_radius))

        for branch, side, _ in members:
            centerline = branch["centerline"]
            radii_arr = branch["radii"]
            limit = min(8, len(centerline))
            if side == "start":
                indexes = range(limit)
            else:
                indexes = range(len(centerline) - 1, len(centerline) - 1 - limit, -1)
            for offset, point_idx in enumerate(indexes):
                alpha = 1.0 - offset / max(limit - 1, 1)
                centerline[point_idx] = alpha * node + (1.0 - alpha) * centerline[point_idx]
                radii_arr[point_idx] = alpha * node_radius + (1.0 - alpha) * radii_arr[point_idx]
    return junctions


def write_obj(path: Path, parts: List[Tuple[str, str, list, list]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path.with_suffix(".mtl"), "w", encoding="utf-8") as f:
        for name, color in MATERIAL_COLORS.items():
            f.write(f"newmtl {name}\nKd {color[0]} {color[1]} {color[2]}\n")
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"mtllib {path.with_suffix('.mtl').name}\n")
        f.write("# Full-tree hybrid reconstruction. Materials encode confidence.\n")
        offset = 1
        for name, material, vertices, faces in parts:
            f.write(f"o {name}\nusemtl {material}\n")
            for vertex in vertices:
                f.write(f"v {vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f}\n")
            for a, b, c in faces:
                f.write(f"f {a + offset} {b + offset} {c + offset}\n")
            offset += len(vertices)


def draw_full_tree_overlay(original: np.ndarray, mask: np.ndarray, branches: List[Dict[str, object]], view: Dict[str, object], output_path: Path):
    canvas = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
    red = np.zeros_like(canvas)
    red[mask > 0] = (0, 0, 255)
    canvas = cv2.addWeighted(canvas, 0.78, red, 0.32, 0)
    colors = {"reliable": (230, 230, 230), "usable": (255, 120, 0), "estimated": (0, 165, 255)}
    h, w = canvas.shape[:2]
    for branch in branches:
        pts = project_points(branch["centerline"], view)
        finite = np.isfinite(pts).all(axis=1)
        pixels = np.rint(pts[finite]).astype(int)
        if len(pixels) < 2:
            continue
        color = colors.get(branch["status"], (0, 255, 255))
        for a, b in zip(pixels[:-1], pixels[1:]):
            if -w <= a[0] <= 2 * w and -h <= a[1] <= 2 * h and -w <= b[0] <= 2 * w and -h <= b[1] <= 2 * h:
                cv2.line(canvas, tuple(a), tuple(b), color, 2, cv2.LINE_AA)
        mid = pixels[len(pixels) // 2]
        if 0 <= mid[0] < w and 0 <= mid[1] < h:
            cv2.putText(canvas, str(branch["branch_id"]), tuple(mid), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
    cv2.imwrite(str(output_path), canvas)


def reconstruct(candidate_dir: Path, output_dir: Path) -> Dict[str, object]:
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
    dist_a = cv2.distanceTransform((clean_a == 0).astype(np.uint8), cv2.DIST_L2, 5)
    dist_b = cv2.distanceTransform((clean_b == 0).astype(np.uint8), cv2.DIST_L2, 5)

    matches = one_to_one_matches(branches_a, branches_b, view_a, view_b, dist_a, dist_b)
    accepted = []
    branch_status: Dict[int, str] = {}
    branch_match: Dict[int, Dict[str, object]] = {}
    for match in matches:
        status = str(match["status"])
        if status not in {"reliable", "usable"}:
            continue
        optimized = optimize_branch(match["score"]["points_3d"], match["score"]["samples_a"], match["score"]["samples_b"], view_a, view_b)
        match["optimized_points"] = optimized
        accepted.append(match)
        branch_id = int(match["branch_a"]["branch_id"])
        branch_status[branch_id] = status
        branch_match[branch_id] = match

    depth_model, z_low, z_high = build_depth_model(accepted)
    mag = float(view_a["source_detector_distance_mm"]) / max(float(view_a["source_patient_distance_mm"]), 1e-6)
    px_to_object_mm = float(view_a["imager_pixel_spacing_mm"][0]) / max(mag, 1e-6)

    full_branches = []
    report = []
    for branch in branches_a:
        branch_id = int(branch["branch_id"])
        status = branch_status.get(branch_id, "estimated")
        n = min(96, max(28, int(branch["points"])))
        coords = resample_polyline(branch["centerline_yx"], n)
        xy = object_xy_from_yx(coords, view_a)
        if depth_model is not None:
            z = depth_model(coords).reshape(-1)
            if status == "estimated":
                z = np.clip(z, z_low, z_high)
            else:
                z = np.clip(z, z_low - 6.0, z_high + 6.0)
        else:
            z = np.zeros(len(coords), dtype=np.float64)
        centerline = np.column_stack([xy, z])
        centerline = smooth_centerline(centerline, window=9, iterations=2)
        radii = sample_radius(radius_dt_a, coords, px_to_object_mm)
        radii = np.clip(smooth_1d(radii, min(9, max(3, len(radii) // 2 * 2 - 1))), 0.12, 2.4)
        if status == "estimated":
            radii *= 0.88
        full_branches.append(
            {
                "branch_id": branch_id,
                "status": status,
                "centerline": centerline,
                "radii": radii,
                "matched_view_b_branch_id": int(branch_match[branch_id]["branch_b"]["branch_id"]) if branch_id in branch_match else "",
            }
        )

    junctions = blend_junctions(full_branches)
    parts = []
    for branch in full_branches:
        vertices, faces = variable_tube(branch["centerline"], branch["radii"], segments=20)
        parts.append((f"branch_{branch['branch_id']:02d}_{branch['status']}", branch["status"], vertices, faces))
        min_radius = float(np.min(branch["radii"]))
        ref_radius = float(np.percentile(branch["radii"], 80))
        report.append(
            {
                "branch_id": branch["branch_id"],
                "status": branch["status"],
                "matched_view_b_branch_id": branch["matched_view_b_branch_id"],
                "points": len(branch["centerline"]),
                "min_radius_mm": round(min_radius, 4),
                "ref_radius_mm_p80": round(ref_radius, 4),
                "estimated_diameter_stenosis_pct": round(max(0.0, (1.0 - min_radius / max(ref_radius, 1e-6)) * 100.0), 2),
                "confidence_source": "two_view_epipolar" if branch["status"] in {"reliable", "usable"} else "depth_interpolated_from_reference_view",
            }
        )
    for idx, (center, radius) in enumerate(junctions):
        vertices, faces = make_sphere(center, max(0.35, min(1.5, radius * 1.05)))
        parts.append((f"junction_{idx:02d}", "junction", vertices, faces))

    output_obj = output_dir / "full_tree_hybrid_confidence.obj"
    write_obj(output_obj, parts)
    write_csv(output_dir / "full_tree_branch_report.csv", report)
    write_json(
        output_dir / "full_tree_centerlines.json",
        {
            "branches": [
                {
                    "branch_id": branch["branch_id"],
                    "status": branch["status"],
                    "matched_view_b_branch_id": branch["matched_view_b_branch_id"],
                    "points_xyz": branch["centerline"].tolist(),
                    "radii_mm": branch["radii"].tolist(),
                }
                for branch in full_branches
            ]
        },
    )
    draw_full_tree_overlay(original_a, clean_a, full_branches, view_a, output_dir / "view_a_full_tree_reprojection.png")
    draw_full_tree_overlay(original_b, clean_b, full_branches, view_b, output_dir / "view_b_full_tree_reprojection.png")

    counts: Dict[str, int] = {}
    for branch in full_branches:
        counts[branch["status"]] = counts.get(branch["status"], 0) + 1
    summary = {
        "candidate_dir": str(candidate_dir),
        "output_dir": str(output_dir),
        "reference_view": {
            "clip_index": graph_a["clip_index"],
            "frame": graph_a["frame"],
            "primary_angle_deg": view_a["primary_angle_deg"],
            "secondary_angle_deg": view_a["secondary_angle_deg"],
        },
        "secondary_view": {
            "clip_index": graph_b["clip_index"],
            "frame": graph_b["frame"],
            "primary_angle_deg": view_b["primary_angle_deg"],
            "secondary_angle_deg": view_b["secondary_angle_deg"],
        },
        "num_reference_branches": len(branches_a),
        "num_secondary_branches": len(branches_b),
        "branch_status_counts": counts,
        "num_junctions": len(junctions),
        "outputs": {
            "obj": str(output_obj),
            "branch_report": str(output_dir / "full_tree_branch_report.csv"),
            "view_a_reprojection": str(output_dir / "view_a_full_tree_reprojection.png"),
            "view_b_reprojection": str(output_dir / "view_b_full_tree_reprojection.png"),
        },
        "color_meaning": {
            "reliable": "gray/white, strict two-view validated",
            "usable": "blue, two-view supported",
            "estimated": "orange, preserved full reference-view branch with interpolated depth",
        },
    }
    write_json(output_dir / "full_tree_hybrid_summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    print(json.dumps(reconstruct(args.candidate_dir, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
