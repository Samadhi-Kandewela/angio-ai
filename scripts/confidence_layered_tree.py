"""
Build a confidence-layered coronary tree:

- reliable/usable branches: validated two-view epipolar 3D centerlines
- estimated branches: full IM0 tree shape with depth interpolated from the
  validated 3D anchors

This intentionally separates visual completeness from geometric certainty.

Example:
    python scripts/confidence_layered_tree.py --input-dir dicom_pipeline_output_v2 --epipolar-dir dicom_epipolar_optimized --output-dir dicom_confidence_layered
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from dicom_3d_pipeline import extract_graph, ray_from_pixel, resample_polyline, smooth_1d, variable_tube  # noqa: E402


MATERIALS = {
    "reliable": (0.82, 0.82, 0.82),
    "usable": (0.18, 0.42, 1.0),
    "estimated": (1.0, 0.58, 0.06),
}


def load_gray(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(path)
    return image


def write_obj(path: Path, parts: List[Tuple[str, str, List[np.ndarray], List[Tuple[int, int, int]]]]):
    with open(path.with_suffix(".mtl"), "w", encoding="utf-8") as f:
        for name, color in MATERIALS.items():
            f.write(f"newmtl {name}\n")
            f.write(f"Kd {color[0]} {color[1]} {color[2]}\n")
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"mtllib {path.with_suffix('.mtl').name}\n")
        f.write("# Confidence-layered artery tree: validated branches plus estimated completion.\n")
        offset = 1
        for name, material, vertices, faces in parts:
            f.write(f"o {name}\nusemtl {material}\n")
            for vertex in vertices:
                f.write(f"v {vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f}\n")
            for a, b, c in faces:
                f.write(f"f {a + offset} {b + offset} {c + offset}\n")
            offset += len(vertices)


def load_validated(epipolar_dir: Path) -> Dict[int, Dict[str, object]]:
    path = epipolar_dir / "epipolar_optimized_centerlines.json"
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    out = {}
    for branch in payload.get("branches", []):
        branch_id = int(branch["branch_id"])
        out[branch_id] = {
            "branch_id": branch_id,
            "matched_view_b_branch_id": int(branch["matched_view_b_branch_id"]),
            "status": branch["status"],
            "points_xyz": np.asarray(branch["points_xyz"], dtype=np.float64),
            "samples_view_a_yx": np.asarray(branch["samples_view_a_yx"], dtype=np.float64),
            "samples_view_b_yx": np.asarray(branch["samples_view_b_yx"], dtype=np.float64),
        }
    return out


def point_on_im0_ray_near_target(point_yx: np.ndarray, target_xyz: np.ndarray, view_a: Dict[str, object]) -> np.ndarray:
    source, direction = ray_from_pixel(point_yx, view_a)
    t = float(np.dot(target_xyz - source, direction))
    return source + t * direction


def smooth_centerline(points: np.ndarray, window: int = 9) -> np.ndarray:
    if len(points) < 4:
        return points
    out = points.copy()
    actual_window = min(window, max(3, len(points) // 2 * 2 - 1))
    for dim in range(3):
        out[:, dim] = smooth_1d(out[:, dim], actual_window)
    out[0] = points[0]
    out[-1] = points[-1]
    return out


def build_anchor_cloud(validated: Dict[int, Dict[str, object]]):
    anchors_yx = []
    anchors_xyz = []
    for branch in validated.values():
        anchors_yx.extend(branch["samples_view_a_yx"])
        anchors_xyz.extend(branch["points_xyz"])
    anchors_yx = np.asarray(anchors_yx, dtype=np.float64)
    anchors_xyz = np.asarray(anchors_xyz, dtype=np.float64)
    if len(anchors_yx) < 8:
        raise RuntimeError("Need at least 8 validated anchor points for confidence-layered interpolation.")
    return anchors_yx, anchors_xyz


def estimate_branch_3d(coords_yx: np.ndarray, view_a: Dict[str, object], anchors_yx: np.ndarray, anchors_xyz: np.ndarray) -> np.ndarray:
    # Conservative nearest-anchor interpolation avoids dangerous RBF
    # extrapolation for branches far away from the validated core.
    xyz_low = np.percentile(anchors_xyz, 3, axis=0) - np.array([8.0, 8.0, 12.0])
    xyz_high = np.percentile(anchors_xyz, 97, axis=0) + np.array([8.0, 8.0, 12.0])
    targets = []
    k = min(18, len(anchors_yx))
    for coord in coords_yx:
        distances = np.linalg.norm(anchors_yx - coord[None, :], axis=1)
        nearest = np.argsort(distances)[:k]
        local_dist = distances[nearest]
        weights = 1.0 / np.maximum(local_dist, 2.0) ** 2
        target = np.average(anchors_xyz[nearest], axis=0, weights=weights)
        targets.append(np.clip(target, xyz_low, xyz_high))
    points = np.asarray([point_on_im0_ray_near_target(yx, xyz, view_a) for yx, xyz in zip(coords_yx, targets)], dtype=np.float64)
    points = np.clip(points, xyz_low, xyz_high)
    return smooth_centerline(points, 9)


def branch_anchor_distance_px(coords_yx: np.ndarray, anchors_yx: np.ndarray) -> Tuple[float, float]:
    dists = []
    for point in coords_yx:
        d = np.linalg.norm(anchors_yx - point[None, :], axis=1)
        dists.append(float(np.min(d)))
    values = np.asarray(dists, dtype=np.float64)
    return float(np.median(values)), float(np.min(values))


def sample_radius(dt: np.ndarray, coords_yx: np.ndarray, pixel_to_mm: float, confidence: str) -> np.ndarray:
    y = np.clip(np.rint(coords_yx[:, 0]).astype(int), 0, dt.shape[0] - 1)
    x = np.clip(np.rint(coords_yx[:, 1]).astype(int), 0, dt.shape[1] - 1)
    radii = np.clip(smooth_1d(dt[y, x] * pixel_to_mm, 9), 0.14, 2.4)
    if confidence == "estimated":
        radii *= 0.82
    return radii


def draw_confidence_overlay(original: np.ndarray, mask: np.ndarray, centerlines: Dict[int, Dict[str, object]], output_path: Path):
    canvas = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR) if original.ndim == 2 else original.copy()
    red = canvas.copy()
    red[mask > 0] = (35, 35, 255)
    canvas = cv2.addWeighted(red, 0.28, canvas, 0.72, 0)
    colors = {
        "reliable": (225, 225, 225),
        "usable": (255, 90, 35),
        "estimated": (0, 170, 255),
    }
    for branch_id, item in centerlines.items():
        coords = np.asarray(item["samples_view_a_yx"], dtype=np.float64)
        pixels = np.rint(np.column_stack([coords[:, 1], coords[:, 0]])).astype(int)
        color = colors.get(str(item["confidence"]), (0, 170, 255))
        for a, b in zip(pixels[:-1], pixels[1:]):
            cv2.line(canvas, tuple(a), tuple(b), color, 1, cv2.LINE_AA)
        if len(pixels):
            cv2.putText(canvas, str(branch_id), tuple(pixels[len(pixels) // 2]), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)
    cv2.imwrite(str(output_path), canvas)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=ROOT / "dicom_pipeline_output_v2")
    parser.add_argument("--epipolar-dir", type=Path, default=ROOT / "dicom_epipolar_optimized")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "dicom_confidence_layered")
    args = parser.parse_args()

    input_dir = args.input_dir
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    with open(input_dir / "pipeline_summary.json", "r", encoding="utf-8") as f:
        summary = json.load(f)
    view_a = summary["view_a"]

    original_a = load_gray(input_dir / "view_a_original.png")
    mask_a = load_gray(input_dir / "view_a_clean_mask.png")
    clean_a, _, dt_a, branches_a = extract_graph(mask_a)
    validated = load_validated(args.epipolar_dir)
    anchors_yx, anchors_xyz = build_anchor_cloud(validated)

    mag_factor = float(view_a["source_detector_distance_mm"]) / max(float(view_a["source_patient_distance_mm"]), 1e-6)
    pixel_to_object_mm = float(view_a["imager_pixel_spacing_mm"][0]) / max(mag_factor, 1e-6)

    parts = []
    report = []
    centerlines: Dict[int, Dict[str, object]] = {}
    for branch in branches_a:
        branch_id = int(branch["branch_id"])
        n = min(64, max(20, int(branch["points"])))
        coords_yx = resample_polyline(branch["centerline_yx"], n)
        if branch_id in validated:
            confidence = str(validated[branch_id]["status"])
            points_xyz = np.asarray(validated[branch_id]["points_xyz"], dtype=np.float64)
            coords_for_radius = np.asarray(validated[branch_id]["samples_view_a_yx"], dtype=np.float64)
            anchor_median_px, anchor_min_px = 0.0, 0.0
            note = "two_view_validated"
        else:
            confidence = "estimated"
            points_xyz = estimate_branch_3d(coords_yx, view_a, anchors_yx, anchors_xyz)
            coords_for_radius = coords_yx
            anchor_median_px, anchor_min_px = branch_anchor_distance_px(coords_yx, anchors_yx)
            note = "im0_shape_depth_interpolated"

        radii = sample_radius(dt_a, coords_for_radius, pixel_to_object_mm, confidence)
        if len(radii) != len(points_xyz):
            radii = np.interp(
                np.linspace(0.0, 1.0, len(points_xyz)),
                np.linspace(0.0, 1.0, len(radii)),
                radii,
            )
        vertices, faces = variable_tube(points_xyz, radii, segments=16)
        parts.append((f"branch_{branch_id:02d}_{confidence}", confidence, vertices, faces))
        centerlines[branch_id] = {
            "branch_id": branch_id,
            "confidence": confidence,
            "points_xyz": points_xyz.tolist(),
            "samples_view_a_yx": coords_for_radius.tolist(),
        }
        report.append(
            {
                "branch_id": branch_id,
                "confidence": confidence,
                "source": note,
                "points": len(points_xyz),
                "anchor_median_distance_px": anchor_median_px,
                "anchor_min_distance_px": anchor_min_px,
                "min_radius_mm": float(np.min(radii)),
                "median_radius_mm": float(np.median(radii)),
                "max_radius_mm": float(np.max(radii)),
            }
        )

    obj_path = out / "confidence_layered_tree.obj"
    write_obj(obj_path, parts)
    with open(out / "confidence_layered_centerlines.json", "w", encoding="utf-8") as f:
        json.dump({"branches": list(centerlines.values())}, f, indent=2)
    with open(out / "confidence_layered_branch_report.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(report[0].keys()))
        writer.writeheader()
        writer.writerows(report)
    shutil.copyfile(out / "confidence_layered_branch_report.csv", out / "branch_quality_report.csv")
    draw_confidence_overlay(original_a, clean_a, centerlines, out / "view_a_confidence_layers.png")

    shutil.copyfile(input_dir / "pipeline_summary.json", out / "pipeline_summary.json")
    for name in (
        "view_a_original.png",
        "view_b_original.png",
        "view_a_overlay.png",
        "view_b_overlay.png",
        "view_a_mask.png",
        "view_b_mask.png",
        "view_a_skeleton.png",
        "view_b_skeleton.png",
    ):
        source = input_dir / name
        if source.exists():
            shutil.copyfile(source, out / name)

    counts = {}
    for row in report:
        counts[row["confidence"]] = counts.get(row["confidence"], 0) + 1
    output_summary = {
        "input_dir": str(input_dir),
        "epipolar_dir": str(args.epipolar_dir),
        "output_dir": str(out),
        "num_branches": len(report),
        "confidence_counts": counts,
        "outputs": {
            "obj": str(obj_path),
            "centerlines": str(out / "confidence_layered_centerlines.json"),
            "branch_report": str(out / "confidence_layered_branch_report.csv"),
            "view_a_overlay": str(out / "view_a_confidence_layers.png"),
        },
        "meaning": {
            "reliable": "two-view validated geometry",
            "usable": "two-view supported but lower confidence geometry",
            "estimated": "IM0-visible branch with depth interpolated from validated 3D anchors",
        },
    }
    with open(out / "confidence_layered_summary.json", "w", encoding="utf-8") as f:
        json.dump(output_summary, f, indent=2)
    print(json.dumps(output_summary, indent=2))


if __name__ == "__main__":
    main()
