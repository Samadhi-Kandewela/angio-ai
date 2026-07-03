"""
Validate a reconstructed coronary OBJ by projecting its branch centerlines back
onto the original two DICOM segmentation views.

Outputs:
    reprojection_validation_report.csv
    reprojection_validation_summary.json
    reprojection_validation_report.md
    view_a_reprojection_validation.png
    view_b_reprojection_validation.png

Example:
    python scripts/reprojection_validation_report.py --input-dir dicom_pipeline_output_v2
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ORIENTATION_MODES = ("normal", "rotate_180", "mirror_h", "mirror_v")


@dataclass
class BranchCenterline:
    branch_id: int
    status: str
    points: np.ndarray


def infer_segments(vertex_count: int) -> int:
    for candidate in (20, 16, 12, 10, 8):
        if vertex_count >= candidate * 2 and vertex_count % candidate == 0:
            return candidate
    return 20


def parse_obj_centerlines(path: Path) -> List[BranchCenterline]:
    vertices: List[Tuple[float, float, float]] = []
    branches: List[BranchCenterline] = []
    current_name = ""
    current_status = "reliable"
    current_start = 0
    current_count = 0

    def flush_current():
        nonlocal current_name, current_start, current_count, current_status
        if not current_name or current_count <= 0:
            return
        match = re.search(r"branch[_-](\d+)", current_name)
        if not match:
            return
        branch_id = int(match.group(1))
        segments = infer_segments(current_count)
        usable_count = current_count - (current_count % segments)
        if usable_count < segments * 2:
            return
        data = np.asarray(vertices[current_start: current_start + usable_count], dtype=np.float64)
        rings = data.reshape(usable_count // segments, segments, 3)
        centerline = rings.mean(axis=1)
        branches.append(BranchCenterline(branch_id=branch_id, status=current_status, points=centerline))

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split()
            if fields[0] == "o":
                flush_current()
                current_name = " ".join(fields[1:])
                current_status = "reliable"
                current_start = len(vertices)
                current_count = 0
            elif fields[0] == "usemtl":
                current_status = fields[1] if len(fields) > 1 else "reliable"
            elif fields[0] == "v" and len(fields) >= 4:
                vertices.append((float(fields[1]), float(fields[2]), float(fields[3])))
                current_count += 1
    flush_current()
    return sorted(branches, key=lambda branch: branch.branch_id)


def rotation_from_angles(primary_deg: float, secondary_deg: float) -> np.ndarray:
    primary = math.radians(primary_deg)
    secondary = math.radians(secondary_deg)
    ry = np.array(
        [
            [math.cos(primary), 0.0, math.sin(primary)],
            [0.0, 1.0, 0.0],
            [-math.sin(primary), 0.0, math.cos(primary)],
        ],
        dtype=np.float64,
    )
    rx = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, math.cos(secondary), -math.sin(secondary)],
            [0.0, math.sin(secondary), math.cos(secondary)],
        ],
        dtype=np.float64,
    )
    return ry @ rx


def project_points(points_xyz: np.ndarray, view: Dict[str, object]) -> np.ndarray:
    rows, cols = [int(v) for v in view["image_size_px"]]
    spacing_y, spacing_x = [float(v) for v in view["imager_pixel_spacing_mm"]]
    sdd = float(view["source_detector_distance_mm"])
    spd = float(view["source_patient_distance_mm"])
    det_z = sdd - spd
    source_local = np.array([0.0, 0.0, -spd], dtype=np.float64)
    rotation = rotation_from_angles(float(view["primary_angle_deg"]), float(view["secondary_angle_deg"]))

    projected = []
    for point in points_xyz:
        local = rotation.T @ point
        denom = local[2] - source_local[2]
        if abs(float(denom)) < 1e-6:
            projected.append((np.nan, np.nan))
            continue
        t = (det_z - source_local[2]) / denom
        detector = source_local + t * (local - source_local)
        x = detector[0] / spacing_x + (cols - 1) * 0.5
        y = detector[1] / spacing_y + (rows - 1) * 0.5
        projected.append((x, y))
    return np.asarray(projected, dtype=np.float64)


def orient_pixels(points_xy: np.ndarray, mode: str, width: int, height: int) -> np.ndarray:
    out = points_xy.copy()
    if mode == "rotate_180":
        out[:, 0] = width - 1 - out[:, 0]
        out[:, 1] = height - 1 - out[:, 1]
    elif mode == "mirror_h":
        out[:, 0] = width - 1 - out[:, 0]
    elif mode == "mirror_v":
        out[:, 1] = height - 1 - out[:, 1]
    return out


def distance_stats(points_xy: np.ndarray, distance_map: np.ndarray) -> Dict[str, float]:
    height, width = distance_map.shape[:2]
    finite = np.isfinite(points_xy).all(axis=1)
    inside = (
        finite
        & (points_xy[:, 0] >= 0)
        & (points_xy[:, 0] <= width - 1)
        & (points_xy[:, 1] >= 0)
        & (points_xy[:, 1] <= height - 1)
    )
    if not np.any(inside):
        return {
            "samples": int(len(points_xy)),
            "inside_pct": 0.0,
            "median_px": float("nan"),
            "mean_px": float("nan"),
            "p90_px": float("nan"),
            "max_px": float("nan"),
        }
    pixels = np.rint(points_xy[inside]).astype(int)
    distances = distance_map[pixels[:, 1], pixels[:, 0]].astype(np.float64)
    return {
        "samples": int(len(points_xy)),
        "inside_pct": float(np.count_nonzero(inside) / max(len(points_xy), 1) * 100.0),
        "median_px": float(np.median(distances)),
        "mean_px": float(np.mean(distances)),
        "p90_px": float(np.percentile(distances, 90)),
        "max_px": float(np.max(distances)),
    }


def choose_orientation(branches: Iterable[BranchCenterline], view: Dict[str, object], distance_map: np.ndarray) -> Tuple[str, Dict[str, float]]:
    height, width = distance_map.shape[:2]
    all_points = np.vstack([project_points(branch.points, view) for branch in branches])
    scores = {}
    for mode in ORIENTATION_MODES:
        stats = distance_stats(orient_pixels(all_points, mode, width, height), distance_map)
        objective = stats["median_px"]
        if not math.isfinite(objective):
            objective = 1e9
        objective += (100.0 - stats["inside_pct"]) * 0.05
        scores[mode] = {**stats, "objective": float(objective)}
    best_mode = min(scores, key=lambda key: scores[key]["objective"])
    return best_mode, scores


def status_from_error(median_px: float, p90_px: float, inside_pct: float) -> str:
    if not math.isfinite(median_px) or inside_pct < 50.0:
        return "bad"
    if median_px <= 3.0 and p90_px <= 8.0:
        return "good"
    if median_px <= 7.0 and p90_px <= 16.0:
        return "review"
    return "bad"


def draw_overlay(
    original: np.ndarray,
    mask: np.ndarray,
    projected_by_branch: Dict[int, np.ndarray],
    rows: List[Dict[str, object]],
    output_path: Path,
):
    if original.ndim == 2:
        canvas = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
    else:
        canvas = original.copy()
    if canvas.shape[:2] != mask.shape[:2]:
        canvas = cv2.resize(canvas, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_AREA)

    red = canvas.copy()
    red[mask > 0] = (35, 35, 255)
    canvas = cv2.addWeighted(red, 0.35, canvas, 0.65, 0)

    row_by_branch = {int(row["branch_id"]): row for row in rows}
    colors = {
        "good": (0, 255, 80),
        "review": (0, 210, 255),
        "bad": (0, 0, 255),
    }
    for branch_id, points in projected_by_branch.items():
        row = row_by_branch.get(branch_id, {})
        color = colors.get(str(row.get("validation_status", "bad")), (0, 0, 255))
        finite = np.isfinite(points).all(axis=1)
        pts = np.rint(points[finite]).astype(int)
        if len(pts) < 2:
            continue
        for a, b in zip(pts[:-1], pts[1:]):
            cv2.line(canvas, tuple(a), tuple(b), color, 1, cv2.LINE_AA)
        cv2.circle(canvas, tuple(pts[0]), 3, color, -1, cv2.LINE_AA)
        cv2.putText(canvas, str(branch_id), tuple(pts[len(pts) // 2]), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)
    cv2.imwrite(str(output_path), canvas)


def load_gray(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(path)
    return image


def validate_view(
    view_key: str,
    branches: List[BranchCenterline],
    view: Dict[str, object],
    original: np.ndarray,
    mask: np.ndarray,
    output_dir: Path,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    vessel = (mask > 0).astype(np.uint8)
    distance_map = cv2.distanceTransform((vessel == 0).astype(np.uint8), cv2.DIST_L2, 5)
    height, width = distance_map.shape[:2]
    best_orientation, orientation_scores = choose_orientation(branches, view, distance_map)

    rows = []
    projected_by_branch = {}
    spacing_y, spacing_x = [float(v) for v in view["imager_pixel_spacing_mm"]]
    px_to_detector_mm = (spacing_x + spacing_y) * 0.5
    for branch in branches:
        projected = orient_pixels(project_points(branch.points, view), best_orientation, width, height)
        projected_by_branch[branch.branch_id] = projected
        stats = distance_stats(projected, distance_map)
        validation_status = status_from_error(stats["median_px"], stats["p90_px"], stats["inside_pct"])
        rows.append(
            {
                "view": view_key,
                "branch_id": branch.branch_id,
                "branch_status": branch.status,
                "best_orientation": best_orientation,
                "validation_status": validation_status,
                "samples": stats["samples"],
                "inside_pct": stats["inside_pct"],
                "median_error_px": stats["median_px"],
                "mean_error_px": stats["mean_px"],
                "p90_error_px": stats["p90_px"],
                "max_error_px": stats["max_px"],
                "median_error_detector_mm": stats["median_px"] * px_to_detector_mm if math.isfinite(stats["median_px"]) else float("nan"),
                "p90_error_detector_mm": stats["p90_px"] * px_to_detector_mm if math.isfinite(stats["p90_px"]) else float("nan"),
            }
        )

    draw_overlay(original, mask, projected_by_branch, rows, output_dir / f"{view_key}_reprojection_validation.png")
    summary = {
        "best_orientation": best_orientation,
        "orientation_scores": orientation_scores,
        "median_branch_error_px": float(np.nanmedian([row["median_error_px"] for row in rows])),
        "p90_branch_error_px": float(np.nanmedian([row["p90_error_px"] for row in rows])),
        "good_branches": int(sum(row["validation_status"] == "good" for row in rows)),
        "review_branches": int(sum(row["validation_status"] == "review" for row in rows)),
        "bad_branches": int(sum(row["validation_status"] == "bad" for row in rows)),
    }
    return rows, summary


def write_markdown(path: Path, summary: Dict[str, object], csv_name: str):
    lines = [
        "# Reprojection Validation Report",
        "",
        "This report projects the reconstructed 3D artery centerlines back onto the two selected DICOM segmentation views.",
        "Lower error means the 3D branch lands closer to the 2D segmented artery mask.",
        "",
        "Important: errors are measured against the current automatic segmentation mask, not against expert ground truth.",
        "",
        f"CSV: `{csv_name}`",
        "",
    ]
    for view_key in ("view_a", "view_b"):
        view_summary = summary["views"][view_key]
        lines.extend(
            [
                f"## {view_key}",
                "",
                f"- Best image orientation: `{view_summary['best_orientation']}`",
                f"- Median branch error: `{view_summary['median_branch_error_px']:.2f}` px",
                f"- P90 branch error: `{view_summary['p90_branch_error_px']:.2f}` px",
                f"- Good / review / bad branches: `{view_summary['good_branches']} / {view_summary['review_branches']} / {view_summary['bad_branches']}`",
                f"- Overlay: `{view_key}_reprojection_validation.png`",
                "",
            ]
        )
    lines.extend(
        [
            "## How to read the overlay",
            "",
            "- Red transparent area: 2D segmentation mask.",
            "- Green centerline: good reprojection.",
            "- Yellow centerline: needs review.",
            "- Red centerline: bad reprojection.",
            "",
            "A clinically accurate 3D reconstruction should have low reprojection error in both views, especially around bifurcations and stenosis regions.",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=ROOT / "dicom_pipeline_output_v2")
    parser.add_argument("--output-dir", type=Path, help="Where to write validation outputs. Defaults to input dir.")
    parser.add_argument("--obj", type=Path, help="OBJ to validate. Defaults to smoothed OBJ, then raw pipeline OBJ.")
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir or input_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    obj_path = args.obj
    if obj_path is None:
        smoothed = input_dir / "pipeline_hybrid_qca_tree_smoothed.obj"
        raw = input_dir / "pipeline_hybrid_qca_tree.obj"
        obj_path = smoothed if smoothed.exists() else raw

    summary_path = input_dir / "pipeline_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    with open(summary_path, "r", encoding="utf-8") as f:
        pipeline_summary = json.load(f)

    branches = parse_obj_centerlines(obj_path)
    if not branches:
        raise RuntimeError(f"No branch centerlines could be recovered from {obj_path}")

    view_inputs = {
        "view_a": {
            "view": pipeline_summary["view_a"],
            "original": load_gray(input_dir / "view_a_original.png"),
            "mask": load_gray(input_dir / "view_a_clean_mask.png"),
        },
        "view_b": {
            "view": pipeline_summary["view_b"],
            "original": load_gray(input_dir / "view_b_original.png"),
            "mask": load_gray(input_dir / "view_b_clean_mask.png"),
        },
    }

    all_rows: List[Dict[str, object]] = []
    view_summaries: Dict[str, object] = {}
    for view_key, data in view_inputs.items():
        rows, view_summary = validate_view(
            view_key=view_key,
            branches=branches,
            view=data["view"],
            original=data["original"],
            mask=data["mask"],
            output_dir=output_dir,
        )
        all_rows.extend(rows)
        view_summaries[view_key] = view_summary

    csv_path = output_dir / "reprojection_validation_report.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    full_summary = {
        "input_obj": str(obj_path),
        "num_branches": len(branches),
        "views": view_summaries,
        "outputs": {
            "csv": str(csv_path),
            "markdown": str(output_dir / "reprojection_validation_report.md"),
            "view_a_overlay": str(output_dir / "view_a_reprojection_validation.png"),
            "view_b_overlay": str(output_dir / "view_b_reprojection_validation.png"),
        },
    }
    with open(output_dir / "reprojection_validation_summary.json", "w", encoding="utf-8") as f:
        json.dump(full_summary, f, indent=2)
    write_markdown(output_dir / "reprojection_validation_report.md", full_summary, csv_path.name)
    print(json.dumps(full_summary, indent=2))


if __name__ == "__main__":
    main()
