"""
Export smooth, 2D-reference-faithful coronary tree meshes.

This is a visual reconstruction stage, not a clinically validated depth solve.
It preserves the full artery shape from a selected reference angiogram frame and
adds shallow depth only for viewing. It is useful when epipolar reconstruction is
too sparse but we still need a clean, doctor-readable full tree.

Example:
    python scripts/export_visual_reference_tree.py ^
        --case-id case_026 ^
        --lca-clip 2 --lca-frame 30 ^
        --rca-clip 6 --rca-frame 31 ^
        --output-dir dicom_case_3d_reconstructions/case_026_visual_reference
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from dicom_3d_pipeline import extract_graph, resample_polyline, variable_tube  # noqa: E402
from reconstruct_full_tree_hybrid import make_sphere, smooth_centerline, smooth_1d, write_obj  # noqa: E402


MATERIALS = {
    "lca_visual": (0.86, 0.86, 0.84),
    "rca_visual": (0.20, 0.48, 1.0),
    "junction": (0.74, 0.74, 0.72),
}


def write_materials(path: Path):
    with open(path, "w", encoding="utf-8") as f:
        for name, color in MATERIALS.items():
            f.write(f"newmtl {name}\nKd {color[0]} {color[1]} {color[2]}\n")


def load_gray(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(path)
    return image


def mm_from_yx(coords_yx: np.ndarray, scale_mm_per_px: float = 0.18) -> np.ndarray:
    y = coords_yx[:, 0]
    x = coords_yx[:, 1]
    out_x = (x - 255.5) * scale_mm_per_px
    out_y = -(y - 255.5) * scale_mm_per_px
    return np.column_stack([out_x, out_y])


def shallow_depth(xy: np.ndarray, depth_mm: float = 12.0) -> np.ndarray:
    if len(xy) == 0:
        return np.zeros(0, dtype=np.float64)
    x = xy[:, 0]
    y = xy[:, 1]
    x_norm = (x - np.mean(x)) / max(np.ptp(x), 1.0)
    y_norm = (y - np.mean(y)) / max(np.ptp(y), 1.0)
    return depth_mm * (0.55 * np.sin(math.pi * x_norm) + 0.45 * np.cos(math.pi * y_norm))


def blend_close_endpoints(branches: List[Dict[str, object]], radius_mm: float = 5.0):
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
        cluster = [item]
        used.add(idx)
        changed = True
        while changed:
            changed = False
            center = np.mean([member[2] for member in cluster], axis=0)
            for other_idx, other in enumerate(endpoints):
                if other_idx in used:
                    continue
                if np.linalg.norm(other[2] - center) <= radius_mm:
                    used.add(other_idx)
                    cluster.append(other)
                    changed = True
        if len(cluster) < 2:
            continue
        node = np.mean([member[2] for member in cluster], axis=0)
        node_radius = float(np.mean([member[0]["radii"][0] if member[1] == "start" else member[0]["radii"][-1] for member in cluster]))
        junctions.append((node, node_radius))
        for branch, side, _ in cluster:
            limit = min(10, len(branch["centerline"]))
            indexes = range(limit) if side == "start" else range(len(branch["centerline"]) - 1, len(branch["centerline"]) - 1 - limit, -1)
            for offset, point_idx in enumerate(indexes):
                alpha = 1.0 - offset / max(limit - 1, 1)
                branch["centerline"][point_idx] = alpha * node + (1.0 - alpha) * branch["centerline"][point_idx]
                branch["radii"][point_idx] = alpha * node_radius + (1.0 - alpha) * branch["radii"][point_idx]
    return junctions


def build_tree(mask_path: Path, artery: str, offset_xyz: Tuple[float, float, float] = (0, 0, 0)) -> Tuple[List[Tuple[str, str, list, list]], Dict[str, object]]:
    mask = load_gray(mask_path)
    clean, _, radius_dt, branches = extract_graph(mask)
    parts = []
    branch_payloads = []
    offset = np.asarray(offset_xyz, dtype=np.float64)
    for branch in branches:
        if float(branch["length_px"]) < 45.0:
            continue
        n = min(120, max(40, int(branch["points"]) * 2))
        coords = resample_polyline(branch["centerline_yx"], n)
        xy = mm_from_yx(coords)
        z = shallow_depth(xy)
        centerline = np.column_stack([xy, z])
        centerline = smooth_centerline(centerline, window=11, iterations=3) + offset
        radii = []
        for y, x in coords:
            yy = int(np.clip(round(y), 0, radius_dt.shape[0] - 1))
            xx = int(np.clip(round(x), 0, radius_dt.shape[1] - 1))
            radii.append(max(0.55, min(2.2, float(radius_dt[yy, xx]) * 0.18)))
        radii = smooth_1d(np.asarray(radii, dtype=np.float64), 11)
        branch_payloads.append({"branch_id": int(branch["branch_id"]), "centerline": centerline, "radii": radii})

    junctions = blend_close_endpoints(branch_payloads, radius_mm=5.5)
    material = f"{artery}_visual"
    for branch in branch_payloads:
        vertices, faces = variable_tube(branch["centerline"], branch["radii"], segments=24)
        parts.append((f"{artery}_branch_{branch['branch_id']:02d}", material, vertices, faces))
    for idx, (center, radius) in enumerate(junctions):
        vertices, faces = make_sphere(center, max(0.8, min(2.0, radius * 1.15)))
        parts.append((f"{artery}_junction_{idx:02d}", "junction", vertices, faces))

    summary = {
        "artery": artery.upper(),
        "mask": str(mask_path),
        "source_branch_count": len(branches),
        "exported_branch_count": len(branch_payloads),
        "junction_count": len(junctions),
    }
    return parts, summary


def write_custom_obj(path: Path, parts: List[Tuple[str, str, list, list]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    write_materials(path.with_suffix(".mtl"))
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"mtllib {path.with_suffix('.mtl').name}\n")
        f.write("# Visual reference coronary tree. Shallow depth is for viewing only.\n")
        vertex_offset = 1
        for name, material, vertices, faces in parts:
            f.write(f"o {name}\nusemtl {material}\n")
            for vertex in vertices:
                f.write(f"v {vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f}\n")
            for a, b, c in faces:
                f.write(f"f {a + vertex_offset} {b + vertex_offset} {c + vertex_offset}\n")
            vertex_offset += len(vertices)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--processed-dir", type=Path, default=ROOT / "dicom_dataset_processed")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--lca-clip", type=int, required=True)
    parser.add_argument("--lca-frame", type=int, required=True)
    parser.add_argument("--rca-clip", type=int, required=True)
    parser.add_argument("--rca-frame", type=int, required=True)
    args = parser.parse_args()

    frame_dir = args.processed_dir / args.case_id / "sampled_frames"
    lca_mask = frame_dir / f"clip_{args.lca_clip:02d}_frame_{args.lca_frame:03d}_mask.png"
    rca_mask = frame_dir / f"clip_{args.rca_clip:02d}_frame_{args.rca_frame:03d}_mask.png"
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    lca_parts, lca_summary = build_tree(lca_mask, "lca", offset_xyz=(-45.0, 0.0, 0.0))
    rca_parts, rca_summary = build_tree(rca_mask, "rca", offset_xyz=(45.0, 0.0, 0.0))
    write_custom_obj(output_dir / f"{args.case_id}_lca_visual_tree.obj", lca_parts)
    write_custom_obj(output_dir / f"{args.case_id}_rca_visual_tree.obj", rca_parts)
    write_custom_obj(output_dir / f"{args.case_id}_combined_visual_coronary_tree.obj", lca_parts + rca_parts)

    summary = {
        "case_id": args.case_id,
        "warning": "This is a visual reference tree. It preserves 2D artery shape but does not claim clinically validated 3D depth.",
        "lca": lca_summary,
        "rca": rca_summary,
        "outputs": {
            "lca_obj": str(output_dir / f"{args.case_id}_lca_visual_tree.obj"),
            "rca_obj": str(output_dir / f"{args.case_id}_rca_visual_tree.obj"),
            "combined_obj": str(output_dir / f"{args.case_id}_combined_visual_coronary_tree.obj"),
        },
    }
    with open(output_dir / "visual_reference_tree_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
