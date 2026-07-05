"""
Smoothed, junction-aware QCA-radius artery mesh generator.

Input:
    OBJ + CSV produced by scripts/dicom_3d_pipeline.py

Output:
    A cleaner OBJ with smoothed centerlines/radii, branch-end blending, and
    junction spheres that visually fuse nearby branch endpoints.

Example:
    python scripts/smooth_junction_mesh.py --input-dir dicom_pipeline_output_v2
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class BranchMesh:
    branch_id: int
    status: str
    vertices: np.ndarray
    rings: int
    segments: int
    radii: np.ndarray
    centerline: np.ndarray


def parse_branch_obj(path: Path) -> List[BranchMesh]:
    vertices: List[Tuple[float, float, float]] = []
    branches: List[BranchMesh] = []
    current_name = ""
    current_status = "reliable"
    current_start_vertex = 0
    current_vertex_count = 0

    def flush_current():
        nonlocal current_vertex_count, current_start_vertex, current_name, current_status
        if not current_name or current_vertex_count <= 0:
            return
        match = re.search(r"branch[_-](\d+)", current_name)
        if not match:
            return
        branch_id = int(match.group(1))
        segments = infer_segments(current_vertex_count)
        rings = current_vertex_count // segments
        data = np.asarray(vertices[current_start_vertex: current_start_vertex + current_vertex_count], dtype=np.float64)
        ring_points = data.reshape(rings, segments, 3)
        centerline = ring_points.mean(axis=1)
        radii = np.linalg.norm(ring_points - centerline[:, None, :], axis=2).mean(axis=1)
        branches.append(
            BranchMesh(
                branch_id=branch_id,
                status=current_status,
                vertices=data,
                rings=rings,
                segments=segments,
                radii=radii,
                centerline=centerline,
            )
        )

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
                current_start_vertex = len(vertices)
                current_vertex_count = 0
            elif fields[0] == "usemtl":
                current_status = fields[1] if len(fields) > 1 else "reliable"
            elif fields[0] == "v" and len(fields) >= 4:
                vertices.append((float(fields[1]), float(fields[2]), float(fields[3])))
                current_vertex_count += 1
    flush_current()
    return branches


def infer_segments(vertex_count: int) -> int:
    for candidate in (16, 12, 10, 8):
        if vertex_count % candidate == 0:
            return candidate
    return 12


def load_report(input_dir: Path) -> Dict[int, Dict[str, str]]:
    for name in ("pipeline_branch_quality_radius.csv", "hybrid_qca_radius_report.csv"):
        path = input_dir / name
        if path.exists():
            with open(path, newline="", encoding="utf-8") as f:
                return {int(row["branch_id"]): row for row in csv.DictReader(f)}
    return {}


def resample_polyline(points: np.ndarray, values: np.ndarray, n: int) -> Tuple[np.ndarray, np.ndarray]:
    if len(points) <= 1:
        return np.repeat(points[:1], n, axis=0), np.repeat(values[:1], n)
    dist = np.zeros(len(points), dtype=np.float64)
    dist[1:] = np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))
    if dist[-1] <= 1e-6:
        return np.repeat(points[:1], n, axis=0), np.repeat(values[:1], n)
    target = np.linspace(0.0, dist[-1], n)
    out = np.column_stack([np.interp(target, dist, points[:, i]) for i in range(3)])
    out_values = np.interp(target, dist, values)
    return out, out_values


def smooth_1d(values: np.ndarray, window: int) -> np.ndarray:
    if len(values) < window:
        return values
    if window % 2 == 0:
        window += 1
    pad = window // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(padded, kernel, mode="valid")


def smooth_centerline(points: np.ndarray, window: int = 9, iterations: int = 2) -> np.ndarray:
    out = points.copy()
    for _ in range(iterations):
        for dim in range(3):
            out[:, dim] = smooth_1d(out[:, dim], min(window, max(3, len(out) // 2 * 2 - 1)))
        out[0] = points[0]
        out[-1] = points[-1]
    return out


def robust_radius_smooth(radii: np.ndarray, stenosis_preserve: bool = True) -> np.ndarray:
    r = np.asarray(radii, dtype=np.float64)
    if len(r) >= 7:
        median = np.median(r)
        lo = max(0.12, np.percentile(r, 5) * 0.8)
        hi = max(lo + 0.05, np.percentile(r, 95) * 1.15)
        r = np.clip(r, lo, hi)
        smooth = smooth_1d(r, 11)
        if stenosis_preserve:
            # Preserve true local narrowings while removing ring-to-ring chatter.
            r = np.minimum(smooth * 1.03, np.maximum(r, smooth * 0.72))
            r = smooth_1d(r, 7)
        else:
            r = smooth
    return np.clip(r, 0.12, 2.4)


def cluster_endpoints(branches: List[BranchMesh], radius: float = 3.0):
    endpoints = []
    for branch in branches:
        endpoints.append((branch, "start", branch.centerline[0]))
        endpoints.append((branch, "end", branch.centerline[-1]))

    clusters = []
    used = set()
    for idx, item in enumerate(endpoints):
        if idx in used:
            continue
        used.add(idx)
        members = [item]
        changed = True
        while changed:
            changed = False
            center = np.mean([m[2] for m in members], axis=0)
            for j, other in enumerate(endpoints):
                if j in used:
                    continue
                if np.linalg.norm(other[2] - center) <= radius:
                    used.add(j)
                    members.append(other)
                    changed = True
        clusters.append(members)
    return clusters


def blend_junctions(branches: List[BranchMesh]):
    clusters = cluster_endpoints(branches)
    junctions = []
    by_id = {branch.branch_id: branch for branch in branches}
    for members in clusters:
        if len(members) < 2:
            continue
        weights = []
        points = []
        radii = []
        for branch, side, point in members:
            weight = 3.0 if branch.status == "reliable" else 2.0 if branch.status == "usable" else 0.8
            radius = branch.radii[0] if side == "start" else branch.radii[-1]
            weights.append(weight)
            points.append(point)
            radii.append(radius)
        node = np.average(np.asarray(points), axis=0, weights=np.asarray(weights))
        node_radius = float(np.average(np.asarray(radii), weights=np.asarray(weights)))
        junctions.append((node, node_radius))

        for branch, side, _ in members:
            limit = min(10, len(branch.centerline))
            if side == "start":
                for i in range(limit):
                    alpha = 1.0 - i / max(limit - 1, 1)
                    branch.centerline[i] = alpha * node + (1.0 - alpha) * branch.centerline[i]
                    branch.radii[i] = alpha * node_radius + (1.0 - alpha) * branch.radii[i]
            else:
                for offset in range(limit):
                    i = len(branch.centerline) - 1 - offset
                    alpha = 1.0 - offset / max(limit - 1, 1)
                    branch.centerline[i] = alpha * node + (1.0 - alpha) * branch.centerline[i]
                    branch.radii[i] = alpha * node_radius + (1.0 - alpha) * branch.radii[i]
    return junctions


def endpoint_tangent(branch: BranchMesh, side: str) -> np.ndarray:
    if len(branch.centerline) < 2:
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if side == "start":
        tangent = branch.centerline[0] - branch.centerline[1]
    else:
        tangent = branch.centerline[-1] - branch.centerline[-2]
    norm = np.linalg.norm(tangent)
    return tangent / max(norm, 1e-9)


def endpoint_radius(branch: BranchMesh, side: str) -> float:
    return float(branch.radii[0] if side == "start" else branch.radii[-1])


def make_estimated_connectors(branches: List[BranchMesh], min_gap: float = 3.2, max_gap: float = 15.0, max_connectors: int = 10):
    endpoints = []
    for branch in branches:
        endpoints.append(
            {
                "branch": branch,
                "branch_id": branch.branch_id,
                "side": "start",
                "point": branch.centerline[0],
                "tangent": endpoint_tangent(branch, "start"),
                "radius": endpoint_radius(branch, "start"),
            }
        )
        endpoints.append(
            {
                "branch": branch,
                "branch_id": branch.branch_id,
                "side": "end",
                "point": branch.centerline[-1],
                "tangent": endpoint_tangent(branch, "end"),
                "radius": endpoint_radius(branch, "end"),
            }
        )

    candidates = []
    for i, a in enumerate(endpoints):
        for b in endpoints[i + 1:]:
            if a["branch_id"] == b["branch_id"]:
                continue
            delta = b["point"] - a["point"]
            gap = float(np.linalg.norm(delta))
            if gap < min_gap or gap > max_gap:
                continue
            direction = delta / max(gap, 1e-9)
            tangent_a = float(np.dot(a["tangent"], direction))
            tangent_b = float(np.dot(b["tangent"], -direction))
            if tangent_a < -0.15 or tangent_b < -0.15:
                continue
            radius = max(0.35, min(1.2, 0.85 * 0.5 * (a["radius"] + b["radius"])))
            score = gap - 2.0 * max(tangent_a, 0.0) - 2.0 * max(tangent_b, 0.0)
            candidates.append((score, gap, tangent_a, tangent_b, radius, a, b))

    selected = []
    used_endpoints = set()
    for _, gap, tangent_a, tangent_b, radius, a, b in sorted(candidates, key=lambda item: item[0]):
        key_a = (a["branch_id"], a["side"])
        key_b = (b["branch_id"], b["side"])
        if key_a in used_endpoints or key_b in used_endpoints:
            continue
        used_endpoints.add(key_a)
        used_endpoints.add(key_b)
        p0 = a["point"]
        p3 = b["point"]
        handle = min(gap * 0.38, 5.0)
        p1 = p0 + a["tangent"] * handle
        p2 = p3 + b["tangent"] * handle
        t = np.linspace(0.0, 1.0, 16)[:, None]
        curve = ((1 - t) ** 3) * p0 + 3 * ((1 - t) ** 2) * t * p1 + 3 * (1 - t) * (t ** 2) * p2 + (t ** 3) * p3
        curve = smooth_centerline(curve, window=5, iterations=1)
        radii = np.full(len(curve), radius, dtype=np.float64)
        selected.append(
            {
                "branch_a": int(a["branch_id"]),
                "side_a": a["side"],
                "branch_b": int(b["branch_id"]),
                "side_b": b["side"],
                "gap_mm": round(gap, 3),
                "tangent_a": round(tangent_a, 3),
                "tangent_b": round(tangent_b, 3),
                "radius_mm": round(radius, 3),
                "points": curve,
                "radii": radii,
            }
        )
        if len(selected) >= max_connectors:
            break
    return selected


def make_frame(tangent: np.ndarray):
    tangent = tangent / max(np.linalg.norm(tangent), 1e-9)
    ref = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(np.dot(tangent, ref))) > 0.9:
        ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    normal = np.cross(tangent, ref)
    normal /= max(np.linalg.norm(normal), 1e-9)
    binormal = np.cross(tangent, normal)
    binormal /= max(np.linalg.norm(binormal), 1e-9)
    return normal, binormal


def make_tube(points: np.ndarray, radii: np.ndarray, segments: int = 20):
    vertices, faces = [], []
    for i, point in enumerate(points):
        tangent = points[min(i + 1, len(points) - 1)] - points[max(i - 1, 0)]
        normal, binormal = make_frame(tangent)
        for seg in range(segments):
            angle = 2.0 * np.pi * seg / segments
            vertices.append(point + radii[i] * (np.cos(angle) * normal + np.sin(angle) * binormal))
    for i in range(len(points) - 1):
        a = i * segments
        b = (i + 1) * segments
        for seg in range(segments):
            faces.append((a + seg, b + seg, b + ((seg + 1) % segments)))
            faces.append((a + seg, b + ((seg + 1) % segments), a + ((seg + 1) % segments)))
    return vertices, faces


def make_sphere(center: np.ndarray, radius: float, rings: int = 10, segments: int = 16):
    vertices, faces = [], []
    for ring in range(rings + 1):
        phi = np.pi * ring / rings
        for seg in range(segments):
            theta = 2.0 * np.pi * seg / segments
            vertices.append(center + radius * np.array([
                np.sin(phi) * np.cos(theta),
                np.sin(phi) * np.sin(theta),
                np.cos(phi),
            ]))
    for ring in range(rings):
        for seg in range(segments):
            a = ring * segments + seg
            b = ring * segments + ((seg + 1) % segments)
            c = (ring + 1) * segments + seg
            d = (ring + 1) * segments + ((seg + 1) % segments)
            faces.append((a, c, d))
            faces.append((a, d, b))
    return vertices, faces


def write_obj(path: Path, parts):
    with open(path.with_suffix(".mtl"), "w", encoding="utf-8") as f:
        f.write("newmtl reliable\nKd 0.78 0.78 0.78\n")
        f.write("newmtl usable\nKd 0.45 0.65 1.0\n")
        f.write("newmtl single_view_preserved\nKd 1.0 0.58 0.12\n")
        f.write("newmtl estimated_connector\nKd 0.68 0.70 0.72\n")
        f.write("newmtl junction\nKd 0.86 0.86 0.86\n")
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"mtllib {path.with_suffix('.mtl').name}\n")
        f.write("# Smoothed junction-aware QCA-radius artery mesh.\n")
        offset = 1
        for name, material, vertices, faces in parts:
            f.write(f"o {name}\nusemtl {material}\n")
            for v in vertices:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            for a, b, c in faces:
                f.write(f"f {a + offset} {b + offset} {c + offset}\n")
            offset += len(vertices)


def process(input_dir: Path, output_obj: Path):
    source_obj = input_dir / "pipeline_hybrid_qca_tree.obj"
    if not source_obj.exists():
        raise FileNotFoundError(source_obj)

    branches = parse_branch_obj(source_obj)
    report = load_report(input_dir)

    processed = []
    for branch in branches:
        row = report.get(branch.branch_id, {})
        status = row.get("status", branch.status)
        branch.status = status
        target_n = max(28, min(96, int(np.ceil(len(branch.centerline) * 0.9))))
        points, radii = resample_polyline(branch.centerline, branch.radii, target_n)
        branch.centerline = smooth_centerline(points, window=9, iterations=2)
        branch.radii = robust_radius_smooth(radii, stenosis_preserve=(status != "single_view_preserved"))
        if status == "single_view_preserved":
            branch.radii *= 0.88
        processed.append(branch)

    junctions = blend_junctions(processed)

    parts = []
    smooth_report = []
    for branch in processed:
        vertices, faces = make_tube(branch.centerline, branch.radii, segments=20)
        parts.append((f"branch_{branch.branch_id:02d}_{branch.status}", branch.status, vertices, faces))
        min_radius = float(np.min(branch.radii))
        ref_radius = float(np.percentile(branch.radii, 80))
        smooth_report.append({
            "branch_id": branch.branch_id,
            "status": branch.status,
            "min_radius_mm": round(min_radius, 3),
            "ref_radius_mm_p80": round(ref_radius, 3),
            "estimated_diameter_stenosis_pct": round(max(0.0, (1.0 - min_radius / max(ref_radius, 1e-6)) * 100.0), 1),
            "points": len(branch.centerline),
        })

    for idx, (center, radius) in enumerate(junctions):
        vertices, faces = make_sphere(center, max(0.45, min(1.3, radius * 1.08)))
        parts.append((f"junction_{idx:02d}", "junction", vertices, faces))

    connectors = make_estimated_connectors(processed)
    connector_report = []
    for idx, connector in enumerate(connectors):
        vertices, faces = make_tube(connector["points"], connector["radii"], segments=10)
        parts.append((f"estimated_connector_{idx:02d}_branch_{connector['branch_a']:02d}_{connector['branch_b']:02d}", "estimated_connector", vertices, faces))
        connector_report.append(
            {
                "connector_id": idx,
                "branch_a": connector["branch_a"],
                "side_a": connector["side_a"],
                "branch_b": connector["branch_b"],
                "side_b": connector["side_b"],
                "gap_mm": connector["gap_mm"],
                "tangent_a": connector["tangent_a"],
                "tangent_b": connector["tangent_b"],
                "radius_mm": connector["radius_mm"],
                "status": "estimated_connector",
            }
        )

    write_obj(output_obj, parts)
    with open(output_obj.with_name("smoothed_junction_radius_report.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(smooth_report[0].keys()))
        writer.writeheader()
        writer.writerows(smooth_report)
    with open(output_obj.with_name("smoothed_junction_summary.json"), "w", encoding="utf-8") as f:
        json.dump({
            "input_obj": str(source_obj),
            "output_obj": str(output_obj),
            "branches": len(processed),
            "junction_blend_nodes": len(junctions),
            "estimated_connectors": len(connectors),
            "report": str(output_obj.with_name("smoothed_junction_radius_report.csv")),
            "connector_report": str(output_obj.with_name("estimated_connector_report.csv")),
        }, f, indent=2)
    if connector_report:
        with open(output_obj.with_name("estimated_connector_report.csv"), "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(connector_report[0].keys()))
            writer.writeheader()
            writer.writerows(connector_report)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("dicom_pipeline_output_v2"))
    parser.add_argument("--output-obj", type=Path)
    args = parser.parse_args()
    output_obj = args.output_obj or args.input_dir / "pipeline_hybrid_qca_tree_smoothed.obj"
    process(args.input_dir, output_obj)
    print(f"Wrote: {output_obj}")
    print(f"Report: {output_obj.with_name('smoothed_junction_radius_report.csv')}")


if __name__ == "__main__":
    main()
