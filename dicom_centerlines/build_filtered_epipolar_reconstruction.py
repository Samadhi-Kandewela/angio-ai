from pathlib import Path
import csv
import json

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RBFInterpolator

from triangulate_rough_skeleton import (
    closest_point_between_rays,
    project_point,
    ray_from_pixel,
    resample_polyline,
)


ROOT = Path(__file__).resolve().parents[1]
CENTERLINE_DIR = ROOT / "dicom_centerlines"
SELECTED_DIR = ROOT / "dicom_selected_segmentation"
OUT_DIR = ROOT / "dicom_3d_filtered"

VIEW_A = {
    "name": "IM0_frame_030",
    "primary_angle_deg": -0.1,
    "secondary_angle_deg": -21.9,
    "source_detector_distance_mm": 1027.0,
    "source_patient_distance_mm": 720.0,
    "imager_pixel_spacing_mm": [0.2375, 0.2375],
    "image_size_px": [512, 512],
}

VIEW_B = {
    "name": "IM2_frame_022",
    "primary_angle_deg": 40.7,
    "secondary_angle_deg": -40.7,
    "source_detector_distance_mm": 1188.0,
    "source_patient_distance_mm": 720.0,
    "imager_pixel_spacing_mm": [0.2375, 0.2375],
    "image_size_px": [512, 512],
}


def load_graph(view_name):
    with open(CENTERLINE_DIR / f"{view_name}_centerline_graph.json", "r", encoding="utf-8") as f:
        graph = json.load(f)
    return [branch for branch in graph["branches"] if int(branch["points"]) >= 23]


def branch_points(branch, reverse=False):
    pts = list(branch["centerline_yx"])
    if reverse:
        pts = list(reversed(pts))
    return pts


def score_pair(branch_a, branch_b, rev_a, rev_b, samples=48):
    pts_a = resample_polyline(branch_points(branch_a, rev_a), samples)
    pts_b = resample_polyline(branch_points(branch_b, rev_b), samples)
    residuals = []
    points_3d = []
    for point_a, point_b in zip(pts_a, pts_b):
        source_a, dir_a = ray_from_pixel(point_a, VIEW_A)
        source_b, dir_b = ray_from_pixel(point_b, VIEW_B)
        point_3d, residual = closest_point_between_rays(source_a, dir_a, source_b, dir_b)
        residuals.append(residual)
        points_3d.append(point_3d)
    residuals = np.asarray(residuals, dtype=np.float64)
    return {
        "mean": float(np.mean(residuals)),
        "median": float(np.median(residuals)),
        "p90": float(np.percentile(residuals, 90)),
        "max": float(np.max(residuals)),
        "points_3d": np.asarray(points_3d, dtype=np.float64),
        "samples_a": pts_a,
        "samples_b": pts_b,
        "residuals": residuals,
    }


def classify_match(score, branch_a):
    length = float(branch_a["length_px"])
    if score["median"] <= 3.0 and score["p90"] <= 7.0:
        return "reliable"
    if score["median"] <= 5.5 and score["p90"] <= 12.0 and length >= 55:
        return "usable"
    return "single_view_preserved"


def find_best_matches(branches_a, branches_b):
    results = []
    for branch_a in branches_a:
        best = None
        for branch_b in branches_b:
            for rev_a in (False, True):
                for rev_b in (False, True):
                    samples = int(min(64, max(24, min(branch_a["points"], branch_b["points"]))))
                    score = score_pair(branch_a, branch_b, rev_a, rev_b, samples=samples)
                    length_ratio = max(float(branch_a["length_px"]), 1.0) / max(float(branch_b["length_px"]), 1.0)
                    length_penalty = abs(np.log(length_ratio))
                    objective = score["median"] + 0.15 * score["p90"] + 1.5 * length_penalty
                    candidate = {
                        "branch_a": branch_a,
                        "branch_b": branch_b,
                        "reverse_a": rev_a,
                        "reverse_b": rev_b,
                        "score": score,
                        "objective": float(objective),
                        "length_ratio": float(length_ratio),
                    }
                    if best is None or candidate["objective"] < best["objective"]:
                        best = candidate
        best["status"] = classify_match(best["score"], branch_a)
        results.append(best)
    return results


def load_depth_initializer(stereo_results):
    yx = []
    xyz = []
    for result in stereo_results:
        if result["status"] in {"reliable", "usable"}:
            for point_yx, point_3d in zip(result["score"]["samples_a"], result["score"]["points_3d"]):
                yx.append(point_yx)
                xyz.append(point_3d)
    if len(yx) < 8:
        raise RuntimeError("Not enough reliable stereo samples for depth interpolation.")
    return RBFInterpolator(np.asarray(yx, dtype=np.float64), np.asarray(xyz, dtype=np.float64), neighbors=40, smoothing=3.0)


def single_view_3d(points_yx, interpolator):
    pts = np.asarray(points_yx, dtype=np.float64)
    return interpolator(pts)


def cluster_endpoints(branch_records, radius_px=9.0):
    endpoints = []
    for record in branch_records:
        pts = np.asarray(record["im0_points_yx"], dtype=np.float64)
        endpoints.append((record, "start", pts[0]))
        endpoints.append((record, "end", pts[-1]))

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
            center = np.mean([member[2] for member in members], axis=0)
            for j, other in enumerate(endpoints):
                if j in used:
                    continue
                if np.linalg.norm(other[2] - center) <= radius_px:
                    used.add(j)
                    members.append(other)
                    changed = True
        clusters.append(members)
    return clusters


def snap_shared_nodes(branch_records):
    clusters = cluster_endpoints(branch_records)
    node_rows = []
    for node_id, members in enumerate(clusters):
        if len(members) < 2:
            continue
        candidates = []
        for record, side, _ in members:
            point = record["points_3d"][0] if side == "start" else record["points_3d"][-1]
            weight = 3.0 if record["status"] == "reliable" else 2.0 if record["status"] == "usable" else 0.8
            candidates.append((point, weight))
        node = sum(point * weight for point, weight in candidates) / sum(weight for _, weight in candidates)
        node_rows.append({"node_id": node_id, "members": len(members), "xyz": node.tolist()})
        for record, side, _ in members:
            points = record["points_3d"].copy()
            if side == "start":
                limit = min(8, len(points))
                for i in range(limit):
                    w = 1.0 - i / max(limit - 1, 1)
                    points[i] = w * node + (1.0 - w) * points[i]
                points[0] = node
            else:
                limit = min(8, len(points))
                for offset in range(limit):
                    i = len(points) - 1 - offset
                    w = 1.0 - offset / max(limit - 1, 1)
                    points[i] = w * node + (1.0 - w) * points[i]
                points[-1] = node
            record["points_3d"] = points
    return node_rows


def smooth_points(points, window=7):
    if len(points) < window:
        return points
    pad = window // 2
    padded = np.pad(points, ((pad, pad), (0, 0)), mode="edge")
    kernel = np.ones(window, dtype=np.float64) / window
    return np.column_stack([np.convolve(padded[:, dim], kernel, mode="valid") for dim in range(3)])


def make_frame(tangent):
    tangent = tangent / max(np.linalg.norm(tangent), 1e-9)
    ref = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(np.dot(tangent, ref))) > 0.9:
        ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    normal = np.cross(tangent, ref)
    normal /= max(np.linalg.norm(normal), 1e-9)
    binormal = np.cross(tangent, normal)
    binormal /= max(np.linalg.norm(binormal), 1e-9)
    return normal, binormal


def tube_for_polyline(points, radius=0.5, segments=12):
    points = smooth_points(points)
    vertices = []
    faces = []
    if len(points) < 2:
        return vertices, faces
    for idx, point in enumerate(points):
        if idx == 0:
            tangent = points[1] - points[0]
        elif idx == len(points) - 1:
            tangent = points[-1] - points[-2]
        else:
            tangent = points[idx + 1] - points[idx - 1]
        normal, binormal = make_frame(tangent)
        for seg in range(segments):
            angle = 2.0 * np.pi * seg / segments
            vertices.append(point + radius * (np.cos(angle) * normal + np.sin(angle) * binormal))
    for idx in range(len(points) - 1):
        base_a = idx * segments
        base_b = (idx + 1) * segments
        for seg in range(segments):
            a0 = base_a + seg
            a1 = base_a + ((seg + 1) % segments)
            b0 = base_b + seg
            b1 = base_b + ((seg + 1) % segments)
            faces.append((a0, b0, b1))
            faces.append((a0, b1, a1))
    return vertices, faces


def write_obj(path, parts):
    mtl_name = path.with_suffix(".mtl").name
    with open(path.with_suffix(".mtl"), "w", encoding="utf-8") as f:
        f.write("newmtl reliable\nKd 0.82 0.82 0.82\n")
        f.write("newmtl usable\nKd 0.55 0.72 1.0\n")
        f.write("newmtl single_view_preserved\nKd 1.0 0.65 0.18\n")
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"mtllib {mtl_name}\n")
        f.write("# Filtered epipolar reconstruction. Reliable/usable branches are stereo-supported; orange branches are preserved single-view branches.\n")
        offset = 1
        for name, status, vertices, faces in parts:
            f.write(f"o {name}\nusemtl {status}\n")
            for vertex in vertices:
                f.write(f"v {vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f}\n")
            for a, b, c in faces:
                f.write(f"f {a + offset} {b + offset} {c + offset}\n")
            offset += len(vertices)


def project_points(records, view):
    rows = []
    for record in records:
        for point in record["points_3d"]:
            yx = project_point(point, view)
            rows.append((record["status"], yx))
    return rows


def draw_reprojection(path, image_name, projected):
    image = cv2.imread(str(SELECTED_DIR / f"{image_name}_original.png"), cv2.IMREAD_GRAYSCALE)
    canvas = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    colors = {
        "reliable": (0, 255, 0),
        "usable": (255, 180, 0),
        "single_view_preserved": (0, 165, 255),
    }
    for status, yx in projected:
        y, x = yx
        if 0 <= x < 512 and 0 <= y < 512:
            cv2.circle(canvas, (int(round(x)), int(round(y))), 1, colors[status], -1)
    cv2.imwrite(str(path), canvas)


def main():
    OUT_DIR.mkdir(exist_ok=True)
    branches_a = load_graph(VIEW_A["name"])
    branches_b = load_graph(VIEW_B["name"])
    matches = find_best_matches(branches_a, branches_b)
    interpolator = load_depth_initializer(matches)

    branch_records = []
    for match in matches:
        branch_a = match["branch_a"]
        status = match["status"]
        if status in {"reliable", "usable"}:
            im0_points = match["score"]["samples_a"]
            points_3d = match["score"]["points_3d"]
            im2_branch = int(match["branch_b"]["branch_id"])
        else:
            # Preserve small/missing branches using IM0 topology and depth inferred
            # from nearby reliable stereo-supported vessels.
            im0_points = resample_polyline(branch_a["centerline_yx"], min(48, max(20, int(branch_a["points"]))))
            points_3d = single_view_3d(im0_points, interpolator)
            im2_branch = None

        branch_records.append(
            {
                "branch_id": int(branch_a["branch_id"]),
                "status": status,
                "im2_branch_id": im2_branch,
                "im0_points_yx": np.asarray(im0_points, dtype=np.float64),
                "points_3d": np.asarray(points_3d, dtype=np.float64),
                "mean_residual_mm": match["score"]["mean"],
                "median_residual_mm": match["score"]["median"],
                "p90_residual_mm": match["score"]["p90"],
                "max_residual_mm": match["score"]["max"],
                "mean_diameter_px": float(branch_a["mean_diameter_px"]),
                "points": int(branch_a["points"]),
            }
        )

    node_rows = snap_shared_nodes(branch_records)

    parts = []
    for record in branch_records:
        radius = max(0.28, min(0.85, record["mean_diameter_px"] * VIEW_A["imager_pixel_spacing_mm"][0] * 0.24))
        if record["status"] == "single_view_preserved":
            radius *= 0.72
        vertices, faces = tube_for_polyline(record["points_3d"], radius=radius, segments=12)
        parts.append((f"branch_{record['branch_id']:02d}_{record['status']}", record["status"], vertices, faces))
    write_obj(OUT_DIR / "filtered_epipolar_full_tree.obj", parts)

    draw_reprojection(OUT_DIR / "filtered_reprojection_IM0.png", VIEW_A["name"], project_points(branch_records, VIEW_A))
    draw_reprojection(OUT_DIR / "filtered_reprojection_IM2.png", VIEW_B["name"], project_points(branch_records, VIEW_B))

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")
    color_map = {"reliable": "green", "usable": "dodgerblue", "single_view_preserved": "orange"}
    for record in branch_records:
        pts = record["points_3d"]
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=color_map[record["status"]], linewidth=1.5)
    ax.set_title("Filtered Epipolar Full Artery Tree")
    ax.set_xlabel("X mm")
    ax.set_ylabel("Y mm")
    ax.set_zlabel("Z mm")
    ax.view_init(elev=24, azim=-58)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "filtered_epipolar_preview.png", dpi=180)
    plt.close(fig)

    report_rows = []
    for record in branch_records:
        report_rows.append(
            {
                "branch_id": record["branch_id"],
                "status": record["status"],
                "matched_im2_branch_id": record["im2_branch_id"],
                "points_original": record["points"],
                "mean_residual_mm": round(record["mean_residual_mm"], 3),
                "median_residual_mm": round(record["median_residual_mm"], 3),
                "p90_residual_mm": round(record["p90_residual_mm"], 3),
                "max_residual_mm": round(record["max_residual_mm"], 3),
            }
        )
    with open(OUT_DIR / "branch_quality_report.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(report_rows[0].keys()))
        writer.writeheader()
        writer.writerows(report_rows)

    counts = {status: sum(1 for row in report_rows if row["status"] == status) for status in ["reliable", "usable", "single_view_preserved"]}
    summary = {
        "status": "filtered epipolar full-tree reconstruction",
        "meaning": "Reliable/usable branches are supported by two-view epipolar triangulation. Single-view branches are preserved from IM0 topology using interpolated depth, so the full artery tree is not discarded.",
        "counts": counts,
        "num_shared_nodes": len(node_rows),
        "outputs": {
            "obj": str(OUT_DIR / "filtered_epipolar_full_tree.obj"),
            "preview_png": str(OUT_DIR / "filtered_epipolar_preview.png"),
            "reprojection_IM0": str(OUT_DIR / "filtered_reprojection_IM0.png"),
            "reprojection_IM2": str(OUT_DIR / "filtered_reprojection_IM2.png"),
            "quality_report": str(OUT_DIR / "branch_quality_report.csv"),
        },
    }
    with open(OUT_DIR / "filtered_epipolar_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
