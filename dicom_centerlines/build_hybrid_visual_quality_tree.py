from pathlib import Path
import csv
import json

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RBFInterpolator


ROOT = Path(__file__).resolve().parents[1]
CENTERLINE_DIR = ROOT / "dicom_centerlines"
FILTERED_DIR = ROOT / "dicom_3d_filtered"
OUT_DIR = ROOT / "dicom_3d_filtered"

SPACING_MM = 0.2375
IMAGE_CENTER = np.array([255.5, 255.5], dtype=np.float64)


def load_graph():
    with open(CENTERLINE_DIR / "IM0_frame_030_centerline_graph.json", "r", encoding="utf-8") as f:
        graph = json.load(f)
    return {int(branch["branch_id"]): branch for branch in graph["branches"]}


def load_quality():
    quality = {}
    with open(FILTERED_DIR / "branch_quality_report.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            quality[int(row["branch_id"])] = row
    return quality


def load_filtered_points_for_depth():
    rows = []
    # Reconstruct from the OBJ is annoying, so use filtered report + centerline graph
    # and the reliable stereo samples from branch-quality scoring indirectly by
    # reading refined points as stable depth anchors.
    refined_csv = ROOT / "dicom_3d_rough" / "refined_3d_points.csv"
    with open(refined_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(
                (
                    [float(row["view_a_y"]), float(row["view_a_x"])],
                    [float(row["z_mm"])],
                )
            )
    yx = np.asarray([r[0] for r in rows], dtype=np.float64)
    z = np.asarray([r[1] for r in rows], dtype=np.float64)
    return RBFInterpolator(yx, z, neighbors=45, smoothing=6.0), z


def yx_to_xy(coords_yx):
    coords = np.asarray(coords_yx, dtype=np.float64)
    x = (coords[:, 1] - IMAGE_CENTER[1]) * SPACING_MM
    y = -(coords[:, 0] - IMAGE_CENTER[0]) * SPACING_MM
    return np.column_stack([x, y])


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
    with open(path.with_suffix(".mtl"), "w", encoding="utf-8") as f:
        f.write("newmtl reliable\nKd 0.78 0.78 0.78\n")
        f.write("newmtl usable\nKd 0.45 0.65 1.0\n")
        f.write("newmtl single_view_preserved\nKd 1.0 0.58 0.12\n")
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"mtllib {path.with_suffix('.mtl').name}\n")
        f.write("# Hybrid visual-quality tree: IM0 shape preserved, stereo-informed/clamped depth, branch quality materials.\n")
        offset = 1
        for name, status, vertices, faces in parts:
            f.write(f"o {name}\nusemtl {status}\n")
            for vertex in vertices:
                f.write(f"v {vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f}\n")
            for a, b, c in faces:
                f.write(f"f {a + offset} {b + offset} {c + offset}\n")
            offset += len(vertices)


def main():
    graph = load_graph()
    quality = load_quality()
    z_model, anchor_z = load_filtered_points_for_depth()
    z_low = float(np.percentile(anchor_z, 5) - 8.0)
    z_high = float(np.percentile(anchor_z, 95) + 8.0)

    parts = []
    plot_records = []
    for branch_id, branch in graph.items():
        if int(branch["points"]) < 23:
            continue
        row = quality.get(branch_id)
        if not row:
            continue
        status = row["status"]
        coords_yx = np.asarray(branch["centerline_yx"], dtype=np.float64)
        xy = yx_to_xy(coords_yx)
        z = z_model(coords_yx)[:, 0]
        if status == "single_view_preserved":
            # Avoid visual nonsense when the second view does not constrain this
            # branch. It stays attached and visible, but depth is bounded.
            z = np.clip(z, z_low, z_high)
            radius_scale = 0.65
        else:
            z = np.clip(z, z_low - 6.0, z_high + 6.0)
            radius_scale = 1.0
        points = np.column_stack([xy, z])
        radius = max(0.25, min(0.82, float(branch["mean_diameter_px"]) * SPACING_MM * 0.24 * radius_scale))
        vertices, faces = tube_for_polyline(points, radius=radius, segments=12)
        parts.append((f"branch_{branch_id:02d}_{status}", status, vertices, faces))
        plot_records.append((branch_id, status, points))

    out_obj = OUT_DIR / "hybrid_visual_quality_tree.obj"
    write_obj(out_obj, parts)

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")
    colors = {"reliable": "green", "usable": "dodgerblue", "single_view_preserved": "orange"}
    for _, status, points in plot_records:
        ax.plot(points[:, 0], points[:, 1], points[:, 2], color=colors[status], linewidth=1.4)
    ax.set_title("Hybrid Visual + Quality Full Tree")
    ax.set_xlabel("X mm")
    ax.set_ylabel("Y mm")
    ax.set_zlabel("Z mm")
    ax.view_init(elev=78, azim=-90)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "hybrid_visual_quality_preview.png", dpi=180)
    plt.close(fig)

    summary = {
        "status": "hybrid visual-quality tree",
        "meaning": "Preserves full IM0 artery-tree shape for doctor-facing visual continuity. Reliable/usable branches use stereo-informed depth; single-view branches are preserved and depth-clamped, not discarded.",
        "obj": str(out_obj),
        "preview_png": str(OUT_DIR / "hybrid_visual_quality_preview.png"),
    }
    with open(OUT_DIR / "hybrid_visual_quality_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
