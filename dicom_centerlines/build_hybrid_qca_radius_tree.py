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

SPACING_DETECTOR_MM = 0.2375
MAGNIFICATION_IM0 = 1.42639
OBJECT_PIXEL_MM = SPACING_DETECTOR_MM / MAGNIFICATION_IM0
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


def load_depth_model():
    yx = []
    z = []
    with open(ROOT / "dicom_3d_rough" / "refined_3d_points.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            yx.append([float(row["view_a_y"]), float(row["view_a_x"])])
            z.append([float(row["z_mm"])])
    yx = np.asarray(yx, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    return RBFInterpolator(yx, z, neighbors=45, smoothing=6.0), z


def yx_to_xy(coords_yx):
    coords = np.asarray(coords_yx, dtype=np.float64)
    x = (coords[:, 1] - IMAGE_CENTER[1]) * SPACING_DETECTOR_MM
    y = -(coords[:, 0] - IMAGE_CENTER[0]) * SPACING_DETECTOR_MM
    return np.column_stack([x, y])


def smooth_1d(values, window=9):
    if len(values) < window:
        return values
    if window % 2 == 0:
        window += 1
    pad = window // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(padded, kernel, mode="valid")


def smooth_points(points, window=7):
    if len(points) < window:
        return points
    pad = window // 2
    padded = np.pad(points, ((pad, pad), (0, 0)), mode="edge")
    kernel = np.ones(window, dtype=np.float64) / window
    return np.column_stack([np.convolve(padded[:, dim], kernel, mode="valid") for dim in range(3)])


def estimate_radii_mm(coords_yx, distance_transform):
    coords = np.asarray(coords_yx, dtype=np.float64)
    y = np.clip(np.rint(coords[:, 0]).astype(int), 0, distance_transform.shape[0] - 1)
    x = np.clip(np.rint(coords[:, 1]).astype(int), 0, distance_transform.shape[1] - 1)
    radius_px = distance_transform[y, x]
    radius_mm = radius_px * OBJECT_PIXEL_MM
    radius_mm = smooth_1d(radius_mm, window=9)
    return np.clip(radius_mm, 0.18, 2.3)


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


def variable_tube(points, radii, segments=16):
    points = smooth_points(points)
    radii = smooth_1d(np.asarray(radii, dtype=np.float64), window=7)
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
        radius = float(radii[idx])
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


def sphere(center, radius=1.2, rings=8, segments=12):
    vertices = []
    faces = []
    for ring in range(rings + 1):
        phi = np.pi * ring / rings
        for seg in range(segments):
            theta = 2.0 * np.pi * seg / segments
            vertices.append(
                center
                + radius
                * np.array(
                    [np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)],
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


def write_obj(path, parts):
    with open(path.with_suffix(".mtl"), "w", encoding="utf-8") as f:
        f.write("newmtl reliable\nKd 0.78 0.78 0.78\n")
        f.write("newmtl usable\nKd 0.45 0.65 1.0\n")
        f.write("newmtl single_view_preserved\nKd 1.0 0.58 0.12\n")
        f.write("newmtl stenosis_marker\nKd 1.0 0.02 0.02\n")
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"mtllib {path.with_suffix('.mtl').name}\n")
        f.write("# Hybrid full tree with QCA/distance-transform variable radius.\n")
        f.write("# Reliable/usable/single-view branch material encodes stereo confidence; red spheres mark narrowest branch points.\n")
        offset = 1
        for name, material, vertices, faces in parts:
            f.write(f"o {name}\nusemtl {material}\n")
            for vertex in vertices:
                f.write(f"v {vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f}\n")
            for a, b, c in faces:
                f.write(f"f {a + offset} {b + offset} {c + offset}\n")
            offset += len(vertices)


def main():
    graph = load_graph()
    quality = load_quality()
    z_model, anchor_z = load_depth_model()
    clean_mask = cv2.imread(str(CENTERLINE_DIR / "IM0_frame_030_clean_mask.png"), cv2.IMREAD_GRAYSCALE)
    if clean_mask is None:
        raise FileNotFoundError("IM0 clean mask missing")
    dt = cv2.distanceTransform((clean_mask > 0).astype(np.uint8), cv2.DIST_L2, 5)

    z_low = float(np.percentile(anchor_z, 5) - 8.0)
    z_high = float(np.percentile(anchor_z, 95) + 8.0)

    parts = []
    report_rows = []
    plot_records = []
    for branch_id, branch in graph.items():
        if int(branch["points"]) < 23 or branch_id not in quality:
            continue
        status = quality[branch_id]["status"]
        coords_yx = np.asarray(branch["centerline_yx"], dtype=np.float64)
        xy = yx_to_xy(coords_yx)
        z = z_model(coords_yx)[:, 0]
        if status == "single_view_preserved":
            z = np.clip(z, z_low, z_high)
        else:
            z = np.clip(z, z_low - 6.0, z_high + 6.0)
        points = np.column_stack([xy, z])

        radii = estimate_radii_mm(coords_yx, dt)
        if status == "single_view_preserved":
            # Keep uncertain distal/small branches visually present but avoid
            # overclaiming diameter precision.
            radii = np.clip(radii * 0.85, 0.14, 1.8)

        vertices, faces = variable_tube(points, radii, segments=16)
        parts.append((f"branch_{branch_id:02d}_{status}", status, vertices, faces))

        min_idx = int(np.argmin(radii))
        min_radius = float(radii[min_idx])
        ref_radius = float(np.percentile(radii, 80))
        stenosis_pct = max(0.0, (1.0 - min_radius / max(ref_radius, 1e-6)) * 100.0)
        if stenosis_pct >= 30.0 and status != "single_view_preserved":
            marker_radius = max(0.8, min(1.6, ref_radius * 0.7))
            marker_vertices, marker_faces = sphere(points[min_idx], radius=marker_radius, rings=8, segments=12)
            parts.append((f"stenosis_marker_branch_{branch_id:02d}", "stenosis_marker", marker_vertices, marker_faces))

        report_rows.append(
            {
                "branch_id": branch_id,
                "status": status,
                "points": int(branch["points"]),
                "min_radius_mm": round(min_radius, 3),
                "ref_radius_mm_p80": round(ref_radius, 3),
                "estimated_diameter_stenosis_pct": round(stenosis_pct, 1),
                "min_point_y": int(coords_yx[min_idx, 0]),
                "min_point_x": int(coords_yx[min_idx, 1]),
                "marker_added": bool(stenosis_pct >= 30.0 and status != "single_view_preserved"),
            }
        )
        plot_records.append((branch_id, status, points, radii))

    out_obj = OUT_DIR / "hybrid_qca_radius_tree.obj"
    write_obj(out_obj, parts)

    with open(OUT_DIR / "hybrid_qca_radius_report.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(report_rows[0].keys()))
        writer.writeheader()
        writer.writerows(report_rows)

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")
    colors = {"reliable": "green", "usable": "dodgerblue", "single_view_preserved": "orange"}
    for _, status, points, _ in plot_records:
        ax.plot(points[:, 0], points[:, 1], points[:, 2], color=colors[status], linewidth=1.4)
    ax.set_title("Hybrid Full Tree With QCA Radius")
    ax.set_xlabel("X mm")
    ax.set_ylabel("Y mm")
    ax.set_zlabel("Z mm")
    ax.view_init(elev=78, azim=-90)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "hybrid_qca_radius_preview.png", dpi=180)
    plt.close(fig)

    summary = {
        "status": "hybrid full tree with QCA-derived variable radius",
        "meaning": "The tree preserves full IM0 topology, uses stereo-informed/clamped depth, and uses IM0 distance-transform/QCA radius so stenosis-like narrowing appears in the mesh.",
        "object_pixel_mm_used": OBJECT_PIXEL_MM,
        "obj": str(out_obj),
        "preview_png": str(OUT_DIR / "hybrid_qca_radius_preview.png"),
        "radius_report": str(OUT_DIR / "hybrid_qca_radius_report.csv"),
    }
    with open(OUT_DIR / "hybrid_qca_radius_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
