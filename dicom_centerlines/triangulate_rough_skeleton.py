from pathlib import Path
import csv
import json
import math
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
WORK_DIR = ROOT / "dicom_centerlines"
OUT_DIR = ROOT / "dicom_3d_rough"


def rotation_from_angles(primary_deg, secondary_deg):
    """Approximate C-arm rotation from DICOM primary/secondary angles."""
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


def ray_from_pixel(point_yx, view):
    """Return world-space source point and unit ray direction for a 2D pixel."""
    y, x = point_yx
    rows, cols = view["image_size_px"]
    spacing_y, spacing_x = view["imager_pixel_spacing_mm"]
    sdd = float(view["source_detector_distance_mm"])
    spd = float(view["source_patient_distance_mm"])
    primary = float(view["primary_angle_deg"])
    secondary = float(view["secondary_angle_deg"])

    # Detector coordinates in the camera frame. The isocenter is at z=0, source
    # is at -SPD, and detector is at SDD-SPD.
    det_x = (float(x) - (cols - 1) / 2.0) * spacing_x
    det_y = (float(y) - (rows - 1) / 2.0) * spacing_y
    detector_point_cam = np.array([det_x, det_y, sdd - spd], dtype=np.float64)
    source_cam = np.array([0.0, 0.0, -spd], dtype=np.float64)

    rot = rotation_from_angles(primary, secondary)
    source_world = rot @ source_cam
    detector_world = rot @ detector_point_cam
    direction = detector_world - source_world
    direction /= np.linalg.norm(direction)
    return source_world, direction


def closest_point_between_rays(source_a, dir_a, source_b, dir_b):
    """Least-squares closest point between two 3D rays/lines."""
    a = np.column_stack([dir_a, -dir_b])
    rhs = source_b - source_a
    params, *_ = np.linalg.lstsq(a, rhs, rcond=None)
    point_a = source_a + params[0] * dir_a
    point_b = source_b + params[1] * dir_b
    midpoint = 0.5 * (point_a + point_b)
    residual = float(np.linalg.norm(point_a - point_b))
    return midpoint, residual


def load_graph(view_name):
    with open(WORK_DIR / f"{view_name}_centerline_graph.json", "r", encoding="utf-8") as f:
        graph = json.load(f)
    return {int(branch["branch_id"]): branch for branch in graph["branches"]}


def resample_polyline(points_yx, n):
    pts = np.asarray(points_yx, dtype=np.float64)
    if len(pts) == 0:
        return pts
    if len(pts) == 1:
        return np.repeat(pts, n, axis=0)

    distances = np.zeros(len(pts), dtype=np.float64)
    distances[1:] = np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))
    total = distances[-1]
    if total <= 1e-6:
        return np.repeat(pts[:1], n, axis=0)

    targets = np.linspace(0.0, total, n)
    ys = np.interp(targets, distances, pts[:, 0])
    xs = np.interp(targets, distances, pts[:, 1])
    return np.column_stack([ys, xs])


def project_point(point_world, view):
    """Project a world point back into one view for diagnostics."""
    rows, cols = view["image_size_px"]
    spacing_y, spacing_x = view["imager_pixel_spacing_mm"]
    sdd = float(view["source_detector_distance_mm"])
    spd = float(view["source_patient_distance_mm"])
    rot = rotation_from_angles(float(view["primary_angle_deg"]), float(view["secondary_angle_deg"]))
    cam = rot.T @ point_world
    source_z = -spd
    detector_z = sdd - spd
    t = (detector_z - source_z) / (cam[2] - source_z)
    det = np.array([0.0, 0.0, source_z]) + t * (cam - np.array([0.0, 0.0, source_z]))
    x = det[0] / spacing_x + (cols - 1) / 2.0
    y = det[1] / spacing_y + (rows - 1) / 2.0
    return np.array([y, x], dtype=np.float64)


def write_ply(path, points, labels):
    palette = {
        "proximal_main_trunk": (230, 70, 70),
        "inferior_long_branch": (70, 140, 240),
        "right_distal_branch": (70, 200, 120),
        "upper_branch": (230, 160, 50),
    }
    with open(path, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for point, label in zip(points, labels):
            r, g, b = palette.get(label, (255, 255, 255))
            f.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} {r} {g} {b}\n")


def make_3d_plot(points_by_match):
    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")
    for match_id, pts in points_by_match.items():
        pts = np.asarray(pts)
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], marker="o", markersize=2, linewidth=1.5, label=match_id)
    ax.set_title("Rough Triangulated 3D Coronary Skeleton")
    ax.set_xlabel("X mm")
    ax.set_ylabel("Y mm")
    ax.set_zlabel("Z mm")
    ax.legend(loc="best", fontsize=8)
    ax.view_init(elev=24, azim=-58)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "rough_3d_skeleton_plot.png", dpi=180)
    plt.close(fig)


def make_reprojection_plot(rows, view_a, view_b):
    for view, suffix in [(view_a, "view_a"), (view_b, "view_b")]:
        image = cv2.imread(str(ROOT / "dicom_selected_segmentation" / f"{view['name']}_original.png"), cv2.IMREAD_GRAYSCALE)
        canvas = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        for row in rows:
            point = np.array([row["x_mm"], row["y_mm"], row["z_mm"]], dtype=np.float64)
            yx = project_point(point, view)
            cv2.circle(canvas, (int(round(yx[1])), int(round(yx[0]))), 2, (0, 0, 255), -1)
        cv2.imwrite(str(OUT_DIR / f"rough_3d_reprojection_{suffix}.png"), canvas)


def main():
    OUT_DIR.mkdir(exist_ok=True)
    match_path = Path(sys.argv[1]) if len(sys.argv) > 1 else WORK_DIR / "rough_branch_matches.json"
    with open(match_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    output_prefix = config.get("output_prefix", "rough_3d")

    view_a = config["view_a"]
    view_b = config["view_b"]
    graph_a = load_graph(view_a["name"])
    graph_b = load_graph(view_b["name"])

    rows = []
    all_points = []
    all_labels = []
    points_by_match = {}
    residuals = []

    for match in config["matches"]:
        branch_a = list(graph_a[int(match["view_a_branch_id"])]["centerline_yx"])
        branch_b = list(graph_b[int(match["view_b_branch_id"])]["centerline_yx"])
        if match.get("reverse_view_a"):
            branch_a = list(reversed(branch_a))
        if match.get("reverse_view_b"):
            branch_b = list(reversed(branch_b))

        samples = int(match.get("samples", 80))
        samples_a = resample_polyline(branch_a, samples)
        samples_b = resample_polyline(branch_b, samples)

        match_points = []
        for idx, (point_a, point_b) in enumerate(zip(samples_a, samples_b)):
            source_a, dir_a = ray_from_pixel(point_a, view_a)
            source_b, dir_b = ray_from_pixel(point_b, view_b)
            point_3d, residual = closest_point_between_rays(source_a, dir_a, source_b, dir_b)
            residuals.append(residual)
            match_points.append(point_3d)
            all_points.append(point_3d)
            all_labels.append(match["match_id"])
            rows.append(
                {
                    "match_id": match["match_id"],
                    "sample_index": idx,
                    "view_a_y": float(point_a[0]),
                    "view_a_x": float(point_a[1]),
                    "view_b_y": float(point_b[0]),
                    "view_b_x": float(point_b[1]),
                    "x_mm": float(point_3d[0]),
                    "y_mm": float(point_3d[1]),
                    "z_mm": float(point_3d[2]),
                    "ray_residual_mm": residual,
                }
            )
        points_by_match[match["match_id"]] = match_points

    with open(OUT_DIR / f"{output_prefix}_points.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    write_ply(OUT_DIR / f"{output_prefix}_skeleton_points.ply", all_points, all_labels)
    make_3d_plot(points_by_match)
    (OUT_DIR / "rough_3d_skeleton_plot.png").replace(OUT_DIR / f"{output_prefix}_skeleton_plot.png")
    make_reprojection_plot(rows, view_a, view_b)
    (OUT_DIR / "rough_3d_reprojection_view_a.png").replace(OUT_DIR / f"{output_prefix}_reprojection_view_a.png")
    (OUT_DIR / "rough_3d_reprojection_view_b.png").replace(OUT_DIR / f"{output_prefix}_reprojection_view_b.png")

    summary = {
        "num_points": len(all_points),
        "num_matches": len(config["matches"]),
        "mean_ray_residual_mm": float(np.mean(residuals)),
        "median_ray_residual_mm": float(np.median(residuals)),
        "max_ray_residual_mm": float(np.max(residuals)),
        "important_warning": "This is a rough first-pass reconstruction using approximate C-arm geometry and provisional branch correspondences. It is for inspection and pipeline validation, not clinical measurement.",
        "match_file": str(match_path),
        "outputs": {
            "points_csv": str(OUT_DIR / f"{output_prefix}_points.csv"),
            "points_ply": str(OUT_DIR / f"{output_prefix}_skeleton_points.ply"),
            "plot_png": str(OUT_DIR / f"{output_prefix}_skeleton_plot.png"),
            "reprojection_view_a": str(OUT_DIR / f"{output_prefix}_reprojection_view_a.png"),
            "reprojection_view_b": str(OUT_DIR / f"{output_prefix}_reprojection_view_b.png"),
        },
    }
    with open(OUT_DIR / f"{output_prefix}_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Output folder: {OUT_DIR}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
