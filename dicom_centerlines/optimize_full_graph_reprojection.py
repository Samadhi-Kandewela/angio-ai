from pathlib import Path
import csv
import json

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RBFInterpolator
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial import cKDTree

from triangulate_rough_skeleton import project_point, ray_from_pixel


ROOT = Path(__file__).resolve().parents[1]
CENTERLINE_DIR = ROOT / "dicom_centerlines"
SELECTED_DIR = ROOT / "dicom_selected_segmentation"
ROUGH_DIR = ROOT / "dicom_3d_rough"
OUT_DIR = ROOT / "dicom_3d_optimized"


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


def load_refined_world_initializer():
    image_points = []
    world_points = []
    with open(ROUGH_DIR / "refined_3d_points.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            image_points.append([float(row["view_a_y"]), float(row["view_a_x"])])
            world_points.append([float(row["x_mm"]), float(row["y_mm"]), float(row["z_mm"])])
    return RBFInterpolator(
        np.asarray(image_points, dtype=np.float64),
        np.asarray(world_points, dtype=np.float64),
        neighbors=50,
        smoothing=4.0,
    )


def load_full_graph():
    with open(CENTERLINE_DIR / "IM0_frame_030_centerline_graph.json", "r", encoding="utf-8") as f:
        graph = json.load(f)
    branches = []
    for branch in graph["branches"]:
        if branch["points"] >= 30:
            branches.append(
                {
                    "branch_id": int(branch["branch_id"]),
                    "points_yx": [tuple(map(int, point)) for point in branch["centerline_yx"]],
                    "mean_diameter_px": float(branch["mean_diameter_px"]),
                }
            )
    return branches


def resample_yx(points_yx, max_points=48):
    pts = np.asarray(points_yx, dtype=np.float64)
    if len(pts) <= max_points:
        return [tuple(map(int, point)) for point in pts]
    distances = np.zeros(len(pts), dtype=np.float64)
    distances[1:] = np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))
    total = distances[-1]
    if total <= 1e-6:
        return [tuple(map(int, pts[0]))]
    targets = np.linspace(0.0, total, max_points)
    ys = np.interp(targets, distances, pts[:, 0])
    xs = np.interp(targets, distances, pts[:, 1])
    return [tuple(map(int, point)) for point in np.column_stack([np.rint(ys), np.rint(xs)])]


def unique_points(branches):
    point_to_idx = {}
    points = []
    branch_indices = []
    for branch in branches:
        indices = []
        for point in branch["points_yx"]:
            if point not in point_to_idx:
                point_to_idx[point] = len(points)
                points.append(point)
            indices.append(point_to_idx[point])
        branch_indices.append((branch, indices))
    return points, branch_indices


def build_jacobian_sparsity(num_points, branch_indices):
    # Residual layout must match residual(): data 2N, prior N, then branch smooth/first terms.
    rows = 2 * num_points + num_points
    for _, indices in branch_indices:
        if len(indices) >= 3:
            rows += 3 * (len(indices) - 2)
        if len(indices) >= 2:
            rows += len(indices) - 1

    sparsity = lil_matrix((rows, num_points), dtype=np.int8)
    row = 0
    for idx in range(num_points):
        sparsity[row, idx] = 1
        sparsity[row + 1, idx] = 1
        row += 2
    for idx in range(num_points):
        sparsity[row, idx] = 1
        row += 1
    for _, indices in branch_indices:
        if len(indices) >= 3:
            for pos in range(len(indices) - 2):
                cols = [indices[pos], indices[pos + 1], indices[pos + 2]]
                for _ in range(3):
                    for col in cols:
                        sparsity[row, col] = 1
                    row += 1
        if len(indices) >= 2:
            for pos in range(len(indices) - 1):
                sparsity[row, indices[pos]] = 1
                sparsity[row, indices[pos + 1]] = 1
                row += 1
    return sparsity.tocsr()


def load_im2_target_tree():
    skeleton = cv2.imread(str(CENTERLINE_DIR / "IM2_frame_022_skeleton.png"), cv2.IMREAD_GRAYSCALE)
    if skeleton is None:
        raise FileNotFoundError("IM2 skeleton missing")
    target_yx = np.column_stack(np.where(skeleton > 0)).astype(np.float64)
    return cKDTree(target_yx), target_yx


def build_rays(points_yx, init_model):
    sources = []
    dirs = []
    t0 = []
    init_world = init_model(np.asarray(points_yx, dtype=np.float64))
    for point_yx, world_guess in zip(points_yx, init_world):
        source, direction = ray_from_pixel(point_yx, VIEW_A)
        t = float(np.dot(world_guess - source, direction))
        if not np.isfinite(t):
            t = float(VIEW_A["source_patient_distance_mm"])
        t = float(np.clip(t, 580.0, 880.0))
        sources.append(source)
        dirs.append(direction)
        t0.append(t)
    return np.asarray(sources), np.asarray(dirs), np.asarray(t0)


def project_all(points_world, view):
    return np.vstack([project_point(point, view) for point in points_world])


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
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Full graph reprojection-optimized artery tree prototype.\n")
        f.write("# IM0 projection is ray-constrained; IM2 projection is optimized to the IM2 centerline.\n")
        f.write("# Prototype only, not clinical measurement.\n")
        offset = 1
        for name, vertices, faces in parts:
            f.write(f"o {name}\n")
            for vertex in vertices:
                f.write(f"v {vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f}\n")
            for a, b, c in faces:
                f.write(f"f {a + offset} {b + offset} {c + offset}\n")
            offset += len(vertices)


def draw_reprojection(path, image_name, projected_yx, color=(0, 0, 255)):
    image = cv2.imread(str(SELECTED_DIR / f"{image_name}_original.png"), cv2.IMREAD_GRAYSCALE)
    canvas = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for y, x in projected_yx:
        if 0 <= x < 512 and 0 <= y < 512:
            cv2.circle(canvas, (int(round(x)), int(round(y))), 1, color, -1)
    cv2.imwrite(str(path), canvas)


def main():
    OUT_DIR.mkdir(exist_ok=True)
    branches = load_full_graph()
    for branch in branches:
        branch["points_yx"] = resample_yx(branch["points_yx"], max_points=48)
    points_yx, branch_indices = unique_points(branches)
    target_tree, _ = load_im2_target_tree()
    init_model = load_refined_world_initializer()
    sources, directions, t0 = build_rays(points_yx, init_model)
    t_ref = t0.copy()

    point_count = len(points_yx)
    print(f"Optimizing {point_count} shared centerline points across {len(branches)} branches...")

    data_weight = 0.18
    smooth_weight = 0.06
    first_weight = 0.015
    depth_prior_weight = 0.008
    jac_sparsity = build_jacobian_sparsity(point_count, branch_indices)

    def world_from_t(t):
        return sources + directions * t[:, None]

    def residual(t):
        world = world_from_t(t)
        proj_b = project_all(world, VIEW_B)
        _, nearest_idx = target_tree.query(proj_b)
        nearest = target_tree.data[nearest_idx]
        res = [((proj_b - nearest) * data_weight).ravel()]

        # Keep the solution close to the stereo-initialized depth unless IM2
        # evidence strongly improves it.
        res.append((t - t_ref) * depth_prior_weight)

        for _, indices in branch_indices:
            idx = np.asarray(indices, dtype=np.int32)
            if len(idx) >= 3:
                branch_world = world[idx]
                second = branch_world[:-2] - 2.0 * branch_world[1:-1] + branch_world[2:]
                res.append((second * smooth_weight).ravel())
            if len(idx) >= 2:
                first = np.diff(t[idx])
                res.append(first * first_weight)
        return np.concatenate(res)

    before_world = world_from_t(t0)
    before_proj_b = project_all(before_world, VIEW_B)
    before_dist, _ = target_tree.query(before_proj_b)

    result = least_squares(
        residual,
        t0,
        bounds=(t0 - 90.0, t0 + 90.0),
        loss="soft_l1",
        f_scale=1.0,
        max_nfev=70,
        jac_sparsity=jac_sparsity,
        verbose=1,
    )

    optimized_world = world_from_t(result.x)
    proj_a = project_all(optimized_world, VIEW_A)
    proj_b = project_all(optimized_world, VIEW_B)
    after_dist, _ = target_tree.query(proj_b)

    draw_reprojection(OUT_DIR / "optimized_reprojection_IM0.png", VIEW_A["name"], proj_a)
    draw_reprojection(OUT_DIR / "optimized_reprojection_IM2.png", VIEW_B["name"], proj_b)

    parts = []
    branch_rows = []
    for branch, indices in branch_indices:
        idx = np.asarray(indices, dtype=np.int32)
        branch_world = optimized_world[idx]
        radius = max(0.35, min(0.9, branch["mean_diameter_px"] * VIEW_A["imager_pixel_spacing_mm"][0] * 0.25))
        vertices, faces = tube_for_polyline(branch_world, radius=radius, segments=12)
        parts.append((f"branch_{branch['branch_id']:02d}", vertices, faces))
        branch_rows.append(
            {
                "branch_id": branch["branch_id"],
                "points": len(indices),
                "visual_radius_mm": radius,
            }
        )

    write_obj(OUT_DIR / "optimized_full_tree.obj", parts)

    with open(OUT_DIR / "optimized_full_tree_points.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["point_id", "im0_y", "im0_x", "x_mm", "y_mm", "z_mm", "im2_y", "im2_x", "im2_nearest_dist_px"],
        )
        writer.writeheader()
        for idx, (yx, xyz, p2, dist) in enumerate(zip(points_yx, optimized_world, proj_b, after_dist)):
            writer.writerow(
                {
                    "point_id": idx,
                    "im0_y": yx[0],
                    "im0_x": yx[1],
                    "x_mm": xyz[0],
                    "y_mm": xyz[1],
                    "z_mm": xyz[2],
                    "im2_y": p2[0],
                    "im2_x": p2[1],
                    "im2_nearest_dist_px": dist,
                }
            )

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")
    for branch, indices in branch_indices:
        pts = optimized_world[np.asarray(indices, dtype=np.int32)]
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], linewidth=1.2)
    ax.set_title("Optimized Full Graph 3D Tree")
    ax.set_xlabel("X mm")
    ax.set_ylabel("Y mm")
    ax.set_zlabel("Z mm")
    ax.view_init(elev=24, azim=-58)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "optimized_full_tree_preview.png", dpi=180)
    plt.close(fig)

    summary = {
        "status": "full connected graph reprojection optimization prototype",
        "warning": "This is a research prototype. It optimizes full IM0 topology against IM2 centerline reprojection, but still needs exact DICOM calibration, cardiac phase validation, and clinical validation.",
        "num_unique_points": point_count,
        "num_branches": len(branches),
        "initial_im2_mean_distance_px": float(np.mean(before_dist)),
        "initial_im2_median_distance_px": float(np.median(before_dist)),
        "optimized_im2_mean_distance_px": float(np.mean(after_dist)),
        "optimized_im2_median_distance_px": float(np.median(after_dist)),
        "optimized_im2_95pct_distance_px": float(np.percentile(after_dist, 95)),
        "optimizer_cost": float(result.cost),
        "optimizer_status": int(result.status),
        "optimizer_message": result.message,
        "branch_rows": branch_rows,
        "outputs": {
            "obj": str(OUT_DIR / "optimized_full_tree.obj"),
            "preview_png": str(OUT_DIR / "optimized_full_tree_preview.png"),
            "reprojection_IM0": str(OUT_DIR / "optimized_reprojection_IM0.png"),
            "reprojection_IM2": str(OUT_DIR / "optimized_reprojection_IM2.png"),
            "points_csv": str(OUT_DIR / "optimized_full_tree_points.csv"),
        },
    }
    with open(OUT_DIR / "optimized_full_tree_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
