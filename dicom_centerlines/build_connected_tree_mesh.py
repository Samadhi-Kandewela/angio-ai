from pathlib import Path
import csv
import json

import cv2
import numpy as np
from scipy.interpolate import RBFInterpolator


ROOT = Path(__file__).resolve().parents[1]
CENTERLINE_DIR = ROOT / "dicom_centerlines"
ROUGH_DIR = ROOT / "dicom_3d_rough"
OUT_DIR = ROOT / "dicom_3d_connected"


def load_refined_samples():
    image_points = []
    world_points = []
    with open(ROUGH_DIR / "refined_3d_points.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            image_points.append([float(row["view_a_y"]), float(row["view_a_x"])])
            world_points.append([float(row["x_mm"]), float(row["y_mm"]), float(row["z_mm"])])
    return np.asarray(image_points, dtype=np.float64), np.asarray(world_points, dtype=np.float64)


def smooth_points(points, window=9):
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


def tube_for_polyline(points, radius=0.55, segments=10):
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


def write_obj(path, mesh_parts):
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Connected topology-preserving coronary prototype.\n")
        f.write("# Built from the IM0 2D vessel graph with 3D coordinates interpolated from refined stereo samples.\n")
        f.write("# This is a visual prototype, not a clinically validated reconstruction.\n")
        vertex_offset = 1
        for name, vertices, faces in mesh_parts:
            f.write(f"o {name}\n")
            for vertex in vertices:
                f.write(f"v {vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f}\n")
            for a, b, c in faces:
                f.write(f"f {a + vertex_offset} {b + vertex_offset} {c + vertex_offset}\n")
            vertex_offset += len(vertices)


def write_ply(path, branch_points):
    points = np.vstack([pts for _, pts in branch_points])
    with open(path, "w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for _, pts in branch_points:
            for point in pts:
                f.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} 230 230 230\n")


def make_plot(path, branch_points):
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")
    for name, points in branch_points:
        ax.plot(points[:, 0], points[:, 1], points[:, 2], linewidth=1.6, label=name)
    ax.set_title("Connected 3D Artery Tree Prototype")
    ax.set_xlabel("X mm")
    ax.set_ylabel("Y mm")
    ax.set_zlabel("Z mm")
    ax.view_init(elev=24, azim=-58)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    OUT_DIR.mkdir(exist_ok=True)
    image_points, world_points = load_refined_samples()

    # RBF interpolation gives a smooth 3D coordinate estimate for every point on
    # the connected IM0 vessel graph. This intentionally preserves topology.
    interpolator = RBFInterpolator(image_points, world_points, neighbors=40, smoothing=2.0)

    with open(CENTERLINE_DIR / "IM0_frame_030_centerline_graph.json", "r", encoding="utf-8") as f:
        graph = json.load(f)

    branch_points = []
    mesh_parts = []
    kept_branches = []
    for branch in graph["branches"]:
        if branch["points"] < 35:
            continue
        coords_yx = np.asarray(branch["centerline_yx"], dtype=np.float64)
        points_3d = interpolator(coords_yx)
        branch_name = f"branch_{branch['branch_id']:02d}"
        branch_points.append((branch_name, points_3d))
        vertices, faces = tube_for_polyline(points_3d, radius=0.55, segments=10)
        mesh_parts.append((branch_name, vertices, faces))
        kept_branches.append(
            {
                "branch_id": branch["branch_id"],
                "points": branch["points"],
                "length_px": branch["length_px"],
                "mean_diameter_px": branch["mean_diameter_px"],
            }
        )

    obj_path = OUT_DIR / "connected_tree_prototype.obj"
    ply_path = OUT_DIR / "connected_tree_centerline_points.ply"
    plot_path = OUT_DIR / "connected_tree_preview.png"
    write_obj(obj_path, mesh_parts)
    write_ply(ply_path, branch_points)
    make_plot(plot_path, branch_points)

    summary = {
        "method": "Topology-preserving visual prototype from IM0 2D vessel graph, with 3D coordinates interpolated from refined stereo samples.",
        "warning": "This fixes the disconnected-pipe visualization problem, but it is still a prototype. Next step is true topology-aware stereo optimization, not only interpolation.",
        "num_branches": len(branch_points),
        "kept_branches": kept_branches,
        "outputs": {
            "obj": str(obj_path),
            "ply": str(ply_path),
            "preview_png": str(plot_path),
        },
    }
    with open(OUT_DIR / "connected_tree_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
