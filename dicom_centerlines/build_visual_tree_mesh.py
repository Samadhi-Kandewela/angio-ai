from pathlib import Path
import csv
import json

import numpy as np
from scipy.interpolate import RBFInterpolator


ROOT = Path(__file__).resolve().parents[1]
CENTERLINE_DIR = ROOT / "dicom_centerlines"
ROUGH_DIR = ROOT / "dicom_3d_rough"
OUT_DIR = ROOT / "dicom_3d_connected"

SPACING_MM = 0.2375
IMAGE_CENTER = np.array([255.5, 255.5], dtype=np.float64)


def load_depth_samples():
    image_points = []
    depths = []
    with open(ROUGH_DIR / "refined_3d_points.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            image_points.append([float(row["view_a_y"]), float(row["view_a_x"])])
            depths.append([float(row["z_mm"])])
    return np.asarray(image_points, dtype=np.float64), np.asarray(depths, dtype=np.float64)


def yx_to_visual_3d(coords_yx, depth_model):
    coords_yx = np.asarray(coords_yx, dtype=np.float64)
    z = depth_model(coords_yx)[:, 0]
    x = (coords_yx[:, 1] - IMAGE_CENTER[1]) * SPACING_MM
    y = -(coords_yx[:, 0] - IMAGE_CENTER[0]) * SPACING_MM
    return np.column_stack([x, y, z])


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


def tube_for_polyline(points, radius=0.55, segments=12):
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


def sphere(center, radius=0.95, rings=8, segments=12):
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
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Visual connected artery tree preview.\n")
        f.write("# X/Y preserve IM0 artery shape; Z is estimated from two-view refined stereo samples.\n")
        f.write("# Use for visualization only, not clinical measurement.\n")
        offset = 1
        for name, vertices, faces in parts:
            f.write(f"o {name}\n")
            for vertex in vertices:
                f.write(f"v {vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f}\n")
            for a, b, c in faces:
                f.write(f"f {a + offset} {b + offset} {c + offset}\n")
            offset += len(vertices)


def main():
    OUT_DIR.mkdir(exist_ok=True)
    sample_yx, sample_z = load_depth_samples()
    depth_model = RBFInterpolator(sample_yx, sample_z, neighbors=50, smoothing=5.0)

    with open(CENTERLINE_DIR / "IM0_frame_030_centerline_graph.json", "r", encoding="utf-8") as f:
        graph = json.load(f)

    parts = []
    for branch in graph["branches"]:
        if branch["points"] < 30:
            continue
        points_3d = yx_to_visual_3d(branch["centerline_yx"], depth_model)
        vertices, faces = tube_for_polyline(points_3d, radius=0.55, segments=12)
        parts.append((f"branch_{branch['branch_id']:02d}", vertices, faces))

    # Junction spheres bridge tiny gaps caused by skeleton branch decomposition.
    junctions = np.asarray(graph["junctions_yx"], dtype=np.float64)
    if len(junctions):
        # Cluster dense junction pixels into a few centers.
        remaining = list(range(len(junctions)))
        clusters = []
        while remaining:
            seed = remaining.pop(0)
            center = junctions[seed]
            cluster = [seed]
            changed = True
            while changed:
                changed = False
                new_remaining = []
                current = junctions[cluster].mean(axis=0)
                for idx in remaining:
                    if np.linalg.norm(junctions[idx] - current) <= 10.0:
                        cluster.append(idx)
                        changed = True
                    else:
                        new_remaining.append(idx)
                remaining = new_remaining
            clusters.append(junctions[cluster].mean(axis=0))

        for idx, center_yx in enumerate(clusters):
            center_3d = yx_to_visual_3d([center_yx], depth_model)[0]
            vertices, faces = sphere(center_3d, radius=1.0, rings=8, segments=12)
            parts.append((f"junction_{idx:02d}", vertices, faces))

    obj_path = OUT_DIR / "visual_connected_artery_tree.obj"
    write_obj(obj_path, parts)
    print(f"Wrote: {obj_path}")
    print("This is the connected visual preview you should open in Blender.")


if __name__ == "__main__":
    main()
