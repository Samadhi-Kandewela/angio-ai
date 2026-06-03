from pathlib import Path
import csv
from collections import defaultdict

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
IN_CSV = ROOT / "dicom_3d_rough" / "refined_3d_points.csv"
OUT_DIR = ROOT / "dicom_3d_rough"


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


def tube_for_polyline(points, radius=0.8, segments=12):
    vertices = []
    faces = []
    points = smooth_points(points)

    for idx, point in enumerate(points):
        if idx == 0:
            tangent = points[1] - points[0]
        elif idx == len(points) - 1:
            tangent = points[-1] - points[-2]
        else:
            tangent = points[idx + 1] - points[idx - 1]
        normal, binormal = make_frame(tangent)
        ring = []
        for seg in range(segments):
            angle = 2.0 * np.pi * seg / segments
            ring_point = point + radius * (np.cos(angle) * normal + np.sin(angle) * binormal)
            ring.append(ring_point)
        vertices.extend(ring)

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
        f.write("# Rough coronary tube mesh generated from refined two-view triangulation.\n")
        f.write("# Constant radius is used for visualization only, not clinical measurement.\n")
        vertex_offset = 1
        for match_id, vertices, faces in mesh_parts:
            f.write(f"o {match_id}\n")
            for vertex in vertices:
                f.write(f"v {vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f}\n")
            for a, b, c in faces:
                f.write(f"f {a + vertex_offset} {b + vertex_offset} {c + vertex_offset}\n")
            vertex_offset += len(vertices)


def main():
    grouped = defaultdict(list)
    with open(IN_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            grouped[row["match_id"]].append(
                (
                    int(row["sample_index"]),
                    np.array([float(row["x_mm"]), float(row["y_mm"]), float(row["z_mm"])], dtype=np.float64),
                )
            )

    mesh_parts = []
    for match_id, rows in grouped.items():
        rows = sorted(rows, key=lambda item: item[0])
        points = np.vstack([point for _, point in rows])
        vertices, faces = tube_for_polyline(points, radius=0.8, segments=12)
        mesh_parts.append((match_id, vertices, faces))

    out_obj = OUT_DIR / "refined_3d_constant_radius_tube.obj"
    write_obj(out_obj, mesh_parts)
    print(f"Wrote: {out_obj}")
    print("Note: constant-radius tube is for visualization only.")


if __name__ == "__main__":
    main()
