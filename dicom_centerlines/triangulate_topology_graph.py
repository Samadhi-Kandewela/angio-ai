from pathlib import Path
import csv
import json

import cv2
import matplotlib.pyplot as plt
import numpy as np

from triangulate_rough_skeleton import (
    closest_point_between_rays,
    project_point,
    ray_from_pixel,
    resample_polyline,
)


ROOT = Path(__file__).resolve().parents[1]
WORK_DIR = ROOT / "dicom_centerlines"
OUT_DIR = ROOT / "dicom_3d_topology"


def load_graph(view_name):
    with open(WORK_DIR / f"{view_name}_centerline_graph.json", "r", encoding="utf-8") as f:
        graph = json.load(f)
    return {int(branch["branch_id"]): branch for branch in graph["branches"]}


def get_oriented_branch(graph, branch_id, reverse):
    pts = list(graph[int(branch_id)]["centerline_yx"])
    if reverse:
        pts = list(reversed(pts))
    return pts


def triangulate_samples(samples_a, samples_b, view_a, view_b):
    points = []
    residuals = []
    for point_a, point_b in zip(samples_a, samples_b):
        source_a, dir_a = ray_from_pixel(point_a, view_a)
        source_b, dir_b = ray_from_pixel(point_b, view_b)
        point_3d, residual = closest_point_between_rays(source_a, dir_a, source_b, dir_b)
        points.append(point_3d)
        residuals.append(residual)
    return np.asarray(points, dtype=np.float64), np.asarray(residuals, dtype=np.float64)


def snap_endpoint(points, node_point, at_start, blend=8):
    points = points.copy()
    if len(points) == 0:
        return points
    if at_start:
        limit = min(blend, len(points))
        for idx in range(limit):
            weight = 1.0 - idx / max(limit - 1, 1)
            points[idx] = weight * node_point + (1.0 - weight) * points[idx]
        points[0] = node_point
    else:
        limit = min(blend, len(points))
        for offset in range(limit):
            idx = len(points) - 1 - offset
            weight = 1.0 - offset / max(limit - 1, 1)
            points[idx] = weight * node_point + (1.0 - weight) * points[idx]
        points[-1] = node_point
    return points


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


def sphere(center, radius=0.9, rings=8, segments=12):
    vertices = []
    faces = []
    for ring in range(rings + 1):
        phi = np.pi * ring / rings
        for seg in range(segments):
            theta = 2.0 * np.pi * seg / segments
            vertices.append(
                center
                + radius * np.array(
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
        f.write("# Topology-aware connected main-vessel reconstruction.\n")
        f.write("# Shared bifurcation nodes are triangulated once and reused by connected edges.\n")
        f.write("# Prototype for reconstruction validation, not clinical measurement.\n")
        offset = 1
        for name, vertices, faces in parts:
            f.write(f"o {name}\n")
            for vertex in vertices:
                f.write(f"v {vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f}\n")
            for a, b, c in faces:
                f.write(f"f {a + offset} {b + offset} {c + offset}\n")
            offset += len(vertices)


def make_plot(edge_points):
    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")
    for edge_id, points in edge_points.items():
        ax.plot(points[:, 0], points[:, 1], points[:, 2], marker="o", markersize=1.8, linewidth=1.6, label=edge_id)
    ax.set_title("Topology-Aware Connected 3D Centerline")
    ax.set_xlabel("X mm")
    ax.set_ylabel("Y mm")
    ax.set_zlabel("Z mm")
    ax.legend(loc="best", fontsize=7)
    ax.view_init(elev=24, azim=-58)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "topology_connected_centerline.png", dpi=180)
    plt.close(fig)


def draw_reprojection(edge_rows, view, output_name):
    image = cv2.imread(str(ROOT / "dicom_selected_segmentation" / f"{view['name']}_original.png"), cv2.IMREAD_GRAYSCALE)
    canvas = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for row in edge_rows:
        point = np.array([row["x_mm"], row["y_mm"], row["z_mm"]], dtype=np.float64)
        yx = project_point(point, view)
        cv2.circle(canvas, (int(round(yx[1])), int(round(yx[0]))), 2, (0, 0, 255), -1)
    cv2.imwrite(str(OUT_DIR / output_name), canvas)


def main():
    OUT_DIR.mkdir(exist_ok=True)
    with open(WORK_DIR / "topology_branch_matches.json", "r", encoding="utf-8") as f:
        config = json.load(f)

    view_a = config["view_a"]
    view_b = config["view_b"]
    graph_a = load_graph(view_a["name"])
    graph_b = load_graph(view_b["name"])

    edge_records = []
    node_candidates = {}
    residuals = {}

    for edge in config["edges"]:
        branch_a = get_oriented_branch(graph_a, edge["view_a_branch_id"], edge.get("reverse_view_a", False))
        branch_b = get_oriented_branch(graph_b, edge["view_b_branch_id"], edge.get("reverse_view_b", False))
        samples = int(edge["samples"])
        samples_a = resample_polyline(branch_a, samples)
        samples_b = resample_polyline(branch_b, samples)
        points_3d, ray_residuals = triangulate_samples(samples_a, samples_b, view_a, view_b)

        node_candidates.setdefault(edge["start_node"], []).append(points_3d[0])
        node_candidates.setdefault(edge["end_node"], []).append(points_3d[-1])
        residuals[edge["edge_id"]] = ray_residuals
        edge_records.append(
            {
                "edge": edge,
                "samples_a": samples_a,
                "samples_b": samples_b,
                "points_3d": points_3d,
                "residuals": ray_residuals,
            }
        )

    node_points = {name: np.mean(points, axis=0) for name, points in node_candidates.items()}

    edge_points = {}
    rows = []
    for record in edge_records:
        edge = record["edge"]
        points = record["points_3d"]
        points = snap_endpoint(points, node_points[edge["start_node"]], at_start=True)
        points = snap_endpoint(points, node_points[edge["end_node"]], at_start=False)
        edge_points[edge["edge_id"]] = points
        for idx, point in enumerate(points):
            rows.append(
                {
                    "edge_id": edge["edge_id"],
                    "sample_index": idx,
                    "start_node": edge["start_node"],
                    "end_node": edge["end_node"],
                    "x_mm": float(point[0]),
                    "y_mm": float(point[1]),
                    "z_mm": float(point[2]),
                    "ray_residual_mm": float(record["residuals"][idx]),
                    "view_a_y": float(record["samples_a"][idx][0]),
                    "view_a_x": float(record["samples_a"][idx][1]),
                    "view_b_y": float(record["samples_b"][idx][0]),
                    "view_b_x": float(record["samples_b"][idx][1]),
                }
            )

    with open(OUT_DIR / "topology_connected_points.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    graph_out = {
        "nodes": {name: [float(v) for v in point] for name, point in node_points.items()},
        "edges": [
            {
                "edge_id": edge["edge_id"],
                "start_node": edge["start_node"],
                "end_node": edge["end_node"],
                "samples": int(edge["samples"]),
                "mean_residual_mm": float(np.mean(residuals[edge["edge_id"]])),
                "median_residual_mm": float(np.median(residuals[edge["edge_id"]])),
                "max_residual_mm": float(np.max(residuals[edge["edge_id"]])),
                "confidence": edge.get("confidence", ""),
            }
            for edge in config["edges"]
        ],
    }
    with open(OUT_DIR / "topology_connected_graph.json", "w", encoding="utf-8") as f:
        json.dump(graph_out, f, indent=2)

    parts = []
    for edge_id, points in edge_points.items():
        vertices, faces = tube_for_polyline(points, radius=0.55, segments=12)
        parts.append((edge_id, vertices, faces))
    for node_name, point in node_points.items():
        vertices, faces = sphere(point, radius=0.85, rings=8, segments=12)
        parts.append((f"node_{node_name}", vertices, faces))
    write_obj(OUT_DIR / "topology_connected_main_tree.obj", parts)

    make_plot(edge_points)
    draw_reprojection(rows, view_a, "topology_reprojection_IM0.png")
    draw_reprojection(rows, view_b, "topology_reprojection_IM2.png")

    all_res = np.concatenate(list(residuals.values()))
    summary = {
        "status": "topology-aware connected main-vessel prototype",
        "warning": "This is a connected graph reconstruction of the main vessel chain. It is a stronger validation artifact than disconnected tubes, but still needs global reprojection optimization and radius modeling before clinical use.",
        "num_nodes": len(node_points),
        "num_edges": len(edge_points),
        "num_points": len(rows),
        "mean_residual_mm": float(np.mean(all_res)),
        "median_residual_mm": float(np.median(all_res)),
        "max_residual_mm": float(np.max(all_res)),
        "outputs": {
            "obj": str(OUT_DIR / "topology_connected_main_tree.obj"),
            "graph_json": str(OUT_DIR / "topology_connected_graph.json"),
            "points_csv": str(OUT_DIR / "topology_connected_points.csv"),
            "preview_png": str(OUT_DIR / "topology_connected_centerline.png"),
            "reprojection_IM0": str(OUT_DIR / "topology_reprojection_IM0.png"),
            "reprojection_IM2": str(OUT_DIR / "topology_reprojection_IM2.png"),
        },
    }
    with open(OUT_DIR / "topology_connected_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
