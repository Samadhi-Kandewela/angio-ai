from pathlib import Path
import csv
import json

import cv2
import numpy as np
from skimage.morphology import skeletonize


ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "dicom_selected_segmentation"
OUT_DIR = ROOT / "dicom_centerlines"

SELECTED = [
    ("IM0_frame_030", "-0.1", "-21.90"),
    ("IM2_frame_022", "40.70", "-40.70"),
]

NEI8 = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]

COLORS = [
    (255, 0, 0), (0, 180, 255), (0, 255, 0), (255, 0, 255),
    (255, 255, 0), (0, 128, 255), (180, 255, 0), (255, 128, 0),
]


def clean_mask(mask):
    _, bw = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=1)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel, iterations=1)

    count, labels, stats, _ = cv2.connectedComponentsWithStats((bw > 0).astype(np.uint8), 8)
    out = np.zeros_like(bw)
    for idx in range(1, count):
        if stats[idx, cv2.CC_STAT_AREA] >= 120:
            out[labels == idx] = 255
    return out


def adjacency(skel):
    sk = skel > 0
    ys, xs = np.where(sk)
    pts = set(zip(ys.tolist(), xs.tolist()))
    adj = {p: [] for p in pts}

    for y, x in list(pts):
        for dy, dx in NEI8:
            q = (y + dy, x + dx)
            if q in pts:
                adj[(y, x)].append(q)
    return adj


def trace_branches(adj, min_len=20):
    degree = {node: len(neighbors) for node, neighbors in adj.items()}
    starts = [node for node, deg in degree.items() if deg != 2]
    visited = set()
    branches = []

    for start in starts:
        for neighbor in adj[start]:
            edge = tuple(sorted([start, neighbor]))
            if edge in visited:
                continue

            path = [start, neighbor]
            visited.add(edge)
            prev, cur = start, neighbor

            while degree.get(cur, 0) == 2:
                next_nodes = [node for node in adj[cur] if node != prev]
                if not next_nodes:
                    break
                nxt = next_nodes[0]
                edge = tuple(sorted([cur, nxt]))
                if edge in visited:
                    break
                visited.add(edge)
                path.append(nxt)
                prev, cur = cur, nxt

            if len(path) >= min_len:
                branches.append(path)

    return branches


def endpoints_and_junctions(skel):
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
    counts = cv2.filter2D((skel > 0).astype(np.uint8), -1, kernel, borderType=cv2.BORDER_CONSTANT)
    endpoints = list(zip(*np.where((skel > 0) & (counts == 1))))
    junctions = list(zip(*np.where((skel > 0) & (counts >= 3))))
    return endpoints, junctions


def branch_metrics(branch, dt):
    pts = np.asarray(branch, dtype=np.float32)
    if len(pts) > 1:
        length = float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())
    else:
        length = 0.0

    ys = pts[:, 0].astype(int)
    xs = pts[:, 1].astype(int)
    diameters = 2.0 * dt[ys, xs]
    return {
        "points": int(len(branch)),
        "length_px": round(length, 2),
        "mean_diameter_px": round(float(np.mean(diameters)), 2),
        "min_diameter_px": round(float(np.min(diameters)), 2),
        "start_y": int(branch[0][0]),
        "start_x": int(branch[0][1]),
        "end_y": int(branch[-1][0]),
        "end_x": int(branch[-1][1]),
    }


def draw_branch_id(img, branch_id, point):
    y, x = point
    org = (int(x) + 4, int(y) - 4)
    cv2.putText(img, str(branch_id), org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, str(branch_id), org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)


def process_view(label, primary, secondary):
    original = cv2.imread(str(INPUT_DIR / f"{label}_original.png"), cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(str(INPUT_DIR / f"{label}_mask.png"), cv2.IMREAD_GRAYSCALE)
    if original is None or mask is None:
        raise FileNotFoundError(label)

    clean = clean_mask(mask)
    skel = (skeletonize(clean > 0).astype(np.uint8)) * 255
    adj = adjacency(skel)
    branches = trace_branches(adj, min_len=20)
    dt = cv2.distanceTransform((clean > 0).astype(np.uint8), cv2.DIST_L2, 5)
    branches = sorted(branches, key=lambda branch: branch_metrics(branch, dt)["length_px"], reverse=True)
    endpoints, junctions = endpoints_and_junctions(skel)

    base = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
    skel_vis = base.copy()
    skel_vis[skel > 0] = (0, 255, 255)
    skel_vis = cv2.addWeighted(skel_vis, 0.75, base, 0.25, 0)
    for y, x in endpoints:
        cv2.circle(skel_vis, (int(x), int(y)), 4, (0, 255, 255), -1)
    for y, x in junctions:
        cv2.circle(skel_vis, (int(x), int(y)), 3, (255, 0, 255), 1)

    branch_vis = base.copy()
    branch_data = []
    rows = []
    for branch_id, branch in enumerate(branches):
        color = COLORS[branch_id % len(COLORS)]
        for y, x in branch:
            cv2.circle(branch_vis, (int(x), int(y)), 1, color, -1)
        draw_branch_id(branch_vis, branch_id, branch[len(branch) // 2])

        metrics = branch_metrics(branch, dt)
        row = {
            "view": label,
            "branch_id": branch_id,
            "primary_angle": primary,
            "secondary_angle": secondary,
            **metrics,
        }
        rows.append(row)
        branch_data.append({**row, "centerline_yx": [[int(y), int(x)] for y, x in branch]})

    for y, x in endpoints:
        cv2.circle(branch_vis, (int(x), int(y)), 4, (0, 255, 255), -1)

    cv2.imwrite(str(OUT_DIR / f"{label}_clean_mask.png"), clean)
    cv2.imwrite(str(OUT_DIR / f"{label}_skeleton.png"), skel)
    cv2.imwrite(str(OUT_DIR / f"{label}_skeleton_nodes.png"), skel_vis)
    cv2.imwrite(str(OUT_DIR / f"{label}_branches_colored.png"), branch_vis)

    graph = {
        "view": label,
        "primary_angle": primary,
        "secondary_angle": secondary,
        "image_size": [512, 512],
        "num_skeleton_pixels": int((skel > 0).sum()),
        "num_endpoints": len(endpoints),
        "num_junction_pixels": len(junctions),
        "endpoints_yx": [[int(y), int(x)] for y, x in endpoints],
        "junctions_yx": [[int(y), int(x)] for y, x in junctions],
        "branches": branch_data,
    }
    with open(OUT_DIR / f"{label}_centerline_graph.json", "w", encoding="utf-8") as f:
        json.dump(graph, f, indent=2)

    summary = {
        "view": label,
        "branches": len(branches),
        "skeleton_pixels": int((skel > 0).sum()),
        "endpoints": len(endpoints),
        "junction_pixels": len(junctions),
        "top_branch_lengths": [item["points"] for item in branch_data[:6]],
    }
    return summary, rows


def build_contact_sheet():
    try:
        from PIL import Image, ImageDraw, ImageFont
        font = ImageFont.truetype("arial.ttf", 18)
    except Exception:
        return

    sheet = Image.new("RGB", (512 * 3, 552 * len(SELECTED)), "white")
    draw = ImageDraw.Draw(sheet)
    for row, (label, primary, secondary) in enumerate(SELECTED):
        images = [
            Image.open(INPUT_DIR / f"{label}_overlay.png").convert("RGB"),
            Image.open(OUT_DIR / f"{label}_skeleton_nodes.png").convert("RGB"),
            Image.open(OUT_DIR / f"{label}_branches_colored.png").convert("RGB"),
        ]
        draw.text(
            (8, row * 552 + 10),
            f"{label} | P {primary} S {secondary} | overlay, skeleton nodes, branch IDs",
            fill=(0, 0, 0),
            font=font,
        )
        for col, image in enumerate(images):
            sheet.paste(image, (col * 512, row * 552 + 40))
    sheet.save(OUT_DIR / "centerline_extraction_contact_sheet.png")


def main():
    OUT_DIR.mkdir(exist_ok=True)
    summaries = []
    all_rows = []
    for label, primary, secondary in SELECTED:
        summary, rows = process_view(label, primary, secondary)
        summaries.append(summary)
        all_rows.extend(rows)

    fieldnames = [
        "view", "branch_id", "primary_angle", "secondary_angle", "points",
        "length_px", "mean_diameter_px", "min_diameter_px",
        "start_y", "start_x", "end_y", "end_x",
    ]
    with open(OUT_DIR / "branch_metrics.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    with open(OUT_DIR / "centerline_summary.json", "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)

    build_contact_sheet()
    print(f"Output folder: {OUT_DIR}")
    print(f"Contact sheet: {OUT_DIR / 'centerline_extraction_contact_sheet.png'}")
    for summary in summaries:
        print(summary)


if __name__ == "__main__":
    main()
