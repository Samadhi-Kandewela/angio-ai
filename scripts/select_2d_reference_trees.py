"""
Select and repair 2D reference artery trees before 3D reconstruction.

This stage ranks sampled frames by whether they contain a complete, connected
artery tree. It creates doctor-review overlays with numbered branches. The 3D
pipeline should use these reference trees first, then use other views for depth.

Example:
    python scripts/select_2d_reference_trees.py --processed-dir dicom_dataset_processed --review-dir dicom_batch_3d_review --output-dir dicom_2d_reference_trees
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from skimage.morphology import skeletonize


ROOT = Path(__file__).resolve().parents[1]
SAMPLE_RE = re.compile(r"clip_(\d+)_frame_(\d+)_mask\.png$")
NEI8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str] | None = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def to_float(value: str, default: float = 0.0) -> float:
    try:
        if value in ("", None):
            return default
        return float(value)
    except Exception:
        return default


def to_int(value: str, default: int = 0) -> int:
    try:
        if value in ("", None):
            return default
        return int(float(value))
    except Exception:
        return default


def clean_and_repair_mask(mask: np.ndarray) -> np.ndarray:
    _, bw = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    # Gentle repair: close small segmentation gaps without inventing large vessels.
    for size, iterations in ((3, 1), (5, 1), (7, 1)):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)

    count, labels, stats, _ = cv2.connectedComponentsWithStats((bw > 0).astype(np.uint8), 8)
    out = np.zeros_like(bw)
    areas = []
    for idx in range(1, count):
        area = int(stats[idx, cv2.CC_STAT_AREA])
        if area >= 80:
            areas.append((area, idx))
    if not areas:
        return out
    areas.sort(reverse=True)
    largest = areas[0][0]
    for area, idx in areas:
        if area >= max(120, largest * 0.08):
            out[labels == idx] = 255
    return out


def adjacency(skel: np.ndarray):
    ys, xs = np.where(skel > 0)
    pts = set(zip(ys.tolist(), xs.tolist()))
    adj = {p: [] for p in pts}
    for y, x in list(pts):
        for dy, dx in NEI8:
            q = (y + dy, x + dx)
            if q in pts:
                adj[(y, x)].append(q)
    return adj


def trace_branches(adj, min_len: int = 18):
    if not adj:
        return []
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
    return sorted(branches, key=len, reverse=True)


def graph_metrics(mask: np.ndarray) -> Tuple[np.ndarray, Dict[str, object], List[Dict[str, object]]]:
    repaired = clean_and_repair_mask(mask)
    skel = (skeletonize(repaired > 0).astype(np.uint8)) * 255
    adj = adjacency(skel)
    branches_raw = trace_branches(adj)
    total_len = 0.0
    branches = []
    for branch_id, path in enumerate(branches_raw):
        pts = np.asarray(path, dtype=np.float32)
        length = float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum()) if len(pts) > 1 else 0.0
        total_len += length
        branches.append(
            {
                "branch_id": branch_id + 1,
                "points": len(path),
                "length_px": round(length, 2),
                "centerline_yx": [[int(y), int(x)] for y, x in path],
            }
        )

    count, labels, stats, _ = cv2.connectedComponentsWithStats((repaired > 0).astype(np.uint8), 8)
    comp_areas = [int(stats[idx, cv2.CC_STAT_AREA]) for idx in range(1, count)]
    comp_areas.sort(reverse=True)
    largest = comp_areas[0] if comp_areas else 0
    total_area = int(sum(comp_areas))
    largest_pct = float(largest / max(total_area, 1) * 100.0)
    degrees = [len(n) for n in adj.values()]
    endpoints = int(sum(1 for d in degrees if d == 1))
    junctions = int(sum(1 for d in degrees if d >= 3))
    long_branches = int(sum(1 for b in branches if float(b["length_px"]) >= 80.0))

    metrics = {
        "vessel_area_px": total_area,
        "component_count": len(comp_areas),
        "largest_component_area_px": largest,
        "largest_component_pct": round(largest_pct, 2),
        "skeleton_pixels": int(np.count_nonzero(skel)),
        "branch_count": len(branches),
        "long_branch_count": long_branches,
        "centerline_length_px": round(total_len, 2),
        "endpoint_count": endpoints,
        "junction_count": junctions,
    }
    return repaired, metrics, branches


def topology_score(metrics: Dict[str, object], segmentation_quality: float = 0.0) -> float:
    area = float(metrics["vessel_area_px"])
    largest_pct = float(metrics["largest_component_pct"])
    length = float(metrics["centerline_length_px"])
    branch_count = float(metrics["branch_count"])
    long_branches = float(metrics["long_branch_count"])
    junctions = float(metrics["junction_count"])
    components = float(metrics["component_count"])
    endpoints = float(metrics["endpoint_count"])

    area_score = min(area / 16000.0, 1.0) * 18.0
    connected_score = min(largest_pct / 95.0, 1.0) * 22.0
    length_score = min(length / 1800.0, 1.0) * 22.0
    branch_score = min(branch_count / 12.0, 1.0) * 13.0
    long_branch_score = min(long_branches / 5.0, 1.0) * 10.0
    junction_score = min(junctions / 5.0, 1.0) * 8.0
    segmentation_score = min(segmentation_quality / 100.0, 1.0) * 12.0
    fragment_penalty = max(0.0, components - 2.0) * 4.0
    endpoint_penalty = max(0.0, endpoints - 14.0) * 0.8
    return round(max(0.0, area_score + connected_score + length_score + branch_score + long_branch_score + junction_score + segmentation_score - fragment_penalty - endpoint_penalty), 3)


def draw_labeled_overlay(original: np.ndarray, repaired: np.ndarray, branches: List[Dict[str, object]]) -> np.ndarray:
    canvas = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
    red = np.zeros_like(canvas)
    red[repaired > 0] = (0, 0, 255)
    canvas = cv2.addWeighted(canvas, 0.72, red, 0.36, 0)
    for branch in branches:
        pts = np.asarray(branch["centerline_yx"], dtype=np.int32)
        if len(pts) < 2:
            continue
        pixels = np.column_stack([pts[:, 1], pts[:, 0]])
        cv2.polylines(canvas, [pixels], False, (0, 255, 0), 2, cv2.LINE_AA)
        mid = tuple(int(v) for v in pixels[len(pixels) // 2])
        cv2.circle(canvas, mid, 6, (0, 255, 255), -1, cv2.LINE_AA)
        cv2.putText(canvas, str(branch["branch_id"]), (mid[0] + 6, mid[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
    return canvas


def make_review_sheet(original: np.ndarray, overlay: np.ndarray, title: str) -> np.ndarray:
    original_bgr = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
    h, w = original_bgr.shape[:2]
    header_h = 34
    sheet = np.zeros((h + header_h, w * 2, 3), dtype=np.uint8)
    sheet[header_h:, :w] = original_bgr
    sheet[header_h:, w:] = overlay
    cv2.putText(sheet, title, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(sheet, "Original", (10, header_h + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(sheet, "Numbered extracted tree", (w + 10, header_h + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
    return sheet


def score_lookup(case_dir: Path) -> Dict[Tuple[int, int], float]:
    lookup = {}
    for row in read_csv(case_dir / "sampled_frame_scores.csv"):
        lookup[(to_int(row["clip_index"]), to_int(row["frame"]))] = to_float(row.get("segmentation_quality_score", ""))
    return lookup


def clip_group_lookup(review_case_dir: Path) -> Dict[int, str]:
    lookup = {}
    for row in read_csv(review_case_dir / "clip_level_doctor_summary.csv"):
        lookup[to_int(row["clip_index"])] = row.get("auto_review_group", "")
    return lookup


def process_case(case_dir: Path, review_dir: Path, output_case_dir: Path, top_n: int) -> List[Dict[str, object]]:
    case_id = case_dir.name
    frame_dir = case_dir / "sampled_frames"
    output_case_dir.mkdir(parents=True, exist_ok=True)
    scores = score_lookup(case_dir)
    groups = clip_group_lookup(review_dir / case_id)
    rows = []

    for mask_path in sorted(frame_dir.glob("clip_*_frame_*_mask.png")):
        match = SAMPLE_RE.match(mask_path.name)
        if not match:
            continue
        clip_idx = int(match.group(1))
        frame_no = int(match.group(2))
        original_path = frame_dir / f"clip_{clip_idx:02d}_frame_{frame_no:03d}_original.png"
        original = cv2.imread(str(original_path), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if original is None or mask is None:
            continue
        repaired, metrics, branches = graph_metrics(mask)
        seg_q = scores.get((clip_idx, frame_no), 0.0)
        score = topology_score(metrics, seg_q)
        group = groups.get(clip_idx, "")
        stem = f"clip_{clip_idx:02d}_frame_{frame_no:03d}"
        original_out = output_case_dir / f"{stem}_original.png"
        overlay_out = output_case_dir / f"{stem}_numbered_branches.png"
        repaired_out = output_case_dir / f"{stem}_repaired_mask.png"
        graph_out = output_case_dir / f"{stem}_branch_graph.json"
        review_sheet_out = output_case_dir / f"{stem}_review_sheet.png"
        overlay = draw_labeled_overlay(original, repaired, branches)
        row = {
            "case_id": case_id,
            "clip_index": clip_idx,
            "frame": frame_no,
            "auto_review_group": group,
            "topology_score": score,
            "segmentation_quality_score": round(seg_q, 3),
            **metrics,
            "doctor_reference_tree_yes_no": "",
            "doctor_artery_lca_rca_uncertain": "",
            "doctor_notes": "",
            "original_path": str(original_out),
            "overlay_path": str(overlay_out),
            "review_sheet_path": str(review_sheet_out),
            "repaired_mask_path": str(repaired_out),
            "branch_graph_path": str(graph_out),
        }
        cv2.imwrite(str(original_out), original)
        cv2.imwrite(str(repaired_out), repaired)
        cv2.imwrite(str(overlay_out), overlay)
        cv2.imwrite(str(review_sheet_out), make_review_sheet(original, overlay, f"{case_id} C{clip_idx} F{frame_no} {group}"))
        write_json(
            graph_out,
            {
                "case_id": case_id,
                "clip_index": clip_idx,
                "frame": frame_no,
                "auto_review_group": group,
                "topology_score": score,
                "metrics": metrics,
                "branches": branches,
            },
        )
        rows.append(row)

    rows.sort(key=lambda r: float(r["topology_score"]), reverse=True)
    fieldnames = list(rows[0].keys()) if rows else []
    write_csv(output_case_dir / "reference_tree_candidates.csv", rows, fieldnames)
    write_csv(output_case_dir / "top_reference_tree_candidates.csv", rows[:top_n], fieldnames)

    # Group-level shortlist for likely LCA/RCA acquisition blocks.
    by_group: Dict[str, List[Dict[str, object]]] = {}
    for row in rows:
        by_group.setdefault(str(row["auto_review_group"] or "ungrouped"), []).append(row)
    group_rows = []
    for group, group_items in by_group.items():
        for rank, row in enumerate(group_items[:top_n], start=1):
            item = dict(row)
            item["group_rank"] = rank
            item["group_name"] = group
            group_rows.append(item)
    write_csv(output_case_dir / "top_reference_tree_candidates_by_group.csv", group_rows, list(group_rows[0].keys()) if group_rows else fieldnames)
    write_doctor_review_package(output_case_dir, by_group, per_group=2)
    return rows


def write_doctor_review_package(output_case_dir: Path, by_group: Dict[str, List[Dict[str, object]]], per_group: int = 2):
    review_dir = output_case_dir / "doctor_reference_review"
    review_dir.mkdir(parents=True, exist_ok=True)
    review_rows = []
    contact_items = []
    for group in sorted(by_group):
        for rank, row in enumerate(by_group[group][:per_group], start=1):
            prefix = f"{group}_rank_{rank}_C{int(row['clip_index']):02d}_F{int(row['frame']):03d}"
            copies = {
                "original": review_dir / f"{prefix}_original.png",
                "numbered_overlay": review_dir / f"{prefix}_numbered_overlay.png",
                "review_sheet": review_dir / f"{prefix}_review_sheet.png",
                "repaired_mask": review_dir / f"{prefix}_repaired_mask.png",
                "branch_graph": review_dir / f"{prefix}_branch_graph.json",
            }
            for source_key, target in (
                ("original_path", copies["original"]),
                ("overlay_path", copies["numbered_overlay"]),
                ("review_sheet_path", copies["review_sheet"]),
                ("repaired_mask_path", copies["repaired_mask"]),
                ("branch_graph_path", copies["branch_graph"]),
            ):
                source = Path(str(row[source_key]))
                if source.exists():
                    target.write_bytes(source.read_bytes())
            review_rows.append(
                {
                    "group_name": group,
                    "group_rank": rank,
                    "clip_index": row["clip_index"],
                    "frame": row["frame"],
                    "topology_score": row["topology_score"],
                    "segmentation_quality_score": row["segmentation_quality_score"],
                    "original": str(copies["original"]),
                    "numbered_overlay": str(copies["numbered_overlay"]),
                    "review_sheet": str(copies["review_sheet"]),
                    "doctor_reference_tree_yes_no": "",
                    "doctor_artery_lca_rca_uncertain": "",
                    "doctor_notes": "",
                }
            )
            sheet = cv2.imread(str(copies["review_sheet"]), cv2.IMREAD_COLOR)
            if sheet is not None:
                contact_items.append((f"{group} rank {rank}", sheet))

    if review_rows:
        write_csv(review_dir / "doctor_reference_review_index.csv", review_rows)
    if contact_items:
        thumbs = []
        target_w = 720
        for label, image in contact_items:
            scale = target_w / image.shape[1]
            resized = cv2.resize(image, (target_w, int(image.shape[0] * scale)), interpolation=cv2.INTER_AREA)
            cv2.putText(resized, label, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
            thumbs.append(resized)
        max_h = max(img.shape[0] for img in thumbs)
        padded = []
        for img in thumbs:
            if img.shape[0] < max_h:
                pad = np.zeros((max_h - img.shape[0], img.shape[1], 3), dtype=np.uint8)
                img = np.vstack([img, pad])
            padded.append(img)
        cols = 2
        rows = int(math.ceil(len(padded) / cols))
        blank = np.zeros_like(padded[0])
        while len(padded) < rows * cols:
            padded.append(blank.copy())
        grid = np.vstack([np.hstack(padded[r * cols:(r + 1) * cols]) for r in range(rows)])
        cv2.imwrite(str(review_dir / "doctor_reference_contact_sheet.png"), grid)


def build_batch(processed_dir: Path, review_dir: Path, output_dir: Path, top_n: int, max_cases: int | None):
    output_dir.mkdir(parents=True, exist_ok=True)
    index_rows = []
    case_count = 0
    for case_dir in sorted(processed_dir.glob("case_*")):
        if max_cases is not None and case_count >= max_cases:
            break
        if not (case_dir / "sampled_frames").exists():
            continue
        rows = process_case(case_dir, review_dir, output_dir / case_dir.name, top_n)
        case_count += 1
        for rank, row in enumerate(rows[:top_n], start=1):
            item = dict(row)
            item["case_rank"] = rank
            index_rows.append(item)
    index_rows.sort(key=lambda r: float(r["topology_score"]), reverse=True)
    write_csv(output_dir / "batch_reference_tree_ranking.csv", index_rows, list(index_rows[0].keys()) if index_rows else [])
    summary = {
        "processed_dir": str(processed_dir),
        "review_dir": str(review_dir),
        "output_dir": str(output_dir),
        "num_cases_processed": case_count,
        "num_ranked_candidates": len(index_rows),
        "outputs": {
            "batch_reference_tree_ranking": str(output_dir / "batch_reference_tree_ranking.csv"),
        },
        "meaning": {
            "topology_score": "Ranks frames by connected complete artery-tree quality before 3D.",
            "numbered_overlay": "Doctor-facing 2D branch labels for reference-tree validation.",
        },
    }
    write_json(output_dir / "reference_tree_selection_summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", type=Path, default=ROOT / "dicom_dataset_processed")
    parser.add_argument("--review-dir", type=Path, default=ROOT / "dicom_batch_3d_review")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "dicom_2d_reference_trees")
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--max-cases", type=int, default=None)
    args = parser.parse_args()

    print(json.dumps(build_batch(args.processed_dir, args.review_dir, args.output_dir, args.top_n, args.max_cases), indent=2))


if __name__ == "__main__":
    main()
