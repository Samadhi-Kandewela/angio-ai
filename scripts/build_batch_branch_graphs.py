"""
Generate branch graphs and branch-match candidates for the batch 3D review queue.

Input:
    dicom_dataset_processed/
    dicom_batch_3d_review/

Output per selected case/group:
    branch_graph_view_a.json
    branch_graph_view_b.json
    branch_overlay_view_a.png
    branch_overlay_view_b.png
    branch_match_candidates.csv
    branch_match_overlay.png

This is still a review/preparation stage. It proposes branch correspondences; it
does not treat them as doctor-validated truth.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from dicom_3d_pipeline import Clip, extract_graph, view_from_clip  # noqa: E402
from epipolar_optimized_centerline import one_to_one_matches  # noqa: E402


FRAME_RE = re.compile(r"(\d+)\s*\(Q")


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


def to_int(value: str, default: int = 0) -> int:
    try:
        if value in ("", None):
            return default
        return int(float(value))
    except Exception:
        return default


def to_float(value: str, default: float = 0.0) -> float:
    try:
        if value in ("", None):
            return default
        return float(value)
    except Exception:
        return default


def parse_first_frame(text: str) -> int | None:
    match = FRAME_RE.search(text or "")
    if not match:
        return None
    return int(match.group(1))


def clip_from_row(row: Dict[str, str]) -> Clip:
    spacing_text = row.get("imager_pixel_spacing_mm", "")
    parts = [part for part in spacing_text.replace(",", ";").split(";") if part.strip()]
    spacing = (to_float(parts[0], 1.0), to_float(parts[1], 1.0)) if len(parts) >= 2 else (1.0, 1.0)
    return Clip(
        path=ROOT / row["raw_path"],
        index=to_int(row["clip_index"]),
        frames=to_int(row["frames"]),
        rows=to_int(row["rows"]),
        cols=to_int(row["cols"]),
        primary=to_float(row["primary_angle_deg"]),
        secondary=to_float(row["secondary_angle_deg"]),
        sdd=to_float(row["source_detector_distance_mm"], 1000.0),
        spd=to_float(row["source_patient_distance_mm"], 750.0),
        spacing=spacing,
        acquisition_time=row.get("acquisition_time", ""),
    )


def load_or_make_frame_assets(case_dir: Path, clip: Clip, frame_no: int) -> Tuple[np.ndarray, np.ndarray]:
    frame_dir = case_dir / "sampled_frames"
    stem = f"clip_{clip.index:02d}_frame_{frame_no:03d}"
    original_path = frame_dir / f"{stem}_original.png"
    mask_path = frame_dir / f"{stem}_mask.png"
    if original_path.exists() and mask_path.exists():
        gray = cv2.imread(str(original_path), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if gray is not None and mask is not None:
            return gray, mask
    raise FileNotFoundError(f"Missing sampled mask/original for C{clip.index} frame {frame_no}: {mask_path}")


def branch_json(branches: List[Dict[str, object]]) -> List[Dict[str, object]]:
    clean = []
    for branch in branches:
        clean.append(
            {
                "branch_id": int(branch["branch_id"]),
                "points": int(branch["points"]),
                "length_px": round(float(branch["length_px"]), 3),
                "mean_diameter_px": round(float(branch["mean_diameter_px"]), 3),
                "centerline_yx": branch["centerline_yx"],
            }
        )
    return clean


def draw_branch_overlay(gray: np.ndarray, clean_mask: np.ndarray, branches: List[Dict[str, object]]) -> np.ndarray:
    canvas = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    red = np.zeros_like(canvas)
    red[clean_mask > 0] = (0, 0, 255)
    canvas = cv2.addWeighted(canvas, 0.78, red, 0.35, 0)
    for branch in branches:
        pts = np.asarray(branch["centerline_yx"], dtype=np.int32)
        if len(pts) < 2:
            continue
        pixels = np.column_stack([pts[:, 1], pts[:, 0]])
        cv2.polylines(canvas, [pixels], False, (0, 255, 0), 2, cv2.LINE_AA)
        mid = tuple(int(v) for v in pixels[len(pixels) // 2])
        cv2.circle(canvas, mid, 4, (0, 255, 255), -1, cv2.LINE_AA)
        cv2.putText(canvas, str(branch["branch_id"]), (mid[0] + 5, mid[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
    return canvas


def draw_match_overlay(
    gray_a: np.ndarray,
    gray_b: np.ndarray,
    branches_a: List[Dict[str, object]],
    branches_b: List[Dict[str, object]],
    matches: List[Dict[str, object]],
) -> np.ndarray:
    left = cv2.cvtColor(gray_a, cv2.COLOR_GRAY2BGR)
    right = cv2.cvtColor(gray_b, cv2.COLOR_GRAY2BGR)
    colors = {
        "reliable": (255, 255, 255),
        "usable": (255, 120, 0),
        "single_view_preserved": (0, 165, 255),
    }
    branch_b_by_id = {int(branch["branch_id"]): branch for branch in branches_b}
    for item in matches:
        status = str(item["status"])
        color = colors.get(status, (80, 80, 80))
        branch_a = item["branch_a"]
        branch_b = branch_b_by_id.get(int(item["branch_b"]["branch_id"]))
        for canvas, branch in ((left, branch_a), (right, branch_b)):
            if branch is None:
                continue
            pts = np.asarray(branch["centerline_yx"], dtype=np.int32)
            if len(pts) < 2:
                continue
            pixels = np.column_stack([pts[:, 1], pts[:, 0]])
            cv2.polylines(canvas, [pixels], False, color, 2, cv2.LINE_AA)
            mid = tuple(int(v) for v in pixels[len(pixels) // 2])
            label = f"{int(branch['branch_id'])}"
            cv2.putText(canvas, label, (mid[0] + 4, mid[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
    gap = np.zeros((left.shape[0], 12, 3), dtype=np.uint8)
    return np.hstack([left, gap, right])


def selected_pairs(pair_rows: List[Dict[str, str]], max_groups: int, pairs_per_group: int) -> List[Dict[str, str]]:
    selected = []
    group_counts: Dict[str, int] = {}
    for row in pair_rows:
        group = row.get("review_group", "all")
        if group not in group_counts and len(group_counts) >= max_groups:
            continue
        count = group_counts.get(group, 0)
        if count >= pairs_per_group:
            continue
        group_counts[group] = count + 1
        selected.append(row)
    return selected


def build_pair_graphs(case_id: str, pair_row: Dict[str, str], processed_dir: Path, output_dir: Path) -> Dict[str, object]:
    case_dir = processed_dir / case_id
    clips = {to_int(row["clip_index"]): clip_from_row(row) for row in read_csv(case_dir / "case_clips_summary.csv")}
    clip_a_idx = to_int(pair_row["view_a_clip_index"])
    clip_b_idx = to_int(pair_row["view_b_clip_index"])
    frame_a = parse_first_frame(pair_row.get("view_a_best_frames", "")) or max(1, round(clips[clip_a_idx].frames * 0.50))
    frame_b = parse_first_frame(pair_row.get("view_b_best_frames", "")) or max(1, round(clips[clip_b_idx].frames * 0.50))

    clip_a = clips[clip_a_idx]
    clip_b = clips[clip_b_idx]
    gray_a, mask_a = load_or_make_frame_assets(case_dir, clip_a, frame_a)
    gray_b, mask_b = load_or_make_frame_assets(case_dir, clip_b, frame_b)
    clean_a, skel_a, dt_a, branches_a = extract_graph(mask_a)
    clean_b, skel_b, dt_b, branches_b = extract_graph(mask_b)
    dist_to_mask_a = cv2.distanceTransform((clean_a == 0).astype(np.uint8), cv2.DIST_L2, 5)
    dist_to_mask_b = cv2.distanceTransform((clean_b == 0).astype(np.uint8), cv2.DIST_L2, 5)

    group_name = pair_row.get("review_group", "pair")
    pair_dir = output_dir / case_id / group_name / f"C{clip_a_idx:02d}_F{frame_a:03d}__C{clip_b_idx:02d}_F{frame_b:03d}"
    pair_dir.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(pair_dir / "view_a_original.png"), gray_a)
    cv2.imwrite(str(pair_dir / "view_b_original.png"), gray_b)
    cv2.imwrite(str(pair_dir / "view_a_mask.png"), mask_a)
    cv2.imwrite(str(pair_dir / "view_b_mask.png"), mask_b)
    cv2.imwrite(str(pair_dir / "view_a_skeleton.png"), skel_a)
    cv2.imwrite(str(pair_dir / "view_b_skeleton.png"), skel_b)
    cv2.imwrite(str(pair_dir / "branch_overlay_view_a.png"), draw_branch_overlay(gray_a, clean_a, branches_a))
    cv2.imwrite(str(pair_dir / "branch_overlay_view_b.png"), draw_branch_overlay(gray_b, clean_b, branches_b))

    view_a = view_from_clip(clip_a, "view_a")
    view_b = view_from_clip(clip_b, "view_b")
    matches = one_to_one_matches(branches_a, branches_b, view_a, view_b, dist_to_mask_a, dist_to_mask_b)
    match_rows = []
    for item in matches:
        score = item["score"]
        match_rows.append(
            {
                "case_id": case_id,
                "review_group": group_name,
                "view_a_clip_index": clip_a_idx,
                "view_a_frame": frame_a,
                "view_a_branch_id": int(item["branch_a"]["branch_id"]),
                "view_b_clip_index": clip_b_idx,
                "view_b_frame": frame_b,
                "view_b_branch_id": int(item["branch_b"]["branch_id"]),
                "status": item["status"],
                "objective": round(float(score["objective"]), 3),
                "ray_median_mm": round(float(score["ray_median_mm"]), 3),
                "ray_p90_mm": round(float(score["ray_p90_mm"]), 3),
                "reproj_median_px": round(float(score["reproj_median_px"]), 3),
                "inside_a_pct": round(float(score["inside_a_pct"]), 2),
                "inside_b_pct": round(float(score["inside_b_pct"]), 2),
                "branch_a_length_px": round(float(item["branch_a"]["length_px"]), 3),
                "branch_b_length_px": round(float(item["branch_b"]["length_px"]), 3),
                "reverse_a": bool(score["reverse_a"]),
                "reverse_b": bool(score["reverse_b"]),
                "doctor_same_branch_yes_no_uncertain": "",
                "doctor_anatomical_label": "",
                "doctor_notes": "",
            }
        )
    write_csv(pair_dir / "branch_match_candidates.csv", match_rows)
    cv2.imwrite(str(pair_dir / "branch_match_overlay.png"), draw_match_overlay(gray_a, gray_b, branches_a, branches_b, matches))

    write_json(
        pair_dir / "branch_graph_view_a.json",
        {
            "case_id": case_id,
            "clip_index": clip_a_idx,
            "frame": frame_a,
            "view_geometry": view_a,
            "num_branches": len(branches_a),
            "branches": branch_json(branches_a),
        },
    )
    write_json(
        pair_dir / "branch_graph_view_b.json",
        {
            "case_id": case_id,
            "clip_index": clip_b_idx,
            "frame": frame_b,
            "view_geometry": view_b,
            "num_branches": len(branches_b),
            "branches": branch_json(branches_b),
        },
    )

    summary = {
        "case_id": case_id,
        "review_group": group_name,
        "view_a_clip_index": clip_a_idx,
        "view_a_frame": frame_a,
        "view_b_clip_index": clip_b_idx,
        "view_b_frame": frame_b,
        "num_view_a_branches": len(branches_a),
        "num_view_b_branches": len(branches_b),
        "num_match_candidates": len(match_rows),
        "num_reliable": sum(1 for row in match_rows if row["status"] == "reliable"),
        "num_usable": sum(1 for row in match_rows if row["status"] == "usable"),
        "output_dir": str(pair_dir),
    }
    write_json(pair_dir / "branch_pair_summary.json", summary)
    return summary


def build_batch(
    processed_dir: Path,
    review_dir: Path,
    output_dir: Path,
    max_cases: int | None,
    max_groups_per_case: int,
    pairs_per_group: int,
) -> Dict[str, object]:
    summaries = []
    for case_dir in sorted(review_dir.glob("case_*")):
        if max_cases is not None and len(summaries) >= max_cases * max_groups_per_case * pairs_per_group:
            break
        case_id = case_dir.name
        rows = read_csv(case_dir / "top_candidate_view_pairs.csv")
        for pair_row in selected_pairs(rows, max_groups=max_groups_per_case, pairs_per_group=pairs_per_group):
            summaries.append(build_pair_graphs(case_id, pair_row, processed_dir, output_dir))
    index_fields = [
        "case_id",
        "review_group",
        "view_a_clip_index",
        "view_a_frame",
        "view_b_clip_index",
        "view_b_frame",
        "num_view_a_branches",
        "num_view_b_branches",
        "num_match_candidates",
        "num_reliable",
        "num_usable",
        "output_dir",
    ]
    write_csv(output_dir / "batch_branch_graph_index.csv", summaries, index_fields)
    payload = {
        "processed_dir": str(processed_dir),
        "review_dir": str(review_dir),
        "output_dir": str(output_dir),
        "num_pair_graphs": len(summaries),
        "outputs": {
            "batch_branch_graph_index": str(output_dir / "batch_branch_graph_index.csv"),
        },
    }
    write_json(output_dir / "batch_branch_graph_summary.json", payload)
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", type=Path, default=ROOT / "dicom_dataset_processed")
    parser.add_argument("--review-dir", type=Path, default=ROOT / "dicom_batch_3d_review")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "dicom_batch_branch_graphs")
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--max-groups-per-case", type=int, default=2)
    parser.add_argument("--pairs-per-group", type=int, default=1)
    args = parser.parse_args()

    summary = build_batch(
        args.processed_dir,
        args.review_dir,
        args.output_dir,
        args.max_cases,
        args.max_groups_per_case,
        args.pairs_per_group,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
