"""
Rank batch branch-graph outputs for 3D reconstruction readiness.

The purpose is to identify which case/view-pairs should be reconstructed first,
based on strict epipolar branch-match quality. This avoids creating misleading
3D meshes for weak or mismatched pairs.

Example:
    python scripts/rank_3d_reconstruction_candidates.py --branch-dir dicom_batch_branch_graphs --output-dir dicom_3d_candidate_ranking
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import median
from typing import Dict, List


ROOT = Path(__file__).resolve().parents[1]


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


def readiness_label(score: float, usable_or_reliable: int, reliable: int) -> str:
    if reliable >= 2 and usable_or_reliable >= 4 and score >= 70:
        return "strong_reconstruction_candidate"
    if usable_or_reliable >= 3 and score >= 45:
        return "moderate_reconstruction_candidate"
    if usable_or_reliable >= 1 and score >= 20:
        return "weak_research_candidate"
    return "not_ready"


def score_pair(summary: Dict[str, str], matches: List[Dict[str, str]]) -> Dict[str, object]:
    non_rejected = [row for row in matches if row.get("status") in {"reliable", "usable"}]
    reliable = [row for row in non_rejected if row.get("status") == "reliable"]
    usable = [row for row in non_rejected if row.get("status") == "usable"]
    rejected = [row for row in matches if row.get("status") == "rejected"]

    ray_values = [to_float(row.get("ray_median_mm", "")) for row in non_rejected]
    reproj_values = [to_float(row.get("reproj_median_px", "")) for row in non_rejected]
    inside_values = [
        min(to_float(row.get("inside_a_pct", "")), to_float(row.get("inside_b_pct", "")))
        for row in non_rejected
    ]
    length_values = [
        min(to_float(row.get("branch_a_length_px", "")), to_float(row.get("branch_b_length_px", "")))
        for row in non_rejected
    ]

    median_ray = median(ray_values) if ray_values else 999.0
    median_reproj = median(reproj_values) if reproj_values else 999.0
    median_inside = median(inside_values) if inside_values else 0.0
    supported_length = sum(length_values)

    support_score = 16.0 * len(reliable) + 9.0 * len(usable)
    geometry_score = max(0.0, 25.0 - 3.0 * median_ray - 1.8 * median_reproj)
    mask_score = max(0.0, min(median_inside, 100.0) * 0.12)
    length_score = min(supported_length / 35.0, 18.0)
    rejection_penalty = min(len(rejected) * 1.2, 12.0)
    score = max(0.0, round(support_score + geometry_score + mask_score + length_score - rejection_penalty, 3))

    usable_or_reliable = len(non_rejected)
    return {
        "case_id": summary.get("case_id", ""),
        "review_group": summary.get("review_group", ""),
        "view_a_clip_index": summary.get("view_a_clip_index", ""),
        "view_a_frame": summary.get("view_a_frame", ""),
        "view_b_clip_index": summary.get("view_b_clip_index", ""),
        "view_b_frame": summary.get("view_b_frame", ""),
        "num_view_a_branches": summary.get("num_view_a_branches", ""),
        "num_view_b_branches": summary.get("num_view_b_branches", ""),
        "num_match_candidates": len(matches),
        "num_reliable": len(reliable),
        "num_usable": len(usable),
        "num_rejected": len(rejected),
        "usable_or_reliable_matches": usable_or_reliable,
        "median_supported_ray_mm": round(median_ray, 3) if ray_values else "",
        "median_supported_reproj_px": round(median_reproj, 3) if reproj_values else "",
        "median_supported_inside_pct": round(median_inside, 2) if inside_values else "",
        "supported_branch_length_px": round(supported_length, 2),
        "reconstruction_readiness_score": score,
        "readiness_label": readiness_label(score, usable_or_reliable, len(reliable)),
        "pair_output_dir": summary.get("output_dir", ""),
    }


def load_summary(pair_dir: Path) -> Dict[str, str]:
    path = pair_dir / "branch_pair_summary.json"
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    return {k: str(v) for k, v in payload.items()}


def rank_candidates(branch_dir: Path, output_dir: Path) -> Dict[str, object]:
    rows = []
    for match_path in branch_dir.rglob("branch_match_candidates.csv"):
        pair_dir = match_path.parent
        summary = load_summary(pair_dir)
        if not summary:
            continue
        matches = read_csv(match_path)
        rows.append(score_pair(summary, matches))

    rows.sort(key=lambda row: to_float(str(row["reconstruction_readiness_score"])), reverse=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = [
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
        "num_rejected",
        "usable_or_reliable_matches",
        "median_supported_ray_mm",
        "median_supported_reproj_px",
        "median_supported_inside_pct",
        "supported_branch_length_px",
        "reconstruction_readiness_score",
        "readiness_label",
        "pair_output_dir",
    ]
    write_csv(output_dir / "ranked_3d_reconstruction_candidates.csv", rows, fieldnames)

    case_best: Dict[str, Dict[str, object]] = {}
    for row in rows:
        case_best.setdefault(str(row["case_id"]), row)
    case_rows = sorted(case_best.values(), key=lambda row: to_float(str(row["reconstruction_readiness_score"])), reverse=True)
    write_csv(output_dir / "best_candidate_per_case.csv", case_rows, fieldnames)

    label_counts: Dict[str, int] = {}
    for row in rows:
        label_counts[str(row["readiness_label"])] = label_counts.get(str(row["readiness_label"]), 0) + 1

    summary = {
        "branch_dir": str(branch_dir),
        "output_dir": str(output_dir),
        "num_pair_candidates": len(rows),
        "num_cases_with_candidates": len(case_rows),
        "readiness_label_counts": label_counts,
        "top_10": rows[:10],
        "outputs": {
            "ranked_candidates": str(output_dir / "ranked_3d_reconstruction_candidates.csv"),
            "best_candidate_per_case": str(output_dir / "best_candidate_per_case.csv"),
        },
    }
    write_json(output_dir / "ranking_summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--branch-dir", type=Path, default=ROOT / "dicom_batch_branch_graphs")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "dicom_3d_candidate_ranking")
    args = parser.parse_args()

    summary = rank_candidates(args.branch_dir, args.output_dir)
    print(json.dumps({k: v for k, v in summary.items() if k != "top_10"}, indent=2))


if __name__ == "__main__":
    main()
