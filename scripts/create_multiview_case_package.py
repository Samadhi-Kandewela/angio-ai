"""
Create a doctor-reviewed multi-view package for one DICOM case.

This does not reconstruct 3D yet. It turns doctor-confirmed clip grouping into
machine-readable config files for later multi-view reconstruction:

- LCA/RCA clip groups
- candidate main pairs inside each group
- optional validation clips
- sampled-frame score shortlist, if available
- review CSV templates for matching cardiac phase and branch correspondences

Example:
    python scripts/create_multiview_case_package.py --case-id case_001 --lca-clips 1,2,3 --lca-weak-clips 0,4 --rca-clips 5,6
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from pathlib import Path
from typing import Dict, List, Sequence


ROOT = Path(__file__).resolve().parents[1]


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Dict[str, object]]):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def parse_clip_list(text: str) -> List[int]:
    if not text:
        return []
    return [int(item.strip()) for item in text.split(",") if item.strip() != ""]


def clip_by_index(clips: List[Dict[str, str]]) -> Dict[int, Dict[str, str]]:
    return {int(row["clip_index"]): row for row in clips}


def angle(row: Dict[str, str], key: str) -> float:
    try:
        return float(row[key])
    except Exception:
        return 0.0


def angle_separation(a: Dict[str, str], b: Dict[str, str]) -> float:
    return math.hypot(angle(a, "primary_angle_deg") - angle(b, "primary_angle_deg"), angle(a, "secondary_angle_deg") - angle(b, "secondary_angle_deg"))


def frame_scores_by_clip(case_dir: Path) -> Dict[int, List[Dict[str, str]]]:
    rows = read_csv(case_dir / "sampled_frame_scores.csv")
    by_clip: Dict[int, List[Dict[str, str]]] = {}
    for row in rows:
        by_clip.setdefault(int(row["clip_index"]), []).append(row)
    for clip_rows in by_clip.values():
        clip_rows.sort(key=lambda r: float(r.get("segmentation_quality_score", 0.0)), reverse=True)
    return by_clip


def best_frames_text(scores: Dict[int, List[Dict[str, str]]], clip_index: int, top_n: int = 3) -> str:
    rows = scores.get(clip_index, [])[:top_n]
    return ";".join(f"{row['frame']} (Q{row['segmentation_quality_score']})" for row in rows)


def candidate_pairs_for_group(case_id: str, artery_system: str, group_clips: Sequence[int], clips: Dict[int, Dict[str, str]], scores: Dict[int, List[Dict[str, str]]]):
    rows = []
    for i, clip_a in enumerate(group_clips):
        for clip_b in group_clips[i + 1:]:
            if clip_a not in clips or clip_b not in clips:
                continue
            a, b = clips[clip_a], clips[clip_b]
            sep = angle_separation(a, b)
            rows.append(
                {
                    "case_id": case_id,
                    "artery_system": artery_system,
                    "view_a_clip_index": clip_a,
                    "view_b_clip_index": clip_b,
                    "angle_separation_deg": round(sep, 3),
                    "view_a_primary_angle_deg": a["primary_angle_deg"],
                    "view_a_secondary_angle_deg": a["secondary_angle_deg"],
                    "view_b_primary_angle_deg": b["primary_angle_deg"],
                    "view_b_secondary_angle_deg": b["secondary_angle_deg"],
                    "view_a_best_sampled_frames": best_frames_text(scores, clip_a),
                    "view_b_best_sampled_frames": best_frames_text(scores, clip_b),
                    "doctor_main_pair_yes_no": "",
                    "doctor_best_frame_a": "",
                    "doctor_best_frame_b": "",
                    "doctor_notes": "",
                }
            )
    return sorted(rows, key=lambda row: row["angle_separation_deg"], reverse=True)


def frame_review_rows(case_id: str, artery_system: str, group_clips: Sequence[int], scores: Dict[int, List[Dict[str, str]]]):
    rows = []
    for clip_index in group_clips:
        for row in scores.get(clip_index, []):
            rows.append(
                {
                    "case_id": case_id,
                    "artery_system": artery_system,
                    "clip_index": clip_index,
                    "frame": row["frame"],
                    "segmentation_quality_score": row["segmentation_quality_score"],
                    "branch_count": row["branch_count"],
                    "vessel_area_pct": row["vessel_area_pct"],
                    "centerline_length_px": row["centerline_length_px"],
                    "doctor_same_phase_rank": "",
                    "doctor_use_frame_yes_no": "",
                    "doctor_notes": "",
                }
            )
    return rows


def copy_group_images(case_dir: Path, output_dir: Path, group_name: str, group_clips: Sequence[int]):
    source_frame_dir = case_dir / "sampled_frames"
    if not source_frame_dir.exists():
        return
    target_dir = output_dir / group_name / "sampled_frame_overlays"
    target_dir.mkdir(parents=True, exist_ok=True)
    for clip_index in group_clips:
        for image in source_frame_dir.glob(f"clip_{clip_index:02d}_frame_*_overlay.png"):
            shutil.copyfile(image, target_dir / image.name)


def make_package(
    processed_dir: Path,
    output_dir: Path,
    case_id: str,
    lca_clips: List[int],
    rca_clips: List[int],
    lca_weak_clips: List[int],
    rca_weak_clips: List[int],
):
    case_dir = processed_dir / case_id
    if not case_dir.exists():
        raise FileNotFoundError(case_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    clips = clip_by_index(read_csv(case_dir / "case_clips_summary.csv"))
    scores = frame_scores_by_clip(case_dir)

    if (case_dir / "clip_contact_sheet.png").exists():
        shutil.copyfile(case_dir / "clip_contact_sheet.png", output_dir / "clip_contact_sheet.png")

    groups = {
        "LCA": {
            "primary_clips": lca_clips,
            "weak_or_optional_clips": lca_weak_clips,
        },
        "RCA": {
            "primary_clips": rca_clips,
            "weak_or_optional_clips": rca_weak_clips,
        },
    }

    all_pair_rows = []
    all_frame_rows = []
    for artery_system, group in groups.items():
        primary_clips = group["primary_clips"]
        if not primary_clips:
            continue
        pair_rows = candidate_pairs_for_group(case_id, artery_system, primary_clips, clips, scores)
        frame_rows = frame_review_rows(case_id, artery_system, primary_clips + group["weak_or_optional_clips"], scores)
        group_dir = output_dir / artery_system.lower()
        write_csv(group_dir / "candidate_multiview_pairs.csv", pair_rows)
        write_csv(group_dir / "candidate_frame_review.csv", frame_rows)
        copy_group_images(case_dir, output_dir, artery_system.lower(), primary_clips + group["weak_or_optional_clips"])
        all_pair_rows.extend(pair_rows)
        all_frame_rows.extend(frame_rows)

    config = {
        "case_id": case_id,
        "processed_case_dir": str(case_dir),
        "doctor_confirmed_groups": groups,
        "next_steps": [
            "Doctor confirms main pair and frame for each artery system.",
            "Generate branch graphs for confirmed frames.",
            "Generate epipolar branch match candidates.",
            "Doctor validates branch correspondences.",
            "Run multi-view reconstruction using validated views and branch matches.",
        ],
    }
    write_json(output_dir / "multiview_case_config.json", config)
    write_csv(output_dir / "all_candidate_multiview_pairs.csv", all_pair_rows)
    write_csv(output_dir / "all_candidate_frame_review.csv", all_frame_rows)

    summary = {
        "case_id": case_id,
        "output_dir": str(output_dir),
        "lca_primary_clips": lca_clips,
        "lca_weak_clips": lca_weak_clips,
        "rca_primary_clips": rca_clips,
        "rca_weak_clips": rca_weak_clips,
        "outputs": {
            "config": str(output_dir / "multiview_case_config.json"),
            "all_candidate_pairs": str(output_dir / "all_candidate_multiview_pairs.csv"),
            "all_candidate_frame_review": str(output_dir / "all_candidate_frame_review.csv"),
            "contact_sheet": str(output_dir / "clip_contact_sheet.png"),
        },
    }
    write_json(output_dir / "multiview_case_package_summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", type=Path, default=ROOT / "dicom_dataset_processed")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "dicom_multiview_review")
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--lca-clips", default="")
    parser.add_argument("--lca-weak-clips", default="")
    parser.add_argument("--rca-clips", default="")
    parser.add_argument("--rca-weak-clips", default="")
    args = parser.parse_args()

    summary = make_package(
        processed_dir=args.processed_dir,
        output_dir=args.output_dir / args.case_id,
        case_id=args.case_id,
        lca_clips=parse_clip_list(args.lca_clips),
        rca_clips=parse_clip_list(args.rca_clips),
        lca_weak_clips=parse_clip_list(args.lca_weak_clips),
        rca_weak_clips=parse_clip_list(args.rca_weak_clips),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
