"""
Build a compact batch review queue for DICOM 3D reconstruction.

This script sits between raw dataset preparation and full 3D reconstruction. It
does not modify raw DICOM files and does not create a final clinical model. It
summarizes each processed case so we can make progress on many cases first, then
ask the doctor to validate only the high-value case-level decisions.

Example:
    python scripts/build_batch_3d_review_queue.py --processed-dir dicom_dataset_processed --output-dir dicom_batch_3d_review
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


ROOT = Path(__file__).resolve().parents[1]


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: Sequence[str] | None = None):
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


def angle_separation(a: Dict[str, str], b: Dict[str, str]) -> float:
    return math.hypot(
        to_float(a.get("primary_angle_deg", "")) - to_float(b.get("primary_angle_deg", "")),
        to_float(a.get("secondary_angle_deg", "")) - to_float(b.get("secondary_angle_deg", "")),
    )


def frame_scores_by_clip(case_dir: Path) -> Dict[int, List[Dict[str, str]]]:
    rows = read_csv(case_dir / "sampled_frame_scores.csv")
    by_clip: Dict[int, List[Dict[str, str]]] = {}
    for row in rows:
        by_clip.setdefault(to_int(row.get("clip_index", "")), []).append(row)
    for clip_rows in by_clip.values():
        clip_rows.sort(key=lambda r: to_float(r.get("segmentation_quality_score", "")), reverse=True)
    return by_clip


def best_quality(scores: Dict[int, List[Dict[str, str]]], clip_index: int) -> float:
    rows = scores.get(clip_index, [])
    if not rows:
        return 0.0
    return max(to_float(row.get("segmentation_quality_score", "")) for row in rows)


def best_frames_text(scores: Dict[int, List[Dict[str, str]]], clip_index: int, top_n: int = 3) -> str:
    rows = scores.get(clip_index, [])[:top_n]
    if not rows:
        return ""
    return "; ".join(f"{row.get('frame', '')} (Q{row.get('segmentation_quality_score', '')})" for row in rows)


def pair_score(pair: Dict[str, object]) -> float:
    sep = float(pair["angle_separation_deg"])
    # For sparse angiogram reconstruction, very small separations give poor
    # depth and extremely wide separations often produce mismatched trees.
    if sep < 15:
        sep_score = sep / 15.0 * 20.0
    elif sep <= 75:
        sep_score = 45.0 + (1.0 - abs(sep - 45.0) / 30.0) * 25.0
    else:
        sep_score = max(15.0, 70.0 - (sep - 75.0) * 1.5)

    qa = float(pair.get("view_a_quality", 0.0))
    qb = float(pair.get("view_b_quality", 0.0))
    if qa > 0.0 and qb > 0.0:
        quality_score = 0.25 * qa + 0.25 * qb
        weak_clip_penalty = max(0.0, 60.0 - min(qa, qb)) * 1.2
    else:
        quality_score = 0.0
        weak_clip_penalty = 0.0
    return round(max(0.0, sep_score + quality_score - weak_clip_penalty), 3)


def make_pair_rows(
    case_id: str,
    clips: List[Dict[str, str]],
    scores: Dict[int, List[Dict[str, str]]],
    review_group: str = "all_clips",
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for i, a in enumerate(clips):
        for b in clips[i + 1:]:
            if a.get("primary_angle_deg", "") == "" or b.get("primary_angle_deg", "") == "":
                continue
            clip_a = to_int(a.get("clip_index", ""))
            clip_b = to_int(b.get("clip_index", ""))
            sep = angle_separation(a, b)
            if sep < 15:
                continue
            row: Dict[str, object] = {
                "case_id": case_id,
                "review_group": review_group,
                "view_a_clip_index": clip_a,
                "view_b_clip_index": clip_b,
                "angle_separation_deg": round(sep, 3),
                "view_a_primary_angle_deg": a.get("primary_angle_deg", ""),
                "view_a_secondary_angle_deg": a.get("secondary_angle_deg", ""),
                "view_b_primary_angle_deg": b.get("primary_angle_deg", ""),
                "view_b_secondary_angle_deg": b.get("secondary_angle_deg", ""),
                "view_a_quality": round(best_quality(scores, clip_a), 2),
                "view_b_quality": round(best_quality(scores, clip_b), 2),
                "view_a_best_frames": best_frames_text(scores, clip_a),
                "view_b_best_frames": best_frames_text(scores, clip_b),
                "auto_pair_score": 0.0,
                "doctor_same_artery_tree": "",
                "doctor_use_for_3d": "",
                "doctor_notes": "",
            }
            row["auto_pair_score"] = pair_score(row)
            rows.append(row)
    return sorted(rows, key=lambda row: float(row["auto_pair_score"]), reverse=True)


def default_group_suggestion(clips: List[Dict[str, str]]) -> Dict[str, List[int]]:
    """Suggest broad review groups without claiming LCA/RCA certainty.

    Many cath lab clips are acquired in blocks, so this is intentionally a
    review shortcut, not a clinical classifier.
    """
    indexes = [to_int(row.get("clip_index", "")) for row in clips]
    if len(indexes) <= 4:
        return {"group_a_primary": indexes, "group_b_primary": []}
    if len(indexes) <= 8:
        split = max(2, min(len(indexes) - 2, int(round(len(indexes) * 0.70))))
    else:
        split = max(3, min(len(indexes) - 3, int(round(len(indexes) * 0.60))))
    return {
        "group_a_primary": indexes[:split],
        "group_b_primary": indexes[split:],
    }


def clip_review_rows(case_id: str, clips: List[Dict[str, str]], scores: Dict[int, List[Dict[str, str]]]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for clip in clips:
        clip_index = to_int(clip.get("clip_index", ""))
        rows.append(
            {
                "case_id": case_id,
                "clip_index": clip_index,
                "frames": clip.get("frames", ""),
                "primary_angle_deg": clip.get("primary_angle_deg", ""),
                "secondary_angle_deg": clip.get("secondary_angle_deg", ""),
                "best_segmentation_quality": round(best_quality(scores, clip_index), 2),
                "best_sampled_frames": best_frames_text(scores, clip_index),
                "auto_review_group": "",
                "doctor_artery_system_lca_rca_uncertain": "",
                "doctor_useful_for_3d_yes_no": "",
                "doctor_notes": "",
            }
        )
    return rows


def copy_if_exists(source: Path, target: Path):
    if source.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)


def build_case_review(case_dir: Path, output_case_dir: Path) -> Dict[str, object]:
    case_id = case_dir.name
    clips = read_csv(case_dir / "case_clips_summary.csv")
    scores = frame_scores_by_clip(case_dir)
    output_case_dir.mkdir(parents=True, exist_ok=True)

    copy_if_exists(case_dir / "clip_contact_sheet.png", output_case_dir / "clip_contact_sheet.png")
    copy_if_exists(case_dir / "case_clips_summary.csv", output_case_dir / "case_clips_summary.csv")

    groups = default_group_suggestion(clips)
    clips_by_index = {to_int(row.get("clip_index", "")): row for row in clips}
    group_a_clips = [clips_by_index[idx] for idx in groups["group_a_primary"] if idx in clips_by_index]
    group_b_clips = [clips_by_index[idx] for idx in groups["group_b_primary"] if idx in clips_by_index]

    all_pairs = make_pair_rows(case_id, clips, scores, "all_clips")
    group_a_pairs = make_pair_rows(case_id, group_a_clips, scores, "group_a_sequence_block")
    group_b_pairs = make_pair_rows(case_id, group_b_clips, scores, "group_b_sequence_block")

    write_csv(output_case_dir / "all_candidate_view_pairs_for_audit.csv", all_pairs[:30])
    write_csv(output_case_dir / "group_a_candidate_view_pairs.csv", group_a_pairs[:20])
    write_csv(output_case_dir / "group_b_candidate_view_pairs.csv", group_b_pairs[:20])
    write_csv(output_case_dir / "top_candidate_view_pairs.csv", (group_a_pairs[:10] + group_b_pairs[:10]))

    clip_rows = clip_review_rows(case_id, clips, scores)
    group_a = set(groups["group_a_primary"])
    group_b = set(groups["group_b_primary"])
    for row in clip_rows:
        idx = int(row["clip_index"])
        if idx in group_a:
            row["auto_review_group"] = "group_a_sequence_block"
        elif idx in group_b:
            row["auto_review_group"] = "group_b_sequence_block"
    write_csv(output_case_dir / "clip_level_doctor_summary.csv", clip_rows)

    summary = {
        "case_id": case_id,
        "num_clips": len(clips),
        "has_segmentation_scores": bool(scores),
        "suggested_group_a_primary": groups["group_a_primary"],
        "suggested_group_b_primary": groups["group_b_primary"],
        "top_group_a_pair": group_a_pairs[0] if group_a_pairs else None,
        "top_group_b_pair": group_b_pairs[0] if group_b_pairs else None,
        "top_global_pair_for_audit": all_pairs[0] if all_pairs else None,
        "outputs": {
            "contact_sheet": str(output_case_dir / "clip_contact_sheet.png"),
            "top_candidate_view_pairs": str(output_case_dir / "top_candidate_view_pairs.csv"),
            "group_a_candidate_view_pairs": str(output_case_dir / "group_a_candidate_view_pairs.csv"),
            "group_b_candidate_view_pairs": str(output_case_dir / "group_b_candidate_view_pairs.csv"),
            "all_candidate_view_pairs_for_audit": str(output_case_dir / "all_candidate_view_pairs_for_audit.csv"),
            "clip_level_doctor_summary": str(output_case_dir / "clip_level_doctor_summary.csv"),
        },
    }
    write_json(output_case_dir / "case_review_summary.json", summary)
    return summary


def build_batch_review(processed_dir: Path, output_dir: Path) -> Dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    case_summaries = []
    index_rows: List[Dict[str, object]] = []
    for case_dir in sorted(processed_dir.glob("case_*")):
        if not case_dir.is_dir():
            continue
        case_output = output_dir / case_dir.name
        summary = build_case_review(case_dir, case_output)
        case_summaries.append(summary)
        top_a = summary.get("top_group_a_pair") or {}
        top_b = summary.get("top_group_b_pair") or {}
        audit_pair = summary.get("top_global_pair_for_audit") or {}
        index_rows.append(
            {
                "case_id": summary["case_id"],
                "num_clips": summary["num_clips"],
                "has_segmentation_scores": summary["has_segmentation_scores"],
                "suggested_group_a_primary": ";".join(map(str, summary["suggested_group_a_primary"])),
                "suggested_group_b_primary": ";".join(map(str, summary["suggested_group_b_primary"])),
                "top_group_a_pair": format_pair(top_a),
                "top_group_b_pair": format_pair(top_b),
                "top_global_pair_for_audit": format_pair(audit_pair),
                "contact_sheet": summary["outputs"]["contact_sheet"],
                "doctor_ready_for_reconstruction_yes_no": "",
                "doctor_notes": "",
            }
        )

    write_csv(output_dir / "batch_case_review_index.csv", index_rows)
    payload = {
        "processed_dir": str(processed_dir),
        "output_dir": str(output_dir),
        "num_cases": len(case_summaries),
        "purpose": "Case-level doctor review queue before automatic multi-view 3D reconstruction.",
        "case_summaries": case_summaries,
        "outputs": {
            "batch_case_review_index": str(output_dir / "batch_case_review_index.csv"),
        },
    }
    write_json(output_dir / "batch_3d_review_summary.json", payload)
    return payload


def format_pair(pair: Dict[str, object]) -> str:
    if not pair:
        return ""
    return (
        f"C{pair.get('view_a_clip_index', '')}-C{pair.get('view_b_clip_index', '')} "
        f"sep={pair.get('angle_separation_deg', '')} score={pair.get('auto_pair_score', '')}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", type=Path, default=ROOT / "dicom_dataset_processed")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "dicom_batch_3d_review")
    args = parser.parse_args()

    summary = build_batch_review(args.processed_dir, args.output_dir)
    print(json.dumps({k: v for k, v in summary.items() if k != "case_summaries"}, indent=2))


if __name__ == "__main__":
    main()
