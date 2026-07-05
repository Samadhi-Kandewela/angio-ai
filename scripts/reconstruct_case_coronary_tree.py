"""
Build separate LCA/RCA hybrid reconstructions and a combined coronary OBJ.

This script uses the first-pass visual clip classification to avoid mixing LCA
and RCA views. For each artery system it:

- chooses candidate clip pairs from the classified clips
- builds branch graphs and epipolar match candidates
- ranks the pair candidates
- reconstructs a full-tree hybrid mesh from the best pair
- exports LCA, RCA, and combined OBJ files

Example:
    python scripts/reconstruct_case_coronary_tree.py ^
        --case-id case_026 ^
        --processed-dir dicom_dataset_processed ^
        --classification-csv dicom_2d_reference_trees/visual_clip_classification_first_pass.csv ^
        --output-dir dicom_case_3d_reconstructions/case_026
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Dict, Iterable, List


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from build_batch_branch_graphs import build_pair_graphs  # noqa: E402
from reconstruct_full_tree_hybrid import reconstruct  # noqa: E402
from rank_3d_reconstruction_candidates import score_pair  # noqa: E402


MATERIAL_COLORS = {
    "reliable": (0.82, 0.82, 0.82),
    "usable": (0.18, 0.42, 1.0),
    "estimated": (1.0, 0.56, 0.06),
    "junction": (0.72, 0.72, 0.72),
}


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


def parse_clip_list(text: str) -> List[int]:
    clips = []
    for part in (text or "").replace(",", ";").split(";"):
        part = part.strip().upper()
        if not part:
            continue
        if part.startswith("C"):
            part = part[1:]
        if part.isdigit():
            clips.append(int(part))
    return clips


def load_classification(path: Path, case_id: str) -> Dict[str, List[int]]:
    for row in read_csv(path):
        if row.get("case_id") == case_id:
            return {
                "lca": parse_clip_list(row.get("likely_lca_clips", "")),
                "rca": parse_clip_list(row.get("likely_rca_clips", "")),
            }
    raise ValueError(f"Case {case_id} was not found in {path}")


def best_frames_by_clip(case_dir: Path) -> Dict[int, List[Dict[str, str]]]:
    by_clip: Dict[int, List[Dict[str, str]]] = {}
    for row in read_csv(case_dir / "sampled_frame_scores.csv"):
        by_clip.setdefault(to_int(row.get("clip_index", "")), []).append(row)
    for rows in by_clip.values():
        rows.sort(key=lambda r: to_float(r.get("segmentation_quality_score", "")), reverse=True)
    return by_clip


def angle_separation(a: Dict[str, str], b: Dict[str, str]) -> float:
    return math.hypot(
        to_float(a.get("primary_angle_deg", "")) - to_float(b.get("primary_angle_deg", "")),
        to_float(a.get("secondary_angle_deg", "")) - to_float(b.get("secondary_angle_deg", "")),
    )


def pair_score(angle_sep: float, qa: float, qb: float) -> float:
    if angle_sep < 15.0:
        return -1.0
    if angle_sep <= 75.0:
        angle_score = 60.0 - abs(angle_sep - 45.0) * 0.7
    else:
        angle_score = max(15.0, 45.0 - (angle_sep - 75.0) * 1.5)
    quality_score = 0.22 * qa + 0.22 * qb
    weak_penalty = max(0.0, 45.0 - min(qa, qb)) * 0.9
    return round(max(0.0, angle_score + quality_score - weak_penalty), 3)


def best_frame_text(frame_rows: List[Dict[str, str]], top_n: int = 3) -> str:
    return "; ".join(
        f"{row.get('frame', '')} (Q{row.get('segmentation_quality_score', '')})"
        for row in frame_rows[:top_n]
    )


def reference_strength(frame_row: Dict[str, str]) -> float:
    """Prefer the view that contains the fuller 2D tree as reconstruction reference."""
    quality = to_float(frame_row.get("segmentation_quality_score", ""))
    branches = to_float(frame_row.get("branch_count", ""))
    length = to_float(frame_row.get("centerline_length_px", ""))
    return quality + branches * 16.0 + min(length / 20.0, 70.0)


def make_candidate_rows(
    case_id: str,
    system_name: str,
    clip_indexes: List[int],
    case_dir: Path,
) -> List[Dict[str, object]]:
    clips = {to_int(row.get("clip_index", "")): row for row in read_csv(case_dir / "case_clips_summary.csv")}
    frame_scores = best_frames_by_clip(case_dir)
    usable = [idx for idx in clip_indexes if idx in clips and frame_scores.get(idx)]
    rows: List[Dict[str, object]] = []
    for pos, a_idx in enumerate(usable):
        for b_idx in usable[pos + 1 :]:
            # The hybrid reconstruction preserves every branch from view A.
            # Therefore view A should be the fuller reference tree, not merely
            # the lower clip index.
            if reference_strength(frame_scores[b_idx][0]) > reference_strength(frame_scores[a_idx][0]):
                a_idx, b_idx = b_idx, a_idx
            a = clips[a_idx]
            b = clips[b_idx]
            sep = angle_separation(a, b)
            qa = to_float(frame_scores[a_idx][0].get("segmentation_quality_score", ""))
            qb = to_float(frame_scores[b_idx][0].get("segmentation_quality_score", ""))
            score = pair_score(sep, qa, qb)
            if score < 0:
                continue
            rows.append(
                {
                    "case_id": case_id,
                    "review_group": system_name,
                    "view_a_clip_index": a_idx,
                    "view_b_clip_index": b_idx,
                    "angle_separation_deg": round(sep, 3),
                    "view_a_primary_angle_deg": a.get("primary_angle_deg", ""),
                    "view_a_secondary_angle_deg": a.get("secondary_angle_deg", ""),
                    "view_b_primary_angle_deg": b.get("primary_angle_deg", ""),
                    "view_b_secondary_angle_deg": b.get("secondary_angle_deg", ""),
                    "view_a_quality": round(qa, 2),
                    "view_b_quality": round(qb, 2),
                    "view_a_best_frames": best_frame_text(frame_scores[a_idx]),
                    "view_b_best_frames": best_frame_text(frame_scores[b_idx]),
                    "auto_pair_score": score,
                    "doctor_same_artery_tree": "",
                    "doctor_use_for_3d": "",
                    "doctor_notes": "",
                }
            )
    return sorted(rows, key=lambda row: float(row["auto_pair_score"]), reverse=True)


def rank_pair_dirs(pair_dirs: Iterable[Path]) -> List[Dict[str, object]]:
    ranked = []
    for pair_dir in pair_dirs:
        summary_path = pair_dir / "branch_pair_summary.json"
        match_path = pair_dir / "branch_match_candidates.csv"
        if not summary_path.exists() or not match_path.exists():
            continue
        with open(summary_path, encoding="utf-8") as f:
            summary = {k: str(v) for k, v in json.load(f).items()}
        matches = read_csv(match_path)
        row = score_pair(summary, matches)
        ranked.append(row)
    return sorted(ranked, key=lambda row: to_float(str(row["reconstruction_readiness_score"])), reverse=True)


def copy_reconstruction_outputs(recon_dir: Path, output_obj: Path):
    source_obj = recon_dir / "full_tree_hybrid_confidence.obj"
    source_mtl = source_obj.with_suffix(".mtl")
    output_obj.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_obj, output_obj)
    if source_mtl.exists():
        shutil.copyfile(source_mtl, output_obj.with_suffix(".mtl"))


def write_combined_obj(output_obj: Path, inputs: List[Dict[str, str]]):
    output_obj.parent.mkdir(parents=True, exist_ok=True)
    with open(output_obj.with_suffix(".mtl"), "w", encoding="utf-8") as f:
        for material, color in MATERIAL_COLORS.items():
            f.write(f"newmtl {material}\nKd {color[0]} {color[1]} {color[2]}\n")

    vertex_offset = 0
    with open(output_obj, "w", encoding="utf-8") as out:
        out.write(f"mtllib {output_obj.with_suffix('.mtl').name}\n")
        out.write("# Combined LCA/RCA coronary tree. Object names are prefixed by artery system.\n")
        for item in inputs:
            artery = item["artery"]
            obj_path = Path(item["obj"])
            vertices_in_object = 0
            current_object = artery
            with open(obj_path, encoding="utf-8") as src:
                for line in src:
                    if line.startswith("mtllib ") or line.startswith("#"):
                        continue
                    if line.startswith("o "):
                        current_object = f"{artery}_{line[2:].strip()}"
                        out.write(f"o {current_object}\n")
                    elif line.startswith("v "):
                        vertices_in_object += 1
                        out.write(line)
                    elif line.startswith("f "):
                        parts = line.strip().split()[1:]
                        shifted = []
                        for part in parts:
                            values = part.split("/")
                            values[0] = str(int(values[0]) + vertex_offset)
                            shifted.append("/".join(values))
                        out.write("f " + " ".join(shifted) + "\n")
                    else:
                        out.write(line)
            vertex_offset += vertices_in_object


def reconstruct_system(
    case_id: str,
    system_name: str,
    clip_indexes: List[int],
    processed_dir: Path,
    output_dir: Path,
    max_pairs: int,
) -> Dict[str, object]:
    case_dir = processed_dir / case_id
    candidates = make_candidate_rows(case_id, system_name, clip_indexes, case_dir)
    system_dir = output_dir / system_name
    write_csv(system_dir / f"{system_name}_candidate_view_pairs.csv", candidates)
    if not candidates:
        return {"system": system_name, "status": "no_candidate_pairs", "clip_indexes": clip_indexes}

    pair_dirs = []
    pair_root = system_dir / "pair_candidates"
    for row in candidates[:max_pairs]:
        try:
            result = build_pair_graphs(case_id, {k: str(v) for k, v in row.items()}, processed_dir, pair_root)
            pair_dirs.append(Path(result["output_dir"]))
        except Exception as exc:
            row["build_error"] = str(exc)

    ranked = rank_pair_dirs(pair_dirs)
    write_csv(system_dir / f"{system_name}_ranked_pair_candidates.csv", ranked)
    if not ranked:
        return {"system": system_name, "status": "no_ranked_pairs", "clip_indexes": clip_indexes}

    best = ranked[0]
    best_pair_dir = ROOT / str(best["pair_output_dir"])
    recon_dir = system_dir / "reconstruction"
    recon_summary = reconstruct(best_pair_dir, recon_dir)
    final_obj = output_dir / f"{case_id}_{system_name}_tree.obj"
    copy_reconstruction_outputs(recon_dir, final_obj)
    return {
        "system": system_name,
        "status": "reconstructed",
        "clip_indexes": clip_indexes,
        "best_pair": best,
        "reconstruction": recon_summary,
        "outputs": {
            "obj": str(final_obj),
            "mtl": str(final_obj.with_suffix(".mtl")),
            "candidate_pairs": str(system_dir / f"{system_name}_candidate_view_pairs.csv"),
            "ranked_pairs": str(system_dir / f"{system_name}_ranked_pair_candidates.csv"),
            "reconstruction_dir": str(recon_dir),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--processed-dir", type=Path, default=ROOT / "dicom_dataset_processed")
    parser.add_argument(
        "--classification-csv",
        type=Path,
        default=ROOT / "dicom_2d_reference_trees" / "visual_clip_classification_first_pass.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "dicom_case_3d_reconstructions")
    parser.add_argument("--max-pairs-per-system", type=int, default=5)
    args = parser.parse_args()

    output_dir = args.output_dir / args.case_id if args.output_dir.name != args.case_id else args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    groups = load_classification(args.classification_csv, args.case_id)

    summaries = []
    for system_name, clips in (("lca", groups["lca"]), ("rca", groups["rca"])):
        if len(clips) < 2:
            summaries.append({"system": system_name, "status": "not_enough_clips", "clip_indexes": clips})
            continue
        summaries.append(
            reconstruct_system(
                args.case_id,
                system_name,
                clips,
                args.processed_dir,
                output_dir,
                max(1, args.max_pairs_per_system),
            )
        )

    combined_inputs = []
    for summary in summaries:
        if summary.get("status") == "reconstructed":
            combined_inputs.append({"artery": str(summary["system"]).upper(), "obj": summary["outputs"]["obj"]})
    combined_obj = output_dir / f"{args.case_id}_combined_coronary_tree.obj"
    if combined_inputs:
        write_combined_obj(combined_obj, combined_inputs)

    final_summary = {
        "case_id": args.case_id,
        "classification": groups,
        "output_dir": str(output_dir),
        "systems": summaries,
        "combined_model": str(combined_obj) if combined_inputs else "",
        "color_meaning": {
            "reliable": "gray/white, strict two-view validated",
            "usable": "blue, two-view supported",
            "estimated": "orange, preserved reference-view branch with interpolated depth",
            "junction": "light gray blended junction helper",
        },
    }
    write_json(output_dir / "case_coronary_reconstruction_summary.json", final_summary)
    print(json.dumps(final_summary, indent=2))


if __name__ == "__main__":
    main()
