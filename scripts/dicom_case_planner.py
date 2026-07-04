"""
Case-level DICOM planner for doctor-facing coronary 3D reconstruction.

This script does not build the final 3D mesh. It creates the safety layer that
should run before reconstruction:

1. Scan a DICOM folder for XA cine clips.
2. Score candidate frames using the vessel segmentation model.
3. Classify clips conservatively as left/right/uncertain when metadata allows.
4. Rank feasible view/frame pairs for each artery group.
5. Write a reconstruction plan that can be reviewed before mesh generation.

Example:
    python scripts/dicom_case_planner.py --dicom-dir dicom_files --output-dir dicom_case_plan
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import cv2
import numpy as np
import onnxruntime as ort
import pydicom
from scipy.optimize import linear_sum_assignment


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from dicom_3d_pipeline import (  # noqa: E402
    Clip,
    extract_graph,
    score_pair,
    segment_frame,
    view_from_clip,
)


DEFAULT_MODEL_CANDIDATES = [
    ROOT / "checkpoints" / "mobileunetv3" / "mobileunetv3_augmented_best.onnx",
    ROOT / "checkpoints" / "mobileunetv3_augmented.onnx",
    ROOT / "checkpoints" / "mobileunetv3_best.onnx",
    ROOT / "checkpoints" / "model.onnx",
]


@dataclass
class FrameCandidate:
    clip: Clip
    frame: int
    gray: np.ndarray
    overlay: Optional[np.ndarray]
    clean_mask: Optional[np.ndarray]
    skeleton: Optional[np.ndarray]
    branches: List[Dict[str, object]]
    area_pct: float
    total_length_px: float
    quality_score: float


@dataclass
class CaseGroup:
    case_key: str
    root_hint: str
    clips: List[Clip]


def existing_default_model() -> Optional[Path]:
    for path in DEFAULT_MODEL_CANDIDATES:
        if path.exists():
            return path
    return None


def load_dicom_header(path: Path):
    return pydicom.dcmread(str(path), stop_before_pixels=True, force=True)


def iter_probable_dicom_files(dicom_dir: Path) -> Iterable[Path]:
    skip_suffixes = {
        ".dll",
        ".exe",
        ".bin",
        ".inf",
        ".txt",
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".xml",
        ".ini",
        ".ico",
    }
    for path in dicom_dir.rglob("*"):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix in skip_suffixes:
            continue
        if suffix == ".dcm" or path.name.upper().startswith("IM") or path.name.lower() == "dicomdir":
            yield path


def find_xa_clips_fast(dicom_dir: Path) -> List[Clip]:
    clips: List[Clip] = []
    for path in iter_probable_dicom_files(dicom_dir):
        try:
            ds = load_dicom_header(path)
        except Exception:
            continue
        if getattr(ds, "Modality", "") != "XA":
            continue
        required = (
            "NumberOfFrames",
            "Rows",
            "Columns",
            "PositionerPrimaryAngle",
            "PositionerSecondaryAngle",
            "DistanceSourceToDetector",
            "DistanceSourceToPatient",
            "ImagerPixelSpacing",
        )
        if any(not hasattr(ds, name) for name in required):
            continue
        try:
            spacing = tuple(float(v) for v in ds.ImagerPixelSpacing)
            clips.append(
                Clip(
                    path=path,
                    index=len(clips),
                    frames=int(ds.NumberOfFrames),
                    rows=int(ds.Rows),
                    cols=int(ds.Columns),
                    primary=float(ds.PositionerPrimaryAngle),
                    secondary=float(ds.PositionerSecondaryAngle),
                    sdd=float(ds.DistanceSourceToDetector),
                    spd=float(ds.DistanceSourceToPatient),
                    spacing=(spacing[0], spacing[1]),
                    acquisition_time=str(getattr(ds, "AcquisitionTime", "")),
                )
            )
        except Exception:
            continue
    return sorted(clips, key=lambda clip: clip.acquisition_time)


def safe_float(value, default: Optional[float] = None) -> Optional[float]:
    try:
        return float(value)
    except Exception:
        return default


def metadata_text(ds) -> str:
    fields = [
        "SeriesDescription",
        "ProtocolName",
        "ViewCodeSequence",
        "PerformedProtocolCodeSequence",
        "ImageComments",
        "Laterality",
        "BodyPartExamined",
    ]
    chunks = []
    for name in fields:
        value = getattr(ds, name, "")
        if value:
            chunks.append(str(value))
    return " ".join(chunks).upper()


def artery_hint_for_clip(clip: Clip) -> Dict[str, str]:
    try:
        ds = load_dicom_header(clip.path)
        text = metadata_text(ds)
    except Exception:
        text = ""

    left_tokens = (" LCA", "LEFT CORONARY", "LEFT MAIN", " LAD", " LCX", "LEFT ")
    right_tokens = (" RCA", "RIGHT CORONARY", "RIGHT ")
    has_left = any(token in f" {text} " for token in left_tokens)
    has_right = any(token in f" {text} " for token in right_tokens)
    if has_left and not has_right:
        return {"artery_group": "left_coronary", "source": "dicom_text"}
    if has_right and not has_left:
        return {"artery_group": "right_coronary", "source": "dicom_text"}
    if has_left and has_right:
        return {"artery_group": "both_or_uncertain", "source": "dicom_text"}
    return {"artery_group": "uncertain_coronary", "source": "not_enough_metadata"}


def case_key_for_clip(clip: Clip, dicom_dir: Path) -> tuple[str, str]:
    try:
        ds = load_dicom_header(clip.path)
        study_uid = str(getattr(ds, "StudyInstanceUID", "")).strip()
        series_uid = str(getattr(ds, "SeriesInstanceUID", "")).strip()
        if study_uid:
            return study_uid, root_hint_for_path(clip.path, dicom_dir)
        if series_uid:
            return f"series:{series_uid}", root_hint_for_path(clip.path, dicom_dir)
    except Exception:
        pass
    return root_hint_for_path(clip.path, dicom_dir), root_hint_for_path(clip.path, dicom_dir)


def root_hint_for_path(path: Path, dicom_dir: Path) -> str:
    try:
        rel = path.relative_to(dicom_dir)
    except ValueError:
        return str(path.parent)
    return rel.parts[0] if len(rel.parts) > 1 else str(path.parent)


def group_clips_by_case(clips: List[Clip], dicom_dir: Path) -> List[CaseGroup]:
    grouped: Dict[str, CaseGroup] = {}
    for clip in clips:
        key, root_hint = case_key_for_clip(clip, dicom_dir)
        if key not in grouped:
            grouped[key] = CaseGroup(case_key=key, root_hint=root_hint, clips=[])
        grouped[key].clips.append(clip)
    cases = sorted(grouped.values(), key=lambda item: item.root_hint)
    for case_index, case in enumerate(cases):
        for clip_index, clip in enumerate(sorted(case.clips, key=lambda c: c.acquisition_time)):
            clip.index = clip_index
        case.clips = sorted(case.clips, key=lambda c: c.index)
    return cases


def case_rows(cases: List[CaseGroup]) -> List[Dict[str, object]]:
    rows = []
    for idx, case in enumerate(cases):
        angles = sorted({f"{clip.primary:.1f}/{clip.secondary:.1f}" for clip in case.clips})
        rows.append(
            {
                "case_index": idx,
                "case_key": case.case_key,
                "root_hint": case.root_hint,
                "xa_clip_count": len(case.clips),
                "total_frames": sum(clip.frames for clip in case.clips),
                "angle_pairs": " | ".join(angles[:18]),
            }
        )
    return rows


def sample_frame_numbers(frame_count: int, max_frames: int) -> List[int]:
    if frame_count <= max_frames:
        return list(range(1, frame_count + 1))
    fractions = np.linspace(0.15, 0.92, max_frames)
    return sorted({int(np.clip(round(frame_count * frac), 1, frame_count)) for frac in fractions})


def load_clip_pixel_array(clip: Clip) -> np.ndarray:
    ds = pydicom.dcmread(str(clip.path), force=True)
    arr = ds.pixel_array
    if arr.ndim == 2:
        arr = arr[None, :, :]
    return arr


def normalize_frame(frame: np.ndarray) -> np.ndarray:
    if frame.dtype == np.uint8:
        out = frame
    else:
        f = frame.astype(np.float32)
        lo, hi = np.percentile(f, [1, 99])
        out = np.clip((f - lo) / max(hi - lo, 1e-6) * 255.0, 0, 255).astype(np.uint8)
    return cv2.resize(out, (512, 512), interpolation=cv2.INTER_AREA)


def score_frame(clean_mask: np.ndarray, branches: List[Dict[str, object]]) -> tuple[float, float, float]:
    area_pct = float(np.count_nonzero(clean_mask) / clean_mask.size * 100.0)
    total_length = float(sum(float(branch["length_px"]) for branch in branches))
    branch_count = len(branches)

    area_score = min(area_pct / 8.0, 1.0)
    length_score = min(total_length / 1300.0, 1.0)
    branch_score = min(branch_count / 10.0, 1.0)
    too_big_penalty = max(0.0, (area_pct - 18.0) / 18.0)
    too_small_penalty = max(0.0, (0.35 - area_pct) / 0.35)
    quality = 100.0 * (0.35 * area_score + 0.45 * length_score + 0.20 * branch_score)
    quality -= 35.0 * too_big_penalty + 50.0 * too_small_penalty
    return area_pct, total_length, float(max(0.0, quality))


def build_frame_candidates(
    clips: List[Clip],
    session,
    input_name: str,
    threshold: float,
    max_frames_per_clip: int,
) -> List[FrameCandidate]:
    candidates: List[FrameCandidate] = []
    for clip in clips:
        pixel_array = load_clip_pixel_array(clip)
        for frame in sample_frame_numbers(clip.frames, max_frames_per_clip):
            gray = normalize_frame(pixel_array[frame - 1])
            mask, overlay = segment_frame(session, input_name, gray, threshold)
            clean_mask, skeleton, _, branches = extract_graph(mask)
            area_pct, total_length, quality = score_frame(clean_mask, branches)
            candidates.append(
                FrameCandidate(
                    clip=clip,
                    frame=frame,
                    gray=gray,
                    overlay=overlay,
                    clean_mask=clean_mask,
                    skeleton=skeleton,
                    branches=branches,
                    area_pct=area_pct,
                    total_length_px=total_length,
                    quality_score=quality,
                )
            )
    return candidates


def keep_best_frames_per_clip(candidates: List[FrameCandidate], top_k: int) -> List[FrameCandidate]:
    by_clip: Dict[int, List[FrameCandidate]] = {}
    for candidate in candidates:
        by_clip.setdefault(candidate.clip.index, []).append(candidate)
    kept = []
    for frames in by_clip.values():
        kept.extend(sorted(frames, key=lambda item: item.quality_score, reverse=True)[:top_k])
    return sorted(kept, key=lambda item: (item.clip.index, item.frame))


def angle_separation(a: Clip, b: Clip) -> float:
    return float(math.hypot(a.primary - b.primary, a.secondary - b.secondary))


def angle_score(separation: float) -> float:
    if separation < 18.0:
        return 0.0
    if separation <= 55.0:
        return separation / 55.0
    if separation <= 90.0:
        return 1.0 - (separation - 55.0) / 70.0
    return 0.45


def branch_pair_objective(score: Dict[str, object], length_ratio: float) -> float:
    return float(score["median"] + 0.18 * score["p90"] + 2.0 * abs(math.log(max(length_ratio, 1e-6))))


def pair_epipolar_score(a: FrameCandidate, b: FrameCandidate) -> Optional[Dict[str, object]]:
    if len(a.branches) < 2 or len(b.branches) < 2:
        return None
    view_a = view_from_clip(a.clip, "view_a")
    view_b = view_from_clip(b.clip, "view_b")
    top_a = sorted(a.branches, key=lambda branch: float(branch["length_px"]), reverse=True)[:5]
    top_b = sorted(b.branches, key=lambda branch: float(branch["length_px"]), reverse=True)[:5]
    cost = np.full((len(top_a), len(top_b)), 1e6, dtype=np.float64)
    score_grid: List[List[Optional[Dict[str, object]]]] = [[None for _ in top_b] for _ in top_a]

    for i, branch_a in enumerate(top_a):
        for j, branch_b in enumerate(top_b):
            best = None
            for reverse_a in (False, True):
                for reverse_b in (False, True):
                    score = score_pair(branch_a, branch_b, view_a, view_b, reverse_a, reverse_b, samples=22)
                    ratio = max(float(branch_a["length_px"]), 1.0) / max(float(branch_b["length_px"]), 1.0)
                    objective = branch_pair_objective(score, ratio)
                    if best is None or objective < best["objective"]:
                        best = {
                            "objective": objective,
                            "median": float(score["median"]),
                            "p90": float(score["p90"]),
                            "branch_a": int(branch_a["branch_id"]),
                            "branch_b": int(branch_b["branch_id"]),
                        }
            if best is not None:
                cost[i, j] = best["objective"]
                score_grid[i][j] = best

    rows, cols = linear_sum_assignment(cost)
    matches = []
    reliable = 0
    usable = 0
    for i, j in zip(rows, cols):
        best = score_grid[i][j]
        if best is None or best["objective"] >= 1e5:
            continue
        matches.append(best)
        if best["median"] <= 3.0 and best["p90"] <= 7.0:
            reliable += 1
        elif best["median"] <= 5.5 and best["p90"] <= 12.0:
            usable += 1

    if not matches:
        return None
    return {
        "median_epipolar_residual_mm": float(np.median([item["median"] for item in matches])),
        "p90_epipolar_residual_mm": float(np.median([item["p90"] for item in matches])),
        "pair_objective": float(np.median([item["objective"] for item in matches])),
        "estimated_reliable_matches": reliable,
        "estimated_usable_matches": usable,
        "top_branch_matches": matches[:5],
    }


def score_candidate_pair(
    a: FrameCandidate,
    b: FrameCandidate,
    min_angle_deg: float,
) -> Optional[Dict[str, object]]:
    if a.clip.index == b.clip.index:
        return None
    sep = angle_separation(a.clip, b.clip)
    if sep < min_angle_deg:
        return None
    epi = pair_epipolar_score(a, b)
    if epi is None:
        return None

    quality = (a.quality_score + b.quality_score) * 0.5
    branch_ratio = min(len(a.branches), len(b.branches)) / max(len(a.branches), len(b.branches), 1)
    score = (
        30.0 * angle_score(sep)
        + 0.24 * quality
        + 12.0 * epi["estimated_reliable_matches"]
        + 5.5 * epi["estimated_usable_matches"]
        + 26.0 * branch_ratio
        - 1.8 * epi["pair_objective"]
        - 0.12 * epi["p90_epipolar_residual_mm"]
    )
    feasibility = "candidate_for_3d_trial"
    reasons = []
    if epi["estimated_reliable_matches"] < 4 or (epi["estimated_reliable_matches"] + epi["estimated_usable_matches"]) < 5:
        feasibility = "review_required"
        reasons.append("not enough strong two-view branch matches for near-clinical confidence")
    if epi["median_epipolar_residual_mm"] > 3.5 or epi["p90_epipolar_residual_mm"] > 8.0:
        feasibility = "review_required"
        reasons.append("expected epipolar residual is too high for automatic acceptance")
    if min(len(a.branches), len(b.branches)) < 6:
        feasibility = "review_required"
        reasons.append("one view has limited visible branch coverage")

    return {
        "view_a_clip_index": a.clip.index,
        "view_a_frame": a.frame,
        "view_b_clip_index": b.clip.index,
        "view_b_frame": b.frame,
        "view_a_primary_angle_deg": a.clip.primary,
        "view_a_secondary_angle_deg": a.clip.secondary,
        "view_b_primary_angle_deg": b.clip.primary,
        "view_b_secondary_angle_deg": b.clip.secondary,
        "angle_separation_deg": sep,
        "score": float(score),
        "feasibility": feasibility,
        "review_reasons": reasons,
        "view_a_quality": a.quality_score,
        "view_b_quality": b.quality_score,
        "view_a_branches": len(a.branches),
        "view_b_branches": len(b.branches),
        "view_a_area_pct": a.area_pct,
        "view_b_area_pct": b.area_pct,
        **epi,
    }


def rank_pairs_for_group(
    candidates: List[FrameCandidate],
    min_angle_deg: float,
) -> List[Dict[str, object]]:
    rows = []
    for i, a in enumerate(candidates):
        for b in candidates[i + 1:]:
            pair = score_candidate_pair(a, b, min_angle_deg)
            if pair is not None:
                rows.append(pair)
    return sorted(rows, key=lambda item: item["score"], reverse=True)


def clip_rows(clips: List[Clip]) -> List[Dict[str, object]]:
    rows = []
    for clip in clips:
        hint = artery_hint_for_clip(clip)
        rows.append(
            {
                "clip_index": clip.index,
                "dicom_path": str(clip.path),
                "frames": clip.frames,
                "rows": clip.rows,
                "cols": clip.cols,
                "primary_angle_deg": clip.primary,
                "secondary_angle_deg": clip.secondary,
                "source_detector_distance_mm": clip.sdd,
                "source_patient_distance_mm": clip.spd,
                "imager_pixel_spacing_y_mm": clip.spacing[0],
                "imager_pixel_spacing_x_mm": clip.spacing[1],
                "acquisition_time": clip.acquisition_time,
                "artery_group_hint": hint["artery_group"],
                "artery_hint_source": hint["source"],
            }
        )
    return rows


def frame_rows(candidates: List[FrameCandidate]) -> List[Dict[str, object]]:
    rows = []
    for item in candidates:
        hint = artery_hint_for_clip(item.clip)
        rows.append(
            {
                "clip_index": item.clip.index,
                "frame": item.frame,
                "artery_group_hint": hint["artery_group"],
                "quality_score": item.quality_score,
                "branches": len(item.branches),
                "area_pct": item.area_pct,
                "total_length_px": item.total_length_px,
                "primary_angle_deg": item.clip.primary,
                "secondary_angle_deg": item.clip.secondary,
                "dicom_path": str(item.clip.path),
            }
        )
    return rows


def write_csv(path: Path, rows: List[Dict[str, object]]):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_preview_grid(candidates: List[FrameCandidate], output_path: Path, max_items: int = 20):
    selected = sorted(candidates, key=lambda item: item.quality_score, reverse=True)[:max_items]
    thumbs = []
    for item in selected:
        image = item.overlay if item.overlay is not None else cv2.cvtColor(item.gray, cv2.COLOR_GRAY2BGR)
        preview = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
        label = f"Clip {item.clip.index} F{item.frame} Q{item.quality_score:.0f}"
        cv2.putText(preview, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        thumbs.append(preview)
    if not thumbs:
        return
    cols = min(4, len(thumbs))
    rows = int(math.ceil(len(thumbs) / cols))
    blank = np.zeros_like(thumbs[0])
    while len(thumbs) < rows * cols:
        thumbs.append(blank.copy())
    grid_rows = [np.hstack(thumbs[row * cols: (row + 1) * cols]) for row in range(rows)]
    cv2.imwrite(str(output_path), np.vstack(grid_rows))


def group_candidates(candidates: List[FrameCandidate]) -> Dict[str, List[FrameCandidate]]:
    grouped: Dict[str, List[FrameCandidate]] = {
        "left_coronary": [],
        "right_coronary": [],
        "uncertain_coronary": [],
    }
    for item in candidates:
        group = artery_hint_for_clip(item.clip)["artery_group"]
        if group in {"left_coronary", "right_coronary"}:
            grouped[group].append(item)
        else:
            grouped["uncertain_coronary"].append(item)
    return grouped


def build_reconstruction_plan(
    clips: List[Clip],
    candidates: List[FrameCandidate],
    min_angle_deg: float,
) -> Dict[str, object]:
    grouped = group_candidates(candidates)
    artery_plans = {}
    for group_name, group_candidates_ in grouped.items():
        if len({item.clip.index for item in group_candidates_}) < 2:
            artery_plans[group_name] = {
                "status": "not_reconstructable",
                "reason": "fewer than two usable clips in this artery group",
                "candidate_clips": sorted({item.clip.index for item in group_candidates_}),
                "ranked_pairs": [],
            }
            continue
        ranked = rank_pairs_for_group(group_candidates_, min_angle_deg)
        if not ranked:
            artery_plans[group_name] = {
                "status": "not_reconstructable",
                "reason": "no candidate pair passed angle and epipolar checks",
                "candidate_clips": sorted({item.clip.index for item in group_candidates_}),
                "ranked_pairs": [],
            }
            continue
        best = ranked[0]
        status = "candidate_for_3d_trial" if best["feasibility"] == "candidate_for_3d_trial" else "review_before_reconstruction"
        artery_plans[group_name] = {
            "status": status,
            "reason": "best pair selected; final reprojection validation is still mandatory" if status == "candidate_for_3d_trial" else "; ".join(best["review_reasons"]),
            "candidate_clips": sorted({item.clip.index for item in group_candidates_}),
            "recommended_pair": best,
            "ranked_pairs": ranked[:10],
        }

    found_labeled = any(
        artery_plans[name]["candidate_clips"]
        for name in ("left_coronary", "right_coronary")
    )
    doctor_message = (
        "No DICOM metadata clearly labels left/right coronary views. Treat the uncertain group as a review queue before clinical display."
        if not found_labeled
        else "Left/right grouping is based on available DICOM metadata and should be reviewed."
    )
    return {
        "dicom_dir": "",
        "num_xa_clips_found": len(clips),
        "clinical_safety_note": (
            "This is a pre-reconstruction feasibility plan. A final artery tree should be shown as validated only after reprojection validation passes."
        ),
        "doctor_message": doctor_message,
        "artery_plans": artery_plans,
    }


def make_pipeline_command(pair: Dict[str, object], dicom_dir: Path, output_dir: Path, model: Path, threshold: float) -> str:
    return (
        "python scripts\\dicom_3d_pipeline.py "
        f"--dicom-dir {dicom_dir} "
        f"--output-dir {output_dir} "
        f"--model {model} "
        f"--threshold {threshold} "
        f"--view-a-index {pair['view_a_clip_index']} --frame-a {pair['view_a_frame']} "
        f"--view-b-index {pair['view_b_clip_index']} --frame-b {pair['view_b_frame']}"
    )


def add_commands_to_plan(plan: Dict[str, object], dicom_dir: Path, output_dir: Path, model: Path, threshold: float):
    for group_name, group_plan in plan["artery_plans"].items():
        pair = group_plan.get("recommended_pair")
        if not pair:
            continue
        group_out = output_dir / group_name
        group_plan["next_pipeline_command"] = make_pipeline_command(pair, dicom_dir, group_out, model, threshold)


def selected_case_dicom_dir(root: Path, case: CaseGroup) -> Path:
    candidate = root / case.root_hint
    return candidate if candidate.exists() else root


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dicom-dir", type=Path, default=ROOT / "dicom_files")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "dicom_case_plan")
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max-frames-per-clip", type=int, default=5)
    parser.add_argument("--top-frames-per-clip", type=int, default=3)
    parser.add_argument("--min-angle-deg", type=float, default=18.0)
    parser.add_argument("--case-index", type=int, help="Case index from cases.csv when --dicom-dir contains multiple studies.")
    args = parser.parse_args()

    model = args.model or existing_default_model()
    if model is None or not model.exists():
        raise FileNotFoundError("No ONNX segmentation model found. Pass --model explicitly.")

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    all_clips = find_xa_clips_fast(args.dicom_dir)
    if len(all_clips) < 1:
        raise RuntimeError(f"No usable XA cine clips found under {args.dicom_dir}")
    cases = group_clips_by_case(all_clips, args.dicom_dir)
    write_csv(out / "cases.csv", case_rows(cases))
    if len(cases) > 1 and args.case_index is None:
        raise RuntimeError(
            f"{args.dicom_dir} contains {len(cases)} separate DICOM studies. "
            f"Review {out / 'cases.csv'} and rerun with --case-index N, or pass a single case folder as --dicom-dir."
        )
    if args.case_index is not None:
        if args.case_index < 0 or args.case_index >= len(cases):
            raise IndexError(f"--case-index must be between 0 and {len(cases) - 1}")
        selected_case = cases[args.case_index]
    else:
        selected_case = cases[0]
    clips = selected_case.clips
    case_dicom_dir = args.dicom_dir if len(cases) == 1 else selected_case_dicom_dir(args.dicom_dir, selected_case)

    session = ort.InferenceSession(str(model))
    input_name = session.get_inputs()[0].name
    all_candidates = build_frame_candidates(clips, session, input_name, args.threshold, args.max_frames_per_clip)
    candidates = keep_best_frames_per_clip(all_candidates, args.top_frames_per_clip)

    manifest = {
        "dicom_dir": str(args.dicom_dir),
        "case_dicom_dir": str(case_dicom_dir),
        "case_key": selected_case.case_key,
        "case_root_hint": selected_case.root_hint,
        "model": str(model),
        "num_xa_clips_found": len(clips),
        "clips": clip_rows(clips),
        "outputs": {
            "cases": str(out / "cases.csv"),
            "clip_summary": str(out / "clip_summary.csv"),
            "frame_candidates": str(out / "frame_candidates.csv"),
            "preview_grid": str(out / "frame_quality_preview.png"),
            "reconstruction_plan": str(out / "reconstruction_plan.json"),
        },
    }
    write_csv(out / "clip_summary.csv", clip_rows(clips))
    write_csv(out / "frame_candidates.csv", frame_rows(candidates))
    save_preview_grid(candidates, out / "frame_quality_preview.png")

    plan = build_reconstruction_plan(clips, candidates, args.min_angle_deg)
    plan["dicom_dir"] = str(args.dicom_dir)
    plan["case_dicom_dir"] = str(case_dicom_dir)
    plan["case_key"] = selected_case.case_key
    plan["case_root_hint"] = selected_case.root_hint
    plan["model"] = str(model)
    add_commands_to_plan(plan, case_dicom_dir, out, model, args.threshold)

    with open(out / "case_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    with open(out / "reconstruction_plan.json", "w", encoding="utf-8") as f:
        json.dump(plan, f, indent=2)

    print(json.dumps(plan, indent=2))


if __name__ == "__main__":
    main()
