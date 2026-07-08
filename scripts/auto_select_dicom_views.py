"""
Automatically select angiography clips and matching frames for two-view 3D
coronary reconstruction.

This script does not reconstruct the final 3D mesh. It creates the missing
decision layer:

1. Inspect all DICOM XA clips.
2. Sample useful frames from each clip.
3. Segment vessels and score frame quality.
4. Rank view/frame pairs using angle separation + epipolar branch consistency.
5. Recommend a main reconstruction pair and optional third validation view.

Example:
    python scripts/auto_select_dicom_views.py --dicom-dir dicom_sample --output-dir dicom_auto_selection
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

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
    DEFAULT_MODEL,
    Clip,
    extract_graph,
    find_xa_clips,
    score_pair,
    segment_frame,
    view_from_clip,
)


@dataclass
class FrameCandidate:
    clip: Clip
    frame: int
    gray: np.ndarray
    mask: np.ndarray
    overlay: np.ndarray
    clean_mask: np.ndarray
    skeleton: np.ndarray
    branches: List[Dict[str, object]]
    area_pct: float
    total_length_px: float
    quality_score: float


def sample_frame_numbers(frame_count: int, max_frames: int) -> List[int]:
    if frame_count <= max_frames:
        return list(range(1, frame_count + 1))
    fractions = np.linspace(0.18, 0.90, max_frames)
    frames = sorted({int(np.clip(round(frame_count * frac), 1, frame_count)) for frac in fractions})
    return frames


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
        out = np.clip((f - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
    return cv2.resize(out, (512, 512), interpolation=cv2.INTER_AREA)


def score_frame(mask: np.ndarray, branches: List[Dict[str, object]]) -> tuple[float, float, float]:
    area_pct = float(np.count_nonzero(mask) / mask.size * 100.0)
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
                    mask=mask,
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
    if separation < 15.0:
        return 0.0
    if separation <= 55.0:
        return separation / 55.0
    if separation <= 90.0:
        return 1.0 - (separation - 55.0) / 70.0
    return 0.45


def branch_pair_objective(score: Dict[str, object], length_ratio: float) -> float:
    return float(score["median"] + 0.18 * score["p90"] + 2.0 * abs(math.log(max(length_ratio, 1e-6))))


def score_candidate_pair(a: FrameCandidate, b: FrameCandidate) -> Optional[Dict[str, object]]:
    if a.clip.index == b.clip.index:
        return None
    if len(a.branches) < 2 or len(b.branches) < 2:
        return None
    sep = angle_separation(a.clip, b.clip)
    if sep < 18.0:
        return None

    view_a = view_from_clip(a.clip, "view_a")
    view_b = view_from_clip(b.clip, "view_b")
    top_a = sorted(a.branches, key=lambda branch: float(branch["length_px"]), reverse=True)[:4]
    top_b = sorted(b.branches, key=lambda branch: float(branch["length_px"]), reverse=True)[:4]

    cost = np.full((len(top_a), len(top_b)), 1e6, dtype=np.float64)
    score_grid: List[List[Optional[Dict[str, object]]]] = [[None for _ in top_b] for _ in top_a]
    for i, branch_a in enumerate(top_a):
        for j, branch_b in enumerate(top_b):
            best = None
            for reverse_a in (False, True):
                for reverse_b in (False, True):
                    score = score_pair(branch_a, branch_b, view_a, view_b, reverse_a, reverse_b, samples=20)
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

    assigned_rows, assigned_cols = linear_sum_assignment(cost)
    best_scores = []
    reliable = 0
    usable = 0
    for i, j in zip(assigned_rows, assigned_cols):
        best = score_grid[i][j]
        if best is None or best["objective"] >= 1e5:
            continue
        best_scores.append(best)
        if best["median"] <= 3.0 and best["p90"] <= 7.0:
            reliable += 1
        elif best["median"] <= 5.5 and best["p90"] <= 12.0:
            usable += 1

    if not best_scores:
        return None

    median_epipolar = float(np.median([item["median"] for item in best_scores]))
    p90_epipolar = float(np.median([item["p90"] for item in best_scores]))
    objective = float(np.median([item["objective"] for item in best_scores]))
    quality = (a.quality_score + b.quality_score) * 0.5
    branch_count_ratio = min(len(a.branches), len(b.branches)) / max(len(a.branches), len(b.branches), 1)
    area_ratio = min(a.area_pct, b.area_pct) / max(a.area_pct, b.area_pct, 1e-6)
    coverage_penalty = 0.0
    if min(len(a.branches), len(b.branches)) < 7:
        coverage_penalty += 14.0
    if branch_count_ratio < 0.55:
        coverage_penalty += (0.55 - branch_count_ratio) * 45.0
    score = (
        28.0 * angle_score(sep)
        + 0.22 * quality
        + 11.0 * reliable
        + 5.0 * usable
        + 35.0 * branch_count_ratio
        + 4.0 * area_ratio
        - 1.6 * objective
        - 0.15 * p90_epipolar
        - coverage_penalty
    )

    return {
        "view_a_clip_index": a.clip.index,
        "view_a_frame": a.frame,
        "view_b_clip_index": b.clip.index,
        "view_b_frame": b.frame,
        "angle_separation_deg": sep,
        "score": float(score),
        "pair_objective": objective,
        "median_epipolar_residual_mm": median_epipolar,
        "p90_epipolar_residual_mm": p90_epipolar,
        "estimated_reliable_matches": reliable,
        "estimated_usable_matches": usable,
        "view_a_quality": a.quality_score,
        "view_b_quality": b.quality_score,
        "view_a_branches": len(a.branches),
        "view_b_branches": len(b.branches),
        "view_a_area_pct": a.area_pct,
        "view_b_area_pct": b.area_pct,
        "top_branch_matches": best_scores[:5],
    }


def rank_pairs(candidates: List[FrameCandidate]) -> List[Dict[str, object]]:
    pairs = []
    for i, a in enumerate(candidates):
        for b in candidates[i + 1:]:
            scored = score_candidate_pair(a, b)
            if scored:
                # Keep the lower clip index as view A only for stable reporting.
                pairs.append(scored)
    return sorted(pairs, key=lambda item: item["score"], reverse=True)


def choose_validation_view(best_pair: Dict[str, object], candidates: List[FrameCandidate]) -> Optional[Dict[str, object]]:
    used = {int(best_pair["view_a_clip_index"]), int(best_pair["view_b_clip_index"])}
    validation_candidates = [item for item in candidates if item.clip.index not in used and item.quality_score >= 30.0]
    if not validation_candidates:
        return None

    def score_validation(candidate: FrameCandidate) -> float:
        sep_a = math.hypot(candidate.clip.primary - best_pair.get("view_a_primary_angle_deg", 0.0), candidate.clip.secondary - best_pair.get("view_a_secondary_angle_deg", 0.0))
        sep_b = math.hypot(candidate.clip.primary - best_pair.get("view_b_primary_angle_deg", 0.0), candidate.clip.secondary - best_pair.get("view_b_secondary_angle_deg", 0.0))
        return candidate.quality_score + 12.0 * angle_score(max(sep_a, sep_b))

    best = max(validation_candidates, key=score_validation)
    return {
        "clip_index": best.clip.index,
        "frame": best.frame,
        "primary_angle_deg": best.clip.primary,
        "secondary_angle_deg": best.clip.secondary,
        "quality_score": best.quality_score,
        "branches": len(best.branches),
        "area_pct": best.area_pct,
        "role": "optional_third_view_validation",
    }


def candidate_to_row(candidate: FrameCandidate) -> Dict[str, object]:
    return {
        "clip_index": candidate.clip.index,
        "frame": candidate.frame,
        "primary_angle_deg": candidate.clip.primary,
        "secondary_angle_deg": candidate.clip.secondary,
        "quality_score": candidate.quality_score,
        "branches": len(candidate.branches),
        "area_pct": candidate.area_pct,
        "total_length_px": candidate.total_length_px,
        "dicom_path": str(candidate.clip.path),
    }


def save_preview_grid(candidates: List[FrameCandidate], output_path: Path, max_items: int = 16):
    selected = sorted(candidates, key=lambda item: item.quality_score, reverse=True)[:max_items]
    thumbs = []
    for item in selected:
        preview = cv2.resize(item.overlay, (256, 256), interpolation=cv2.INTER_AREA)
        label = f"IM{item.clip.index} F{item.frame} Q{item.quality_score:.0f}"
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


def enrich_pair_with_angles(pair: Dict[str, object], clips_by_index: Dict[int, Clip]) -> Dict[str, object]:
    a = clips_by_index[int(pair["view_a_clip_index"])]
    b = clips_by_index[int(pair["view_b_clip_index"])]
    return {
        **pair,
        "view_a_primary_angle_deg": a.primary,
        "view_a_secondary_angle_deg": a.secondary,
        "view_b_primary_angle_deg": b.primary,
        "view_b_secondary_angle_deg": b.secondary,
        "view_a_dicom_path": str(a.path),
        "view_b_dicom_path": str(b.path),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dicom-dir", type=Path, default=ROOT / "dicom_sample")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "dicom_auto_selection")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max-frames-per-clip", type=int, default=5)
    parser.add_argument("--top-frames-per-clip", type=int, default=2)
    args = parser.parse_args()

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    clips = find_xa_clips(args.dicom_dir)
    if len(clips) < 2:
        raise RuntimeError("Need at least two XA clips for automatic selection.")

    session = ort.InferenceSession(str(args.model))
    input_name = session.get_inputs()[0].name
    all_candidates = build_frame_candidates(clips, session, input_name, args.threshold, args.max_frames_per_clip)
    candidates = keep_best_frames_per_clip(all_candidates, args.top_frames_per_clip)
    ranked_pairs = rank_pairs(candidates)
    if not ranked_pairs:
        raise RuntimeError("No usable view/frame pairs were found.")

    clips_by_index = {clip.index: clip for clip in clips}
    best_pair = enrich_pair_with_angles(ranked_pairs[0], clips_by_index)
    validation_view = choose_validation_view(best_pair, candidates)

    with open(out / "frame_candidates.csv", "w", newline="", encoding="utf-8") as f:
        rows = [candidate_to_row(candidate) for candidate in candidates]
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    pair_rows = [enrich_pair_with_angles(pair, clips_by_index) for pair in ranked_pairs[:30]]
    pair_csv_rows = [{k: v for k, v in row.items() if k != "top_branch_matches"} for row in pair_rows]
    with open(out / "view_pair_rankings.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(pair_csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(pair_csv_rows)

    save_preview_grid(candidates, out / "top_frame_candidates.png")

    recommendation = {
        "dicom_dir": str(args.dicom_dir),
        "num_xa_clips_found": len(clips),
        "clips": [
            {
                "clip_index": clip.index,
                "dicom_path": str(clip.path),
                "frames": clip.frames,
                "primary_angle_deg": clip.primary,
                "secondary_angle_deg": clip.secondary,
                "acquisition_time": clip.acquisition_time,
            }
            for clip in clips
        ],
        "recommended_main_pair": best_pair,
        "optional_validation_view": validation_view,
        "next_pipeline_command": (
            "python scripts\\dicom_3d_pipeline.py "
            f"--dicom-dir {args.dicom_dir} --output-dir AUTO_PIPELINE_OUTPUT "
            f"--view-a-index {best_pair['view_a_clip_index']} --frame-a {best_pair['view_a_frame']} "
            f"--view-b-index {best_pair['view_b_clip_index']} --frame-b {best_pair['view_b_frame']}"
        ),
        "outputs": {
            "frame_candidates": str(out / "frame_candidates.csv"),
            "view_pair_rankings": str(out / "view_pair_rankings.csv"),
            "preview_grid": str(out / "top_frame_candidates.png"),
        },
    }
    with open(out / "auto_selection_report.json", "w", encoding="utf-8") as f:
        json.dump(recommendation, f, indent=2)
    print(json.dumps(recommendation, indent=2))


if __name__ == "__main__":
    main()
