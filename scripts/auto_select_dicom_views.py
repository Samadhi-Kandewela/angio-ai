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
    endpoint_count: int
    bifurcation_count: int
    shape_descriptor: List[float]
    quality_score: float
    base_mask_quality_score: float
    sharpness_score: float
    contrast_score: float
    vessel_contrast_score: float
    continuity_score: float
    anatomy_visibility_score: float
    fragmentation_penalty: float
    exposure_penalty: float
    clinical_quality_score: float
    quality_gate: str
    quality_reason: str
    artery_group: str
    artery_source: str


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


def metadata_text(path: Path) -> str:
    try:
        ds = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
    except Exception:
        return ""
    fields = [
        "SeriesDescription",
        "ProtocolName",
        "ImageComments",
        "Laterality",
        "BodyPartExamined",
        "RequestedProcedureDescription",
        "PerformedProcedureStepDescription",
    ]
    chunks = []
    for name in fields:
        value = getattr(ds, name, "")
        if value:
            chunks.append(str(value))
    return " ".join(chunks).upper()


def classify_clip_artery(clip: Clip) -> Dict[str, str]:
    text = f" {metadata_text(clip.path)} "
    left_tokens = (" LCA ", " LEFT CORONARY ", " LEFT MAIN ", " LAD ", " LCX ")
    right_tokens = (" RCA ", " RIGHT CORONARY ")
    has_left = any(token in text for token in left_tokens)
    has_right = any(token in text for token in right_tokens)
    if has_left and not has_right:
        return {"artery_group": "LCA", "artery_source": "dicom_text"}
    if has_right and not has_left:
        return {"artery_group": "RCA", "artery_source": "dicom_text"}
    if has_left and has_right:
        return {"artery_group": "unknown", "artery_source": "conflicting_dicom_text"}
    return {"artery_group": "unknown", "artery_source": "not_in_dicom_text"}


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


def connected_component_stats(mask: np.ndarray) -> tuple[int, float]:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
    if num_labels <= 1:
        return 0, 0.0
    areas = stats[1:, cv2.CC_STAT_AREA].astype(np.float64)
    largest = float(np.max(areas))
    total = float(np.sum(areas))
    small_components = int(np.sum(areas < max(20.0, largest * 0.08)))
    largest_ratio = largest / max(total, 1.0)
    return small_components, largest_ratio


def image_quality_metrics(gray: np.ndarray, mask: np.ndarray, skeleton: np.ndarray, branches: List[Dict[str, object]], base_quality: float) -> Dict[str, object]:
    vessel = mask > 0
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)

    lap = cv2.Laplacian(clahe, cv2.CV_64F)
    if np.count_nonzero(vessel) > 20:
        sharpness_raw = float(np.var(lap[vessel]))
    else:
        sharpness_raw = float(np.var(lap))
    sharpness_score = 100.0 * min(sharpness_raw / 120.0, 1.0)

    p2, p98 = np.percentile(gray, [2, 98])
    global_contrast_raw = float(p98 - p2)
    contrast_score = 100.0 * min(global_contrast_raw / 105.0, 1.0)

    dilated = cv2.dilate((mask > 0).astype(np.uint8), np.ones((11, 11), np.uint8), iterations=1) > 0
    local_bg = np.logical_and(dilated, ~vessel)
    if np.count_nonzero(vessel) > 20 and np.count_nonzero(local_bg) > 20:
        vessel_mean = float(np.mean(clahe[vessel]))
        bg_mean = float(np.mean(clahe[local_bg]))
        vessel_contrast_raw = abs(vessel_mean - bg_mean)
    else:
        vessel_contrast_raw = 0.0
    vessel_contrast_score = 100.0 * min(vessel_contrast_raw / 32.0, 1.0)

    skeleton_len = float(np.count_nonzero(skeleton))
    branch_count = len(branches)
    continuity_score = 100.0 * min(skeleton_len / max(260.0 + 30.0 * branch_count, 1.0), 1.0)
    lengths = sorted((float(branch.get("length_px", 0.0)) for branch in branches), reverse=True)
    longest = lengths[0] if lengths else 0.0
    second = lengths[1] if len(lengths) > 1 else 0.0
    area_pct = float(np.count_nonzero(mask) / mask.size * 100.0)
    anatomy_visibility_score = 100.0 * (
        0.28 * min(branch_count / 8.0, 1.0)
        + 0.26 * min(skeleton_len / 950.0, 1.0)
        + 0.18 * min(area_pct / 4.8, 1.0)
        + 0.16 * min(longest / 360.0, 1.0)
        + 0.12 * min(second / 220.0, 1.0)
    )

    small_components, largest_ratio = connected_component_stats(mask)
    fragmentation_penalty = min(45.0, 4.0 * small_components + 35.0 * max(0.0, 0.58 - largest_ratio))

    dark_pct = float(np.mean(gray <= 8) * 100.0)
    bright_pct = float(np.mean(gray >= 247) * 100.0)
    exposure_penalty = min(45.0, 1.8 * max(0.0, dark_pct - 1.0) + 1.8 * max(0.0, bright_pct - 1.0))

    clinical_quality = (
        0.22 * base_quality
        + 0.13 * sharpness_score
        + 0.11 * contrast_score
        + 0.15 * vessel_contrast_score
        + 0.14 * continuity_score
        + 0.25 * anatomy_visibility_score
        - fragmentation_penalty
        - exposure_penalty
    )
    clinical_quality = float(max(0.0, min(100.0, clinical_quality)))

    reasons = []
    if sharpness_score < 22.0:
        reasons.append("low_sharpness")
    if vessel_contrast_score < 18.0:
        reasons.append("weak_vessel_contrast")
    if continuity_score < 22.0:
        reasons.append("poor_centerline_continuity")
    if fragmentation_penalty > 22.0:
        reasons.append("fragmented_mask")
    if exposure_penalty > 18.0:
        reasons.append("exposure_artifact")
    if base_quality < 18.0:
        reasons.append("low_vessel_tree_quality")
    if anatomy_visibility_score < 42.0:
        reasons.append("weak_anatomy_visibility")
    if branch_count < 3:
        reasons.append("too_few_visible_branches")
    gate = "pass" if clinical_quality >= 34.0 and anatomy_visibility_score >= 38.0 and len(reasons) <= 2 else "review" if clinical_quality >= 22.0 else "reject"
    return {
        "sharpness_score": float(sharpness_score),
        "contrast_score": float(contrast_score),
        "vessel_contrast_score": float(vessel_contrast_score),
        "continuity_score": float(continuity_score),
        "anatomy_visibility_score": float(anatomy_visibility_score),
        "fragmentation_penalty": float(fragmentation_penalty),
        "exposure_penalty": float(exposure_penalty),
        "clinical_quality_score": clinical_quality,
        "quality_gate": gate,
        "quality_reason": ";".join(reasons) if reasons else "ok",
    }


def skeleton_counts(skeleton: np.ndarray) -> tuple[int, int]:
    ys, xs = np.where(skeleton > 0)
    points = set(zip(ys.tolist(), xs.tolist()))
    endpoints = 0
    bifurcations = 0
    for y, x in points:
        degree = 0
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                if (y + dy, x + dx) in points:
                    degree += 1
        if degree == 1:
            endpoints += 1
        elif degree >= 3:
            bifurcations += 1
    return endpoints, bifurcations


def shape_descriptor(area_pct: float, total_length_px: float, branches: List[Dict[str, object]], endpoints: int, bifurcations: int) -> List[float]:
    lengths = sorted((float(branch["length_px"]) for branch in branches), reverse=True)
    top = lengths[:3] + [0.0] * max(0, 3 - len(lengths))
    total = max(total_length_px, 1.0)
    return [
        area_pct / 20.0,
        total_length_px / 1800.0,
        len(branches) / 18.0,
        endpoints / 30.0,
        bifurcations / 40.0,
        top[0] / total,
        top[1] / total,
        top[2] / total,
    ]


def build_frame_candidates(
    clips: List[Clip],
    session,
    input_name: str,
    threshold: float,
    max_frames_per_clip: int,
) -> List[FrameCandidate]:
    candidates: List[FrameCandidate] = []
    for clip in clips:
        artery = classify_clip_artery(clip)
        pixel_array = load_clip_pixel_array(clip)
        for frame in sample_frame_numbers(clip.frames, max_frames_per_clip):
            gray = normalize_frame(pixel_array[frame - 1])
            mask, overlay = segment_frame(session, input_name, gray, threshold)
            clean_mask, skeleton, _, branches = extract_graph(mask)
            area_pct, total_length, quality = score_frame(clean_mask, branches)
            metrics = image_quality_metrics(gray, clean_mask, skeleton, branches, quality)
            endpoints, bifurcations = skeleton_counts(skeleton)
            descriptor = shape_descriptor(area_pct, total_length, branches, endpoints, bifurcations)
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
                    endpoint_count=endpoints,
                    bifurcation_count=bifurcations,
                    shape_descriptor=descriptor,
                    quality_score=metrics["clinical_quality_score"],
                    base_mask_quality_score=quality,
                    sharpness_score=metrics["sharpness_score"],
                    contrast_score=metrics["contrast_score"],
                    vessel_contrast_score=metrics["vessel_contrast_score"],
                    continuity_score=metrics["continuity_score"],
                    anatomy_visibility_score=metrics["anatomy_visibility_score"],
                    fragmentation_penalty=metrics["fragmentation_penalty"],
                    exposure_penalty=metrics["exposure_penalty"],
                    clinical_quality_score=metrics["clinical_quality_score"],
                    quality_gate=metrics["quality_gate"],
                    quality_reason=metrics["quality_reason"],
                    artery_group=artery["artery_group"],
                    artery_source=artery["artery_source"],
                )
            )
    return candidates


def keep_best_frames_per_clip(candidates: List[FrameCandidate], top_k: int) -> List[FrameCandidate]:
    by_clip: Dict[int, List[FrameCandidate]] = {}
    for candidate in candidates:
        by_clip.setdefault(candidate.clip.index, []).append(candidate)
    kept = []
    for frames in by_clip.values():
        pass_frames = [frame for frame in frames if frame.quality_gate == "pass"]
        review_frames = [frame for frame in frames if frame.quality_gate == "review"]
        pool = pass_frames if pass_frames else review_frames if review_frames else frames
        kept.extend(sorted(pool, key=lambda item: item.quality_score, reverse=True)[:top_k])
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


def artery_pair_allowed(a: FrameCandidate, b: FrameCandidate) -> tuple[bool, str]:
    if a.artery_group in {"LCA", "RCA"} and b.artery_group in {"LCA", "RCA"} and a.artery_group != b.artery_group:
        return False, "blocked_mixed_lca_rca"
    if a.artery_group == b.artery_group and a.artery_group in {"LCA", "RCA"}:
        return True, f"same_{a.artery_group}"
    return True, "unknown_or_unlabeled_review_required"


def same_phase_shape_score(a: FrameCandidate, b: FrameCandidate) -> Dict[str, float]:
    desc_a = np.asarray(a.shape_descriptor, dtype=np.float64)
    desc_b = np.asarray(b.shape_descriptor, dtype=np.float64)
    distance = float(np.linalg.norm(desc_a - desc_b))
    length_ratio = min(a.total_length_px, b.total_length_px) / max(a.total_length_px, b.total_length_px, 1e-6)
    area_ratio = min(a.area_pct, b.area_pct) / max(a.area_pct, b.area_pct, 1e-6)
    branch_ratio = min(len(a.branches), len(b.branches)) / max(len(a.branches), len(b.branches), 1)
    bifurcation_similarity = 1.0 - min(abs(a.bifurcation_count - b.bifurcation_count) / 25.0, 1.0)
    phase_score = float(
        100.0
        * (
            0.30 * max(0.0, 1.0 - distance)
            + 0.24 * length_ratio
            + 0.18 * area_ratio
            + 0.18 * branch_ratio
            + 0.10 * bifurcation_similarity
        )
    )
    return {
        "shape_distance": distance,
        "cardiac_phase_similarity_score": phase_score,
        "length_ratio": float(length_ratio),
        "area_ratio": float(area_ratio),
        "branch_count_ratio": float(branch_ratio),
        "bifurcation_similarity": float(bifurcation_similarity),
    }


def branch_pair_objective(score: Dict[str, object], length_ratio: float) -> float:
    return float(score["median"] + 0.18 * score["p90"] + 2.0 * abs(math.log(max(length_ratio, 1e-6))))


def score_candidate_pair(a: FrameCandidate, b: FrameCandidate) -> Optional[Dict[str, object]]:
    if a.clip.index == b.clip.index:
        return None
    allowed, artery_pair_status = artery_pair_allowed(a, b)
    if not allowed:
        return None
    if len(a.branches) < 2 or len(b.branches) < 2:
        return None
    if a.quality_gate == "reject" or b.quality_gate == "reject":
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
    min_quality = min(a.quality_score, b.quality_score)
    phase = same_phase_shape_score(a, b)
    min_branches = min(len(a.branches), len(b.branches))
    branch_count_ratio = min_branches / max(len(a.branches), len(b.branches), 1)
    # Absolute richness: two frames with few-but-matching branches (e.g. 5/6)
    # otherwise score as well on branch_count_ratio as two genuinely rich
    # frames (e.g. 14/9) -- reward min branch count directly, and only let
    # the ratio bonus count fully once there's enough anatomy on both sides
    # to be worth ratio-matching in the first place.
    richness_score = min(min_branches / 12.0, 1.0)
    ratio_gate = min(min_branches / 8.0, 1.0)
    area_ratio = min(a.area_pct, b.area_pct) / max(a.area_pct, b.area_pct, 1e-6)
    coverage_penalty = 0.0
    if min(len(a.branches), len(b.branches)) < 7:
        coverage_penalty += (7 - min(len(a.branches), len(b.branches))) * 6.0
    if min(len(a.branches), len(b.branches)) < 4:
        coverage_penalty += 18.0
    if min(a.anatomy_visibility_score, b.anatomy_visibility_score) < 52.0:
        coverage_penalty += (52.0 - min(a.anatomy_visibility_score, b.anatomy_visibility_score)) * 0.55
    if min(a.anatomy_visibility_score, b.anatomy_visibility_score) < 62.0:
        coverage_penalty += (62.0 - min(a.anatomy_visibility_score, b.anatomy_visibility_score)) * 0.65
    epipolar_penalty = 0.0
    if objective > 7.0:
        epipolar_penalty += (objective - 7.0) * 4.2
    if median_epipolar > 5.5:
        epipolar_penalty += (median_epipolar - 5.5) * 3.0
    if p90_epipolar > 11.0:
        epipolar_penalty += (p90_epipolar - 11.0) * 1.4
    if branch_count_ratio < 0.55:
        coverage_penalty += (0.55 - branch_count_ratio) * 45.0
    clinical_quality_penalty = 0.0
    if min_quality < 35.0:
        clinical_quality_penalty += (35.0 - min_quality) * 0.9
    if a.quality_gate == "review":
        clinical_quality_penalty += 7.0
    if b.quality_gate == "review":
        clinical_quality_penalty += 7.0
    score = (
        28.0 * angle_score(sep)
        + 0.42 * quality
        + 11.0 * reliable
        + 5.0 * usable
        + 24.0 * richness_score
        + 20.0 * branch_count_ratio * ratio_gate
        + 4.0 * area_ratio
        + 0.14 * phase["cardiac_phase_similarity_score"]
        - 1.6 * objective
        - 0.15 * p90_epipolar
        - coverage_penalty
        - clinical_quality_penalty
        - epipolar_penalty
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
        "view_a_quality_gate": a.quality_gate,
        "view_b_quality_gate": b.quality_gate,
        "view_a_quality_reason": a.quality_reason,
        "view_b_quality_reason": b.quality_reason,
        "view_a_sharpness_score": a.sharpness_score,
        "view_b_sharpness_score": b.sharpness_score,
        "view_a_vessel_contrast_score": a.vessel_contrast_score,
        "view_b_vessel_contrast_score": b.vessel_contrast_score,
        "view_a_continuity_score": a.continuity_score,
        "view_b_continuity_score": b.continuity_score,
        "view_a_anatomy_visibility_score": a.anatomy_visibility_score,
        "view_b_anatomy_visibility_score": b.anatomy_visibility_score,
        "clinical_quality_penalty": clinical_quality_penalty,
        "coverage_penalty": coverage_penalty,
        "epipolar_penalty": epipolar_penalty,
        "view_a_branches": len(a.branches),
        "view_b_branches": len(b.branches),
        "view_a_area_pct": a.area_pct,
        "view_b_area_pct": b.area_pct,
        "artery_pair_status": artery_pair_status,
        "view_a_artery_group": a.artery_group,
        "view_b_artery_group": b.artery_group,
        "cardiac_phase_similarity_score": phase["cardiac_phase_similarity_score"],
        "mask_shape_distance": phase["shape_distance"],
        "mask_length_ratio": phase["length_ratio"],
        "mask_area_ratio": phase["area_ratio"],
        "branch_count_ratio": phase["branch_count_ratio"],
        "bifurcation_similarity": phase["bifurcation_similarity"],
        "view_a_bifurcations": a.bifurcation_count,
        "view_b_bifurcations": b.bifurcation_count,
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
        "base_mask_quality_score": candidate.base_mask_quality_score,
        "sharpness_score": candidate.sharpness_score,
        "contrast_score": candidate.contrast_score,
        "vessel_contrast_score": candidate.vessel_contrast_score,
        "continuity_score": candidate.continuity_score,
        "anatomy_visibility_score": candidate.anatomy_visibility_score,
        "fragmentation_penalty": candidate.fragmentation_penalty,
        "exposure_penalty": candidate.exposure_penalty,
        "clinical_quality_score": candidate.clinical_quality_score,
        "quality_gate": candidate.quality_gate,
        "quality_reason": candidate.quality_reason,
        "branches": len(candidate.branches),
        "area_pct": candidate.area_pct,
        "total_length_px": candidate.total_length_px,
        "endpoint_count": candidate.endpoint_count,
        "bifurcation_count": candidate.bifurcation_count,
        "artery_group": candidate.artery_group,
        "artery_source": candidate.artery_source,
        "dicom_path": str(candidate.clip.path),
    }


def clip_classification_rows(clips: List[Clip]) -> List[Dict[str, object]]:
    rows = []
    for clip in clips:
        artery = classify_clip_artery(clip)
        rows.append(
            {
                "clip_index": clip.index,
                "artery_group": artery["artery_group"],
                "artery_source": artery["artery_source"],
                "primary_angle_deg": clip.primary,
                "secondary_angle_deg": clip.secondary,
                "frames": clip.frames,
                "dicom_path": str(clip.path),
            }
        )
    return rows


def save_preview_grid(candidates: List[FrameCandidate], output_path: Path, max_items: int = 16):
    selected = sorted(candidates, key=lambda item: item.quality_score, reverse=True)[:max_items]
    thumbs = []
    for item in selected:
        preview = cv2.resize(item.overlay, (256, 256), interpolation=cv2.INTER_AREA)
        label = f"IM{item.clip.index} F{item.frame} Q{item.quality_score:.0f} {item.quality_gate}"
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
    parser.add_argument("--max-frames-per-clip", type=int, default=12)
    parser.add_argument("--top-frames-per-clip", type=int, default=5)
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
        rows = [candidate_to_row(candidate) for candidate in all_candidates]
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    selected_rows = [candidate_to_row(candidate) for candidate in candidates]
    with open(out / "selected_frame_candidates.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(selected_rows[0].keys()))
        writer.writeheader()
        writer.writerows(selected_rows)

    non_pass_rows = [candidate_to_row(candidate) for candidate in all_candidates if candidate.quality_gate != "pass"]
    with open(out / "rejected_frame_report.csv", "w", newline="", encoding="utf-8") as f:
        if non_pass_rows:
            writer = csv.DictWriter(f, fieldnames=list(non_pass_rows[0].keys()))
            writer.writeheader()
            writer.writerows(non_pass_rows)
        else:
            f.write("clip_index,frame,quality_gate,quality_reason\n")

    with open(out / "clip_classification_report.csv", "w", newline="", encoding="utf-8") as f:
        rows = clip_classification_rows(clips)
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    pair_rows = [enrich_pair_with_angles(pair, clips_by_index) for pair in ranked_pairs[:30]]
    pair_csv_rows = [{k: v for k, v in row.items() if k != "top_branch_matches"} for row in pair_rows]
    with open(out / "view_pair_rankings.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(pair_csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(pair_csv_rows)
    with open(out / "frame_pair_candidates.csv", "w", newline="", encoding="utf-8") as f:
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
                **classify_clip_artery(clip),
            }
            for clip in clips
        ],
        "selection_method": {
            "artery_rule": "known LCA and RCA clips are never paired together; unknown pairs are allowed but marked for review",
            "frame_pair_rule": "candidate frames are scored as pairs using vessel quality, epipolar consistency, and image-based cardiac phase/shape similarity",
            "frame_quality_rule": "frames are gated by sharpness, contrast, vessel/background contrast, continuity, fragmentation, and exposure artifacts",
        },
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
            "selected_frame_candidates": str(out / "selected_frame_candidates.csv"),
            "rejected_frame_report": str(out / "rejected_frame_report.csv"),
            "clip_classification_report": str(out / "clip_classification_report.csv"),
            "frame_pair_candidates": str(out / "frame_pair_candidates.csv"),
            "view_pair_rankings": str(out / "view_pair_rankings.csv"),
            "preview_grid": str(out / "top_frame_candidates.png"),
        },
    }
    with open(out / "auto_selection_report.json", "w", encoding="utf-8") as f:
        json.dump(recommendation, f, indent=2)
    print(json.dumps(recommendation, indent=2))


if __name__ == "__main__":
    main()
