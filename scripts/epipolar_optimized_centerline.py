"""
Epipolar-guided two-view coronary centerline reconstruction.

This is a stricter research reconstruction than the visual hybrid pipeline:

1. Extract centerline branches from the two saved segmentation masks.
2. Match branches one-to-one using epipolar/ray residual, length, radius, and
   local reprojection checks.
3. Triangulate matched branch centerlines from DICOM geometry.
4. Refine each 3D branch with a small reprojection + smoothness optimization.
5. Export an OBJ, centerline JSON, match report, and reprojection overlays.

Example:
    python scripts/epipolar_optimized_centerline.py --input-dir dicom_pipeline_output_v2 --output-dir dicom_epipolar_optimized
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from scipy.optimize import least_squares, linear_sum_assignment


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from dicom_3d_pipeline import (  # noqa: E402
    closest_point_between_rays,
    extract_graph,
    ray_from_pixel,
    resample_polyline,
    rotation_from_angles,
    smooth_1d,
    variable_tube,
    write_obj,
)

from localization import localize_point  # noqa: E402


def load_gray(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(path)
    return image


def clahe_enhance_rgb(gray: np.ndarray) -> np.ndarray:
    """Matches src/frame_pipeline.py::preprocess_frame's enhancement recipe,
    which the localization model was trained/deployed against."""
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return cv2.cvtColor(cv2.merge((clahe.apply(l), a, b)), cv2.COLOR_LAB2RGB)


# Learned SYNTAX artery labels (src/localization_labels.py) collapsed onto the
# LM/LAD/LCX/side_branch roles that branch_role_probabilities already scores.
_ROLE_BY_ARTERY = {
    "LM": "LM",
    "LAD": "LAD",
    "LCX": "LCX",
    "LCX/PDA": "LCX",
    "RCA": "side_branch",
    "RCA/PDA": "side_branch",
}


def sample_branch_anatomy(branch: Dict[str, object], class_map, confidence_map) -> Dict[str, object] | None:
    """Majority-votes the learned per-pixel anatomy class map along a
    branch's centerline. Returns the winning (artery, group) with its
    LM/LAD/LCX/side_branch role and average confidence, or None if no
    learned map is available or every sampled point was background."""
    if class_map is None:
        return None
    points = branch["centerline_yx"]
    step = max(1, len(points) // 12)
    votes: Dict[Tuple[str, str], int] = {}
    conf_sum: Dict[Tuple[str, str], float] = {}
    for y, x in points[::step]:
        result = localize_point(class_map, confidence_map, (y, x), radius=7)
        if result["artery"] in ("unknown", "background"):
            continue
        key = (result["artery"], result["group"])
        votes[key] = votes.get(key, 0) + 1
        conf_sum[key] = conf_sum.get(key, 0.0) + result["confidence"]
    if not votes:
        return None
    best_key = max(votes, key=lambda k: votes[k])
    artery, group = best_key
    return {
        "artery": artery,
        "group": group,
        "role": _ROLE_BY_ARTERY.get(artery, "side_branch"),
        "confidence": float(conf_sum[best_key] / votes[best_key]),
        "vote_fraction": float(votes[best_key] / sum(votes.values())),
    }


def project_points(points_xyz: np.ndarray, view: Dict[str, object]) -> np.ndarray:
    rows, cols = [int(v) for v in view["image_size_px"]]
    spacing_y, spacing_x = [float(v) for v in view["imager_pixel_spacing_mm"]]
    sdd = float(view["source_detector_distance_mm"])
    spd = float(view["source_patient_distance_mm"])
    detector_z = sdd - spd
    source = np.array([0.0, 0.0, -spd], dtype=np.float64)
    rotation = rotation_from_angles(float(view["primary_angle_deg"]), float(view["secondary_angle_deg"]))

    out = []
    for point in points_xyz:
        local = rotation.T @ point
        denom = local[2] - source[2]
        if abs(float(denom)) < 1e-8:
            out.append((np.nan, np.nan))
            continue
        t = (detector_z - source[2]) / denom
        detector = source + t * (local - source)
        x = detector[0] / spacing_x + (cols - 1) * 0.5
        y = detector[1] / spacing_y + (rows - 1) * 0.5
        out.append((x, y))
    return np.asarray(out, dtype=np.float64)


def distance_to_mask_stats(points_xy: np.ndarray, distance_map: np.ndarray) -> Dict[str, float]:
    h, w = distance_map.shape[:2]
    finite = np.isfinite(points_xy).all(axis=1)
    inside = finite & (points_xy[:, 0] >= 0) & (points_xy[:, 0] < w) & (points_xy[:, 1] >= 0) & (points_xy[:, 1] < h)
    if not np.any(inside):
        return {"inside_pct": 0.0, "median_px": float("inf"), "p90_px": float("inf")}
    pix = np.rint(points_xy[inside]).astype(int)
    pix[:, 0] = np.clip(pix[:, 0], 0, w - 1)
    pix[:, 1] = np.clip(pix[:, 1], 0, h - 1)
    d = distance_map[pix[:, 1], pix[:, 0]].astype(np.float64)
    return {
        "inside_pct": float(np.count_nonzero(inside) / max(len(points_xy), 1) * 100.0),
        "median_px": float(np.median(d)),
        "p90_px": float(np.percentile(d, 90)),
    }


def sample_radius(dt: np.ndarray, coords_yx: np.ndarray, pixel_to_mm: float) -> np.ndarray:
    y = np.clip(np.rint(coords_yx[:, 0]).astype(int), 0, dt.shape[0] - 1)
    x = np.clip(np.rint(coords_yx[:, 1]).astype(int), 0, dt.shape[1] - 1)
    return np.clip(smooth_1d(dt[y, x] * pixel_to_mm, 9), 0.14, 2.4)


def triangulate_samples(samples_a: np.ndarray, samples_b: np.ndarray, view_a: Dict[str, object], view_b: Dict[str, object]):
    points = []
    ray_gaps = []
    for pa, pb in zip(samples_a, samples_b):
        source_a, dir_a = ray_from_pixel(pa, view_a)
        source_b, dir_b = ray_from_pixel(pb, view_b)
        point, gap = closest_point_between_rays(source_a, dir_a, source_b, dir_b)
        points.append(point)
        ray_gaps.append(gap)
    return np.asarray(points, dtype=np.float64), np.asarray(ray_gaps, dtype=np.float64)


def score_branch_pair(
    branch_a: Dict[str, object],
    branch_b: Dict[str, object],
    view_a: Dict[str, object],
    view_b: Dict[str, object],
    dist_a: np.ndarray,
    dist_b: np.ndarray,
    reverse_a: bool,
    reverse_b: bool,
    samples: int = 56,
) -> Dict[str, object]:
    pts_a = branch_a["centerline_yx"][::-1] if reverse_a else branch_a["centerline_yx"]
    pts_b = branch_b["centerline_yx"][::-1] if reverse_b else branch_b["centerline_yx"]
    n = min(samples, max(18, min(len(pts_a), len(pts_b))))
    samples_a = resample_polyline(pts_a, n)
    samples_b = resample_polyline(pts_b, n)
    points_3d, ray_gaps = triangulate_samples(samples_a, samples_b, view_a, view_b)

    proj_a = project_points(points_3d, view_a)
    proj_b = project_points(points_3d, view_b)
    reproj_a = np.linalg.norm(proj_a - np.column_stack([samples_a[:, 1], samples_a[:, 0]]), axis=1)
    reproj_b = np.linalg.norm(proj_b - np.column_stack([samples_b[:, 1], samples_b[:, 0]]), axis=1)
    mask_a = distance_to_mask_stats(proj_a, dist_a)
    mask_b = distance_to_mask_stats(proj_b, dist_b)

    length_ratio = max(float(branch_a["length_px"]), 1.0) / max(float(branch_b["length_px"]), 1.0)
    diameter_ratio = max(float(branch_a["mean_diameter_px"]), 0.2) / max(float(branch_b["mean_diameter_px"]), 0.2)
    ray_median = float(np.median(ray_gaps))
    ray_p90 = float(np.percentile(ray_gaps, 90))
    reproj_median = float(np.median(np.r_[reproj_a, reproj_b]))
    mask_median = float((mask_a["median_px"] + mask_b["median_px"]) * 0.5)
    objective = (
        ray_median
        + 0.18 * ray_p90
        + 0.65 * reproj_median
        + 0.35 * mask_median
        + 3.2 * abs(math.log(length_ratio))
        + 1.4 * abs(math.log(diameter_ratio))
    )

    return {
        "objective": float(objective),
        "samples_a": samples_a,
        "samples_b": samples_b,
        "points_3d": points_3d,
        "ray_median_mm": ray_median,
        "ray_p90_mm": ray_p90,
        "reproj_median_px": reproj_median,
        "mask_a_median_px": mask_a["median_px"],
        "mask_b_median_px": mask_b["median_px"],
        "inside_a_pct": mask_a["inside_pct"],
        "inside_b_pct": mask_b["inside_pct"],
        "reverse_a": reverse_a,
        "reverse_b": reverse_b,
    }


def best_pair_score(branch_a, branch_b, view_a, view_b, dist_a, dist_b) -> Dict[str, object]:
    best = None
    for reverse_a in (False, True):
        for reverse_b in (False, True):
            score = score_branch_pair(branch_a, branch_b, view_a, view_b, dist_a, dist_b, reverse_a, reverse_b)
            if best is None or score["objective"] < best["objective"]:
                best = score
    return best


def classify_match(score: Dict[str, object], branch_a: Dict[str, object]) -> str:
    if (
        score["ray_median_mm"] <= 2.5
        and score["ray_p90_mm"] <= 6.0
        and score["reproj_median_px"] <= 2.5
        and score["inside_a_pct"] >= 80.0
        and score["inside_b_pct"] >= 70.0
    ):
        return "reliable"
    if (
        score["ray_median_mm"] <= 4.5
        and score["ray_p90_mm"] <= 10.0
        and score["reproj_median_px"] <= 4.0
        and float(branch_a["length_px"]) >= 45.0
        and score["inside_a_pct"] >= 65.0
        and score["inside_b_pct"] >= 55.0
    ):
        return "usable"
    if (
        score["ray_median_mm"] <= 2.6
        and score["ray_p90_mm"] <= 4.5
        and score["reproj_median_px"] <= 7.5
        and score["mask_a_median_px"] <= 1.0
        and score["mask_b_median_px"] <= 1.0
        and score["inside_a_pct"] >= 90.0
        and score["inside_b_pct"] >= 90.0
    ):
        return "usable"
    return "rejected"


def proximal_bifurcation_risk(branch: Dict[str, object]) -> bool:
    """Conservative screen for short branches near the LCA inlet/bifurcation area.

    This is a safety gate, not final anatomy detection. It prevents a short
    proximal skeleton fragment from becoming a strong 3D anchor before the
    bifurcation-first matcher is available.
    """
    length_px = float(branch.get("length_px", 0.0))
    if length_px > 75.0:
        return False
    points = np.asarray(branch.get("centerline_yx", []), dtype=np.float64)
    if len(points) == 0:
        return False
    min_y = float(np.min(points[:, 0]))
    min_x = float(np.min(points[:, 1]))
    mean_x = float(np.mean(points[:, 1]))
    return min_x <= 230.0 and min_y <= 230.0 and mean_x <= 260.0


def very_strong_short_branch_evidence(score: Dict[str, object]) -> bool:
    return (
        score["ray_median_mm"] <= 1.0
        and score["ray_p90_mm"] <= 2.0
        and score["reproj_median_px"] <= 2.0
        and score["mask_a_median_px"] <= 1.0
        and score["mask_b_median_px"] <= 1.0
        and score["inside_a_pct"] >= 95.0
        and score["inside_b_pct"] >= 95.0
    )


def apply_safety_gates(status: str, score: Dict[str, object], branch_a: Dict[str, object]) -> tuple[str, str]:
    if status in {"reliable", "usable"} and proximal_bifurcation_risk(branch_a):
        if not very_strong_short_branch_evidence(score):
            return (
                "uncertain",
                "short proximal/bifurcation-region branch; requires bifurcation-first anatomy confirmation",
            )
    return status, ""


def load_json_if_exists(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def branch_endpoints(branch: Dict[str, object]) -> tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(branch.get("centerline_yx", []), dtype=np.float64)
    if len(pts) == 0:
        return np.zeros(2), np.zeros(2)
    return pts[0], pts[-1]


def branch_vector_from_node(branch: Dict[str, object], node_yx: np.ndarray) -> tuple[np.ndarray, float]:
    start, end = branch_endpoints(branch)
    if np.linalg.norm(start - node_yx) <= np.linalg.norm(end - node_yx):
        other = end
    else:
        other = start
    vec = other - node_yx
    norm = float(np.linalg.norm(vec))
    return vec / max(norm, 1e-6), norm


def node_branch_ids(node: Dict[str, object], graph: Dict[str, object], branches: List[Dict[str, object]], radius: float = 10.0) -> list[int]:
    node_id = str(node.get("node_id"))
    direct = list(graph.get("node_to_branch_ids", {}).get(node_id, []))
    if len(direct) >= 2:
        return [int(v) for v in direct]
    node_yx = np.asarray(node["yx"], dtype=np.float64)
    near = set(int(v) for v in direct)
    for branch in branches:
        start, end = branch_endpoints(branch)
        if min(np.linalg.norm(start - node_yx), np.linalg.norm(end - node_yx)) <= radius:
            near.add(int(branch["branch_id"]))
    return sorted(near)


def softmax_scores(scores: Dict[str, float]) -> Dict[str, float]:
    keys = list(scores)
    values = np.asarray([scores[k] for k in keys], dtype=np.float64)
    values = values - np.max(values)
    exp = np.exp(values)
    probs = exp / max(float(np.sum(exp)), 1e-9)
    return {key: float(value) for key, value in zip(keys, probs)}


def branch_role_probabilities(
    branch: Dict[str, object],
    bif_yx: np.ndarray,
    connected: bool,
    lm_branch_id: int | None = None,
    lad_branch_id: int | None = None,
    lcx_branch_id: int | None = None,
    learned_role: str | None = None,
    learned_confidence: float = 0.0,
) -> Dict[str, float]:
    branch_id = int(branch["branch_id"])
    if lm_branch_id is not None and branch_id == lm_branch_id:
        return {"LM": 0.84, "LAD": 0.04, "LCX": 0.04, "side_branch": 0.03, "unknown": 0.05}
    if lad_branch_id is not None and branch_id == lad_branch_id:
        return {"LM": 0.03, "LAD": 0.78, "LCX": 0.06, "side_branch": 0.08, "unknown": 0.05}
    if lcx_branch_id is not None and branch_id == lcx_branch_id:
        return {"LM": 0.03, "LAD": 0.07, "LCX": 0.76, "side_branch": 0.09, "unknown": 0.05}

    unit, distal_len = branch_vector_from_node(branch, bif_yx)
    dy, dx = float(unit[0]), float(unit[1])
    length = float(branch.get("length_px", 0.0))
    diameter = float(branch.get("mean_diameter_px", 0.0))
    start, end = branch_endpoints(branch)
    other_x = float(start[1] if np.linalg.norm(start - bif_yx) > np.linalg.norm(end - bif_yx) else end[1])
    bif_x = float(bif_yx[1])
    connected_bonus = 1.2 if connected else -0.5

    scores = {
        "LM": connected_bonus + 1.4 * (diameter / 20.0) + 1.0 * max(0.0, (bif_x - other_x) / 120.0) - 0.6 * max(0.0, (length - 95.0) / 120.0),
        "LAD": connected_bonus + 1.2 * max(0.0, dy) + 0.9 * min(length / 170.0, 1.2) + 0.2 * (diameter / 16.0),
        "LCX": connected_bonus + 1.1 * max(0.0, dx) + 0.7 * min(length / 160.0, 1.2) + 0.15 * (diameter / 16.0),
        "side_branch": 0.25 + 0.6 * max(0.0, abs(dx) + abs(dy) - 0.8) + 0.5 * min(length / 90.0, 1.0),
        "unknown": 0.2,
    }
    if learned_role in scores and learned_confidence > 0.0:
        # Bias toward the trained anatomical-localization model's vote
        # (src/frame_pipeline.py::LocalizationModel) instead of relying on
        # position/length/diameter geometry alone.
        scores[learned_role] += 3.0 * learned_confidence
    probs = softmax_scores(scores)
    return probs


def graph_node_by_id(graph: Dict[str, object]) -> Dict[int, Dict[str, object]]:
    return {int(node["node_id"]): node for node in graph.get("nodes", [])}


def branch_by_id(branches: List[Dict[str, object]]) -> Dict[int, Dict[str, object]]:
    return {int(branch["branch_id"]): branch for branch in branches}


def branch_other_node(branch_graph: Dict[str, object], node_id: int) -> int | None:
    start = branch_graph.get("start_node")
    end = branch_graph.get("end_node")
    if start == node_id:
        return int(end) if end is not None else None
    if end == node_id:
        return int(start) if start is not None else None
    return None


def endpoint_ostium_candidates(graph: Dict[str, object], branches: List[Dict[str, object]]) -> list[Dict[str, object]]:
    nodes = graph.get("nodes", [])
    branch_graphs = {int(item["branch_id"]): item for item in graph.get("branches", [])}
    branch_map = branch_by_id(branches)
    out = []
    for node in nodes:
        if node.get("kind") != "endpoint":
            continue
        ids = [branch_id for branch_id in node_branch_ids(node, graph, branches, radius=8.0) if branch_id in branch_map]
        if not ids:
            continue
        best_branch_id = max(ids, key=lambda bid: float(branch_map[bid].get("mean_diameter_px", 0.0)))
        branch = branch_map[best_branch_id]
        y, x = [float(v) for v in node["yx"]]
        leftness = max(0.0, 1.0 - x / 512.0)
        upper_bias = max(0.0, 1.0 - y / 420.0)
        diameter_score = min(float(branch["mean_diameter_px"]) / 18.0, 1.4)
        length = float(branch["length_px"])
        lm_length_score = 1.0 - min(abs(length - 80.0) / 150.0, 1.0)
        score = 35.0 * leftness + 16.0 * upper_bias + 22.0 * diameter_score + 12.0 * lm_length_score
        bg = branch_graphs.get(best_branch_id, {})
        out.append(
            {
                "node_id": int(node["node_id"]),
                "yx": [int(y), int(x)],
                "candidate_lm_branch_id": int(best_branch_id),
                "candidate_lm_other_node": branch_other_node(bg, int(node["node_id"])),
                "confidence_score": float(score),
                "diameter_px": float(branch["mean_diameter_px"]),
                "branch_length_px": length,
            }
        )
    return sorted(out, key=lambda item: item["confidence_score"], reverse=True)


def split_child_candidates(
    bif_node_id: int,
    lm_branch_id: int,
    graph: Dict[str, object],
    branches: List[Dict[str, object]],
) -> list[Dict[str, object]]:
    branch_map = branch_by_id(branches)
    node_map = graph_node_by_id(graph)
    node = node_map.get(int(bif_node_id))
    if not node:
        return []
    ids = [bid for bid in node_branch_ids(node, graph, branches, radius=12.0) if bid in branch_map and bid != lm_branch_id]
    bif_yx = np.asarray(node["yx"], dtype=np.float64)
    children = []
    for bid in ids:
        branch = branch_map[bid]
        unit, extent = branch_vector_from_node(branch, bif_yx)
        length = float(branch["length_px"])
        diameter = float(branch["mean_diameter_px"])
        main_score = 0.55 * min(length / 150.0, 1.3) + 0.45 * min(diameter / 16.0, 1.3)
        children.append(
            {
                "branch_id": int(bid),
                "length_px": length,
                "diameter_px": diameter,
                "unit_yx": [float(unit[0]), float(unit[1])],
                "extent_px": float(extent),
                "main_score": float(main_score),
            }
        )
    return sorted(children, key=lambda item: item["main_score"], reverse=True)


def assign_lad_lcx_from_children(children: list[Dict[str, object]]) -> tuple[int | None, int | None, float]:
    if len(children) < 2:
        return None, None, 0.0
    best_pair = None
    for i, a in enumerate(children):
        for b in children[i + 1:]:
            va = np.asarray(a["unit_yx"], dtype=np.float64)
            vb = np.asarray(b["unit_yx"], dtype=np.float64)
            angle_sep = 1.0 - float(np.dot(va, vb))
            score = a["main_score"] + b["main_score"] + 0.55 * angle_sep
            if best_pair is None or score > best_pair[0]:
                best_pair = (score, a, b, angle_sep)
    if best_pair is None:
        return None, None, 0.0
    _, a, b, angle_sep = best_pair
    # In image coordinates, larger +y usually tracks descending/apical direction.
    # Larger absolute/rightward x is treated as lateral LCx evidence.
    def lad_score(child):
        unit = child["unit_yx"]
        return 1.1 * max(0.0, unit[0]) + 0.45 * child["main_score"] + 0.15 * min(child["length_px"] / 180.0, 1.0)

    def lcx_score(child):
        unit = child["unit_yx"]
        return 0.8 * max(0.0, abs(unit[1])) + 0.5 * child["main_score"] + 0.2 * max(0.0, unit[1])

    if lad_score(a) + lcx_score(b) >= lad_score(b) + lcx_score(a):
        lad, lcx = a, b
    else:
        lad, lcx = b, a
    confidence = min(100.0, 35.0 * min(angle_sep, 1.6) + 25.0 * min(lad["length_px"] / 160.0, 1.2) + 25.0 * min(lcx["length_px"] / 130.0, 1.2))
    return int(lad["branch_id"]), int(lcx["branch_id"]), float(confidence)


def trace_lm_bifurcation_candidates(graph: Dict[str, object], branches: List[Dict[str, object]]) -> list[Dict[str, object]]:
    node_map = graph_node_by_id(graph)
    branch_graphs = {int(item["branch_id"]): item for item in graph.get("branches", [])}
    branch_map = branch_by_id(branches)
    candidates = []
    for ostium in endpoint_ostium_candidates(graph, branches)[:8]:
        lm_branch_id = int(ostium["candidate_lm_branch_id"])
        lm_graph = branch_graphs.get(lm_branch_id, {})
        bif_node_id = ostium.get("candidate_lm_other_node")
        if bif_node_id is None or bif_node_id not in node_map:
            continue
        bif_node = node_map[bif_node_id]
        if bif_node.get("kind") != "bifurcation":
            # Keep as a weak candidate only if it is close to a bifurcation.
            bif_yx = np.asarray(bif_node["yx"], dtype=np.float64)
            nearby = [
                node for node in node_map.values()
                if node.get("kind") == "bifurcation" and np.linalg.norm(np.asarray(node["yx"], dtype=np.float64) - bif_yx) <= 18.0
            ]
            if nearby:
                bif_node = max(nearby, key=lambda node: len(node_branch_ids(node, graph, branches)))
                bif_node_id = int(bif_node["node_id"])
            else:
                continue
        children = split_child_candidates(int(bif_node_id), lm_branch_id, graph, branches)
        lad_id, lcx_id, trunk_conf = assign_lad_lcx_from_children(children)
        lm_branch = branch_map.get(lm_branch_id, {})
        lm_len = float(lm_branch.get("length_px", 0.0))
        lm_diam = float(lm_branch.get("mean_diameter_px", 0.0))
        lm_shape = 1.0 - min(abs(lm_len - 80.0) / 160.0, 1.0)
        child_count_score = min(len(children) / 2.0, 1.4)
        confidence = (
            0.34 * float(ostium["confidence_score"])
            + 18.0 * child_count_score
            + 18.0 * min(lm_diam / 18.0, 1.3)
            + 14.0 * lm_shape
            + 0.45 * trunk_conf
        )
        candidates.append(
            {
                "ostium": ostium,
                "lm_branch_id": lm_branch_id,
                "lm_length_px": lm_len,
                "lm_diameter_px": lm_diam,
                "bifurcation_node_id": int(bif_node_id),
                "bifurcation_yx": [int(v) for v in bif_node["yx"]],
                "child_candidates": children[:6],
                "lad_branch_id": lad_id,
                "lcx_branch_id": lcx_id,
                "lad_lcx_confidence": trunk_conf,
                "confidence_score": float(confidence),
                "confidence": "high" if confidence >= 82.0 and lad_id is not None and lcx_id is not None else "moderate" if confidence >= 58.0 else "low",
            }
        )
    return sorted(candidates, key=lambda item: item["confidence_score"], reverse=True)


def detect_anatomy_anchors(
    branches: List[Dict[str, object]],
    graph: Dict[str, object],
    original: np.ndarray,
    mask: np.ndarray,
    output_path: Path,
    view_name: str,
    class_map=None,
    confidence_map=None,
) -> Dict[str, object]:
    branch_by_id = {int(branch["branch_id"]): branch for branch in branches}
    ostium_candidates = endpoint_ostium_candidates(graph, branches)
    lm_trace_candidates = trace_lm_bifurcation_candidates(graph, branches)
    nodes = [node for node in graph.get("nodes", []) if node.get("kind") == "bifurcation"]
    candidates = []
    for node in nodes:
        ids = [branch_id for branch_id in node_branch_ids(node, graph, branches) if branch_id in branch_by_id]
        if len(ids) < 2:
            continue
        connected = [branch_by_id[branch_id] for branch_id in ids]
        lengths = sorted([float(branch["length_px"]) for branch in connected], reverse=True)
        diameters = sorted([float(branch["mean_diameter_px"]) for branch in connected], reverse=True)
        y, x = [float(v) for v in node["yx"]]
        leftness = max(0.0, 1.0 - x / 512.0)
        vertical_ok = 1.0 - min(abs(y - 240.0) / 260.0, 1.0)
        score = (
            25.0 * min(len(ids) / 3.0, 1.4)
            + 22.0 * leftness
            + 10.0 * vertical_ok
            + 0.06 * sum(lengths[:3])
            + 1.2 * sum(diameters[:3])
        )
        candidates.append(
            {
                "node_id": int(node["node_id"]),
                "yx": [int(y), int(x)],
                "connected_branch_ids": ids,
                "confidence_score": float(score),
                "connected_branch_count": len(ids),
                "top_lengths_px": lengths[:3],
                "top_diameters_px": diameters[:3],
            }
        )
    candidates = sorted(candidates, key=lambda item: item["confidence_score"], reverse=True)
    selected_trace = lm_trace_candidates[0] if lm_trace_candidates else None
    selected = None
    if selected_trace:
        selected = {
            "node_id": int(selected_trace["bifurcation_node_id"]),
            "yx": selected_trace["bifurcation_yx"],
            "connected_branch_ids": [
                int(v)
                for v in [
                    selected_trace.get("lm_branch_id"),
                    selected_trace.get("lad_branch_id"),
                    selected_trace.get("lcx_branch_id"),
                ]
                if v is not None
            ],
            "confidence_score": float(selected_trace["confidence_score"]),
            "confidence": selected_trace["confidence"],
            "source": "ostium_lm_trace",
            "ostium_node_id": selected_trace["ostium"]["node_id"],
            "ostium_yx": selected_trace["ostium"]["yx"],
            "lm_branch_id": selected_trace.get("lm_branch_id"),
            "lad_branch_id": selected_trace.get("lad_branch_id"),
            "lcx_branch_id": selected_trace.get("lcx_branch_id"),
        }
    elif candidates:
        selected = {**candidates[0], "source": "fallback_bifurcation_node", "confidence": "low"}

    # Learned anatomy is independent of whether a bifurcation anchor was
    # found -- collect it for every branch so assign_coronary_anatomy_labels.py
    # can use it even when the geometric LM/bifurcation trace fails.
    learned_branch_anatomy = {}
    for branch in branches:
        learned = sample_branch_anatomy(branch, class_map, confidence_map)
        if learned is not None:
            learned_branch_anatomy[str(int(branch["branch_id"]))] = learned

    branch_probs = {}
    if selected:
        bif_yx = np.asarray(selected["yx"], dtype=np.float64)
        connected_ids = set(int(v) for v in selected["connected_branch_ids"])
        lm_branch_id = selected.get("lm_branch_id")
        lad_branch_id = selected.get("lad_branch_id")
        lcx_branch_id = selected.get("lcx_branch_id")
        for branch in branches:
            learned = learned_branch_anatomy.get(str(int(branch["branch_id"])))
            probs = branch_role_probabilities(
                branch,
                bif_yx,
                int(branch["branch_id"]) in connected_ids,
                lm_branch_id=lm_branch_id,
                lad_branch_id=lad_branch_id,
                lcx_branch_id=lcx_branch_id,
                learned_role=learned["role"] if learned else None,
                learned_confidence=learned["confidence"] if learned else 0.0,
            )
            branch_probs[str(int(branch["branch_id"]))] = probs

    draw_bifurcation_overlay(original, mask, branches, selected, branch_probs, output_path, view_name)
    return {
        "view": view_name,
        "selected_bifurcation": selected,
        "ostium_candidates": ostium_candidates[:8],
        "lm_trace_candidates": lm_trace_candidates[:8],
        "bifurcation_candidates": candidates[:8],
        "branch_role_probabilities": branch_probs,
        "learned_branch_anatomy": learned_branch_anatomy,
    }


def draw_bifurcation_overlay(
    original: np.ndarray,
    mask: np.ndarray,
    branches: List[Dict[str, object]],
    selected: Dict[str, object] | None,
    branch_probs: Dict[str, Dict[str, float]],
    output_path: Path,
    view_name: str,
):
    canvas = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR) if original.ndim == 2 else original.copy()
    red = canvas.copy()
    red[mask > 0] = (35, 35, 255)
    canvas = cv2.addWeighted(red, 0.28, canvas, 0.72, 0)
    colors = {"LM": (255, 255, 255), "LAD": (0, 255, 80), "LCX": (255, 190, 40), "side_branch": (0, 210, 255), "unknown": (190, 190, 190)}
    forced = {}
    if selected:
        if selected.get("lm_branch_id") is not None:
            forced[int(selected["lm_branch_id"])] = "LM"
        if selected.get("lad_branch_id") is not None:
            forced[int(selected["lad_branch_id"])] = "LAD"
        if selected.get("lcx_branch_id") is not None:
            forced[int(selected["lcx_branch_id"])] = "LCX"
    for branch in branches:
        pts = np.asarray(branch["centerline_yx"], dtype=np.int32)
        if len(pts) < 2:
            continue
        probs = branch_probs.get(str(int(branch["branch_id"])), {})
        label = forced.get(int(branch["branch_id"])) or (max(probs, key=probs.get) if probs else "unknown")
        color = colors.get(label, (190, 190, 190))
        thickness = 2 if int(branch["branch_id"]) in forced else 1
        xy = np.column_stack([pts[:, 1], pts[:, 0]])
        for a, b in zip(xy[:-1], xy[1:]):
            cv2.line(canvas, tuple(a), tuple(b), color, thickness, cv2.LINE_AA)
        mid = tuple(xy[len(xy) // 2])
        cv2.putText(canvas, f"{branch['branch_id']} {label}", mid, cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)
    if selected:
        y, x = selected["yx"]
        cv2.circle(canvas, (int(x), int(y)), 7, (0, 255, 255), 2, cv2.LINE_AA)
        conf = selected.get("confidence", "unknown")
        cv2.putText(canvas, f"{view_name} LM bif {conf}", (int(x) + 8, int(y) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
        if selected.get("source") == "ostium_lm_trace" and selected.get("ostium_node_id") is not None:
            # Find ostium coordinates from the selected trace candidate stored in the selected payload.
            for key in ("ostium_yx",):
                if selected.get(key):
                    oy, ox = selected[key]
                    cv2.circle(canvas, (int(ox), int(oy)), 6, (255, 255, 0), 2, cv2.LINE_AA)
                    cv2.putText(canvas, "ostium", (int(ox) + 6, int(oy) + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 0), 1, cv2.LINE_AA)
    cv2.imwrite(str(output_path), canvas)


def candidate_branches_for_role(branches: List[Dict[str, object]], anchors: Dict[str, object], role: str) -> list[Dict[str, object]]:
    probs = anchors.get("branch_role_probabilities", {})
    ranked = []
    for branch in branches:
        prob = float(probs.get(str(int(branch["branch_id"])), {}).get(role, 0.0))
        if prob > 0.08:
            ranked.append((prob, float(branch["length_px"]), branch))
    ranked = sorted(ranked, key=lambda item: (item[0], item[1]), reverse=True)
    return [item[2] for item in ranked[:5]]


def role_match_score(
    branch_a: Dict[str, object],
    branch_b: Dict[str, object],
    role: str,
    anchors_a: Dict[str, object],
    anchors_b: Dict[str, object],
    view_a,
    view_b,
    dist_a,
    dist_b,
) -> Dict[str, object]:
    base = best_pair_score(branch_a, branch_b, view_a, view_b, dist_a, dist_b)
    probs_a = anchors_a.get("branch_role_probabilities", {}).get(str(int(branch_a["branch_id"])), {})
    probs_b = anchors_b.get("branch_role_probabilities", {}).get(str(int(branch_b["branch_id"])), {})
    pa = float(probs_a.get(role, 0.0))
    pb = float(probs_b.get(role, 0.0))
    anatomy_penalty = 10.0 * (1.0 - pa) + 10.0 * (1.0 - pb)
    length = min(float(branch_a["length_px"]), float(branch_b["length_px"]))
    short_main_penalty = 0.0
    if role in {"LAD", "LCX"} and length < 75.0:
        short_main_penalty = 18.0
    if role == "LM" and length > 130.0:
        short_main_penalty = 8.0
    anatomy_objective = float(base["objective"] + anatomy_penalty + short_main_penalty)
    return {
        **base,
        "anatomy_objective": anatomy_objective,
        "anatomy_probability_a": pa,
        "anatomy_probability_b": pb,
        "anatomy_penalty": anatomy_penalty,
        "short_main_penalty": short_main_penalty,
    }


def classify_anatomy_match(role: str, score: Dict[str, object], branch_a: Dict[str, object], lad_lcx_stable: bool) -> tuple[str, str]:
    if role in {"LAD", "LCX"} and min(score["anatomy_probability_a"], score["anatomy_probability_b"]) < 0.22:
        return "rejected", f"{role} anatomy probability too low"
    if role in {"LAD", "LCX"} and float(branch_a.get("length_px", 0.0)) < 75.0:
        return "uncertain", f"{role} candidate is too short for a main-vessel anchor"
    status = classify_match(score, branch_a)
    status, reason = apply_safety_gates(status, score, branch_a)
    if status in {"reliable", "usable"} and role not in {"LM", "LAD", "LCX"} and not lad_lcx_stable:
        return "uncertain", "side branch withheld until LAD and LCx anchors are stable"
    return status, reason


def low_bifurcation_confidence(anchors_a: Dict[str, object], anchors_b: Dict[str, object]) -> bool:
    selected_a = anchors_a.get("selected_bifurcation") or {}
    selected_b = anchors_b.get("selected_bifurcation") or {}
    if not selected_a or not selected_b:
        return True
    if selected_a.get("source") != "ostium_lm_trace" or selected_b.get("source") != "ostium_lm_trace":
        return True
    if selected_a.get("confidence") == "low" or selected_b.get("confidence") == "low":
        return True
    return False


def anatomy_aware_matches(branches_a, branches_b, view_a, view_b, dist_a, dist_b, anchors_a, anchors_b):
    matches = []
    used_a: set[int] = set()
    used_b: set[int] = set()
    main_roles = ("LM", "LAD", "LCX")
    bif_low = low_bifurcation_confidence(anchors_a, anchors_b)
    for role in main_roles:
        best = None
        for branch_a in candidate_branches_for_role(branches_a, anchors_a, role):
            if int(branch_a["branch_id"]) in used_a:
                continue
            for branch_b in candidate_branches_for_role(branches_b, anchors_b, role):
                if int(branch_b["branch_id"]) in used_b:
                    continue
                score = role_match_score(branch_a, branch_b, role, anchors_a, anchors_b, view_a, view_b, dist_a, dist_b)
                if best is None or score["anatomy_objective"] < best["score"]["anatomy_objective"]:
                    best = {"branch_a": branch_a, "branch_b": branch_b, "score": score, "anatomy_role": role}
        if best is None:
            continue
        status, reason = classify_anatomy_match(role, best["score"], best["branch_a"], lad_lcx_stable=False)
        if bif_low and status in {"reliable", "usable"}:
            status = "uncertain"
            reason = "LM bifurcation confidence is not high enough for reliable main-vessel anchoring"
        best["status"] = status
        best["status_reason"] = reason
        matches.append(best)
        if status in {"reliable", "usable", "uncertain"}:
            used_a.add(int(best["branch_a"]["branch_id"]))
            used_b.add(int(best["branch_b"]["branch_id"]))

    stable_roles = {m["anatomy_role"] for m in matches if m["status"] in {"reliable", "usable"}}
    lad_lcx_stable = {"LAD", "LCX"}.issubset(stable_roles)

    remaining_a = [branch for branch in branches_a if int(branch["branch_id"]) not in used_a]
    remaining_b = [branch for branch in branches_b if int(branch["branch_id"]) not in used_b]
    if remaining_a and remaining_b:
        side_matches = one_to_one_geometry_matches(remaining_a, remaining_b, view_a, view_b, dist_a, dist_b)
        for match in side_matches:
            match["anatomy_role"] = "side_branch"
            if not lad_lcx_stable and match["status"] in {"reliable", "usable"}:
                match["status"] = "uncertain"
                match["status_reason"] = "side branch withheld until LAD and LCx anchors are stable"
            matches.append(match)
    return matches


def one_to_one_geometry_matches(branches_a, branches_b, view_a, view_b, dist_a, dist_b):
    score_table: List[List[Dict[str, object]]] = []
    cost = np.zeros((len(branches_a), len(branches_b)), dtype=np.float64)
    for i, branch_a in enumerate(branches_a):
        row = []
        for j, branch_b in enumerate(branches_b):
            score = best_pair_score(branch_a, branch_b, view_a, view_b, dist_a, dist_b)
            row.append(score)
            cost[i, j] = score["objective"]
        score_table.append(row)

    row_idx, col_idx = linear_sum_assignment(cost)
    matches = []
    for i, j in zip(row_idx, col_idx):
        score = score_table[i][j]
        status = classify_match(score, branches_a[i])
        status, status_reason = apply_safety_gates(status, score, branches_a[i])
        matches.append(
            {
                "branch_a": branches_a[i],
                "branch_b": branches_b[j],
                "score": score,
                "status": status,
                "status_reason": status_reason,
            }
        )
    return matches


def write_anchor_outputs(path: Path, anchors_a: Dict[str, object], anchors_b: Dict[str, object]):
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"view_a": anchors_a, "view_b": anchors_b}, f, indent=2)


def write_landmark_debug_outputs(out: Path, anchors_a: Dict[str, object], anchors_b: Dict[str, object]):
    payloads = {
        "ostium_candidates_view_a.json": anchors_a.get("ostium_candidates", []),
        "ostium_candidates_view_b.json": anchors_b.get("ostium_candidates", []),
        "lm_trace_view_a.json": anchors_a.get("lm_trace_candidates", []),
        "lm_trace_view_b.json": anchors_b.get("lm_trace_candidates", []),
        "lm_bifurcation_candidates_view_a.json": anchors_a.get("bifurcation_candidates", []),
        "lm_bifurcation_candidates_view_b.json": anchors_b.get("bifurcation_candidates", []),
    }
    for name, payload in payloads.items():
        with open(out / name, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)


def write_main_vessel_matches(path: Path, matches: List[Dict[str, object]]):
    rows = []
    for match in matches:
        if match.get("anatomy_role") not in {"LM", "LAD", "LCX"}:
            continue
        score = match["score"]
        rows.append(
            {
                "anatomy_role": match.get("anatomy_role", "unknown"),
                "view_a_branch_id": int(match["branch_a"]["branch_id"]),
                "view_b_branch_id": int(match["branch_b"]["branch_id"]),
                "status": match["status"],
                "status_reason": match.get("status_reason", ""),
                "ray_median_mm": score.get("ray_median_mm"),
                "ray_p90_mm": score.get("ray_p90_mm"),
                "reproj_median_px": score.get("reproj_median_px"),
                "anatomy_probability_a": score.get("anatomy_probability_a", ""),
                "anatomy_probability_b": score.get("anatomy_probability_b", ""),
                "anatomy_objective": score.get("anatomy_objective", score.get("objective")),
            }
        )
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"main_vessel_matches": rows}, f, indent=2)


def optimize_branch(points_3d: np.ndarray, samples_a: np.ndarray, samples_b: np.ndarray, view_a, view_b) -> np.ndarray:
    if len(points_3d) < 4:
        return points_3d
    target_a = np.column_stack([samples_a[:, 1], samples_a[:, 0]])
    target_b = np.column_stack([samples_b[:, 1], samples_b[:, 0]])
    initial = points_3d.copy()
    scale = max(float(np.ptp(initial[:, 0])), float(np.ptp(initial[:, 1])), float(np.ptp(initial[:, 2])), 1.0)

    def residual(flat):
        pts = flat.reshape((-1, 3))
        proj_a = project_points(pts, view_a)
        proj_b = project_points(pts, view_b)
        data = np.r_[
            (proj_a - target_a).ravel() * 0.055,
            (proj_b - target_b).ravel() * 0.055,
        ]
        smooth = []
        if len(pts) >= 3:
            smooth = ((pts[:-2] - 2.0 * pts[1:-1] + pts[2:]) / scale * 0.9).ravel()
        anchor = ((pts - initial) / scale * 0.12).ravel()
        return np.r_[data, smooth, anchor]

    result = least_squares(residual, initial.ravel(), max_nfev=80, ftol=1e-4, xtol=1e-4, verbose=0)
    optimized = result.x.reshape((-1, 3))
    for dim in range(3):
        optimized[:, dim] = smooth_1d(optimized[:, dim], min(7, max(3, len(optimized) // 2 * 2 - 1)))
    optimized[0] = points_3d[0]
    optimized[-1] = points_3d[-1]
    return optimized


def write_centerline_json(path: Path, accepted):
    payload = []
    for item in accepted:
        branch = item["branch_a"]
        score = item["score"]
        payload.append(
            {
                "branch_id": int(branch["branch_id"]),
                "matched_view_b_branch_id": int(item["branch_b"]["branch_id"]),
                "status": item["status"],
                "points_xyz": item["optimized_points"].tolist(),
                "samples_view_a_yx": score["samples_a"].tolist(),
                "samples_view_b_yx": score["samples_b"].tolist(),
            }
        )
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"branches": payload}, f, indent=2)


def draw_match_overlay(original, mask, accepted, view, output_path: Path):
    canvas = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR) if original.ndim == 2 else original.copy()
    red = canvas.copy()
    red[mask > 0] = (35, 35, 255)
    canvas = cv2.addWeighted(red, 0.32, canvas, 0.68, 0)
    colors = {"reliable": (0, 255, 80), "usable": (0, 220, 255), "uncertain": (0, 210, 255)}
    for item in accepted:
        pts = project_points(item["optimized_points"], view)
        finite = np.isfinite(pts).all(axis=1)
        pixels = np.rint(pts[finite]).astype(int)
        if len(pixels) < 2:
            continue
        color = colors.get(item["status"], (0, 0, 255))
        for a, b in zip(pixels[:-1], pixels[1:]):
            cv2.line(canvas, tuple(a), tuple(b), color, 1, cv2.LINE_AA)
        cv2.putText(canvas, str(item["branch_a"]["branch_id"]), tuple(pixels[len(pixels) // 2]), cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)
    cv2.imwrite(str(output_path), canvas)


DEFAULT_LOCALIZATION_MODEL_PATHS = [
    ROOT / "checkpoints" / "multitask_localization_v2" / "multitask_latest.onnx",
    ROOT / "checkpoints" / "multitask_localization_v2" / "multitask_latest.pth",
    ROOT / "checkpoints" / "multitask_localization_v2" / "multitask_best.onnx",
    ROOT / "checkpoints" / "multitask_localization_v2" / "multitask_best.pth",
]


def load_localization_model(model_path: Path | None):
    """Best-effort load of the trained anatomical-localization model. Anatomy
    matching falls back to the geometric heuristic alone if it's missing."""
    candidates = [model_path] if model_path else DEFAULT_LOCALIZATION_MODEL_PATHS
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            try:
                from frame_pipeline import LocalizationModel

                return LocalizationModel(str(candidate))
            except Exception as exc:
                print(f"[warn] Could not load localization model {candidate}: {exc}")
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=ROOT / "dicom_pipeline_output_v2")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "dicom_epipolar_optimized")
    parser.add_argument("--localization-model", type=Path, default=None, help="Optional override for the anatomy-localization model checkpoint.")
    args = parser.parse_args()

    input_dir = args.input_dir
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    with open(input_dir / "pipeline_summary.json", "r", encoding="utf-8") as f:
        summary = json.load(f)
    view_a = summary["view_a"]
    view_b = summary["view_b"]

    mask_a = load_gray(input_dir / "view_a_clean_mask.png")
    mask_b = load_gray(input_dir / "view_b_clean_mask.png")
    original_a = load_gray(input_dir / "view_a_original.png")
    original_b = load_gray(input_dir / "view_b_original.png")
    clean_a, _, dt_a, branches_a = extract_graph(mask_a)
    clean_b, _, dt_b, branches_b = extract_graph(mask_b)
    dist_a = cv2.distanceTransform((clean_a == 0).astype(np.uint8), cv2.DIST_L2, 5)
    dist_b = cv2.distanceTransform((clean_b == 0).astype(np.uint8), cv2.DIST_L2, 5)

    class_map_a = class_map_b = confidence_map_a = confidence_map_b = None
    loc_model = load_localization_model(args.localization_model)
    if loc_model is not None:
        class_map_a, confidence_map_a = loc_model.predict(clahe_enhance_rgb(original_a))
        class_map_b, confidence_map_b = loc_model.predict(clahe_enhance_rgb(original_b))

    graph_a = load_json_if_exists(input_dir / "view_a_vessel_graph.json")
    graph_b = load_json_if_exists(input_dir / "view_b_vessel_graph.json")
    anchors_a = detect_anatomy_anchors(
        branches_a,
        graph_a,
        original_a,
        clean_a,
        out / "bifurcation_detection_overlay_view_a.png",
        "view_a",
        class_map=class_map_a,
        confidence_map=confidence_map_a,
    )
    anchors_b = detect_anatomy_anchors(
        branches_b,
        graph_b,
        original_b,
        clean_b,
        out / "bifurcation_detection_overlay_view_b.png",
        "view_b",
        class_map=class_map_b,
        confidence_map=confidence_map_b,
    )
    write_anchor_outputs(out / "anatomy_anchor_candidates.json", anchors_a, anchors_b)
    write_landmark_debug_outputs(out, anchors_a, anchors_b)

    matches = anatomy_aware_matches(branches_a, branches_b, view_a, view_b, dist_a, dist_b, anchors_a, anchors_b)
    mag_factor = float(view_a["source_detector_distance_mm"]) / max(float(view_a["source_patient_distance_mm"]), 1e-6)
    px_to_object_mm = float(view_a["imager_pixel_spacing_mm"][0]) / max(mag_factor, 1e-6)

    accepted = []
    report = []
    parts = []
    for match in matches:
        score = match["score"]
        row = {
            "branch_id": match["branch_a"]["branch_id"],
            "matched_view_b_branch_id": match["branch_b"]["branch_id"],
            "anatomy_role": match.get("anatomy_role", "unknown"),
            "status": match["status"],
            "status_reason": match.get("status_reason", ""),
            "objective": score["objective"],
            "anatomy_objective": score.get("anatomy_objective", score["objective"]),
            "anatomy_probability_a": score.get("anatomy_probability_a", ""),
            "anatomy_probability_b": score.get("anatomy_probability_b", ""),
            "anatomy_penalty": score.get("anatomy_penalty", ""),
            "short_main_penalty": score.get("short_main_penalty", ""),
            "ray_median_mm": score["ray_median_mm"],
            "ray_p90_mm": score["ray_p90_mm"],
            "reproj_median_px": score["reproj_median_px"],
            "mask_a_median_px": score["mask_a_median_px"],
            "mask_b_median_px": score["mask_b_median_px"],
            "inside_a_pct": score["inside_a_pct"],
            "inside_b_pct": score["inside_b_pct"],
            "reverse_a": score["reverse_a"],
            "reverse_b": score["reverse_b"],
        }
        report.append(row)
        if match["status"] not in {"reliable", "usable", "uncertain"}:
            continue
        optimized = optimize_branch(score["points_3d"], score["samples_a"], score["samples_b"], view_a, view_b)
        match["optimized_points"] = optimized
        accepted.append(match)
        radii = sample_radius(dt_a, score["samples_a"], px_to_object_mm)
        vertices, faces = variable_tube(optimized, radii, segments=16)
        parts.append((f"branch_{match['branch_a']['branch_id']:02d}_{match['status']}", match["status"], vertices, faces))

    if parts:
        write_obj(out / "epipolar_optimized_centerline_tree.obj", parts)
    write_centerline_json(out / "epipolar_optimized_centerlines.json", accepted)
    with open(out / "epipolar_branch_match_report.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(report[0].keys()))
        writer.writeheader()
        writer.writerows(report)
    with open(out / "anatomy_aware_match_report.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(report[0].keys()))
        writer.writeheader()
        writer.writerows(report)
    write_main_vessel_matches(out / "main_vessel_matches.json", matches)

    draw_match_overlay(original_a, clean_a, accepted, view_a, out / "view_a_epipolar_reprojection.png")
    draw_match_overlay(original_b, clean_b, accepted, view_b, out / "view_b_epipolar_reprojection.png")

    status_counts = {}
    for row in report:
        status_counts[row["status"]] = status_counts.get(row["status"], 0) + 1
    output_summary = {
        "input_dir": str(input_dir),
        "output_dir": str(out),
        "num_view_a_branches": len(branches_a),
        "num_view_b_branches": len(branches_b),
        "num_matches": len(matches),
        "num_accepted": sum(1 for item in accepted if item["status"] in {"reliable", "usable"}),
        "num_uncertain_visualized": sum(1 for item in accepted if item["status"] == "uncertain"),
        "status_counts": status_counts,
        "outputs": {
            "obj": str(out / "epipolar_optimized_centerline_tree.obj"),
            "centerlines": str(out / "epipolar_optimized_centerlines.json"),
            "match_report": str(out / "epipolar_branch_match_report.csv"),
            "anatomy_anchor_candidates": str(out / "anatomy_anchor_candidates.json"),
            "main_vessel_matches": str(out / "main_vessel_matches.json"),
            "anatomy_aware_match_report": str(out / "anatomy_aware_match_report.csv"),
            "view_a_bifurcation_overlay": str(out / "bifurcation_detection_overlay_view_a.png"),
            "view_b_bifurcation_overlay": str(out / "bifurcation_detection_overlay_view_b.png"),
            "view_a_overlay": str(out / "view_a_epipolar_reprojection.png"),
            "view_b_overlay": str(out / "view_b_epipolar_reprojection.png"),
        },
    }
    with open(out / "epipolar_optimized_summary.json", "w", encoding="utf-8") as f:
        json.dump(output_summary, f, indent=2)
    print(json.dumps(output_summary, indent=2))


if __name__ == "__main__":
    main()
