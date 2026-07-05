import os
import glob
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import cv2
import numpy as np
import matplotlib.pyplot as plt

from skimage.morphology import skeletonize
from scipy.ndimage import distance_transform_edt
from scipy.interpolate import splprep, splev


# =========================
# Config
# =========================
@dataclass
class QCAConfig:
    min_component_pixels: int = 200          # remove tiny mask components
    close_kernel: int = 3                    # morphological closing
    close_iters: int = 1                     # 1 iter prevents bridging nearby-but-separate vessels
    open_kernel: int = 3                     # morphological opening
    open_iters: int = 1

    prune_spur_len: int = 6                  # prune skeleton spurs shorter than this (px)
    smooth_win: int = 9                      # diameter smoothing window (odd); capped adaptively per branch

    lesion_alpha: float = 0.82               # boundary threshold: lesion region where d < alpha*RVD
    ref_win_prox: int = 40                   # points before lesion for RVD estimate
    ref_win_dist: int = 40                   # points after lesion for RVD estimate
    min_lesion_points: int = 3               # minimum centerline span to count as a lesion

    # Severity thresholds — four-tier scheme per the JACIT/ARC-2 hierarchical
    # consensus (QCA_CAL.md): repeat revascularization is clinically indicated
    # for (1) DS > 50% WITH recurrent symptoms or a positive functional test,
    # or (2) DS > 70% regardless of other criteria. Below 50% the split at 30%
    # is a non-obstructive/mild-vs-moderate convenience boundary (QCA_CAL.md is
    # silent below 50%; ACC/AHA/ESC practice commonly uses 30% here):
    #   Mild        : DS  < 30%  — non-obstructive
    #   Moderate    : 30% ≤ DS < 50%  — watch, not independently actionable
    #   Significant : 50% ≤ DS < 70%  — actionable if symptomatic / positive functional test
    #   Severe      : DS ≥ 70%        — intervention indicated regardless of symptoms
    severe_threshold: float = 70.0
    significant_threshold: float = 50.0
    moderate_threshold: float = 30.0

    min_branch_len: int = 15                 # minimum branch length (skeleton px) to analyze

    px_to_mm: Optional[float] = None         # set if you have calibration (mm per pixel)
    # JACIT-recommended calibration range for the standard catheter/isocenter
    # technique (QCA_CAL.md); used to sanity-check a supplied px_to_mm.
    px_to_mm_min: float = 0.18
    px_to_mm_max: float = 0.22

    total_occlusion_px_threshold: float = 0.5  # DT at original branch pt below which total occlusion declared
    stent_mode: bool = False                    # use stent-specific (5mm edge subsegment) reference windows
    stent_edge_mm: float = 5.0                  # DES analysis: proximal/distal edge subsegment length

    huo_kassab_exponent: float = 7.0 / 3.0      # Huo-Kassab (Murray's law) bifurcation exponent


def validate_calibration(cfg: QCAConfig) -> Optional[float]:
    """
    Validate cfg.px_to_mm is within the JACIT-recommended range (0.18-0.22
    mm/px for the standard catheter/isocenter technique). Returns the
    validated value (or None if no calibration is set). Raises ValueError if
    a calibration value is set but implausible -- catches a wrong catheter
    measurement or a copy-paste config error before it silently produces
    wrong mm measurements.
    """
    if cfg.px_to_mm is None:
        return None
    if not (cfg.px_to_mm_min <= cfg.px_to_mm <= cfg.px_to_mm_max):
        raise ValueError(
            f"px_to_mm={cfg.px_to_mm:.4f} is outside the JACIT-recommended range "
            f"[{cfg.px_to_mm_min}, {cfg.px_to_mm_max}] mm/pixel."
        )
    return cfg.px_to_mm


# =========================
# Helpers: mask cleanup
# =========================
def to_binary_mask(mask_img: np.ndarray) -> np.ndarray:
    """Ensure binary mask uint8 with values 0/255."""
    if mask_img.ndim == 3:
        mask_img = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(mask_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return bw

def keep_significant_components(bw: np.ndarray, min_pixels: int) -> np.ndarray:
    """Keep ALL connected components above min_pixels (not just the largest)."""
    num, labels, stats, _ = cv2.connectedComponentsWithStats((bw > 0).astype(np.uint8), connectivity=8)
    if num <= 1:
        return bw

    areas = stats[1:, cv2.CC_STAT_AREA]
    out = np.zeros_like(bw)
    any_valid = False
    for i, a in enumerate(areas):
        if a >= min_pixels:
            out[labels == (i + 1)] = 255
            any_valid = True

    if not any_valid:
        largest = 1 + int(np.argmax(areas))
        out = (labels == largest).astype(np.uint8) * 255
    return out

def morph_cleanup(bw: np.ndarray, cfg: QCAConfig) -> np.ndarray:
    """Close gaps, remove specks, keep all significant vessel components.

    Bridge prevention: any pixel that morphological closing adds between two
    previously-separate components is detected and removed before opening.
    This stops nearby-but-distinct vessel branches from being merged into a
    single connected region whose skeleton then contains a thin bridge —
    which would otherwise be measured as a severe stenosis.
    """
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg.close_kernel, cfg.close_kernel))
    k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg.open_kernel,  cfg.open_kernel))
    dilate_k = np.ones((3, 3), dtype=np.uint8)   # 8-connected dilation kernel

    # Snapshot the original binary mask and its connected components
    bw_bin = (bw > 0).astype(np.uint8)
    n_comp, labels_orig = cv2.connectedComponents(bw_bin, connectivity=8)
    # n_comp includes label 0 (background), so n_comp > 2 means ≥2 foreground pieces

    # Apply morphological closing (fills holes and smooths edges within vessels)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k_close, iterations=cfg.close_iters)

    # --- Bridge-prevention pass ---
    # Pixels added by closing that are adjacent to ≥2 different original components
    # are bridge pixels — they connect separate vessels and must be removed.
    if n_comp > 2:
        new_px = (bw > 0) & (bw_bin == 0)     # pixels that closing added
        if np.any(new_px):
            # For each original foreground component, dilate it by 1 pixel.
            # A new pixel "touched" by ≥2 components is a bridge.
            bridge_count = np.zeros(bw.shape, dtype=np.int32)
            for i in range(1, n_comp):
                comp_dilated = cv2.dilate((labels_orig == i).astype(np.uint8), dilate_k)
                bridge_count += comp_dilated.astype(np.int32)
            bw[new_px & (bridge_count >= 2)] = 0   # excise bridge pixels

    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, k_open, iterations=cfg.open_iters)
    bw = keep_significant_components(bw, cfg.min_component_pixels)
    return bw


# =========================
# Helpers: skeleton graph + longest path
# =========================
_NEI8 = [(-1, -1), (-1, 0), (-1, 1),
         ( 0, -1),          ( 0, 1),
         ( 1, -1), ( 1, 0), ( 1, 1)]

def skeleton_endpoints(skel: np.ndarray) -> List[Tuple[int, int]]:
    """Endpoints = skeleton pixels with exactly 1 neighbor (8-connect) using fast convolution."""
    sk = (skel > 0).astype(np.uint8)
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]], dtype=np.uint8)
    neighbor_count = cv2.filter2D(sk, -1, kernel, borderType=cv2.BORDER_CONSTANT)
    endpoints_mask = (sk == 1) & (neighbor_count == 1)
    ys, xs = np.where(endpoints_mask)
    return list(zip(ys, xs))

def skeleton_junctions(skel: np.ndarray) -> List[Tuple[int, int]]:
    """Junction points = skeleton pixels with >= 3 neighbors using fast convolution."""
    sk = (skel > 0).astype(np.uint8)
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]], dtype=np.uint8)
    neighbor_count = cv2.filter2D(sk, -1, kernel, borderType=cv2.BORDER_CONSTANT)
    junction_mask = (sk == 1) & (neighbor_count >= 3)
    ys, xs = np.where(junction_mask)
    return list(zip(ys, xs))

def build_adjacency(skel: np.ndarray) -> Dict[Tuple[int,int], List[Tuple[int,int]]]:
    """Fast adjacency graph builder using vectorized shifts."""
    sk = (skel > 0)
    ys, xs = np.where(sk)

    H, W = sk.shape
    idx_map = np.full((H, W), -1, dtype=np.int32)
    idx_map[ys, xs] = np.arange(len(ys))

    pts = list(zip(ys, xs))
    adj: Dict[Tuple[int,int], List[Tuple[int,int]]] = {p: [] for p in pts}

    for dy, dx in _NEI8:
        ny = ys + dy
        nx = xs + dx
        valid = (ny >= 0) & (ny < H) & (nx >= 0) & (nx < W)
        v_ys = ys[valid]
        v_xs = xs[valid]
        v_ny = ny[valid]
        v_nx = nx[valid]
        is_skel = sk[v_ny, v_nx]
        for curr_y, curr_x, n_y, n_x in zip(v_ys[is_skel], v_xs[is_skel], v_ny[is_skel], v_nx[is_skel]):
            adj[(curr_y, curr_x)].append((n_y, n_x))

    return adj

def bfs_farthest(adj: Dict[Tuple[int,int], List[Tuple[int,int]]],
                 start: Tuple[int,int]) -> Tuple[Tuple[int,int], Dict[Tuple[int,int], Tuple[int,int]], Dict[Tuple[int,int], int]]:
    """Unweighted BFS on skeleton graph: returns farthest node, parent map, distance map."""
    from collections import deque
    q = deque([start])
    parent = {start: None}
    dist = {start: 0}
    far = start
    while q:
        u = q.popleft()
        if dist[u] > dist[far]:
            far = u
        for v in adj.get(u, []):
            if v not in dist:
                dist[v] = dist[u] + 1
                parent[v] = u
                q.append(v)
    return far, parent, dist

def extract_path(parent: Dict[Tuple[int,int], Optional[Tuple[int,int]]],
                 end: Tuple[int,int]) -> List[Tuple[int,int]]:
    path = []
    cur = end
    while cur is not None:
        path.append(cur)
        cur = parent.get(cur, None)
    path.reverse()
    return path

def prune_spurs(skel: np.ndarray, max_len: int) -> np.ndarray:
    """
    Iteratively remove short spur branches:
    - detect endpoints
    - walk from endpoint until junction (deg>=3) or another endpoint
    - if walked length <= max_len -> delete spur pixels
    """
    sk = skel.copy().astype(np.uint8)
    changed = True
    while changed:
        changed = False
        adj = build_adjacency(sk)
        endpoints = skeleton_endpoints(sk)

        sset = set(adj.keys())
        for ep in endpoints:
            if ep not in sset:
                continue
            path = [ep]
            prev = None
            cur = ep
            while True:
                nbrs = adj.get(cur, [])
                deg = len(nbrs)
                if deg >= 3 and cur != ep:
                    break
                next_nodes = [n for n in nbrs if n != prev]
                if not next_nodes:
                    break
                nxt = next_nodes[0]
                path.append(nxt)
                prev, cur = cur, nxt
                if len(path) > max_len:
                    path = []
                    break

            if path and 2 <= len(path) <= max_len:
                for (y, x) in path[:-1]:
                    sk[y, x] = 0
                changed = True
    return sk

def extract_all_branches(skel: np.ndarray, min_branch_len: int) -> List[List[Tuple[int, int]]]:
    """
    Decompose skeleton into individual branch segments.
    A branch runs from endpoint/junction to endpoint/junction.
    Returns list of ordered (y,x) paths, one per branch.
    """
    adj = build_adjacency(skel)
    if not adj:
        return []

    degree = {}
    for node, nbrs in adj.items():
        degree[node] = len(nbrs)

    visited_edges = set()
    branches = []

    start_nodes = [n for n, d in degree.items() if d == 1 or d >= 3]
    if not start_nodes:
        start_nodes = [list(adj.keys())[0]]

    for start in start_nodes:
        for nbr in adj[start]:
            edge_key = frozenset([start, nbr])
            if edge_key in visited_edges:
                continue

            path = [start, nbr]
            visited_edges.add(edge_key)
            prev, cur = start, nbr

            while degree.get(cur, 0) == 2:
                next_nodes = [n for n in adj[cur] if n != prev]
                if not next_nodes:
                    break
                nxt = next_nodes[0]
                ek = frozenset([cur, nxt])
                if ek in visited_edges:
                    break
                visited_edges.add(ek)
                path.append(nxt)
                prev, cur = cur, nxt

            # Junction-to-junction connectors are always kept even if shorter
            # than min_branch_len -- dropping them leaves a visual/measurement
            # gap in the vessel tree right at bifurcation clusters, since a
            # short connecting segment between two junctions is still part of
            # the vessel, not a spur to be pruned.
            start_is_junction = degree.get(start, 0) >= 3
            end_is_junction = degree.get(cur, 0) >= 3
            if len(path) >= min_branch_len or (start_is_junction and end_is_junction):
                branches.append(path)

    if not branches:
        nodes = list(adj.keys())
        if len(nodes) >= 10:
            eps = skeleton_endpoints(skel)
            s = eps[0] if eps else nodes[0]
            a, _, _ = bfs_farthest(adj, s)
            b, parent_b, _ = bfs_farthest(adj, a)
            path = extract_path(parent_b, b)
            if len(path) >= min_branch_len:
                branches.append(path)

    return branches

def ordered_centerline_from_mask(bw_mask: np.ndarray, cfg: QCAConfig) -> List[Tuple[int,int]]:
    """
    Skeletonize -> prune spurs -> find 'longest' path using 2x BFS (tree-diameter heuristic).
    Optimized: Crops the mask to the bounding box of the vessel before skeletonization.
    Returns ordered list of (y,x) points.
    """
    ys, xs = np.where(bw_mask > 0)
    if len(ys) == 0:
        raise RuntimeError("Mask is empty.")

    y_min, y_max = np.min(ys), np.max(ys)
    x_min, x_max = np.min(xs), np.max(xs)

    pad = 10
    H, W = bw_mask.shape
    y1, y2 = max(0, y_min - pad), min(H, y_max + pad + 1)
    x1, x2 = max(0, x_min - pad), min(W, x_max + pad + 1)

    cropped_mask = bw_mask[y1:y2, x1:x2]

    cr_skel = skeletonize((cropped_mask > 0)).astype(np.uint8)
    cr_skel = prune_spurs(cr_skel, cfg.prune_spur_len)

    skel = np.zeros_like(bw_mask)
    skel[y1:y2, x1:x2] = cr_skel

    adj = build_adjacency(skel)
    nodes = list(adj.keys())
    if len(nodes) < 10:
        raise RuntimeError("Skeleton too small. Mask may be empty or too thin.")

    eps = skeleton_endpoints(skel)
    start = eps[0] if eps else nodes[0]

    a, _, _ = bfs_farthest(adj, start)
    b, parent_b, _ = bfs_farthest(adj, a)

    path = extract_path(parent_b, b)
    if len(path) < 10:
        raise RuntimeError("Failed to extract a valid centerline path.")
    return path

def get_skeleton_from_mask(bw_mask: np.ndarray, cfg: QCAConfig) -> np.ndarray:
    """Skeletonize and prune spurs, returning the cleaned skeleton image."""
    skel = skeletonize((bw_mask > 0)).astype(np.uint8)
    skel = prune_spurs(skel, cfg.prune_spur_len)
    return skel


# =========================
# QCA: core signal helpers
# =========================
def smooth_1d(x: np.ndarray, win: int) -> np.ndarray:
    if win <= 1:
        return x
    if win % 2 == 0:
        win += 1
    pad = win // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    kernel = np.ones(win, dtype=np.float32) / win
    return np.convolve(xp, kernel, mode="valid")

def local_minima_indices(x: np.ndarray) -> List[int]:
    mins = []
    for i in range(1, len(x) - 1):
        if x[i] <= x[i-1] and x[i] < x[i+1]:
            mins.append(i)
    return mins

def arc_length(centerline: List[Tuple[int,int]], L: int, R: int) -> float:
    """Calculates arc length of a centerline segment using vectorized numpy ops."""
    if R <= L:
        return 0.0
    pts = np.array(centerline[L:R+1])
    diffs = np.diff(pts, axis=0)
    distances = np.linalg.norm(diffs, axis=1)
    return float(np.sum(distances))


# =========================
# V6 Enhancement Functions
# =========================

def _tapered_rvd(d_s: np.ndarray, L: int, R: int, m: int,
                 prox_win: int, dist_win: int) -> float:
    """
    Compute tapered (position-interpolated) RVD from proximal and distal
    reference windows. Handles tapering vessels more accurately than flat average.
    """
    N = len(d_s)
    pL = max(0, L - prox_win)
    pR = max(0, L - 1)
    dL = min(N - 1, R + 1)
    dR = min(N - 1, R + dist_win)

    prox_pts = d_s[pL:pR + 1]
    dist_pts = d_s[dL:dR + 1]

    RVD_prox = float(np.percentile(prox_pts, 80)) if len(prox_pts) >= 3 else None
    RVD_dist = float(np.percentile(dist_pts, 80)) if len(dist_pts) >= 3 else None

    if RVD_prox is None and RVD_dist is None:
        return 0.0
    if RVD_prox is None:
        return RVD_dist
    if RVD_dist is None:
        return RVD_prox

    span = (R - L) if R > L else 1
    t = max(0.0, min(1.0, (m - L) / span))
    return (1.0 - t) * RVD_prox + t * RVD_dist


def smooth_centerline_spline(branch: List[Tuple[int, int]],
                              smoothing: float = 3.0,
                              num_pts: Optional[int] = None) -> List[Tuple[int, int]]:
    """
    Fit a parametric cubic spline through branch pixel coordinates and
    re-sample at the same number of points. Reduces jagged centerline
    artifacts that inflate diameter noise.
    Falls back to original branch if spline fitting fails (< 6 pts).
    """
    if len(branch) < 6:
        return branch
    ys = np.array([p[0] for p in branch], dtype=float)
    xs = np.array([p[1] for p in branch], dtype=float)
    try:
        tck, u = splprep([xs, ys], s=smoothing * len(branch), k=3)
        n_out = num_pts or len(branch)
        u_new = np.linspace(0, 1, n_out)
        xs_s, ys_s = splev(u_new, tck)
        return [(int(round(y)), int(round(x))) for x, y in zip(xs_s, ys_s)]
    except Exception:
        return branch


def _tangent_at(branch_arr: np.ndarray, idx: int) -> np.ndarray:
    """Unit tangent vector at index idx using central differences (window ±3)."""
    n = len(branch_arr)
    lo = max(0, idx - 3)
    hi = min(n - 1, idx + 3)
    vec = branch_arr[hi] - branch_arr[lo]
    norm = np.linalg.norm(vec)
    if norm < 1e-6:
        return np.array([1.0, 0.0])
    return vec / norm


def orthogonal_diameter(mask: np.ndarray, branch_arr: np.ndarray,
                        idx: int, max_radius: int = 40) -> float:
    """
    Cast two rays perpendicular to the centerline tangent and measure where
    they exit the vessel mask. Returns diameter in pixels.

    Handles spline-smoothed centerlines that may drift slightly off the binary
    mask by snapping the starting pixel to the nearest vessel pixel before
    casting rays. Without this, an off-mask start gives MLD = 1 (both rays
    exit immediately), which falsely triggers total-occlusion detection.
    """
    cy, cx = branch_arr[idx]
    cy_i, cx_i = int(round(cy)), int(round(cx))

    # Snap starting pixel to nearest vessel pixel (handles spline drift off mask)
    if not (0 <= cy_i < mask.shape[0] and 0 <= cx_i < mask.shape[1]
            and mask[cy_i, cx_i] > 0):
        found = False
        for r in range(1, 8):
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    if abs(dy) != r and abs(dx) != r:
                        continue  # only check the perimeter of each ring
                    ny2, nx2 = cy_i + dy, cx_i + dx
                    if (0 <= ny2 < mask.shape[0] and 0 <= nx2 < mask.shape[1]
                            and mask[ny2, nx2] > 0):
                        cy_i, cx_i = ny2, nx2
                        found = True
                        break
                if found:
                    break
            if found:
                break

    tang = _tangent_at(branch_arr, idx)   # (dy, dx) in (y,x) space
    perp_y, perp_x = -tang[1], tang[0]   # perpendicular direction

    def cast(dy: float, dx: float) -> int:
        for r in range(1, max_radius + 1):
            iy = int(round(cy_i + dy * r))
            ix = int(round(cx_i + dx * r))
            if iy < 0 or iy >= mask.shape[0] or ix < 0 or ix >= mask.shape[1]:
                return r - 1
            if mask[iy, ix] == 0:
                return r - 1
        return max_radius

    r1 = cast(perp_y, perp_x)
    r2 = cast(-perp_y, -perp_x)
    return float(r1 + r2 + 1)


def refine_diameter_subpixel(angio_gray: np.ndarray, mask: np.ndarray,
                              branch_arr: np.ndarray, idx: int,
                              rough_radius: float, search_extra: int = 4) -> float:
    """
    Refine diameter at a centerline point by sampling the angiogram intensity
    gradient along the perpendicular direction. Provides sub-pixel accuracy
    at the vessel wall. Falls back to rough_radius if angio_gray is None.
    """
    if angio_gray is None:
        return rough_radius

    cy, cx = branch_arr[idx]
    tang = _tangent_at(branch_arr, idx)
    ny, nx = -tang[1], tang[0]
    half = int(math.ceil(rough_radius / 2)) + search_extra

    def sample_profile(dy: float, dx: float) -> np.ndarray:
        vals = []
        for r in range(half + 1):
            iy = int(round(cy + dy * r))
            ix = int(round(cx + dx * r))
            if 0 <= iy < angio_gray.shape[0] and 0 <= ix < angio_gray.shape[1]:
                vals.append(float(angio_gray[iy, ix]))
            else:
                vals.append(0.0)
        return np.array(vals)

    def edge_pos(profile: np.ndarray) -> float:
        g = np.abs(np.gradient(profile))
        return float(np.argmax(g))

    p1 = sample_profile(ny, nx)
    p2 = sample_profile(-ny, -nx)
    return edge_pos(p1) + edge_pos(p2) + 1.0


def curvature_derivative_gate(d_s: np.ndarray, minima: List[int],
                               min_slope_ratio: float = 0.05) -> List[int]:
    """
    Keep only minima with a clear negative slope before and positive recovery
    after. Removes flat-profile false positives where no real stenosis exists.
    """
    grad = np.gradient(d_s.astype(float))
    gated = []
    for m in minima:
        lo = max(0, m - 5)
        pre_slope = float(np.mean(grad[lo:m])) if m > lo else -1.0
        hi = min(len(d_s) - 1, m + 5)
        post_slope = float(np.mean(grad[m:hi])) if hi > m else 1.0
        ref_d = float(d_s[m]) if d_s[m] > 0 else 1.0
        threshold = min_slope_ratio * ref_d
        if pre_slope < -threshold and post_slope > threshold:
            gated.append(m)
    return gated


def depth_gate(d_s: np.ndarray, minima: List[int],
               ref_win: int = 40, min_ds_percent: float = 15.0) -> List[int]:
    """
    Keep only minima that represent a clinically meaningful diameter reduction.

    The curvature/slope gate fails for gradual stenoses: a 50% stenosis that
    develops over 20 centerline points has a per-step slope that is often below
    the slope threshold, so it gets rejected even though the total drop is large.

    This gate uses the clinical QCA definition directly:
        DS% = (reference_diameter - MLD) / reference_diameter × 100
    and keeps a minimum only when the DT-profile drop is >= min_ds_percent.
    Any real stenosis — focal or diffuse, sharp or gradual — passes equally.
    """
    N = len(d_s)
    gated = []
    for m in minima:
        left0 = max(0, m - ref_win)
        right0 = min(N - 1, m + ref_win)
        # Reference = healthy diameter adjacent to the dip (exclude the dip itself)
        ref_pts = list(d_s[left0:m]) + list(d_s[m + 1:right0 + 1])
        if len(ref_pts) < 3:
            continue
        local_ref = float(np.percentile(ref_pts, 80))
        if local_ref <= 1e-6:
            continue
        ds_pct = (1.0 - float(d_s[m]) / local_ref) * 100.0
        if ds_pct >= min_ds_percent:
            gated.append(m)
    return gated


def calibrate_from_catheter(mask_catheter: np.ndarray,
                             known_diameter_mm: float = 2.0) -> Optional[float]:
    """
    Estimate px_to_mm by measuring catheter shaft width across multiple
    horizontal cross-sections. Returns mm/px or None if measurement fails.
    A 6F guiding catheter has outer diameter ~2.0 mm.
    """
    ys, _ = np.where(mask_catheter > 0)
    if len(ys) == 0:
        return None
    widths = []
    for row in range(mask_catheter.shape[0]):
        cols = np.where(mask_catheter[row] > 0)[0]
        if len(cols) > 3:
            widths.append(int(cols[-1]) - int(cols[0]) + 1)
    if len(widths) < 5:
        return None
    median_w = float(np.median(widths))
    return known_diameter_mm / median_w if median_w > 0 else None


def classify_severity(ds_percent: float, cfg: QCAConfig) -> str:
    """
    Four-tier JACIT/ARC-2 severity classification (see QCAConfig's threshold
    doc-comment for the clinical rationale):
      SEVERE       — DS >= severe_threshold        (unconditional)
      SIGNIFICANT  — DS >= significant_threshold   (actionable with symptoms / +ve functional test)
      MODERATE     — DS >= moderate_threshold
      MILD         — below moderate_threshold
    """
    if ds_percent >= cfg.severe_threshold:
        return "SEVERE"
    if ds_percent >= cfg.significant_threshold:
        return "SIGNIFICANT"
    if ds_percent >= cfg.moderate_threshold:
        return "MODERATE"
    return "MILD"


def stent_reference_windows(branch: List[Tuple[int, int]], L: int, R: int,
                            cfg: QCAConfig) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """
    DES (drug-eluting stent) analysis mode: use fixed edge subsegments
    (cfg.stent_edge_mm, default 5mm) on the proximal and distal ends of the
    stented/lesion range to define the reference diameter, per the JACIT
    standard DES algorithm (QCA_CAL.md), instead of the general tapered-RVD
    windows. Returns (prox_slice, dist_slice) as (start, end) index pairs.
    """
    if cfg.px_to_mm and cfg.px_to_mm > 0:
        edge_px = int(round(cfg.stent_edge_mm / cfg.px_to_mm))
    else:
        edge_px = 25   # fallback ~5mm at the JACIT-standard 0.20 mm/px

    N = len(branch)
    prox_start = max(0, L - edge_px)
    prox_end = max(0, L - 1)
    dist_start = min(N - 1, R + 1)
    dist_end = min(N - 1, R + edge_px)
    return (prox_start, prox_end), (dist_start, dist_end)


# =========================
# QCA: diameter profile + lesion metrics (V6-enhanced)
# =========================
def detect_lesions_on_branch(branch: List[Tuple[int,int]], dt: np.ndarray,
                              cfg: QCAConfig,
                              mask: Optional[np.ndarray] = None,
                              angio_gray: Optional[np.ndarray] = None) -> List[dict]:
    """
    Run lesion detection on a single branch centerline with V6 enhancements:
      - Spline-smoothed centerline to reduce jagged artifacts
      - Curvature-derivative gating to reject flat-profile false positives
      - Orthogonal boundary diameter at lesion minima (more accurate than DT)
      - Sub-pixel gradient edge refinement when angio_gray is provided
      - Tapered RVD for accurate reference in tapering vessels
      - Confidence score per lesion
    """
    # Spline-smooth the centerline — used for tangent direction only.
    # The DT profile is always built from the original branch (skeleton pixels
    # guaranteed to be inside the vessel mask), preventing artificial diameter
    # dips where the spline drifts off-mask.
    branch_s = smooth_centerline_spline(branch, smoothing=3.0)
    N = len(branch)                        # both branch and branch_s have same length
    branch_arr = np.array(branch_s)        # shape (N, 2) = (y, x), used for tangents

    # DT profile on ORIGINAL branch points (always inside the mask).
    # Adaptive smooth_win: cap at N//3 so short branches are not over-smoothed
    # (e.g., a 15-pt branch with win=11 would destroy all spatial detail).
    adapt_win = max(3, min(cfg.smooth_win, N // 3 if N >= 9 else 3))
    if adapt_win % 2 == 0:
        adapt_win += 1
    d_raw = np.array([2.0 * dt[y, x] for y, x in branch], dtype=np.float32)
    d_s = smooth_1d(d_raw, adapt_win)

    # Depth gate: keep only minima that represent a meaningful diameter drop.
    # This directly implements the QCA clinical definition of stenosis
    # (DS% = (RVD - MLD)/RVD) and works for both focal and gradual stenoses,
    # unlike slope-based gates which miss stenoses that narrow gradually.
    minima_raw = local_minima_indices(d_s)
    minima = depth_gate(d_s, minima_raw,
                        ref_win=cfg.ref_win_prox, min_ds_percent=20.0)

    lesions = []

    # Branch-level median diameter (from raw, unsmoothed DT).
    # Used to exclude catheter / aortic segments from the reference pool.
    # The catheter is far wider than any coronary vessel — its DT is typically
    # 30–80 px while coronary vessels are 3–20 px. By capping reference
    # candidates at 2× the branch median, we prevent the catheter from
    # inflating the reference diameter and falsely creating a large DS%.
    branch_median_dt = float(np.median(d_raw))
    ref_cap = branch_median_dt * 2.0        # reference ceiling in smoothed-diameter space

    for m in minima:
        left0 = max(0, m - cfg.ref_win_prox)
        right0 = min(N - 1, m + cfg.ref_win_dist)
        prox_avail = m - left0
        dist_avail = right0 - m

        # Reference diameter: healthy vessel on EITHER side of the lesion.
        # Values above ref_cap are catheter / aortic territory and are excluded.
        # Per clinical consensus, reference = adjacent healthy-segment diameter.
        ref_candidates: List[float] = []
        if prox_avail >= 3:
            ref_candidates.extend([v for v in d_s[left0:m] if v <= ref_cap])
        if dist_avail >= 3:
            ref_candidates.extend([v for v in d_s[m + 1:right0 + 1] if v <= ref_cap])

        if len(ref_candidates) >= 5:
            local_ref = float(np.percentile(ref_candidates, 80))
        elif len(ref_candidates) > 0:
            local_ref = float(np.median(ref_candidates))
        else:
            # No adjacent reference at all → cannot establish a reliable baseline
            continue

        if local_ref <= 1e-6:
            continue

        thr = cfg.lesion_alpha * local_ref
        L, R = m, m
        while L > 0 and d_s[L] < thr:
            L -= 1
        while R < N - 1 and d_s[R] < thr:
            R += 1

        if (R - L + 1) < cfg.min_lesion_points:
            continue

        # Skip minima within 5 pts of branch endpoints — these are junction/endpoint
        # pixels where the vessel geometry changes abruptly and measurements are
        # unreliable (skeleton degree transitions, natural tapering at tips).
        edge_margin = max(5, cfg.min_lesion_points)
        if m < edge_margin or m > N - 1 - edge_margin:
            continue

        # DT-based diameter at this exact original-branch point (no smoothing).
        # Used as a ground-truth sanity check for the orthogonal measurement.
        orig_y, orig_x = branch[m]
        dt_at_orig = float(dt[orig_y, orig_x])
        dt_mld_raw  = 2.0 * dt_at_orig   # DT diameter in pixels at the minimum

        # MLD: orthogonal rays then optional sub-pixel refinement.
        if mask is not None:
            mld_orth = orthogonal_diameter(mask, branch_arr, m)

            # Sub-pixel refinement — only accept if the result stays within ±50% of
            # the orthogonal estimate.  Large deviations indicate edge-detection
            # failures at overlapping vessels or high-curvature sections.
            if angio_gray is not None:
                refined = refine_diameter_subpixel(angio_gray, mask, branch_arr, m, mld_orth)
                if mld_orth > 0 and 0.5 <= refined / mld_orth <= 1.5:
                    mld_orth = refined

            # Cross-validate orthogonal against DT.  At tight curves and junction
            # points the spline tangent is wrong, causing the orthogonal ray to exit
            # the vessel on the inner wall too early.  The DT gives the true minimum
            # distance to the vessel wall regardless of direction.
            # If orthogonal is < 65% of the DT diameter, it exited the wrong wall.
            if dt_mld_raw > 2.0 and mld_orth < dt_mld_raw * 0.65:
                MLD    = float(d_s[m])   # fall back to smoothed DT estimate
                method = "dt"
            else:
                MLD    = float(mld_orth)
                method = "orthogonal"
        else:
            MLD    = float(d_s[m])
            method = "dt"

        # Total occlusion: only when DT at the original skeleton point confirms
        # essentially zero vessel width (< 0.5 px from wall).
        is_total_occlusion = (dt_at_orig < cfg.total_occlusion_px_threshold)

        if is_total_occlusion:
            MLD = 0.0
            percent_DS = 100.0
            RVD = local_ref          # use adjacent healthy diameter, not global ref
        else:
            if cfg.stent_mode:
                # DES mode: JACIT standard uses fixed 5mm proximal/distal edge
                # subsegments rather than the general tapered-percentile window,
                # since post-PCI/follow-up reference segments are defined
                # relative to the stented range, not the whole branch.
                (pS, pE), (dS, dE) = stent_reference_windows(branch, L, R, cfg)
                if pE - pS + 1 >= 3 and dE - dS + 1 >= 3:
                    RVD_prox = float(np.percentile(d_s[pS:pE + 1], 80))
                    RVD_dist = float(np.percentile(d_s[dS:dE + 1], 80))
                    span = (R - L) if R > L else 1
                    t = max(0.0, min(1.0, (m - L) / span))
                    RVD = (1.0 - t) * RVD_prox + t * RVD_dist
                else:
                    RVD = _tapered_rvd(d_s, L, R, m, cfg.ref_win_prox, cfg.ref_win_dist)
            else:
                RVD = _tapered_rvd(d_s, L, R, m, cfg.ref_win_prox, cfg.ref_win_dist)
            if RVD <= 1e-6:
                RVD = local_ref      # fallback to the reference window we already computed
            percent_DS = max(0.0, (1.0 - MLD / RVD) * 100.0)

        length_px = arc_length(branch, L, R)  # use original branch for geometry
        px = cfg.px_to_mm
        MLD_mm = MLD * px if px else None
        RVD_mm = RVD * px if px else None
        length_mm = length_px * px if px else None

        severity = classify_severity(percent_DS, cfg)

        # Confidence score: edge sharpness (0.5) + ref quality (0.3) + length (0.2)
        edge_sharpness = min(1.0, (local_ref - MLD) / (local_ref + 1e-6))
        ref_quality = 1.0 if (prox_avail >= 20 and dist_avail >= 20) else 0.5
        len_score = min(1.0, (R - L + 1) / 20.0)
        confidence = round(edge_sharpness * 0.5 + ref_quality * 0.3 + len_score * 0.2, 3)

        lesions.append({
            "L_idx":           int(L),
            "R_idx":           int(R),
            "min_idx":         int(m),
            "MLD_px":          MLD,
            "RVD_px":          RVD,
            "DS_percent":      percent_DS,
            "length_px":       length_px,
            "MLD_mm":          MLD_mm,
            "RVD_mm":          RVD_mm,
            "length_mm":       length_mm,
            "severity":        severity,
            "total_occlusion": is_total_occlusion,
            "min_pt":          branch[m],   # original branch pixel — always inside mask
            "branch":          branch,
            "branch_smooth":   branch_s,
            "confidence":      confidence,
            "method":          method,
            "edge_sharpness":  round(float(edge_sharpness), 3),
            "ref_quality":     round(float(ref_quality), 3),
            "len_score":       round(float(len_score), 3),
        })

    # Merge overlapping lesions on this branch
    lesions = sorted(lesions, key=lambda z: (z["L_idx"], z["R_idx"]))
    merged = []
    for les in lesions:
        if not merged:
            merged.append(les)
            continue
        prev = merged[-1]
        if les["L_idx"] <= prev["R_idx"]:
            best = les if les["DS_percent"] > prev["DS_percent"] else prev
            best = dict(best)
            best["L_idx"] = min(prev["L_idx"], les["L_idx"])
            best["R_idx"] = max(prev["R_idx"], les["R_idx"])
            merged[-1] = best
        else:
            merged.append(les)

    return merged


# =========================
# Bifurcation analysis (Huo-Kassab / Murray's law step-down check)
# =========================
# Standalone and not wired into qca_from_mask's return value on purpose: the
# rest of the pipeline (frame_pipeline.py, report_engine.py, desktop_app_qca.py,
# dicom_analysis_thread.py) all unpack qca_from_mask's 3-tuple, and changing
# that arity would force edits to every caller. Call these directly wherever
# bifurcation detail is wanted, passing the branches/dt/skel already produced
# by qca_from_mask.
def huo_kassab_check(d_mother: float, d_large: float, d_small: float,
                     exponent: float = 7.0 / 3.0) -> Tuple[float, float]:
    """
    Huo-Kassab relation (a generalization of Murray's law) for the diameter
    step-down at a coronary bifurcation, per QCA_CAL.md's bifurcation-analysis
    section:
        D_mother^n = D_large_daughter^n + D_small_daughter^n,   n = 7/3

    Returns (predicted_mother_diameter, relative_error) in the same units as
    the inputs (px or mm) -- a large relative_error flags either a genuine
    diseased/atypical bifurcation or a measurement error at that junction.
    """
    if d_large <= 0 or d_small <= 0:
        return float("nan"), float("nan")
    predicted_mother = (d_large ** exponent + d_small ** exponent) ** (1.0 / exponent)
    rel_err = abs(predicted_mother - d_mother) / max(d_mother, 1e-9)
    return predicted_mother, rel_err


def analyse_bifurcations(branches: List[List[Tuple[int, int]]],
                         dt: np.ndarray,
                         skel: np.ndarray,
                         cfg: QCAConfig) -> List[dict]:
    """
    Finds skeleton junction points, matches them to the branches meeting
    there, and applies the Huo-Kassab step-down check across each
    bifurcation (mother vessel vs. the two daughter vessels).
    Returns a list of bifurcation dicts, one per detected junction with >=3
    branches meeting there.
    """
    junctions = skeleton_junctions(skel)
    if not junctions:
        return []

    results = []
    for jy, jx in junctions:
        near = []
        for bi, br in enumerate(branches):
            for pt in br[:5] + br[-5:]:
                if abs(pt[0] - jy) <= 3 and abs(pt[1] - jx) <= 3:
                    sample = br[:10] if pt in br[:5] else br[-10:]
                    diam_samples = [2.0 * dt[y, x] for y, x in sample if dt[y, x] > 0]
                    if diam_samples:
                        near.append((bi, float(np.mean(diam_samples))))
                    break

        if len(near) < 3:
            continue  # need a mother branch + 2 daughters

        near_sorted = sorted(near, key=lambda t: t[1], reverse=True)
        mother_id, d_mother = near_sorted[0]
        large_id, d_large = near_sorted[1]
        small_id, d_small = near_sorted[2]

        pred_mother, rel_err = huo_kassab_check(d_mother, d_large, d_small, cfg.huo_kassab_exponent)
        px = cfg.px_to_mm
        results.append({
            "junction_pt": (jy, jx),
            "mother_branch_id": mother_id,
            "large_daughter_id": large_id,
            "small_daughter_id": small_id,
            "D_mother_px": d_mother,
            "D_large_px": d_large,
            "D_small_px": d_small,
            "D_mother_predicted_px": pred_mother,
            "huo_kassab_rel_error": rel_err,
            "D_mother_mm": d_mother * px if px else None,
            "D_large_mm": d_large * px if px else None,
            "D_small_mm": d_small * px if px else None,
        })
    return results


def qca_from_mask(bw_mask: np.ndarray, cfg: QCAConfig,
                  angio_gray: Optional[np.ndarray] = None):
    """
    Multi-branch QCA with V6 enhancements: skeletonize, decompose into branches,
    detect lesions on each branch with orthogonal + sub-pixel refinement.
    Optimized: Crops the mask to bounding box before expensive operations.
    Returns: all_branches, all_lesions (sorted by severity), dt
    """
    ys, xs = np.where(bw_mask > 0)
    if len(ys) == 0:
        return [], [], np.zeros_like(bw_mask, dtype=np.float32)

    y_min, y_max = np.min(ys), np.max(ys)
    x_min, x_max = np.min(xs), np.max(xs)

    pad = 10
    H, W = bw_mask.shape
    y1, y2 = max(0, y_min - pad), min(H, y_max + pad + 1)
    x1, x2 = max(0, x_min - pad), min(W, x_max + pad + 1)

    cropped_mask = bw_mask[y1:y2, x1:x2]

    cr_skel = get_skeleton_from_mask(cropped_mask, cfg)
    cr_dt = distance_transform_edt(cropped_mask > 0)

    skel = np.zeros_like(bw_mask)
    skel[y1:y2, x1:x2] = cr_skel

    dt = np.zeros_like(bw_mask, dtype=np.float32)
    dt[y1:y2, x1:x2] = cr_dt

    branches = extract_all_branches(skel, cfg.min_branch_len)

    if not branches:
        try:
            cl = ordered_centerline_from_mask(bw_mask, cfg)
            branches = [cl]
        except RuntimeError:
            return [], [], dt

    all_lesions = []
    for branch in branches:
        lesions = detect_lesions_on_branch(branch, dt, cfg,
                                           mask=bw_mask,
                                           angio_gray=angio_gray)
        all_lesions.extend(lesions)

    all_lesions = sorted(all_lesions, key=lambda z: z["DS_percent"], reverse=True)
    return branches, all_lesions, dt


# =========================
# Visualization & Reporting
# =========================
_BRANCH_COLORS = [
    (0, 255, 255),   # yellow
    (255, 200, 0),   # cyan-ish
    (200, 100, 255), # pink
    (100, 255, 100), # light green
    (255, 150, 50),  # light blue
    (100, 200, 255), # orange
]

_SEVERITY_BGR = {
    "SEVERE":      (0,  0,  255),   # Red
    "SIGNIFICANT": (0, 140, 255),   # Orange
    "MODERATE":    (0, 215, 255),   # Gold
    "MILD":        (0, 255, 255),   # Yellow
}
_SEVERITY_RADIUS = {
    "SEVERE":      3,
    "SIGNIFICANT": 2,
    "MODERATE":    1,
    "MILD":        1,
}

def draw_overlay(angio: np.ndarray, mask: np.ndarray, branches: List[List[Tuple[int,int]]],
                 lesions: List[dict]) -> np.ndarray:
    if angio.ndim == 2:
        vis = cv2.cvtColor(angio, cv2.COLOR_GRAY2BGR)
    else:
        vis = angio.copy()

    # mask outline
    contours, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, contours, -1, (0, 255, 0), 1)

    # draw each branch centerline in a different colour
    for bi, branch in enumerate(branches):
        color = _BRANCH_COLORS[bi % len(_BRANCH_COLORS)]
        for (y, x) in branch:
            vis[y, x] = color

    # Only SIGNIFICANT (>=50% DS) and SEVERE (>=70% DS) lesions are drawn --
    # i.e. the two tiers that are independently clinically actionable per the
    # ARC-2 hierarchical definition (QCA_CAL.md). MODERATE (30-49%) and MILD
    # (<30%) are still detected and written to the CSV/report but are not
    # shown on the overlay, to keep the image clinically actionable.
    # Color coding:
    #   SEVERE      (>=70%)   = Red    (circles r=3, prominent label)
    #   SIGNIFICANT (50-69%)  = Orange (circles r=2)
    display_lesions = [l for l in lesions
                       if l.get("severity", "MILD") in ("SEVERE", "SIGNIFICANT")]
    for k, les in enumerate(display_lesions):
        sev = les.get("severity", "MILD")
        color  = _SEVERITY_BGR.get(sev, _SEVERITY_BGR["MILD"])
        radius = _SEVERITY_RADIUS.get(sev, 1)
        branch = les["branch"]
        L, R = les["L_idx"], les["R_idx"]
        for i in range(L, min(R + 1, len(branch))):
            y, x = branch[i]
            cv2.circle(vis, (x, y), radius, color, -1)

        y0, x0 = les["min_pt"]
        occ    = " [OCC]" if les.get("total_occlusion") else ""
        ds_str = f"{les['DS_percent']:.1f}%"
        label  = f"#{k+1} {ds_str} {sev[:3]}{occ}"
        font_scale = 0.45 if sev == "SEVERE" else 0.38
        thickness  = 1
        cv2.putText(vis, label, (x0 + 6, y0 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)

    return vis

def save_diameter_plot(out_path: Path, branches: List[List[Tuple[int,int]]],
                      dt: np.ndarray, lesions: List[dict], cfg: QCAConfig):
    n_branches = len(branches)
    if n_branches == 0:
        return
    fig, axes = plt.subplots(n_branches, 1, figsize=(10, 3 * n_branches), squeeze=False)
    for bi, branch in enumerate(branches):
        ax = axes[bi, 0]
        d_raw = np.array([2.0 * dt[y, x] for y, x in branch], dtype=np.float32)
        d_s = smooth_1d(d_raw, cfg.smooth_win)
        ax.plot(d_raw, alpha=0.5, label="raw")
        ax.plot(d_s, label="smooth")
        # highlight lesions that belong to this branch
        for les in lesions:
            if les["branch"] is branch:
                L, R = les["L_idx"], les["R_idx"]
                ax.axvspan(L, R, alpha=0.25, color="red")
                ax.scatter([les["min_idx"]], [d_s[les["min_idx"]]], marker="v", color="red", zorder=5)
                ax.annotate(f"{les['DS_percent']:.1f}%", (les["min_idx"], d_s[les["min_idx"]]),
                            textcoords="offset points", xytext=(5, 8), fontsize=8, color="red")
        ax.set_title(f"Branch {bi+1} ({len(branch)} pts)")
        ax.set_xlabel("Centerline index")
        ax.set_ylabel("Diameter (px)")
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def build_lesion_figure(angio: np.ndarray, mask: np.ndarray, dt: np.ndarray,
                        les: dict, cfg: QCAConfig, title: str,
                        heatmap: Optional[np.ndarray] = None) -> "plt.Figure":
    """
    Builds the 3-panel explainable figure (cropped angiogram, width heatmap,
    localized diameter profile) for a single lesion. Returns an open
    matplotlib Figure — caller is responsible for saving/embedding and closing it.

    `heatmap` may be passed in precomputed (BGR colormap of the full-frame
    distance transform) to avoid recomputing it per lesion when called in a loop
    over several lesions from the same frame; otherwise it's derived from `dt`.
    """
    if heatmap is None:
        dt_norm = cv2.normalize(dt, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        heatmap = cv2.applyColorMap(dt_norm, cv2.COLORMAP_JET)

    if angio.ndim == 2:
        angio_bgr = cv2.cvtColor(angio, cv2.COLOR_GRAY2BGR)
    else:
        angio_bgr = angio.copy()

    branch = les["branch"]
    L, R = les["L_idx"], les["R_idx"]
    m_idx = les["min_idx"]

    y0, x0 = les["min_pt"]
    crop_size = 80
    H, W = angio.shape[:2]

    y1, y2 = max(0, y0 - crop_size), min(H, y0 + crop_size)
    x1, x2 = max(0, x0 - crop_size), min(W, x0 + crop_size)

    patch_angio = angio_bgr[y1:y2, x1:x2].copy()
    patch_heat = heatmap[y1:y2, x1:x2].copy()
    patch_mask = mask[y1:y2, x1:x2]

    patch_heat[patch_mask == 0] = patch_angio[patch_mask == 0]
    patch_blended = cv2.addWeighted(patch_angio, 0.4, patch_heat, 0.6, 0)

    for (py, px) in branch[L:R+1]:
        if y1 <= py < y2 and x1 <= px < x2:
            cv2.circle(patch_angio, (px - x1, py - y1), 1, (0, 0, 255), -1)
            cv2.circle(patch_blended, (px - x1, py - y1), 1, (255, 255, 255), -1)

    pad = 60
    disp_L = max(0, L - pad)
    disp_R = min(len(branch) - 1, R + pad)

    d_raw = np.array([2.0 * dt[y, x] for y, x in branch], dtype=np.float32)
    d_s = smooth_1d(d_raw, cfg.smooth_win)

    fig = plt.figure(figsize=(14, 5))

    ax1 = plt.subplot(1, 3, 1)
    ax1.imshow(cv2.cvtColor(patch_angio, cv2.COLOR_BGR2RGB))
    ax1.set_title("Lesion Angiogram")
    ax1.axis('off')

    ax2 = plt.subplot(1, 3, 2)
    ax2.imshow(cv2.cvtColor(patch_blended, cv2.COLOR_BGR2RGB))
    ax2.set_title("Width Heatmap (Red=Wide, Blue=Narrow)")
    ax2.axis('off')

    ax3 = plt.subplot(1, 3, 3)
    region_raw = d_raw[disp_L:disp_R+1]
    region_smooth = d_s[disp_L:disp_R+1]
    x_axis = range(disp_L, disp_R+1)

    ax3.plot(x_axis, region_raw, alpha=0.4, color='gray', label="Raw DT Width")
    ax3.plot(x_axis, region_smooth, color='blue', label="Smoothed Width")
    ax3.axvspan(L, R, alpha=0.2, color="red", label="Stenosis Region")
    ax3.axhline(les["RVD_px"], color='green', linestyle='--', label=f"RVD: {les['RVD_px']:.1f}px")
    ax3.plot(m_idx, les["MLD_px"], 'rv', markersize=8, label=f"MLD: {les['MLD_px']:.1f}px")
    ax3.set_title("Local Diameter Profile")
    ax3.set_ylabel("Diameter (pixels)")
    ax3.legend(fontsize=9)

    occ_str = " [TOTAL OCCLUSION]" if les.get("total_occlusion") else ""
    conf_str = f"  conf={les.get('confidence', '?')}  method={les.get('method', '?')}"
    fig.suptitle(
        f"{title} | {les['DS_percent']:.1f}% DS ({les['severity']}){occ_str}{conf_str}",
        fontsize=14, fontweight='bold'
    )
    plt.tight_layout()
    return fig


def save_explainable_report(out_dir: Path, stem: str, angio: np.ndarray, mask: np.ndarray,
                            branches: List[List[Tuple[int,int]]], lesions: List[dict],
                            dt: np.ndarray, cfg: QCAConfig, top_k: int = 3):
    """
    Generates a detailed, multi-panel PNG report for the top K severe lesions.
    Shows cropped angiogram, heatmap overlay, and localized diameter profile.
    """
    if not lesions:
        return

    dt_norm = cv2.normalize(dt, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heatmap = cv2.applyColorMap(dt_norm, cv2.COLORMAP_JET)

    for k, les in enumerate(lesions[:top_k], start=1):
        fig = build_lesion_figure(angio, mask, dt, les, cfg, f"{stem} - Lesion #{k}", heatmap=heatmap)
        report_path = out_dir / f"{stem}_lesion_{k}_report.png"
        plt.savefig(report_path, dpi=150, bbox_inches='tight')
        plt.close(fig)


# =========================
# Batch runner
# =========================
_CSV_FIELDS_BATCH = [
    "file", "lesion_rank", "severity", "DS_percent",
    "MLD_px", "RVD_px", "length_px",
    "MLD_mm", "RVD_mm", "length_mm",
    "L_idx", "R_idx", "min_idx",
    "total_occlusion", "confidence", "method",
]

def run_qca_batch(angiogram_dir: str, mask_dir: str, out_dir: str,
                  cfg: QCAConfig,
                  angio_exts=(".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
    validate_calibration(cfg)
    angio_dir = Path(angiogram_dir)
    mask_dirp = Path(mask_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    angio_files = []
    for ext in angio_exts:
        angio_files += list(angio_dir.glob(f"*{ext}"))

    if not angio_files:
        raise RuntimeError(f"No angiogram images found in: {angiogram_dir}")

    report_csv = out / "qca_report.csv"
    with open(report_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS_BATCH)
        writer.writeheader()

        for img_path in sorted(angio_files):
            stem = img_path.stem
            mask_path = None
            for ext in angio_exts:
                cand = mask_dirp / f"{stem}{ext}"
                if cand.exists():
                    mask_path = cand
                    break
            if mask_path is None:
                for ext in angio_exts:
                    cand = mask_dirp / f"{stem}_mask{ext}"
                    if cand.exists():
                        mask_path = cand
                        break

            if mask_path is None:
                print(f"[SKIP] No mask found for {img_path.name}")
                continue

            angio = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            mask_img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if angio is None or mask_img is None:
                print(f"[SKIP] Failed to read {img_path.name} or its mask.")
                continue

            bw = to_binary_mask(mask_img)
            bw = morph_cleanup(bw, cfg)

            try:
                branches, lesions, dt = qca_from_mask(bw, cfg, angio_gray=angio)
            except Exception as e:
                print(f"[FAIL] {img_path.name}: {e}")
                continue

            overlay = draw_overlay(angio, bw, branches, lesions)
            cv2.imwrite(str(out / f"{stem}_overlay.png"), overlay)
            save_diameter_plot(out / f"{stem}_diameter.png", branches, dt, lesions, cfg)
            save_explainable_report(out, stem, angio, bw, branches, lesions, dt, cfg)

            for rank, les in enumerate(lesions[:5], start=1):
                writer.writerow({
                    "file":            img_path.name,
                    "lesion_rank":     rank,
                    "severity":        les["severity"],
                    "DS_percent":      f"{les['DS_percent']:.4f}",
                    "MLD_px":          f"{les['MLD_px']:.4f}",
                    "RVD_px":          f"{les['RVD_px']:.4f}",
                    "length_px":       f"{les['length_px']:.4f}",
                    "MLD_mm":          "" if les["MLD_mm"] is None else f"{les['MLD_mm']:.4f}",
                    "RVD_mm":          "" if les["RVD_mm"] is None else f"{les['RVD_mm']:.4f}",
                    "length_mm":       "" if les["length_mm"] is None else f"{les['length_mm']:.4f}",
                    "L_idx":           les["L_idx"],
                    "R_idx":           les["R_idx"],
                    "min_idx":         les["min_idx"],
                    "total_occlusion": les.get("total_occlusion", False),
                    "confidence":      les.get("confidence", ""),
                    "method":          les.get("method", "dt"),
                })

            print(f"[OK] {img_path.name}: lesions={len(lesions)}, branches={len(branches)}")

    print(f"\nSaved report: {report_csv}")
    print(f"Saved outputs to: {out}")


# =========================
# Single image runner
# =========================
_CSV_FIELDS_SINGLE = [
    "file", "lesion_rank", "branch_id", "severity", "DS_percent",
    "MLD_px", "RVD_px", "length_px",
    "MLD_mm", "RVD_mm", "length_mm",
    "total_occlusion", "confidence", "method",
]

def run_qca_single(angio_path: str, mask_path: str, out_dir: str, cfg: QCAConfig):
    validate_calibration(cfg)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    img_path = Path(angio_path)
    stem = img_path.stem

    angio = cv2.imread(angio_path, cv2.IMREAD_GRAYSCALE)
    mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if angio is None or mask_img is None:
        print(f"[FAIL] Failed to read image or mask.")
        return

    bw = to_binary_mask(mask_img)
    bw = morph_cleanup(bw, cfg)

    try:
        branches, lesions, dt = qca_from_mask(bw, cfg, angio_gray=angio)
    except Exception as e:
        print(f"[FAIL] {img_path.name}: {e}")
        return

    print(f"  Branches found : {len(branches)}")
    print(f"  Lesions detected: {len(lesions)}")
    for i, les in enumerate(lesions):
        occ = " [TOTAL OCCLUSION]" if les.get("total_occlusion") else ""
        mld_str = (f"MLD={les['MLD_mm']:.2f}mm" if les["MLD_mm"] is not None
                   else f"MLD={les['MLD_px']:.1f}px")
        rvd_str = (f"RVD={les['RVD_mm']:.2f}mm" if les["RVD_mm"] is not None
                   else f"RVD={les['RVD_px']:.1f}px")
        print(f"    #{i+1}: DS={les['DS_percent']:.1f}% ({les['severity']}){occ}  "
              f"{mld_str}  {rvd_str}  "
              f"conf={les.get('confidence','?')}  method={les.get('method','?')}")

    overlay = draw_overlay(angio, bw, branches, lesions)
    cv2.imwrite(str(out / f"{stem}_overlay.png"), overlay)
    save_diameter_plot(out / f"{stem}_diameter.png", branches, dt, lesions, cfg)
    save_explainable_report(out, stem, angio, bw, branches, lesions, dt, cfg)

    report_csv = out / f"{stem}_qca_report.csv"
    with open(report_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS_SINGLE)
        writer.writeheader()

        for rank, les in enumerate(lesions[:10], start=1):
            b_id = -1
            for bi, br in enumerate(branches):
                if les["branch"] is br:
                    b_id = bi + 1
                    break
            writer.writerow({
                "file":            img_path.name,
                "lesion_rank":     rank,
                "branch_id":       b_id,
                "severity":        les["severity"],
                "DS_percent":      f"{les['DS_percent']:.4f}",
                "MLD_px":          f"{les['MLD_px']:.4f}",
                "RVD_px":          f"{les['RVD_px']:.4f}",
                "length_px":       f"{les['length_px']:.4f}",
                "MLD_mm":          "" if les["MLD_mm"] is None else f"{les['MLD_mm']:.4f}",
                "RVD_mm":          "" if les["RVD_mm"] is None else f"{les['RVD_mm']:.4f}",
                "length_mm":       "" if les["length_mm"] is None else f"{les['length_mm']:.4f}",
                "total_occlusion": les.get("total_occlusion", False),
                "confidence":      les.get("confidence", ""),
                "method":          les.get("method", "dt"),
            })

    print(f"[OK] {img_path.name}: {len(lesions)} lesions across {len(branches)} branches")
    print(f"\nSaved report : {report_csv}")
    print(f"Saved outputs to: {out}")


if __name__ == "__main__":
    import tkinter as tk
    from tkinter import filedialog

    cfg = QCAConfig(
        # px_to_mm=0.20,   # set if you know calibration (mm per pixel); use calibrate_from_catheter()
        # Thresholds follow ACC/AHA/ESC standard — do not override without clinical justification
    )

    root = tk.Tk()
    root.attributes("-topmost", True)
    root.withdraw()

    print("Please select the angiogram image...")
    angio_file = filedialog.askopenfilename(
        title="Select Angiogram Image",
        filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"), ("All files", "*.*")]
    )

    if not angio_file:
        print("No angiogram image selected. Exiting.")
        exit()

    print("Please select the corresponding mask image...")
    mask_file = filedialog.askopenfilename(
        title="Select Mask Image",
        filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"), ("All files", "*.*")]
    )

    if not mask_file:
        print("No mask image selected. Exiting.")
        exit()

    run_qca_single(
        angio_path=angio_file,
        mask_path=mask_file,
        out_dir="qca_outputs_single/",
        cfg=cfg
    )
