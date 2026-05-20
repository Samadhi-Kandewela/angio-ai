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


# =========================
# Config
# =========================
@dataclass
class QCAConfig:
    min_component_pixels: int = 200          # remove tiny mask components
    close_kernel: int = 3                    # morphological closing
    close_iters: int = 2
    open_kernel: int = 3                     # morphological opening
    open_iters: int = 1

    prune_spur_len: int = 12                 # prune small skeleton spurs
    smooth_win: int = 11                     # diameter smoothing (odd recommended)

    lesion_alpha: float = 0.80               # recovery threshold: d >= alpha * RVD_local
    ref_win_prox: int = 40                   # points before lesion for RVD estimate
    ref_win_dist: int = 40                   # points after lesion for RVD estimate
    min_lesion_points: int = 5               # ignore tiny lesions on centerline
    severe_threshold: float = 50.0           # %DS >= severe_threshold -> severe
    moderate_threshold: float = 30.0         # %DS >= moderate_threshold -> moderate
    min_branch_len: int = 30                 # minimum branch length (skeleton px) to analyze

    px_to_mm: Optional[float] = None         # set if you have calibration (mm per pixel)


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
        # fallback: keep absolute largest non-background
        largest = 1 + int(np.argmax(areas))
        out = (labels == largest).astype(np.uint8) * 255
    return out

def morph_cleanup(bw: np.ndarray, cfg: QCAConfig) -> np.ndarray:
    """Close gaps, remove specks, keep all significant vessel components."""
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg.close_kernel, cfg.close_kernel))
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k_close, iterations=cfg.close_iters)

    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg.open_kernel, cfg.open_kernel))
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
    # skeleton is 0 or 255. Divide by 255 to get 0 or 1.
    sk = (skel > 0).astype(np.uint8)
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]], dtype=np.uint8)
    neighbor_count = cv2.filter2D(sk, -1, kernel, borderType=cv2.BORDER_CONSTANT)
    # Endpoints are where sk == 1 AND neighbor_count == 1
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
    # Junctions are where sk == 1 AND neighbor_count >= 3
    junction_mask = (sk == 1) & (neighbor_count >= 3)
    ys, xs = np.where(junction_mask)
    return list(zip(ys, xs))

def build_adjacency(skel: np.ndarray) -> Dict[Tuple[int,int], List[Tuple[int,int]]]:
    """Fast adjacency graph builder using vectorized shifts."""
    sk = (skel > 0)
    ys, xs = np.where(sk)
    
    # Map each (y,x) to an integer index for fast lookup
    H, W = sk.shape
    idx_map = np.full((H, W), -1, dtype=np.int32)
    idx_map[ys, xs] = np.arange(len(ys))
    
    pts = list(zip(ys, xs))
    adj: Dict[Tuple[int,int], List[Tuple[int,int]]] = {p: [] for p in pts}
    
    for dy, dx in _NEI8:
        # shifted coordinates
        ny = ys + dy
        nx = xs + dx
        
        # valid bounds
        valid = (ny >= 0) & (ny < H) & (nx >= 0) & (nx < W)
        
        # filter valid
        v_ys = ys[valid]
        v_xs = xs[valid]
        v_ny = ny[valid]
        v_nx = nx[valid]
        
        # check if shifted pixel is also part of skeleton
        is_skel = sk[v_ny, v_nx]
        
        # add to adjacency
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
            # walk from endpoint
            path = [ep]
            prev = None
            cur = ep
            while True:
                nbrs = adj.get(cur, [])
                deg = len(nbrs)
                if deg >= 3 and cur != ep:
                    break  # junction reached
                # choose next node not equal prev
                next_nodes = [n for n in nbrs if n != prev]
                if not next_nodes:
                    break
                nxt = next_nodes[0]
                path.append(nxt)
                prev, cur = cur, nxt
                # stop if too long
                if len(path) > max_len:
                    path = []
                    break

            if path and 2 <= len(path) <= max_len:
                # remove spur pixels except the junction pixel (last one) if it is junction
                for (y, x) in path[:-1]:
                    sk[y, x] = 0
                changed = True
    return sk

# skeleton_junctions moved above

def extract_all_branches(skel: np.ndarray, min_branch_len: int) -> List[List[Tuple[int, int]]]:
    """
    Decompose skeleton into individual branch segments.
    A branch runs from endpoint/junction to endpoint/junction.
    Returns list of ordered (y,x) paths, one per branch.
    """
    adj = build_adjacency(skel)
    if not adj:
        return []

    # classify node degrees
    degree = {}
    for node, nbrs in adj.items():
        degree[node] = len(nbrs)

    # walk from each endpoint or junction neighbor to trace branches
    visited_edges = set()  # frozenset pairs
    branches = []

    start_nodes = [n for n, d in degree.items() if d == 1 or d >= 3]
    if not start_nodes:
        # no junctions or endpoints (a loop) — use any node, trace whole thing
        start_nodes = [list(adj.keys())[0]]

    for start in start_nodes:
        for nbr in adj[start]:
            edge_key = frozenset([start, nbr])
            if edge_key in visited_edges:
                continue

            # trace from start through nbr until we hit another junction/endpoint
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

            if len(path) >= min_branch_len:
                branches.append(path)

    # If no branch extracted (very simple skeleton), fall back to longest path
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
    # 1. Find bounding box of the mask to crop it
    ys, xs = np.where(bw_mask > 0)
    if len(ys) == 0:
        raise RuntimeError("Mask is empty.")
        
    y_min, y_max = np.min(ys), np.max(ys)
    x_min, x_max = np.min(xs), np.max(xs)
    
    pad = 10
    H, W = bw_mask.shape
    y1, y2 = max(0, y_min - pad), min(H, y_max + pad + 1)
    x1, x2 = max(0, x_min - pad), min(W, x_max + pad + 1)
    
    # Process only cropped region
    cropped_mask = bw_mask[y1:y2, x1:x2]
    
    cr_skel = skeletonize((cropped_mask > 0)).astype(np.uint8)
    cr_skel = prune_spurs(cr_skel, cfg.prune_spur_len)
    
    # Restore full sizes
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
# QCA: diameter profile + lesion metrics
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
    # diffs is roughly (N-1, 2) shaped [dy, dx]
    # distance = sqrt(dy^2 + dx^2)
    distances = np.linalg.norm(diffs, axis=1)
    return float(np.sum(distances))

def detect_lesions_on_branch(branch: List[Tuple[int,int]], dt: np.ndarray, cfg: QCAConfig) -> List[dict]:
    """
    Run lesion detection on a single branch centerline.
    dt is the distance transform of the full mask.
    Returns list of lesion dicts (with branch-local indices AND absolute (y,x) coords).
    """
    N = len(branch)
    d_raw = np.zeros(N, dtype=np.float32)
    for i, (y, x) in enumerate(branch):
        d_raw[i] = 2.0 * dt[y, x]

    d_s = smooth_1d(d_raw, cfg.smooth_win)
    minima = local_minima_indices(d_s)
    lesions = []

    for m in minima:
        left0 = max(0, m - cfg.ref_win_prox)
        right0 = min(N - 1, m + cfg.ref_win_dist)
        local_ref = np.median(d_s[left0:right0+1])
        if local_ref <= 1e-6:
            continue

        thr = cfg.lesion_alpha * local_ref

        L = m
        while L > 0 and d_s[L] < thr:
            L -= 1
        R = m
        while R < N - 1 and d_s[R] < thr:
            R += 1

        if (R - L + 1) < cfg.min_lesion_points:
            continue

        prox_L = max(0, L - cfg.ref_win_prox)
        prox_R = max(0, L - 1)
        dist_L = min(N - 1, R + 1)
        dist_R = min(N - 1, R + cfg.ref_win_dist)

        if prox_R - prox_L + 1 < 5 or dist_R - dist_L + 1 < 5:
            continue

        RVD_prox = float(np.percentile(d_s[prox_L:prox_R+1], 80))
        RVD_dist = float(np.percentile(d_s[dist_L:dist_R+1], 80))
        RVD = 0.5 * (RVD_prox + RVD_dist)
        if RVD <= 1e-6:
            continue

        MLD = float(np.min(d_s[L:R+1]))
        percent_DS = (1.0 - (MLD / RVD)) * 100.0

        length_px = arc_length(branch, L, R)

        if cfg.px_to_mm is not None:
            MLD_mm = MLD * cfg.px_to_mm
            RVD_mm = RVD * cfg.px_to_mm
            length_mm = length_px * cfg.px_to_mm
        else:
            MLD_mm = None
            RVD_mm = None
            length_mm = None

        severity = "MILD"
        if percent_DS >= cfg.severe_threshold:
            severity = "SEVERE"
        elif percent_DS >= cfg.moderate_threshold:
            severity = "MODERATE"

        lesions.append({
            "L_idx": int(L),
            "R_idx": int(R),
            "min_idx": int(m),
            "MLD_px": MLD,
            "RVD_px": RVD,
            "DS_percent": percent_DS,
            "length_px": length_px,
            "MLD_mm": MLD_mm,
            "RVD_mm": RVD_mm,
            "length_mm": length_mm,
            "severity": severity,
            "min_pt": branch[m],          # (y,x) of narrowest point
            "branch": branch,             # reference to the branch path
        })

    # merge overlapping on this branch
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


def qca_from_mask(bw_mask: np.ndarray, cfg: QCAConfig):
    """
    Multi-branch QCA: skeletonize, decompose into branches,
    detect lesions on each branch independently.
    Optimized: Crops the mask to the bounding box of the vessel before expensive 
    skeletonization and distance_transform operations.
    Returns:
      all_branches, all_lesions (sorted by severity), dt
    """
    # 1. Find bounding box of the mask to crop it
    ys, xs = np.where(bw_mask > 0)
    if len(ys) == 0:
        return [], [], np.zeros_like(bw_mask, dtype=np.float32)
        
    y_min, y_max = np.min(ys), np.max(ys)
    x_min, x_max = np.min(xs), np.max(xs)
    
    pad = 10
    H, W = bw_mask.shape
    y1, y2 = max(0, y_min - pad), min(H, y_max + pad + 1)
    x1, x2 = max(0, x_min - pad), min(W, x_max + pad + 1)
    
    # Process only cropped region
    cropped_mask = bw_mask[y1:y2, x1:x2]
    
    cr_skel = get_skeleton_from_mask(cropped_mask, cfg)
    cr_dt = distance_transform_edt(cropped_mask > 0)
    
    # Restore full sizes
    skel = np.zeros_like(bw_mask)
    skel[y1:y2, x1:x2] = cr_skel
    
    dt = np.zeros_like(bw_mask, dtype=np.float32)
    dt[y1:y2, x1:x2] = cr_dt

    branches = extract_all_branches(skel, cfg.min_branch_len)

    # Fallback: if branch decomposition yields nothing, use longest-path
    if not branches:
        try:
            cl = ordered_centerline_from_mask(bw_mask, cfg)
            branches = [cl]
        except RuntimeError:
            return [], [], dt

    all_lesions = []
    for branch in branches:
        lesions = detect_lesions_on_branch(branch, dt, cfg)
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

    # lesions: mark segment in red and label
    for k, les in enumerate(lesions[:5]):  # show top 5
        branch = les["branch"]
        L, R = les["L_idx"], les["R_idx"]
        for i in range(L, min(R + 1, len(branch))):
            y, x = branch[i]
            cv2.circle(vis, (x, y), 2, (0, 0, 255), -1)
        # label at narrowest point
        y0, x0 = les["min_pt"]
        text = f"#{k+1} {les['DS_percent']:.1f}% {les['severity']}"
        cv2.putText(vis, text, (x0 + 8, y0 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)

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

def save_explainable_report(out_dir: Path, stem: str, angio: np.ndarray, mask: np.ndarray,
                            branches: List[List[Tuple[int,int]]], lesions: List[dict],
                            dt: np.ndarray, cfg: QCAConfig, top_k: int = 3):
    """
    Generates a detailed, multi-panel PDF/PNG report for the top K severe lesions.
    Shows cropped angiogram, heatmap overlay, and localized diameter profile.
    """
    if not lesions:
        return

    # Create JET colormap of distance transform for thickness heatmap
    dt_norm = cv2.normalize(dt, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heatmap = cv2.applyColorMap(dt_norm, cv2.COLORMAP_JET)
    
    # ensure angio is BGR for overlaying
    if angio.ndim == 2:
        angio_bgr = cv2.cvtColor(angio, cv2.COLOR_GRAY2BGR)
    else:
        angio_bgr = angio.copy()

    for k, les in enumerate(lesions[:top_k], start=1):
        branch = les["branch"]
        L, R = les["L_idx"], les["R_idx"]
        m_idx = les["min_idx"]
        
        # Crop region around the lesion
        y0, x0 = les["min_pt"]
        crop_size = 80
        H, W = angio.shape[:2]
        
        y1, y2 = max(0, y0 - crop_size), min(H, y0 + crop_size)
        x1, x2 = max(0, x0 - crop_size), min(W, x0 + crop_size)
        
        patch_angio = angio_bgr[y1:y2, x1:x2].copy()
        patch_heat = heatmap[y1:y2, x1:x2].copy()
        patch_mask = mask[y1:y2, x1:x2]
        
        # Apply mask to heatmap so it only colors the vessel, not background
        patch_heat[patch_mask == 0] = patch_angio[patch_mask == 0]
        
        # Blend heatmap with original angiogram
        patch_blended = cv2.addWeighted(patch_angio, 0.4, patch_heat, 0.6, 0)
        
        # Draw lesion markers on patches
        for (py, px) in branch[L:R+1]:
            if y1 <= py < y2 and x1 <= px < x2:
                cv2.circle(patch_angio, (px - x1, py - y1), 1, (0, 0, 255), -1)
                cv2.circle(patch_blended, (px - x1, py - y1), 1, (255, 255, 255), -1)
        
        # Centerline profile data for THIS lesion specifically (+ pad)
        pad = 60
        disp_L = max(0, L - pad)
        disp_R = min(len(branch) - 1, R + pad)
        
        d_raw = np.array([2.0 * dt[y, x] for y, x in branch], dtype=np.float32)
        d_s = smooth_1d(d_raw, cfg.smooth_win)
        
        fig = plt.figure(figsize=(14, 5))
        
        # Panel 1: Original Patched Angiogram
        ax1 = plt.subplot(1, 3, 1)
        ax1.imshow(cv2.cvtColor(patch_angio, cv2.COLOR_BGR2RGB))
        ax1.set_title(f"Lesion #{k} Angiogram")
        ax1.axis('off')
        
        # Panel 2: Heatmap Overlay
        ax2 = plt.subplot(1, 3, 2)
        ax2.imshow(cv2.cvtColor(patch_blended, cv2.COLOR_BGR2RGB))
        ax2.set_title(f"Width Heatmap (Red=Wide, Blue=Narrow)")
        ax2.axis('off')
        
        # Panel 3: Diameter Profile
        ax3 = plt.subplot(1, 3, 3)
        region_raw = d_raw[disp_L:disp_R+1]
        region_smooth = d_s[disp_L:disp_R+1]
        x_axis = range(disp_L, disp_R+1)
        
        ax3.plot(x_axis, region_raw, alpha=0.4, color='gray', label="Raw DT Width")
        ax3.plot(x_axis, region_smooth, color='blue', label="Smoothed Width")
        
        # Highlight MLD and RVD lines
        ax3.axvspan(L, R, alpha=0.2, color="red", label="Stenosis Region")
        ax3.axhline(les["RVD_px"], color='green', linestyle='--', label=f"Ref Vess Dia (RVD): {les['RVD_px']:.1f}px")
        ax3.plot(m_idx, les["MLD_px"], 'rv', markersize=8, label=f"Min Lumen Dia (MLD): {les['MLD_px']:.1f}px")
        
        ax3.set_title("Local Diameter Profile")
        ax3.set_ylabel("Diameter (pixels)")
        ax3.legend(fontsize=9)
        
        # Add overarching title with metrics
        fig.suptitle(f"{stem} - Lesion #{k} Report | {les['DS_percent']:.1f}% DS ({les['severity']})", fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        report_path = out_dir / f"{stem}_lesion_{k}_report.png"
        plt.savefig(report_path, dpi=150, bbox_inches='tight')
        plt.close(fig)


# =========================
# Batch runner
# =========================
def run_qca_batch(angiogram_dir: str, mask_dir: str, out_dir: str,
                  cfg: QCAConfig,
                  angio_exts=(".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
    angio_dir = Path(angiogram_dir)
    mask_dirp = Path(mask_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # match by stem name
    angio_files = []
    for ext in angio_exts:
        angio_files += list(angio_dir.glob(f"*{ext}"))

    if not angio_files:
        raise RuntimeError(f"No angiogram images found in: {angiogram_dir}")

    report_csv = out / "qca_report.csv"
    with open(report_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "file",
            "lesion_rank",
            "severity",
            "DS_percent",
            "MLD_px", "RVD_px", "length_px",
            "MLD_mm", "RVD_mm", "length_mm",
            "L_idx", "R_idx", "min_idx"
        ])
        writer.writeheader()

        for img_path in sorted(angio_files):
            stem = img_path.stem
            # find corresponding mask
            mask_path = None
            for ext in angio_exts:
                cand = mask_dirp / f"{stem}{ext}"
                if cand.exists():
                    mask_path = cand
                    break
            if mask_path is None:
                # also try _mask naming
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
                branches, lesions, dt = qca_from_mask(bw, cfg)
            except Exception as e:
                print(f"[FAIL] {img_path.name}: {e}")
                continue

            # save overlay, plot, and explainable report
            overlay = draw_overlay(angio, bw, branches, lesions)
            cv2.imwrite(str(out / f"{stem}_overlay.png"), overlay)
            save_diameter_plot(out / f"{stem}_diameter.png", branches, dt, lesions, cfg)
            save_explainable_report(out, stem, angio, bw, branches, lesions, dt, cfg)

            # write CSV rows (top K lesions)
            for rank, les in enumerate(lesions[:5], start=1):
                writer.writerow({
                    "file": img_path.name,
                    "lesion_rank": rank,
                    "severity": les["severity"],
                    "DS_percent": f"{les['DS_percent']:.4f}",
                    "MLD_px": f"{les['MLD_px']:.4f}",
                    "RVD_px": f"{les['RVD_px']:.4f}",
                    "length_px": f"{les['length_px']:.4f}",
                    "MLD_mm": "" if les["MLD_mm"] is None else f"{les['MLD_mm']:.4f}",
                    "RVD_mm": "" if les["RVD_mm"] is None else f"{les['RVD_mm']:.4f}",
                    "length_mm": "" if les["length_mm"] is None else f"{les['length_mm']:.4f}",
                    "L_idx": les["L_idx"],
                    "R_idx": les["R_idx"],
                    "min_idx": les["min_idx"],
                })

            print(f"[OK] {img_path.name}: lesions={len(lesions)}")

    print(f"\nSaved report: {report_csv}")
    print(f"Saved outputs to: {out}")


# =========================
# Example usage
# =========================
# =========================
# Single image runner
# =========================
def run_qca_single(angio_path: str, mask_path: str, out_dir: str, cfg: QCAConfig):
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
        branches, lesions, dt = qca_from_mask(bw, cfg)
    except Exception as e:
        print(f"[FAIL] {img_path.name}: {e}")
        return

    print(f"  Branches found: {len(branches)}")
    print(f"  Lesions detected: {len(lesions)}")
    for i, les in enumerate(lesions):
        print(f"    #{i+1}: DS={les['DS_percent']:.1f}% ({les['severity']})  MLD={les['MLD_px']:.1f}px  RVD={les['RVD_px']:.1f}px")

    # save overlay, plot, and explainable reports
    overlay = draw_overlay(angio, bw, branches, lesions)
    cv2.imwrite(str(out / f"{stem}_overlay.png"), overlay)
    save_diameter_plot(out / f"{stem}_diameter.png", branches, dt, lesions, cfg)
    save_explainable_report(out, stem, angio, bw, branches, lesions, dt, cfg)

    report_csv = out / f"{stem}_qca_report.csv"
    with open(report_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "file",
            "lesion_rank",
            "branch_id",
            "severity",
            "DS_percent",
            "MLD_px", "RVD_px", "length_px",
            "MLD_mm", "RVD_mm", "length_mm",
        ])
        writer.writeheader()

        for rank, les in enumerate(lesions[:10], start=1):
            # find which branch index
            b_id = -1
            for bi, br in enumerate(branches):
                if les["branch"] is br:
                    b_id = bi + 1
                    break
            writer.writerow({
                "file": img_path.name,
                "lesion_rank": rank,
                "branch_id": b_id,
                "severity": les["severity"],
                "DS_percent": f"{les['DS_percent']:.4f}",
                "MLD_px": f"{les['MLD_px']:.4f}",
                "RVD_px": f"{les['RVD_px']:.4f}",
                "length_px": f"{les['length_px']:.4f}",
                "MLD_mm": "" if les["MLD_mm"] is None else f"{les['MLD_mm']:.4f}",
                "RVD_mm": "" if les["RVD_mm"] is None else f"{les['RVD_mm']:.4f}",
                "length_mm": "" if les["length_mm"] is None else f"{les['length_mm']:.4f}",
            })

    print(f"[OK] {img_path.name}: {len(lesions)} lesions across {len(branches)} branches")
    print(f"\nSaved report: {report_csv}")
    print(f"Saved outputs to: {out}")


if __name__ == "__main__":
    import tkinter as tk
    from tkinter import filedialog
    
    cfg = QCAConfig(
        # px_to_mm=0.05,   # <- set this if you know calibration (mm per pixel)
        severe_threshold=50.0
    )

    root = tk.Tk()
    root.attributes("-topmost", True)
    root.withdraw() # Hide the main window

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