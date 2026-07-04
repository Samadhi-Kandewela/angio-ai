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
try:
    from skimage.filters import frangi
except Exception:  # keep script usable even if scikit-image lacks frangi
    frangi = None
from scipy.ndimage import distance_transform_edt


# =========================
# Config
# =========================
@dataclass
class QCAConfig:
    min_component_pixels: int = 200          # remove tiny mask components
    max_hole_area: int = 100                 # fill only small background holes to avoid bridging loops
    close_kernel: int = 3                    # morphological closing
    close_iters: int = 0
    open_kernel: int = 3                     # morphological opening
    open_iters: int = 1

    prune_spur_len: int = 6                  # prune small skeleton spurs (keep short stenotic branches)
    smooth_win: int = 15                     # diameter profile smoothing (odd recommended; increased for smoother profiles)
    centerline_smooth_win: int = 9           # coordinate smoothing window for skeleton centerline

    lesion_alpha: float = 0.80               # recovery threshold: d >= alpha * RVD_local
    ref_win_prox: int = 40                   # points before lesion for RVD estimate
    ref_win_dist: int = 40                   # points after lesion for RVD estimate
    min_lesion_points: int = 5               # ignore tiny lesions on centerline
    # ACC/AHA/ESC thresholds: ≥70% = SEVERE (revascularise regardless); 50–69% = MODERATE
    severe_threshold: float = 70.0
    moderate_threshold: float = 50.0
    min_branch_len: int = 10                 # minimum branch length (skeleton px) to analyze

    px_to_mm: Optional[float] = None         # set if you have calibration (mm per pixel)

    # Shape-aware QCA improvements
    # QCA should compare the measured lumen curve against an expected reference
    # diameter curve, because normal coronary arteries naturally taper.
    use_shape_aware_reference: bool = True
    reference_percentile: float = 85.0        # upper-envelope percentile for healthy reference
    reference_win: int = 41                  # window for local reference envelope
    min_candidate_ds: float = 25.0           # ignore tiny dips below this %DS
    boundary_ds: float = 15.0                # lesion boundary where deficit falls below this %DS
    min_valley_depth_px: float = 0.75        # dip depth below reference needed to be a candidate
    plot_use_perpendicular_width: bool = True # make plots match the QCA measurement method

    # Paper-inspired vessel-structure guidance (Mao et al., QIMS 2025)
    # Their CNN used Frangi-filtered vessel structure as an auxiliary output.
    # Here we cannot train their CNN without labelled data, so we reuse the
    # same idea as an interpretable anatomical guidance map for the mask/QCA step.
    use_frangi_guidance: bool = True
    frangi_black_ridges: bool = True          # coronary lumen is usually dark in XCA
    frangi_sigmas: Tuple[float, ...] = (1.0, 1.5, 2.0, 2.5, 3.0)
    frangi_threshold_percentile: float = 92.0
    frangi_near_mask_dilate: int = 5          # only add Frangi pixels near supplied mask
    frangi_min_support: float = 0.08          # quality warning threshold, 0..1
    low_contrast_warning_threshold: float = 5.0

    # Five clinically meaningful grades used in the DL paper
    use_five_grade_output: bool = True


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


def fill_small_holes(bw: np.ndarray, max_area: int) -> np.ndarray:
    """Fill only small background holes to avoid merging vessels that form loops."""
    bg = (bw == 0).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(bg, connectivity=8)
    if num <= 1:
        return bw.copy()
    
    out = bw.copy()
    areas = stats[:, cv2.CC_STAT_AREA]
    main_bg_label = np.argmax(areas)
    
    for i in range(num):
        if i == main_bg_label:
            continue
        if stats[i, cv2.CC_STAT_AREA] <= max_area:
            out[labels == i] = 255
            
    return out


def morph_cleanup(bw: np.ndarray, cfg: QCAConfig) -> np.ndarray:
    """Close gaps and remove specks PER connected component.

    Processing each component in isolation prevents morphological closing
    from bridging nearby-but-separate vessels.  After closing, any pixels
    where multiple closed components overlap are removed to guarantee
    vessel separation is preserved.
    """
    # 0. Fill internal holes (only small ones)
    bw = fill_small_holes(bw, cfg.max_hole_area)

    # 1. Remove tiny specks first so they don't interfere
    bw = keep_significant_components(bw, cfg.min_component_pixels)

    # 2. Label components BEFORE any closing
    num, labels, stats, _ = cv2.connectedComponentsWithStats(
        (bw > 0).astype(np.uint8), connectivity=8)
    if num <= 1:
        return bw

    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                        (cfg.close_kernel, cfg.close_kernel))
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (cfg.open_kernel, cfg.open_kernel))

    # Close & open each component independently
    closed_components = []
    for i in range(1, num):
        comp = (labels == i).astype(np.uint8) * 255
        comp = cv2.morphologyEx(comp, cv2.MORPH_CLOSE, k_close,
                                iterations=cfg.close_iters)
        comp = cv2.morphologyEx(comp, cv2.MORPH_OPEN, k_open,
                                iterations=cfg.open_iters)
        closed_components.append(comp)

    # Combine components but REMOVE overlapping pixels to prevent bridging
    overlap_count = np.zeros(bw.shape, dtype=np.int32)
    result = np.zeros_like(bw)
    for comp in closed_components:
        overlap_count += (comp > 0).astype(np.int32)
        result = cv2.bitwise_or(result, comp)

    # Any pixel claimed by 2+ components is a false bridge — remove it
    result[overlap_count > 1] = 0

    return result


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

def make_maximum_spanning_forest(adj: Dict[Tuple[int,int], List[Tuple[int,int]]], dt: np.ndarray) -> Dict[Tuple[int,int], List[Tuple[int,int]]]:
    """Break cycles in the skeleton graph using Kruskal's algorithm to form a maximum spanning forest."""
    edges = []
    seen = set()
    for u, neighbors in adj.items():
        for v in neighbors:
            edge_key = frozenset([u, v])
            if edge_key not in seen:
                seen.add(edge_key)
                weight = float(dt[u[0], u[1]] + dt[v[0], v[1]])
                edges.append((weight, u, v))
                
    # Sort edges by weight in descending order (maximum spanning forest)
    edges.sort(key=lambda x: x[0], reverse=True)
    
    parent = {}
    def find(x):
        if parent.get(x, x) == x:
            return x
        parent[x] = find(parent[x])
        return parent[x]
        
    def union(x, y):
        root_x = find(x)
        root_y = find(y)
        if root_x != root_y:
            parent[root_x] = root_y
            return True
        return False
        
    new_adj = {node: [] for node in adj.keys()}
    for weight, u, v in edges:
        if union(u, v):
            new_adj[u].append(v)
            new_adj[v].append(u)
            
    return new_adj

def extract_all_branches(skel: np.ndarray, min_branch_len: int, dt: np.ndarray) -> List[List[Tuple[int, int]]]:
    """Decompose skeleton into branches, merging collinear segments at junctions."""
    adj = build_adjacency(skel)
    if not adj:
        return []

    # Break skeleton cycles/loops using Maximum Spanning Forest
    adj = make_maximum_spanning_forest(adj, dt)

    # classify node degrees
    degree = {}
    for node, nbrs in adj.items():
        degree[node] = len(nbrs)

    # 1. Extract raw segments (between junctions/endpoints)
    visited_edges = set()
    segments = []

    start_nodes = [n for n, d in degree.items() if d == 1 or d >= 3]
    if not start_nodes:
        # no junctions or endpoints (a loop) — use any node
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

            segments.append(path)

    # If no segments, fallback
    if not segments:
        return []

    # 2. Merge collinear segments at junctions (degree >= 3)
    junctions = [n for n, d in degree.items() if d >= 3]

    # Map from segment index to segment path
    seg_dict = {i: list(seg) for i, seg in enumerate(segments)}
    active_segs = set(seg_dict.keys())

    # Helper to calculate unit direction vector of a segment leaving a junction J
    def get_outgoing_direction(seg_path, J):
        if seg_path[0] == J:
            pts = seg_path
        elif seg_path[-1] == J:
            pts = list(reversed(seg_path))
        else:
            return None
        
        m = min(len(pts) - 1, 5)
        if m < 1:
            return None
        dy = float(pts[m][0] - pts[0][0])
        dx = float(pts[m][1] - pts[0][1])
        length = math.sqrt(dy**2 + dx**2)
        if length < 1e-6:
            return None
        return (dy / length, dx / length)

    for J in junctions:
        # Find all active segments meeting at J
        meeting = []
        for sid in list(active_segs):
            path = seg_dict[sid]
            if path[0] == J or path[-1] == J:
                meeting.append(sid)

        if len(meeting) < 2:
            continue

        dirs = {}
        for sid in meeting:
            d = get_outgoing_direction(seg_dict[sid], J)
            if d is not None:
                dirs[sid] = d

        # Find the pair with the dot product closest to -1
        best_pair = None
        min_dot = 1.0

        meeting_with_dirs = [sid for sid in meeting if sid in dirs]
        for idx1 in range(len(meeting_with_dirs)):
            for idx2 in range(idx1 + 1, len(meeting_with_dirs)):
                sid1 = meeting_with_dirs[idx1]
                sid2 = meeting_with_dirs[idx2]
                d1 = dirs[sid1]
                d2 = dirs[sid2]
                dot = d1[0]*d2[0] + d1[1]*d2[1]
                if dot < min_dot:
                    min_dot = dot
                    best_pair = (sid1, sid2)

        # Merge if the angle is sufficiently straight (dot product < -0.5, i.e. > 120 degrees)
        if best_pair is not None and min_dot < -0.5:
            sid1, sid2 = best_pair
            path1 = seg_dict[sid1]
            path2 = seg_dict[sid2]

            # Perform the merge based on how they connect at J
            if path1[-1] == J and path2[0] == J:
                merged_path = path1 + path2[1:]
            elif path1[0] == J and path2[0] == J:
                merged_path = list(reversed(path1)) + path2[1:]
            elif path1[-1] == J and path2[-1] == J:
                merged_path = path1 + list(reversed(path2))[1:]
            elif path1[0] == J and path2[-1] == J:
                merged_path = path2 + path1[1:]
            else:
                continue

            new_sid = max(seg_dict.keys()) + 1
            seg_dict[new_sid] = merged_path
            active_segs.add(new_sid)

            active_segs.remove(sid1)
            active_segs.remove(sid2)

    # 3. Filter merged branches by minimum length
    merged_branches = []
    for sid in active_segs:
        path = seg_dict[sid]
        if len(path) >= min_branch_len:
            merged_branches.append(path)

    # Fallback to longest path if nothing met min length
    if not merged_branches:
        nodes = list(adj.keys())
        if len(nodes) >= 10:
            eps = skeleton_endpoints(skel)
            s = eps[0] if eps else nodes[0]
            a, _, _ = bfs_farthest(adj, s)
            b, parent_b, _ = bfs_farthest(adj, a)
            path = extract_path(parent_b, b)
            if len(path) >= min_branch_len:
                merged_branches.append(path)

    return merged_branches

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

def _perpendicular_diameter(branch: List[Tuple[int,int]], bw_mask: np.ndarray, idx: int) -> float:
    """Measure vessel diameter strictly perpendicular to the centerline at index idx.

    The distance-transform approach measures to the nearest background pixel in ANY
    direction.  Near junctions, an adjacent vessel boundary can be diagonally close,
    causing the DT to return an inflated value and severely underestimate %DS.

    This function scans in the true perpendicular direction of the vessel axis, so
    neighbouring vessels that are not perpendicular to the local tangent cannot
    contaminate the measurement.

    Returns diameter in pixels (sum of both half-widths).
    """
    H, W = bw_mask.shape
    N = len(branch)
    y0, x0 = branch[idx]

    # Tangent: use ±2 neighbours for a more stable local direction
    i_bwd = max(0, idx - 2)
    i_fwd = min(N - 1, idx + 2)
    dy = float(branch[i_fwd][0] - branch[i_bwd][0])
    dx = float(branch[i_fwd][1] - branch[i_bwd][1])
    length = math.sqrt(dy * dy + dx * dx)
    if length < 1e-6:
        return 0.0

    # Unit perpendicular vector (rotate tangent 90°)
    ny = -dx / length
    nx =  dy / length

    def _scan(sign: float) -> float:
        for step in range(1, 80):
            y = int(round(y0 + sign * step * ny))
            x = int(round(x0 + sign * step * nx))
            if y < 0 or y >= H or x < 0 or x >= W or bw_mask[y, x] == 0:
                # Sub-pixel refinement: the edge is between step-1 and step
                return max(0.0, float(step) - 0.5)
        return 79.0

    return _scan(+1.0) + _scan(-1.0)


def smooth_branch_coords(branch: List[Tuple[int,int]], win: int, H: int, W: int) -> List[Tuple[int,int]]:
    """Smooth skeleton centerline (y,x) coordinates to reduce pixel-level jaggedness.

    Applies a sliding-window average to each coordinate axis independently,
    then clamps to image bounds.  The smoothed path stays on integer pixels
    so dt can still be looked up by direct indexing.
    """
    if len(branch) < win or win <= 1:
        return branch
    pts = np.array(branch, dtype=np.float32)
    ys = smooth_1d(pts[:, 0], win)
    xs = smooth_1d(pts[:, 1], win)
    ys = np.clip(ys, 0, H - 1)
    xs = np.clip(xs, 0, W - 1)
    
    smoothed = [(int(round(float(y))), int(round(float(x)))) for y, x in zip(ys, xs)]
    # Keep the original endpoints fixed to preserve connectivity at junctions
    if len(smoothed) > 0:
        smoothed[0] = branch[0]
    if len(smoothed) > 1:
        smoothed[-1] = branch[-1]
    return smoothed

def local_minima_indices(x: np.ndarray) -> List[int]:
    """Find local minima, including branch endpoints.

    Endpoint stenoses (stenosis right at a bifurcation junction, which lands at
    index 0 or N-1 of the extracted branch segment) would be silently missed by
    an interior-only search.  Adding the endpoint checks catches these cases.
    """
    N = len(x)
    mins = []
    for i in range(1, N - 1):
        if x[i] <= x[i-1] and x[i] < x[i+1]:
            mins.append(i)
    # Endpoints: a branch can start or end right at a stenosis (junction-adjacent)
    if N >= 2 and x[0] < x[1]:
        mins.append(0)
    if N >= 2 and x[N - 1] < x[N - 2]:
        mins.append(N - 1)
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

def _find_vessel_shoulders(d_s: np.ndarray, m_idx: int, cfg: QCAConfig,
                            branch_ref_fallback: float) -> Tuple[float, int, float, int]:
    """Locate proximal/distal shoulder peaks flanking a stenosis minimum.

    Fixed-window RVD estimation fails for diffuse lesions because the reference
    window lands inside the stenotic segment (Suzuki et al. 2020, Fig 3), causing
    the walk threshold to be set below the actual MLD and the lesion to collapse
    to a single point.  This function searches up to 3× the standard reference
    window for the last local maximum before the minimum (proximal shoulder) and
    the first local maximum after it (distal shoulder), falling back to the highest
    point in the search region when no peak exists (monotonic decline, e.g. Branch 1).

    Returns (RVD_prox, prox_idx, RVD_dist, dist_idx).
    """
    N = len(d_s)

    # ── Proximal shoulder ─────────────────────────────────────────────────────
    prox_start = max(0, m_idx - cfg.ref_win_prox * 3)
    prox_idx = prox_start
    prox_val = float(d_s[prox_start]) if N > 0 else branch_ref_fallback

    found_prox = False
    for i in range(m_idx - 1, prox_start, -1):
        left_ok  = d_s[i] >= d_s[i - 1]
        right_ok = (i + 1 >= m_idx) or (d_s[i] >= d_s[i + 1])
        if left_ok and right_ok:
            prox_idx = i
            prox_val = float(d_s[i])
            found_prox = True
            break

    if not found_prox:
        seg = d_s[prox_start:m_idx]
        if len(seg) > 0:
            rel = int(np.argmax(seg))
            prox_idx = prox_start + rel
            prox_val = float(seg[rel])
        else:
            prox_val = branch_ref_fallback

    # ── Distal shoulder ───────────────────────────────────────────────────────
    dist_end = min(N - 1, m_idx + cfg.ref_win_dist * 3)
    dist_idx = dist_end
    dist_val = float(d_s[dist_end]) if N > 0 else branch_ref_fallback

    found_dist = False
    for i in range(m_idx + 1, dist_end):
        left_ok  = d_s[i] >= d_s[i - 1]
        right_ok = (i + 1 > dist_end) or (d_s[i] >= d_s[i + 1])
        if left_ok and right_ok:
            dist_idx = i
            dist_val = float(d_s[i])
            found_dist = True
            break

    if not found_dist:
        seg = d_s[m_idx + 1:dist_end + 1]
        if len(seg) > 0:
            rel = int(np.argmax(seg))
            dist_idx = m_idx + 1 + rel
            dist_val = float(seg[rel])
        else:
            dist_val = branch_ref_fallback

    return prox_val, prox_idx, dist_val, dist_idx


def detect_lesions_on_branch(branch: List[Tuple[int,int]], dt: np.ndarray,
                             cfg: QCAConfig, bw_mask: Optional[np.ndarray] = None) -> List[dict]:
    """
    Run lesion detection on a single branch centerline.
    dt is the distance transform of the full mask.
    bw_mask (optional): when provided, perpendicular diameter replaces DT-based width.
      Perpendicular measurement eliminates diagonal contamination from nearby vessels,
      which is the primary cause of severe stenoses being underestimated near junctions.
    Returns list of lesion dicts (with branch-local indices AND absolute (y,x) coords).
    """
    N = len(branch)
    d_raw = np.zeros(N, dtype=np.float32)
    for i, (y, x) in enumerate(branch):
        if bw_mask is not None:
            pd = _perpendicular_diameter(branch, bw_mask, i)
            # Fall back to DT if perpendicular scan gives a clearly wrong value
            # (e.g. centerline pixel landed outside mask after coordinate smoothing)
            d_raw[i] = pd if pd > 0.5 else 2.0 * dt[y, x]
        else:
            d_raw[i] = 2.0 * dt[y, x]

    # Coarse smooth: stable boundary detection + reference diameter
    d_s = smooth_1d(d_raw, cfg.smooth_win)
    # Fine smooth: preserves true MLD for sharp/localised stenoses.
    # A win-15 average blurs a 3-px neck up to ~4–5×, causing severe stenoses
    # to appear moderate.  A win-5 average retains the actual minimum.
    d_fine = smooth_1d(d_raw, max(3, cfg.smooth_win // 3))

    minima = local_minima_indices(d_s)

    # Global branch reference: fallback when a lesion sits at a branch endpoint
    # and one reference window is too short to compute a local estimate.
    branch_ref_fallback = float(np.percentile(d_s, 80)) if N > 0 else 0.0

    # ── Endpoint stenosis candidates via d_fine ───────────────────────────────
    # A 1–3 px neck at a branch endpoint is completely washed out by the coarse
    # smooth (win=15 pads 7 wide values on the outside), so local_minima_indices
    # on d_s never sees it.  Using d_fine (win≈5) preserves the narrow minimum,
    # and comparing against the branch global reference flags it as a candidate.
    if N >= 4 and branch_ref_fallback > 1e-6:
        em = min(15, N // 4)  # endpoint margin in branch-index space
        # Start of branch (junction-proximal end)
        s_idx = int(np.argmin(d_fine[:em]))
        if d_fine[s_idx] < cfg.lesion_alpha * branch_ref_fallback and s_idx not in minima:
            minima.append(s_idx)
        # End of branch (junction-distal end)
        e_rel = int(np.argmin(d_fine[N - em:]))
        e_idx = N - em + e_rel
        if d_fine[e_idx] < cfg.lesion_alpha * branch_ref_fallback and e_idx not in minima:
            minima.append(e_idx)

    lesions = []
    # Track which minima came from the endpoint d_fine scan (not interior d_s scan)
    # so we can use d_fine for their boundary walk (d_s may be too smoothed to extend L/R)
    _endpoint_candidates: set = set()
    if N >= 4 and branch_ref_fallback > 1e-6:
        em = min(15, N // 4)
        for _m in [int(np.argmin(d_fine[:em])),
                   N - em + int(np.argmin(d_fine[N - em:]))]:
            if _m in minima:
                _endpoint_candidates.add(_m)

    for m in minima:
        # ── Shoulder-based reference (handles diffuse/long lesions) ───────────
        # For a gradual monotonic decline (e.g. Branch 1), the fixed narrow
        # window around m lands entirely inside the stenotic segment, making
        # local_ref ≈ MLD and thr < MLD, so the walk never expands beyond m.
        # Shoulders find the actual proximal/distal normal vessel diameter even
        # when there is no local max (flat healthy segment) — they fall back to
        # the highest point in a 3× wider search region.
        RVD_prox_sh, prox_sh_idx, RVD_dist_sh, dist_sh_idx = \
            _find_vessel_shoulders(d_s, m, cfg, branch_ref_fallback)

        if RVD_prox_sh <= 1e-6 and RVD_dist_sh <= 1e-6:
            continue

        # Directional walk thresholds: proximal shoulder anchors L boundary,
        # distal shoulder anchors R boundary (handles asymmetric lesions).
        thr_prox = cfg.lesion_alpha * max(RVD_prox_sh, branch_ref_fallback * 0.5)
        thr_dist = cfg.lesion_alpha * max(RVD_dist_sh, branch_ref_fallback * 0.5)

        # For endpoint candidates use d_fine (coarse smooth is too flat there).
        d_walk = d_fine if m in _endpoint_candidates else d_s

        L = m
        while L > 0 and d_walk[L] < thr_prox:
            L -= 1
        R = m
        while R < N - 1 and d_walk[R] < thr_dist:
            R += 1

        if (R - L + 1) < cfg.min_lesion_points:
            continue

        # ── RVD: prefer shoulder value when shoulder lies outside [L, R] ─────
        # If the shoulder index is within the detected lesion span, it is itself
        # diseased — fall back to fixed-window percentile or global reference.
        prox_L = max(0, L - cfg.ref_win_prox)
        prox_R = max(0, L - 1)
        dist_L = min(N - 1, R + 1)
        dist_R = min(N - 1, R + cfg.ref_win_dist)

        if prox_sh_idx < L:
            RVD_prox = RVD_prox_sh
        elif prox_R - prox_L + 1 < 5:
            RVD_prox = branch_ref_fallback
        else:
            RVD_prox = float(np.percentile(d_s[prox_L:prox_R + 1], 80))

        if dist_sh_idx > R:
            RVD_dist = RVD_dist_sh
        elif dist_R - dist_L + 1 < 5:
            RVD_dist = branch_ref_fallback
        else:
            RVD_dist = float(np.percentile(d_s[dist_L:dist_R + 1], 80))

        # Standard QCA interpolated reference (Suzuki et al. 2020): RVD at
        # the MLD location is linearly interpolated between proximal and distal
        # reference to account for natural coronary artery taper.
        if dist_sh_idx > prox_sh_idx:
            t = float(np.clip((m - prox_sh_idx) / (dist_sh_idx - prox_sh_idx), 0.0, 1.0))
        else:
            t = 0.5
        RVD = RVD_prox * (1.0 - t) + RVD_dist * t
        if RVD <= 1e-6:
            continue

        MLD = float(np.min(d_fine[L:R + 1]))
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

        # ── Gradient (rate of diameter change) ───────────────────────────────
        # Entry gradient: rate of lumen narrowing from lesion entry to MLD.
        # Exit gradient:  rate of lumen recovery from MLD to lesion exit.
        # Steep entry + gradual exit → eccentric/asymmetric plaque (clinically
        # relevant shape information beyond simple %DS).
        grad_win = max(3, min(8, max(1, (R - L + 1) // 4)))
        entry_seg = d_s[L:min(L + grad_win + 1, m + 1)]
        exit_seg  = d_s[max(m, R - grad_win):R + 1]
        entry_grad = float((entry_seg[-1] - entry_seg[0]) / max(1, len(entry_seg) - 1)) \
            if len(entry_seg) > 1 else 0.0
        exit_grad  = float((exit_seg[-1] - exit_seg[0]) / max(1, len(exit_seg) - 1)) \
            if len(exit_seg) > 1 else 0.0

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
            "RVD_prox_px": RVD_prox,
            "RVD_dist_px": RVD_dist,
            "prox_sh_idx": int(prox_sh_idx),
            "dist_sh_idx": int(dist_sh_idx),
            "DS_percent": percent_DS,
            "length_px": length_px,
            "MLD_mm": MLD_mm,
            "RVD_mm": RVD_mm,
            "length_mm": length_mm,
            "entry_grad": entry_grad,   # px/step, negative = narrowing
            "exit_grad": exit_grad,     # px/step, positive = recovery
            "severity": severity,
            "min_pt": branch[m],
            "branch": branch,
            "d_smooth": d_s,            # coarse profile for this branch (plot reference)
            "d_fine": d_fine,
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


def _separate_touching_vessels(bw_mask: np.ndarray) -> np.ndarray:
    """Separate touching vessels using erosion to find vessel cores.

    Erodes the mask until touching vessels split into separate components,
    labels them, then dilates each label back to claim its original territory.
    Returns a label image (int32) where each separated vessel region
    has a unique positive label. Background is 0.
    """
    bw_bin = (bw_mask > 0).astype(np.uint8)

    # Progressively erode until we get more components than the original
    num_orig, _ = cv2.connectedComponents(bw_bin, connectivity=8)

    best_labels = None
    best_n = num_orig
    k_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    eroded = bw_bin.copy()
    for _ in range(8):  # up to 8 erosion steps
        eroded = cv2.erode(eroded, k_erode, iterations=1)
        if cv2.countNonZero(eroded) < 50:
            break
        n_cc, labels = cv2.connectedComponents(eroded, connectivity=8)
        if n_cc > best_n:
            best_n = n_cc
            best_labels = labels.copy()

    if best_labels is None or best_n <= num_orig:
        # No split found — return original mask as single region
        return bw_bin.astype(np.int32)

    # Dilate each labelled core back to fill original mask via nearest-label
    # Use distance-based Voronoi: for each foreground pixel, assign the
    # label of the nearest core pixel
    from scipy.ndimage import distance_transform_edt as _dt_edt

    result = np.zeros_like(bw_mask, dtype=np.int32)
    # For each label, compute distance from mask pixels to that label's core
    n_labels = int(best_labels.max())
    dist_maps = np.full((n_labels, *bw_mask.shape), np.inf, dtype=np.float64)
    for li in range(1, n_labels + 1):
        # distance from every pixel to the nearest pixel of label li
        core = (best_labels == li).astype(np.uint8)
        dist_maps[li - 1] = _dt_edt(1 - core)

    # For each foreground pixel, assign the label with minimum distance
    fg_mask = bw_mask > 0
    min_dist_label = np.argmin(dist_maps, axis=0) + 1  # 1-based labels
    result[fg_mask] = min_dist_label[fg_mask]

    return result




# =========================
# Paper-inspired Frangi vessel guidance + five-grade stenosis output
# =========================
def _normalize01(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    mn, mx = float(np.min(x)), float(np.max(x))
    if mx - mn < 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - mn) / (mx - mn)).astype(np.float32)


def compute_frangi_vessel_map(angio: np.ndarray, cfg: QCAConfig) -> np.ndarray:
    """Create a Frangi-based vesselness map from the angiogram.

    Mao et al. used a precomputed Frangi vessel map as an auxiliary anatomical
    learning target for their dual-output CNN.  In this script we use the same
    vessel-structure cue without training a CNN: it guides mask repair, provides
    a vessel-support score for each detected lesion, and is saved for debugging.
    """
    if angio is None or frangi is None:
        return np.zeros((0, 0), dtype=np.float32)

    if angio.ndim == 3:
        gray = cv2.cvtColor(angio, cv2.COLOR_BGR2GRAY)
    else:
        gray = angio.copy()

    # Improve local vessel/background contrast before Frangi filtering.
    gray_u8 = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    eq = clahe.apply(gray_u8)
    img = _normalize01(eq)

    try:
        vessel = frangi(
            img,
            sigmas=getattr(cfg, "frangi_sigmas", (1.0, 1.5, 2.0, 2.5, 3.0)),
            black_ridges=bool(getattr(cfg, "frangi_black_ridges", True)),
        )
    except TypeError:
        # Compatibility with older scikit-image signatures.
        vessel = frangi(img, black_ridges=bool(getattr(cfg, "frangi_black_ridges", True)))
    except Exception:
        return np.zeros_like(img, dtype=np.float32)

    vessel = np.nan_to_num(vessel, nan=0.0, posinf=0.0, neginf=0.0)
    return _normalize01(vessel)


def refine_mask_with_frangi(bw_mask: np.ndarray, frangi_map: np.ndarray, cfg: QCAConfig) -> np.ndarray:
    """Use the Frangi vessel map as a conservative anatomical constraint.

    We keep the user-provided/segmentation mask as the primary source.  Frangi
    only adds high-confidence vessel pixels very close to the mask.  This can
    reconnect tiny gaps or short under-segmented stenotic regions while avoiding
    large false bridges from unrelated vessels.
    """
    if frangi_map.size == 0 or frangi_map.shape != bw_mask.shape:
        return bw_mask
    if not bool(getattr(cfg, "use_frangi_guidance", True)):
        return bw_mask

    fg = (bw_mask > 0).astype(np.uint8)
    if cv2.countNonZero(fg) == 0:
        return bw_mask

    pct = float(getattr(cfg, "frangi_threshold_percentile", 92.0))
    positive = frangi_map[frangi_map > 0]
    if positive.size == 0:
        return bw_mask
    thr = float(np.percentile(positive, pct))
    aux = (frangi_map >= thr).astype(np.uint8)

    k = int(getattr(cfg, "frangi_near_mask_dilate", 5))
    if k % 2 == 0:
        k += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(3, k), max(3, k)))
    near_mask = cv2.dilate(fg, kernel, iterations=1)
    guided = ((fg > 0) | ((aux > 0) & (near_mask > 0))).astype(np.uint8) * 255

    # Light cleanup after adding auxiliary vessel pixels.
    guided = keep_significant_components(guided, max(20, int(cfg.min_component_pixels * 0.25)))
    return guided


def qims_five_grade(percent_ds: float) -> str:
    """Five stenosis grades used by Mao et al. QIMS 2025."""
    if percent_ds < 1.0:
        return "0_no_stenosis"
    if percent_ds < 50.0:
        return "1_49_nonobstructive"
    if percent_ds <= 70.0:
        return "50_70_intermediate"
    if percent_ds <= 95.0:
        return "71_95_severe"
    if percent_ds < 100.0:
        return "96_99_subtotal_occlusion"
    return "100_total_occlusion"


def _lesion_frangi_support(branch: List[Tuple[int, int]], L: int, R: int,
                           frangi_map: Optional[np.ndarray]) -> float:
    if frangi_map is None or frangi_map.size == 0:
        return float("nan")
    vals = []
    H, W = frangi_map.shape
    for y, x in branch[max(0, L):min(len(branch), R + 1)]:
        if 0 <= y < H and 0 <= x < W:
            vals.append(float(frangi_map[y, x]))
    return float(np.mean(vals)) if vals else float("nan")


def _local_angiographic_contrast(angio: Optional[np.ndarray], mask: np.ndarray,
                                center: Tuple[int, int], radius: int = 12) -> float:
    """Simple local contrast around lesion; low values warn about unreliable XCA."""
    if angio is None:
        return float("nan")
    if angio.ndim == 3:
        gray = cv2.cvtColor(angio, cv2.COLOR_BGR2GRAY)
    else:
        gray = angio
    y, x = center
    H, W = gray.shape[:2]
    y1, y2 = max(0, y - radius), min(H, y + radius + 1)
    x1, x2 = max(0, x - radius), min(W, x + radius + 1)
    patch = gray[y1:y2, x1:x2].astype(np.float32)
    patch_mask = mask[y1:y2, x1:x2] > 0
    if patch.size == 0 or np.sum(patch_mask) < 3 or np.sum(~patch_mask) < 3:
        return float("nan")
    return float(abs(np.mean(patch[patch_mask]) - np.mean(patch[~patch_mask])))


def enrich_lesions_with_image_guidance(lesions: List[dict], frangi_map: Optional[np.ndarray],
                                       angio: Optional[np.ndarray], mask: np.ndarray,
                                       cfg: QCAConfig) -> List[dict]:
    """Attach paper-inspired grading and image-quality fields to lesion rows."""
    for les in lesions:
        les["qims_grade"] = qims_five_grade(float(les.get("DS_percent", 0.0)))
        support = _lesion_frangi_support(les["branch"], les["L_idx"], les["R_idx"], frangi_map)
        les["frangi_support"] = support
        contrast = _local_angiographic_contrast(angio, mask, les.get("min_pt", (0, 0)))
        les["local_contrast"] = contrast
        warnings = []
        if not np.isnan(support) and support < float(getattr(cfg, "frangi_min_support", 0.08)):
            warnings.append("low_frangi_vessel_support")
        if not np.isnan(contrast) and contrast < float(getattr(cfg, "low_contrast_warning_threshold", 5.0)):
            warnings.append("low_local_contrast")
        if les.get("shape_type") in {"diffuse_tapering_deficit", "asymmetric_or_edge_lesion"}:
            warnings.append("needs_review_complex_shape")
        if les.get("DS_percent", 0.0) >= 96.0:
            warnings.append("critical_96_100_range_review")
        les["quality_warnings"] = ";".join(warnings)
    return lesions


def save_frangi_debug(out_path: Path, frangi_map: np.ndarray):
    if frangi_map is None or frangi_map.size == 0:
        return
    img = (255.0 * _normalize01(frangi_map)).astype(np.uint8)
    cv2.imwrite(str(out_path), img)

def qca_from_mask(bw_mask: np.ndarray, cfg: QCAConfig, angio: Optional[np.ndarray] = None):
    """
    Multi-branch QCA: skeletonize, decompose into branches,
    detect lesions on each branch independently.

    Fused vessels are split via watershed (seeded by skeleton branches)
    so that the distance transform is computed per-vessel-region.  This
    prevents touching arteries from inflating each other's width values
    and ensures narrow branches are correctly measured for stenosis.

    Returns:
      all_branches, all_lesions (sorted by severity), dt
    """
    # Optional Frangi auxiliary vessel map from the angiogram, inspired by the
    # dual-output CNN paper. It acts as vessel-structure guidance for this
    # modular QCA script, not as a trained deep-learning prediction.
    frangi_map = compute_frangi_vessel_map(angio, cfg) if angio is not None else np.zeros((0, 0), dtype=np.float32)
    if frangi_map.size and bool(getattr(cfg, "use_frangi_guidance", True)):
        bw_mask = refine_mask_with_frangi(bw_mask, frangi_map, cfg)

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

    # -- Per-vessel distance transform via erosion-based separation --
    vessel_labels = _separate_touching_vessels(cropped_mask)
    n_labels = int(vessel_labels.max())
    cr_dt = np.zeros(cropped_mask.shape, dtype=np.float32)
    for vi in range(1, n_labels + 1):
        region = (vessel_labels == vi).astype(np.uint8)
        cr_dt = np.maximum(cr_dt, distance_transform_edt(region).astype(np.float32))
    
    # Restore full sizes
    skel = np.zeros_like(bw_mask)
    skel[y1:y2, x1:x2] = cr_skel
    
    dt = np.zeros_like(bw_mask, dtype=np.float32)
    dt[y1:y2, x1:x2] = cr_dt

    branches = extract_all_branches(skel, cfg.min_branch_len, dt)

    # Smooth each branch's centerline coordinates to reduce skeleton pixel-level jitter.
    # Smoother centerlines → less noise in the diameter profile → more accurate stenosis detection.
    if cfg.centerline_smooth_win > 1:
        H, W = bw_mask.shape
        branches = [smooth_branch_coords(b, cfg.centerline_smooth_win, H, W) for b in branches]

    # Fallback: if branch decomposition yields nothing, use longest-path
    if not branches:
        try:
            cl = ordered_centerline_from_mask(bw_mask, cfg)
            branches = [cl]
        except RuntimeError:
            return [], [], dt

    all_lesions = []
    for branch in branches:
        lesions = detect_lesions_on_branch(branch, dt, cfg, bw_mask=bw_mask)
        all_lesions.extend(lesions)

    # Detect cannula/catheter region (points in the skeleton with diameter > 9.5px)
    # The cannula/catheter is by far the widest structure and causes false-positive stenoses.
    cannula_mask = (2.0 * dt > 9.5) & (skel > 0)
    cannula_coords = np.argwhere(cannula_mask)
    if len(cannula_coords) > 0:
        print(f"  [INFO] Detected cannula region with {len(cannula_coords)} skeleton points.")

    filtered_lesions = []
    for les in all_lesions:
        min_y, min_x = les["min_pt"]
        if len(cannula_coords) > 0:
            dists = np.linalg.norm(cannula_coords - np.array([min_y, min_x]), axis=1)
            min_dist = np.min(dists)
            if min_dist < 35.0:
                print(f"  [INFO] Ignoring cannula-related lesion at {les['min_pt']} (distance {min_dist:.1f}px to cannula region)")
                continue
        filtered_lesions.append(les)
    all_lesions = filtered_lesions

    all_lesions = sorted(all_lesions, key=lambda z: z["DS_percent"], reverse=True)
    all_lesions = enrich_lesions_with_image_guidance(all_lesions, frangi_map, angio, bw_mask, cfg)
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

    # ── Vessel contours ───────────────────────────────────────────────────────
    # Blur the float mask (sigma 1.0) before thresholding to get sub-pixel
    # smooth edges; polylines + LINE_AA then renders them without staircase.
    mask_f = (mask > 0).astype(np.float32)
    mask_smooth = cv2.GaussianBlur(mask_f, (0, 0), 1.0)
    mask_contour = (mask_smooth > 0.5).astype(np.uint8)
    contours, _ = cv2.findContours(mask_contour, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        cv2.polylines(vis, [cnt], isClosed=True, color=(0, 255, 0),
                      thickness=1, lineType=cv2.LINE_AA)

    # ── Branch centerlines ────────────────────────────────────────────────────
    # Draw as anti-aliased polylines instead of per-pixel coloring.
    for bi, branch in enumerate(branches):
        if len(branch) < 2:
            continue
        color = _BRANCH_COLORS[bi % len(_BRANCH_COLORS)]
        pts = np.array([(x, y) for (y, x) in branch], dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(vis, [pts], isClosed=False, color=color,
                      thickness=1, lineType=cv2.LINE_AA)
        
        # Label the branch with a small font near its middle
        by, bx = branch[len(branch)//2]
        cv2.putText(vis, f"B{bi+1}", (bx + 5, by),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)

    # ── Lesion markers ────────────────────────────────────────────────────────
    for k, les in enumerate(lesions[:5]):
        branch = les["branch"]
        L, R = les["L_idx"], les["R_idx"]
        seg = branch[L:min(R + 1, len(branch))]
        if len(seg) >= 2:
            pts = np.array([(x, y) for (y, x) in seg], dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(vis, [pts], isClosed=False, color=(0, 0, 255),
                          thickness=3, lineType=cv2.LINE_AA)
        elif seg:
            y, x = seg[0]
            cv2.circle(vis, (x, y), 3, (0, 0, 255), -1, lineType=cv2.LINE_AA)
        # Label at narrowest point
        y0, x0 = les["min_pt"]
        sev_color = (0, 0, 255) if les["severity"] == "SEVERE" else (0, 140, 255)
        text = f"#{k+1} {les['DS_percent']:.1f}% {les.get('qims_grade', les['severity'])}"
        cv2.putText(vis, text, (x0 + 8, y0 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, sev_color, 1, cv2.LINE_AA)

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

        branch_lesions = [les for les in lesions if les["branch"] is branch]
        for les in branch_lesions:
            L, R, m = les["L_idx"], les["R_idx"], les["min_idx"]
            sev = les["severity"]
            span_color = {"SEVERE": "red", "MODERATE": "orange", "MILD": "gold"}.get(sev, "red")

            ax.axvspan(L, R, alpha=0.22, color=span_color)
            ax.scatter([m], [d_s[m]], marker="v", color=span_color, zorder=5)

            # DS% and gradient annotation at MLD point
            entry_g = les.get("entry_grad", 0.0)
            exit_g  = les.get("exit_grad",  0.0)
            label = (f"{les['DS_percent']:.1f}% {sev[0]}\n"
                     f"↓{abs(entry_g):.2f}  ↑{exit_g:.2f} px/pt")
            ax.annotate(label, (m, d_s[m]),
                        textcoords="offset points", xytext=(5, 6),
                        fontsize=7, color=span_color)

            # Proximal and distal shoulder reference lines
            RVD_prox = les.get("RVD_prox_px", les["RVD_px"])
            RVD_dist = les.get("RVD_dist_px", les["RVD_px"])
            prox_sh  = les.get("prox_sh_idx", L)
            dist_sh  = les.get("dist_sh_idx", R)

            # Draw reference lines only within the shoulder-to-shoulder span
            ax.hlines(RVD_prox, prox_sh, m, colors="steelblue",
                      linestyles="--", linewidth=1.0, label=f"RVD_prox {RVD_prox:.1f}px" if bi == 0 else "")
            ax.hlines(RVD_dist, m, dist_sh, colors="darkorange",
                      linestyles="--", linewidth=1.0, label=f"RVD_dist {RVD_dist:.1f}px" if bi == 0 else "")
            ax.hlines(les["RVD_px"], L, R, colors="green",
                      linestyles=":", linewidth=0.8)

        ax.set_title(f"Branch {bi+1} ({len(branch)} pts)")
        ax.set_xlabel("Centerline index")
        ax.set_ylabel("Diameter (px)")
        ax.legend(fontsize=7)
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
            "qims_grade",
            "shape_type",
            "DS_percent",
            "frangi_support",
            "local_contrast",
            "quality_warnings",
            "MLD_px", "RVD_px", "RVD_curve_px", "length_px",
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
                branches, lesions, dt = qca_from_mask(bw, cfg, angio=angio)
            except Exception as e:
                print(f"[FAIL] {img_path.name}: {e}")
                continue

            # save overlay, plot, and explainable report
            overlay = draw_overlay(angio, bw, branches, lesions)
            cv2.imwrite(str(out / f"{stem}_overlay.png"), overlay)
            # Debug output matching the paper-inspired auxiliary Frangi vessel map
            save_frangi_debug(out / f"{stem}_frangi_vessel_map.png", compute_frangi_vessel_map(angio, cfg))
            save_diameter_plot(out / f"{stem}_diameter.png", branches, dt, lesions, cfg)
            save_explainable_report(out, stem, angio, bw, branches, lesions, dt, cfg)

            # write CSV rows (top K lesions)
            for rank, les in enumerate(lesions[:5], start=1):
                writer.writerow({
                    "file": img_path.name,
                    "lesion_rank": rank,
                    "severity": les["severity"],
                    "qims_grade": les.get("qims_grade", ""),
                    "shape_type": les.get("shape_type", ""),
                    "DS_percent": f"{les['DS_percent']:.4f}",
                    "frangi_support": "" if np.isnan(les.get("frangi_support", float("nan"))) else f"{les.get('frangi_support', 0.0):.4f}",
                    "local_contrast": "" if np.isnan(les.get("local_contrast", float("nan"))) else f"{les.get('local_contrast', 0.0):.4f}",
                    "quality_warnings": les.get("quality_warnings", ""),
                    "MLD_px": f"{les['MLD_px']:.4f}",
                    "RVD_px": f"{les['RVD_px']:.4f}",
                    "RVD_curve_px": f"{les.get('RVD_curve_px', les['RVD_px']):.4f}",
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
# Shape-aware QCA overrides
# =========================
def _odd_int(value: int, minimum: int = 3) -> int:
    value = max(minimum, int(value))
    return value if value % 2 == 1 else value + 1


def _diameter_profile(branch: List[Tuple[int, int]], dt: np.ndarray,
                      cfg: QCAConfig, bw_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Return the diameter profile used by QCA.

    Important: this uses the same perpendicular diameter measurement as lesion
    calculation when a binary mask is available.  This keeps the diameter plot,
    MLD, RVD and %DS consistent.
    """
    d = np.zeros(len(branch), dtype=np.float32)
    for i, (y, x) in enumerate(branch):
        if bw_mask is not None and getattr(cfg, "plot_use_perpendicular_width", True):
            pd = _perpendicular_diameter(branch, bw_mask, i)
            d[i] = pd if pd > 0.5 else 2.0 * dt[y, x]
        else:
            d[i] = 2.0 * dt[y, x]
    return d


def _rolling_percentile(x: np.ndarray, win: int, pct: float) -> np.ndarray:
    """Centered rolling percentile with edge padding."""
    if len(x) == 0:
        return x.copy()
    win = _odd_int(min(win, max(3, len(x) if len(x) % 2 == 1 else len(x) - 1)))
    if win < 3:
        return x.copy()
    pad = win // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    out = np.zeros_like(x, dtype=np.float32)
    for i in range(len(x)):
        out[i] = np.percentile(xp[i:i + win], pct)
    return out


def _estimate_reference_curve(d_s: np.ndarray, cfg: QCAConfig) -> np.ndarray:
    """Estimate the expected healthy reference diameter curve.

    This is a practical 2D-QCA approximation of the interpolated reference
    diameter: build an upper envelope from the diameter profile, then fit a
    gentle linear taper line through the likely normal portions.  This prevents
    normal distal taper from being mistaken for stenosis, while preserving focal
    V-shaped deficits as lesion candidates.
    """
    N = len(d_s)
    if N == 0:
        return d_s.copy()
    if N < 8:
        return np.full_like(d_s, float(np.percentile(d_s, 80)))

    x = np.arange(N, dtype=np.float32)
    win = min(getattr(cfg, "reference_win", 41), max(5, (N // 2) * 2 - 1))
    win = _odd_int(win)
    pct = float(getattr(cfg, "reference_percentile", 85.0))

    # Local upper envelope suppresses focal V-shaped stenosis dips.
    env = _rolling_percentile(d_s, win, pct)
    env = smooth_1d(env.astype(np.float32), max(5, min(win, 21)))

    # Fit using the upper half of envelope-supported points.  This approximates
    # the vessel diameter if no atherosclerotic narrowing were present.
    try:
        resid = d_s - env
        keep = resid >= np.percentile(resid, 35)  # exclude obvious low valleys
        if int(np.sum(keep)) >= max(4, N // 5):
            coef = np.polyfit(x[keep], env[keep], deg=1)
        else:
            coef = np.polyfit(x, env, deg=1)
        ref = np.polyval(coef, x).astype(np.float32)
    except Exception:
        ref = env.astype(np.float32)

    # Do not allow the reference curve to collapse below the local upper envelope.
    # A small blend keeps the line smooth while still following real taper.
    ref = np.maximum(ref, 0.85 * env)
    ref = smooth_1d(ref.astype(np.float32), 9)

    # Numerical guard: reference must be positive and at least mildly above the
    # smallest measured diameter, otherwise %DS becomes unstable.
    floor = max(1e-3, 0.5 * float(np.percentile(d_s, 75)))
    ref = np.maximum(ref, floor).astype(np.float32)
    return ref


def _candidate_minima_from_reference(d_fine: np.ndarray, ref_curve: np.ndarray,
                                     cfg: QCAConfig) -> List[int]:
    """Find valley candidates using measured-vs-reference diameter deficit."""
    if len(d_fine) == 0:
        return []
    ds_curve = (1.0 - (d_fine / np.maximum(ref_curve, 1e-6))) * 100.0
    min_candidate_ds = float(getattr(cfg, "min_candidate_ds", 25.0))
    min_depth = float(getattr(cfg, "min_valley_depth_px", 0.75))
    bad = (ds_curve >= min_candidate_ds) & ((ref_curve - d_fine) >= min_depth)

    mins: List[int] = []
    i = 0
    N = len(d_fine)
    while i < N:
        if not bad[i]:
            i += 1
            continue
        j = i
        while j + 1 < N and bad[j + 1]:
            j += 1
        # choose point with maximal %DS inside this candidate span
        m = i + int(np.argmax(ds_curve[i:j + 1]))
        mins.append(m)
        i = j + 1
    return mins


def _classify_profile_shape(d_s: np.ndarray, ref_curve: np.ndarray,
                            L: int, R: int, m: int) -> str:
    """Human-readable profile-shape label for reporting/debugging."""
    if R <= L or len(d_s) == 0:
        return "uncertain"
    seg = d_s[L:R + 1]
    ref_seg = ref_curve[L:R + 1]
    deficit = ref_seg - seg
    dip = float(np.max(deficit)) if len(deficit) else 0.0
    recovery_left = float(d_s[L] - d_s[m]) if m >= L else 0.0
    recovery_right = float(d_s[R] - d_s[m]) if R >= m else 0.0

    if dip < 0:
        return "bulge_or_overlap"
    if recovery_left > 0.5 and recovery_right > 0.5:
        return "focal_v_shape"
    if recovery_left > 0.5 or recovery_right > 0.5:
        return "asymmetric_or_edge_lesion"
    # Mostly downward trend with little recovery: could be diffuse lesion or taper.
    slope = float((d_s[R] - d_s[L]) / max(1, R - L))
    if slope < -0.03:
        return "diffuse_tapering_deficit"
    return "short_local_deficit"


def detect_lesions_on_branch(branch: List[Tuple[int,int]], dt: np.ndarray,
                             cfg: QCAConfig, bw_mask: Optional[np.ndarray] = None) -> List[dict]:
    """Shape-aware QCA lesion detection on one branch.

    Improvements over the earlier version:
    1. The diameter profile is compared with an estimated reference/taper curve.
       A normal diagonal decrease is treated as natural taper unless it falls
       below the reference curve enough to form a negative valley.
    2. Focal V-shaped narrowing is detected from the deficit curve.
    3. Final severity still uses the QCA formula: %DS = (1 - MLD/RVD) * 100.
    4. The plot and calculation can use the same perpendicular width values.
    """
    N = len(branch)
    if N < max(3, cfg.min_branch_len):
        return []

    d_raw = _diameter_profile(branch, dt, cfg, bw_mask)
    d_s = smooth_1d(d_raw, cfg.smooth_win)
    d_fine = smooth_1d(d_raw, max(3, cfg.smooth_win // 3))

    if getattr(cfg, "use_shape_aware_reference", True):
        ref_curve = _estimate_reference_curve(d_s, cfg)
    else:
        ref_curve = np.full_like(d_s, float(np.percentile(d_s, 80)))

    # Candidate minima from both local minima and reference-deficit valleys.
    minima = set(local_minima_indices(d_s))
    minima.update(_candidate_minima_from_reference(d_fine, ref_curve, cfg))

    branch_ref_fallback = float(np.percentile(d_s, 85)) if N > 0 else 0.0
    lesions: List[dict] = []

    for m in sorted(minima):
        if m < 0 or m >= N:
            continue

        # Skip upside-down V / bulge regions: measured diameter is not below reference.
        if d_fine[m] >= ref_curve[m] - float(getattr(cfg, "min_valley_depth_px", 0.75)):
            continue

        # Define lesion borders from the reference-deficit curve.
        ds_curve = (1.0 - (d_fine / np.maximum(ref_curve, 1e-6))) * 100.0
        boundary_ds = float(getattr(cfg, "boundary_ds", 15.0))
        L = int(m)
        while L > 0 and ds_curve[L] >= boundary_ds:
            L -= 1
        R = int(m)
        while R < N - 1 and ds_curve[R] >= boundary_ds:
            R += 1

        # Also use shoulder-based expansion from the original algorithm for
        # diffuse lesions where the deficit curve may be broad.
        RVD_prox_sh, prox_sh_idx, RVD_dist_sh, dist_sh_idx = \
            _find_vessel_shoulders(d_s, int(m), cfg, branch_ref_fallback)
        thr_prox = cfg.lesion_alpha * max(RVD_prox_sh, branch_ref_fallback * 0.5)
        thr_dist = cfg.lesion_alpha * max(RVD_dist_sh, branch_ref_fallback * 0.5)
        L2 = int(m)
        while L2 > 0 and d_s[L2] < thr_prox:
            L2 -= 1
        R2 = int(m)
        while R2 < N - 1 and d_s[R2] < thr_dist:
            R2 += 1
        L = min(L, L2)
        R = max(R, R2)

        if (R - L + 1) < cfg.min_lesion_points:
            continue

        # RVD at the MLD location: use the taper/reference curve as the primary
        # estimate, backed up by proximal/distal shoulder interpolation.
        if dist_sh_idx > prox_sh_idx:
            t = float(np.clip((m - prox_sh_idx) / (dist_sh_idx - prox_sh_idx), 0.0, 1.0))
        else:
            t = 0.5
        RVD_shoulder = RVD_prox_sh * (1.0 - t) + RVD_dist_sh * t
        RVD_curve = float(ref_curve[m])
        RVD = max(RVD_curve, 0.90 * RVD_shoulder)
        if RVD <= 1e-6:
            continue

        MLD = float(np.min(d_fine[L:R + 1]))
        percent_DS = (1.0 - (MLD / RVD)) * 100.0

        # Candidate-quality gate: avoids reporting tiny wiggles/noise as disease.
        if percent_DS < float(getattr(cfg, "min_candidate_ds", 25.0)):
            continue
        if (RVD - MLD) < float(getattr(cfg, "min_valley_depth_px", 0.75)):
            continue

        length_px = arc_length(branch, L, R)
        if cfg.px_to_mm is not None:
            MLD_mm = MLD * cfg.px_to_mm
            RVD_mm = RVD * cfg.px_to_mm
            length_mm = length_px * cfg.px_to_mm
        else:
            MLD_mm = None
            RVD_mm = None
            length_mm = None

        grad_win = max(3, min(8, max(1, (R - L + 1) // 4)))
        entry_seg = d_s[L:min(L + grad_win + 1, m + 1)]
        exit_seg  = d_s[max(m, R - grad_win):R + 1]
        entry_grad = float((entry_seg[-1] - entry_seg[0]) / max(1, len(entry_seg) - 1)) \
            if len(entry_seg) > 1 else 0.0
        exit_grad  = float((exit_seg[-1] - exit_seg[0]) / max(1, len(exit_seg) - 1)) \
            if len(exit_seg) > 1 else 0.0

        severity = "MILD"
        if percent_DS >= cfg.severe_threshold:
            severity = "SEVERE"
        elif percent_DS >= cfg.moderate_threshold:
            severity = "MODERATE"

        shape_type = _classify_profile_shape(d_s, ref_curve, L, R, int(m))

        lesions.append({
            "L_idx": int(L),
            "R_idx": int(R),
            "min_idx": int(m),
            "MLD_px": MLD,
            "RVD_px": float(RVD),
            "RVD_curve_px": float(RVD_curve),
            "RVD_shoulder_px": float(RVD_shoulder),
            "RVD_prox_px": float(RVD_prox_sh),
            "RVD_dist_px": float(RVD_dist_sh),
            "prox_sh_idx": int(prox_sh_idx),
            "dist_sh_idx": int(dist_sh_idx),
            "DS_percent": percent_DS,
            "length_px": length_px,
            "MLD_mm": MLD_mm,
            "RVD_mm": RVD_mm,
            "length_mm": length_mm,
            "entry_grad": entry_grad,
            "exit_grad": exit_grad,
            "severity": severity,
            "shape_type": shape_type,
            "min_pt": branch[int(m)],
            "branch": branch,
            "d_raw": d_raw,
            "d_smooth": d_s,
            "d_fine": d_fine,
            "ref_curve": ref_curve,
        })

    # Merge overlapping lesions on this branch, keeping the strongest %DS.
    lesions = sorted(lesions, key=lambda z: (z["L_idx"], z["R_idx"]))
    merged: List[dict] = []
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


def save_diameter_plot(out_path: Path, branches: List[List[Tuple[int,int]]],
                      dt: np.ndarray, lesions: List[dict], cfg: QCAConfig):
    """Save branch diameter plots that match the actual QCA measurement method."""
    n_branches = len(branches)
    if n_branches == 0:
        return
    fig, axes = plt.subplots(n_branches, 1, figsize=(10, 3 * n_branches), squeeze=False)
    for bi, branch in enumerate(branches):
        ax = axes[bi, 0]

        # If lesions exist for this branch, use stored profiles. Otherwise fall
        # back to DT because save_diameter_plot does not receive the mask.
        branch_lesions = [les for les in lesions if les["branch"] is branch]
        if branch_lesions and "d_raw" in branch_lesions[0]:
            d_raw = branch_lesions[0]["d_raw"]
            d_s = branch_lesions[0]["d_smooth"]
            ref_curve = branch_lesions[0].get("ref_curve", _estimate_reference_curve(d_s, cfg))
        else:
            d_raw = np.array([2.0 * dt[y, x] for y, x in branch], dtype=np.float32)
            d_s = smooth_1d(d_raw, cfg.smooth_win)
            ref_curve = _estimate_reference_curve(d_s, cfg)

        ax.plot(d_raw, alpha=0.45, label="raw diameter")
        ax.plot(d_s, label="smooth diameter")
        ax.plot(ref_curve, linestyle="--", linewidth=1.1, label="estimated reference/taper")

        for les in branch_lesions:
            L, R, m = les["L_idx"], les["R_idx"], les["min_idx"]
            sev = les["severity"]
            span_color = {"SEVERE": "red", "MODERATE": "orange", "MILD": "gold"}.get(sev, "red")

            ax.axvspan(L, R, alpha=0.22, color=span_color)
            y_m = d_s[m] if 0 <= m < len(d_s) else les["MLD_px"]
            ax.scatter([m], [y_m], marker="v", color=span_color, zorder=5)

            entry_g = les.get("entry_grad", 0.0)
            exit_g  = les.get("exit_grad",  0.0)
            shape = les.get("shape_type", "")
            label = (f"{les['DS_percent']:.1f}% {sev[0]}\n"
                     f"{shape}\n"
                     f"↓{abs(entry_g):.2f} ↑{exit_g:.2f} px/pt")
            ax.annotate(label, (m, y_m), textcoords="offset points", xytext=(5, 6),
                        fontsize=7, color=span_color)

            ax.hlines(les["RVD_px"], L, R, colors="green", linestyles=":", linewidth=0.9,
                      label="RVD at MLD" if bi == 0 else "")

        ax.set_title(f"Branch {bi+1} ({len(branch)} pts)")
        ax.set_xlabel("Centerline index")
        ax.set_ylabel("Diameter (px)")
        ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


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
        branches, lesions, dt = qca_from_mask(bw, cfg, angio=angio)
    except Exception as e:
        print(f"[FAIL] {img_path.name}: {e}")
        return

    print(f"  Branches found: {len(branches)}")
    print(f"  Lesions detected: {len(lesions)}")
    for i, les in enumerate(lesions):
        print(f"    #{i+1}: DS={les['DS_percent']:.1f}% ({les['severity']} / {les.get('qims_grade','')})  MLD={les['MLD_px']:.1f}px  RVD={les['RVD_px']:.1f}px  warnings={les.get('quality_warnings','')}")

    # save overlay, plot, and explainable reports
    overlay = draw_overlay(angio, bw, branches, lesions)
    cv2.imwrite(str(out / f"{stem}_overlay.png"), overlay)
    # Debug output matching the paper-inspired auxiliary Frangi vessel map
    save_frangi_debug(out / f"{stem}_frangi_vessel_map.png", compute_frangi_vessel_map(angio, cfg))
    save_diameter_plot(out / f"{stem}_diameter.png", branches, dt, lesions, cfg)
    save_explainable_report(out, stem, angio, bw, branches, lesions, dt, cfg)

    report_csv = out / f"{stem}_qca_report.csv"
    with open(report_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "file",
            "lesion_rank",
            "branch_id",
            "severity",
            "qims_grade",
            "shape_type",
            "DS_percent",
            "frangi_support",
            "local_contrast",
            "quality_warnings",
            "MLD_px", "RVD_px", "RVD_curve_px", "length_px",
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
                "qims_grade": les.get("qims_grade", ""),
                "shape_type": les.get("shape_type", ""),
                "DS_percent": f"{les['DS_percent']:.4f}",
                "frangi_support": "" if np.isnan(les.get("frangi_support", float("nan"))) else f"{les.get('frangi_support', 0.0):.4f}",
                "local_contrast": "" if np.isnan(les.get("local_contrast", float("nan"))) else f"{les.get('local_contrast', 0.0):.4f}",
                "quality_warnings": les.get("quality_warnings", ""),
                "MLD_px": f"{les['MLD_px']:.4f}",
                "RVD_px": f"{les['RVD_px']:.4f}",
                "RVD_curve_px": f"{les.get('RVD_curve_px', les['RVD_px']):.4f}",
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