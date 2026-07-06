"""
Offline, whole-video QCA analysis for the clinical report generator.

A cine run produces a noisy per-frame stream of lesion measurements (the same
narrowing measures slightly differently frame to frame). This module runs the
frame_pipeline across every frame of one angiographic view ("angle"), clusters
detections that belong to the same anatomical lesion into a LesionTrack, and
picks each track's representative measurement from its highest-confidence
frame. It also picks one "summary frame" per angle that shows as many of the
angle's lesions simultaneously as possible, for the 2D overview diagram.
"""

from dataclasses import dataclass, field
from typing import Callable, List, Optional

import cv2
import numpy as np

from qca import QCAConfig, _BRANCH_COLORS, _SEVERITY_BGR, _SEVERITY_RADIUS
from frame_pipeline import (
    SegmentationModel, LocalizationModel,
    preprocess_frame, segment_frame, run_localization_frame, run_qca_frame,
)

# Offline analysis can afford to run localization more often than the live
# preview (frame_pipeline / VideoThread use interval=15 to hold real-time FPS).
LOC_FRAME_INTERVAL = 5

# Fallback spatial clustering bucket size (pixels) used only when no
# localization model is available to group detections by anatomical segment.
SPATIAL_BUCKET_PX = 20

# Anatomical segments (e.g. "mid LAD") can span a long stretch of vessel, long
# enough to hold two separate real lesions (tandem/serial stenoses). Detections
# that share a segment but sit farther apart than this are treated as distinct
# lesions rather than collapsed into one track -- see _split_by_proximity.
LESION_PROXIMITY_PX = 45

# Key frames are picked to guarantee every SEVERE/SIGNIFICANT lesion track has
# supporting evidence in at least one frame (see select_key_frames), capped at
# this many so the UI never has to lay out an unbounded number of panels.
KEY_FRAME_MAX_COUNT = 5
KEY_FRAME_MIN_GAP = 3


@dataclass
class FrameRecord:
    frame_idx: int
    img_gray: np.ndarray
    img_rgb: np.ndarray
    bw_mask: np.ndarray
    dt: np.ndarray
    branches: list
    lesions: list  # lesion dicts, each carries a "frame_idx" key


@dataclass
class LesionTrack:
    track_id: str
    label: str
    artery: str
    group: str
    detections: List[dict] = field(default_factory=list)

    @property
    def representative(self) -> dict:
        return max(self.detections, key=lambda d: d.get("confidence", 0.0))

    @property
    def frame_indices(self):
        return sorted({d["frame_idx"] for d in self.detections})


@dataclass
class AngleResult:
    angle_label: str
    video_path: str
    n_frames_total: int
    n_frames_analyzed: int
    frame_records: List[FrameRecord]
    tracks: List[LesionTrack]
    summary_frame_idx: Optional[int]
    key_frame_indices: List[int]
    has_localization: bool

    def get_frame_record(self, frame_idx: int) -> Optional[FrameRecord]:
        for rec in self.frame_records:
            if rec.frame_idx == frame_idx:
                return rec
        return None


def _analyze_frame_iterable(frame_iter, n_frames_total: int, angle_label: str, source_label: str,
                            seg_model: SegmentationModel,
                            loc_model: Optional[LocalizationModel],
                            cfg: QCAConfig, threshold: float,
                            progress_cb: Optional[Callable[[int, int], None]]) -> AngleResult:
    """
    Shared whole-run QCA analysis loop: runs segmentation + QCA over every
    frame yielded by frame_iter. Only frames containing >=1 detected lesion
    are retained in memory, then detections are clustered into per-lesion
    tracks across the whole run. Used by both analyze_angle_video() (reads
    from a video file) and analyze_frame_list() (an already-decoded list,
    e.g. from a loaded DICOM series).
    """
    frame_records: List[FrameRecord] = []
    loc_class_map = None
    loc_confidence_map = None
    frame_idx = 0

    for frame in frame_iter:
        img_rgb_original, img_rgb_enhanced, img_batch, img_gray = preprocess_frame(frame)
        mask_binary = segment_frame(seg_model, img_batch, threshold)

        if loc_model is not None and (
            loc_class_map is None or frame_idx % LOC_FRAME_INTERVAL == 0
        ):
            try:
                loc_class_map, loc_confidence_map = run_localization_frame(loc_model, img_rgb_enhanced, mask_binary)
            except Exception:
                loc_class_map, loc_confidence_map = None, None

        branches, lesions, dt, bw = run_qca_frame(
            img_gray, mask_binary, cfg,
            class_map=loc_class_map, confidence_map=loc_confidence_map,
            use_merged_labels=(loc_model.use_merged_labels if loc_model is not None else False),
        )

        if lesions:
            for les in lesions:
                les["frame_idx"] = frame_idx
            frame_records.append(FrameRecord(
                frame_idx=frame_idx, img_gray=img_gray, img_rgb=img_rgb_original,
                bw_mask=bw, dt=dt, branches=branches, lesions=lesions,
            ))

        frame_idx += 1
        if progress_cb is not None:
            progress_cb(frame_idx, n_frames_total)

    has_localization = loc_model is not None
    tracks = build_lesion_tracks(frame_records, has_localization)
    summary_frame_idx = select_summary_frame(tracks, frame_records)
    key_frame_indices = select_key_frames(frame_records, tracks)

    return AngleResult(
        angle_label=angle_label,
        video_path=source_label,
        n_frames_total=n_frames_total,
        n_frames_analyzed=frame_idx,
        frame_records=frame_records,
        tracks=tracks,
        summary_frame_idx=summary_frame_idx,
        key_frame_indices=key_frame_indices,
        has_localization=has_localization,
    )


def analyze_angle_video(video_path: str, angle_label: str,
                        seg_model: SegmentationModel,
                        loc_model: Optional[LocalizationModel],
                        cfg: QCAConfig,
                        threshold: float = 0.5,
                        progress_cb: Optional[Callable[[int, int], None]] = None) -> AngleResult:
    """
    Runs segmentation + QCA over every frame of one angiographic view video.
    Only frames containing >=1 detected lesion are retained in memory, then
    detections are clustered into per-lesion tracks across the whole run.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    n_frames_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    def _frames():
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                yield frame
        finally:
            cap.release()

    return _analyze_frame_iterable(
        _frames(), n_frames_total, angle_label, video_path,
        seg_model, loc_model, cfg, threshold, progress_cb,
    )


def analyze_frame_list(frames: List[np.ndarray], angle_label: str,
                       seg_model: SegmentationModel,
                       loc_model: Optional[LocalizationModel],
                       cfg: QCAConfig,
                       threshold: float = 0.5,
                       progress_cb: Optional[Callable[[int, int], None]] = None,
                       source_label: Optional[str] = None) -> AngleResult:
    """
    Same whole-run QCA analysis as analyze_angle_video(), but over an
    already-decoded list of frames (e.g. from dicom_loader.load_series_frames())
    instead of opening a video file -- used by the DICOM analysis page, which
    already has the series' frames loaded in memory.

    `source_label` should identify the actual source (e.g. the DICOM series
    file path) when the caller has one, so a saved AngleResult can later be
    recognized as "the same series analyzed again" rather than just a generic
    frame count -- falls back to a generic in-memory description if omitted.
    """
    return _analyze_frame_iterable(
        iter(frames), len(frames), angle_label, source_label or f"<{len(frames)} in-memory frames>",
        seg_model, loc_model, cfg, threshold, progress_cb,
    )


def _track_key(les: dict, has_localization: bool):
    """Clustering key for grouping the same anatomical lesion across frames."""
    if has_localization and "localization" in les:
        return ("segment", les["localization"]["segment_id"])
    y, x = les["min_pt"]
    return ("xy", int(y // SPATIAL_BUCKET_PX), int(x // SPATIAL_BUCKET_PX))


def _split_by_proximity(detections: list, threshold_px: float) -> List[list]:
    """
    Greedily splits a list of same-segment detections into spatially distinct
    groups: each detection joins the nearest existing group if within
    threshold_px of that group's running centroid, otherwise it starts a new
    group. Frame-to-frame jitter of the *same* physical lesion is much
    smaller than this threshold, while two genuinely separate lesions (e.g.
    tandem stenoses) that merely happen to share a broad anatomical segment
    sit much farther apart -- so this recovers them as distinct lesions
    instead of one track silently absorbing both.
    """
    groups = []  # each: {"members": [...], "centroid": (y, x)}
    for det in detections:
        y, x = det["min_pt"]
        best, best_dist = None, None
        for g in groups:
            cy, cx = g["centroid"]
            d = ((cy - y) ** 2 + (cx - x) ** 2) ** 0.5
            if d <= threshold_px and (best_dist is None or d < best_dist):
                best, best_dist = g, d
        if best is not None:
            best["members"].append(det)
            n = len(best["members"])
            cy, cx = best["centroid"]
            best["centroid"] = (cy + (y - cy) / n, cx + (x - cx) / n)
        else:
            groups.append({"members": [det], "centroid": (float(y), float(x))})
    return [g["members"] for g in groups]


def build_lesion_tracks(frame_records: List[FrameRecord], has_localization: bool) -> List[LesionTrack]:
    """
    Clusters per-frame lesion detections into per-lesion tracks. Prefers the
    AHA/SYNTAX segment id from the localization model (robust to per-frame
    pixel jitter); falls back to a spatial grid bucket of the lesion's minimum
    point when no localization model was supplied. Segment-id alone can span
    two real, separate lesions (a long segment holding tandem stenoses), so
    each segment bucket is further split by spatial proximity (see
    _split_by_proximity) before becoming tracks.
    """
    buckets = {}
    for rec in frame_records:
        for les in rec.lesions:
            key = _track_key(les, has_localization)
            buckets.setdefault(key, []).append(les)

    tracks = []
    for dets in buckets.values():
        for group in _split_by_proximity(dets, LESION_PROXIMITY_PX):
            rep = max(group, key=lambda d: d.get("confidence", 0.0))
            if has_localization and "localization" in rep:
                loc = rep["localization"]
                label, artery, group_name = loc["label"], loc["artery"], loc["group"]
            else:
                y, x = rep["min_pt"]
                label, artery, group_name = f"unlabeled region near ({x}, {y})", "unknown", "unknown"

            tracks.append(LesionTrack(
                track_id=f"L{len(tracks) + 1}", label=label, artery=artery, group=group_name, detections=group,
            ))

    tracks.sort(key=lambda t: t.representative["DS_percent"], reverse=True)
    for i, t in enumerate(tracks, start=1):
        t.track_id = f"L{i}"
    return tracks



# A frame only qualifies for the summary diagram if its opacified/traceable
# vessel length is at least this fraction of the best-opacified frame in the
# run -- otherwise a partially-filled frame (early/late in the contrast
# injection, "ink" not yet spread) could win purely on an incidental lesion
# coincidence while looking washed-out and largely vessel-free.
MIN_VESSEL_COVERAGE_FRACTION = 0.7


def _vessel_coverage(rec: FrameRecord) -> int:
    """Total traced skeleton length across all branches -- a proxy for how
    much of the coronary tree is opacified and visible in this frame."""
    return sum(len(b) for b in rec.branches)


def select_summary_frame(tracks: List[LesionTrack], frame_records: List[FrameRecord]) -> Optional[int]:
    """
    Picks the single stored frame to use for the per-angle 2D overview
    diagram. First narrows to well-opacified frames (vessel coverage within
    MIN_VESSEL_COVERAGE_FRACTION of the best frame in the run), then among
    those picks the one showing the most distinct lesion tracks simultaneously
    (tie-break: highest summed confidence) -- so the diagram never favors a
    partially-contrasted, hard-to-read frame just because it happens to
    contain one extra incidental lesion.
    """
    if not frame_records:
        return None

    max_coverage = max((_vessel_coverage(rec) for rec in frame_records), default=0)
    if max_coverage > 0:
        threshold = max_coverage * MIN_VESSEL_COVERAGE_FRACTION
        candidates = [rec for rec in frame_records if _vessel_coverage(rec) >= threshold]
    else:
        candidates = frame_records

    track_of_lesion = {}
    for t in tracks:
        for les in t.detections:
            track_of_lesion[(les["frame_idx"], id(les))] = t.track_id

    best_idx, best_score = None, (-1, -1.0)
    for rec in candidates:
        track_ids_here = set()
        conf_sum = 0.0
        for les in rec.lesions:
            tid = track_of_lesion.get((rec.frame_idx, id(les)))
            if tid is not None:
                track_ids_here.add(tid)
            conf_sum += float(les.get("confidence", 0.0))
        score = (len(track_ids_here), conf_sum)
        if score > best_score:
            best_score = score
            best_idx = rec.frame_idx

    return best_idx


def select_key_frames(frame_records: List[FrameRecord], tracks: List[LesionTrack],
                      max_count: int = KEY_FRAME_MAX_COUNT,
                      min_gap: int = KEY_FRAME_MIN_GAP) -> List[int]:
    """
    Greedily picks the smallest set of frames (capped at max_count) that
    shows every clinically-actionable (SEVERE/SIGNIFICANT) lesion track at
    least once -- so each finding has real supporting evidence in the
    key-frames view, instead of being selected purely by vessel coverage and
    hoping every lesion happens to land in the top few frames (a run with
    more distinct significant lesions than frames picked could otherwise
    silently drop some of them).

    Each step picks the frame that covers the most not-yet-covered
    significant tracks; ties (and any picks once everything is already
    covered) are broken by vessel coverage, so frame quality still matters
    wherever it doesn't conflict with coverage. Picks prefer to stay
    >= min_gap frames apart (to avoid near-identical consecutive frames from
    the same opacification peak), but will ignore that spacing if it's the
    only way left to cover a remaining track.

    If there are no significant tracks at all (nothing to cover), falls back
    to the highest-coverage frames so the panel isn't empty.
    """
    if not frame_records:
        return []

    track_of_lesion = {id(les): t for t in tracks for les in t.detections}
    significant_ids = {t.track_id for t in tracks if t.representative["severity"] in ("SEVERE", "SIGNIFICANT")}

    frame_tracks = {}
    for rec in frame_records:
        ids_here = {
            track_of_lesion[id(les)].track_id
            for les in rec.lesions
            if id(les) in track_of_lesion and track_of_lesion[id(les)].track_id in significant_ids
        }
        frame_tracks[rec.frame_idx] = ids_here

    coverage = {rec.frame_idx: _vessel_coverage(rec) for rec in frame_records}

    picked: List[int] = []
    covered: set = set()
    remaining = list(frame_records)

    while remaining and len(picked) < max_count and covered != significant_ids:
        # Coverage always wins: min_gap is only a tie-break among frames that
        # would add the same number of new tracks, never a reason to exclude
        # the one frame that happens to be the sole source of a still-missing
        # track (excluding it outright can leave that track uncovered forever
        # even with picks to spare under max_count).
        def _score(rec):
            new_tracks = len(frame_tracks[rec.frame_idx] - covered)
            gap_ok = all(abs(rec.frame_idx - p) >= min_gap for p in picked)
            return (new_tracks, int(gap_ok), coverage[rec.frame_idx])

        best = max(remaining, key=_score)
        if covered and _score(best)[0] == 0:
            break  # nothing left would add new coverage -- stop spending picks
        picked.append(best.frame_idx)
        covered |= frame_tracks[best.frame_idx]
        remaining.remove(best)

    if not picked:
        ranked = sorted(frame_records, key=_vessel_coverage, reverse=True)
        for rec in ranked:
            if len(picked) >= max_count:
                break
            if all(abs(rec.frame_idx - p) >= min_gap for p in picked):
                picked.append(rec.frame_idx)
        if len(picked) < max_count:
            for rec in ranked:
                if len(picked) >= max_count:
                    break
                if rec.frame_idx not in picked:
                    picked.append(rec.frame_idx)

    return picked


def draw_angle_summary_bgr(rec: FrameRecord, tracks: List[LesionTrack]) -> np.ndarray:
    """
    Renders one frame's vessel mask + SEVERE/SIGNIFICANT lesions using each
    lesion's track-representative DS%/severity (the stable, whole-run value),
    not this single frame's own noisier per-frame reading. Shared by the PDF
    report and the in-app key-frames summary so the two never show different
    numbers for the same lesion.
    """
    vis = cv2.cvtColor(rec.img_gray, cv2.COLOR_GRAY2BGR)

    contours, _ = cv2.findContours((rec.bw_mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, contours, -1, (0, 255, 0), 1)

    for bi, branch in enumerate(rec.branches):
        color = _BRANCH_COLORS[bi % len(_BRANCH_COLORS)]
        for (y, x) in branch:
            vis[y, x] = color

    track_of_lesion = {id(les): t for t in tracks for les in t.detections}
    drawn_track_ids = set()

    for les in rec.lesions:
        t = track_of_lesion.get(id(les))
        if t is None or t.track_id in drawn_track_ids:
            continue
        rep = t.representative
        sev = rep["severity"]
        if sev not in ("SEVERE", "SIGNIFICANT"):
            continue
        drawn_track_ids.add(t.track_id)

        color = _SEVERITY_BGR.get(sev, _SEVERITY_BGR["MILD"])
        radius = _SEVERITY_RADIUS.get(sev, 1)
        branch = les["branch"]
        L, R = les["L_idx"], les["R_idx"]
        for i in range(L, min(R + 1, len(branch))):
            y, x = branch[i]
            cv2.circle(vis, (x, y), radius, color, -1)

        y0, x0 = les["min_pt"]
        occ = " [OCC]" if rep.get("total_occlusion") else ""
        label = f"{t.track_id} {rep['DS_percent']:.1f}% {sev[:3]}{occ}"
        font_scale = 0.45 if sev == "SEVERE" else 0.38
        cv2.putText(vis, label, (x0 + 6, y0 - 4), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1, cv2.LINE_AA)

    return vis


def _rects_overlap(a, b, pad=3):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return not (ax2 + pad < bx1 or bx2 + pad < ax1 or ay2 + pad < by1 or by2 + pad < ay1)


# 8 compass directions to radiate label candidates out from a circle in,
# closest/most-natural first (right and upper-right read best for a
# left-to-right label, so they're tried before the others).
_LABEL_DIRECTIONS = [
    (1, 0), (1, -1), (1, 1), (0, -1), (-1, -1), (-1, 0), (0, 1), (-1, 1),
]


def _find_label_box(center, text_size, base_radius, occupied, img_shape):
    """
    Finds a label bounding box near `center` that avoids every rect in
    `occupied` (other circles' bounding boxes and already-placed labels),
    by radiating candidate positions out in 8 directions at increasing
    distance. Falls back to the least-overlapping candidate tried if every
    one collides, so placement never fails outright.

    Returns (box, anchor_xy) where box = (x1, y1, x2, y2) and anchor_xy is
    the point closest to the circle -- used to draw a leader line back to it
    when the label ended up displaced from its default position.
    """
    tw, th = text_size
    h, w = img_shape[:2]
    cx, cy = center

    best_box, best_anchor, best_overlap = None, None, None

    for dist in (base_radius + 8, base_radius + 28, base_radius + 52, base_radius + 80, base_radius + 115):
        for dx, dy in _LABEL_DIRECTIONS:
            px, py = cx + dx * dist, cy + dy * dist

            x1 = px if dx >= 0 else px - tw
            x2 = px + tw if dx >= 0 else px
            y1 = py - th if dy <= 0 else py
            y2 = py if dy <= 0 else py + th

            x1c, x2c = np.clip([x1, x2], 4, w - 4)
            y1c, y2c = np.clip([y1, y2], 4, h - 4)
            if x2c - x1c < tw * 0.6 or y2c - y1c < th * 0.6:
                continue  # clipped too much to be readable, skip this candidate
            box = (x1c, y1c, x2c, y2c)

            overlap_count = sum(1 for r in occupied if _rects_overlap(box, r))
            if overlap_count == 0:
                anchor = (px if dx >= 0 else px, py)
                return box, (float(np.clip(px, 0, w)), float(np.clip(py, 0, h)))
            if best_overlap is None or overlap_count < best_overlap:
                best_box, best_anchor, best_overlap = box, (px, py), overlap_count

    return best_box, best_anchor


def build_frame_lesion_specs(rec: FrameRecord, tracks: List[LesionTrack]) -> List[dict]:
    """
    Circle+tag specs for one frame's OWN detected lesions (deduplicated to
    one entry per track, in case this frame independently detected the same
    track on more than one branch -- e.g. at a bifurcation), using each
    track's whole-run representative severity/DS% but THIS frame's own
    captured position -- always accurate, since it's the exact frame that
    measurement came from.
    """
    track_of_lesion = {id(les): t for t in tracks for les in t.detections}
    drawn_track_ids = set()
    specs = []
    for les in rec.lesions:
        t = track_of_lesion.get(id(les))
        if t is None or t.track_id in drawn_track_ids:
            continue
        rep = t.representative
        if rep["severity"] not in ("SEVERE", "SIGNIFICANT"):
            continue
        drawn_track_ids.add(t.track_id)

        y0, x0 = les["min_pt"]
        radius = 16 if rep["severity"] == "SEVERE" else 13
        specs.append({
            "center": (x0, y0), "radius": radius,
            "color": _SEVERITY_BGR.get(rep["severity"], _SEVERITY_BGR["MILD"]),
            "label": t.track_id, "severity": rep["severity"],
        })
    return specs


def draw_frame_stenosis_only(rec: FrameRecord, tracks: List[LesionTrack]) -> np.ndarray:
    """
    Plain-frame view of one frame's own detected stenosis: the original
    image plus a circle + short track-id tag (e.g. "L1") for each
    significant lesion -- no vessel mask outline, no colored branch
    skeleton. Used for the Key Frames panels so each picture stays
    uncluttered; full detail per tag belongs in that panel's caption text
    instead of crowding the image.
    """
    background_bgr = cv2.cvtColor(rec.img_rgb, cv2.COLOR_RGB2BGR)
    return draw_lesion_markers_bgr(background_bgr, build_frame_lesion_specs(rec, tracks))


def build_live_lesion_specs(lesions: List[dict]) -> List[dict]:
    """
    Builds circle+label specs directly from one frame's raw QCA lesions (no
    whole-run track clustering available yet during live Play) -- used by
    the live per-frame view, which has no LesionTrack to pull a stable
    representative measurement from, so each lesion's own DS%/severity is
    shown as-is.
    """
    specs = []
    for les in lesions:
        sev = les.get("severity", "MILD")
        if sev not in ("SEVERE", "SIGNIFICANT"):
            continue
        y0, x0 = les["min_pt"]
        radius = 16 if sev == "SEVERE" else 13
        occ = " [OCC]" if les.get("total_occlusion") else ""
        loc = les.get("localization")
        loc_part = f" - {loc['label']}" if loc and loc.get("artery", "unknown") != "unknown" else ""
        label = f"{les['DS_percent']:.0f}% {sev}{occ}{loc_part}"
        specs.append({
            "center": (x0, y0), "radius": radius,
            "color": _SEVERITY_BGR.get(sev, _SEVERITY_BGR["MILD"]),
            "label": label, "severity": sev,
        })
    return specs


def draw_lesion_markers_bgr(base_bgr: np.ndarray, lesion_specs: List[dict], show_labels: bool = True) -> np.ndarray:
    """
    Draws a circle for each spec in `lesion_specs` (see build_live_lesion_specs
    / draw_stenosis_overview's track-based spec building) onto a copy of
    `base_bgr`. When show_labels is True (default), also draws a
    collision-avoided severity/DS%/location label next to each circle (see
    _find_label_box), with a thin leader line back to the circle whenever a
    label had to be pushed away to avoid overlapping another. When False,
    only the circles are drawn -- e.g. the live Play view, which wants a
    clean, uncluttered marker for every finding without text competing for
    space while frames advance.
    """
    vis = base_bgr.copy()

    # Most severe first, so SEVERE lesions win the closest/cleanest label spot.
    lesion_specs = sorted(lesion_specs, key=lambda s: 0 if s["severity"] == "SEVERE" else 1)

    occupied = []
    for spec in lesion_specs:
        x0, y0 = spec["center"]
        r = spec["radius"]
        cv2.circle(vis, (x0, y0), r, spec["color"], 2)
        occupied.append((x0 - r, y0 - r, x0 + r, y0 + r))

    if not show_labels:
        return vis

    font_scale = 0.48
    for spec in lesion_specs:
        x0, y0 = spec["center"]
        r, color, label = spec["radius"], spec["color"], spec["label"]
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)

        box, anchor = _find_label_box((x0, y0), (tw, th + baseline), r, occupied, vis.shape)
        if box is None:
            continue
        x1, y1, x2, y2 = (int(v) for v in box)
        occupied.append(box)

        if anchor is not None and (abs(anchor[0] - x0) > r + 14 or abs(anchor[1] - y0) > r + 14):
            edge_x = x0 + int(r * np.sign(anchor[0] - x0 or 1))
            edge_y = y0 + int(r * np.sign(anchor[1] - y0 or 1))
            cv2.line(vis, (edge_x, edge_y), (int(anchor[0]), int(anchor[1])), color, 1, cv2.LINE_AA)

        tx, ty = x1 + 2, y2 - baseline - 2
        # 1px black halo in all 8 directions at the *same* thickness as the final
        # colored pass -- keeps the label readable against both bright and dark
        # parts of the angiogram without a solid box. (A single thicker black
        # pass underneath looks doubled/smeared on this small stroke-based font,
        # since a different thickness rasterizes each glyph's curves differently.)
        for dx, dy in ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)):
            cv2.putText(vis, label, (tx + dx, ty + dy), cv2.FONT_HERSHEY_SIMPLEX,
                       font_scale, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(vis, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1, cv2.LINE_AA)

    return vis


def draw_live_stenosis_overlay(img_gray: np.ndarray, bw: np.ndarray, branches: list,
                               lesions: List[dict], show_labels: bool = False) -> np.ndarray:
    """
    Live per-frame QCA visualization for the DICOM analysis viewer: vessel
    mask outline + colored branch skeleton (context, same as qca.draw_overlay),
    plus a plain circle around every significant lesion (see
    build_live_lesion_specs) instead of qca.draw_overlay's small filled-dot
    markers -- easier to read at a glance during continuous Play. Labels are
    off by default here (unlike draw_stenosis_overview) since text next to
    every circle, updating as frames advance, gets busy fast during Play;
    pass show_labels=True to add them back.
    """
    vis = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

    contours, _ = cv2.findContours((bw > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, contours, -1, (0, 255, 0), 1)

    for bi, branch in enumerate(branches):
        color = _BRANCH_COLORS[bi % len(_BRANCH_COLORS)]
        for (y, x) in branch:
            vis[y, x] = color

    return draw_lesion_markers_bgr(vis, build_live_lesion_specs(lesions), show_labels=show_labels)


def generate_reasoning(track: LesionTrack, cfg: QCAConfig) -> str:
    """Builds an explainable-AI reasoning paragraph for a lesion track's representative measurement."""
    les = track.representative
    ds = les["DS_percent"]
    sev = les["severity"]

    if les.get("MLD_mm") is not None:
        mld, rvd, unit = les["MLD_mm"], les["RVD_mm"], "mm"
    else:
        mld, rvd, unit = les["MLD_px"], les["RVD_px"], "px"

    loc_sentence = ""
    if track.artery != "unknown":
        loc_sentence = f" in the {track.label} ({track.artery})"
        loc = les.get("localization")
        if loc:
            loc_sentence += f", localization confidence {loc['confidence']:.2f}"

    occ_sentence = ""
    if les.get("total_occlusion"):
        occ_sentence = " This segment shows a total occlusion (no measurable lumen)."

    conf = les.get("confidence")
    conf_sentence = ""
    if conf is not None:
        parts = []
        if les.get("edge_sharpness") is not None:
            parts.append(f"edge sharpness {les['edge_sharpness']:.2f}")
        if les.get("ref_quality") is not None:
            parts.append(f"reference-segment quality {les['ref_quality']:.2f}")
        if les.get("len_score") is not None:
            parts.append(f"lesion-span stability {les['len_score']:.2f}")
        basis = ", ".join(parts) if parts else "the underlying diameter profile"
        conf_sentence = f" Detection confidence {conf:.2f}, derived from {basis}."

    method_note = ""
    if les.get("method") == "dt":
        method_note = (
            " Measured from the mask distance transform after the orthogonal edge "
            "estimate was rejected as unreliable at this point (e.g. vessel overlap or high curvature)."
        )

    n_frames = len(track.frame_indices)
    frame_note = f" Consistently detected across {n_frames} analyzed frame(s) in this view." if n_frames > 1 else ""

    return (
        f"{ds:.1f}% diameter stenosis (MLD {mld:.1f}{unit} vs reference vessel diameter "
        f"{rvd:.1f}{unit}){loc_sentence} — classified {sev} per the JACIT/ARC-2 hierarchical "
        f"consensus (Severe ≥{cfg.severe_threshold:.0f}% DS regardless of symptoms; "
        f"Significant ≥{cfg.significant_threshold:.0f}% DS if symptomatic or a positive "
        f"functional test; Moderate ≥{cfg.moderate_threshold:.0f}% DS)."
        f"{occ_sentence}{conf_sentence}{method_note}{frame_note}"
    )
