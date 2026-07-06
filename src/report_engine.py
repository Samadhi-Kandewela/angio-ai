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

from qca import QCAConfig
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

    return AngleResult(
        angle_label=angle_label,
        video_path=source_label,
        n_frames_total=n_frames_total,
        n_frames_analyzed=frame_idx,
        frame_records=frame_records,
        tracks=tracks,
        summary_frame_idx=summary_frame_idx,
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
                       progress_cb: Optional[Callable[[int, int], None]] = None) -> AngleResult:
    """
    Same whole-run QCA analysis as analyze_angle_video(), but over an
    already-decoded list of frames (e.g. from dicom_loader.load_series_frames())
    instead of opening a video file -- used by the DICOM analysis page, which
    already has the series' frames loaded in memory.
    """
    return _analyze_frame_iterable(
        iter(frames), len(frames), angle_label, f"<{len(frames)} in-memory frames>",
        seg_model, loc_model, cfg, threshold, progress_cb,
    )


def _track_key(les: dict, has_localization: bool):
    """Clustering key for grouping the same anatomical lesion across frames."""
    if has_localization and "localization" in les:
        return ("segment", les["localization"]["segment_id"])
    y, x = les["min_pt"]
    return ("xy", int(y // SPATIAL_BUCKET_PX), int(x // SPATIAL_BUCKET_PX))


def build_lesion_tracks(frame_records: List[FrameRecord], has_localization: bool) -> List[LesionTrack]:
    """
    Clusters per-frame lesion detections into per-lesion tracks. Prefers the
    AHA/SYNTAX segment id from the localization model (robust to per-frame
    pixel jitter); falls back to a spatial grid bucket of the lesion's minimum
    point when no localization model was supplied.
    """
    buckets = {}
    for rec in frame_records:
        for les in rec.lesions:
            key = _track_key(les, has_localization)
            buckets.setdefault(key, []).append(les)

    tracks = []
    for i, dets in enumerate(buckets.values()):
        rep = max(dets, key=lambda d: d.get("confidence", 0.0))
        if has_localization and "localization" in rep:
            loc = rep["localization"]
            label, artery, group = loc["label"], loc["artery"], loc["group"]
        else:
            y, x = rep["min_pt"]
            label, artery, group = f"unlabeled region near ({x}, {y})", "unknown", "unknown"

        tracks.append(LesionTrack(
            track_id=f"L{i + 1}", label=label, artery=artery, group=group, detections=dets,
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
