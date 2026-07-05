"""
End-to-end research pipeline for two-view coronary 3D reconstruction.

This script is intentionally offline and reproducible. It does not modify the
desktop app. For clinical work, keep manual review of the chosen view/frame pair.

Example:
    python scripts/dicom_3d_pipeline.py ^
        --dicom-dir dicom_sample ^
        --view-a-index 0 --frame-a 30 ^
        --view-b-index 2 --frame-b 22
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import pydicom
from scipy.interpolate import RBFInterpolator
from skimage.morphology import skeletonize


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "checkpoints" / "mobileunetv3" / "mobileunetv3_augmented_best.onnx"


@dataclass
class Clip:
    path: Path
    index: int
    frames: int
    rows: int
    cols: int
    primary: float
    secondary: float
    sdd: float
    spd: float
    spacing: tuple[float, float]
    acquisition_time: str


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def find_xa_clips(dicom_dir: Path) -> list[Clip]:
    clips = []
    for path in dicom_dir.rglob("*"):
        if not path.is_file():
            continue
        try:
            ds = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
        except Exception:
            continue
        if getattr(ds, "Modality", "") != "XA":
            continue
        if not hasattr(ds, "NumberOfFrames") or not hasattr(ds, "PositionerPrimaryAngle"):
            continue
        try:
            frames = int(ds.NumberOfFrames)
            rows = int(ds.Rows)
            cols = int(ds.Columns)
            spacing = tuple(float(v) for v in ds.ImagerPixelSpacing)
            clips.append(
                Clip(
                    path=path,
                    index=len(clips),
                    frames=frames,
                    rows=rows,
                    cols=cols,
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
    return sorted(clips, key=lambda c: c.acquisition_time)


def choose_default_pair(clips: list[Clip]) -> tuple[int, int]:
    if len(clips) >= 3:
        # For left-coronary studies, the first and third views often give a
        # useful angular separation. The UI/researcher should still review this.
        return 0, 2
    best = None
    for i, a in enumerate(clips):
        for j, b in enumerate(clips):
            if i >= j:
                continue
            sep = math.hypot(a.primary - b.primary, a.secondary - b.secondary)
            if best is None or sep > best[0]:
                best = (sep, i, j)
    if best is None:
        raise RuntimeError("Need at least two XA clips for two-view 3D reconstruction.")
    return best[1], best[2]


def default_frame(clip: Clip, fraction: float) -> int:
    return int(np.clip(round(clip.frames * fraction), 1, clip.frames))


def read_frame(clip: Clip, frame_number: int) -> np.ndarray:
    ds = pydicom.dcmread(str(clip.path), force=True)
    arr = ds.pixel_array
    if arr.ndim == 2:
        arr = arr[None, :, :]
    frame = arr[frame_number - 1]
    if frame.dtype != np.uint8:
        f = frame.astype(np.float32)
        lo, hi = np.percentile(f, [1, 99])
        frame = np.clip((f - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
    return cv2.resize(frame, (512, 512), interpolation=cv2.INTER_AREA)


def preprocess_for_model(gray512: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rgb = cv2.cvtColor(gray512, cv2.COLOR_GRAY2RGB)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = cv2.cvtColor(cv2.merge((clahe.apply(l), a, b)), cv2.COLOR_LAB2RGB)
    batch = np.expand_dims(np.transpose(enhanced.astype(np.float32) / 255.0, (2, 0, 1)), axis=0)
    return batch.astype(np.float32), rgb


def segment_frame(session, input_name: str, gray512: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    batch, rgb = preprocess_for_model(gray512)
    prob = sigmoid(session.run(None, {input_name: batch})[0]).squeeze().astype(np.float32)
    mask = (prob > threshold).astype(np.uint8) * 255
    overlay = rgb.copy()
    overlay[mask > 0] = np.array([255, 0, 0], dtype=np.uint8)
    blended = cv2.addWeighted(overlay, 0.4, rgb, 0.6, 0)
    return mask, cv2.cvtColor(blended, cv2.COLOR_RGB2BGR)


def clean_mask(mask: np.ndarray) -> np.ndarray:
    _, bw = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=1)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel, iterations=1)
    count, labels, stats, _ = cv2.connectedComponentsWithStats((bw > 0).astype(np.uint8), 8)
    out = np.zeros_like(bw)
    for idx in range(1, count):
        if stats[idx, cv2.CC_STAT_AREA] >= 120:
            out[labels == idx] = 255
    return out


NEI8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def adjacency(skel: np.ndarray):
    ys, xs = np.where(skel > 0)
    pts = set(zip(ys.tolist(), xs.tolist()))
    adj = {p: [] for p in pts}
    for y, x in list(pts):
        for dy, dx in NEI8:
            q = (y + dy, x + dx)
            if q in pts:
                adj[(y, x)].append(q)
    return adj


def trace_branches(adj, min_len=20):
    degree = {node: len(neighbors) for node, neighbors in adj.items()}
    starts = [node for node, deg in degree.items() if deg != 2]
    visited = set()
    branches = []
    for start in starts:
        for neighbor in adj[start]:
            edge = tuple(sorted([start, neighbor]))
            if edge in visited:
                continue
            path = [start, neighbor]
            visited.add(edge)
            prev, cur = start, neighbor
            while degree.get(cur, 0) == 2:
                next_nodes = [node for node in adj[cur] if node != prev]
                if not next_nodes:
                    break
                nxt = next_nodes[0]
                edge = tuple(sorted([cur, nxt]))
                if edge in visited:
                    break
                visited.add(edge)
                path.append(nxt)
                prev, cur = cur, nxt
            if len(path) >= min_len:
                branches.append(path)
    return sorted(branches, key=len, reverse=True)


def extract_graph(mask: np.ndarray):
    clean = clean_mask(mask)
    skel = (skeletonize(clean > 0).astype(np.uint8)) * 255
    dt = cv2.distanceTransform((clean > 0).astype(np.uint8), cv2.DIST_L2, 5)
    branches = []
    for branch_id, path in enumerate(trace_branches(adjacency(skel))):
        pts = np.asarray(path, dtype=np.float32)
        length = float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum()) if len(pts) > 1 else 0.0
        ys = pts[:, 0].astype(int)
        xs = pts[:, 1].astype(int)
        diam = 2.0 * dt[ys, xs]
        branches.append(
            {
                "branch_id": branch_id,
                "points": len(path),
                "length_px": length,
                "mean_diameter_px": float(np.mean(diam)),
                "centerline_yx": [[int(y), int(x)] for y, x in path],
            }
        )
    return clean, skel, dt, branches


def rotation_from_angles(primary_deg, secondary_deg):
    primary = math.radians(primary_deg)
    secondary = math.radians(secondary_deg)
    ry = np.array([[math.cos(primary), 0, math.sin(primary)], [0, 1, 0], [-math.sin(primary), 0, math.cos(primary)]])
    rx = np.array([[1, 0, 0], [0, math.cos(secondary), -math.sin(secondary)], [0, math.sin(secondary), math.cos(secondary)]])
    return ry @ rx


def view_from_clip(clip: Clip, name: str):
    return {
        "name": name,
        "primary_angle_deg": clip.primary,
        "secondary_angle_deg": clip.secondary,
        "source_detector_distance_mm": clip.sdd,
        "source_patient_distance_mm": clip.spd,
        "imager_pixel_spacing_mm": [clip.spacing[0], clip.spacing[1]],
        "image_size_px": [512, 512],
    }


def ray_from_pixel(point_yx, view):
    y, x = point_yx
    rows, cols = view["image_size_px"]
    spacing_y, spacing_x = view["imager_pixel_spacing_mm"]
    sdd = float(view["source_detector_distance_mm"])
    spd = float(view["source_patient_distance_mm"])
    det_x = (float(x) - (cols - 1) / 2.0) * spacing_x
    det_y = (float(y) - (rows - 1) / 2.0) * spacing_y
    detector = np.array([det_x, det_y, sdd - spd], dtype=np.float64)
    source = np.array([0.0, 0.0, -spd], dtype=np.float64)
    rot = rotation_from_angles(float(view["primary_angle_deg"]), float(view["secondary_angle_deg"]))
    source_w = rot @ source
    detector_w = rot @ detector
    direction = detector_w - source_w
    direction /= np.linalg.norm(direction)
    return source_w, direction


def closest_point_between_rays(source_a, dir_a, source_b, dir_b):
    mat = np.column_stack([dir_a, -dir_b])
    rhs = source_b - source_a
    params, *_ = np.linalg.lstsq(mat, rhs, rcond=None)
    pa = source_a + params[0] * dir_a
    pb = source_b + params[1] * dir_b
    return 0.5 * (pa + pb), float(np.linalg.norm(pa - pb))


def resample_polyline(points_yx, n):
    pts = np.asarray(points_yx, dtype=np.float64)
    if len(pts) <= 1:
        return np.repeat(pts, n, axis=0)
    dist = np.zeros(len(pts), dtype=np.float64)
    dist[1:] = np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))
    targets = np.linspace(0.0, dist[-1], n)
    return np.column_stack([np.interp(targets, dist, pts[:, 0]), np.interp(targets, dist, pts[:, 1])])


def score_pair(branch_a, branch_b, view_a, view_b, rev_a=False, rev_b=False, samples=48):
    pa = branch_a["centerline_yx"][::-1] if rev_a else branch_a["centerline_yx"]
    pb = branch_b["centerline_yx"][::-1] if rev_b else branch_b["centerline_yx"]
    samples = min(samples, max(20, min(len(pa), len(pb))))
    ra = resample_polyline(pa, samples)
    rb = resample_polyline(pb, samples)
    residuals = []
    points = []
    for a, b in zip(ra, rb):
        sa, da = ray_from_pixel(a, view_a)
        sb, db = ray_from_pixel(b, view_b)
        p3, res = closest_point_between_rays(sa, da, sb, db)
        residuals.append(res)
        points.append(p3)
    residuals = np.asarray(residuals)
    return {
        "samples_a": ra,
        "samples_b": rb,
        "points_3d": np.asarray(points),
        "mean": float(np.mean(residuals)),
        "median": float(np.median(residuals)),
        "p90": float(np.percentile(residuals, 90)),
        "max": float(np.max(residuals)),
    }


def classify(score, branch):
    if score["median"] <= 3.0 and score["p90"] <= 7.0:
        return "reliable"
    if score["median"] <= 5.5 and score["p90"] <= 12.0 and branch["length_px"] >= 55:
        return "usable"
    return "single_view_preserved"


def find_branch_matches(branches_a, branches_b, view_a, view_b):
    matches = []
    for branch_a in branches_a:
        best = None
        for branch_b in branches_b:
            for rev_a in (False, True):
                for rev_b in (False, True):
                    score = score_pair(branch_a, branch_b, view_a, view_b, rev_a, rev_b)
                    ratio = max(branch_a["length_px"], 1.0) / max(branch_b["length_px"], 1.0)
                    objective = score["median"] + 0.15 * score["p90"] + 1.5 * abs(math.log(ratio))
                    candidate = (objective, branch_b, rev_a, rev_b, score)
                    if best is None or objective < best[0]:
                        best = candidate
        _, branch_b, rev_a, rev_b, score = best
        matches.append(
            {
                "branch_a": branch_a,
                "branch_b": branch_b,
                "reverse_a": rev_a,
                "reverse_b": rev_b,
                "score": score,
                "status": classify(score, branch_a),
            }
        )
    return matches


def yx_to_xy(coords_yx):
    coords = np.asarray(coords_yx, dtype=np.float64)
    spacing = 0.2375
    center = np.array([255.5, 255.5], dtype=np.float64)
    x = (coords[:, 1] - center[1]) * spacing
    y = -(coords[:, 0] - center[0]) * spacing
    return np.column_stack([x, y])


def smooth_1d(values, window=9):
    if len(values) < window:
        return values
    pad = window // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    return np.convolve(padded, np.ones(window) / window, mode="valid")


def make_frame(tangent):
    tangent = tangent / max(np.linalg.norm(tangent), 1e-9)
    ref = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(tangent, ref))) > 0.9:
        ref = np.array([0.0, 1.0, 0.0])
    normal = np.cross(tangent, ref)
    normal /= max(np.linalg.norm(normal), 1e-9)
    binormal = np.cross(tangent, normal)
    binormal /= max(np.linalg.norm(binormal), 1e-9)
    return normal, binormal


def variable_tube(points, radii, segments=16):
    radii = smooth_1d(np.asarray(radii), 9)
    vertices, faces = [], []
    if len(points) < 2:
        return vertices, faces
    for idx, point in enumerate(points):
        tangent = points[min(idx + 1, len(points) - 1)] - points[max(idx - 1, 0)]
        normal, binormal = make_frame(tangent)
        for seg in range(segments):
            angle = 2.0 * np.pi * seg / segments
            vertices.append(point + float(radii[idx]) * (np.cos(angle) * normal + np.sin(angle) * binormal))
    for idx in range(len(points) - 1):
        a, b = idx * segments, (idx + 1) * segments
        for seg in range(segments):
            faces.append((a + seg, b + seg, b + ((seg + 1) % segments)))
            faces.append((a + seg, b + ((seg + 1) % segments), a + ((seg + 1) % segments)))
    return vertices, faces


def write_obj(path, parts):
    with open(path.with_suffix(".mtl"), "w", encoding="utf-8") as f:
        f.write("newmtl reliable\nKd 0.78 0.78 0.78\n")
        f.write("newmtl usable\nKd 0.45 0.65 1.0\n")
        f.write("newmtl single_view_preserved\nKd 1.0 0.58 0.12\n")
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"mtllib {path.with_suffix('.mtl').name}\n")
        f.write("# DICOM-to-3D pipeline output. Materials encode branch confidence.\n")
        offset = 1
        for name, material, vertices, faces in parts:
            f.write(f"o {name}\nusemtl {material}\n")
            for vertex in vertices:
                f.write(f"v {vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f}\n")
            for a, b, c in faces:
                f.write(f"f {a + offset} {b + offset} {c + offset}\n")
            offset += len(vertices)


def build_mesh(matches, dt_a, out_dir: Path, mag_factor: float):
    anchors_yx, anchors_xyz = [], []
    for match in matches:
        if match["status"] in {"reliable", "usable"}:
            anchors_yx.extend(match["score"]["samples_a"])
            anchors_xyz.extend(match["score"]["points_3d"])
    if len(anchors_yx) < 8:
        raise RuntimeError("Not enough two-view-supported branches to build depth model.")
    depth_model = RBFInterpolator(np.asarray(anchors_yx), np.asarray(anchors_xyz), neighbors=45, smoothing=6.0)
    object_px_mm = 0.2375 / max(mag_factor, 1e-6)
    anchor_z = np.asarray(anchors_xyz, dtype=np.float64)[:, 2]
    z_low = float(np.percentile(anchor_z, 5) - 8.0)
    z_high = float(np.percentile(anchor_z, 95) + 8.0)

    parts, report = [], []
    for match in matches:
        branch = match["branch_a"]
        status = match["status"]

        # Doctor-facing output preserves the full IM0 vessel-tree shape for
        # visual continuity. Stereo-supported branches inform depth and quality;
        # weak branches are kept but depth-clamped rather than discarded.
        coords = resample_polyline(branch["centerline_yx"], min(64, max(20, branch["points"])))
        xy = yx_to_xy(coords)
        z = depth_model(coords)[:, 2]
        if status == "single_view_preserved":
            z = np.clip(z, z_low, z_high)
        else:
            z = np.clip(z, z_low - 6.0, z_high + 6.0)
        points = np.column_stack([xy, z])

        y = np.clip(np.rint(coords[:, 0]).astype(int), 0, 511)
        x = np.clip(np.rint(coords[:, 1]).astype(int), 0, 511)
        radii = np.clip(smooth_1d(dt_a[y, x] * object_px_mm, 9), 0.16, 2.2)
        if status == "single_view_preserved":
            radii *= 0.85
        vertices, faces = variable_tube(points, radii)
        parts.append((f"branch_{branch['branch_id']:02d}_{status}", status, vertices, faces))
        min_r = float(np.min(radii))
        ref_r = float(np.percentile(radii, 80))
        report.append(
            {
                "branch_id": branch["branch_id"],
                "status": status,
                "matched_im2_branch_id": match["branch_b"]["branch_id"],
                "median_residual_mm": match["score"]["median"],
                "p90_residual_mm": match["score"]["p90"],
                "min_radius_mm": min_r,
                "ref_radius_mm_p80": ref_r,
                "estimated_diameter_stenosis_pct": max(0.0, (1.0 - min_r / max(ref_r, 1e-6)) * 100.0),
            }
        )
    write_obj(out_dir / "pipeline_hybrid_qca_tree.obj", parts)
    with open(out_dir / "pipeline_branch_quality_radius.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(report[0].keys()))
        writer.writeheader()
        writer.writerows(report)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dicom-dir", type=Path, default=ROOT / "dicom_sample")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "dicom_pipeline_output")
    parser.add_argument("--view-a-index", type=int)
    parser.add_argument("--view-b-index", type=int)
    parser.add_argument("--frame-a", type=int)
    parser.add_argument("--frame-b", type=int)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    clips = find_xa_clips(args.dicom_dir)
    if len(clips) < 2:
        raise RuntimeError("Need at least two usable XA cine clips in the DICOM folder.")

    idx_a, idx_b = (args.view_a_index, args.view_b_index) if args.view_a_index is not None and args.view_b_index is not None else choose_default_pair(clips)
    clip_a, clip_b = clips[idx_a], clips[idx_b]
    frame_a = args.frame_a or default_frame(clip_a, 0.86)
    frame_b = args.frame_b or default_frame(clip_b, 0.58)

    session = ort.InferenceSession(str(args.model))
    input_name = session.get_inputs()[0].name

    gray_a, gray_b = read_frame(clip_a, frame_a), read_frame(clip_b, frame_b)
    mask_a, overlay_a = segment_frame(session, input_name, gray_a, args.threshold)
    mask_b, overlay_b = segment_frame(session, input_name, gray_b, args.threshold)
    cv2.imwrite(str(out / "view_a_original.png"), gray_a)
    cv2.imwrite(str(out / "view_b_original.png"), gray_b)
    cv2.imwrite(str(out / "view_a_overlay.png"), overlay_a)
    cv2.imwrite(str(out / "view_b_overlay.png"), overlay_b)
    cv2.imwrite(str(out / "view_a_mask.png"), mask_a)
    cv2.imwrite(str(out / "view_b_mask.png"), mask_b)

    clean_a, skel_a, dt_a, branches_a = extract_graph(mask_a)
    clean_b, skel_b, dt_b, branches_b = extract_graph(mask_b)
    cv2.imwrite(str(out / "view_a_clean_mask.png"), clean_a)
    cv2.imwrite(str(out / "view_b_clean_mask.png"), clean_b)
    cv2.imwrite(str(out / "view_a_skeleton.png"), skel_a)
    cv2.imwrite(str(out / "view_b_skeleton.png"), skel_b)

    view_a = view_from_clip(clip_a, "view_a")
    view_b = view_from_clip(clip_b, "view_b")
    matches = find_branch_matches(branches_a, branches_b, view_a, view_b)
    report = build_mesh(matches, dt_a, out, getattr(pydicom.dcmread(str(clip_a.path), stop_before_pixels=True, force=True), "EstimatedRadiographicMagnificationFactor", 1.0))

    summary = {
        "dicom_dir": str(args.dicom_dir),
        "view_a": {**view_a, "clip_index": idx_a, "dicom_path": str(clip_a.path), "frame": frame_a},
        "view_b": {**view_b, "clip_index": idx_b, "dicom_path": str(clip_b.path), "frame": frame_b},
        "num_xa_clips_found": len(clips),
        "num_view_a_branches": len(branches_a),
        "num_view_b_branches": len(branches_b),
        "branch_status_counts": {
            "reliable": sum(1 for m in matches if m["status"] == "reliable"),
            "usable": sum(1 for m in matches if m["status"] == "usable"),
            "single_view_preserved": sum(1 for m in matches if m["status"] == "single_view_preserved"),
        },
        "outputs": {
            "obj": str(out / "pipeline_hybrid_qca_tree.obj"),
            "quality_radius_report": str(out / "pipeline_branch_quality_radius.csv"),
            "view_a_overlay": str(out / "view_a_overlay.png"),
            "view_b_overlay": str(out / "view_b_overlay.png"),
        },
    }
    with open(out / "pipeline_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
