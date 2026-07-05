"""
Prepare a raw DICOM angiography dataset for 3D reconstruction review.

This script keeps raw files untouched. It scans a raw DICOM folder, detects XA
cine clips, groups them into cases, writes metadata inventories, generates clip
contact sheets, and optionally runs quick segmentation on sampled frames.

Example:
    python scripts/prepare_dicom_dataset.py --raw-dir dicom_files --output-dir dicom_dataset_processed

With quick segmentation:
    python scripts/prepare_dicom_dataset.py --raw-dir dicom_files --output-dir dicom_dataset_processed --run-segmentation
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pydicom


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from dicom_3d_pipeline import DEFAULT_MODEL, extract_graph, segment_frame  # noqa: E402


@dataclass
class ClipRecord:
    case_id: str
    clip_index: int
    raw_path: str
    relative_path: str
    patient_id: str
    study_instance_uid: str
    series_instance_uid: str
    sop_instance_uid: str
    modality: str
    frames: int
    rows: int
    cols: int
    primary_angle_deg: Optional[float]
    secondary_angle_deg: Optional[float]
    source_detector_distance_mm: Optional[float]
    source_patient_distance_mm: Optional[float]
    imager_pixel_spacing_mm: str
    acquisition_time: str
    study_date: str
    series_description: str
    estimated_clip_type: str = "uncertain"
    usable_for_3d: str = "unknown"


def safe_str(value, default: str = "") -> str:
    if value is None:
        return default
    return str(value)


def safe_float(value) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def safe_int(value, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except Exception:
        return default


def spacing_to_text(ds) -> str:
    try:
        spacing = [float(v) for v in ds.ImagerPixelSpacing]
        return f"{spacing[0]:.6g};{spacing[1]:.6g}"
    except Exception:
        return ""


def is_probably_dicom(path: Path) -> bool:
    if not path.is_file():
        return False
    if path.suffix.lower() in {".exe", ".dll", ".inf", ".bin", ".txt", ".xml", ".jpg", ".png"}:
        return False
    return True


def read_header(path: Path):
    try:
        return pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
    except Exception:
        return None


def case_key(ds) -> Tuple[str, str]:
    study_uid = safe_str(getattr(ds, "StudyInstanceUID", ""))
    patient_id = safe_str(getattr(ds, "PatientID", "unknown"))
    if study_uid:
        return patient_id, study_uid
    return patient_id, safe_str(getattr(ds, "StudyDate", "unknown"))


def is_xa_cine(ds) -> bool:
    if safe_str(getattr(ds, "Modality", "")) != "XA":
        return False
    if not hasattr(ds, "NumberOfFrames"):
        return False
    if safe_int(getattr(ds, "NumberOfFrames", 0)) < 2:
        return False
    return True


def scan_clips(raw_dir: Path) -> Tuple[List[ClipRecord], Dict[str, Dict[str, str]]]:
    dicom_headers = []
    for path in raw_dir.rglob("*"):
        if not is_probably_dicom(path):
            continue
        ds = read_header(path)
        if ds is None:
            continue
        if not is_xa_cine(ds):
            continue
        dicom_headers.append((path, ds))

    case_keys = sorted({case_key(ds) for _, ds in dicom_headers}, key=lambda item: (item[0], item[1]))
    case_id_by_key = {key: f"case_{idx + 1:03d}" for idx, key in enumerate(case_keys)}
    clip_counters: Dict[str, int] = {}
    clips: List[ClipRecord] = []
    cases: Dict[str, Dict[str, str]] = {}

    for path, ds in sorted(dicom_headers, key=lambda item: (case_id_by_key[case_key(item[1])], safe_str(getattr(item[1], "AcquisitionTime", "")), str(item[0]))):
        key = case_key(ds)
        case_id = case_id_by_key[key]
        clip_index = clip_counters.get(case_id, 0)
        clip_counters[case_id] = clip_index + 1
        cases.setdefault(
            case_id,
            {
                "case_id": case_id,
                "patient_id": safe_str(getattr(ds, "PatientID", "")),
                "study_instance_uid": safe_str(getattr(ds, "StudyInstanceUID", "")),
                "study_date": safe_str(getattr(ds, "StudyDate", "")),
            },
        )
        clips.append(
            ClipRecord(
                case_id=case_id,
                clip_index=clip_index,
                raw_path=str(path),
                relative_path=str(path.relative_to(raw_dir)),
                patient_id=safe_str(getattr(ds, "PatientID", "")),
                study_instance_uid=safe_str(getattr(ds, "StudyInstanceUID", "")),
                series_instance_uid=safe_str(getattr(ds, "SeriesInstanceUID", "")),
                sop_instance_uid=safe_str(getattr(ds, "SOPInstanceUID", "")),
                modality=safe_str(getattr(ds, "Modality", "")),
                frames=safe_int(getattr(ds, "NumberOfFrames", 0)),
                rows=safe_int(getattr(ds, "Rows", 0)),
                cols=safe_int(getattr(ds, "Columns", 0)),
                primary_angle_deg=safe_float(getattr(ds, "PositionerPrimaryAngle", None)),
                secondary_angle_deg=safe_float(getattr(ds, "PositionerSecondaryAngle", None)),
                source_detector_distance_mm=safe_float(getattr(ds, "DistanceSourceToDetector", None)),
                source_patient_distance_mm=safe_float(getattr(ds, "DistanceSourceToPatient", None)),
                imager_pixel_spacing_mm=spacing_to_text(ds),
                acquisition_time=safe_str(getattr(ds, "AcquisitionTime", "")),
                study_date=safe_str(getattr(ds, "StudyDate", "")),
                series_description=safe_str(getattr(ds, "SeriesDescription", "")),
            )
        )

    return clips, cases


def normalize_frame(frame: np.ndarray) -> np.ndarray:
    if frame.dtype == np.uint8:
        out = frame
    else:
        f = frame.astype(np.float32)
        lo, hi = np.percentile(f, [1, 99])
        out = np.clip((f - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
    return out


def read_clip_pixels(path: Path) -> Optional[np.ndarray]:
    try:
        ds = pydicom.dcmread(str(path), force=True)
        arr = ds.pixel_array
        if arr.ndim == 2:
            arr = arr[None, :, :]
        return arr
    except Exception:
        return None


def sample_frames(frame_count: int, samples: int) -> List[int]:
    if frame_count <= 0:
        return []
    if frame_count <= samples:
        return list(range(1, frame_count + 1))
    return sorted({int(np.clip(round(frame_count * frac), 1, frame_count)) for frac in np.linspace(0.15, 0.90, samples)})


def resize_with_label(gray: np.ndarray, label: str, size: int = 220) -> np.ndarray:
    thumb = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
    bgr = cv2.cvtColor(thumb, cv2.COLOR_GRAY2BGR)
    cv2.putText(bgr, label, (7, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(bgr, label, (7, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 1, cv2.LINE_AA)
    return bgr


def write_contact_sheet(case_clips: List[ClipRecord], case_dir: Path, samples_per_clip: int):
    thumbs = []
    for clip in case_clips:
        arr = read_clip_pixels(Path(clip.raw_path))
        if arr is None:
            continue
        for frame_no in sample_frames(min(clip.frames, len(arr)), samples_per_clip):
            gray = normalize_frame(arr[frame_no - 1])
            label = f"C{clip.clip_index} F{frame_no} P{clip.primary_angle_deg} S{clip.secondary_angle_deg}"
            thumbs.append(resize_with_label(gray, label))
    if not thumbs:
        return
    cols = min(4, len(thumbs))
    rows = int(math.ceil(len(thumbs) / cols))
    blank = np.zeros_like(thumbs[0])
    while len(thumbs) < rows * cols:
        thumbs.append(blank.copy())
    grid = np.vstack([np.hstack(thumbs[row * cols:(row + 1) * cols]) for row in range(rows)])
    cv2.imwrite(str(case_dir / "clip_contact_sheet.png"), grid)


def segmentation_score(clean_mask: np.ndarray, branches: List[Dict[str, object]]) -> Tuple[float, float, float, float]:
    vessel_area_pct = float(np.count_nonzero(clean_mask) / clean_mask.size * 100.0)
    branch_count = float(len(branches))
    centerline_length = float(sum(float(branch.get("length_px", 0.0)) for branch in branches))
    area_score = min(vessel_area_pct / 8.0, 1.0)
    length_score = min(centerline_length / 1300.0, 1.0)
    branch_score = min(branch_count / 10.0, 1.0)
    quality = max(0.0, 100.0 * (0.35 * area_score + 0.45 * length_score + 0.20 * branch_score))
    return vessel_area_pct, branch_count, centerline_length, quality


def run_segmentation_for_case(case_clips: List[ClipRecord], case_dir: Path, model_path: Path, threshold: float, samples_per_clip: int):
    import onnxruntime as ort

    session = ort.InferenceSession(str(model_path))
    input_name = session.get_inputs()[0].name
    frame_dir = case_dir / "sampled_frames"
    frame_dir.mkdir(exist_ok=True)
    rows = []
    for clip in case_clips:
        arr = read_clip_pixels(Path(clip.raw_path))
        if arr is None:
            continue
        for frame_no in sample_frames(min(clip.frames, len(arr)), samples_per_clip):
            gray = cv2.resize(normalize_frame(arr[frame_no - 1]), (512, 512), interpolation=cv2.INTER_AREA)
            mask, overlay = segment_frame(session, input_name, gray, threshold)
            clean, skel, _, branches = extract_graph(mask)
            area_pct, branch_count, centerline_length, quality = segmentation_score(clean, branches)
            stem = f"clip_{clip.clip_index:02d}_frame_{frame_no:03d}"
            cv2.imwrite(str(frame_dir / f"{stem}_original.png"), gray)
            cv2.imwrite(str(frame_dir / f"{stem}_mask.png"), mask)
            cv2.imwrite(str(frame_dir / f"{stem}_overlay.png"), overlay)
            cv2.imwrite(str(frame_dir / f"{stem}_skeleton.png"), skel)
            rows.append(
                {
                    "case_id": clip.case_id,
                    "clip_index": clip.clip_index,
                    "frame": frame_no,
                    "vessel_area_pct": round(area_pct, 4),
                    "branch_count": int(branch_count),
                    "centerline_length_px": round(centerline_length, 2),
                    "segmentation_quality_score": round(quality, 2),
                    "doctor_usefulness": "",
                    "doctor_notes": "",
                }
            )
    if rows:
        write_csv(case_dir / "sampled_frame_scores.csv", rows)


def write_csv(path: Path, rows: List[Dict[str, object]]):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def classify_clip_for_review(clip: ClipRecord, frame_scores: Optional[List[Dict[str, object]]] = None) -> str:
    if clip.primary_angle_deg is None or clip.secondary_angle_deg is None:
        return "uncertain"
    if frame_scores:
        best = max(float(row["segmentation_quality_score"]) for row in frame_scores)
        if best < 15:
            return "not_useful"
    return "uncertain_coronary_needs_doctor_review"


def candidate_pairs(case_clips: List[ClipRecord]) -> List[Dict[str, object]]:
    rows = []
    for i, a in enumerate(case_clips):
        for b in case_clips[i + 1:]:
            if a.primary_angle_deg is None or b.primary_angle_deg is None or a.secondary_angle_deg is None or b.secondary_angle_deg is None:
                continue
            sep = math.hypot(a.primary_angle_deg - b.primary_angle_deg, a.secondary_angle_deg - b.secondary_angle_deg)
            if sep < 15:
                continue
            rows.append(
                {
                    "case_id": a.case_id,
                    "view_a_clip_index": a.clip_index,
                    "view_b_clip_index": b.clip_index,
                    "angle_separation_deg": round(sep, 3),
                    "view_a_primary_angle_deg": a.primary_angle_deg,
                    "view_a_secondary_angle_deg": a.secondary_angle_deg,
                    "view_b_primary_angle_deg": b.primary_angle_deg,
                    "view_b_secondary_angle_deg": b.secondary_angle_deg,
                    "doctor_same_artery_tree": "",
                    "doctor_lca_rca": "",
                    "doctor_use_for_3d": "",
                    "doctor_notes": "",
                }
            )
    return sorted(rows, key=lambda row: row["angle_separation_deg"], reverse=True)


def process_dataset(
    raw_dir: Path,
    output_dir: Path,
    run_segmentation: bool,
    model_path: Path,
    threshold: float,
    contact_samples: int,
    segmentation_samples: int,
    segmentation_case_ids: Optional[set[str]] = None,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    clips, cases = scan_clips(raw_dir)
    if not clips:
        raise RuntimeError(f"No XA cine DICOM clips found under {raw_dir}")

    inventory_rows = [asdict(clip) for clip in clips]
    write_csv(output_dir / "dataset_inventory.csv", inventory_rows)
    write_json(output_dir / "dataset_inventory.json", {"raw_dir": str(raw_dir), "num_cases": len(cases), "num_xa_clips": len(clips), "clips": inventory_rows})
    write_csv(output_dir / "cases_summary.csv", list(cases.values()))
    write_json(output_dir / "cases_summary.json", {"cases": list(cases.values())})

    for case_id in sorted(cases):
        case_dir = output_dir / case_id
        case_dir.mkdir(exist_ok=True)
        case_clips = [clip for clip in clips if clip.case_id == case_id]
        clip_rows = [asdict(clip) for clip in case_clips]
        write_csv(case_dir / "case_clips_summary.csv", clip_rows)
        write_json(case_dir / "case_clips_summary.json", {"case": cases[case_id], "clips": clip_rows})
        write_contact_sheet(case_clips, case_dir, contact_samples)
        pair_rows = candidate_pairs(case_clips)
        write_csv(case_dir / "candidate_view_pairs.csv", pair_rows)
        write_json(case_dir / "candidate_view_pairs.json", {"candidate_view_pairs": pair_rows})
        if run_segmentation and (not segmentation_case_ids or case_id in segmentation_case_ids):
            run_segmentation_for_case(case_clips, case_dir, model_path, threshold, segmentation_samples)

    doctor_rows = []
    for clip in clips:
        doctor_rows.append(
            {
                "case_id": clip.case_id,
                "clip_index": clip.clip_index,
                "frames": clip.frames,
                "primary_angle_deg": clip.primary_angle_deg,
                "secondary_angle_deg": clip.secondary_angle_deg,
                "relative_path": clip.relative_path,
                "doctor_clip_type": "",
                "doctor_usefulness": "",
                "doctor_best_frame": "",
                "doctor_notes": "",
            }
        )
    write_csv(output_dir / "doctor_clip_review_template.csv", doctor_rows)

    return {
        "raw_dir": str(raw_dir),
        "output_dir": str(output_dir),
        "num_cases": len(cases),
        "num_xa_clips": len(clips),
        "run_segmentation": run_segmentation,
        "segmentation_case_ids": sorted(segmentation_case_ids) if segmentation_case_ids else "all" if run_segmentation else [],
        "outputs": {
            "dataset_inventory_csv": str(output_dir / "dataset_inventory.csv"),
            "dataset_inventory_json": str(output_dir / "dataset_inventory.json"),
            "cases_summary_csv": str(output_dir / "cases_summary.csv"),
            "doctor_clip_review_template": str(output_dir / "doctor_clip_review_template.csv"),
        },
    }


def build_doctor_review_folder(processed_dir: Path, review_dir: Path):
    review_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for case_dir in sorted(processed_dir.glob("case_*")):
        if not case_dir.is_dir():
            continue
        case_id = case_dir.name
        target_case_dir = review_dir / case_id
        target_case_dir.mkdir(exist_ok=True)
        for name in ("clip_contact_sheet.png", "case_clips_summary.csv", "candidate_view_pairs.csv"):
            source = case_dir / name
            if source.exists():
                target = target_case_dir / name
                target.write_bytes(source.read_bytes())
        rows.append(
            {
                "case_id": case_id,
                "contact_sheet": str(target_case_dir / "clip_contact_sheet.png"),
                "clip_summary": str(target_case_dir / "case_clips_summary.csv"),
                "candidate_pairs": str(target_case_dir / "candidate_view_pairs.csv"),
                "doctor_lca_clip_indexes": "",
                "doctor_rca_clip_indexes": "",
                "doctor_best_lca_pair": "",
                "doctor_best_rca_pair": "",
                "doctor_notes": "",
            }
        )
    write_csv(review_dir / "doctor_case_review_index.csv", rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "dicom_files")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "dicom_dataset_processed")
    parser.add_argument("--run-segmentation", action="store_true")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--contact-samples", type=int, default=4)
    parser.add_argument("--segmentation-samples", type=int, default=5)
    parser.add_argument("--segmentation-case-id", action="append", help="Run segmentation only for this case id. Repeatable, e.g. --segmentation-case-id case_001")
    parser.add_argument("--doctor-review-dir", type=Path, default=ROOT / "dicom_doctor_review")
    parser.add_argument("--skip-doctor-review-folder", action="store_true")
    args = parser.parse_args()

    summary = process_dataset(
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        run_segmentation=args.run_segmentation,
        model_path=args.model,
        threshold=args.threshold,
        contact_samples=args.contact_samples,
        segmentation_samples=args.segmentation_samples,
        segmentation_case_ids=set(args.segmentation_case_id or []),
    )
    write_json(args.output_dir / "prepare_dicom_dataset_summary.json", summary)
    if not args.skip_doctor_review_folder:
        build_doctor_review_folder(args.output_dir, args.doctor_review_dir)
        summary["outputs"]["doctor_review_dir"] = str(args.doctor_review_dir)
        write_json(args.output_dir / "prepare_dicom_dataset_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
