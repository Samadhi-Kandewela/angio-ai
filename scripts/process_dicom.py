"""
Extract keyframes + metadata directly from original multi-frame DICOM files
(one .dcm per series, e.g. "Dicom/000005.dcm").

This supersedes the JPG+OCR pipeline (select_keyframes.py + extract_metadata.py)
whenever raw DICOMs are available -- verified against this hospital's export
(Philips AlluraXper): PixelData contains *no* burned-in text at all. The
patient/technical overlay seen in previously-exported JPGs is composited at
export time from a separate embedded Presentation State object, not baked
into the actual image, so there is nothing to black out here. Every metadata
field also comes from structured DICOM tags instead of noisy OCR -- exact
angle/exposure values, no O/0 confusion, no dropped decimal points.

Patient *name* is still never stored, same as extract_metadata.py -- PatientID
(a real DICOM tag here, not an OCR guess) is hashed into an opaque patient_id.
DICOM has no PatientAge tag for this data (BirthDate is blank too), so age is
still parsed out of the PatientName string, which is the one field that -- as
in the JPG path -- has age jammed into it by the hospital's own data entry.

Usage:
  python process_dicom.py --patient-dir ".../C31904 , MR. L W K G CRISTY , 02.01.2025" \
      --out D:/.../dicom_out

  python process_dicom.py --patients-root ".../2026-2-1" --out D:/.../dicom_out
"""
import argparse
import hashlib
import json
import os
import re
import sys
from multiprocessing import Pool, cpu_count

import cv2
import numpy as np
import pydicom

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from select_keyframes import vesselness_score, smooth, select_from_curve  # noqa: E402

AGE_RE = re.compile(r"\b(\d{1,3})\s*Y\b", re.IGNORECASE)


def hash_patient_id(patient_id_tag, salt=""):
    if not patient_id_tag:
        return None
    return hashlib.sha256((salt + str(patient_id_tag)).encode()).hexdigest()[:16]


def projection_from_positioner(primary, secondary):
    """DICOM C.8.7.5: PositionerPrimaryAngle >0 = LAO, <0 = RAO;
    PositionerSecondaryAngle >0 = CRA (cranial), <0 = CAU (caudal).
    Cross-checked against this hospital's own on-screen LAO/CAU readout for
    the same series (32.2 / -35 -> "LAO 32.20 CAU 35.00"): sign convention
    matches."""
    if primary is None or secondary is None:
        return None
    primary, secondary = float(primary), float(secondary)
    proj = {"LAO" if primary >= 0 else "RAO": round(abs(primary), 2)}
    proj["CRA" if secondary >= 0 else "CAU"] = round(abs(secondary), 2)
    return proj


def save_contact_sheet(frames, selected, out_path, thumb_w=160, cols=8):
    h, w = frames[0].shape[:2]
    thumb_h = int(thumb_w * h / w)
    rows = (len(frames) + cols - 1) // cols
    canvas = np.zeros((rows * (thumb_h + 18), cols * thumb_w, 3), dtype=np.uint8)

    for i, frame in enumerate(frames):
        thumb = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR), (thumb_w, thumb_h))
        r, c = divmod(i, cols)
        y0, x0 = r * (thumb_h + 18), c * thumb_w
        if i in selected:
            cv2.rectangle(thumb, (0, 0), (thumb_w - 1, thumb_h - 1), (0, 0, 255), 3)
        canvas[y0:y0 + thumb_h, x0:x0 + thumb_w] = thumb
        cv2.putText(canvas, str(i), (x0 + 4, y0 + thumb_h + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imwrite(out_path, canvas)


def process_series_dicom(args):
    dcm_path, top_n, plateau_ratio, min_relative_rise, min_peak_area = args
    ds = pydicom.dcmread(dcm_path)
    arr = ds.pixel_array
    if arr.ndim == 2:  # single-frame series -- rare, but handle it
        arr = arr[None, ...]
    frames = [arr[i] for i in range(arr.shape[0])]

    scores = [vesselness_score(f, scale=1.0) for f in frames]
    has_contrast, selected = select_from_curve(scores, top_n, plateau_ratio, min_relative_rise, min_peak_area)

    age_m = AGE_RE.search(str(getattr(ds, "PatientName", "")))
    primary = getattr(ds, "PositionerPrimaryAngle", None)
    secondary = getattr(ds, "PositionerSecondaryAngle", None)

    return {
        "dcm_path": dcm_path,
        "series_number": str(getattr(ds, "SeriesNumber", "")),
        "n_frames": len(frames),
        "frames": frames,
        "scores": scores,
        "has_contrast": has_contrast,
        "selected": selected,
        "patient_fields": {
            "patient_id_hash_input": getattr(ds, "PatientID", None),
            "age_years": int(age_m.group(1)) if age_m else None,
            "sex": getattr(ds, "PatientSex", None) or None,
            "study_date": getattr(ds, "StudyDate", None),
            "study_time": getattr(ds, "StudyTime", None),
        },
        "series_fields": {
            "view_label": getattr(ds, "SeriesDescription", None),
            "fps": int(ds.CineRate) if getattr(ds, "CineRate", None) else None,
            "projection_deg": projection_from_positioner(primary, secondary),
            "mA": float(ds.XRayTubeCurrent) if getattr(ds, "XRayTubeCurrent", None) else None,
            "kV": float(ds.KVP) if getattr(ds, "KVP", None) else None,
        },
    }


def majority(values):
    values = [v for v in values if v is not None]
    return max(set(values), key=values.count) if values else None


def process_patient(args):
    patient_dir, out_root, top_n, plateau_ratio, min_relative_rise, min_peak_area, salt, plot = args
    dicom_dir = os.path.join(patient_dir, "Dicom")
    if not os.path.isdir(dicom_dir):
        return None
    dcm_files = sorted(f for f in os.listdir(dicom_dir) if f.lower().endswith(".dcm"))

    results = []
    for fname in dcm_files:
        results.append(process_series_dicom(
            (os.path.join(dicom_dir, fname), top_n, plateau_ratio, min_relative_rise, min_peak_area)
        ))

    patient_id = hash_patient_id(results[0]["patient_fields"]["patient_id_hash_input"] if results else None, salt)
    patient_label = os.path.basename(patient_dir)  # local folder name only, not written to output

    out_dir = os.path.join(out_root, patient_id or "unknown")
    os.makedirs(out_dir, exist_ok=True)

    series_meta = {}
    manifest_rows = []
    for r in results:
        series_key = f"Series-{r['series_number']}"
        series_out = os.path.join(out_dir, series_key)

        for idx in r["selected"]:
            os.makedirs(series_out, exist_ok=True)
            cv2.imwrite(os.path.join(series_out, f"frame-{idx:04d}.png"), r["frames"][idx])
        if r["has_contrast"] and plot:
            os.makedirs(series_out, exist_ok=True)
            save_contact_sheet(r["frames"], r["selected"], os.path.join(series_out, "_contact_sheet.png"))

        series_meta[series_key] = {**r["series_fields"], "has_contrast": r["has_contrast"],
                                    "n_frames": r["n_frames"], "n_selected": len(r["selected"])}
        for idx in range(r["n_frames"]):
            manifest_rows.append({
                "patient_id": patient_id, "series": series_key, "frame_idx": idx,
                "score": round(r["scores"][idx], 4), "has_contrast_series": r["has_contrast"],
                "selected": idx in r["selected"],
            })

    meta = {
        "patient_id": patient_id,
        "age_years": majority([r["patient_fields"]["age_years"] for r in results]),
        "sex": majority([r["patient_fields"]["sex"] for r in results]),
        "study_date": majority([r["patient_fields"]["study_date"] for r in results]),
        "study_time": majority([r["patient_fields"]["study_time"] for r in results]),
        "series": series_meta,
    }
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    return {"patient_label": patient_label, "patient_id": patient_id, "meta": meta, "manifest_rows": manifest_rows}


def find_patient_dirs(patients_root=None, patient_dir=None):
    if patient_dir:
        return [patient_dir]
    return [os.path.join(patients_root, d) for d in sorted(os.listdir(patients_root))
            if os.path.isdir(os.path.join(patients_root, d)) and os.path.isdir(os.path.join(patients_root, d, "Dicom"))]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--patient-dir", help="Path to a single patient folder (containing a Dicom/ subfolder)")
    ap.add_argument("--patients-root", help="Path containing multiple patient folders")
    ap.add_argument("--out", required=True)
    ap.add_argument("--top-n", type=int, default=3)
    ap.add_argument("--plateau-ratio", type=float, default=0.6)
    ap.add_argument("--min-relative-rise", type=float, default=0.8)
    ap.add_argument("--min-peak-area", type=float, default=4.0)
    ap.add_argument("--id-salt", default="", help="Optional secret salt for the patient-id hash; keep constant across runs")
    ap.add_argument("--plot", action="store_true", help="Save a contact-sheet PNG per contrast series (QA)")
    ap.add_argument("--workers", type=int, default=max(1, cpu_count() - 1))
    args = ap.parse_args()

    if not args.patient_dir and not args.patients_root:
        ap.error("pass --patient-dir or --patients-root")

    patient_dirs = find_patient_dirs(args.patients_root, args.patient_dir)
    os.makedirs(args.out, exist_ok=True)
    print(f"{len(patient_dirs)} patient folder(s) found")

    tasks = [(pd, args.out, args.top_n, args.plateau_ratio, args.min_relative_rise,
              args.min_peak_area, args.id_salt, args.plot) for pd in patient_dirs]

    all_manifest_rows = []
    with Pool(args.workers) as pool:
        for i, result in enumerate(pool.imap_unordered(process_patient, tasks), 1):
            if result is None:
                continue
            n_contrast = sum(1 for v in result["meta"]["series"].values() if v["has_contrast"])
            n_series = len(result["meta"]["series"])
            print(f"[{i}/{len(tasks)}] {result['patient_label']} -> patient_id={result['patient_id']} "
                  f"age={result['meta']['age_years']} sex={result['meta']['sex']} "
                  f"({n_contrast}/{n_series} series with contrast)")
            all_manifest_rows.extend(result["manifest_rows"])

    if all_manifest_rows:
        import csv
        manifest_path = os.path.join(args.out, "manifest.csv")
        with open(manifest_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_manifest_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_manifest_rows)
        print(f"\nDone. Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
