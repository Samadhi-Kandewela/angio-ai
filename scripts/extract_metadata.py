"""
Extract patient/study/series metadata from the burned-in overlay text on raw
angiogram frames using OCR. There are no original DICOM files for this data
(only extracted JPGs), so the overlay text is the only source for age, sex,
projection angle, exposure technique, etc.

Patient *name* is never parsed out or stored. Instead, the small ID codes also
printed on the overlay (e.g. "C31904 A28857") are OCR'd and one-way hashed into
an opaque patient_id, so re-running this script on the same patient's data
reproduces the same ID without persisting anything identifying. Even so, this
output (age/sex/exact procedure timestamp) is still sensitive in combination --
treat metadata_out/ with the same care as PHI, separately from the de-identified
image dataset produced by select_keyframes.py.

Usage:
  python extract_metadata.py --study-dir "D:/.../Study-125-XA-Cardiac[20260102]" \
      --out D:/.../metadata_out

  python extract_metadata.py --studies-root D:/.../raw_studies --out D:/.../metadata_out
"""
import argparse
import hashlib
import json
import os
import re
import sys
from multiprocessing import Pool, cpu_count

import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from select_keyframes import find_overlay_bands, find_study_dirs, IMG_EXTS  # noqa: E402

_READER = None  # one instance per worker process; loaded lazily via pool initializer


def _init_worker():
    global _READER
    import easyocr
    _READER = easyocr.Reader(["en"], gpu=False, verbose=False)


def ocr_region(img_bgr, upscale=3):
    if img_bgr.size == 0:
        return ""
    # This overlay text is small (~10-13px cap height); upscaling before OCR
    # measurably improved recognition in testing -- it's what recovered the
    # trailing sex letter and the full HH:MM:SS timestamp that were dropped
    # entirely at native resolution.
    if upscale != 1:
        img_bgr = cv2.resize(img_bgr, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC)
    results = _READER.readtext(img_bgr, detail=0, paragraph=False)
    return " | ".join(results)


# Regexes are deliberately loose (OCR on compressed bitmap text drops/garbles
# characters) -- tune these against the raw_top_text/raw_bottom_text saved in
# the output JSON if a different machine's overlay format doesn't match this
# hospital's AlluraXper layout.
AGE_RE = re.compile(r"\b(\d{1,3})\s*Y\b", re.IGNORECASE)
SEX_RE = re.compile(r"\b([MF])\b(?!.*\b[MF]\b)")  # last standalone M/F token
ID_CODE_RE = re.compile(r"\b([A-Z]\d{4,6}|[A-Z]\d[-\d]{3,8})\b")
DATE_RE = re.compile(r"(\d{1,2})[-\s]([A-Za-z]+)[-\s](\d{4})")
TIME_RE = re.compile(r"\b(\d{1,2}):(\d{2})(?::(\d{2}))?\b")
FPS_RE = re.compile(r"(\d+)\s*fps", re.IGNORECASE)
# Tolerates two OCR quirks seen in testing: '0' (zero) misread as the letter
# O/o (e.g. "125.00kV" -> "125.OOkV"), and the decimal point itself being
# dropped with the number split into two separate OCR boxes/tokens instead
# (e.g. "5.20" -> "5 | 20", "118.36kV" -> "118. 36kV").
_NUM = r"[\dOo]+\s?\.?\s?[\dOo]*"
ANGLE_RE = re.compile(
    rf"\b(LAO|RAO)\D{{0,4}}({_NUM}).{{0,15}}\b(CAU|CRA)\D{{0,4}}({_NUM})", re.IGNORECASE
)
EXPOSURE_RE = re.compile(rf"({_NUM})\s*m?A\D{{0,15}}?({_NUM})\s*kV", re.IGNORECASE)
# Curated view-label lookup rather than a general layout parser -- OCR splits
# this line inconsistently (e.g. "Left Coronary 15" / "fps" as separate boxes),
# so matching known clinical view names against the whole line is more robust
# than trying to regex out "whatever precedes the fps number". Ordered
# most-to-least specific and matched in that order: "Cardiac" is a generic
# procedure-type word that appears earlier in the text on every series
# (contrast or not), so a plain unordered search always matches it first and
# never reaches the actual view name later in the string -- only fall back to
# it if nothing more specific hit.
VIEW_LABELS = ["Left Coronary", "Right Coronary", "Fluoroscopy", "LV Gram", "LV", "Aortic", "Bypass", "Cardiac"]
VIEW_RES = [re.compile(re.escape(v), re.IGNORECASE) for v in VIEW_LABELS]


def find_view_label(text):
    for label, pattern in zip(VIEW_LABELS, VIEW_RES):
        if pattern.search(text):
            return label
    return None


def _num(s):
    return float(s.replace(" ", "").replace("O", "0").replace("o", "0"))


def hash_patient_id(id_codes, salt=""):
    if not id_codes:
        return None
    joined = "|".join(sorted(set(id_codes)))
    return hashlib.sha256((salt + joined).encode()).hexdigest()[:16]


def parse_patient_fields(text):
    age = AGE_RE.search(text)
    sex = SEX_RE.search(text)
    ids = ID_CODE_RE.findall(text)
    date_m = DATE_RE.search(text)
    time_m = TIME_RE.search(text)
    return {
        "age_years": int(age.group(1)) if age else None,
        "sex": sex.group(1).upper() if sex else None,
        "study_date": f"{date_m.group(1)}-{date_m.group(2)}-{date_m.group(3)}" if date_m else None,
        "study_time": time_m.group(0) if time_m else None,
    }, ids


def parse_series_fields(text):
    fps = FPS_RE.search(text)
    angle = ANGLE_RE.search(text)
    exposure = EXPOSURE_RE.search(text)
    projection = None
    if angle:
        projection = {angle.group(1).upper(): float(angle.group(2)), angle.group(3).upper(): float(angle.group(4))}
    return {
        "view_label": find_view_label(text),
        "fps": int(fps.group(1)) if fps else None,
        "projection_deg": projection,
        "mA": _num(exposure.group(1)) if exposure else None,
        "kV": _num(exposure.group(2)) if exposure else None,
    }


def process_series_metadata(args):
    study_dir, series_name = args
    series_dir = os.path.join(study_dir, series_name)
    frames = sorted(f for f in os.listdir(series_dir) if f.lower().endswith(IMG_EXTS))
    if not frames:
        return None

    first = cv2.imread(os.path.join(series_dir, frames[0]))
    gray = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)
    top_band, bottom_band = find_overlay_bands(gray)

    top_text = ocr_region(first[top_band[0]:top_band[1], :]) if top_band[1] > top_band[0] else ""
    bottom_text = ocr_region(first[bottom_band[0]:bottom_band[1], :]) if bottom_band[1] > bottom_band[0] else ""

    # regexes match across token-box boundaries (e.g. "15" and "fps" landing in
    # separate OCR boxes), so search on a plain-space join, not the "|"-joined
    # display text.
    search_text = (top_text + " " + bottom_text).replace("|", " ")
    patient_fields, id_codes = parse_patient_fields(search_text)
    series_fields = parse_series_fields(search_text)

    return {
        "study": os.path.basename(study_dir),
        "series": series_name,
        "patient_fields": patient_fields,
        "id_codes": id_codes,
        "series_fields": series_fields,
        "raw_top_text": top_text,
        "raw_bottom_text": bottom_text,
    }


def majority(values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    return max(set(values), key=values.count)


def aggregate_study(series_results, salt):
    all_ids = [c for r in series_results for c in r["id_codes"]]
    patient_id = hash_patient_id(all_ids, salt=salt)

    ages = [r["patient_fields"]["age_years"] for r in series_results]
    sexes = [r["patient_fields"]["sex"] for r in series_results]
    dates = [r["patient_fields"]["study_date"] for r in series_results]
    times = [r["patient_fields"]["study_time"] for r in series_results]

    return {
        "patient_id": patient_id,
        "age_years": majority(ages),
        "sex": majority(sexes),
        "study_date": majority(dates),
        "study_time": majority(times),
        "series": {
            r["series"]: {**r["series_fields"], "raw_top_text": r["raw_top_text"], "raw_bottom_text": r["raw_bottom_text"]}
            for r in series_results
        },
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--study-dir", help="Path to a single Study-* folder")
    ap.add_argument("--studies-root", help="Path containing multiple Study-* folders")
    ap.add_argument("--out", required=True, help="Output directory for per-study metadata JSON")
    ap.add_argument("--id-salt", default="",
                     help="Optional secret salt mixed into the patient-id hash. Keep it constant across runs "
                          "for the same dataset so the same patient always hashes to the same id.")
    ap.add_argument("--workers", type=int, default=max(1, min(4, cpu_count() - 1)),
                     help="EasyOCR is memory-heavy per worker; defaults to a lower cap than select_keyframes.py")
    args = ap.parse_args()

    if not args.study_dir and not args.studies_root:
        ap.error("pass --study-dir or --studies-root")

    study_dirs = find_study_dirs(args.studies_root, args.study_dir)
    os.makedirs(args.out, exist_ok=True)

    tasks = []
    for sd in study_dirs:
        for series_name in sorted(os.listdir(sd)):
            if series_name.startswith("Series-") and os.path.isdir(os.path.join(sd, series_name)):
                tasks.append((sd, series_name))

    print(f"{len(study_dirs)} study folder(s), {len(tasks)} series to OCR, {args.workers} workers")

    by_study = {}
    with Pool(args.workers, initializer=_init_worker) as pool:
        for i, result in enumerate(pool.imap_unordered(process_series_metadata, tasks), 1):
            if result is None:
                continue
            by_study.setdefault(result["study"], []).append(result)
            print(f"[{i}/{len(tasks)}] {result['study']}/{result['series']} OCR'd")

    for study, series_results in by_study.items():
        meta = aggregate_study(series_results, args.id_salt)
        out_path = os.path.join(args.out, f"{study}.json")
        with open(out_path, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"{study}: patient_id={meta['patient_id']} age={meta['age_years']} sex={meta['sex']} -> {out_path}")


if __name__ == "__main__":
    main()
