"""
Filesystem organization for patient cases.

Each case gets its own folder containing:
    metadata.json         -- patient demographics + clinical history
    dicom/                -- copies of the original DICOM source(s), preserving
                             each source's structure (a study export is often a
                             whole folder -- a DICOMDIR index, a Dicom/ subfolder
                             of numbered .dcm files, etc. -- not a single file)
    analysis_results/     -- empty; later pipeline stages (segmentation,
                             QCA, PDF reports) write their outputs here

No Qt dependency -- reusable from any UI layer, a CLI, or a batch tool.
"""
from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional

# angio-ai/src/clinical_app/patient_store.py -> parents[2] == angio-ai/
DEFAULT_PATIENT_DATA_ROOT = Path(__file__).resolve().parents[2] / "patient_data"


@dataclass
class PatientCase:
    case_id: str
    case_dir: Path
    metadata_path: Path
    dicom_dir: Path
    analysis_dir: Path
    dicom_files: List[Path]


def _sanitize(value: str) -> str:
    value = value.strip().replace(" ", "_")
    value = re.sub(r"[^A-Za-z0-9_.-]", "", value)
    return value or "unknown"


def _unique_case_dir(root: Path, base_name: str) -> Path:
    candidate = root / base_name
    if not candidate.exists():
        return candidate
    i = 1
    while (root / f"{base_name}_{i}").exists():
        i += 1
    return root / f"{base_name}_{i}"


def _copy_recursive(src: Path, dst: Path, progress_cb: Optional[Callable[[str], None]] = None) -> List[Path]:
    """
    Copies a file, or an entire folder tree, from src to dst. Returns the list
    of files actually copied (not directories). Used instead of shutil.copytree
    so callers can get live per-file progress on large studies.
    """
    copied = []
    if src.is_dir():
        dst.mkdir(parents=True, exist_ok=True)
        for child in sorted(src.iterdir()):
            copied.extend(_copy_recursive(child, dst / child.name, progress_cb))
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied.append(dst)
        if progress_cb:
            progress_cb(src.name)
    return copied


def create_patient_case(metadata: dict, dicom_paths: List[str],
                        root_dir: Optional[Path] = None,
                        progress_cb: Optional[Callable[[str], None]] = None) -> PatientCase:
    """
    Creates <root>/<case_id>/ with metadata.json, a dicom/ subfolder holding
    copies of the supplied DICOM source(s) -- each entry in `dicom_paths` may
    be a single file or a whole folder (e.g. a cath-lab CD export containing a
    DICOMDIR index and a Dicom/ subfolder of frames), copied recursively with
    its structure intact -- and an empty analysis_results/ subfolder.

    `progress_cb`, if given, is called with each copied file's name as it's
    written, so a caller can show live progress on large studies.

    Raises OSError if the filesystem operations fail.
    """
    root = Path(root_dir) if root_dir else DEFAULT_PATIENT_DATA_ROOT
    root.mkdir(parents=True, exist_ok=True)

    patient_id = _sanitize(str(metadata.get("patient_id", "")))
    study_date = _sanitize(str(metadata.get("study_date", "")))
    if patient_id != "unknown":
        base_name = f"{patient_id}_{study_date}" if study_date != "unknown" else patient_id
    else:
        base_name = f"case_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    case_dir = _unique_case_dir(root, base_name)
    dicom_dir = case_dir / "dicom"
    analysis_dir = case_dir / "analysis_results"
    case_dir.mkdir(parents=True)
    dicom_dir.mkdir()
    analysis_dir.mkdir()

    copied = []
    sources_summary = []
    for src in dicom_paths:
        src_path = Path(src)
        if not src_path.exists():
            continue
        dst_path = dicom_dir / src_path.name
        newly_copied = _copy_recursive(src_path, dst_path, progress_cb)
        copied.extend(newly_copied)
        sources_summary.append({
            "name": src_path.name,
            "type": "folder" if src_path.is_dir() else "file",
            "file_count": len(newly_copied),
        })

    full_metadata = dict(metadata)
    full_metadata["case_id"] = case_dir.name
    full_metadata["created_at"] = datetime.now().isoformat(timespec="seconds")
    full_metadata["dicom_sources"] = sources_summary
    full_metadata["dicom_file_count"] = len(copied)

    metadata_path = case_dir / "metadata.json"
    metadata_path.write_text(json.dumps(full_metadata, indent=2), encoding="utf-8")

    _append_case_index(root, full_metadata)

    return PatientCase(
        case_id=case_dir.name, case_dir=case_dir, metadata_path=metadata_path,
        dicom_dir=dicom_dir, analysis_dir=analysis_dir, dicom_files=copied,
    )


def _append_case_index(root: Path, metadata: dict) -> None:
    """Maintains a flat case_index.json at the storage root for a future patient-list page."""
    index_path = root / "case_index.json"
    entries = []
    if index_path.exists():
        try:
            entries = json.loads(index_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            entries = []

    entries.append({
        "case_id": metadata["case_id"],
        "patient_id": metadata.get("patient_id", ""),
        "full_name": metadata.get("full_name", ""),
        "study_date": metadata.get("study_date", ""),
        "created_at": metadata["created_at"],
    })
    index_path.write_text(json.dumps(entries, indent=2), encoding="utf-8")


def list_cases(root_dir: Optional[Path] = None) -> list:
    """Returns the flat case index (most recent last) for a future patient-list page."""
    root = Path(root_dir) if root_dir else DEFAULT_PATIENT_DATA_ROOT
    index_path = root / "case_index.json"
    if not index_path.exists():
        return []
    try:
        return json.loads(index_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []


def get_case_dir(case_id: str, root_dir: Optional[Path] = None) -> Path:
    root = Path(root_dir) if root_dir else DEFAULT_PATIENT_DATA_ROOT
    return root / case_id


def get_case_dicom_dir(case_id: str, root_dir: Optional[Path] = None) -> Path:
    return get_case_dir(case_id, root_dir) / "dicom"


def get_case_analysis_dir(case_id: str, root_dir: Optional[Path] = None) -> Path:
    return get_case_dir(case_id, root_dir) / "analysis_results"


def load_metadata(case_id: str, root_dir: Optional[Path] = None) -> dict:
    """Loads a case's saved metadata.json (patient/study info), e.g. for report title pages."""
    metadata_path = get_case_dir(case_id, root_dir) / "metadata.json"
    if not metadata_path.exists():
        return {}
    try:
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
