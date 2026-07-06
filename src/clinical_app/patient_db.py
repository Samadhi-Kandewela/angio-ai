"""
SQLite-backed patient case index.

Each patient case's real record is still its own metadata.json inside
patient_data/<case_id>/ (see patient_store.py) -- that file alone is enough
to know everything about a case, and stays portable/human-readable on its
own. This module is only a fast, queryable INDEX over those files, replacing
the old flat case_index.json with a real (if small) database so the app can
search/filter/sort cases instead of scanning every folder.

Because it's just an index, it's fully disposable: rebuild_from_disk() wipes
it and re-derives every row from the metadata.json files on disk, so a
missing or corrupted patients.db never loses patient data.

Uses Python's built-in sqlite3 -- no new dependency.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import List, Optional

DEFAULT_PATIENT_DATA_ROOT = Path(__file__).resolve().parents[2] / "patient_data"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS cases (
    case_id             TEXT PRIMARY KEY,
    patient_id          TEXT,
    full_name           TEXT,
    age                 INTEGER,
    gender              TEXT,
    study_date          TEXT,
    operator            TEXT,
    referring_physician TEXT,
    indication          TEXT,
    risk_factors        TEXT,
    notes               TEXT,
    dicom_file_count    INTEGER,
    created_at          TEXT
);
CREATE INDEX IF NOT EXISTS idx_cases_patient_id ON cases(patient_id);
CREATE INDEX IF NOT EXISTS idx_cases_full_name ON cases(full_name);
CREATE INDEX IF NOT EXISTS idx_cases_created_at ON cases(created_at);
"""


def _db_path(root_dir: Optional[Path] = None) -> Path:
    root = Path(root_dir) if root_dir else DEFAULT_PATIENT_DATA_ROOT
    return root / "patients.db"


def _connect(root_dir: Optional[Path] = None) -> sqlite3.Connection:
    path = _db_path(root_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    return conn


def init_db(root_dir: Optional[Path] = None) -> None:
    """Creates patients.db and the cases table if they don't already exist. Safe to call every startup."""
    with _connect(root_dir) as conn:
        conn.executescript(_SCHEMA)


def _row_from_metadata(metadata: dict) -> dict:
    risk_factors = metadata.get("risk_factors") or []
    if isinstance(risk_factors, list):
        risk_factors = ", ".join(risk_factors)
    return {
        "case_id": metadata.get("case_id"),
        "patient_id": metadata.get("patient_id", ""),
        "full_name": metadata.get("full_name", ""),
        "age": metadata.get("age"),
        "gender": metadata.get("gender", ""),
        "study_date": metadata.get("study_date", ""),
        "operator": metadata.get("operator", ""),
        "referring_physician": metadata.get("referring_physician", ""),
        "indication": metadata.get("indication", ""),
        "risk_factors": risk_factors,
        "notes": metadata.get("notes", ""),
        "dicom_file_count": metadata.get("dicom_file_count"),
        "created_at": metadata.get("created_at", ""),
    }


def upsert_case(metadata: dict, root_dir: Optional[Path] = None) -> None:
    """Inserts or updates a case's index row from its metadata dict (as saved to metadata.json)."""
    row = _row_from_metadata(metadata)
    if not row["case_id"]:
        raise ValueError("metadata must include case_id")

    with _connect(root_dir) as conn:
        conn.executescript(_SCHEMA)  # no-op if already applied; guards against a caller skipping init_db()
        conn.execute(
            """
            INSERT INTO cases (case_id, patient_id, full_name, age, gender, study_date, operator,
                               referring_physician, indication, risk_factors, notes,
                               dicom_file_count, created_at)
            VALUES (:case_id, :patient_id, :full_name, :age, :gender, :study_date, :operator,
                   :referring_physician, :indication, :risk_factors, :notes,
                   :dicom_file_count, :created_at)
            ON CONFLICT(case_id) DO UPDATE SET
                patient_id=excluded.patient_id, full_name=excluded.full_name, age=excluded.age,
                gender=excluded.gender, study_date=excluded.study_date, operator=excluded.operator,
                referring_physician=excluded.referring_physician, indication=excluded.indication,
                risk_factors=excluded.risk_factors, notes=excluded.notes,
                dicom_file_count=excluded.dicom_file_count, created_at=excluded.created_at
            """,
            row,
        )


def list_cases(root_dir: Optional[Path] = None, search: Optional[str] = None) -> List[dict]:
    """
    Returns cases as plain dicts, most recently created first. If `search` is
    given, filters to cases whose patient_id, full_name, or indication contain
    it (case-insensitive) -- for a future patient search box.
    """
    with _connect(root_dir) as conn:
        conn.executescript(_SCHEMA)
        if search:
            like = f"%{search}%"
            cur = conn.execute(
                """
                SELECT * FROM cases
                WHERE patient_id LIKE ? COLLATE NOCASE
                   OR full_name LIKE ? COLLATE NOCASE
                   OR indication LIKE ? COLLATE NOCASE
                ORDER BY created_at DESC
                """,
                (like, like, like),
            )
        else:
            cur = conn.execute("SELECT * FROM cases ORDER BY created_at DESC")
        return [dict(r) for r in cur.fetchall()]


def delete_case(case_id: str, root_dir: Optional[Path] = None) -> None:
    """Removes a case's index row (e.g. if its folder was deleted). Does not touch any files."""
    with _connect(root_dir) as conn:
        conn.execute("DELETE FROM cases WHERE case_id = ?", (case_id,))


def rebuild_from_disk(root_dir: Optional[Path] = None) -> int:
    """
    Wipes and re-derives the entire index from every patient_data/<case_id>/metadata.json
    on disk. Self-healing: run at app startup so a missing/corrupted/out-of-date
    patients.db (or cases created before this feature existed) never loses data --
    the folders on disk are always the real record. Returns the number of cases indexed.
    """
    root = Path(root_dir) if root_dir else DEFAULT_PATIENT_DATA_ROOT
    init_db(root)

    count = 0
    with _connect(root) as conn:
        conn.execute("DELETE FROM cases")
        if root.exists():
            for metadata_path in sorted(root.glob("*/metadata.json")):
                try:
                    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError):
                    continue
                metadata.setdefault("case_id", metadata_path.parent.name)
                row = _row_from_metadata(metadata)
                conn.execute(
                    """
                    INSERT INTO cases (case_id, patient_id, full_name, age, gender, study_date, operator,
                                       referring_physician, indication, risk_factors, notes,
                                       dicom_file_count, created_at)
                    VALUES (:case_id, :patient_id, :full_name, :age, :gender, :study_date, :operator,
                           :referring_physician, :indication, :risk_factors, :notes,
                           :dicom_file_count, :created_at)
                    """,
                    row,
                )
                count += 1
    return count
