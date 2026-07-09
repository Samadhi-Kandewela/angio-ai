"""
Case-level DICOM analysis and 3D reconstruction workflow.

This worker is intentionally filesystem-first: each stage writes durable case
outputs so the UI can recover after restart and the 3D Viewer can load a case
without depending on in-memory analysis objects.
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from PySide6.QtCore import QThread, Signal
import numpy as np

APP_DIR = Path(__file__).resolve().parent
SRC_DIR = APP_DIR.parent
for _path in (APP_DIR, SRC_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import analysis_results_store
import patient_store
from dicom_loader import discover_series, load_series_frames
from frame_pipeline import LocalizationModel, SegmentationModel
from qca import QCAConfig
from report_engine import analyze_frame_list


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SEGMENTATION_MODEL_PATHS = [
    PROJECT_ROOT / "checkpoints" / "mobileunetv3" / "mobileunetv3_augmented_best.onnx",
    PROJECT_ROOT / "checkpoints" / "mobileunetv3" / "mobileunetv3_augmented_best.pth",
]
DEFAULT_LOCALIZATION_MODEL_PATHS = [
    PROJECT_ROOT / "checkpoints" / "multitask_localization_v2" / "multitask_latest.onnx",
    PROJECT_ROOT / "checkpoints" / "multitask_localization_v2" / "multitask_latest.pth",
    PROJECT_ROOT / "checkpoints" / "multitask_localization_v2" / "multitask_best.onnx",
    PROJECT_ROOT / "checkpoints" / "multitask_localization_v2" / "multitask_best.pth",
]


def first_existing(paths: list[Path]) -> str:
    for path in paths:
        if path.exists():
            return str(path)
    return ""


def status_path(case_id: str) -> Path:
    return patient_store.get_case_analysis_dir(case_id) / "case_workflow_status.json"


def reconstruction_dir(case_id: str) -> Path:
    return patient_store.get_case_dir(case_id) / "reconstruction_3d"


def reconstruction_obj(case_id: str) -> Path:
    return reconstruction_dir(case_id) / "hybrid_smoothed_confidence_colored.obj"


def lesions_3d_path(case_id: str) -> Path:
    return reconstruction_dir(case_id) / "lesions_3d.json"


def lesion_panel_path(case_id: str) -> Path:
    return reconstruction_dir(case_id) / "lesion_panel.json"


def write_status(case_id: str, state: str, message: str, **extra: Any) -> dict:
    data = {
        "case_id": case_id,
        "state": state,
        "message": message,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        **extra,
    }
    path = status_path(case_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return data


def read_status(case_id: str) -> dict:
    path = status_path(case_id)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def is_3d_ready(case_id: str) -> bool:
    return reconstruction_obj(case_id).exists() and lesions_3d_path(case_id).exists()


def _load_obj_vertices(obj_path: Path) -> list[tuple[float, float, float]]:
    vertices = []
    if not obj_path.exists():
        return vertices
    with obj_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.split()
                if len(parts) >= 4:
                    vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
    return vertices


def _load_obj_branch_vertices(obj_path: Path) -> dict[int, list[tuple[float, float, float]]]:
    branches: dict[int, list[tuple[float, float, float]]] = {}
    current_branch = None
    if not obj_path.exists():
        return branches
    with obj_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("o "):
                name = line.split(maxsplit=1)[1].strip()
                current_branch = None
                if name.startswith("branch_"):
                    try:
                        current_branch = int(name.split("_")[1])
                        branches.setdefault(current_branch, [])
                    except (IndexError, ValueError):
                        current_branch = None
            elif current_branch is not None and line.startswith("v "):
                parts = line.split()
                if len(parts) >= 4:
                    branches[current_branch].append((float(parts[1]), float(parts[2]), float(parts[3])))
    return branches


def _load_branch_labels(recon_dir: Path) -> list[dict]:
    path = recon_dir / "anatomy_branch_labels.json"
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("branches", [])
    except (json.JSONDecodeError, OSError):
        return []


def _selected_reconstruction_sources(recon_dir: Path) -> tuple[set[str], set[int]]:
    summary_path = recon_dir / "full_3d_reconstruction_summary.json"
    selection_path = recon_dir / "auto_selection_report.json"
    selected_indices: set[int] = set()
    if summary_path.exists():
        try:
            selected_pair = json.loads(summary_path.read_text(encoding="utf-8")).get("selected_pair", {})
            for key in ("view_a_clip_index", "view_b_clip_index"):
                if selected_pair.get(key) is not None:
                    selected_indices.add(int(selected_pair[key]))
        except (json.JSONDecodeError, OSError, TypeError, ValueError):
            pass

    selected_paths: set[str] = set()
    if selection_path.exists() and selected_indices:
        try:
            report = json.loads(selection_path.read_text(encoding="utf-8"))
            for clip in report.get("clips", []):
                if int(clip.get("clip_index", -1)) in selected_indices:
                    selected_paths.add(str(Path(clip["dicom_path"]).resolve()).lower())
        except (json.JSONDecodeError, OSError, TypeError, ValueError, KeyError):
            pass
    return selected_paths, selected_indices


def _view_number_from_label(label: str) -> Optional[int]:
    label = str(label).strip().lower()
    if len(label) >= 2 and label[:2].isdigit():
        return int(label[:2])
    if label.startswith("cag"):
        digits = "".join(ch for ch in label[3:] if ch.isdigit())
        if digits:
            return int(digits)
    return None


def _filter_views_to_reconstruction_pair(views: list[dict], recon_dir: Path) -> list[dict]:
    selected_paths, selected_indices = _selected_reconstruction_sources(recon_dir)
    if not selected_paths and not selected_indices:
        return views

    path_matches = []
    for view in views:
        source = str(view.get("source", ""))
        if source and not source.startswith("<"):
            try:
                source_key = str(Path(source).resolve()).lower()
            except OSError:
                source_key = source.lower()
            if source_key in selected_paths:
                path_matches.append(view)

    if path_matches:
        return path_matches

    selected_view_numbers = {idx + 1 for idx in selected_indices}
    label_matches = []
    for view in views:
        number = _view_number_from_label(view.get("view_label", ""))
        if number in selected_view_numbers:
            label_matches.append(view)
    return label_matches


def _lesion_label_tokens(lesion: dict) -> set[str]:
    text = " ".join(str(lesion.get(k, "")) for k in ("artery", "label", "group")).lower()
    tokens = set()
    for token in ("lm", "lad", "lcx", "rca", "om", "diagonal", "d1", "d2", "pl", "pda", "lpl", "lpda"):
        if token in text:
            tokens.add(token)
    return tokens


def _branch_label_tokens(branch: dict) -> set[str]:
    text = " ".join(str(branch.get(k, "")) for k in ("anatomy_label", "display_name", "major_tree")).lower()
    tokens = set()
    for token in ("lm", "lad", "lcx", "rca", "om", "diagonal", "d1", "d2", "pl", "pda", "lpl", "lpda"):
        if token in text:
            tokens.add(token)
    return tokens


def _choose_branch_for_lesion(lesion: dict, branch_labels: list[dict], used_counts: dict[int, int]) -> Optional[int]:
    lesion_tokens = _lesion_label_tokens(lesion)
    if not lesion_tokens:
        return None

    candidates = []
    for branch in branch_labels:
        branch_tokens = _branch_label_tokens(branch)
        score = len(lesion_tokens & branch_tokens)
        if score <= 0:
            continue
        try:
            branch_id = int(branch["branch_id"])
        except (KeyError, TypeError, ValueError):
            continue
        candidates.append((score, used_counts.get(branch_id, 0), branch_id))

    if not candidates:
        return None
    candidates.sort(key=lambda row: (-row[0], row[1], row[2]))
    branch_id = candidates[0][2]
    used_counts[branch_id] = used_counts.get(branch_id, 0) + 1
    return branch_id


def _branch_surface_location(vertices: list[tuple[float, float, float]], ordinal: int,
                             total_on_branch: int) -> dict:
    """Approximate a centerline station + wall radius from a tubular OBJ branch."""
    if not vertices:
        return {
            "position_3d": [0.0, 0.0, 0.0],
            "tangent_3d": [0.0, 0.0, 1.0],
            "vessel_radius_3d": 1.0,
        }

    pts = np.asarray(vertices, dtype=np.float64)
    mean = pts.mean(axis=0)
    centered = pts - mean
    try:
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        axis = vh[0]
    except np.linalg.LinAlgError:
        axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    axis = axis / max(np.linalg.norm(axis), 1e-9)

    proj = centered @ axis
    if total_on_branch <= 1:
        target_proj = float(np.median(proj))
    else:
        t = (ordinal + 1) / (total_on_branch + 1)
        target_proj = float(np.quantile(proj, max(0.05, min(0.95, t))))

    center = mean + axis * target_proj
    distances = np.linalg.norm(centered - np.outer(proj, axis), axis=1)
    radius = float(np.median(distances))
    if not math.isfinite(radius) or radius <= 0:
        radius = 1.0

    return {
        "position_3d": [float(v) for v in center],
        "tangent_3d": [float(v) for v in axis],
        "vessel_radius_3d": radius,
    }


def _severity_color(severity: str) -> str:
    severity_l = str(severity).lower()
    if "occlusion" in severity_l or "severe" in severity_l:
        return "#E5484D"
    if "significant" in severity_l or "moderate" in severity_l:
        return "#F5A623"
    if "mild" in severity_l:
        return "#DBB40C"
    return "#BAED91"


def _is_severe_lesion(lesion: dict) -> bool:
    severity = str(lesion.get("severity", "")).lower()
    ds = float(lesion.get("DS_percent") or 0.0)
    return "severe" in severity or ds >= 70.0 or bool(lesion.get("total_occlusion"))


def _is_3d_display_lesion(lesion: dict) -> bool:
    """Lesions shown in the 3D viewer and side panel."""
    return str(lesion.get("severity", "")).upper() in {"MODERATE", "SIGNIFICANT", "SEVERE"}


def _radius_at_point(vertices: list[tuple[float, float, float]], point: list[float], k: int = 24) -> float:
    """Median distance from `point` to its k nearest mesh vertices on the
    same branch -- an estimate of the tube's local wall radius, since the
    lesion point sits on the centerline and nearby ring vertices sit on the
    wall around it."""
    if not vertices:
        return 1.0
    pts = np.asarray(vertices, dtype=np.float64)
    p = np.asarray(point, dtype=np.float64)
    nearest = np.sort(np.linalg.norm(pts - p, axis=1))[: min(k, len(pts))]
    radius = float(np.median(nearest))
    return radius if math.isfinite(radius) and radius > 0 else 1.0


def _lesions_3d_from_pipeline(recon_dir: Path, branch_vertices: dict[int, list[tuple[float, float, float]]]) -> list[dict] | None:
    """Prefers lesions already attached to their exact mesh branch/point at
    build time (scripts/dicom_3d_pipeline.py::apply_qca_narrowing) over the
    fuzzy post-hoc token-overlap match below -- no matching step needed since
    the QCA lesion detection ran on the same branch the mesh point came from."""
    path = recon_dir / "pipeline_qca_lesions_3d.json"
    if not path.exists():
        return None
    try:
        raw_lesions = json.loads(path.read_text(encoding="utf-8")).get("lesions", [])
    except (json.JSONDecodeError, OSError):
        return None
    if not raw_lesions:
        return None

    lesions_3d = []
    for i, lesion in enumerate(raw_lesions):
        # Same severity tiers as the panel so a lesion whose narrowing is
        # correctly baked into the mesh (via smooth_junction_mesh.py's
        # report-authoritative correction) isn't then shown as "not in 3D"
        # in the panel because of a display-filter mismatch.
        if not _is_3d_display_lesion(lesion):
            continue
        branch_id = int(lesion["branch_id"])
        position = lesion["position_3d"]
        lesions_3d.append({
            "lesion_id": f"L3D_{i + 1}",
            "branch_id": branch_id,
            "source_track_id": lesion.get("source_track_id"),
            "view_label": lesion.get("view_label") or "3D reconstruction (view A)",
            "artery": lesion.get("artery") or "unknown",
            "label": lesion.get("label") or "unknown",
            "group": lesion.get("group") or "unknown",
            "severity": lesion.get("severity", "unknown"),
            "DS_percent": float(lesion.get("DS_percent") or 0.0),
            "MLD_px": lesion.get("MLD_px"),
            "RVD_px": lesion.get("RVD_px"),
            "MLD_mm": lesion.get("MLD_mm"),
            "RVD_mm": lesion.get("RVD_mm"),
            "confidence": lesion.get("confidence"),
            "total_occlusion": bool(lesion.get("total_occlusion")),
            "position_3d": position,
            "tangent_3d": [0.0, 0.0, 1.0],
            "vessel_radius_3d": _radius_at_point(branch_vertices.get(branch_id, []), position),
            "marker_type": "centerline_point",
            "color": _severity_color(str(lesion.get("severity", ""))),
        })
    return lesions_3d or None


def build_lesion_3d_metadata(case_id: str) -> Path:
    """Creates a practical first-pass lesion overlay package for the 3D Viewer."""
    recon_dir = reconstruction_dir(case_id)
    obj_path = reconstruction_obj(case_id)
    branch_vertices = _load_obj_branch_vertices(obj_path)
    branch_labels = _load_branch_labels(recon_dir)

    analysis_dir = patient_store.get_case_analysis_dir(case_id)
    all_views = analysis_results_store.list_view_results(analysis_dir)
    views = _filter_views_to_reconstruction_pair(all_views, recon_dir)

    pipeline_lesions = _lesions_3d_from_pipeline(recon_dir, branch_vertices)
    if pipeline_lesions is not None:
        lesions_3d = pipeline_lesions
        mapping_method = (
            "displayed lesions detected directly on their mesh branch during reconstruction "
            "(scripts/dicom_3d_pipeline.py QCA pass) -- position is the exact centerline point, "
            "no post-hoc branch matching"
        )
    else:
        all_lesions = []
        for view in views:
            for lesion in view.get("lesions", []):
                if _is_3d_display_lesion(lesion):
                    all_lesions.append((view, lesion))

        used_counts: dict[int, int] = {}
        assigned = []
        for view, lesion in all_lesions:
            branch_id = _choose_branch_for_lesion(lesion, branch_labels, used_counts)
            if branch_id is None or branch_id not in branch_vertices:
                continue
            assigned.append((view, lesion, branch_id))

        per_branch_totals: dict[int, int] = {}
        per_branch_seen: dict[int, int] = {}
        for _, _, branch_id in assigned:
            per_branch_totals[branch_id] = per_branch_totals.get(branch_id, 0) + 1

        lesions_3d = []
        for i, (view, lesion, branch_id) in enumerate(assigned):
            ordinal = per_branch_seen.get(branch_id, 0)
            per_branch_seen[branch_id] = ordinal + 1
            surface = _branch_surface_location(branch_vertices[branch_id], ordinal, per_branch_totals[branch_id])
            ds = float(lesion.get("DS_percent") or 0.0)
            lesions_3d.append({
                "lesion_id": f"L3D_{i + 1}",
                "branch_id": branch_id,
                "source_track_id": lesion.get("track_id"),
                "view_label": view.get("view_label", "View"),
                "artery": lesion.get("artery", "unknown"),
                "label": lesion.get("label", "unknown"),
                "group": lesion.get("group", "unknown"),
                "severity": lesion.get("severity", "unknown"),
                "DS_percent": ds,
                "MLD_px": lesion.get("MLD_px"),
                "RVD_px": lesion.get("RVD_px"),
                "MLD_mm": lesion.get("MLD_mm"),
                "RVD_mm": lesion.get("RVD_mm"),
                "confidence": lesion.get("confidence"),
                "total_occlusion": bool(lesion.get("total_occlusion")),
                "position_3d": surface["position_3d"],
                "tangent_3d": surface["tangent_3d"],
                "vessel_radius_3d": surface["vessel_radius_3d"],
                "marker_type": "wall_surface_patch",
                "color": _severity_color(str(lesion.get("severity", ""))),
                "view_report": str(Path(view["_view_dir"]) / "view_report.pdf"),
            })
        mapping_method = (
            "fallback: wall-surface lesion patches matched to a reconstructed branch "
            "by artery-label token overlap (no in-pipeline QCA lesion file was found)"
        )

    package = {
        "case_id": case_id,
        "mesh_obj": str(obj_path),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "mapping_method": mapping_method,
        "display_filter": "MODERATE, SIGNIFICANT, and SEVERE stenosis tiers",
        "views_used": [
            {
                "view_label": v.get("view_label", "View"),
                "n_frames_analyzed": v.get("n_frames_analyzed"),
                "lesion_count": sum(1 for lesion in v.get("lesions", []) if _is_3d_display_lesion(lesion)),
                "view_report": str(Path(v["_view_dir"]) / "view_report.pdf"),
            }
            for v in views
        ],
        "lesions": lesions_3d,
    }
    out_path = lesions_3d_path(case_id)
    recon_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(package, indent=2), encoding="utf-8")
    (recon_dir / "views_used.json").write_text(json.dumps(package["views_used"], indent=2), encoding="utf-8")
    return out_path


def build_lesion_panel(case_id: str) -> Path:
    """Lists every moderate-or-higher stenosis across ALL analyzed views for the
    case (not just the two used for reconstruction), tagging each with
    whether it's actually represented in the 3D mesh (`lesions_3d.json`) so
    the viewer can show the rest as present-but-inactive instead of
    silently omitting them. Must run after build_lesion_3d_metadata."""
    recon_dir = reconstruction_dir(case_id)
    try:
        package = json.loads(lesions_3d_path(case_id).read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        package = {"lesions": []}
    unclaimed_3d = list(package.get("lesions", []))

    analysis_dir = patient_store.get_case_analysis_dir(case_id)
    all_views = analysis_results_store.list_view_results(analysis_dir)
    reconstruction_views = _filter_views_to_reconstruction_pair(all_views, recon_dir)
    reconstruction_view_labels = {v.get("view_label") for v in reconstruction_views}

    def claim(view_label: str, lesion: dict) -> Optional[dict]:
        track_id = lesion.get("track_id")
        for entry in unclaimed_3d:
            if entry.get("source_track_id") is not None and entry.get("view_label") == view_label and entry.get("source_track_id") == track_id:
                unclaimed_3d.remove(entry)
                return entry
        # Fuzzy fallback: only reached when at least one unclaimed entry has
        # no source_track_id at all -- the in-pipeline QCA path
        # (scripts/dicom_3d_pipeline.py::attach_qca_lesions) only gets one
        # when scripts/smooth_junction_mesh.py's report-correction step ran
        # (needs --analysis-dir wired through run_full_3d_reconstruction.py)
        # and found a matching report lesion; otherwise fall back to matching on
        # severity + closest DS% instead.
        if any(entry.get("source_track_id") is None for entry in unclaimed_3d):
            severity = str(lesion.get("severity", "")).upper()
            ds = float(lesion.get("DS_percent") or 0.0)
            best, best_gap = None, 10.0
            for entry in unclaimed_3d:
                if entry.get("source_track_id") is not None:
                    continue
                if str(entry.get("severity", "")).upper() != severity:
                    continue
                gap = abs(float(entry.get("DS_percent") or 0.0) - ds)
                if gap <= best_gap:
                    best, best_gap = entry, gap
            if best is not None:
                unclaimed_3d.remove(best)
                return best
        return None

    def entry_from_2d(view_label: str, lesion: dict, view_dir, matched: Optional[dict]) -> dict:
        return {
            "view_label": view_label,
            "artery": lesion.get("artery", "unknown"),
            "label": lesion.get("label", "unknown"),
            "group": lesion.get("group", "unknown"),
            "severity": lesion.get("severity", "unknown"),
            "DS_percent": float(lesion.get("DS_percent") or 0.0),
            "MLD_px": lesion.get("MLD_px"),
            "RVD_px": lesion.get("RVD_px"),
            "MLD_mm": lesion.get("MLD_mm"),
            "RVD_mm": lesion.get("RVD_mm"),
            "confidence": lesion.get("confidence"),
            "total_occlusion": bool(lesion.get("total_occlusion")),
            "view_report": str(Path(view_dir) / "view_report.pdf") if view_dir else None,
            "in_3d": matched is not None,
            "lesion_3d_id": matched.get("lesion_id") if matched else None,
        }

    entries = []
    for view in all_views:
        view_label = view.get("view_label", "View")
        is_reconstruction_view = view_label in reconstruction_view_labels
        for lesion in view.get("lesions", []):
            if not _is_3d_display_lesion(lesion):
                continue
            matched = claim(view_label, lesion) if is_reconstruction_view else None
            entries.append(entry_from_2d(view_label, lesion, view.get("_view_dir"), matched))

    # Any mesh-drawn lesion nothing in the 2D report matched still needs a
    # panel entry -- never silently drop something that's actually in the mesh.
    for entry in unclaimed_3d:
        entries.append({
            "view_label": entry.get("view_label", "3D reconstruction"),
            "artery": entry.get("artery", "unknown"),
            "label": entry.get("label", "unknown"),
            "group": entry.get("group", "unknown"),
            "severity": entry.get("severity", "unknown"),
            "DS_percent": float(entry.get("DS_percent") or 0.0),
            "MLD_px": entry.get("MLD_px"),
            "RVD_px": entry.get("RVD_px"),
            "MLD_mm": entry.get("MLD_mm"),
            "RVD_mm": entry.get("RVD_mm"),
            "confidence": entry.get("confidence"),
            "total_occlusion": bool(entry.get("total_occlusion")),
            "view_report": None,
            "in_3d": True,
            "lesion_3d_id": entry.get("lesion_id"),
        })

    entries.sort(key=lambda e: (not e["in_3d"], -e["DS_percent"]))
    for i, entry in enumerate(entries, start=1):
        entry["panel_id"] = f"P{i}"

    payload = {
        "case_id": case_id,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "severity_filter": "MODERATE, SIGNIFICANT, and SEVERE tiers",
        "entries": entries,
    }
    out_path = lesion_panel_path(case_id)
    recon_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


class CaseAnalysisWorkflowThread(QThread):
    progress = Signal(str)
    status_changed = Signal(dict)
    ready = Signal(str)
    error = Signal(str)

    def __init__(self, case_id: str, seg_model_path: str = "", loc_model_path: str = "",
                 threshold: float = 0.5, qca_cfg=None, parent=None):
        super().__init__(parent)
        self.case_id = case_id
        self.seg_model_path = seg_model_path or first_existing(DEFAULT_SEGMENTATION_MODEL_PATHS)
        self.loc_model_path = loc_model_path or first_existing(DEFAULT_LOCALIZATION_MODEL_PATHS)
        self.threshold = threshold
        self.qca_cfg = qca_cfg or QCAConfig(severe_threshold=70.0)
        self._process: Optional[subprocess.Popen] = None

    def cancel(self):
        self.requestInterruption()
        if self._process is not None and self._process.poll() is None:
            self._process.terminate()

    def _status(self, state: str, message: str, **extra):
        self.progress.emit(message)
        data = write_status(self.case_id, state, message, **extra)
        self.status_changed.emit(data)

    def run(self):
        try:
            if is_3d_ready(self.case_id):
                self._status("ready_for_3d_view", "3D reconstruction is already ready.")
                self.ready.emit(self.case_id)
                return

            if not self.seg_model_path or not Path(self.seg_model_path).exists():
                raise RuntimeError("No segmentation model found. Load/select a model before automatic case analysis.")

            case_dir = patient_store.get_case_dir(self.case_id)
            dicom_dir = patient_store.get_case_dicom_dir(self.case_id)
            analysis_dir = patient_store.get_case_analysis_dir(self.case_id)
            patient_info = patient_store.load_metadata(self.case_id)

            self._status("discovering_series", "Discovering DICOM series...")
            series_list = discover_series(dicom_dir)
            if not series_list:
                raise RuntimeError("No readable DICOM series found for this case.")

            self._status("loading_models", "Loading AI models...")
            seg_model = SegmentationModel(self.seg_model_path)
            loc_model = None
            if self.loc_model_path and Path(self.loc_model_path).exists():
                loc_model = LocalizationModel(self.loc_model_path)

            saved_views = []
            for i, series in enumerate(series_list, start=1):
                if self.isInterruptionRequested():
                    return
                label = f"{i:02d}_{series.description or series.path.stem}"
                self._status("analyzing_series", f"Analyzing series {i}/{len(series_list)}: {series.path.name}",
                             current=i, total=len(series_list))
                frames = load_series_frames(series.path)

                def _cb(done, total):
                    if done % 5 == 0 or done == total:
                        self.progress.emit(f"Series {i}/{len(series_list)} frame {done}/{total}")

                angle_result = analyze_frame_list(
                    frames, label, seg_model, loc_model, self.qca_cfg, self.threshold, progress_cb=_cb
                )
                angle_result.video_path = str(series.path)
                view_dir = analysis_results_store.save_view_results(
                    analysis_dir, angle_result, patient_info, self.qca_cfg
                )
                saved_views.append(str(view_dir))

            self._status("generating_reports", "Generating final clinical report...", saved_views=saved_views)
            final_report = analysis_results_store.generate_final_clinical_report(
                case_dir, analysis_dir, patient_info, self.qca_cfg
            )

            self._status("reconstructing_3d", "Running 3D reconstruction. This can take several minutes...")
            recon_dir = reconstruction_dir(self.case_id)
            recon_dir.mkdir(parents=True, exist_ok=True)
            cmd = [
                sys.executable,
                "scripts/run_full_3d_reconstruction.py",
                "--dicom-dir",
                str(dicom_dir),
                "--output-dir",
                str(recon_dir),
                "--model",
                str(self.seg_model_path),
                "--threshold",
                str(self.threshold),
                "--analysis-dir",
                str(analysis_dir),
            ]
            self._process = subprocess.Popen(
                cmd,
                cwd=str(PROJECT_ROOT),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            stdout, stderr = self._process.communicate()
            if self.isInterruptionRequested():
                return
            if self._process.returncode != 0:
                raise RuntimeError(stderr or stdout or "3D reconstruction failed.")

            self._status("mapping_lesions_3d", "Mapping saved stenosis lesions onto the 3D reconstruction...")
            lesions_path = build_lesion_3d_metadata(self.case_id)
            panel_path = build_lesion_panel(self.case_id)
            self._status(
                "ready_for_3d_view",
                "3D reconstruction is ready.",
                final_report=str(final_report),
                reconstruction_dir=str(recon_dir),
                lesions_3d=str(lesions_path),
                lesion_panel=str(panel_path),
            )
            self.ready.emit(self.case_id)
        except Exception as exc:
            message = str(exc)
            self._status("failed", message)
            self.error.emit(message)
