"""
Recolor the visually smoother full-tree OBJ using epipolar confidence labels.

This keeps the geometry from pipeline_hybrid_qca_tree_smoothed.obj, but changes
branch materials according to epipolar_branch_match_report.csv:

- reliable: two-view validated
- usable: two-view supported
- uncertain: possible two-view match, requires anatomy review
- estimated: not accepted by epipolar validation, visual estimate only

Example:
    python scripts/recolor_smoothed_tree_by_confidence.py
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from pathlib import Path
from typing import Dict


ROOT = Path(__file__).resolve().parents[1]

MATERIALS = {
    "reliable": (0.82, 0.82, 0.82),
    "usable": (0.18, 0.42, 1.0),
    "uncertain": (1.0, 0.86, 0.15),
    "estimated": (1.0, 0.58, 0.06),
    "estimated_connector": (0.68, 0.70, 0.72),
    "junction": (0.72, 0.72, 0.72),
}


def load_confidence(report_path: Path) -> Dict[int, str]:
    confidence: Dict[int, str] = {}
    with open(report_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            branch_id = int(row["branch_id"])
            status = row.get("status", "rejected")
            confidence[branch_id] = status if status in {"reliable", "usable", "uncertain"} else "estimated"
    return confidence


def add_missing_source_branches(confidence: Dict[int, str], source_report: Path):
    if not source_report.exists():
        return
    with open(source_report, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            confidence.setdefault(int(row["branch_id"]), "estimated")


def write_mtl(path: Path):
    with open(path, "w", encoding="utf-8") as f:
        for name, color in MATERIALS.items():
            f.write(f"newmtl {name}\n")
            f.write(f"Kd {color[0]} {color[1]} {color[2]}\n")


def recolor_obj(source_obj: Path, output_obj: Path, confidence: Dict[int, str]):
    output_obj.parent.mkdir(parents=True, exist_ok=True)
    write_mtl(output_obj.with_suffix(".mtl"))

    current_material = "estimated"
    with open(source_obj, "r", encoding="utf-8", errors="ignore") as src, open(output_obj, "w", encoding="utf-8") as dst:
        dst.write(f"mtllib {output_obj.with_suffix('.mtl').name}\n")
        dst.write("# Smoothed full-tree geometry recolored by epipolar confidence.\n")
        for raw in src:
            line = raw.rstrip("\n")
            if line.startswith("mtllib ") or line.startswith("#"):
                continue
            if line.startswith("o "):
                name = line[2:].strip()
                if name.startswith("estimated_connector"):
                    current_material = "estimated_connector"
                    dst.write(f"o {name}\n")
                    continue
                branch_match = re.search(r"branch[_-](\d+)", name)
                if branch_match:
                    branch_id = int(branch_match.group(1))
                    current_material = confidence.get(branch_id, "estimated")
                    name = f"branch_{branch_id:02d}_{current_material}"
                    dst.write(f"o {name}\n")
                    continue
                if name.startswith("junction"):
                    current_material = "junction"
                    dst.write(f"o {name}\n")
                    continue
            if line.startswith("usemtl "):
                dst.write(f"usemtl {current_material}\n")
                continue
            dst.write(raw)


def write_branch_report(path: Path, confidence: Dict[int, str], source_report: Path):
    source_rows = {}
    if source_report.exists():
        with open(source_report, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                source_rows[int(row["branch_id"])] = row

    rows = []
    for branch_id in sorted(confidence):
        source = source_rows.get(branch_id, {})
        rows.append(
            {
                "branch_id": branch_id,
                "status": confidence[branch_id],
                "estimated_diameter_stenosis_pct": source.get("estimated_diameter_stenosis_pct", ""),
                "min_radius_mm": source.get("min_radius_mm", ""),
                "ref_radius_mm_p80": source.get("ref_radius_mm_p80", ""),
                "median_residual_mm": source.get("median_residual_mm", ""),
                "p90_residual_mm": source.get("p90_residual_mm", ""),
                "confidence_source": (
                    "epipolar_two_view"
                    if confidence[branch_id] in {"reliable", "usable"}
                    else "candidate_two_view_review"
                    if confidence[branch_id] == "uncertain"
                    else "visual_estimate"
                ),
            }
        )
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def copy_viewer_context(source_dir: Path, output_dir: Path):
    for name in (
        "pipeline_summary.json",
        "view_a_original.png",
        "view_b_original.png",
        "view_a_overlay.png",
        "view_b_overlay.png",
        "view_a_mask.png",
        "view_b_mask.png",
        "view_a_skeleton.png",
        "view_b_skeleton.png",
        "view_a_branches.json",
        "view_b_branches.json",
        "view_a_vessel_graph.json",
        "view_b_vessel_graph.json",
    ):
        source = source_dir / name
        if source.exists():
            shutil.copyfile(source, output_dir / name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-obj", type=Path, default=ROOT / "dicom_pipeline_output_v2" / "pipeline_hybrid_qca_tree_smoothed.obj")
    parser.add_argument("--source-dir", type=Path, default=ROOT / "dicom_pipeline_output_v2")
    parser.add_argument("--epipolar-report", type=Path, default=ROOT / "dicom_epipolar_optimized" / "epipolar_branch_match_report.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "dicom_smoothed_confidence")
    args = parser.parse_args()

    output_obj = args.output_dir / "hybrid_smoothed_confidence_colored.obj"
    confidence = load_confidence(args.epipolar_report)
    add_missing_source_branches(confidence, args.source_dir / "pipeline_branch_quality_radius.csv")
    recolor_obj(args.source_obj, output_obj, confidence)
    write_branch_report(
        args.output_dir / "branch_quality_report.csv",
        confidence,
        args.source_dir / "pipeline_branch_quality_radius.csv",
    )
    copy_viewer_context(args.source_dir, args.output_dir)

    counts = {}
    for status in confidence.values():
        counts[status] = counts.get(status, 0) + 1
    summary = {
        "source_geometry": str(args.source_obj),
        "confidence_report": str(args.epipolar_report),
        "output_obj": str(output_obj),
        "branch_counts": counts,
        "meaning": {
            "reliable": "two-view validated branch, visually using smoothed hybrid geometry",
            "usable": "two-view supported branch, visually using smoothed hybrid geometry",
            "uncertain": "possible two-view branch, shown for review but not a clinical anchor",
            "estimated": "full-tree visual branch, not accepted by epipolar validation",
        },
    }
    with open(args.output_dir / "smoothed_confidence_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
