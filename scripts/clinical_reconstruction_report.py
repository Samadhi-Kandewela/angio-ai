"""Create a doctor-facing confidence report for a 3D coronary reconstruction."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional


def load_json(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def first_existing(folder: Path, names: Iterable[str]) -> Optional[Path]:
    for name in names:
        path = folder / name
        if path.exists():
            return path
    return None


def to_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def to_int(value: object, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def angle_separation(summary: Dict[str, object]) -> float:
    view_a = summary.get("view_a", {}) if isinstance(summary.get("view_a"), dict) else {}
    view_b = summary.get("view_b", {}) if isinstance(summary.get("view_b"), dict) else {}
    primary = abs(to_float(view_a.get("primary_angle_deg")) - to_float(view_b.get("primary_angle_deg")))
    secondary = abs(to_float(view_a.get("secondary_angle_deg")) - to_float(view_b.get("secondary_angle_deg")))
    return (primary * primary + secondary * secondary) ** 0.5


def validation_counts(validation_rows: List[Dict[str, str]]) -> Counter:
    return Counter(row.get("validation_status", "").lower() for row in validation_rows)


def branch_counts(branch_rows: List[Dict[str, str]]) -> Counter:
    return Counter(row.get("status", "").lower() for row in branch_rows)


def max_error(validation_summary: Dict[str, object], field: str) -> float:
    views = validation_summary.get("views", {})
    if not isinstance(views, dict):
        return 999.0
    values = []
    for view in views.values():
        if isinstance(view, dict) and field in view:
            values.append(to_float(view.get(field), 999.0))
    return max(values) if values else 999.0


def total_view_count(validation_summary: Dict[str, object], field: str) -> int:
    views = validation_summary.get("views", {})
    if not isinstance(views, dict):
        return 0
    return sum(to_int(view.get(field), 0) for view in views.values() if isinstance(view, dict))


def grade_reconstruction(metrics: Dict[str, object]) -> Dict[str, object]:
    good = to_int(metrics["good_validation_branches"])
    review = to_int(metrics["review_validation_branches"])
    bad = to_int(metrics["bad_validation_branches"])
    estimated = to_int(metrics["estimated_3d_branches"])
    uncertain = to_int(metrics["uncertain_3d_branches"])
    supported = to_int(metrics["supported_branch_count"])
    angle = to_float(metrics["angle_separation_deg"])
    max_median = to_float(metrics["max_median_reprojection_error_px"])
    max_p90 = to_float(metrics["max_p90_reprojection_error_px"])
    total = max(1, good + review + bad)
    bad_ratio = bad / total
    estimated_ratio = estimated / max(1, estimated + uncertain + to_int(metrics["usable_3d_branches"]))

    warnings: List[str] = []
    strengths: List[str] = []

    if angle >= 45:
        strengths.append("The selected views have useful angular separation.")
    else:
        warnings.append("The selected views have limited angular separation, which weakens depth accuracy.")

    if good >= 8:
        strengths.append("Most reprojected branches pass the 2D validation check.")
    elif good < 4:
        warnings.append("Few branches pass reprojection validation.")

    if supported >= 2:
        strengths.append("At least two branches are supported in both views.")
    else:
        warnings.append("Very few branches are supported by both views.")

    if bad > 0:
        warnings.append(f"{bad} projected branch entries are marked bad and should be hidden or ignored.")
    if estimated_ratio > 0.45:
        warnings.append("Many 3D branches are estimated rather than strongly triangulated.")
    if max_p90 > 12:
        warnings.append("High p90 reprojection error suggests unreliable local alignment.")
    elif max_p90 <= 6:
        strengths.append("The high-percentile reprojection error is low.")

    if good >= 10 and bad_ratio <= 0.08 and supported >= 3 and angle >= 45 and max_p90 <= 8 and estimated_ratio <= 0.35:
        grade = "reliable"
        clinical_use = "Good research-quality reconstruction; still requires clinician review before measurement use."
    elif good >= 7 and bad_ratio <= 0.20 and supported >= 2 and angle >= 35 and max_p90 <= 12:
        grade = "usable"
        clinical_use = "Usable for anatomical visualization and review, but not yet clinical measurement-ready."
    elif good >= 3 and bad_ratio <= 0.35:
        grade = "uncertain"
        clinical_use = "Use only as a visual reference; branch identity and depth may be unstable."
    elif good >= 10 and supported >= 3 and bad_ratio <= 0.45:
        grade = "uncertain"
        clinical_use = (
            "Fuller visual tree is available, but too many branches need review; use only as an uncertain "
            "anatomical reference with bad branches hidden."
        )
    else:
        grade = "failed"
        clinical_use = "Do not use this reconstruction for anatomical interpretation."

    return {
        "overall_confidence": grade,
        "clinical_use_statement": clinical_use,
        "strengths": strengths,
        "warnings": warnings,
    }


def markdown_report(report: Dict[str, object]) -> str:
    metrics = report["metrics"]
    grading = report["grading"]
    view_a = report["selected_views"]["view_a"]
    view_b = report["selected_views"]["view_b"]

    lines = [
        "# Clinical 3D Reconstruction Report",
        "",
        f"Overall confidence: **{grading['overall_confidence'].upper()}**",
        "",
        grading["clinical_use_statement"],
        "",
        "## Selected Views",
        "",
        f"- View A: clip {view_a.get('clip_index', 'unknown')}, frame {view_a.get('frame', 'unknown')}, angles {view_a.get('primary_angle_deg', 'unknown')} / {view_a.get('secondary_angle_deg', 'unknown')}",
        f"- View B: clip {view_b.get('clip_index', 'unknown')}, frame {view_b.get('frame', 'unknown')}, angles {view_b.get('primary_angle_deg', 'unknown')} / {view_b.get('secondary_angle_deg', 'unknown')}",
        f"- Estimated angle separation: {metrics['angle_separation_deg']} deg",
        "",
        "## Validation Summary",
        "",
        f"- Good validation branches: {metrics['good_validation_branches']}",
        f"- Review validation branches: {metrics['review_validation_branches']}",
        f"- Bad validation branches: {metrics['bad_validation_branches']}",
        f"- Supported branch count: {metrics['supported_branch_count']}",
        f"- Max median reprojection error: {metrics['max_median_reprojection_error_px']} px",
        f"- Max p90 reprojection error: {metrics['max_p90_reprojection_error_px']} px",
        "",
        "## 3D Branch Confidence",
        "",
        f"- Usable branches: {metrics['usable_3d_branches']}",
        f"- Uncertain branches: {metrics['uncertain_3d_branches']}",
        f"- Estimated branches: {metrics['estimated_3d_branches']}",
        "",
        "## Strengths",
        "",
    ]
    lines.extend(f"- {item}" for item in grading["strengths"])
    if not grading["strengths"]:
        lines.append("- No strong confidence signal was found.")
    lines.extend(["", "## Warnings", ""])
    lines.extend(f"- {item}" for item in grading["warnings"])
    if not grading["warnings"]:
        lines.append("- No major warning was triggered.")
    lines.extend(
        [
            "",
            "## Recommendation",
            "",
            report["recommendation"],
            "",
        ]
    )
    return "\n".join(lines)


def build_report(reconstruction_dir: Path) -> Dict[str, object]:
    pipeline_summary = load_json(reconstruction_dir / "pipeline_summary.json")
    validation_summary = load_json(
        first_existing(
            reconstruction_dir,
            ("final_reprojection_validation_summary.json", "reprojection_validation_summary.json"),
        )
        or Path()
    )
    branch_rows = load_csv(reconstruction_dir / "branch_quality_report.csv")
    validation_rows = load_csv(
        first_existing(
            reconstruction_dir,
            ("final_reprojection_validation_report.csv", "reprojection_validation_report.csv"),
        )
        or Path()
    )

    v_counts = validation_counts(validation_rows)
    b_counts = branch_counts(branch_rows)
    supported_count = len(
        {
            row.get("branch_id")
            for row in validation_rows
            if row.get("validation_status", "").lower() in {"good", "review"}
            and row.get("branch_status", "").lower() in {"reliable", "usable", "uncertain"}
        }
    )

    metrics: Dict[str, object] = {
        "angle_separation_deg": round(angle_separation(pipeline_summary), 2),
        "good_validation_branches": int(v_counts.get("good", 0)),
        "review_validation_branches": int(v_counts.get("review", 0)),
        "bad_validation_branches": int(v_counts.get("bad", 0)),
        "supported_branch_count": supported_count or total_view_count(validation_summary, "supported_branch_ids"),
        "max_median_reprojection_error_px": round(max_error(validation_summary, "median_branch_error_px"), 2),
        "max_p90_reprojection_error_px": round(max_error(validation_summary, "p90_branch_error_px"), 2),
        "usable_3d_branches": int(b_counts.get("usable", 0) + b_counts.get("reliable", 0)),
        "uncertain_3d_branches": int(b_counts.get("uncertain", 0)),
        "estimated_3d_branches": int(b_counts.get("estimated", 0)),
    }
    grading = grade_reconstruction(metrics)
    recommendation = (
        "Show the model with bad branches hidden by default. Use supported/reprojected branches for visual "
        "orientation, and clearly label estimated branches as non-measurement anatomy."
    )
    if grading["overall_confidence"] in {"uncertain", "failed"}:
        recommendation = (
            "Do not present this as a clinical-ready 3D model. Show the 2D validation overlays first and mark "
            "the 3D tree as uncertain until better views or segmentation are available."
        )

    return {
        "reconstruction_dir": str(reconstruction_dir),
        "selected_views": {
            "view_a": pipeline_summary.get("view_a", {}),
            "view_b": pipeline_summary.get("view_b", {}),
        },
        "metrics": metrics,
        "grading": grading,
        "recommendation": recommendation,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reconstruction-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    output_dir = args.output_dir or args.reconstruction_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    report = build_report(args.reconstruction_dir)

    json_path = output_dir / "clinical_reconstruction_report.json"
    md_path = output_dir / "clinical_reconstruction_report.md"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(markdown_report(report))

    print(json.dumps({"json": str(json_path), "markdown": str(md_path), "confidence": report["grading"]["overall_confidence"]}, indent=2))


if __name__ == "__main__":
    main()
