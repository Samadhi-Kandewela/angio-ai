"""Validate top-ranked view/frame pairs and choose the best reconstruction."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List
from collections import Counter


ROOT = Path(__file__).resolve().parents[1]


def run_step(name: str, command: List[str], cwd: Path):
    print(f"\n=== {name} ===")
    print(" ".join(command))
    result = subprocess.run(command, cwd=str(cwd), text=True, capture_output=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"{name} failed with exit code {result.returncode}")


def load_json(path: Path) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_pair_rows(path: Path, top_n: int) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows[:top_n]


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def validation_score(pair_dir: Path, pair_row: Dict[str, str]) -> Dict[str, object]:
    validation_summary = load_json(pair_dir / "05_final_validation" / "reprojection_validation_summary.json")
    epipolar_summary = load_json(pair_dir / "03_epipolar" / "epipolar_optimized_summary.json")
    branch_rows = read_csv_rows(pair_dir / "04_smoothed_confidence" / "branch_quality_report.csv")
    branch_counts = Counter(row.get("status", "").lower() for row in branch_rows)
    score = 0.0
    total_good = 0
    total_review = 0
    total_bad = 0
    supported_union: set[int] = set()
    p90_values = []
    median_values = []

    for view in validation_summary.get("views", {}).values():
        good = int(view.get("good_branches", 0))
        review = int(view.get("review_branches", 0))
        bad = int(view.get("bad_branches", 0))
        total_good += good
        total_review += review
        total_bad += bad
        supported_union.update(int(v) for v in view.get("supported_branch_ids", []))
        if "p90_branch_error_px" in view:
            p90_values.append(float(view["p90_branch_error_px"]))
        if "median_branch_error_px" in view:
            median_values.append(float(view["median_branch_error_px"]))

    accepted = int(epipolar_summary.get("num_accepted", 0))
    uncertain = int(epipolar_summary.get("num_uncertain_visualized", 0))
    max_p90 = max(p90_values) if p90_values else 999.0
    max_median = max(median_values) if median_values else 999.0
    initial_score = float(pair_row.get("score", 0.0) or 0.0)
    total_validation = max(1, total_good + total_review + total_bad)
    bad_ratio = total_bad / total_validation
    visible_3d_branches = len(branch_rows)
    usable_3d_branches = int(branch_counts.get("reliable", 0) + branch_counts.get("usable", 0))
    uncertain_3d_branches = int(branch_counts.get("uncertain", 0))
    estimated_3d_branches = int(branch_counts.get("estimated", 0))
    completeness_bonus = min(visible_3d_branches, 14) * 7.0
    anchor_bonus = 45.0 * accepted + 14.0 * uncertain + 16.0 * len(supported_union)
    tiny_tree_penalty = 85.0 if visible_3d_branches < 8 else 0.0
    no_anchor_penalty = 45.0 if accepted == 0 and len(supported_union) < 3 else 0.0
    bad_branch_penalty = 9.0 * total_bad + 45.0 * max(0.0, bad_ratio - 0.45)
    score = (
        5.0 * total_good
        + 2.0 * total_review
        + anchor_bonus
        + completeness_bonus
        + 18.0 * usable_3d_branches
        + 5.0 * uncertain_3d_branches
        + 1.2 * estimated_3d_branches
        + 0.08 * initial_score
        - bad_branch_penalty
        - tiny_tree_penalty
        - no_anchor_penalty
        - 1.2 * max_p90
        - 2.0 * max_median
    )

    return {
        "validation_score": round(score, 4),
        "total_good_branches": total_good,
        "total_review_branches": total_review,
        "total_bad_branches": total_bad,
        "supported_branch_count": len(supported_union),
        "epipolar_accepted": accepted,
        "epipolar_uncertain": uncertain,
        "visible_3d_branches": visible_3d_branches,
        "usable_3d_branches": usable_3d_branches,
        "uncertain_3d_branches": uncertain_3d_branches,
        "estimated_3d_branches": estimated_3d_branches,
        "bad_branch_ratio": round(bad_ratio, 4),
        "tiny_tree_penalty": round(tiny_tree_penalty, 4),
        "no_anchor_penalty": round(no_anchor_penalty, 4),
        "max_p90_branch_error_px": round(max_p90, 4),
        "max_median_branch_error_px": round(max_median, 4),
    }


def copy_tree_contents(source: Path, target: Path):
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(source, target)


def copy_best_display_artifacts(best_pair_dir: Path, target: Path):
    final_dir = best_pair_dir / "04_smoothed_confidence"
    validation_dir = best_pair_dir / "05_final_validation"
    for name in (
        "hybrid_smoothed_confidence_colored.obj",
        "hybrid_smoothed_confidence_colored.mtl",
        "branch_quality_report.csv",
        "smoothed_confidence_summary.json",
        "pipeline_summary.json",
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
        source = final_dir / name
        if source.exists():
            shutil.copyfile(source, target / name)
    for name in (
        "reprojection_validation_report.md",
        "reprojection_validation_report.csv",
        "reprojection_validation_summary.json",
        "view_a_reprojection_validation.png",
        "view_b_reprojection_validation.png",
        "view_a_passing_reprojection_validation.png",
        "view_b_passing_reprojection_validation.png",
        "view_a_supported_reprojection_validation.png",
        "view_b_supported_reprojection_validation.png",
    ):
        source = validation_dir / name
        if source.exists():
            shutil.copyfile(source, target / f"final_{name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dicom-dir", type=Path, required=True)
    parser.add_argument("--selection-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--top-n", type=int, default=5)
    args = parser.parse_args()

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    pair_rows = read_pair_rows(args.selection_dir / "view_pair_rankings.csv", args.top_n)
    if not pair_rows:
        raise RuntimeError("No pair rows found for top-pair validation.")

    report_rows = []
    for idx, row in enumerate(pair_rows):
        pair_name = (
            f"pair_{idx:02d}_clip{row['view_a_clip_index']}_f{row['view_a_frame']}"
            f"__clip{row['view_b_clip_index']}_f{row['view_b_frame']}"
        )
        pair_dir = out / pair_name
        pipeline_dir = pair_dir / "02_pipeline"
        epipolar_dir = pair_dir / "03_epipolar"
        final_dir = pair_dir / "04_smoothed_confidence"
        validation_dir = pair_dir / "05_final_validation"
        pair_dir.mkdir(parents=True, exist_ok=True)

        run_step(
            f"Pair {idx} Two-View Pipeline",
            [
                sys.executable,
                "scripts/dicom_3d_pipeline.py",
                "--dicom-dir",
                str(args.dicom_dir),
                "--output-dir",
                str(pipeline_dir),
                "--model",
                str(args.model),
                "--threshold",
                str(args.threshold),
                "--view-a-index",
                row["view_a_clip_index"],
                "--frame-a",
                row["view_a_frame"],
                "--view-b-index",
                row["view_b_clip_index"],
                "--frame-b",
                row["view_b_frame"],
            ],
            ROOT,
        )
        run_step(
            f"Pair {idx} Smooth Mesh",
            [sys.executable, "scripts/smooth_junction_mesh.py", "--input-dir", str(pipeline_dir)],
            ROOT,
        )
        run_step(
            f"Pair {idx} Epipolar Validation",
            [sys.executable, "scripts/epipolar_optimized_centerline.py", "--input-dir", str(pipeline_dir), "--output-dir", str(epipolar_dir)],
            ROOT,
        )
        run_step(
            f"Pair {idx} Recolor",
            [
                sys.executable,
                "scripts/recolor_smoothed_tree_by_confidence.py",
                "--source-dir",
                str(pipeline_dir),
                "--source-obj",
                str(pipeline_dir / "pipeline_hybrid_qca_tree_smoothed.obj"),
                "--epipolar-report",
                str(epipolar_dir / "epipolar_branch_match_report.csv"),
                "--output-dir",
                str(final_dir),
            ],
            ROOT,
        )
        run_step(
            f"Pair {idx} Final Reprojection",
            [
                sys.executable,
                "scripts/reprojection_validation_report.py",
                "--input-dir",
                str(pipeline_dir),
                "--obj",
                str(final_dir / "hybrid_smoothed_confidence_colored.obj"),
                "--output-dir",
                str(validation_dir),
            ],
            ROOT,
        )

        metrics = validation_score(pair_dir, row)
        report_rows.append(
            {
                "pair_index": idx,
                "pair_dir": str(pair_dir),
                **{k: v for k, v in row.items() if k != "top_branch_matches"},
                **metrics,
            }
        )

    report_rows = sorted(report_rows, key=lambda item: float(item["validation_score"]), reverse=True)
    best = report_rows[0]
    best_pair_dir = Path(best["pair_dir"])
    best_output_dir = out / "best_pair_reconstruction"
    copy_tree_contents(best_pair_dir, best_output_dir)
    copy_best_display_artifacts(best_pair_dir, best_output_dir)
    run_step(
        "Best Pair Clinical Confidence Report",
        [
            sys.executable,
            "scripts/clinical_reconstruction_report.py",
            "--reconstruction-dir",
            str(best_output_dir),
        ],
        ROOT,
    )

    with open(out / "top_pair_validation_report.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(report_rows[0].keys()))
        writer.writeheader()
        writer.writerows(report_rows)
    with open(out / "top_pair_validation_summary.json", "w", encoding="utf-8") as f:
        json.dump({"best_pair": best, "pairs": report_rows}, f, indent=2)

    print(json.dumps({"best_pair": best, "report": str(out / "top_pair_validation_report.csv")}, indent=2))


if __name__ == "__main__":
    main()
