"""
One-command research pipeline for confidence-colored 3D coronary reconstruction.

Given a DICOM folder, this orchestrates:

1. Automatic view/frame selection.
2. Two-view DICOM 3D pipeline.
3. Smoothed junction-aware mesh generation.
4. Epipolar optimized centerline validation.
5. Confidence recoloring of the smoothed full tree.
6. Final reprojection validation report.

Example:
    python scripts/run_full_3d_reconstruction.py --dicom-dir dicom_sample --output-dir dicom_full_3d_run
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List


ROOT = Path(__file__).resolve().parents[1]


def run_step(name: str, command: List[str], cwd: Path) -> subprocess.CompletedProcess:
    print(f"\n=== {name} ===")
    print(" ".join(command))
    result = subprocess.run(command, cwd=str(cwd), text=True, capture_output=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"{name} failed with exit code {result.returncode}")
    return result


def load_json(path: Path) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def copy_final_artifacts(final_dir: Path, validation_dir: Path, output_dir: Path):
    final_obj = final_dir / "hybrid_smoothed_confidence_colored.obj"
    final_mtl = final_dir / "hybrid_smoothed_confidence_colored.mtl"
    if final_obj.exists():
        shutil.copyfile(final_obj, output_dir / final_obj.name)
    if final_mtl.exists():
        shutil.copyfile(final_mtl, output_dir / final_mtl.name)
    for name in (
        "branch_quality_report.csv",
        "smoothed_confidence_summary.json",
        "pipeline_summary.json",
        "view_a_overlay.png",
        "view_b_overlay.png",
        "view_a_mask.png",
        "view_b_mask.png",
        "view_a_skeleton.png",
        "view_b_skeleton.png",
    ):
        source = final_dir / name
        if source.exists():
            shutil.copyfile(source, output_dir / name)
    for name in (
        "reprojection_validation_report.md",
        "reprojection_validation_report.csv",
        "reprojection_validation_summary.json",
        "view_a_reprojection_validation.png",
        "view_b_reprojection_validation.png",
    ):
        source = validation_dir / name
        if source.exists():
            shutil.copyfile(source, output_dir / f"final_{name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dicom-dir", type=Path, required=True, help="Input DICOM folder containing XA clips.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "dicom_full_3d_run")
    parser.add_argument("--model", type=Path, default=ROOT / "checkpoints" / "mobileunetv3" / "mobileunetv3_augmented_best.onnx")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--top-frames-per-clip", type=int, default=3)
    parser.add_argument("--max-frames-per-clip", type=int, default=5)
    parser.add_argument("--view-a-index", type=int, help="Optional override for selected view A clip.")
    parser.add_argument("--frame-a", type=int, help="Optional override for selected view A frame.")
    parser.add_argument("--view-b-index", type=int, help="Optional override for selected view B clip.")
    parser.add_argument("--frame-b", type=int, help="Optional override for selected view B frame.")
    parser.add_argument("--skip-auto-selection", action="store_true", help="Require manual view/frame overrides and skip selector.")
    args = parser.parse_args()

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    selection_dir = out / "01_auto_selection"
    pipeline_dir = out / "02_pipeline"
    epipolar_dir = out / "03_epipolar"
    final_dir = out / "04_smoothed_confidence"
    validation_dir = out / "05_final_validation"

    started = datetime.now().isoformat(timespec="seconds")
    selected_pair: Dict[str, object]

    if args.skip_auto_selection:
        required = [args.view_a_index, args.frame_a, args.view_b_index, args.frame_b]
        if any(value is None for value in required):
            raise RuntimeError("--skip-auto-selection requires --view-a-index --frame-a --view-b-index --frame-b")
        selected_pair = {
            "view_a_clip_index": args.view_a_index,
            "view_a_frame": args.frame_a,
            "view_b_clip_index": args.view_b_index,
            "view_b_frame": args.frame_b,
            "selection_source": "manual_override",
        }
    else:
        run_step(
            "Auto Select Views/Frames",
            [
                sys.executable,
                "scripts/auto_select_dicom_views.py",
                "--dicom-dir",
                str(args.dicom_dir),
                "--output-dir",
                str(selection_dir),
                "--model",
                str(args.model),
                "--threshold",
                str(args.threshold),
                "--max-frames-per-clip",
                str(args.max_frames_per_clip),
                "--top-frames-per-clip",
                str(args.top_frames_per_clip),
            ],
            ROOT,
        )
        selection_report = load_json(selection_dir / "auto_selection_report.json")
        selected_pair = selection_report["recommended_main_pair"]

    view_a_index = int(selected_pair["view_a_clip_index"])
    frame_a = int(selected_pair["view_a_frame"])
    view_b_index = int(selected_pair["view_b_clip_index"])
    frame_b = int(selected_pair["view_b_frame"])

    run_step(
        "Two-View DICOM 3D Pipeline",
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
            str(view_a_index),
            "--frame-a",
            str(frame_a),
            "--view-b-index",
            str(view_b_index),
            "--frame-b",
            str(frame_b),
        ],
        ROOT,
    )

    run_step(
        "Smooth Junction-Aware Mesh",
        [
            sys.executable,
            "scripts/smooth_junction_mesh.py",
            "--input-dir",
            str(pipeline_dir),
        ],
        ROOT,
    )

    run_step(
        "Epipolar Optimized Centerline Validation",
        [
            sys.executable,
            "scripts/epipolar_optimized_centerline.py",
            "--input-dir",
            str(pipeline_dir),
            "--output-dir",
            str(epipolar_dir),
        ],
        ROOT,
    )

    run_step(
        "Confidence Recolor Smoothed Tree",
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
        "Final Reprojection Validation",
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

    copy_final_artifacts(final_dir, validation_dir, out)

    final_summary = {
        "started": started,
        "finished": datetime.now().isoformat(timespec="seconds"),
        "dicom_dir": str(args.dicom_dir),
        "output_dir": str(out),
        "selected_pair": selected_pair,
        "stage_outputs": {
            "auto_selection": str(selection_dir),
            "pipeline": str(pipeline_dir),
            "epipolar": str(epipolar_dir),
            "smoothed_confidence": str(final_dir),
            "final_validation": str(validation_dir),
        },
        "final_outputs": {
            "obj": str(out / "hybrid_smoothed_confidence_colored.obj"),
            "branch_report": str(out / "branch_quality_report.csv"),
            "summary": str(out / "smoothed_confidence_summary.json"),
            "validation_report": str(out / "final_reprojection_validation_report.md"),
            "view_a_validation_overlay": str(out / "final_view_a_reprojection_validation.png"),
            "view_b_validation_overlay": str(out / "final_view_b_reprojection_validation.png"),
        },
        "viewer_command": f"python src\\artery_3d_vtk_viewer.py --obj {out / 'hybrid_smoothed_confidence_colored.obj'}",
    }
    with open(out / "full_3d_reconstruction_summary.json", "w", encoding="utf-8") as f:
        json.dump(final_summary, f, indent=2)
    print("\n=== Full 3D Reconstruction Complete ===")
    print(json.dumps(final_summary, indent=2))


if __name__ == "__main__":
    main()
