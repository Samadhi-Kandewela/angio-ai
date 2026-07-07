import sys, json, glob
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "clinical_app"))

import patient_store
import analysis_results_store
from dicom_loader import load_series_frames
from frame_pipeline import SegmentationModel, LocalizationModel
from report_engine import analyze_frame_list
from qca import QCAConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # angio-ai/

SEG_PATHS = [
    PROJECT_ROOT / "checkpoints" / "mobileunetv3" / "mobileunetv3_augmented_best.onnx",
    PROJECT_ROOT / "checkpoints" / "mobileunetv3" / "mobileunetv3_augmented_best.pth",
]
LOC_PATHS = [
    PROJECT_ROOT / "checkpoints" / "mask_localization_v2" / "best.onnx",
    PROJECT_ROOT / "checkpoints" / "mask_localization_v2" / "best.pth",
    PROJECT_ROOT / "checkpoints" / "mask_localization_v2" / "latest.pth",
    PROJECT_ROOT / "checkpoints" / "multitask_localization_v2" / "multitask_latest.onnx",
    PROJECT_ROOT / "checkpoints" / "multitask_localization_v2" / "multitask_latest.pth",
    PROJECT_ROOT / "checkpoints" / "multitask_localization_v2" / "multitask_best.onnx",
    PROJECT_ROOT / "checkpoints" / "multitask_localization_v2" / "multitask_best.pth",
]


def _first_existing(paths):
    for p in paths:
        if p.exists():
            return str(p)
    return None


def main():
    case_dir = Path(r"D:\UNI\FYP\Pipeline\angio-ai\patient_data\C31986_2026-07-07")
    case_id = case_dir.name
    analysis_dir = case_dir / "analysis_results"

    seg_path = _first_existing(SEG_PATHS)
    loc_path = _first_existing(LOC_PATHS)
    print("Segmentation model:", seg_path)
    print("Localization model:", loc_path)
    seg_model = SegmentationModel(seg_path)
    loc_model = LocalizationModel(loc_path) if loc_path else None

    cfg = QCAConfig(severe_threshold=70.0)
    threshold = 0.5

    patient_info_path = case_dir / "patient_info.json"
    if patient_info_path.exists():
        patient_info = json.loads(patient_info_path.read_text(encoding="utf-8"))
    else:
        patient_info = {"patient_id": "C31986", "name": "Unknown"}

    view_dirs = sorted(analysis_dir.glob("*/results.json"))
    for rj in view_dirs:
        data = json.loads(rj.read_text(encoding="utf-8"))
        source = data["source"]
        view_label = data["view_label"]
        print(f"\n--- Re-analyzing '{view_label}' ---\nsource: {source}")

        frames = load_series_frames(source)
        print(f"  loaded {len(frames)} frames")

        def _cb(i, n):
            if n and i % 10 == 0:
                print(f"  frame {i}/{n}")

        angle_result = analyze_frame_list(
            frames, view_label, seg_model, loc_model, cfg,
            threshold=threshold, progress_cb=_cb, source_label=source,
        )
        view_dir = analysis_results_store.save_view_results(analysis_dir, angle_result, patient_info, cfg)
        print(f"  saved -> {view_dir}")

    print("\n--- Regenerating final clinical report ---")
    out = analysis_results_store.generate_final_clinical_report(case_dir, analysis_dir, patient_info, cfg)
    print("Final report:", out, out.stat().st_size, "bytes")


if __name__ == "__main__":
    main()
