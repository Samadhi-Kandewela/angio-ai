"""
Two-view coronary 3D reconstruction using manual COCO segmentation masks.

This is an experiment script: it does not change the existing model-based
pipeline. It reads DICOM geometry/frames from the selected case, but replaces
the ONNX segmentation output with masks decoded from a COCO annotations JSON.

Example:
    python scripts/dicom_annotation_3d_pipeline.py ^
        --dicom-dir dicom_files/case_003 ^
        --annotations instances_default.json ^
        --image-dir new_raw_images ^
        --output-dir dicom_case003_annotation_reconstruction ^
        --view-a-index 0 --frame-a 23 --view-a-image frame_023.png ^
        --view-b-index 3 --frame-b 24 --view-b-image frame_024.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pydicom

from dicom_3d_pipeline import (
    build_mesh,
    extract_graph,
    find_branch_matches,
    find_xa_clips,
    read_frame,
    view_from_clip,
)


def decode_uncompressed_rle(segmentation: dict) -> np.ndarray:
    height, width = [int(v) for v in segmentation["size"]]
    counts = segmentation["counts"]
    if not isinstance(counts, list):
        raise ValueError("Only uncompressed COCO RLE counts are supported without pycocotools.")
    flat = np.zeros(height * width, dtype=np.uint8)
    index = 0
    value = 0
    for count in counts:
        count = int(count)
        if value == 1 and count > 0:
            flat[index:index + count] = 1
        index += count
        value = 1 - value
    return flat.reshape((height, width), order="F") * 255


def decode_polygon(segmentation, height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    polygons = segmentation if isinstance(segmentation, list) else []
    for polygon in polygons:
        pts = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
        if len(pts) >= 3:
            cv2.fillPoly(mask, [np.rint(pts).astype(np.int32)], 255)
    return mask


def load_coco_masks(annotation_path: Path) -> dict[str, np.ndarray]:
    data = json.loads(annotation_path.read_text(encoding="utf-8"))
    images = {int(item["id"]): item for item in data.get("images", [])}
    masks = {}
    for image_id, image in images.items():
        height = int(image["height"])
        width = int(image["width"])
        masks[image["file_name"]] = np.zeros((height, width), dtype=np.uint8)

    for annotation in data.get("annotations", []):
        image = images.get(int(annotation["image_id"]))
        if image is None:
            continue
        file_name = image["file_name"]
        height = int(image["height"])
        width = int(image["width"])
        segmentation = annotation.get("segmentation")
        if isinstance(segmentation, dict):
            mask = decode_uncompressed_rle(segmentation)
        else:
            mask = decode_polygon(segmentation, height, width)
        masks[file_name] = cv2.bitwise_or(masks[file_name], mask.astype(np.uint8))
    return masks


def load_annotation_mask(masks: dict[str, np.ndarray], image_name: str) -> np.ndarray:
    if image_name not in masks:
        raise FileNotFoundError(f"{image_name} was not found in annotation JSON.")
    mask = masks[image_name]
    if mask.shape[:2] != (512, 512):
        mask = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)
    return (mask > 0).astype(np.uint8) * 255


def overlay_mask(gray: np.ndarray, mask: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    red = rgb.copy()
    red[mask > 0] = (35, 35, 255)
    return cv2.addWeighted(red, 0.4, rgb, 0.6, 0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dicom-dir", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, help="Raw annotated images folder, kept for traceability.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--view-a-index", type=int, required=True)
    parser.add_argument("--frame-a", type=int, required=True)
    parser.add_argument("--view-a-image", required=True)
    parser.add_argument("--view-b-index", type=int, required=True)
    parser.add_argument("--frame-b", type=int, required=True)
    parser.add_argument("--view-b-image", required=True)
    args = parser.parse_args()

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    clips = find_xa_clips(args.dicom_dir)
    clip_a = clips[args.view_a_index]
    clip_b = clips[args.view_b_index]
    masks = load_coco_masks(args.annotations)

    gray_a = read_frame(clip_a, args.frame_a)
    gray_b = read_frame(clip_b, args.frame_b)
    mask_a = load_annotation_mask(masks, args.view_a_image)
    mask_b = load_annotation_mask(masks, args.view_b_image)

    cv2.imwrite(str(out / "view_a_original.png"), gray_a)
    cv2.imwrite(str(out / "view_b_original.png"), gray_b)
    cv2.imwrite(str(out / "view_a_mask.png"), mask_a)
    cv2.imwrite(str(out / "view_b_mask.png"), mask_b)
    cv2.imwrite(str(out / "view_a_overlay.png"), overlay_mask(gray_a, mask_a))
    cv2.imwrite(str(out / "view_b_overlay.png"), overlay_mask(gray_b, mask_b))

    clean_a, skel_a, dt_a, branches_a = extract_graph(mask_a)
    clean_b, skel_b, _dt_b, branches_b = extract_graph(mask_b)
    cv2.imwrite(str(out / "view_a_clean_mask.png"), clean_a)
    cv2.imwrite(str(out / "view_b_clean_mask.png"), clean_b)
    cv2.imwrite(str(out / "view_a_skeleton.png"), skel_a)
    cv2.imwrite(str(out / "view_b_skeleton.png"), skel_b)

    view_a = view_from_clip(clip_a, "view_a")
    view_b = view_from_clip(clip_b, "view_b")
    matches = find_branch_matches(branches_a, branches_b, view_a, view_b)
    mag_factor = getattr(
        pydicom.dcmread(str(clip_a.path), stop_before_pixels=True, force=True),
        "EstimatedRadiographicMagnificationFactor",
        1.0,
    )
    build_mesh(matches, dt_a, out, mag_factor)

    summary = {
        "dicom_dir": str(args.dicom_dir),
        "annotations": str(args.annotations),
        "image_dir": str(args.image_dir) if args.image_dir else "",
        "view_a": {
            **view_a,
            "clip_index": args.view_a_index,
            "frame": args.frame_a,
            "annotation_image": args.view_a_image,
            "dicom_path": str(clip_a.path),
        },
        "view_b": {
            **view_b,
            "clip_index": args.view_b_index,
            "frame": args.frame_b,
            "annotation_image": args.view_b_image,
            "dicom_path": str(clip_b.path),
        },
        "num_view_a_branches": len(branches_a),
        "num_view_b_branches": len(branches_b),
        "branch_status_counts": {
            "reliable": sum(1 for match in matches if match["status"] == "reliable"),
            "usable": sum(1 for match in matches if match["status"] == "usable"),
            "single_view_preserved": sum(1 for match in matches if match["status"] == "single_view_preserved"),
        },
        "outputs": {
            "obj": str(out / "pipeline_hybrid_qca_tree.obj"),
            "quality_radius_report": str(out / "pipeline_branch_quality_radius.csv"),
            "view_a_overlay": str(out / "view_a_overlay.png"),
            "view_b_overlay": str(out / "view_b_overlay.png"),
        },
    }
    (out / "pipeline_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
