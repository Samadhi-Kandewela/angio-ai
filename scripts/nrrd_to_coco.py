"""
Convert 3D Slicer NRRD segmentation exports to COCO JSON format.

Workflow:
  1. In Slicer: Segment Editor → draw artery mask → File > Export Segmentation > Labelmap > .nrrd
  2. Organize your files (see --help for folder layout)
  3. Run this script to produce train.json / val.json / test.json

Usage:
  python nrrd_to_coco.py --images-dir data/train/images --masks-dir data/train/masks --output data/train/annotations/train.json
"""

import os
import json
import argparse
import numpy as np
import cv2

try:
    import nrrd
except ImportError:
    raise ImportError("Install pynrrd first:  pip install pynrrd")


def nrrd_to_binary_mask(nrrd_path: str) -> np.ndarray:
    """Load a Slicer labelmap NRRD and return a 2D binary uint8 mask.

    Slicer exports labelmaps as (H, W, 1) for 2D images, or (H, W, D) for
    3D volumes. We squeeze single-slice dimensions first, then handle any
    remaining 3D volume by picking the slice with the most foreground.
    """
    data, header = nrrd.read(nrrd_path)

    # Squeeze any size-1 dimensions (handles Slicer's (H, W, 1) format)
    data = np.squeeze(data)

    if data.ndim == 3:
        # True 3D volume — pick the slice (along last axis) with most foreground
        counts = [(data[:, :, i] > 0).sum() for i in range(data.shape[2])]
        data = data[:, :, np.argmax(counts)]

    # Any non-zero label → foreground (vessel)
    mask = (data > 0).astype(np.uint8)

    # Slicer exports in LPS space where axes are (col, row) — transpose to (row, col)
    mask = mask.T

    return mask


def mask_to_coco_polygons(mask: np.ndarray):
    """
    Convert binary mask to a list of COCO polygon segmentations.
    Returns (polygons, area, bbox) or None if no foreground found.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    polygons = []
    total_area = 0.0

    for contour in contours:
        if contour.shape[0] < 3:          # need at least a triangle
            continue
        # Simplify contour slightly to reduce polygon size
        epsilon = 0.5
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if approx.shape[0] < 3:
            continue
        flat = approx.flatten().tolist()
        polygons.append(flat)
        total_area += cv2.contourArea(contour)

    if not polygons:
        return None, 0.0, [0, 0, 0, 0]

    # Bounding box around all contours combined
    all_contours = np.concatenate(contours)
    x, y, w, h = cv2.boundingRect(all_contours)

    return polygons, total_area, [int(x), int(y), int(w), int(h)]


def build_coco(images_dir: str, masks_dir: str, output_path: str):
    """
    Pair each image with its corresponding NRRD mask and build a COCO JSON.

    Naming convention: image  patient01.png  ↔  mask  patient01.nrrd
    (stem must match; extensions can differ)
    """
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
    mask_extension   = '.nrrd'

    # Index masks by stem
    mask_map = {}
    for fname in os.listdir(masks_dir):
        if fname.lower().endswith(mask_extension):
            stem = os.path.splitext(fname)[0]
            mask_map[stem] = os.path.join(masks_dir, fname)

    coco = {
        "images":      [],
        "annotations": [],
        "categories":  [{"id": 1, "name": "vessel", "supercategory": "anatomy"}]
    }

    image_id     = 1
    annotation_id = 1
    skipped      = []

    image_files = sorted([
        f for f in os.listdir(images_dir)
        if os.path.splitext(f)[1].lower() in image_extensions
    ])

    for fname in image_files:
        stem = os.path.splitext(fname)[0]

        if stem not in mask_map:
            print(f"  [WARN] No mask found for {fname} — skipping")
            skipped.append(fname)
            continue

        img_path  = os.path.join(images_dir, fname)
        img       = cv2.imread(img_path)
        if img is None:
            print(f"  [WARN] Could not read image {fname} — skipping")
            skipped.append(fname)
            continue

        h, w = img.shape[:2]

        mask = nrrd_to_binary_mask(mask_map[stem])

        # Resize mask to match image dimensions if Slicer exported at different size
        if mask.shape != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        polygons, area, bbox = mask_to_coco_polygons(mask)

        coco["images"].append({
            "id":        image_id,
            "file_name": fname,
            "height":    h,
            "width":     w
        })

        if polygons:
            coco["annotations"].append({
                "id":          annotation_id,
                "image_id":    image_id,
                "category_id": 1,
                "segmentation": polygons,
                "area":        area,
                "bbox":        bbox,
                "iscrowd":     0
            })
            annotation_id += 1
        else:
            print(f"  [WARN] Empty mask for {fname} — image added but no annotation")

        image_id += 1

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(coco, f, indent=2)

    print(f"\nDone.")
    print(f"  Images:      {len(coco['images'])}")
    print(f"  Annotations: {len(coco['annotations'])}")
    print(f"  Skipped:     {len(skipped)}")
    print(f"  Output:      {output_path}")


def get_args():
    parser = argparse.ArgumentParser(description="Convert Slicer NRRD masks to COCO JSON")
    parser.add_argument('--images-dir', required=True,
                        help='Folder containing the angiography images (PNG/JPG)')
    parser.add_argument('--masks-dir',  required=True,
                        help='Folder containing the Slicer NRRD labelmap files')
    parser.add_argument('--output',     required=True,
                        help='Output COCO JSON path, e.g. data/train/annotations/train.json')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    print(f"Images : {args.images_dir}")
    print(f"Masks  : {args.masks_dir}")
    print(f"Output : {args.output}")
    build_coco(args.images_dir, args.masks_dir, args.output)
