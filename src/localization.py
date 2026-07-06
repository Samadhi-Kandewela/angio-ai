import numpy as np

from localization_labels import (
    merged_segment_artery, merged_segment_group, merged_segment_label,
    segment_artery, segment_group, segment_label,
)


def anatomy_logits_to_map_and_confidence(anatomy_logits):
    """
    Convert anatomy logits/probabilities to class map and confidence map.

    Accepts either:
      - logits shaped (C, H, W)
      - logits shaped (1, C, H, W)
      - probabilities with the same shapes
    """
    arr = np.asarray(anatomy_logits)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim != 3:
        raise ValueError(f"Expected anatomy logits shaped CxHxW or 1xCxHxW, got {arr.shape}")

    # Softmax is stable even if arr already resembles probabilities.
    arr = arr.astype(np.float32)
    arr = arr - np.max(arr, axis=0, keepdims=True)
    exp = np.exp(arr)
    probs = exp / np.maximum(np.sum(exp, axis=0, keepdims=True), 1e-8)

    class_map = np.argmax(probs, axis=0).astype(np.uint8)
    confidence = np.max(probs, axis=0).astype(np.float32)
    return class_map, confidence


def localize_point(class_map, confidence_map, point_yx, radius=7, use_merged=False):
    """
    Localize a point to the dominant non-background segment in a local window.

    `use_merged` selects the label scheme: False for the raw 26-class SYNTAX
    ids (MultiTaskMobileUNetv3), True for the 15-class merged ids produced by
    MaskLocalizationNet (see localization_labels.py).

    Returns a dict with segment_id, label, artery, group, and confidence.
    """
    label_fn = merged_segment_label if use_merged else segment_label
    artery_fn = merged_segment_artery if use_merged else segment_artery
    group_fn = merged_segment_group if use_merged else segment_group

    y, x = int(point_yx[0]), int(point_yx[1])
    h, w = class_map.shape
    y1, y2 = max(0, y - radius), min(h, y + radius + 1)
    x1, x2 = max(0, x - radius), min(w, x + radius + 1)

    patch = class_map[y1:y2, x1:x2]
    conf_patch = confidence_map[y1:y2, x1:x2]
    valid = patch > 0

    if not np.any(valid):
        return {
            "segment_id": 0,
            "label": "unknown",
            "artery": "unknown",
            "group": "unknown",
            "confidence": 0.0,
        }

    ids, counts = np.unique(patch[valid], return_counts=True)
    segment_id = int(ids[np.argmax(counts)])
    segment_pixels = patch == segment_id
    confidence = float(np.mean(conf_patch[segment_pixels]))

    return {
        "segment_id": segment_id,
        "label": label_fn(segment_id),
        "artery": artery_fn(segment_id),
        "group": group_fn(segment_id),
        "confidence": confidence,
    }


def localize_lesions(lesions, class_map, confidence_map, radius=7, use_merged=False):
    """Attach anatomical localization dicts to QCA lesion dictionaries."""
    localized = []
    for lesion in lesions:
        enriched = dict(lesion)
        enriched["localization"] = localize_point(
            class_map,
            confidence_map,
            lesion["min_pt"],
            radius=radius,
            use_merged=use_merged,
        )
        localized.append(enriched)
    return localized
