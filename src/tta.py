"""
Flip-based test-time augmentation (TTA) for the MobileUNetv3 segmentation
model. Averaging predictions over the identity image plus its horizontal,
vertical, and both-axis flips (flipped back before averaging) measured
+0.4-0.6pt Dice/IoU on both the val and test splits with no precision/recall
trade-off (see eval_segmentation_tta.py), at the cost of 4x the segmentation
forward passes.
"""

import torch

# Each entry is the set of dims to flip the image before the forward pass;
# the prediction is flipped back over the same dims before averaging.
_TTA_FLIPS = [
    None,        # identity
    [-1],        # horizontal flip (width)
    [-2],        # vertical flip (height)
    [-2, -1],    # both
]


@torch.no_grad()
def predict_tta(model, image):
    """Average sigmoid probabilities across flips of `image`.

    `model` must return either a raw logits tensor or a dict with an 'out'
    key (as MobileUNetv3 does); `image` is (B, C, H, W).
    """
    probs_sum = None
    for dims in _TTA_FLIPS:
        img = image if dims is None else torch.flip(image, dims=dims)
        out = model(img)
        logits = out["out"] if isinstance(out, dict) else out
        probs = torch.sigmoid(logits)
        if dims is not None:
            probs = torch.flip(probs, dims=dims)
        probs_sum = probs if probs_sum is None else probs_sum + probs
    return probs_sum / len(_TTA_FLIPS)
