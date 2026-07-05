"""
Tversky-family losses for imbalanced binary segmentation.

Standard Dice/BCE weight false positives and false negatives equally. Vessel
segmentation on angiograms is recall-limited (thin/faint branches get missed
more often than background gets mislabeled as vessel), so these losses expose
alpha/beta to bias the gradient toward penalizing false negatives harder.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TverskyLoss(nn.Module):
    """Generalized Dice: alpha weights false positives, beta weights false
    negatives. alpha == beta == 0.5 reduces to standard Dice loss."""

    def __init__(self, alpha=0.3, beta=0.7, smooth=1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def tversky_index(self, pred, target):
        pred = pred.contiguous()
        target = target.contiguous()
        tp = (pred * target).sum(dim=(2, 3))
        fp = (pred * (1 - target)).sum(dim=(2, 3))
        fn = ((1 - pred) * target).sum(dim=(2, 3))
        return (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)

    def forward(self, pred, target):
        return (1 - self.tversky_index(pred, target)).mean()


class FocalTverskyLoss(TverskyLoss):
    """Raises (1 - Tversky index) to the power gamma < 1, which up-weights
    the loss contribution of hard/low-overlap samples (e.g. frames where the
    vessel tree is thin or faint) relative to easy ones."""

    def __init__(self, alpha=0.3, beta=0.7, gamma=0.75, smooth=1.0):
        super().__init__(alpha=alpha, beta=beta, smooth=smooth)
        self.gamma = gamma

    def forward(self, pred, target):
        tversky = self.tversky_index(pred, target)
        return ((1 - tversky) ** self.gamma).mean()


def _soft_erode(img):
    p1 = -F.max_pool2d(-img, kernel_size=(3, 1), stride=1, padding=(1, 0))
    p2 = -F.max_pool2d(-img, kernel_size=(1, 3), stride=1, padding=(0, 1))
    return torch.min(p1, p2)


def _soft_dilate(img):
    return F.max_pool2d(img, kernel_size=3, stride=1, padding=1)


def _soft_open(img):
    return _soft_dilate(_soft_erode(img))


def soft_skeletonize(img, iterations=10):
    """Differentiable approximation of a morphological skeleton, built from
    iterated soft erode/open (Shit et al., "clDice", CVPR 2021). Operates on
    soft (sigmoid) probabilities, not hard 0/1 masks."""
    img1 = _soft_open(img)
    skel = F.relu(img - img1)
    for _ in range(iterations):
        img = _soft_erode(img)
        img1 = _soft_open(img)
        delta = F.relu(img - img1)
        skel = skel + F.relu(delta - skel * delta)
    return skel


class SoftClDiceLoss(nn.Module):
    """clDice: rewards centerline/topological connectivity of thin tubular
    structures (vessels), which pixel-overlap losses (Dice/Tversky) cannot
    distinguish -- two masks with the identical pixel count can differ in
    whether the vessel tree stays connected or is fragmented. Meant to be
    combined with, not replace, a pixel-overlap loss (see
    train_mobileunet_cldice.py)."""

    def __init__(self, iterations=10, smooth=1.0):
        super().__init__()
        self.iterations = iterations
        self.smooth = smooth

    def forward(self, pred, target):
        skel_pred = soft_skeletonize(pred, self.iterations)
        skel_true = soft_skeletonize(target, self.iterations)

        tprec = (torch.sum(skel_pred * target) + self.smooth) / (torch.sum(skel_pred) + self.smooth)
        tsens = (torch.sum(skel_true * pred) + self.smooth) / (torch.sum(skel_true) + self.smooth)
        return 1 - 2.0 * (tprec * tsens) / (tprec + tsens + self.smooth)
