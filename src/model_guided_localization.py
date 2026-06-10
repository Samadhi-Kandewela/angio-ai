"""
Guided single-model localization.

One model does both vessel segmentation and anatomy localization.
The vessel prediction acts as an internal soft mask that guides the anatomy head,
so the anatomy head focuses only on vessel regions.

Input:  RGB angiogram (3, H, W)
Output: {
    "vessel":  (1,  H, W)  binary vessel logits
    "anatomy": (26, H, W)  per-pixel SYNTAX segment logits
}
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large

from localization_labels import NUM_ANATOMY_CLASSES
from model_lightweight import CBAM, DoubleDSConv


class GuidedLocalizationNet(nn.Module):
    """
    MobileNetV3-Large encoder + shared U-Net decoder.

    Key difference from old MultiTaskMobileUNetv3:
      - No stenosis head
      - Vessel prediction masks decoder features before anatomy head
        (anatomy head only sees vessel regions)
    """

    def __init__(self, n_anatomy_classes=NUM_ANATOMY_CLASSES, pretrained=True):
        super().__init__()
        self.n_anatomy_classes = n_anatomy_classes

        weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        self.encoder = mobilenet_v3_large(weights=weights).features

        # Decoder — identical to MultiTaskMobileUNetv3
        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv1 = DoubleDSConv(960 + 112, 512)
        self.att1 = CBAM(512)

        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv2 = DoubleDSConv(512 + 40, 256)
        self.att2 = CBAM(256)

        self.up3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv3 = DoubleDSConv(256 + 24, 128)
        self.att3 = CBAM(128)

        self.up4 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv4 = DoubleDSConv(128 + 16, 64)
        self.att4 = CBAM(64)

        self.up5 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv5 = DoubleDSConv(64 + 3, 32)
        self.att5 = CBAM(32)

        # Vessel head — predicts where vessels are
        self.vessel_head = nn.Conv2d(32, 1, kernel_size=1)

        # Anatomy head — receives vessel-masked features
        self.anatomy_head = nn.Conv2d(32, n_anatomy_classes, kernel_size=1)

    def forward(self, x):
        # ── Encoder ──────────────────────────────────────────────────────────
        x_0 = self.encoder[0:2](x)    # (B, 16,  H/2,  W/2)
        x_1 = self.encoder[2:4](x_0)  # (B, 24,  H/4,  W/4)
        x_2 = self.encoder[4:7](x_1)  # (B, 40,  H/8,  W/8)
        x_3 = self.encoder[7:13](x_2) # (B, 112, H/16, W/16)
        x_4 = self.encoder[13:](x_3)  # (B, 960, H/32, W/32)

        # ── Decoder ──────────────────────────────────────────────────────────
        d1 = self.att1(self.conv1(self._cat(self.up1(x_4), x_3)))
        d2 = self.att2(self.conv2(self._cat(self.up2(d1), x_2)))
        d3 = self.att3(self.conv3(self._cat(self.up3(d2), x_1)))
        d4 = self.att4(self.conv4(self._cat(self.up4(d3), x_0)))
        d5 = self.att5(self.conv5(self._cat(self.up5(d4), x)))

        # ── Vessel prediction ─────────────────────────────────────────────────
        vessel_logits = self.vessel_head(d5)          # (B, 1, H, W)
        vessel_mask = torch.sigmoid(vessel_logits)    # soft mask [0, 1]

        # ── Anatomy prediction guided by vessel mask ──────────────────────────
        # Multiply decoder features by vessel probability so the anatomy head
        # only activates on vessel regions.
        guided = d5 * vessel_mask
        anatomy_logits = self.anatomy_head(guided)    # (B, 26, H, W)

        return {
            "vessel": vessel_logits,
            "anatomy": anatomy_logits,
        }

    @staticmethod
    def _cat(upsampled, skip):
        diff_y = skip.size(2) - upsampled.size(2)
        diff_x = skip.size(3) - upsampled.size(3)
        upsampled = F.pad(
            upsampled,
            [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2],
        )
        return torch.cat([skip, upsampled], dim=1)


if __name__ == "__main__":
    model = GuidedLocalizationNet(pretrained=False)
    x = torch.randn(1, 3, 512, 512)
    y = model(x)
    print("vessel: ", tuple(y["vessel"].shape))
    print("anatomy:", tuple(y["anatomy"].shape))
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Parameters: {params:.1f}M")
