"""
Guided single-model localization with ASPP.

One model does both vessel segmentation and anatomy localization.

Architecture:
  MobileNetV3-Large encoder  (unchanged, pretrained)
          ↓
  ASPP (multi-scale context)  (NEW — captures global artery layout)
          ↓
  U-Net decoder + CBAM attention
          ↓
  vessel_head  → vessel logits
  vessel_mask  = sigmoid(vessel_logits)
  anatomy_head → anatomy logits  (features × vessel_mask before head)

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


# ─── ASPP ─────────────────────────────────────────────────────────────────────

class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling.

    Captures multi-scale context at the encoder bottleneck using parallel
    dilated convolutions + global average pooling.  Important for artery
    localisation because identifying LAD vs RCA vs LCX requires understanding
    the global layout of the whole image, not just a small local patch.

    in_channels  : 960  (MobileNetV3-Large bottleneck output)
    out_channels : 256  (compact representation passed to decoder)
    dilations    : (1, 2, 4, 8)  — chosen for 16×16 feature map size
    """

    def __init__(self, in_channels=960, out_channels=256, dilations=(1, 2, 4, 8)):
        super().__init__()
        mid = out_channels

        # Branch 1: 1×1 conv (dilation=1)
        self.b1 = nn.Sequential(
            nn.Conv2d(in_channels, mid, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
        )
        # Branches 2-4: 3×3 atrous convs (dilations 2, 4, 8)
        self.b2 = self._atrous(in_channels, mid, dilations[1])
        self.b3 = self._atrous(in_channels, mid, dilations[2])
        self.b4 = self._atrous(in_channels, mid, dilations[3])

        # Branch 5: global average pooling (full image context)
        self.b5 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, mid, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
        )

        # Project 5 branches → out_channels
        self.project = nn.Sequential(
            nn.Conv2d(mid * 5, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def _atrous(in_ch, out_ch, dilation):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3,
                      padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        size = x.shape[-2:]
        b5 = F.interpolate(self.b5(x), size=size, mode="bilinear", align_corners=True)
        out = torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x), b5], dim=1)
        return self.project(out)


# ─── Main model ───────────────────────────────────────────────────────────────

class GuidedLocalizationNet(nn.Module):
    """
    MobileNetV3-Large encoder (frozen structure, pretrained weights)
    + ASPP bottleneck
    + U-Net decoder with CBAM attention
    + vessel head + vessel-guided anatomy head
    """

    def __init__(self, n_anatomy_classes=NUM_ANATOMY_CLASSES, pretrained=True):
        super().__init__()
        self.n_anatomy_classes = n_anatomy_classes

        # ── Encoder (MobileNetV3 — structure unchanged) ───────────────────────
        weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        self.encoder = mobilenet_v3_large(weights=weights).features

        # ── ASPP bottleneck (NEW) ─────────────────────────────────────────────
        # Takes 960-ch encoder output, produces 256-ch context-enriched features
        self.aspp = ASPP(in_channels=960, out_channels=256, dilations=(1, 2, 4, 8))

        # ── Decoder ──────────────────────────────────────────────────────────
        # conv1 now takes 256 (ASPP out) + 112 (skip) instead of 960 + 112
        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv1 = DoubleDSConv(256 + 112, 512)
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

        # ── Heads ─────────────────────────────────────────────────────────────
        self.vessel_head  = nn.Conv2d(32, 1, kernel_size=1)
        self.anatomy_head = nn.Conv2d(32, n_anatomy_classes, kernel_size=1)

    def forward(self, x):
        # ── Encoder (MobileNetV3 — unchanged) ────────────────────────────────
        x_0 = self.encoder[0:2](x)    # (B, 16,  H/2,  W/2)
        x_1 = self.encoder[2:4](x_0)  # (B, 24,  H/4,  W/4)
        x_2 = self.encoder[4:7](x_1)  # (B, 40,  H/8,  W/8)
        x_3 = self.encoder[7:13](x_2) # (B, 112, H/16, W/16)
        x_4 = self.encoder[13:](x_3)  # (B, 960, H/32, W/32)

        # ── ASPP (multi-scale context enrichment) ────────────────────────────
        x_4 = self.aspp(x_4)          # (B, 256, H/32, W/32)

        # ── Decoder ──────────────────────────────────────────────────────────
        d1 = self.att1(self.conv1(self._cat(self.up1(x_4), x_3)))
        d2 = self.att2(self.conv2(self._cat(self.up2(d1),  x_2)))
        d3 = self.att3(self.conv3(self._cat(self.up3(d2),  x_1)))
        d4 = self.att4(self.conv4(self._cat(self.up4(d3),  x_0)))
        d5 = self.att5(self.conv5(self._cat(self.up5(d4),  x)))

        # ── Vessel prediction ─────────────────────────────────────────────────
        vessel_logits = self.vessel_head(d5)
        vessel_mask   = torch.sigmoid(vessel_logits)   # soft mask [0, 1]

        # ── Vessel-guided anatomy prediction ─────────────────────────────────
        anatomy_logits = self.anatomy_head(d5 * vessel_mask)

        return {
            "vessel":  vessel_logits,
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
