"""
Mask-input anatomy localization model.

Takes a binary vessel segmentation mask (1 channel) and outputs per-pixel
SYNTAX coronary segment labels (26 classes).

Pipeline:
  Angiogram → [MobileUNetv3 vessel segmentation] → binary mask
           → [MaskLocalizationNet] → per-pixel anatomy map
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large

from localization_labels import MERGED_NUM_ANATOMY_CLASSES as NUM_ANATOMY_CLASSES
from model_lightweight import CBAM, DoubleDSConv


class MaskLocalizationNet(nn.Module):
    """
    MobileNetV3-Large encoder + U-Net decoder for anatomy localization.

    Input:  (B, 1, H, W)  binary vessel mask  (0=background, 1=vessel)
    Output: {"anatomy": (B, 26, H, W)}  per-pixel SYNTAX segment logits
    """

    def __init__(self, n_anatomy_classes=NUM_ANATOMY_CLASSES, pretrained=True):
        super().__init__()
        self.n_anatomy_classes = n_anatomy_classes

        weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        encoder = mobilenet_v3_large(weights=weights).features

        # Patch first Conv2d: 3-channel RGB → 1-channel mask
        # Average the pretrained weights across the channel dim so spatial
        # feature detectors are preserved while accepting 1-channel input.
        first_conv = encoder[0][0]  # Conv2dNormActivation[0] = Conv2d(3, 16, ...)
        new_first = nn.Conv2d(
            1, first_conv.out_channels,
            kernel_size=first_conv.kernel_size,
            stride=first_conv.stride,
            padding=first_conv.padding,
            bias=False,
        )
        if pretrained:
            new_first.weight = nn.Parameter(
                first_conv.weight.mean(dim=1, keepdim=True)
            )
        encoder[0][0] = new_first
        self.encoder = encoder

        # Decoder mirrors MultiTaskMobileUNetv3 exactly, just without vessel/stenosis heads
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

        # Last skip concatenates with the 1-channel input mask (not 3-channel image)
        self.up5 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv5 = DoubleDSConv(64 + 1, 32)
        self.att5 = CBAM(32)

        self.anatomy_head = nn.Conv2d(32, n_anatomy_classes, kernel_size=1)

    def forward(self, x):
        # x: (B, 1, H, W)
        x_0 = self.encoder[0:2](x)    # (B, 16,  H/2,  W/2)
        x_1 = self.encoder[2:4](x_0)  # (B, 24,  H/4,  W/4)
        x_2 = self.encoder[4:7](x_1)  # (B, 40,  H/8,  W/8)
        x_3 = self.encoder[7:13](x_2) # (B, 112, H/16, W/16)
        x_4 = self.encoder[13:](x_3)  # (B, 960, H/32, W/32)

        d1 = self.att1(self.conv1(self._cat(self.up1(x_4), x_3)))
        d2 = self.att2(self.conv2(self._cat(self.up2(d1), x_2)))
        d3 = self.att3(self.conv3(self._cat(self.up3(d2), x_1)))
        d4 = self.att4(self.conv4(self._cat(self.up4(d3), x_0)))
        d5 = self.att5(self.conv5(self._cat(self.up5(d4), x)))

        return {"anatomy": self.anatomy_head(d5)}

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
    model = MaskLocalizationNet(pretrained=False)
    x = torch.zeros(1, 1, 512, 512)
    x[0, 0, 100:300, 200:350] = 1.0  # fake vessel region
    y = model(x)
    print("anatomy:", tuple(y["anatomy"].shape))
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Parameters: {params:.1f}M")
