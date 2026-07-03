import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large

from localization_labels import NUM_ANATOMY_CLASSES
from model_lightweight import CBAM, DoubleDSConv


class MultiTaskMobileUNetv3(nn.Module):
    """
    MobileNetV3-Large encoder with one shared decoder and three prediction heads:
      - vessel: binary vessel segmentation logits
      - anatomy: multi-class SYNTAX anatomical segment logits
      - stenosis: binary stenosis region logits
    """

    def __init__(self, n_anatomy_classes=NUM_ANATOMY_CLASSES, pretrained=True):
        super().__init__()
        self.n_anatomy_classes = n_anatomy_classes

        weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        self.encoder = mobilenet_v3_large(weights=weights).features

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

        self.vessel_head = nn.Conv2d(32, 1, kernel_size=1)
        self.anatomy_head = nn.Conv2d(32, n_anatomy_classes, kernel_size=1)

    def forward(self, x):
        x_0 = self.encoder[0:2](x)
        x_1 = self.encoder[2:4](x_0)
        x_2 = self.encoder[4:7](x_1)
        x_3 = self.encoder[7:13](x_2)
        x_4 = self.encoder[13:](x_3)

        d1 = self._upsample_cat(self.up1(x_4), x_3)
        d1 = self.att1(self.conv1(d1))

        d2 = self._upsample_cat(self.up2(d1), x_2)
        d2 = self.att2(self.conv2(d2))

        d3 = self._upsample_cat(self.up3(d2), x_1)
        d3 = self.att3(self.conv3(d3))

        d4 = self._upsample_cat(self.up4(d3), x_0)
        d4 = self.att4(self.conv4(d4))

        d5 = self._upsample_cat(self.up5(d4), x)
        d5 = self.att5(self.conv5(d5))

        return {
            "vessel": self.vessel_head(d5),
            "anatomy": self.anatomy_head(d5),
            "features": x_4,
        }

    @staticmethod
    def _upsample_cat(x, skip):
        diff_y = skip.size(2) - x.size(2)
        diff_x = skip.size(3) - x.size(3)
        x = F.pad(x, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        return torch.cat([skip, x], dim=1)


if __name__ == "__main__":
    model = MultiTaskMobileUNetv3(pretrained=False)
    x = torch.randn(1, 3, 512, 512)
    y = model(x)
    print("vessel:", tuple(y["vessel"].shape))
    print("anatomy:", tuple(y["anatomy"].shape))
