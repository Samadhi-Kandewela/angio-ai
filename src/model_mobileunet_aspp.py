"""
MobileUNetv3 + ASPP at the encoder/decoder bottleneck.

Four training-recipe interventions this session -- Focal Tversky loss
reweighting, a training-methodology overhaul (real val split, AdamW,
warmup+cosine, norm-based grad clipping, encoder freeze phase), a clDice
topology loss, and simply retraining from scratch -- all converged on the
same ~0.777-0.784 test Dice / ~0.646-0.649 test IoU band (see
train_mobileunet_tversky.py, train_mobileunet_v2.py,
train_mobileunet_cldice.py). That consistency points at a capacity /
receptive-field ceiling rather than a training problem: MobileNetV3-Large
downsamples a 512x512 input by 32x before the decoder ever sees it, so the
whole image is squeezed through a 16x16 feature map. Thin, faint, distal
vessel branches may simply not survive that bottleneck no matter how the
loss or schedule is tuned afterwards.

ASPP (Atrous Spatial Pyramid Pooling, Chen et al., DeepLabV3) widens the
receptive field at that exact bottleneck via parallel dilated convolutions
plus global-average-pooled context, all at the same 16x16 resolution --
i.e. more context per bottleneck pixel without any further downsampling,
which a plain 1x1/3x3 conv cannot provide. Dilation rates here are 2/4/6
rather than DeepLabV3's usual 6/12/18: those larger rates are tuned for
much bigger feature maps (~33x33 at DeepLabV3's native resolution) and
would be dominated by zero-padding on a 16x16 map. Each atrous branch is
depthwise-separable (matching this codebase's existing DSConv convention
in model_lightweight.py) to keep the module mobile-sized rather than
ballooning parameter count.

This is a new class in a new file -- model_lightweight.MobileUNetv3 is not
modified.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_large

from model_lightweight import CBAM, DoubleDSConv


class _DepthwiseAtrousConv(nn.Sequential):
    """Depthwise dilated 3x3 + pointwise 1x1 (the DSConv pattern from
    model_lightweight.py), with a configurable dilation rate."""

    def __init__(self, in_channels, out_channels, dilation):
        super().__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=dilation,
                       dilation=dilation, groups=in_channels, bias=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class _ASPPPooling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        size = x.shape[-2:]
        pooled = x.mean(dim=(2, 3), keepdim=True)
        pooled = self.relu(self.bn(self.conv(pooled)))
        return F.interpolate(pooled, size=size, mode="bilinear", align_corners=False)


class ASPP(nn.Module):
    """Sized for a 16x16 bottleneck feature map: 1x1 branch + three
    depthwise-atrous branches (dilation 2/4/6) + global-pooled branch,
    concatenated and projected back down to out_channels."""

    def __init__(self, in_channels, out_channels=256, atrous_rates=(2, 4, 6)):
        super().__init__()
        self.branch1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.branches = nn.ModuleList(
            _DepthwiseAtrousConv(in_channels, out_channels, rate) for rate in atrous_rates
        )
        self.pooling = _ASPPPooling(in_channels, out_channels)

        n_branches = 2 + len(atrous_rates)
        self.project = nn.Sequential(
            nn.Conv2d(n_branches * out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )

    def forward(self, x):
        feats = [self.branch1x1(x)] + [b(x) for b in self.branches] + [self.pooling(x)]
        return self.project(torch.cat(feats, dim=1))


class MobileUNetv3ASPP(nn.Module):
    """MobileUNetv3 (MobileNetV3-Large encoder + CBAM-attention decoder,
    architecturally identical to model_lightweight.MobileUNetv3) with an
    ASPP module inserted between the encoder and decoder."""

    def __init__(self, n_classes, pretrained=True, aspp_out_channels=256):
        super().__init__()
        self.n_classes = n_classes

        self.encoder = mobilenet_v3_large(pretrained=pretrained).features
        self.aspp = ASPP(in_channels=960, out_channels=aspp_out_channels)

        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv1 = DoubleDSConv(aspp_out_channels + 112, 512)
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

        self.final_conv = nn.Conv2d(32, n_classes, kernel_size=1)

    def forward(self, x):
        x_0 = self.encoder[0:2](x)     # 16ch, 256x256
        x_1 = self.encoder[2:4](x_0)   # 24ch, 128x128
        x_2 = self.encoder[4:7](x_1)   # 40ch, 64x64
        x_3 = self.encoder[7:13](x_2)  # 112ch, 32x32
        x_4 = self.encoder[13:](x_3)   # 960ch, 16x16

        bottleneck = self.aspp(x_4)    # aspp_out_channels, 16x16

        d1 = self.up1(bottleneck)
        d1 = self._pad_cat(d1, x_3)
        d1 = self.att1(self.conv1(d1))

        d2 = self.up2(d1)
        d2 = self._pad_cat(d2, x_2)
        d2 = self.att2(self.conv2(d2))

        d3 = self.up3(d2)
        d3 = self._pad_cat(d3, x_1)
        d3 = self.att3(self.conv3(d3))

        d4 = self.up4(d3)
        d4 = self._pad_cat(d4, x_0)
        d4 = self.att4(self.conv4(d4))

        d5 = self.up5(d4)
        d5 = self._pad_cat(d5, x)
        d5 = self.att5(self.conv5(d5))

        out = self.final_conv(d5)
        return {"out": out, "features": bottleneck}

    @staticmethod
    def _pad_cat(up, skip):
        diff_y = skip.size(2) - up.size(2)
        diff_x = skip.size(3) - up.size(3)
        up = F.pad(up, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        return torch.cat([skip, up], dim=1)


if __name__ == "__main__":
    model = MobileUNetv3ASPP(n_classes=1, pretrained=False)
    x = torch.randn(2, 3, 512, 512)  # batch > 1: BatchNorm needs >1 sample per channel in train mode
    y = model(x)
    print("out:", tuple(y["out"].shape))
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Params: {n_params / 1e6:.2f}M")
