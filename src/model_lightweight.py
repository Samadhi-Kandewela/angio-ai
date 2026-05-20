import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision import models
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large, deeplabv3_resnet101
from torchvision.models import mobilenet_v3_large

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        # self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # self.max_pool = nn.AdaptiveMaxPool2d(1)
        # Use torch.mean/amax for better ONNX support
        pass
        
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        # max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        
        avg_pool = torch.mean(x, dim=(2, 3), keepdim=True)
        max_pool = torch.amax(x, dim=(2, 3), keepdim=True)
        
        avg_out = self.fc2(self.relu1(self.fc1(avg_pool)))
        max_out = self.fc2(self.relu1(self.fc1(max_pool)))
        
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel_size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self, planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        out = x * self.ca(x)
        result = out * self.sa(out)
        return result

class DSConv(nn.Module):
    """Depthwise Separable Convolution"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class DoubleDSConv(nn.Module):
    """(DSConv => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            DSConv(in_channels, out_channels),
            DSConv(out_channels, out_channels)
        )

    def forward(self, x):
        return self.double_conv(x)

class DSCUNet(nn.Module):
    """
    U-Net with Depthwise Separable Convolutions to reduce FLOPs and parameters.
    Architecture is similar to standard U-Net but replacing standard Conv with DSConv.
    """
    def __init__(self, n_channels, n_classes):
        super(DSCUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # Use DoubleDSConv instead of standard DoubleConv
        self.inc = DoubleDSConv(n_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleDSConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleDSConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleDSConv(256, 512))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleDSConv(512, 1024))
        
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv1 = DoubleDSConv(1024 + 512, 512)
        
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv2 = DoubleDSConv(512 + 256, 256)
        
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv3 = DoubleDSConv(256 + 128, 128)
        
        self.up4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv4 = DoubleDSConv(128 + 64, 64)
        
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        x = self.up1(x5)
        # Pad if necessary
        diffY = x4.size()[2] - x.size()[2]
        diffX = x4.size()[3] - x.size()[3]
        x = F.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x4, x], dim=1)
        x = self.conv1(x)
        
        x = self.up2(x)
        diffY = x3.size()[2] - x.size()[2]
        diffX = x3.size()[3] - x.size()[3]
        x = F.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x3, x], dim=1)
        x = self.conv2(x)
        
        x = self.up3(x)
        diffY = x2.size()[2] - x.size()[2]
        diffX = x2.size()[3] - x.size()[3]
        x = F.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x], dim=1)
        x = self.conv3(x)
        
        x = self.up4(x)
        diffY = x1.size()[2] - x.size()[2]
        diffX = x1.size()[3] - x.size()[3]
        x = F.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x1, x], dim=1)
        x = self.conv4(x)
        
        logits = self.outc(x)
        return logits

class MobileUNet(nn.Module):
    """
    U-Net with MobileNetV2 encoder.
    """
    def __init__(self, n_classes, pretrained=True):
        super(MobileUNet, self).__init__()
        self.n_classes = n_classes
        
        # Encoder: MobileNetV2
        mobilenet = models.mobilenet_v2(pretrained=pretrained)
        self.encoder = mobilenet.features
        
        # MobileNetV2 feature map channels:
        # layer 0: 32
        # layer 2: 24
        # layer 4: 32
        # layer 7: 64
        # layer 14: 160
        # layer 18: 1280 (last)
        
        # Decoder (Upsampling)
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv1 = DoubleDSConv(1280 + 96, 512) # 96 comes from layer 13 (96) - wait, let's verify indices
        
        # Let's simplify and pick specific layers for skip connections
        # Input: 3x512x512
        # Layer 1 (inverted_res_block): 16x256x256
        # Layer 3: 24x128x128
        # Layer 6: 32x64x64
        # Layer 13: 96x32x32
        # Layer 18: 1280x16x16
        
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv1 = DoubleDSConv(1280 + 96, 512)
        
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv2 = DoubleDSConv(512 + 32, 256)
        
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv3 = DoubleDSConv(256 + 24, 128)
        
        self.up4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv4 = DoubleDSConv(128 + 16, 64)
        
        self.up5 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv5 = DoubleDSConv(64 + 3, 32) # Skip connection from input image? Or just upsample?
        # Standard U-Net usually has 4 upsamples. MobileNet reduces 32x.
        
        self.final_conv = nn.Conv2d(32, n_classes, kernel_size=1)

    def forward(self, x):
        # Encoder path with skip connections
        # x: 3, 512, 512
        
        # Detailed feature extraction from MobileNetV2
        # We need to manually inspect layers or wrap them
        features = []
        out = x
        
        # Indices for skip connections (approximate for standard MobileNetV2)
        # 0-1: stride 2 -> 256 (16 ch)
        # 2-3: stride 2 -> 128 (24 ch)
        # 4-6: stride 2 -> 64  (32 ch)
        # 7-13: stride 2 -> 32 (96 ch)
        # 14-18: stride 2 -> 16 (1280 ch)
        
        # x_0 = self.encoder[0:2](x)   # 16, 256, 256
        # x_1 = self.encoder[2:4](x_0) # 24, 128, 128
        # x_2 = self.encoder[4:7](x_1) # 32, 64, 64
        # x_3 = self.encoder[7:14](x_2)# 96, 32, 32
        # x_4 = self.encoder[14:](x_3) # 1280, 16, 16
        
        x_0 = self.encoder[0:2](x)
        x_1 = self.encoder[2:4](x_0)
        x_2 = self.encoder[4:7](x_1)
        x_3 = self.encoder[7:14](x_2)
        x_4 = self.encoder[14:](x_3)
        
        # Decoder
        d1 = self.up1(x_4) # 32x32
        diffY = x_3.size()[2] - d1.size()[2]
        diffX = x_3.size()[3] - d1.size()[3]
        d1 = F.pad(d1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        d1 = torch.cat([x_3, d1], dim=1)
        d1 = self.conv1(d1)
        
        d2 = self.up2(d1) # 64x64
        diffY = x_2.size()[2] - d2.size()[2]
        diffX = x_2.size()[3] - d2.size()[3]
        d2 = F.pad(d2, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        d2 = torch.cat([x_2, d2], dim=1)
        d2 = self.conv2(d2)
        
        d3 = self.up3(d2) # 128x128
        diffY = x_1.size()[2] - d3.size()[2]
        diffX = x_1.size()[3] - d3.size()[3]
        d3 = F.pad(d3, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        d3 = torch.cat([x_1, d3], dim=1)
        d3 = self.conv3(d3)
        
        d4 = self.up4(d3) # 256x256
        diffY = x_0.size()[2] - d4.size()[2]
        diffX = x_0.size()[3] - d4.size()[3]
        d4 = F.pad(d4, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        d4 = torch.cat([x_0, d4], dim=1)
        d4 = self.conv4(d4)
        
        d5 = self.up5(d4) # 512x512
        diffY = x.size()[2] - d5.size()[2]
        diffX = x.size()[3] - d5.size()[3]
        d5 = F.pad(d5, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        d5 = torch.cat([x, d5], dim=1)
        d5 = self.conv5(d5)
        
        out = self.final_conv(d5)
        return out

class MobileUNetv2(nn.Module):
    """
    MobileUNet with CBAM Attention Gates (Enhanced Version).
    """
    def __init__(self, n_classes, pretrained=True):
        super(MobileUNetv2, self).__init__()
        self.n_classes = n_classes
        
        # Encoder: MobileNetV2
        mobilenet = models.mobilenet_v2(pretrained=pretrained)
        self.encoder = mobilenet.features
        
        # Decoder (Upsampling) + Attention
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv1 = DoubleDSConv(1280 + 96, 512)
        self.att1 = CBAM(512)
        
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv2 = DoubleDSConv(512 + 32, 256)
        self.att2 = CBAM(256)
        
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv3 = DoubleDSConv(256 + 24, 128)
        self.att3 = CBAM(128)
        
        self.up4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv4 = DoubleDSConv(128 + 16, 64)
        self.att4 = CBAM(64)
        
        self.up5 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv5 = DoubleDSConv(64 + 3, 32)
        self.att5 = CBAM(32)
        
        self.final_conv = nn.Conv2d(32, n_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        x_0 = self.encoder[0:2](x)   # 16, 256, 256
        x_1 = self.encoder[2:4](x_0) # 24, 128, 128
        x_2 = self.encoder[4:7](x_1) # 32, 64, 64
        x_3 = self.encoder[7:14](x_2)# 96, 32, 32
        x_4 = self.encoder[14:](x_3) # 1280, 16, 16
        
        # Decoder
        d1 = self.up1(x_4)
        diffY = x_3.size()[2] - d1.size()[2]
        diffX = x_3.size()[3] - d1.size()[3]
        d1 = F.pad(d1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        d1 = torch.cat([x_3, d1], dim=1)
        d1 = self.conv1(d1)
        d1 = self.att1(d1) # Apply Attention
        
        d2 = self.up2(d1)
        diffY = x_2.size()[2] - d2.size()[2]
        diffX = x_2.size()[3] - d2.size()[3]
        d2 = F.pad(d2, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        d2 = torch.cat([x_2, d2], dim=1)
        d2 = self.conv2(d2)
        d2 = self.att2(d2) # Apply Attention
        
        d3 = self.up3(d2)
        diffY = x_1.size()[2] - d3.size()[2]
        diffX = x_1.size()[3] - d3.size()[3]
        d3 = F.pad(d3, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        d3 = torch.cat([x_1, d3], dim=1)
        d3 = self.conv3(d3)
        d3 = self.att3(d3) # Apply Attention
        
        d4 = self.up4(d3)
        diffY = x_0.size()[2] - d4.size()[2]
        diffX = x_0.size()[3] - d4.size()[3]
        d4 = F.pad(d4, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        d4 = torch.cat([x_0, d4], dim=1)
        d4 = self.conv4(d4)
        d4 = self.att4(d4) # Apply Attention
        
        d5 = self.up5(d4)
        diffY = x.size()[2] - d5.size()[2]
        diffX = x.size()[3] - d5.size()[3]
        d5 = F.pad(d5, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        d5 = torch.cat([x, d5], dim=1)
        d5 = self.conv5(d5)
        d5 = self.att5(d5) # Apply Attention
        
        out = self.final_conv(d5)
        return out

class MobileUNetv3(nn.Module):
    """
    MobileUNet with MobileNetV3-Large backbone + CBAM Attention.
    Super lightweight and accurate.
    """
    def __init__(self, n_classes, pretrained=True):
        super(MobileUNetv3, self).__init__()
        self.n_classes = n_classes
        
        # Encoder: MobileNetV3-Large
        # We need to grab the 'features' part
        self.encoder = mobilenet_v3_large(pretrained=pretrained).features
        
        # Skip connection channels based on our inspection:
        # Layer 1: 16 channels (256x256)
        # Layer 3: 24 channels (128x128)
        # Layer 6: 40 channels (64x64)
        # Layer 12: 112 channels (32x32)
        # Layer 16: 960 channels (16x16)
        
        # Decoder (Upsampling) + Attention
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv1 = DoubleDSConv(960 + 112, 512)
        self.att1 = CBAM(512)
        
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv2 = DoubleDSConv(512 + 40, 256)
        self.att2 = CBAM(256)
        
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv3 = DoubleDSConv(256 + 24, 128)
        self.att3 = CBAM(128)
        
        self.up4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv4 = DoubleDSConv(128 + 16, 64)
        self.att4 = CBAM(64)
        
        self.up5 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv5 = DoubleDSConv(64 + 3, 32)
        self.att5 = CBAM(32)
        
        self.final_conv = nn.Conv2d(32, n_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        # Need to extract intermediate features
        # L0-1: 16ch 256x256
        x_0 = self.encoder[0:2](x)
        # L2-3: 24ch 128x128
        x_1 = self.encoder[2:4](x_0)
        # L4-6: 40ch 64x64
        x_2 = self.encoder[4:7](x_1)
        # L7-12: 112ch 32x32 (actually L7-12 goes to 112, but let's check stride)
        # L7 has stride 2? No, L7 is 32x32.
        x_3 = self.encoder[7:13](x_2) 
        # L13-16: 960ch 16x16
        x_4 = self.encoder[13:](x_3)
        
        # Decoder
        d1 = self.up1(x_4)
        diffY = x_3.size()[2] - d1.size()[2]
        diffX = x_3.size()[3] - d1.size()[3]
        d1 = F.pad(d1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        d1 = torch.cat([x_3, d1], dim=1)
        d1 = self.conv1(d1)
        d1 = self.att1(d1)
        
        d2 = self.up2(d1)
        diffY = x_2.size()[2] - d2.size()[2]
        diffX = x_2.size()[3] - d2.size()[3]
        d2 = F.pad(d2, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        d2 = torch.cat([x_2, d2], dim=1)
        d2 = self.conv2(d2)
        d2 = self.att2(d2)
        
        d3 = self.up3(d2)
        diffY = x_1.size()[2] - d3.size()[2]
        diffX = x_1.size()[3] - d3.size()[3]
        d3 = F.pad(d3, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        d3 = torch.cat([x_1, d3], dim=1)
        d3 = self.conv3(d3)
        d3 = self.att3(d3)
        
        d4 = self.up4(d3)
        diffY = x_0.size()[2] - d4.size()[2]
        diffX = x_0.size()[3] - d4.size()[3]
        d4 = F.pad(d4, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        d4 = torch.cat([x_0, d4], dim=1)
        d4 = self.conv4(d4)
        d4 = self.att4(d4)
        
        d5 = self.up5(d4)
        diffY = x.size()[2] - d5.size()[2]
        diffX = x.size()[3] - d5.size()[3]
        d5 = F.pad(d5, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        d5 = torch.cat([x, d5], dim=1)
        d5 = self.conv5(d5)
        d5 = self.att5(d5)
        
        out = self.final_conv(d5)
        return {'out': out, 'features': x_4}

class DeepLabV3Plus(nn.Module):
    """
    DeepLabV3+ with MobileNetV3-Large Backbone.
    State-of-the-art for mobile segmentation.
    """
    def __init__(self, n_classes, pretrained=True):
        super(DeepLabV3Plus, self).__init__()
        # Load pre-trained DeepLabV3 with MobileNetV3 backbone
        # This model is optimized for COCO (21 classes)
        self.model = deeplabv3_mobilenet_v3_large(pretrained=pretrained)
        
        # Replace the classifier head for our number of classes (1)
        # DeepLabV3 head structure: model.classifier[4] is the last Conv2d
        # The classifier is a DeepLabHead object
        
        # We need to inspect model.classifier structure
        # usually: 0: ASPP, 1: Conv, 2: BN, 3: ReLU, 4: Conv(num_classes)
        
        # Let's replace the last layer
        in_channels = 256 # DeepLabV3 MobileNetV3 classifier hidden dim
        self.model.classifier[4] = nn.Conv2d(in_channels, n_classes, kernel_size=1)
        
        # Aux classifier (if present)
        if self.model.aux_classifier is not None:
             self.model.aux_classifier[4] = nn.Conv2d(10, n_classes, kernel_size=1) 
             # Note: MobileNetV3 aux classifier interaction might be different, let's just disable it to be safe/fast
             self.model.aux_classifier = None

    def forward(self, x):
        # torchvision DeepLab returns a dict: {'out': tensor, 'aux': tensor}
        output = self.model(x)['out']
        return output

class DeepLabV3ResNet(nn.Module):
    """
    DeepLabV3 with standard ResNet101 Backbone.
    Heavy Teacher Model.
    """
    def __init__(self, n_classes, pretrained=True):
        super(DeepLabV3ResNet, self).__init__()
        self.model = deeplabv3_resnet101(pretrained=pretrained)
        
        in_channels = 256 # DeepLabV3 ResNet101 classifier hidden dim
        self.model.classifier[4] = nn.Conv2d(in_channels, n_classes, kernel_size=1)
        
        if self.model.aux_classifier is not None:
             self.model.aux_classifier = None

    def forward(self, x):
        output = self.model(x)['out']
        return output

def get_model(model_name='unet', n_channels=3, n_classes=1):
    if model_name == 'dscunet':
        return DSCUNet(n_channels, n_classes)
    elif model_name == 'mobileunet':
        return MobileUNet(n_classes)
    elif model_name == 'mobileunetv2':
        return MobileUNetv2(n_classes)
    elif model_name == 'mobileunetv2':
        return MobileUNetv2(n_classes)
    elif model_name == 'mobileunetv3':
        return MobileUNetv3(n_classes)
    elif model_name == 'deeplabv3':
        return DeepLabV3Plus(n_classes)
    elif model_name == 'deeplabv3_resnet':
        return DeepLabV3ResNet(n_classes)
    else:
        return None

if __name__ == "__main__":
    print("Testing DSCUNet...")
    net = DSCUNet(n_channels=3, n_classes=1)
    # print(net)
    x = torch.randn(1, 3, 512, 512)
    y = net(x)
    print(f"DSCUNet Output shape: {y.shape}")

    print("Testing MobileUNet...")
    net_mobile = MobileUNet(n_classes=1, pretrained=False) # False to avoid downloading weights in test
    y_mobile = net_mobile(x)
    print(f"MobileUNet Output shape: {y_mobile.shape}")
    
    print("Testing DeepLabV3+...")
    try:
        net_deep = DeepLabV3Plus(n_classes=1, pretrained=False)
        y_deep = net_deep(x)
        print(f"DeepLabV3+ Output shape: {y_deep.shape}")
    except Exception as e:
        print(f"DeepLabV3+ Failed: {e}")
