"""
models.py

One U-Net backbone, two heads:
    - EnhancementUNet: 1 -> 1 channel, sigmoid output (image restoration)
    - SegmentationUNet: 1 -> N channel, raw logits (multi-class segmentation,
      use with CrossEntropyLoss which applies softmax internally)

Same architecture family used for both Brain and Spine - only the head
and loss function differ.
"""

import torch
import torch.nn as nn


def conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class _UNetBackbone(nn.Module):
    def __init__(self, in_channels: int, base_filters: int = 32):
        super().__init__()
        f = base_filters
        self.enc1 = conv_block(in_channels, f)
        self.enc2 = conv_block(f, f * 2)
        self.enc3 = conv_block(f * 2, f * 4)
        self.enc4 = conv_block(f * 4, f * 8)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = conv_block(f * 8, f * 16)
        self.up4 = nn.ConvTranspose2d(f * 16, f * 8, 2, stride=2)
        self.dec4 = conv_block(f * 16, f * 8)
        self.up3 = nn.ConvTranspose2d(f * 8, f * 4, 2, stride=2)
        self.dec3 = conv_block(f * 8, f * 4)
        self.up2 = nn.ConvTranspose2d(f * 4, f * 2, 2, stride=2)
        self.dec2 = conv_block(f * 4, f * 2)
        self.up1 = nn.ConvTranspose2d(f * 2, f, 2, stride=2)
        self.dec1 = conv_block(f * 2, f)
        self.out_filters = f

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return d1


class EnhancementUNet(nn.Module):
    """1-channel MRI slice in -> 1-channel enhanced slice out, values in [0,1]."""

    def __init__(self, base_filters: int = 32):
        super().__init__()
        self.backbone = _UNetBackbone(in_channels=1, base_filters=base_filters)
        self.out_conv = nn.Conv2d(self.backbone.out_filters, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.out_conv(self.backbone(x)))


class SegmentationUNet(nn.Module):
    """N-channel MRI slice in (e.g. 4 stacked modalities: T1/T1ce/T2/FLAIR)
    -> num_classes raw-logit channels out.
    Use nn.CrossEntropyLoss(logits, target) directly - do not apply softmax here."""

    def __init__(self, num_classes: int, in_channels: int = 4, base_filters: int = 32):
        super().__init__()
        self.backbone = _UNetBackbone(in_channels=in_channels, base_filters=base_filters)
        self.out_conv = nn.Conv2d(self.backbone.out_filters, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out_conv(self.backbone(x))
