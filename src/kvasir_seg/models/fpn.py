import torch
import torch.nn.functional as F
from torch import nn

from .encoders import ResNetEncoder


class Conv3x3GNReLU(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, upsample: bool = False):
        super().__init__()
        self.upsample = upsample
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.block(x)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        return x


class FPNBlock(nn.Module):
    """Top-down pathway step: upsample the coarser map and add a 1x1-projected lateral skip."""

    def __init__(self, pyramid_channels: int, skip_channels: int):
        super().__init__()
        self.skip_conv = nn.Conv2d(skip_channels, pyramid_channels, kernel_size=1, bias=False)

    def forward(self, x, skip):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        skip = self.skip_conv(skip)
        if skip.shape[2:] != x.shape[2:]:
            skip = F.interpolate(skip, size=x.shape[2:], mode="bilinear", align_corners=True)
        return x + skip


class SegmentationBlock(nn.Module):
    """Brings one pyramid level to 1/4 resolution with ``n_upsamples`` conv + 2x upsample steps."""

    def __init__(self, in_channels: int, out_channels: int, n_upsamples: int = 0):
        super().__init__()
        blocks = [Conv3x3GNReLU(in_channels, out_channels, upsample=bool(n_upsamples))]
        blocks += [Conv3x3GNReLU(out_channels, out_channels, upsample=True) for _ in range(1, n_upsamples)]
        self.block = nn.Sequential(*blocks)

    def forward(self, x):
        return self.block(x)


class FPNDecoder(nn.Module):
    def __init__(
        self, encoder_channels, pyramid_channels: int = 256, segmentation_channels: int = 128, merge_policy: str = "add"
    ):
        super().__init__()
        if merge_policy not in ("add", "cat"):
            raise ValueError("merge_policy must be 'add' or 'cat'")
        self.merge_policy = merge_policy

        c1, c2, c3, c4 = encoder_channels
        self.p5 = nn.Conv2d(c4, pyramid_channels, kernel_size=1)
        self.p4 = FPNBlock(pyramid_channels, c3)
        self.p3 = FPNBlock(pyramid_channels, c2)
        self.p2 = FPNBlock(pyramid_channels, c1)
        self.seg_blocks = nn.ModuleList(
            [SegmentationBlock(pyramid_channels, segmentation_channels, n_upsamples=n) for n in (3, 2, 1, 0)]
        )
        self.out_channels = segmentation_channels * (4 if merge_policy == "cat" else 1)

    def forward(self, x1, x2, x3, x4):
        p5 = self.p5(x4)
        p4 = self.p4(p5, x3)
        p3 = self.p3(p4, x2)
        p2 = self.p2(p3, x1)
        pyramid = [block(p) for block, p in zip(self.seg_blocks, (p5, p4, p3, p2))]
        return sum(pyramid) if self.merge_policy == "add" else torch.cat(pyramid, dim=1)


class FPN(nn.Module):
    """Feature Pyramid Network (Lin et al., 2017) for semantic segmentation. Outputs raw logits."""

    def __init__(
        self,
        encoder_name: str = "resnet34",
        in_channels: int = 3,
        out_channels: int = 1,
        pyramid_channels: int = 256,
        segmentation_channels: int = 128,
        pretrained: bool = True,
    ):
        super().__init__()
        self.encoder = ResNetEncoder(encoder_name, pretrained=pretrained, in_channels=in_channels)
        self.decoder = FPNDecoder(self.encoder.out_channels, pyramid_channels, segmentation_channels)
        self.head = nn.Conv2d(self.decoder.out_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.head(self.decoder(*self.encoder(x)))
        return F.interpolate(out, size=x.shape[2:], mode="bilinear", align_corners=True)
