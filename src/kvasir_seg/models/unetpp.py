import torch
import torch.nn.functional as F
from torch import nn

from .encoders import ResNetEncoder


def conv_bn_relu(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


def upsample_to(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    return F.interpolate(x, size=ref.shape[2:], mode="bilinear", align_corners=True)


class UNetPlusPlusDecoder(nn.Module):
    """Lightweight UNet++-style decoder with nested, densely connected skip pathways.

    Takes encoder features ordered deepest-first ``(x4, x3, x2, x1)`` and progressively
    re-aggregates them at increasing resolution (node naming follows Zhou et al., 2018).
    """

    def __init__(self, encoder_channels=(512, 256, 128, 64), decoder_channels=(256, 128, 64)):
        super().__init__()
        e, d = encoder_channels, decoder_channels
        self.conv0_0 = conv_bn_relu(e[0], d[0])
        self.conv1_0 = conv_bn_relu(e[1], d[1])
        self.conv1_1 = conv_bn_relu(d[0] + d[1], d[1])
        self.conv2_0 = conv_bn_relu(e[2], d[2])
        self.conv2_1 = conv_bn_relu(d[1] + d[2], d[2])
        self.conv2_2 = conv_bn_relu(d[0] + d[1] + d[2], d[2])
        self.out_channels = d[2]

    def forward(self, x4, x3, x2, x1):
        x0_0 = self.conv0_0(x4)
        x1_0 = self.conv1_0(upsample_to(x3, x2))
        x1_1 = self.conv1_1(torch.cat([x1_0, upsample_to(x0_0, x1_0)], dim=1))
        x2_0 = self.conv2_0(upsample_to(x2, x1_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, upsample_to(x1_1, x2_0)], dim=1))
        x2_2 = self.conv2_2(torch.cat([x2_1, upsample_to(x1_1, x2_1), upsample_to(x0_0, x2_1)], dim=1))
        return x2_2


class UNetPlusPlus(nn.Module):
    """UNet++ with an ImageNet-pretrained ResNet34 encoder. Outputs raw logits."""

    def __init__(self, in_channels: int = 3, out_channels: int = 1, pretrained: bool = True):
        super().__init__()
        self.encoder = ResNetEncoder("resnet34", pretrained=pretrained, in_channels=in_channels)
        self.decoder = UNetPlusPlusDecoder(encoder_channels=self.encoder.out_channels[::-1])
        self.head = nn.Conv2d(self.decoder.out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2, x3, x4 = self.encoder(x)
        out = self.head(self.decoder(x4, x3, x2, x1))
        return F.interpolate(out, size=x.shape[2:], mode="bilinear", align_corners=True)
