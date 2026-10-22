import torch
from torch import nn


def conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    """(Conv3x3 -> BN -> ReLU) x 2"""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    """Classic U-Net (Ronneberger et al., 2015) trained from scratch.

    4 down-sampling stages, a bottleneck and a symmetric decoder with transposed
    convolutions and skip connections. Outputs raw logits.
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 1, init_features: int = 32):
        super().__init__()
        f = init_features
        self.encoders = nn.ModuleList(
            [conv_block(in_channels, f), conv_block(f, f * 2), conv_block(f * 2, f * 4), conv_block(f * 4, f * 8)]
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck = conv_block(f * 8, f * 16)

        self.upconvs = nn.ModuleList(
            [nn.ConvTranspose2d(f * 2 ** (i + 1), f * 2**i, kernel_size=2, stride=2) for i in reversed(range(4))]
        )
        self.decoders = nn.ModuleList([conv_block(f * 2 ** (i + 1), f * 2**i) for i in reversed(range(4))])
        self.head = nn.Conv2d(f, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for encoder in self.encoders:
            x = encoder(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        for upconv, decoder, skip in zip(self.upconvs, self.decoders, reversed(skips)):
            x = decoder(torch.cat([upconv(x), skip], dim=1))

        return self.head(x)
