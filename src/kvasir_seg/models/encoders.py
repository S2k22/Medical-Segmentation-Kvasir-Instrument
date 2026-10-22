import torch
import torchvision
from torch import nn

ENCODER_CHANNELS = {
    "resnet34": [64, 128, 256, 512],
    "resnet50": [256, 512, 1024, 2048],
}


class ResNetEncoder(nn.Module):
    """ResNet backbone that returns the feature maps of layer1..layer4 (strides 4, 8, 16, 32)."""

    def __init__(self, name: str = "resnet34", pretrained: bool = True, in_channels: int = 3):
        super().__init__()
        if name not in ENCODER_CHANNELS:
            raise ValueError(f"Unsupported encoder '{name}'. Choose from {list(ENCODER_CHANNELS)}")

        weights = "DEFAULT" if pretrained else None
        resnet = getattr(torchvision.models, name)(weights=weights)
        if in_channels != 3:
            resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layers = nn.ModuleList([resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4])
        self.out_channels = ENCODER_CHANNELS[name]

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = self.stem(x)
        features = []
        for layer in self.layers:
            x = layer(x)
            features.append(x)
        return features  # shallow -> deep
