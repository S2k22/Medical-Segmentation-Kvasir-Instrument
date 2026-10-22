from torch import nn

from .fpn import FPN
from .unet import UNet
from .unetpp import UNetPlusPlus

MODELS = {
    "unet": lambda pretrained: UNet(),
    "unetpp": lambda pretrained: UNetPlusPlus(pretrained=pretrained),
    "fpn": lambda pretrained: FPN(pretrained=pretrained),
}


def build_model(name: str, pretrained: bool = True) -> nn.Module:
    if name not in MODELS:
        raise ValueError(f"Unknown model '{name}'. Choose from {list(MODELS)}")
    return MODELS[name](pretrained)


__all__ = ["FPN", "UNet", "UNetPlusPlus", "MODELS", "build_model"]
