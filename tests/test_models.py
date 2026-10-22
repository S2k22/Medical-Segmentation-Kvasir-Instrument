import pytest
import torch

from kvasir_seg.metrics import segmentation_stats
from kvasir_seg.models import MODELS, build_model


@pytest.mark.parametrize("name", list(MODELS))
@pytest.mark.parametrize("size", [128, 256])
def test_output_shape_matches_input(name, size):
    model = build_model(name, pretrained=False).eval()
    x = torch.randn(2, 3, size, size)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (2, 1, size, size)


@pytest.mark.parametrize("name", list(MODELS))
def test_backward_pass(name):
    model = build_model(name, pretrained=False)
    loss = torch.nn.BCEWithLogitsLoss()(model(torch.randn(2, 3, 128, 128)), torch.rand(2, 1, 128, 128).round())
    loss.backward()
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad)


def test_metrics_perfect_and_disjoint():
    target = torch.zeros(1, 1, 4, 4)
    target[..., :2, :] = 1
    perfect = segmentation_stats(torch.where(target > 0, 10.0, -10.0), target)
    assert perfect["dice"] == pytest.approx(1.0)
    assert perfect["iou"] == pytest.approx(1.0)
    assert perfect["acc"] == pytest.approx(1.0)

    disjoint = segmentation_stats(torch.where(target > 0, -10.0, 10.0), target)
    assert disjoint["dice"] == pytest.approx(0.0, abs=1e-5)
    assert disjoint["acc"] == pytest.approx(0.0)
