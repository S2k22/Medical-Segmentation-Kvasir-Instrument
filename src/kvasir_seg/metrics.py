import torch


@torch.no_grad()
def segmentation_stats(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5, eps: float = 1e-6):
    """Per-image Dice, IoU and pixel accuracy, summed over the batch.

    Returns a dict of sums so callers can average over the whole dataset
    (avoids the bias of averaging per-batch means when the last batch is smaller).
    """
    preds = (torch.sigmoid(logits) > threshold).float().flatten(1)
    targets = targets.float().flatten(1)

    intersection = (preds * targets).sum(1)
    total = preds.sum(1) + targets.sum(1)
    dice = (2 * intersection + eps) / (total + eps)
    iou = (intersection + eps) / (total - intersection + eps)
    acc = (preds == targets).float().mean(1)

    return {"dice": dice.sum().item(), "iou": iou.sum().item(), "acc": acc.sum().item(), "n": preds.shape[0]}
