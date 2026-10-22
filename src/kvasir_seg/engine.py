"""Training / evaluation loops with LR scheduling, early stopping and best-checkpoint saving."""

from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .metrics import segmentation_stats


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device,
    optimizer: torch.optim.Optimizer | None = None,
    desc: str = "",
) -> dict[str, float]:
    """One pass over ``loader``. Trains if an optimizer is given, otherwise evaluates."""
    training = optimizer is not None
    model.train(training)

    totals = {"loss": 0.0, "dice": 0.0, "iou": 0.0, "acc": 0.0, "n": 0}
    with torch.set_grad_enabled(training):
        for images, masks in tqdm(loader, desc=desc, unit="batch", leave=False):
            images, masks = images.to(device), masks.to(device)
            logits = model(images)
            loss = criterion(logits, masks)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            stats = segmentation_stats(logits.detach(), masks)
            totals["loss"] += loss.item() * stats["n"]
            for k in ("dice", "iou", "acc", "n"):
                totals[k] += stats[k]

    n = totals.pop("n")
    return {k: v / n for k, v in totals.items()}


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer,
    scheduler,
    criterion: nn.Module,
    device,
    num_epochs: int,
    patience: int,
    checkpoint_path: Path,
) -> list[dict]:
    """Train with ReduceLROnPlateau + early stopping on validation loss.

    The checkpoint with the lowest validation loss is written to ``checkpoint_path``.
    Returns the per-epoch history.
    """
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    history, best_val_loss, epochs_without_improvement = [], float("inf"), 0
    for epoch in range(1, num_epochs + 1):
        train = run_epoch(model, train_loader, criterion, device, optimizer, desc=f"train {epoch}/{num_epochs}")
        val = run_epoch(model, val_loader, criterion, device, desc=f"val   {epoch}/{num_epochs}")
        scheduler.step(val["loss"])
        lr = optimizer.param_groups[0]["lr"]

        history.append(
            {
                "epoch": epoch,
                "lr": lr,
                **{f"train_{k}": v for k, v in train.items()},
                **{f"val_{k}": v for k, v in val.items()},
            }
        )
        print(
            f"Epoch {epoch:3d} | lr {lr:.2e} | "
            f"train loss {train['loss']:.4f} dice {train['dice']:.4f} | "
            f"val loss {val['loss']:.4f} dice {val['dice']:.4f} iou {val['iou']:.4f}"
        )

        if val["loss"] < best_val_loss:
            best_val_loss, epochs_without_improvement = val["loss"], 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch} (no val-loss improvement for {patience} epochs)")
                break

    return history
