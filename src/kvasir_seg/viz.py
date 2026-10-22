from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


def plot_history(history: list[dict], save_path: Path, title: str = "") -> None:
    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))
    for ax, metric in zip(axes, ("loss", "acc", "dice", "iou")):
        ax.plot(epochs, [h[f"train_{metric}"] for h in history], label="train")
        ax.plot(epochs, [h[f"val_{metric}"] for h in history], label="val")
        ax.set_title(metric.upper() if metric != "loss" else "BCE loss")
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


@torch.no_grad()
def plot_predictions(model, loader, device, save_path: Path, num_samples: int = 4) -> None:
    model.eval()
    images, masks = next(iter(loader))
    preds = (torch.sigmoid(model(images.to(device))) > 0.5).float().cpu()

    n = min(num_samples, len(images))
    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n), squeeze=False)
    for i in range(n):
        image = images[i].permute(1, 2, 0).numpy() * IMAGENET_STD + IMAGENET_MEAN
        panels = (np.clip(image, 0, 1), masks[i, 0], preds[i, 0])
        for ax, panel, title in zip(axes[i], panels, ("Image", "Ground truth", "Prediction")):
            ax.imshow(panel, cmap=None if panel is panels[0] else "gray")
            ax.set_title(title if i == 0 else "")
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
