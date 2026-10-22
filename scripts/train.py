"""Train and evaluate a segmentation model on Kvasir-Instrument.

Example:
    python scripts/train.py --model unetpp --lr 1e-3 --epochs 60
    python scripts/train.py --model all --no-augment
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from kvasir_seg.data import KvasirInstrumentDataset, get_splits, get_transforms, prepare_dataset
from kvasir_seg.engine import fit, run_epoch
from kvasir_seg.models import MODELS, build_model
from kvasir_seg.utils import get_device, seed_everything
from kvasir_seg.viz import plot_history, plot_predictions


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=[*MODELS, "all"], default="all")
    p.add_argument("--data-root", default="data")
    p.add_argument("--output-dir", default="runs")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=5, help="early-stopping patience (epochs)")
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", default="auto", help="auto | cpu | cuda | mps | directml")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def run(model_name: str, args, loaders: dict, device) -> dict:
    run_name = f"{model_name}_lr{args.lr:g}_{'noaug' if args.no_augment else 'aug'}"
    out_dir = Path(args.output_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== {run_name} on {device} ===")

    seed_everything(args.seed)
    model = build_model(model_name, pretrained=not args.no_pretrained).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    criterion = nn.BCEWithLogitsLoss()

    checkpoint = out_dir / "best_model.pth"
    history = fit(
        model,
        loaders["train"],
        loaders["val"],
        optimizer,
        scheduler,
        criterion,
        device,
        num_epochs=args.epochs,
        patience=args.patience,
        checkpoint_path=checkpoint,
    )

    # Evaluate the best checkpoint on the untouched test split
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    test = run_epoch(model, loaders["test"], criterion, device, desc="test")
    best = min(history, key=lambda h: h["val_loss"])

    plot_history(history, out_dir / "curves.png", title=run_name)
    plot_predictions(model, loaders["test"], device, out_dir / "predictions.png")
    pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)

    summary = {
        "run": run_name,
        "best_epoch": best["epoch"],
        "val_loss": best["val_loss"],
        "val_dice": best["val_dice"],
        "val_iou": best["val_iou"],
        **{f"test_{k}": v for k, v in test.items()},
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Test  dice {test['dice']:.4f}  iou {test['iou']:.4f}  acc {test['acc']:.4f}")
    return summary


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = get_device(args.device)

    splits = get_splits(prepare_dataset(args.data_root), val_fraction=args.val_fraction, seed=args.seed)
    print({k: len(v) for k, v in splits.items()})

    datasets = {
        name: KvasirInstrumentDataset(
            split, get_transforms(train=name == "train", augment=not args.no_augment, image_size=args.image_size)
        )
        for name, split in splits.items()
    }
    loaders = {
        name: DataLoader(ds, batch_size=args.batch_size, shuffle=name == "train", num_workers=args.num_workers)
        for name, ds in datasets.items()
    }

    model_names = list(MODELS) if args.model == "all" else [args.model]
    results = pd.DataFrame([run(name, args, loaders, device) for name in model_names])
    print("\n", results.to_string(index=False, float_format="%.4f"))
    results.to_csv(
        Path(args.output_dir) / "results.csv",
        mode="a",
        index=False,
        header=not (Path(args.output_dir) / "results.csv").exists(),
    )


if __name__ == "__main__":
    main()
