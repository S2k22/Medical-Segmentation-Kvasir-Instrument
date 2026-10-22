"""Dataset download, train/val/test splits, augmentations and the PyTorch Dataset."""

import os
import random
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import requests
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
from tqdm import tqdm

DATASET_URL = "https://datasets.simula.no/downloads/kvasir-instrument.zip"


# --------------------------------------------------------------------------- #
# Download / extraction
# --------------------------------------------------------------------------- #
def _download(url: str, save_path: Path) -> None:
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    total = int(response.headers.get("content-length", 0))
    with (
        open(save_path, "wb") as f,
        tqdm(desc=f"Downloading {save_path.name}", total=total, unit="iB", unit_scale=True, unit_divisor=1024) as bar,
    ):
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))


def prepare_dataset(data_root: str | os.PathLike) -> Path:
    """Download and extract Kvasir-Instrument into ``data_root``. Returns the dataset directory."""
    data_root = Path(data_root)
    data_root.mkdir(parents=True, exist_ok=True)
    dataset_dir = data_root / "kvasir-instrument"
    archive = data_root / "kvasir-instrument.zip"

    if not dataset_dir.exists():
        if not archive.exists():
            _download(DATASET_URL, archive)
        try:
            with zipfile.ZipFile(archive) as zf:
                zf.extractall(data_root)
        except zipfile.BadZipFile:
            archive.unlink()
            raise RuntimeError("Downloaded archive is corrupted - deleted it, please re-run.") from None

    # The zip contains images.tar.gz / masks.tar.gz that must be extracted once more
    for name in ("images", "masks"):
        tar_path = dataset_dir / f"{name}.tar.gz"
        if tar_path.exists() and not (dataset_dir / name / name).exists():
            with tarfile.open(tar_path, "r:gz") as tar:
                if hasattr(tarfile, "data_filter"):  # safe extraction on Python >= 3.10.12
                    tar.extractall(path=dataset_dir / name, filter="data")
                else:
                    tar.extractall(path=dataset_dir / name)

    return dataset_dir


# --------------------------------------------------------------------------- #
# Splits
# --------------------------------------------------------------------------- #
@dataclass
class Split:
    images: list[str]
    masks: list[str]

    def __len__(self) -> int:
        return len(self.images)


def get_splits(dataset_dir: str | os.PathLike, val_fraction: float = 0.1, seed: int = 42) -> dict[str, Split]:
    """Build train/val/test splits.

    The official ``test.txt`` split is kept untouched as a held-out test set, and a
    validation set is carved out of ``train.txt`` for early stopping / LR scheduling,
    so the test set never influences training decisions.
    """
    dataset_dir = Path(dataset_dir)
    images_dir = dataset_dir / "images" / "images"
    masks_dir = dataset_dir / "masks" / "masks"

    def read_ids(name: str) -> list[str]:
        return (dataset_dir / name).read_text().split()

    train_ids, test_ids = read_ids("train.txt"), read_ids("test.txt")
    rng = random.Random(seed)
    rng.shuffle(train_ids)
    n_val = round(len(train_ids) * val_fraction)
    val_ids, train_ids = train_ids[:n_val], train_ids[n_val:]

    def to_split(ids: list[str]) -> Split:
        return Split(
            images=[str(images_dir / f"{i}.jpg") for i in ids],
            masks=[str(masks_dir / f"{i}.png") for i in ids],
        )

    return {"train": to_split(train_ids), "val": to_split(val_ids), "test": to_split(test_ids)}


# --------------------------------------------------------------------------- #
# Transforms
# --------------------------------------------------------------------------- #
def get_transforms(train: bool, augment: bool = True, image_size: int = 256) -> A.Compose:
    """Albumentations pipeline. Normalisation uses ImageNet statistics (matches the pretrained encoders)."""
    ops = []
    if train and augment:
        ops += [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.GaussNoise(p=0.2),
        ]
    ops += [A.Resize(image_size, image_size), A.Normalize(), ToTensorV2()]
    return A.Compose(ops)


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class KvasirInstrumentDataset(Dataset):
    """Returns ``(image [3,H,W] float, mask [1,H,W] float in {0,1})``."""

    def __init__(self, split: Split, transform: A.Compose):
        self.split = split
        self.transform = transform

    def __len__(self) -> int:
        return len(self.split)

    def __getitem__(self, idx: int):
        image_path, mask_path = self.split.images[idx], self.split.masks[idx]

        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Mask not found: {mask_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = (mask > 0).astype(np.float32)

        out = self.transform(image=image, mask=mask)
        return out["image"], out["mask"].unsqueeze(0)
