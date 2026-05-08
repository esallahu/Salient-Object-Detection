"""
data_loader.py
==============

Dataset, preprocessing and augmentation for Salient Object Detection.

Expected layout
---------------
The "split" layout (preferred):

    data_root/
    ├── images/{train,val,test}/*.jpg|png
    └── masks/{train,val,test}/*.png

The "flat" layout (use ``auto_split=True``):

    data_root/
    ├── images/*.jpg|png
    └── masks/*.png

Masks are single-channel, 0 = background, 255 = salient.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def _list_images(folder: Path) -> List[Path]:
    if not folder.exists():
        return []
    return sorted([p for p in folder.iterdir() if p.suffix.lower() in IMG_EXTS])


def _match_mask(image_path: Path, mask_dir: Path) -> Optional[Path]:
    """Find the mask whose stem matches the given image stem."""
    stem = image_path.stem
    for ext in (".png", ".jpg", ".jpeg", ".bmp"):
        candidate = mask_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


class SODDataset(Dataset):
    """Generic SOD dataset working on (image, mask) pairs."""

    def __init__(
        self,
        image_paths: List[Path],
        mask_paths: List[Path],
        img_size: int = 128,
        augment: bool = False,
    ) -> None:
        assert len(image_paths) == len(mask_paths), (
            f"Mismatch: {len(image_paths)} images vs {len(mask_paths)} masks"
        )
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.img_size = img_size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.image_paths)

    # --- augmentations (kept simple and dependency-free) ---------------

    def _augment(self, img: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Horizontal flip
        if random.random() < 0.5:
            img = img[:, ::-1, :].copy()
            mask = mask[:, ::-1].copy()

        # Random crop + resize back
        if random.random() < 0.5:
            h, w = img.shape[:2]
            crop = random.uniform(0.75, 1.0)
            ch, cw = int(h * crop), int(w * crop)
            y = random.randint(0, h - ch)
            x = random.randint(0, w - cw)
            img = img[y : y + ch, x : x + cw, :]
            mask = mask[y : y + ch, x : x + cw]
            img = np.array(
                Image.fromarray(img).resize((self.img_size, self.img_size), Image.BILINEAR)
            )
            mask = np.array(
                Image.fromarray(mask).resize((self.img_size, self.img_size), Image.NEAREST)
            )

        # Brightness variation (multiplicative, clipped to [0, 255])
        if random.random() < 0.5:
            factor = random.uniform(0.7, 1.3)
            img = np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)

        return img, mask

    # ------------------------------------------------------------------

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img = Image.open(self.image_paths[idx]).convert("RGB")
        mask = Image.open(self.mask_paths[idx]).convert("L")

        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        mask = mask.resize((self.img_size, self.img_size), Image.NEAREST)

        img = np.array(img, dtype=np.uint8)
        mask = np.array(mask, dtype=np.uint8)

        if self.augment:
            img, mask = self._augment(img, mask)

        # Normalise image to [0, 1] and mask to {0, 1}
        img_t = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
        mask_t = torch.from_numpy(mask).float().unsqueeze(0) / 255.0
        mask_t = (mask_t > 0.5).float()

        return img_t, mask_t


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------


def _gather_split(images_dir: Path, masks_dir: Path) -> Tuple[List[Path], List[Path]]:
    images = _list_images(images_dir)
    matched_imgs, matched_masks = [], []
    for ip in images:
        mp = _match_mask(ip, masks_dir)
        if mp is not None:
            matched_imgs.append(ip)
            matched_masks.append(mp)
    return matched_imgs, matched_masks


def _auto_split(
    images: List[Path],
    masks: List[Path],
    ratios: Tuple[float, float, float] = (0.70, 0.15, 0.15),
    seed: int = 42,
) -> dict:
    rng = random.Random(seed)
    pairs = list(zip(images, masks))
    rng.shuffle(pairs)
    n = len(pairs)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    train = pairs[:n_train]
    val = pairs[n_train : n_train + n_val]
    test = pairs[n_train + n_val :]

    def unzip(pairs_):
        if not pairs_:
            return [], []
        a, b = zip(*pairs_)
        return list(a), list(b)

    tr_i, tr_m = unzip(train)
    va_i, va_m = unzip(val)
    te_i, te_m = unzip(test)
    return {
        "train": (tr_i, tr_m),
        "val": (va_i, va_m),
        "test": (te_i, te_m),
    }


def build_datasets(
    data_root: str | Path,
    img_size: int = 128,
    augment_train: bool = True,
    auto_split: bool = False,
    seed: int = 42,
) -> Tuple[SODDataset, SODDataset, SODDataset]:
    """Build (train, val, test) datasets from ``data_root``."""

    root = Path(data_root)
    images_root = root / "images"
    masks_root = root / "masks"
    if not images_root.exists() or not masks_root.exists():
        raise FileNotFoundError(
            f"Expected '{images_root}' and '{masks_root}'. "
            f"Got contents: {list(root.iterdir()) if root.exists() else 'missing'}"
        )

    has_splits = all((images_root / s).exists() for s in ("train", "val", "test"))

    if has_splits and not auto_split:
        tr_i, tr_m = _gather_split(images_root / "train", masks_root / "train")
        va_i, va_m = _gather_split(images_root / "val", masks_root / "val")
        te_i, te_m = _gather_split(images_root / "test", masks_root / "test")
    else:
        all_imgs, all_masks = _gather_split(images_root, masks_root)
        if not all_imgs:
            raise RuntimeError(
                f"No matching image/mask pairs found under {root}. "
                "Check that masks share the same stem as images."
            )
        splits = _auto_split(all_imgs, all_masks, seed=seed)
        tr_i, tr_m = splits["train"]
        va_i, va_m = splits["val"]
        te_i, te_m = splits["test"]

    print(
        f"[data_loader] train={len(tr_i)}  val={len(va_i)}  test={len(te_i)}  "
        f"img_size={img_size}  augment_train={augment_train}"
    )

    train_ds = SODDataset(tr_i, tr_m, img_size=img_size, augment=augment_train)
    val_ds = SODDataset(va_i, va_m, img_size=img_size, augment=False)
    test_ds = SODDataset(te_i, te_m, img_size=img_size, augment=False)
    return train_ds, val_ds, test_ds


def build_dataloaders(
    data_root: str | Path,
    img_size: int = 128,
    batch_size: int = 16,
    num_workers: int = 0,
    augment_train: bool = True,
    auto_split: bool = False,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_ds, val_ds, test_ds = build_datasets(
        data_root,
        img_size=img_size,
        augment_train=augment_train,
        auto_split=auto_split,
        seed=seed,
    )

    def make_loader(ds: SODDataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
        )

    return make_loader(train_ds, True), make_loader(val_ds, False), make_loader(test_ds, False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Quick sanity check of the data loader.")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--img_size", type=int, default=128)
    parser.add_argument("--auto_split", action="store_true")
    args = parser.parse_args()

    tr, va, te = build_dataloaders(
        args.data_root,
        img_size=args.img_size,
        batch_size=4,
        auto_split=args.auto_split,
    )
    x, y = next(iter(tr))
    print("batch image shape:", x.shape, "  range:", float(x.min()), float(x.max()))
    print("batch mask shape :", y.shape, "  unique:", torch.unique(y).tolist())
