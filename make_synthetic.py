"""
make_synthetic.py
=================

Generate a tiny synthetic Salient Object Detection dataset, just to verify the
training / evaluation pipeline works without having to download any of the
public benchmarks (DUTS / ECSSD / MSRA10K / SALICON).

Each sample is:
* an image with a random colored circle / rectangle on a noisy textured
  background -> ``images/<id>.png``
* the corresponding binary mask (255 inside the shape, 0 outside) ->
  ``masks/<id>.png``

This is **not** meant to produce a useful saliency model — only to confirm that
the data loader, model, training loop, evaluation and demo all work end-to-end.

Usage
-----
    python make_synthetic.py --out data/synthetic --num 60
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def _noisy_background(size: int, seed: int) -> Image.Image:
    rng = np.random.default_rng(seed)
    base = rng.integers(40, 200, size=3)
    noise = rng.integers(-25, 25, size=(size, size, 3))
    img = np.clip(base + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(img, mode="RGB")


def _draw_shape(img: Image.Image, mask: Image.Image, rng: random.Random) -> None:
    w, h = img.size
    shape_kind = rng.choice(["circle", "rect", "ellipse"])
    color = (rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255))

    margin = int(min(w, h) * 0.1)
    cx = rng.randint(margin, w - margin)
    cy = rng.randint(margin, h - margin)
    rmax = int(min(cx, cy, w - cx, h - cy) * rng.uniform(0.5, 0.95))
    r = max(rmax, 8)

    img_draw = ImageDraw.Draw(img)
    mask_draw = ImageDraw.Draw(mask)

    if shape_kind in ("circle", "ellipse"):
        rx = r if shape_kind == "circle" else int(r * rng.uniform(0.6, 1.4))
        ry = r if shape_kind == "circle" else int(r * rng.uniform(0.6, 1.4))
        bbox = [cx - rx, cy - ry, cx + rx, cy + ry]
        img_draw.ellipse(bbox, fill=color, outline=None)
        mask_draw.ellipse(bbox, fill=255)
    else:
        bbox = [cx - r, cy - r, cx + r, cy + r]
        img_draw.rectangle(bbox, fill=color, outline=None)
        mask_draw.rectangle(bbox, fill=255)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, required=True, help="Output dataset folder.")
    parser.add_argument("--num", type=int, default=60, help="Number of samples.")
    parser.add_argument("--size", type=int, default=128, help="Image size (square).")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    out = Path(args.out)
    img_dir = out / "images"
    mask_dir = out / "masks"
    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    for i in range(args.num):
        img = _noisy_background(args.size, seed=args.seed + i)
        mask = Image.new("L", (args.size, args.size), 0)
        _draw_shape(img, mask, rng)
        name = f"sample_{i:04d}.png"
        img.save(img_dir / name)
        mask.save(mask_dir / name)

    print(f"[synthetic] wrote {args.num} samples -> {out}")
    print("            now train with:")
    print(f"            python train.py --data_root {out} --auto_split --epochs 5")


if __name__ == "__main__":
    main()
