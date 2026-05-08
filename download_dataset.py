"""
download_dataset.py
===================

Download and prepare a public Salient Object Detection dataset so that the
folder layout matches what ``data_loader.py`` expects.

Supported datasets
------------------

* ``ecssd``    -- Extended Complex Scene Saliency Dataset (1000 images).
                  No official train/val/test split. Files are placed flat
                  under ``<out>/images/`` and ``<out>/masks/`` so you can
                  use ``--auto_split`` in ``train.py``.

* ``duts``     -- DUTS dataset (10553 train + 5019 test). The official
                  splits are preserved:
                      <out>/images/train, <out>/images/test
                      <out>/masks/train,  <out>/masks/test
                  A validation split is then carved out of train (default
                  15% of train -> val).

* ``duts-tr``  -- DUTS-TR only (training portion). Placed flat.
* ``duts-te``  -- DUTS-TE only (test portion).      Placed flat.

The script is idempotent: re-running it skips downloads and extractions
that are already done.

Usage
-----
    python download_dataset.py --name ecssd --out data/ecssd
    python download_dataset.py --name duts  --out data/duts
"""

from __future__ import annotations

import argparse
import os
import random
import shutil
import ssl
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm


IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


# ---------------------------------------------------------------------------
# Dataset definitions
# ---------------------------------------------------------------------------


DATASETS: Dict[str, Dict] = {
    "ecssd": {
        "urls": {
            "images.zip": "http://www.cse.cuhk.edu.hk/leojia/projects/hsaliency/data/ECSSD/images.zip",
            "masks.zip": "http://www.cse.cuhk.edu.hk/leojia/projects/hsaliency/data/ECSSD/ground_truth_mask.zip",
        },
        "layout": "flat",
    },
    "duts-tr": {
        "urls": {
            "DUTS-TR.zip": "https://saliencydetection.net/duts/download/DUTS-TR.zip",
        },
        "layout": "flat",
    },
    "duts-te": {
        "urls": {
            "DUTS-TE.zip": "https://saliencydetection.net/duts/download/DUTS-TE.zip",
        },
        "layout": "flat",
    },
    "duts": {
        "urls": {
            "DUTS-TR.zip": "https://saliencydetection.net/duts/download/DUTS-TR.zip",
            "DUTS-TE.zip": "https://saliencydetection.net/duts/download/DUTS-TE.zip",
        },
        "layout": "official_splits",
    },
}


# ---------------------------------------------------------------------------
# Download with progress bar
# ---------------------------------------------------------------------------


def _make_ssl_context(insecure: bool) -> Optional[ssl.SSLContext]:
    if not insecure:
        return None
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def _http_get_total_size(url: str, ssl_ctx: Optional[ssl.SSLContext]) -> Optional[int]:
    """HEAD request (fall back to GET) to learn the total file size."""
    req = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(req, context=ssl_ctx, timeout=30) as resp:
            length = resp.headers.get("Content-Length")
            return int(length) if length else None
    except Exception:
        return None


def _download_one_attempt(
    url: str,
    tmp: Path,
    total: Optional[int],
    ssl_ctx: Optional[ssl.SSLContext],
    chunk_size: int = 64 * 1024,
) -> bool:
    """One download attempt with HTTP Range resume. Returns True on full completion."""
    start_byte = tmp.stat().st_size if tmp.exists() else 0
    headers = {"User-Agent": "ml-sod-downloader/1.0"}
    if start_byte > 0:
        headers["Range"] = f"bytes={start_byte}-"

    req = urllib.request.Request(url, headers=headers)
    try:
        resp = urllib.request.urlopen(req, context=ssl_ctx, timeout=60)
    except urllib.error.HTTPError as e:
        # If server doesn't support Range and we asked for one, restart from 0.
        if start_byte > 0 and e.code in (416,):
            tmp.unlink(missing_ok=True)
            return _download_one_attempt(url, tmp, total, ssl_ctx, chunk_size)
        raise

    # Content-Range header tells us if the resume was honored.
    content_range = resp.headers.get("Content-Range")
    content_length = resp.headers.get("Content-Length")
    if total is None and content_length:
        try:
            total = int(content_length) + (start_byte if content_range else 0)
        except ValueError:
            total = None

    mode = "ab" if start_byte > 0 and content_range else "wb"
    if mode == "wb":
        start_byte = 0

    pbar = tqdm(
        total=total,
        initial=start_byte,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        miniters=1,
        desc=tmp.name,
    )
    try:
        with open(tmp, mode) as f, resp:
            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                pbar.update(len(chunk))
    finally:
        pbar.close()

    return total is not None and tmp.stat().st_size >= total


def _download(
    url: str,
    dest: Path,
    insecure: bool = False,
    max_attempts: int = 8,
    backoff_seconds: float = 3.0,
) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[download] already present, skipping: {dest.name}")
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    ssl_ctx = _make_ssl_context(insecure)

    print(f"[download] {url}")
    total = _http_get_total_size(url, ssl_ctx)
    if total:
        print(f"[download] total size: {total / (1024 * 1024):.1f} MiB")

    last_err: Optional[Exception] = None
    for attempt in range(1, max_attempts + 1):
        try:
            done = _download_one_attempt(url, tmp, total, ssl_ctx)
            current = tmp.stat().st_size if tmp.exists() else 0
            if done or (total is None and current > 0):
                tmp.rename(dest)
                print(f"[download] saved -> {dest}  ({current} bytes)")
                return
            print(
                f"[download] partial (got {current}/{total or '?'} bytes), "
                f"will resume (attempt {attempt}/{max_attempts})..."
            )
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError) as e:
            last_err = e
            current = tmp.stat().st_size if tmp.exists() else 0
            print(
                f"[download] error on attempt {attempt}/{max_attempts}: {e}  "
                f"(have {current} bytes, will resume)"
            )
        time.sleep(backoff_seconds * attempt)

    raise RuntimeError(
        f"Could not finish downloading {url} after {max_attempts} attempts. "
        f"Last error: {last_err}"
    )


def _extract(zip_path: Path, extract_to: Path) -> Path:
    extract_to.mkdir(parents=True, exist_ok=True)
    marker = extract_to / f".extracted_{zip_path.name}"
    if marker.exists():
        print(f"[extract ] already extracted: {zip_path.name}")
        return extract_to
    print(f"[extract ] {zip_path.name} -> {extract_to}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.infolist()
        for m in tqdm(members, desc=f"unzip {zip_path.name}"):
            zf.extract(m, extract_to)
    marker.touch()
    return extract_to


# ---------------------------------------------------------------------------
# Helpers to find images / masks inside extracted folders
# ---------------------------------------------------------------------------


def _walk_images(root: Path) -> List[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS)


def _looks_like_mask_dir(name: str) -> bool:
    n = name.lower()
    return any(tag in n for tag in ("mask", "ground_truth", "groundtruth", "-gt", "_gt", "/gt"))


def _looks_like_image_dir(name: str) -> bool:
    n = name.lower()
    return ("image" in n or "imgs" in n or "/img" in n) and not _looks_like_mask_dir(n)


def _classify_files(files: List[Path]) -> Tuple[List[Path], List[Path]]:
    """Split a flat list of files into (images, masks) using their parent dir name."""
    imgs, masks = [], []
    for f in files:
        parent = f.parent.as_posix().lower()
        if _looks_like_mask_dir(parent):
            masks.append(f)
        elif _looks_like_image_dir(parent):
            imgs.append(f)
        else:
            # Fallback: PNGs in folders that don't say "image" but the file
            # is part of a mask-style folder are caught above; otherwise
            # treat .png in ambiguous folders as masks if there is also a
            # paired .jpg in the same dataset, else as images.
            (masks if f.suffix.lower() == ".png" else imgs).append(f)
    return imgs, masks


def _copy_pairs_flat(
    images: List[Path], masks: List[Path], img_out: Path, mask_out: Path
) -> int:
    """Copy (image, mask) pairs (matched by stem) into flat output folders."""
    img_out.mkdir(parents=True, exist_ok=True)
    mask_out.mkdir(parents=True, exist_ok=True)
    mask_by_stem = {m.stem: m for m in masks}
    n = 0
    for img in tqdm(images, desc="organising"):
        m = mask_by_stem.get(img.stem)
        if m is None:
            continue
        shutil.copy2(img, img_out / img.name)
        shutil.copy2(m, mask_out / m.name)
        n += 1
    return n


# ---------------------------------------------------------------------------
# Per-dataset preparation
# ---------------------------------------------------------------------------


def _prepare_flat(out: Path, downloads: Path, zip_paths: List[Path]) -> int:
    extract_dir = downloads / "extracted"
    for zp in zip_paths:
        _extract(zp, extract_dir)
    files = _walk_images(extract_dir)
    images, masks = _classify_files(files)
    return _copy_pairs_flat(images, masks, out / "images", out / "masks")


def _prepare_duts_official_splits(
    out: Path, downloads: Path, val_ratio: float, seed: int
) -> Dict[str, int]:
    """Place DUTS-TR under train/val and DUTS-TE under test."""

    extract_dir = downloads / "extracted"

    tr_zip = downloads / "DUTS-TR.zip"
    te_zip = downloads / "DUTS-TE.zip"
    tr_dir = extract_dir / "tr"
    te_dir = extract_dir / "te"

    _extract(tr_zip, tr_dir)
    _extract(te_zip, te_dir)

    tr_files = _walk_images(tr_dir)
    te_files = _walk_images(te_dir)
    tr_imgs, tr_masks = _classify_files(tr_files)
    te_imgs, te_masks = _classify_files(te_files)

    counts = {}

    # Train + val (carve val out of TR)
    rng = random.Random(seed)
    pairs_tr = []
    masks_by_stem = {m.stem: m for m in tr_masks}
    for img in tr_imgs:
        m = masks_by_stem.get(img.stem)
        if m is not None:
            pairs_tr.append((img, m))
    rng.shuffle(pairs_tr)
    n_val = int(round(len(pairs_tr) * val_ratio))
    val_pairs = pairs_tr[:n_val]
    train_pairs = pairs_tr[n_val:]

    def write_split(split_name: str, pairs):
        img_dir = out / "images" / split_name
        mask_dir = out / "masks" / split_name
        img_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)
        for img, m in tqdm(pairs, desc=f"organising {split_name}"):
            shutil.copy2(img, img_dir / img.name)
            shutil.copy2(m, mask_dir / m.name)
        return len(pairs)

    counts["train"] = write_split("train", train_pairs)
    counts["val"] = write_split("val", val_pairs)

    # Test from DUTS-TE
    masks_by_stem_te = {m.stem: m for m in te_masks}
    test_pairs = [(i, masks_by_stem_te[i.stem]) for i in te_imgs if i.stem in masks_by_stem_te]
    counts["test"] = write_split("test", test_pairs)

    return counts


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", required=True, choices=sorted(DATASETS.keys()))
    parser.add_argument("--out", required=True, help="Output dataset folder.")
    parser.add_argument("--downloads", default="", help="Where to cache zip files (default: <out>/_downloads).")
    parser.add_argument("--insecure", action="store_true",
                        help="Disable TLS certificate verification (sometimes needed for academic servers).")
    parser.add_argument("--val_ratio", type=float, default=0.15,
                        help="DUTS only: fraction of DUTS-TR moved into the val split.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out = Path(args.out)
    downloads = Path(args.downloads) if args.downloads else (out / "_downloads")
    downloads.mkdir(parents=True, exist_ok=True)

    spec = DATASETS[args.name]
    print(f"\n=== Preparing dataset: {args.name} ===")
    print(f"  out:       {out}")
    print(f"  downloads: {downloads}")
    print(f"  layout:    {spec['layout']}\n")

    # 1. Download all zips
    zip_paths = []
    for filename, url in spec["urls"].items():
        dest = downloads / filename
        try:
            _download(url, dest, insecure=args.insecure)
        except Exception as e:
            print(f"\n[error] could not download {url}: {e}\n"
                  f"        you can download the file manually and place it at:\n"
                  f"            {dest}\n"
                  f"        then re-run this command -- it will skip the download.\n")
            sys.exit(1)
        zip_paths.append(dest)

    # 2. Prepare folder layout
    if spec["layout"] == "flat":
        n = _prepare_flat(out, downloads, zip_paths)
        print(f"\n[done] wrote {n} (image, mask) pairs into:")
        print(f"       {out / 'images'}")
        print(f"       {out / 'masks'}")
        print("       Train with --auto_split, e.g.:")
        print(f"         python train.py --data_root {out} --auto_split --epochs 20")
    else:  # duts official splits
        counts = _prepare_duts_official_splits(out, downloads, args.val_ratio, args.seed)
        print(f"\n[done] wrote DUTS into official splits:")
        for split in ("train", "val", "test"):
            print(f"       {split:5s} -> {counts[split]} pairs")
        print("       Train without --auto_split, e.g.:")
        print(f"         python train.py --data_root {out} --epochs 20")


if __name__ == "__main__":
    main()
