"""
train.py
========

Full training loop for the Salient Object Detection model.

Features
--------
* Adam optimizer (lr = 1e-3 by default, configurable)
* Loss = BCE + 0.5 * (1 - IoU)
* Per-epoch validation + early stopping on validation loss
* Checkpoint save every epoch (``last.pt``) and best-ever (``best.pt``)
* Automatic resume from a checkpoint (``--resume path``)
* JSON training history saved to ``results/history_<tag>.json``

Usage
-----
    python train.py --data_root data/synthetic --auto_split --epochs 20

Resume:

    python train.py --resume checkpoints/last.pt
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, Tuple

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_loader import build_dataloaders
from sod_model import BCEIoULoss, build_model, count_parameters


# ---------------------------------------------------------------------------
# Metrics (computed on binarised predictions, threshold = 0.5)
# ---------------------------------------------------------------------------


@torch.no_grad()
def batch_metrics(prob: torch.Tensor, target: torch.Tensor, thr: float = 0.5, eps: float = 1e-7):
    pred = (prob > thr).float()
    tgt = (target > 0.5).float()
    tp = (pred * tgt).sum(dim=(1, 2, 3))
    fp = (pred * (1 - tgt)).sum(dim=(1, 2, 3))
    fn = ((1 - pred) * tgt).sum(dim=(1, 2, 3))
    iou = tp / (tp + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    mae = (prob - tgt).abs().mean(dim=(1, 2, 3))
    return {
        "iou": iou.mean().item(),
        "precision": precision.mean().item(),
        "recall": recall.mean().item(),
        "f1": f1.mean().item(),
        "mae": mae.mean().item(),
    }


def average_metrics(history: list) -> Dict[str, float]:
    if not history:
        return {}
    keys = history[0].keys()
    return {k: float(sum(h[k] for h in history) / len(history)) for k in keys}


# ---------------------------------------------------------------------------
# Train / validate epochs
# ---------------------------------------------------------------------------


def train_one_epoch(model, loader: DataLoader, criterion, optimizer, device) -> Dict[str, float]:
    model.train()
    losses = []
    metrics_acc = []
    pbar = tqdm(loader, desc="train", leave=False)
    for imgs, masks in pbar:
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(imgs)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        with torch.no_grad():
            metrics_acc.append(batch_metrics(torch.sigmoid(logits), masks))
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    out = {"loss": float(sum(losses) / max(len(losses), 1))}
    out.update(average_metrics(metrics_acc))
    return out


@torch.no_grad()
def validate(model, loader: DataLoader, criterion, device) -> Dict[str, float]:
    model.eval()
    losses = []
    metrics_acc = []
    for imgs, masks in tqdm(loader, desc="val", leave=False):
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        logits = model(imgs)
        loss = criterion(logits, masks)
        losses.append(loss.item())
        metrics_acc.append(batch_metrics(torch.sigmoid(logits), masks))
    out = {"loss": float(sum(losses) / max(len(losses), 1))}
    out.update(average_metrics(metrics_acc))
    return out


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------


def save_checkpoint(path: Path, **kwargs) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(kwargs, path)
    print(f"[ckpt] saved -> {path}")


def load_checkpoint(path: Path, model, optimizer=None, device="cpu") -> Dict:
    print(f"[ckpt] loading <- {path}")
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    if optimizer is not None and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    return ckpt


# ---------------------------------------------------------------------------
# Main training entry point
# ---------------------------------------------------------------------------


def main(args: argparse.Namespace) -> None:
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"[device] {device}")

    # ------------------------------------------------------------------
    # If resuming, restore config from the checkpoint when not overridden.
    # ------------------------------------------------------------------
    resume_state = None
    if args.resume:
        resume_state = torch.load(args.resume, map_location="cpu")
        cfg = resume_state.get("config", {})
        for key in ("data_root", "img_size", "model", "auto_split", "tag", "batch_size"):
            if getattr(args, key) in (None, False, "", 0) and key in cfg:
                setattr(args, key, cfg[key])
        print(f"[resume] using config from checkpoint: {cfg}")

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    train_loader, val_loader, _ = build_dataloaders(
        args.data_root,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augment_train=not args.no_aug,
        auto_split=args.auto_split,
    )

    # ------------------------------------------------------------------
    # Model + optimizer + loss
    # ------------------------------------------------------------------
    model = build_model(args.model).to(device)
    total, trainable = count_parameters(model)
    print(f"[model] {args.model}  params={total:,} (trainable {trainable:,})")

    optimizer = Adam(model.parameters(), lr=args.lr)
    criterion = BCEIoULoss(alpha=0.5)

    start_epoch = 0
    best_val = float("inf")
    epochs_no_improve = 0
    history = []

    if resume_state is not None:
        model.load_state_dict(resume_state["model_state"])
        optimizer.load_state_dict(resume_state["optimizer_state"])
        start_epoch = resume_state.get("epoch", 0) + 1
        best_val = resume_state.get("best_val", float("inf"))
        epochs_no_improve = resume_state.get("epochs_no_improve", 0)
        history = resume_state.get("history", [])
        print(f"[resume] continuing from epoch {start_epoch}  (best_val={best_val:.4f})")

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    tag = args.tag or args.model
    last_ckpt = ckpt_dir / f"last_{tag}.pt"
    best_ckpt = ckpt_dir / f"best_{tag}.pt"

    config = {
        "data_root": args.data_root,
        "img_size": args.img_size,
        "model": args.model,
        "auto_split": args.auto_split,
        "tag": tag,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "epochs": args.epochs,
    }

    print(f"\n[train] starting at epoch {start_epoch + 1} / {args.epochs}\n")

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        train_stats = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_stats = validate(model, val_loader, criterion, device)
        dt = time.time() - t0

        log = {
            "epoch": epoch + 1,
            "train": train_stats,
            "val": val_stats,
            "time_sec": dt,
        }
        history.append(log)

        print(
            f"epoch {epoch + 1:3d}/{args.epochs}  "
            f"train_loss={train_stats['loss']:.4f}  val_loss={val_stats['loss']:.4f}  "
            f"val_IoU={val_stats['iou']:.3f}  val_F1={val_stats['f1']:.3f}  "
            f"({dt:.1f}s)"
        )

        # Save "last"
        save_checkpoint(
            last_ckpt,
            model_state=model.state_dict(),
            optimizer_state=optimizer.state_dict(),
            epoch=epoch,
            best_val=best_val,
            epochs_no_improve=epochs_no_improve,
            history=history,
            config=config,
        )

        # Save "best" if val loss improved
        if val_stats["loss"] < best_val - 1e-6:
            best_val = val_stats["loss"]
            epochs_no_improve = 0
            save_checkpoint(
                best_ckpt,
                model_state=model.state_dict(),
                optimizer_state=optimizer.state_dict(),
                epoch=epoch,
                best_val=best_val,
                epochs_no_improve=epochs_no_improve,
                history=history,
                config=config,
            )
            print(f"[ckpt] *** new best val_loss = {best_val:.4f} ***")
        else:
            epochs_no_improve += 1
            print(f"[early-stop] no improvement for {epochs_no_improve}/{args.patience} epochs")
            if epochs_no_improve >= args.patience:
                print("[early-stop] stopping training.")
                break

    # ------------------------------------------------------------------
    # Save history JSON
    # ------------------------------------------------------------------
    hist_path = Path("results") / f"history_{tag}.json"
    hist_path.write_text(json.dumps(history, indent=2))
    print(f"\n[done] history -> {hist_path}\n[done] best ckpt -> {best_ckpt}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the SOD model.")
    p.add_argument("--data_root", type=str, default="")
    p.add_argument("--img_size", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--model", type=str, default="improved", choices=["baseline", "improved"])
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--patience", type=int, default=5, help="Early-stopping patience.")
    p.add_argument("--no_aug", action="store_true", help="Disable training augmentations.")
    p.add_argument("--auto_split", action="store_true",
                   help="Auto-split flat images/masks folders into 70/15/15.")
    p.add_argument("--tag", type=str, default="", help="Tag for checkpoint/history filenames.")
    p.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    p.add_argument("--resume", type=str, default="", help="Path to a checkpoint to resume from.")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
