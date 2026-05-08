"""
evaluate.py
===========

Evaluate a trained Salient Object Detection model on the test split.

Reports
-------
* IoU, Precision, Recall, F1 (binarised at 0.5)
* Mean Absolute Error (on the continuous probability map)

Visualizations
--------------
For ``--num_visuals`` random test samples, saves a 1×4 panel:
    [Input]  [Ground Truth]  [Predicted Mask]  [Overlay]
into ``results/<tag>/`` plus a ``metrics.json`` file with the numbers.

Usage
-----
    python evaluate.py \
        --data_root data/synthetic \
        --weights checkpoints/best_improved.pt \
        --auto_split --num_visuals 8
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from data_loader import build_datasets
from sod_model import build_model


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


@torch.no_grad()
def compute_metrics(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    thr: float = 0.5,
    eps: float = 1e-7,
) -> Dict[str, float]:
    model.eval()
    tp_sum = fp_sum = fn_sum = 0.0
    mae_sum = 0.0
    n_pixels = 0
    n_images = 0

    for imgs, masks in loader:
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        prob = torch.sigmoid(model(imgs))
        pred = (prob > thr).float()
        tgt = (masks > 0.5).float()

        tp_sum += (pred * tgt).sum().item()
        fp_sum += (pred * (1 - tgt)).sum().item()
        fn_sum += ((1 - pred) * tgt).sum().item()
        mae_sum += (prob - tgt).abs().sum().item()
        n_pixels += tgt.numel()
        n_images += imgs.size(0)

    iou = tp_sum / (tp_sum + fp_sum + fn_sum + eps)
    precision = tp_sum / (tp_sum + fp_sum + eps)
    recall = tp_sum / (tp_sum + fn_sum + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    mae = mae_sum / max(n_pixels, 1)

    return {
        "num_images": n_images,
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mae": mae,
    }


# ---------------------------------------------------------------------------
# Inference + visualization
# ---------------------------------------------------------------------------


@torch.no_grad()
def predict_single(model: torch.nn.Module, image_t: torch.Tensor, device: torch.device):
    """Returns (prob[H,W] in [0,1], inference_time_ms)."""
    model.eval()
    x = image_t.unsqueeze(0).to(device)
    t0 = time.time()
    prob = torch.sigmoid(model(x))[0, 0].cpu().numpy()
    dt_ms = (time.time() - t0) * 1000.0
    return prob, dt_ms


def overlay_mask(image_chw: np.ndarray, mask_hw: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Red-tint overlay of mask on top of the input image."""
    img = image_chw.transpose(1, 2, 0)
    img = np.clip(img, 0, 1)
    color = np.zeros_like(img)
    color[..., 0] = 1.0
    m = mask_hw[..., None]
    return img * (1 - alpha * m) + color * (alpha * m)


def save_panel(input_t, gt, pred, save_path: Path, dt_ms: float) -> None:
    img = input_t.cpu().numpy()
    gt_np = gt.cpu().numpy()[0]
    pred_bin = (pred > 0.5).astype(np.float32)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(img.transpose(1, 2, 0)); axes[0].set_title("Input"); axes[0].axis("off")
    axes[1].imshow(gt_np, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Ground Truth"); axes[1].axis("off")
    axes[2].imshow(pred, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title(f"Predicted ({dt_ms:.1f} ms)"); axes[2].axis("off")
    axes[3].imshow(overlay_mask(img, pred_bin)); axes[3].set_title("Overlay"); axes[3].axis("off")
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def save_visualizations(
    model: torch.nn.Module,
    test_ds,
    device: torch.device,
    out_dir: Path,
    n: int = 8,
) -> List[float]:
    if len(test_ds) == 0:
        print("[evaluate] test split is empty, skipping visualizations")
        return []
    n = min(n, len(test_ds))
    rng = np.random.default_rng(0)
    idxs = rng.choice(len(test_ds), size=n, replace=False)
    inference_times = []
    for k, idx in enumerate(idxs):
        img_t, mask_t = test_ds[int(idx)]
        prob, dt_ms = predict_single(model, img_t, device)
        inference_times.append(dt_ms)
        save_panel(img_t, mask_t, prob, out_dir / f"sample_{k:02d}.png", dt_ms)
    print(f"[evaluate] saved {n} visualizations -> {out_dir}")
    return inference_times


# ---------------------------------------------------------------------------
# Main
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

    ckpt = torch.load(args.weights, map_location=device)
    cfg = ckpt.get("config", {})
    model_name = args.model or cfg.get("model", "improved")
    img_size = args.img_size or cfg.get("img_size", 128)
    auto_split = args.auto_split or cfg.get("auto_split", False)

    model = build_model(model_name).to(device)
    model.load_state_dict(ckpt["model_state"])
    print(f"[load] {args.weights}  model={model_name}  img_size={img_size}")

    _, _, test_ds = build_datasets(
        args.data_root,
        img_size=img_size,
        augment_train=False,
        auto_split=auto_split,
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    metrics = compute_metrics(model, test_loader, device)
    print("\n=== Test metrics ===")
    for k, v in metrics.items():
        print(f"  {k:>10}: {v:.4f}" if isinstance(v, float) else f"  {k:>10}: {v}")

    tag = args.tag or model_name
    out_dir = Path("results") / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    inf_times = save_visualizations(model, test_ds, device, out_dir, n=args.num_visuals)
    if inf_times:
        metrics["mean_inference_ms"] = float(np.mean(inf_times))

    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"[evaluate] metrics -> {out_dir / 'metrics.json'}")

    # Append to comparison CSV (used by the experiments table)
    cmp_csv = Path("results") / "comparison.csv"
    write_header = not cmp_csv.exists()
    with cmp_csv.open("a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["tag", "iou", "precision", "recall", "f1", "mae", "mean_inference_ms"])
        writer.writerow([
            tag,
            f"{metrics['iou']:.4f}",
            f"{metrics['precision']:.4f}",
            f"{metrics['recall']:.4f}",
            f"{metrics['f1']:.4f}",
            f"{metrics['mae']:.4f}",
            f"{metrics.get('mean_inference_ms', 0.0):.2f}",
        ])
    print(f"[evaluate] appended row to {cmp_csv}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate the SOD model on the test split.")
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--weights", type=str, required=True)
    p.add_argument("--img_size", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--model", type=str, default="", choices=["", "baseline", "improved"])
    p.add_argument("--auto_split", action="store_true")
    p.add_argument("--tag", type=str, default="")
    p.add_argument("--num_visuals", type=int, default=8)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
