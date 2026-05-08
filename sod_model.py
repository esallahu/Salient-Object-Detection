"""
sod_model.py
============

CNN encoder–decoder for Salient Object Detection, built from scratch.

Two variants are provided:

* ``BaselineSOD``  : 4 Conv2D + ReLU + MaxPool encoder, 4 ConvTranspose2D + ReLU
                    decoder, single-channel sigmoid output (matches the
                    "minimum requirements" of the project spec).

* ``ImprovedSOD``  : same backbone, but with BatchNorm and Dropout in the
                    encoder. Used for the "Experiments & Improvements" task.

The output is a 1-channel saliency mask the same H×W as the input.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class ConvBlock(nn.Module):
    """Conv -> (BN) -> ReLU -> (Dropout). Single conv per block."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        use_bn: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        layers = [nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=not use_bn)]
        if use_bn:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.ReLU(inplace=True))
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DoubleConvBlock(nn.Module):
    """Two consecutive Conv -> BN -> ReLU layers. Increases feature extraction depth."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        use_bn: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=not use_bn),
        ]
        if use_bn:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=not use_bn))
        if use_bn:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.ReLU(inplace=True))
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpBlock(nn.Module):
    """ConvTranspose2D (stride=2) -> ReLU. Doubles spatial size."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.up(x))


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class BaselineSOD(nn.Module):
    """4-stage encoder / 4-stage decoder. Input: (B, 3, H, W). Output: (B, 1, H, W)."""

    def __init__(self, base_channels: int = 32) -> None:
        super().__init__()
        c1, c2, c3, c4 = (
            base_channels,
            base_channels * 2,
            base_channels * 4,
            base_channels * 8,
        )

        # Encoder
        self.enc1 = ConvBlock(3, c1)
        self.enc2 = ConvBlock(c1, c2)
        self.enc3 = ConvBlock(c2, c3)
        self.enc4 = ConvBlock(c3, c4)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Decoder
        self.up4 = UpBlock(c4, c3)
        self.up3 = UpBlock(c3, c2)
        self.up2 = UpBlock(c2, c1)
        self.up1 = UpBlock(c1, c1)

        # Output: 1-channel mask + sigmoid is applied externally (we expose logits)
        self.head = nn.Conv2d(c1, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.enc1(x); x = self.pool(x)
        x = self.enc2(x); x = self.pool(x)
        x = self.enc3(x); x = self.pool(x)
        x = self.enc4(x); x = self.pool(x)

        x = self.up4(x)
        x = self.up3(x)
        x = self.up2(x)
        x = self.up1(x)

        return self.head(x)  # logits, shape (B, 1, H, W)


class ImprovedSOD(nn.Module):
    """Deeper encoder with DoubleConvBlock (2 convs per stage), BatchNorm, and Dropout.

    Improvements over BaselineSOD:
      1. Double convolution per encoder stage (more feature extraction)
      2. BatchNorm for training stability
      3. Dropout for regularization
    """

    def __init__(self, base_channels: int = 32, dropout: float = 0.2) -> None:
        super().__init__()
        c1, c2, c3, c4 = (
            base_channels,
            base_channels * 2,
            base_channels * 4,
            base_channels * 8,
        )

        self.enc1 = DoubleConvBlock(3, c1, use_bn=True, dropout=dropout)
        self.enc2 = DoubleConvBlock(c1, c2, use_bn=True, dropout=dropout)
        self.enc3 = DoubleConvBlock(c2, c3, use_bn=True, dropout=dropout)
        self.enc4 = DoubleConvBlock(c3, c4, use_bn=True, dropout=dropout)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.up4 = UpBlock(c4, c3)
        self.up3 = UpBlock(c3, c2)
        self.up2 = UpBlock(c2, c1)
        self.up1 = UpBlock(c1, c1)

        self.head = nn.Conv2d(c1, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.enc1(x)
        x = self.pool(x)
        x = self.enc2(x)
        x = self.pool(x)
        x = self.enc3(x)
        x = self.pool(x)
        x = self.enc4(x)
        x = self.pool(x)

        x = self.up4(x)
        x = self.up3(x)
        x = self.up2(x)
        x = self.up1(x)

        return self.head(x)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------


class BCEIoULoss(nn.Module):
    """``BCE + alpha * (1 - IoU)``  — exactly the loss requested in the spec."""

    def __init__(self, alpha: float = 0.5, eps: float = 1e-6) -> None:
        super().__init__()
        self.alpha = alpha
        self.eps = eps
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = self.bce(logits, target)
        prob = torch.sigmoid(logits)
        # Compute IoU on probabilities (soft IoU) to keep gradients smooth.
        intersection = (prob * target).sum(dim=(1, 2, 3))
        union = prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) - intersection
        iou = (intersection + self.eps) / (union + self.eps)
        return bce + self.alpha * (1.0 - iou.mean())


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_model(name: str = "improved", base_channels: int = 32) -> nn.Module:
    name = name.lower()
    if name == "baseline":
        return BaselineSOD(base_channels=base_channels)
    if name == "improved":
        return ImprovedSOD(base_channels=base_channels)
    raise ValueError(f"Unknown model name: {name!r}. Use 'baseline' or 'improved'.")


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


if __name__ == "__main__":
    for name in ("baseline", "improved"):
        m = build_model(name)
        total, trainable = count_parameters(m)
        x = torch.zeros(2, 3, 128, 128)
        y = m(x)
        print(f"{name:8s} -> output {tuple(y.shape)}  params={total:,} (trainable {trainable:,})")
