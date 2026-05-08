"""Generate the 5-slide presentation deck (PDF, landscape) using reportlab.

All numbers are loaded from results/ at run time so the deck stays in sync
with the latest evaluation.
"""
from __future__ import annotations

import json
from pathlib import Path

from reportlab.lib.colors import HexColor, grey, white, black
from reportlab.lib.pagesizes import landscape
from reportlab.lib.units import cm
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader


ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
SLIDES_DIR = ROOT / "slides"
SLIDES_DIR.mkdir(parents=True, exist_ok=True)
OUT_PDF = SLIDES_DIR / "presentation.pdf"


# 16:9 landscape page (33.87 cm x 19.05 cm)
PAGE_W, PAGE_H = (33.87 * cm, 19.05 * cm)

PRIMARY = HexColor("#1f4e79")
ACCENT = HexColor("#2e75b6")
LIGHT = HexColor("#deebf7")
SOFT = HexColor("#f6f8fb")
DARK = HexColor("#0f2c44")


# Real numbers
b = json.loads((RESULTS / "baseline" / "metrics.json").read_text())
i = json.loads((RESULTS / "improved" / "metrics.json").read_text())


def fmt(x: float, n: int = 4) -> str:
    return f"{x:.{n}f}"


def header(c: canvas.Canvas, slide_num: int, total: int):
    """Draw header band + slide number."""
    c.setFillColor(PRIMARY)
    c.rect(0, PAGE_H - 1.4 * cm, PAGE_W, 1.4 * cm, fill=1, stroke=0)
    c.setFillColor(white)
    c.setFont("Helvetica-Bold", 11)
    c.drawString(1 * cm, PAGE_H - 0.92 * cm, "Salient Object Detection — Project #3")
    c.setFont("Helvetica", 9)
    c.drawRightString(PAGE_W - 1 * cm, PAGE_H - 0.92 * cm, f"Slide {slide_num} / {total}")


def title(c: canvas.Canvas, text: str):
    c.setFillColor(DARK)
    c.setFont("Helvetica-Bold", 26)
    c.drawString(1.2 * cm, PAGE_H - 2.6 * cm, text)
    c.setStrokeColor(ACCENT)
    c.setLineWidth(2)
    c.line(1.2 * cm, PAGE_H - 2.85 * cm, 6 * cm, PAGE_H - 2.85 * cm)


def bullet(c: canvas.Canvas, x: float, y: float, text: str, size: int = 13, bold=False, max_w_cm=15):
    c.setFillColor(ACCENT)
    c.circle(x - 0.25 * cm, y + 0.12 * cm, 0.10 * cm, fill=1, stroke=0)
    c.setFillColor(black)
    c.setFont("Helvetica-Bold" if bold else "Helvetica", size)
    # Crude wrapping
    words = text.split()
    line, lines = "", []
    for w in words:
        trial = (line + " " + w).strip()
        if c.stringWidth(trial, "Helvetica-Bold" if bold else "Helvetica", size) > max_w_cm * cm:
            lines.append(line)
            line = w
        else:
            line = trial
    if line:
        lines.append(line)
    for k, ln in enumerate(lines):
        c.drawString(x, y - k * (size + 4), ln)
    return len(lines) * (size + 4)


def fit_image(path: Path, max_w_cm: float, max_h_cm: float):
    img = ImageReader(str(path))
    iw, ih = img.getSize()
    scale = min((max_w_cm * cm) / iw, (max_h_cm * cm) / ih)
    return img, iw * scale, ih * scale


# ---------------------------------------------------------------------------
c = canvas.Canvas(str(OUT_PDF), pagesize=(PAGE_W, PAGE_H))
c.setTitle("SOD Project — Presentation")
c.setAuthor("Ernisa Sallahu")
TOTAL = 5

# =============== SLIDE 1: Cover ===============
c.setFillColor(SOFT)
c.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)
c.setFillColor(PRIMARY)
c.rect(0, PAGE_H - 6 * cm, PAGE_W, 6 * cm, fill=1, stroke=0)

c.setFillColor(white)
c.setFont("Helvetica-Bold", 36)
c.drawCentredString(PAGE_W / 2, PAGE_H - 3.4 * cm, "Salient Object Detection")
c.setFont("Helvetica", 18)
c.drawCentredString(PAGE_W / 2, PAGE_H - 4.6 * cm, "End-to-End ML/DL Project — From-scratch CNN encoder–decoder")

c.setFillColor(DARK)
c.setFont("Helvetica", 14)
c.drawCentredString(PAGE_W / 2, PAGE_H - 8 * cm, "Author: Ernisa Sallahu")
c.drawCentredString(PAGE_W / 2, PAGE_H - 8.8 * cm, "Framework: PyTorch  |  Dataset: DUTS  |  No pretrained weights")

# Result highlight box
c.setFillColor(white)
c.setStrokeColor(ACCENT)
c.setLineWidth(2)
c.roundRect(PAGE_W / 2 - 9 * cm, 3 * cm, 18 * cm, 4 * cm, 0.4 * cm, fill=1, stroke=1)
c.setFillColor(PRIMARY)
c.setFont("Helvetica-Bold", 16)
c.drawCentredString(PAGE_W / 2, 6 * cm, "Final test-set result (improved model, DUTS-TE, 5,019 images)")

c.setFillColor(DARK)
c.setFont("Helvetica-Bold", 22)
metrics_text = f"IoU = {fmt(i['iou'], 3)}    F1 = {fmt(i['f1'], 3)}    MAE = {fmt(i['mae'], 3)}"
c.drawCentredString(PAGE_W / 2, 4.4 * cm, metrics_text)

c.setFillColor(grey)
c.setFont("Helvetica-Oblique", 11)
c.drawCentredString(PAGE_W / 2, 1.5 * cm, "Slide 1 / 5")
c.showPage()


# =============== SLIDE 2: Problem & Pipeline ===============
header(c, 2, TOTAL)
title(c, "Problem & Pipeline")

y = PAGE_H - 4.5 * cm
y -= bullet(c, 1.6 * cm, y, "Goal: predict a pixel-level binary mask of the most visually salient object in an image.", 14, bold=True)
y -= 0.2 * cm
y -= bullet(c, 1.6 * cm, y, "Built end-to-end from scratch in PyTorch — no pretrained backbones.", 13)
y -= 0.1 * cm
y -= bullet(c, 1.6 * cm, y, "Pipeline: data loader → augmentation → CNN encoder–decoder → BCE+IoU loss → eval (IoU/P/R/F1/MAE) → demo.", 13)
y -= 0.1 * cm
y -= bullet(c, 1.6 * cm, y, "Dataset: DUTS — 8,970 train / 1,583 val / 5,019 test (official DUTS-TE held out).", 13)
y -= 0.1 * cm
y -= bullet(c, 1.6 * cm, y, "Preprocessing: resize 128×128, pixels → [0,1], masks binarised at 0.5.", 13)
y -= 0.1 * cm
y -= bullet(c, 1.6 * cm, y, "Augmentation: horizontal flip, random crop+resize, brightness ±30%.", 13)

# Right side: pipeline boxes
def step_box(x, y, label, w=4.6 * cm, h=1.1 * cm, fill=ACCENT):
    c.setFillColor(fill)
    c.setStrokeColor(fill)
    c.roundRect(x, y, w, h, 0.18 * cm, fill=1, stroke=1)
    c.setFillColor(white)
    c.setFont("Helvetica-Bold", 11)
    c.drawCentredString(x + w / 2, y + h / 2 - 4, label)


pipeline_x = PAGE_W - 6.5 * cm
py = PAGE_H - 5.5 * cm
for label in ["Data loader", "Augmentation", "CNN encoder", "CNN decoder", "BCE + IoU loss", "Adam, lr=1e-3", "Eval + Demo"]:
    step_box(pipeline_x, py, label)
    py -= 1.5 * cm

c.showPage()


# =============== SLIDE 3: Model Architecture ===============
header(c, 3, TOTAL)
title(c, "Model Architecture")

y = PAGE_H - 4.5 * cm
c.setFillColor(DARK)
c.setFont("Helvetica-Bold", 14)
c.drawString(1.6 * cm, y, "Baseline (564,833 params)")
y -= 0.7 * cm
y -= bullet(c, 1.9 * cm, y, "Encoder: 4 × (Conv2d → ReLU → MaxPool), channels 32→64→128→256.", 12)
y -= bullet(c, 1.9 * cm, y, "Decoder: 4 × (ConvTranspose2d, stride=2 → ReLU).", 12)
y -= bullet(c, 1.9 * cm, y, "Head: 1×1 Conv → 1-channel logits at 128×128.", 12)

y -= 0.6 * cm
c.setFillColor(DARK)
c.setFont("Helvetica-Bold", 14)
c.drawString(1.6 * cm, y, "Improved (1,349,633 params, ~2.4× baseline)")
y -= 0.7 * cm
y -= bullet(c, 1.9 * cm, y, "DoubleConvBlock per stage (2 convs → BN → ReLU) — deeper feature extraction.", 12)
y -= bullet(c, 1.9 * cm, y, "BatchNorm2d after every Conv2d — stabilises training, faster convergence.", 12)
y -= bullet(c, 1.9 * cm, y, "Dropout2d(p=0.2) at the end of each encoder stage — regularisation.", 12)
y -= bullet(c, 1.9 * cm, y, "Loss: BCEWithLogits + 0.5·(1 − soft-IoU). Optimiser: Adam, lr=1e-3.", 12)
y -= bullet(c, 1.9 * cm, y, "Bonus: full save/resume — last.pt + best.pt with optimiser state.", 12)

# Right: training curves figure
curves = RESULTS / "training_curves.png"
if curves.exists():
    img, w_, h_ = fit_image(curves, max_w_cm=15, max_h_cm=10)
    c.drawImage(img, PAGE_W - 16.5 * cm, 1.5 * cm, width=w_, height=h_)
    c.setFillColor(grey)
    c.setFont("Helvetica-Oblique", 9)
    c.drawString(PAGE_W - 16.5 * cm, 1.1 * cm, "Loss / IoU / F1 across epochs (baseline vs improved).")

c.showPage()


# =============== SLIDE 4: Results & Comparison ===============
header(c, 4, TOTAL)
title(c, "Results: Baseline vs Improved (DUTS-TE, 5,019 images)")

# Metrics table
data = [
    ("Metric",      "Baseline",                   "Improved",                   "Δ"),
    ("IoU",         fmt(b["iou"], 3),             fmt(i["iou"], 3),             f"+{fmt(i['iou']-b['iou'], 3)}"),
    ("Precision",   fmt(b["precision"], 3),       fmt(i["precision"], 3),       f"+{fmt(i['precision']-b['precision'], 3)}"),
    ("Recall",      fmt(b["recall"], 3),          fmt(i["recall"], 3),          f"+{fmt(i['recall']-b['recall'], 3)}"),
    ("F1",          fmt(b["f1"], 3),              fmt(i["f1"], 3),              f"+{fmt(i['f1']-b['f1'], 3)}"),
    ("MAE (↓)",     fmt(b["mae"], 3),             fmt(i["mae"], 3),             f"{fmt(i['mae']-b['mae'], 3)}"),
    ("Inference",   f"{b['mean_inference_ms']:.1f} ms", f"{i['mean_inference_ms']:.1f} ms",
                                                                                f"+{i['mean_inference_ms']-b['mean_inference_ms']:.1f} ms"),
]

x0, y0 = 1.6 * cm, PAGE_H - 5 * cm
col_w = [3.4 * cm, 3.0 * cm, 3.0 * cm, 2.4 * cm]
row_h = 0.95 * cm

for r, row in enumerate(data):
    cy = y0 - r * row_h
    if r == 0:
        c.setFillColor(PRIMARY)
        c.rect(x0, cy - row_h + 0.05 * cm, sum(col_w), row_h, fill=1, stroke=0)
        c.setFillColor(white)
        c.setFont("Helvetica-Bold", 12)
    else:
        c.setFillColor(LIGHT if r % 2 == 0 else white)
        c.rect(x0, cy - row_h + 0.05 * cm, sum(col_w), row_h, fill=1, stroke=0)
        c.setFillColor(DARK)
        c.setFont("Helvetica" if r != len(data) - 1 else "Helvetica", 11)
    cx = x0
    for k, cell in enumerate(row):
        font = "Helvetica-Bold" if r == 0 or k == 0 else "Helvetica"
        c.setFont(font, 12 if r == 0 else 11)
        c.setFillColor(white if r == 0 else DARK)
        c.drawString(cx + 0.25 * cm, cy - row_h + 0.35 * cm, cell)
        cx += col_w[k]

# Right column: a sample visualisation
sample = RESULTS / "improved" / "sample_00.png"
if sample.exists():
    img, w_, h_ = fit_image(sample, max_w_cm=18, max_h_cm=5)
    c.drawImage(img, PAGE_W - 19 * cm, 6.5 * cm, width=w_, height=h_)
    c.setFillColor(grey)
    c.setFont("Helvetica-Oblique", 9)
    c.drawString(PAGE_W - 19 * cm, 6.1 * cm, "Improved model — input | GT | predicted mask | overlay")

# Takeaway band
c.setFillColor(LIGHT)
c.rect(0, 1.4 * cm, PAGE_W, 2.6 * cm, fill=1, stroke=0)
c.setFillColor(PRIMARY)
c.setFont("Helvetica-Bold", 14)
c.drawString(1.6 * cm, 3.3 * cm, "Takeaway")
c.setFillColor(DARK)
c.setFont("Helvetica", 12)
delta_iou = i["iou"] - b["iou"]
delta_f1 = i["f1"] - b["f1"]
c.drawString(
    1.6 * cm,
    2.4 * cm,
    f"Improved model: +{delta_iou:.3f} IoU and +{delta_f1:.3f} F1 over baseline, "
    f"≈{1000/i['mean_inference_ms']:.0f} fps on Apple MPS — substantial quality gain at near-zero latency cost.",
)

c.showPage()


# =============== SLIDE 5: Demo & Conclusion ===============
header(c, 5, TOTAL)
title(c, "Demo & Conclusion")

y = PAGE_H - 4.5 * cm
y -= bullet(c, 1.6 * cm, y, "Streamlit demo (demo.py): upload an image → predicted mask + overlay + inference time.", 13, bold=True)
y -= 0.1 * cm
y -= bullet(c, 1.6 * cm, y, "Notebook demo (demo_notebook.ipynb) for offline review.", 13)
y -= 0.1 * cm
y -= bullet(c, 1.6 * cm, y, f"Real-time on a Mac: ~{i['mean_inference_ms']:.1f} ms / image at 128×128 on Apple MPS.", 13)
y -= 0.4 * cm

c.setFillColor(DARK)
c.setFont("Helvetica-Bold", 14)
c.drawString(1.6 * cm, y, "Deliverables (all completed)")
y -= 0.7 * cm
items = [
    "data_loader.py, sod_model.py, train.py, evaluate.py, demo.py, demo_notebook.ipynb",
    "Train/val/test split (70/15/15-style) on DUTS, with augmentation",
    "Custom CNN encoder–decoder (no pretrained weights), BCE+IoU loss, Adam lr=1e-3",
    "Full training loop with logging, validation, checkpointing + resume (bonus)",
    "Evaluation: IoU, Precision, Recall, F1, MAE + 4-panel sample visualisations",
    "Two experiments: BatchNorm + Dropout, and a deeper DoubleConv encoder",
    "Project report (PDF) and presentation slides (this deck)",
]
for it in items:
    y -= bullet(c, 1.9 * cm, y, it, 12)
    y -= 0.05 * cm

# Right side: final metrics box
c.setFillColor(PRIMARY)
c.roundRect(PAGE_W - 11 * cm, 5.5 * cm, 9.5 * cm, 6.5 * cm, 0.4 * cm, fill=1, stroke=0)
c.setFillColor(white)
c.setFont("Helvetica-Bold", 13)
c.drawCentredString(PAGE_W - 6.25 * cm, 11.2 * cm, "Final test metrics (improved)")

c.setFont("Helvetica-Bold", 22)
c.drawCentredString(PAGE_W - 6.25 * cm, 9.8 * cm, f"IoU  =  {fmt(i['iou'], 3)}")
c.drawCentredString(PAGE_W - 6.25 * cm, 8.8 * cm, f"F1   =  {fmt(i['f1'], 3)}")
c.drawCentredString(PAGE_W - 6.25 * cm, 7.8 * cm, f"MAE  =  {fmt(i['mae'], 3)}")

c.setFont("Helvetica", 11)
c.drawCentredString(PAGE_W - 6.25 * cm, 6.5 * cm, "Evaluated on 5,019 DUTS-TE images")

# Bottom thank you
c.setFillColor(LIGHT)
c.rect(0, 0, PAGE_W, 1.4 * cm, fill=1, stroke=0)
c.setFillColor(PRIMARY)
c.setFont("Helvetica-Bold", 14)
c.drawCentredString(PAGE_W / 2, 0.5 * cm, "Thank you — Questions?")

c.showPage()
c.save()
print(f"[done] wrote {OUT_PDF}")
