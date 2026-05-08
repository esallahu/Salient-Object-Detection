# Salient Object Detection (SOD) — End-to-End ML/DL Project

A complete from-scratch implementation of a Salient Object Detection system
using a CNN encoder–decoder, trained with a combined Binary Cross-Entropy +
IoU loss in PyTorch.

## Project structure

```
ML/
├── data_loader.py        # Dataset, preprocessing, augmentation
├── sod_model.py          # CNN encoder-decoder (baseline + improved)
├── train.py              # Training loop with checkpointing and resume
├── evaluate.py           # IoU / Precision / Recall / F1 / MAE + plots
├── demo.py               # Streamlit demo app
├── demo_notebook.ipynb   # Notebook demo (upload image -> mask)
├── download_dataset.py   # One-shot downloader for ECSSD / DUTS (with resume + retry)
├── make_synthetic.py     # Generates a tiny synthetic SOD dataset for smoke-tests
├── requirements.txt
├── checkpoints/          # Saved model weights and training state
├── results/              # Visualizations, metrics, comparison plots
└── data/                 # Place your dataset here
```

## 1. Setup

```bash
python -m venv .venv
source .venv/bin/activate          # macOS / Linux
pip install -r requirements.txt
```

## 2. Dataset

### Easiest: use the built-in downloader

```bash
# ECSSD (1000 images, ~66 MB total, recommended starting point)
python download_dataset.py --name ecssd --out data/ecssd

# DUTS-TR only (training portion, ~370 MB, 10553 images)
python download_dataset.py --name duts-tr --out data/duts-tr

# DUTS-TE only (test portion, ~140 MB, 5019 images)
python download_dataset.py --name duts-te --out data/duts-te

# DUTS full (TR + TE, places official splits as train/val/test)
python download_dataset.py --name duts --out data/duts
```

The downloader is **idempotent** (re-running skips finished steps), supports
**HTTP Range resume** (partial downloads continue from where they stopped)
and retries up to 8 times with exponential backoff — useful because the
academic CUHK / saliencydetection.net servers are sometimes flaky. If a URL
is permanently dead you can also place the zip manually into
`<out>/_downloads/` and re-run the same command — extraction will pick it
up automatically.

### Expected layout (if you bring your own data)

```
data/<dataset_name>/
├── images/
│   ├── train/   *.jpg|*.png
│   ├── val/
│   └── test/
└── masks/
    ├── train/   *.png  (single channel, 0 = background, 255 = salient)
    ├── val/
    └── test/
```

Supported public datasets: **DUTS**, **ECSSD**, **MSRA10K**, **SALICON**.

If you only have a single `images/` and `masks/` folder (this is what the
downloader produces for ECSSD / DUTS-TR / DUTS-TE), pass `--auto_split` to
`train.py` and it will randomly split into 70 / 15 / 15.

### Quick start without any download

To verify the pipeline runs end-to-end on a tiny synthetic dataset:

```bash
python make_synthetic.py --out data/synthetic --num 60
```

## 3. Training

```bash
python train.py \
    --data_root data/duts \
    --img_size 128 \
    --batch_size 16 \
    --epochs 25 \
    --lr 1e-3 \
    --model improved
```

Resume an interrupted run automatically from the latest checkpoint:

```bash
python train.py --resume checkpoints/last.pt
```

Useful flags:

| Flag             | Meaning                                              |
|------------------|------------------------------------------------------|
| `--model`        | `baseline` or `improved` (BatchNorm + Dropout)       |
| `--img_size`     | 128 or 224                                           |
| `--patience`     | Early-stopping patience (default 5)                  |
| `--no_aug`       | Disable data augmentation                            |
| `--auto_split`   | Auto split flat `images/`+`masks/` into 70/15/15     |

## 4. Evaluation

```bash
python evaluate.py \
    --data_root data/duts \
    --weights checkpoints/best_improved.pt \
    --img_size 128 \
    --num_visuals 8
```

This prints **IoU, Precision, Recall, F1, MAE** on the test split and saves
side-by-side visualizations (input / GT / prediction / overlay) into
`results/`.

## 5. Demo

### Streamlit (recommended for live demo)

```bash
streamlit run demo.py
```

Upload an image, see the predicted saliency mask, the overlay, and the
inference time per image.

### Notebook

Open `demo_notebook.ipynb` and run all cells.

## Presentation

- PDF slides: [slides/presentation.pdf](slides/presentation.pdf)

## 6. Experiments

Run baseline vs improved and compare:

```bash
python train.py --model baseline --tag baseline --data_root data/duts --epochs 20
python train.py --model improved --tag improved --data_root data/duts --epochs 25

python evaluate.py --weights checkpoints/best_baseline.pt --tag baseline \
    --data_root data/duts --num_visuals 8
python evaluate.py --weights checkpoints/best_improved.pt --tag improved \
    --data_root data/duts --num_visuals 8
```

Comparison metrics will be saved to `results/comparison.csv`.

## Notes

- Images are resized to 128×128 (default) or 224×224 and pixel values are
  normalised to **[0, 1]**.
- Loss = `BCE + 0.5 * (1 − IoU)` exactly as required.
- Optimizer: Adam with lr = 1e-3.
- Early stopping is triggered when validation loss stops improving for
  `--patience` consecutive epochs.
