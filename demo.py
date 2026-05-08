"""
demo.py
=======

Streamlit mini-app for the Salient Object Detection model.

Run with:

    streamlit run demo.py

In the sidebar:
  * pick a checkpoint (.pt file under ./checkpoints)
  * pick the input image size (must match training)
Then upload an image. The app shows:
  * the input image
  * the predicted saliency mask
  * the overlay
  * the inference time (ms)
"""

from __future__ import annotations

import io
import time
from pathlib import Path

import numpy as np
import streamlit as st
import torch
from PIL import Image

from sod_model import build_model


st.set_page_config(page_title="Salient Object Detection", layout="wide")
st.title("Salient Object Detection — Demo")
st.caption("Upload an image to see the predicted saliency mask + overlay.")


@st.cache_resource(show_spinner=False)
def load_model(weights_path: str, model_name: str, device: str) -> torch.nn.Module:
    ckpt = torch.load(weights_path, map_location=device)
    cfg = ckpt.get("config", {})
    name = model_name or cfg.get("model", "improved")
    model = build_model(name).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def preprocess(pil_img: Image.Image, size: int) -> torch.Tensor:
    pil_img = pil_img.convert("RGB").resize((size, size), Image.BILINEAR)
    arr = np.array(pil_img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def overlay(pil_img: Image.Image, mask: np.ndarray, alpha: float = 0.45) -> Image.Image:
    img = np.array(pil_img.convert("RGB").resize(mask.shape[::-1], Image.BILINEAR)).astype(
        np.float32
    ) / 255.0
    color = np.zeros_like(img); color[..., 0] = 1.0
    m = mask[..., None]
    out = img * (1 - alpha * m) + color * (alpha * m)
    return Image.fromarray((out * 255).clip(0, 255).astype(np.uint8))


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
ckpt_dir = Path("checkpoints")
ckpts = sorted([p.name for p in ckpt_dir.glob("*.pt")]) if ckpt_dir.exists() else []
if not ckpts:
    st.warning(
        "No checkpoints found in `./checkpoints`. "
        "Train a model first with `python train.py ...`."
    )
    st.stop()

ckpt_choice = st.sidebar.selectbox("Checkpoint", ckpts, index=0)
img_size = st.sidebar.select_slider("Input size", options=[128, 224], value=128)
threshold = st.sidebar.slider("Binarisation threshold", 0.0, 1.0, 0.5, 0.05)
model_name = st.sidebar.selectbox("Model", ["", "baseline", "improved"], index=0,
                                  help="Leave blank to use the value stored in the checkpoint.")

device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
st.sidebar.write(f"**Device:** `{device}`")

model = load_model(str(ckpt_dir / ckpt_choice), model_name, device)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp"])
if uploaded is None:
    st.info("Upload an image to start.")
    st.stop()

pil_img = Image.open(io.BytesIO(uploaded.read()))
x = preprocess(pil_img, img_size).to(device)

with torch.no_grad():
    t0 = time.time()
    prob = torch.sigmoid(model(x))[0, 0].cpu().numpy()
    dt_ms = (time.time() - t0) * 1000.0

binary = (prob > threshold).astype(np.float32)

c1, c2, c3 = st.columns(3)
with c1:
    st.subheader("Input")
    st.image(pil_img.resize((img_size, img_size)))
with c2:
    st.subheader("Predicted mask")
    st.image((prob * 255).astype(np.uint8), clamp=True)
with c3:
    st.subheader("Overlay")
    st.image(overlay(pil_img.resize((img_size, img_size)), binary))

st.success(f"Inference time: **{dt_ms:.1f} ms** on `{device}`")
