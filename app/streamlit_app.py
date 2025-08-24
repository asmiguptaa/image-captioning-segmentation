import streamlit as st
from pathlib import Path
from PIL import Image
import numpy as np
import cv2
import torch
import torch.nn as nn

# ======================
# Setup
# ======================
st.set_page_config(page_title="Image Captioning + Segmentation", layout="wide")
st.title("🖼️ Image Captioning & Segmentation Demo")

ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(exist_ok=True)

last_input_path = ARTIFACTS_DIR / "last_input.jpg"
last_mask_path = ARTIFACTS_DIR / "last_mask.png"

# ======================
# Dummy model stubs
# ======================
class TinyUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(),
            nn.Conv2d(8, 16, 3, padding=1), nn.ReLU()
        )
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(16, 8, 3, padding=1), nn.ReLU(),
            nn.Conv2d(8, 1, 1)
        )
    def forward(self, x): return self.dec(self.enc(x))

class TinyCaptionModel(nn.Module):
    def __init__(self, vocab_size=50, hidden_dim=32):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 8, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        self.fc_img = nn.Linear(8*16*16, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)
    def forward(self, x):
        feats = self.cnn(x)
        feats = self.fc_img(feats).unsqueeze(1)  # (B,1,H)
        out, _ = self.lstm(feats)
        return self.fc_out(out.squeeze(1))

# ======================
# Load models if available
# ======================
use_unet, use_captioner = False, False

if (ARTIFACTS_DIR / "unet.pt").exists():
    unet = TinyUNet()
    unet.load_state_dict(torch.load(ARTIFACTS_DIR / "unet.pt", map_location="cpu"))
    unet.eval()
    use_unet = True

if (ARTIFACTS_DIR / "caption_cnn_lstm.pt").exists():
    caption_model = TinyCaptionModel()
    caption_model.load_state_dict(torch.load(ARTIFACTS_DIR / "caption_cnn_lstm.pt", map_location="cpu"))
    caption_model.eval()
    use_captioner = True

# ======================
# Helper functions
# ======================
def grabcut_segment(img):
    arr = np.array(img)
    mask = np.zeros(arr.shape[:2], np.uint8)
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)
    h,w = arr.shape[:2]
    rect = (int(w*0.1), int(h*0.1), int(w*0.8), int(h*0.8))
    cv2.grabCut(arr, mask, rect, bgdModel, fgdModel, 3, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    return Image.fromarray(mask2*255)

def rule_based_caption(img):
    arr = np.array(img)
    mean_colors = arr.reshape(-1,3).mean(axis=0)
    dom = ["red","green","blue"][int(np.argmax(mean_colors))]
    return f"An image with dominant {dom} tones."

# ======================
# Upload / Sample Image
# ======================
uploaded = st.file_uploader("Upload an image", type=["jpg","png","jpeg"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    img.save(last_input_path)
else:
    st.info("Using sample image (data/sample_images/demo.jpg)")
    img = Image.open("data/sample_images/demo.jpg").convert("RGB")
    img.save(last_input_path)

# ======================
# Segmentation
# ======================
if use_unet:
    st.success("Using trained U-Net model for segmentation")
    arr = np.array(img.resize((64,64)))
    x = torch.tensor(arr.transpose(2,0,1)).float().unsqueeze(0)/255.0
    with torch.no_grad():
        pred = unet(x).squeeze().numpy()
    mask = (pred > 0).astype(np.uint8)*255
    mask_img = Image.fromarray(mask).resize(img.size)
else:
    st.warning("No U-Net found → using GrabCut fallback")
    mask_img = grabcut_segment(img)

mask_img.save(last_mask_path)

# ======================
# Captioning
# ======================
if use_captioner:
    st.success("Using trained Captioning model")
    arr = np.array(img.resize((32,32)))
    x = torch.tensor(arr.transpose(2,0,1)).float().unsqueeze(0)/255.0
    with torch.no_grad():
        pred = caption_model(x)
    word_idx = pred.argmax(1).item()
    caption = f"Generated caption token id: {word_idx}"
else:
    st.warning("No caption model found → using rule-based fallback")
    caption = rule_based_caption(img)

# ======================
# Display
# ======================
col1, col2 = st.columns(2)
with col1:
    st.subheader("Input Image")
    st.image(img, use_column_width=True)
with col2:
    st.subheader("Segmentation Mask")
    st.image(mask_img, use_column_width=True)

st.subheader("Caption")
st.write(caption)
