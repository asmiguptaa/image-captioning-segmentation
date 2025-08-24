
# Image Captioning & Segmentation

This project demonstrates **image captioning** (CNN+LSTM) and **image segmentation** (U-Net), with offline fallbacks for demo (rule-based captioning, GrabCut segmentation).

## Features

Segmentation: U-Net (dummy training stub) → fallback to GrabCut

Captioning: CNN+LSTM (dummy training stub) → fallback to rule-based color captions

Streamlit Web App: Upload images, view segmentation & captions, works with/without training

Artifacts auto-save: input & mask stored in artifacts/ folder

## Quickstart

```bash
# 1. Create environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

# 2. Install deps
pip install -r requirements.txt

# 3. Run demo app
streamlit run app/streamlit_app.py
```
## Tech Stack

Python 3.11 (pinned for Streamlit Cloud via runtime.txt)

PyTorch → for model stubs

OpenCV (headless) → for segmentation fallback

Streamlit → interactive app

Pillow / NumPy → image processing

## Deployment

Deployed on Streamlit Cloud.

requirements.txt includes opencv-python-headless for compatibility.

runtime.txt sets Python 3.11 for stable builds.

Live demo 👉(https://image-captioning-segmentation.streamlit.app/)



