
# Image Captioning & Segmentation

This project demonstrates **image captioning** (CNN+LSTM) and **image segmentation** (U-Net), with offline fallbacks for demo (rule-based captioning, GrabCut segmentation).

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

Upload your own image or use `data/sample_images/demo.jpg`.
