# FoodGuard AI

Food-safety detection for Indian consumers. Upload a photo and get one clear verdict: **SAFE / CAUTION / UNSAFE**, or **REJECTED / MISMATCH** when the image is not usable.

* **Raw produce**: a fine-tuned MobileNetV2 detects spoilage, and Grad-CAM shows where it looked.
* **Packaged food**: EasyOCR reads the nutrition panel, and the label is checked against FSSAI HFSS limits (red‑hexagon warnings). An expiry date can be entered by hand if you want it checked too; it is optional.
* **Guardrails**: three checks run before analysis. Is it food? Does it match the chosen category? Is it the product you named?
* **Review**: every analysed item gets a 0–100 health score with a rating and star display, plus information about the food (nutrition, storage, shelf life, signs of spoilage, share of the daily maximum).

## 1. Setup on a new PC

You need Python 3.10 or newer (3.12 was used) and Git. The trained spoilage model (`models/spoilage_mobilenetv2.pt`) is included in the repository, so **no training is needed** to run the app.

```powershell
git clone https://github.com/Yash7891/Food-Gaurd-AI.git
cd Food-Gaurd-AI
python -m venv .venv
.venv\Scripts\activate

# 1. PyTorch. Pick ONE line:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128   # NVIDIA GPU (RTX 50-series needs cu128)
pip install torch torchvision                                                     # no NVIDIA GPU (runs on CPU, slower)

# 2. Everything else
pip install -r requirements.txt

# 3. Run
python app.py        # open http://127.0.0.1:5000
```

The first run needs internet: it downloads CLIP (~600 MB) and the EasyOCR weights (~100 MB). After that the app starts offline; only product look‑ups use Open Food Facts.

*Additional documentation, data, training scripts, and usage details remain unchanged in the repository.*