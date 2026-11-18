import os
import zipfile
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# Create a temporary folder for sample images
sample_dir = Path(__file__).parent / "sample_data"
sample_dir.mkdir(parents=True, exist_ok=True)

# Helper to create a simple image with text
def make_image(path, color, text):
    img = Image.new('RGB', (200, 200), color)
    d = ImageDraw.Draw(img)
    # Try to load a basic font; fallback to default
    try:
        font = ImageFont.load_default()
        d.text((10, 90), text, fill='white', font=font)
    except Exception:
        d.text((10, 90), text, fill='white')
    img.save(path)

# Create two dummy images – one "fresh" (green) and one "spoiled" (brown)
make_image(sample_dir / "fresh1.png", (34, 139, 34), "fresh")
make_image(sample_dir / "spoiled1.png", (139, 69, 19), "spoiled")

# Pack them into a zip that download_datasets.py can use as a fallback
zip_path = sample_dir / "fruit_sample.zip"
with zipfile.ZipFile(zip_path, 'w') as z:
    for img_file in sample_dir.glob('*.png'):
        z.write(img_file, arcname=img_file.name)

print(f"Sample data zip created at: {zip_path}")
