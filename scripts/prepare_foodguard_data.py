"""Prepare the comprehensive FoodGuard AI Vision dataset.

Merges real Kaggle datasets (raghavrpotdar fresh-and-stale, siavash93 freshness44)
with augmented domain-specific images for all target produce and perishable food items:
  apple, banana, bitter_gourd, bread, capsicum, carrot, cucumber, grapes,
  lemon, mango, onion, orange, paneer_milk, potato, strawberry, tomato
  (32 total classes: fresh_<item> and stale_<item>)

Splits into train, validation, and test sets.
"""
import json
import math
import os
import random
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "dataset_vision"
CACHE_DIR = Path.home() / ".cache" / "kagglehub" / "datasets"

TARGET_PRODUCE = [
    "apple", "banana", "bitter_gourd", "bread", "capsicum", "carrot",
    "cucumber", "grapes", "lemon", "mango", "onion", "orange",
    "paneer_milk", "potato", "strawberry", "tomato"
]

TARGET_CLASSES = []
for p in TARGET_PRODUCE:
    TARGET_CLASSES.append(f"fresh_{p}")
    TARGET_CLASSES.append(f"stale_{p}")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

SPLIT_RATIOS = {"train": 0.70, "validation": 0.15, "test": 0.15}


def normalize_class_name(dir_name: str) -> tuple[str, str] | None:
    """Extract (state, produce) from various directory naming conventions."""
    name = dir_name.lower().replace("-", "_").replace(" ", "_")
    state, produce_part = None, None
    if name.endswith("_fresh") or name.endswith("_freshly"):
        state = "fresh"
        produce_part = name.replace("_freshly", "").replace("_fresh", "")
    elif name.endswith("_rotten") or name.endswith("_stale"):
        state = "stale"
        produce_part = name.replace("_rotten", "").replace("_stale", "")
    elif name.startswith("fresh_"):
        state = "fresh"
        produce_part = name[6:]
    elif name.startswith("stale_"):
        state = "stale"
        produce_part = name[6:]
    elif name.startswith("rotten_"):
        state = "stale"
        produce_part = name[7:]
    elif name.startswith("decayed_"):
        state = "stale"
        produce_part = name[8:]

    if not state or not produce_part:
        return None

    # Match produce_part to TARGET_PRODUCE
    aliases = {
        "bell_pepper": "capsicum",
        "bellpepper": "capsicum",
        "bitter_gourd": "bitter_gourd",
        "bittergourd": "bitter_gourd",
        "karela": "bitter_gourd",
        "lime": "lemon",
        "grape": "grapes",
        "paneer": "paneer_milk",
        "milk": "paneer_milk",
        "strawberries": "strawberry",
        "potatoes": "potato",
        "tomatoes": "tomato",
        "apples": "apple",
        "bananas": "banana",
        "oranges": "orange",
        "lemons": "lemon",
        "carrots": "carrot",
        "cucumbers": "cucumber",
        "onions": "onion",
        "mangoes": "mango",
    }
    produce = aliases.get(produce_part, produce_part)
    if produce in TARGET_PRODUCE:
        return state, produce
    return None


def find_available_kaggle_images() -> dict[str, list[Path]]:
    """Scan Kaggle cache and project data_raw for real images matching target classes."""
    class_images = defaultdict(list)

    scan_dirs = [
        CACHE_DIR,
        ROOT / "data_raw",
    ]

    for base_dir in scan_dirs:
        if not base_dir.exists():
            continue
        print(f"Scanning {base_dir} for datasets...")
        for root, dirs, files in os.walk(base_dir):
            root_path = Path(root)
            norm = normalize_class_name(root_path.name)
            if norm:
                state, produce = norm
                target_cls = f"{state}_{produce}"
                img_files = [root_path / f for f in files if Path(f).suffix.lower() in IMAGE_EXTS]
                if img_files:
                    class_images[target_cls].extend(img_files)

    for cls in TARGET_CLASSES:
        print(f"Found {len(class_images[cls])} real images for {cls}")
    return class_images


# Procedural generation for classes needing augmented/synthetic samples (e.g., bread, paneer, extra items)
COLOR_PALETTES = {
    "fresh_bread": {"base": (225, 190, 140), "crust": (160, 100, 45), "crumb": (245, 235, 210)},
    "stale_bread": {"base": (195, 175, 140), "mold": (45, 95, 65), "white_mold": (220, 230, 225), "dark": (80, 70, 50)},
    "fresh_paneer_milk": {"base": (250, 248, 245), "accent": (240, 235, 225)},
    "stale_paneer_milk": {"base": (225, 215, 160), "slime": (195, 185, 100), "mold": (90, 110, 80)},
    "fresh_potato": {"base": (205, 165, 115), "shadow": (165, 125, 80)},
    "stale_potato": {"base": (145, 115, 75), "green": (110, 145, 60), "rot": (60, 45, 30)},
    "fresh_onion": {"base": (195, 70, 90), "skin": (175, 50, 70), "white": (245, 240, 230)},
    "stale_onion": {"base": (120, 40, 50), "black_mold": (35, 35, 35), "soft": (80, 60, 45)},
    "fresh_mango": {"base": (245, 185, 30), "blush": (225, 85, 40)},
    "stale_mango": {"base": (160, 120, 25), "black_spot": (40, 30, 20), "wrinkle": (90, 70, 20)},
    "fresh_cucumber": {"base": (60, 130, 55), "light": (120, 180, 85)},
    "stale_cucumber": {"base": (115, 125, 60), "yellow": (160, 155, 65), "rot": (75, 65, 35)},
    "fresh_carrot": {"base": (235, 105, 20), "highlight": (250, 140, 45)},
    "stale_carrot": {"base": (160, 75, 20), "dark": (70, 40, 20), "white_rot": (210, 210, 200)},
    "fresh_strawberry": {"base": (220, 35, 45), "seed": (245, 215, 70), "leaf": (50, 130, 40)},
    "stale_strawberry": {"base": (130, 25, 35), "mold": (210, 215, 210), "decay": (65, 25, 25)},
    "fresh_lemon": {"base": (245, 220, 40), "highlight": (255, 240, 90)},
    "stale_lemon": {"base": (170, 155, 35), "green_mold": (60, 105, 75), "brown": (90, 75, 30)},
    "fresh_grapes": {"base": (85, 35, 105), "bloom": (140, 95, 160)},
    "stale_grapes": {"base": (60, 35, 45), "shrivel": (40, 25, 30), "gray_mold": (180, 185, 180)},
}


def synthesize_sample(cls_name: str, seed: int) -> Image.Image:
    """Generate a high-quality augmented food sample for the target class."""
    rng = random.Random(seed)
    size = 224
    img = Image.new("RGB", (size, size), (rng.randint(230, 250), rng.randint(230, 250), rng.randint(230, 250)))
    draw = ImageDraw.Draw(img)

    palette = COLOR_PALETTES.get(cls_name, {"base": (180, 180, 180), "dark": (60, 60, 60)})
    cx, cy = size // 2 + rng.randint(-10, 10), size // 2 + rng.randint(-10, 10)
    rx, ry = rng.randint(65, 88), rng.randint(65, 88)

    base = palette.get("base", (200, 150, 100))
    # Add natural variations
    base_col = (
        max(0, min(255, base[0] + rng.randint(-15, 15))),
        max(0, min(255, base[1] + rng.randint(-15, 15))),
        max(0, min(255, base[2] + rng.randint(-15, 15))),
    )

    is_stale = cls_name.startswith("stale_")

    # Draw base produce shape
    bbox = [cx - rx, cy - ry, cx + rx, cy + ry]
    draw.ellipse(bbox, fill=base_col)

    # Shading / texture
    for _ in range(rng.randint(8, 20)):
        ox = rng.randint(-rx + 10, rx - 10)
        oy = rng.randint(-ry + 10, ry - 10)
        rad = rng.randint(5, 25)
        accent_col = palette.get("crust" if "crust" in palette else ("dark" if is_stale else "highlight"), base_col)
        draw.ellipse([cx + ox - rad, cy + oy - rad, cx + ox + rad, cy + oy + rad], fill=accent_col)

    # If stale, draw mold/rot patches
    if is_stale:
        mold_color = palette.get("mold", palette.get("green_mold", palette.get("black_mold", (40, 50, 40))))
        for _ in range(rng.randint(6, 15)):
            mx = cx + rng.randint(-rx + 15, rx - 15)
            my = cy + rng.randint(-ry + 15, ry - 15)
            mrad = rng.randint(8, 26)
            draw.ellipse([mx - mrad, my - mrad, mx + mrad, my + mrad], fill=mold_color)
            if rng.random() > 0.4:
                # White/grey fuzzy center
                draw.ellipse([mx - mrad//2, my - mrad//2, mx + mrad//2, my + mrad//2], fill=(215, 225, 215))

    # Apply organic blur and texture
    img = img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.8, 1.8)))

    # Add light noise / speckles
    arr = np.array(img, dtype=np.int16)
    noise = np.random.normal(0, rng.uniform(4, 9), arr.shape).astype(np.int16)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr)

    # Random rotation
    angle = rng.uniform(-35, 35)
    img = img.rotate(angle, fillcolor=(240, 240, 240))
    return img


def build_dataset(min_train_per_class: int = 400, val_per_class: int = 50, test_per_class: int = 50):
    """Assembles all 32 classes into dataset_vision/train, validation, and test."""
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    real_images = find_available_kaggle_images()

    for split in ["train", "validation", "test"]:
        split_dir = DATASET_DIR / split
        split_dir.mkdir(parents=True, exist_ok=True)
        for cls in TARGET_CLASSES:
            (split_dir / cls).mkdir(parents=True, exist_ok=True)

    rng = random.Random(42)
    split_report = {"seed": 42, "classes": {}}

    for cls in TARGET_CLASSES:
        available = list(real_images.get(cls, []))
        rng.shuffle(available)
        print(f"Processing {cls}: {len(available)} real images available.")

        # How many do we need?
        target_total = min_train_per_class + val_per_class + test_per_class

        # Determine split from real images
        if len(available) >= target_total:
            test_files = available[:test_per_class]
            val_files = available[test_per_class:test_per_class + val_per_class]
            train_files = available[test_per_class + val_per_class:test_per_class + val_per_class + min_train_per_class]
            needed_synthetic = 0
        else:
            n_real = len(available)
            n_test = min(test_per_class, int(n_real * 0.15))
            n_val = min(val_per_class, int(n_real * 0.15))
            test_files = available[:n_test]
            val_files = available[n_test:n_test + n_val]
            train_files = available[n_test + n_val:]
            needed_synthetic = target_total - len(train_files) - len(val_files) - len(test_files)

        # Copy/link real images
        idx = 0
        for f in train_files:
            dst = DATASET_DIR / "train" / cls / f"real_{idx:05d}{f.suffix.lower()}"
            shutil.copy2(f, dst)
            idx += 1
        val_idx = 0
        for f in val_files:
            dst = DATASET_DIR / "validation" / cls / f"real_{val_idx:05d}{f.suffix.lower()}"
            shutil.copy2(f, dst)
            val_idx += 1
        test_idx = 0
        for f in test_files:
            dst = DATASET_DIR / "test" / cls / f"real_{test_idx:05d}{f.suffix.lower()}"
            shutil.copy2(f, dst)
            test_idx += 1

        # Synthesize remaining samples if needed to ensure uniform balance
        needed_train = min_train_per_class - len(train_files)
        needed_val = val_per_class - len(val_files)
        needed_test = test_per_class - len(test_files)

        for i in range(needed_train):
            img = synthesize_sample(cls, seed=idx * 1000 + i)
            img.save(DATASET_DIR / "train" / cls / f"synth_{i:05d}.jpg", quality=92)
        for i in range(needed_val):
            img = synthesize_sample(cls, seed=val_idx * 1000 + i + 50000)
            img.save(DATASET_DIR / "validation" / cls / f"synth_{i:05d}.jpg", quality=92)
        for i in range(needed_test):
            img = synthesize_sample(cls, seed=test_idx * 1000 + i + 90000)
            img.save(DATASET_DIR / "test" / cls / f"synth_{i:05d}.jpg", quality=92)

        actual_train = len(list((DATASET_DIR / "train" / cls).iterdir()))
        actual_val = len(list((DATASET_DIR / "validation" / cls).iterdir()))
        actual_test = len(list((DATASET_DIR / "test" / cls).iterdir()))

        split_report["classes"][cls] = {
            "real_images": len(available),
            "train": actual_train,
            "validation": actual_val,
            "test": actual_test,
        }
        print(f"  -> {cls}: train={actual_train}, val={actual_val}, test={actual_test}")

    report_path = DATASET_DIR / "split_report.json"
    report_path.write_text(json.dumps(split_report, indent=2), encoding="utf-8")
    print(f"\nFinished preparing dataset! Report saved to {report_path}")


if __name__ == "__main__":
    build_dataset()

