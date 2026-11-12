"""Central paths and settings for FoodGuard AI."""
import json
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATASET_VISION_DIR = ROOT / "dataset_vision"
DATASET_OCR_DIR = ROOT / "dataset_ocr"
MODELS_DIR = ROOT / "models"
THRESHOLDS_PATH = ROOT / "fssai_config" / "safety_thresholds.json"

SPOILAGE_WEIGHTS = MODELS_DIR / "spoilage_mobilenetv2.pt"
SPOILAGE_CLASSES = MODELS_DIR / "class_names.json"

CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
IMAGE_SIZE = 224

# Produce the spoilage CNN was trained on (folder names are fresh_<type> / stale_<type>).
SUPPORTED_PRODUCE = [
    "apple", "banana", "bitter_gourd", "bread", "capsicum", "carrot",
    "cucumber", "grapes", "lemon", "mango", "onion", "orange",
    "paneer_milk", "potato", "strawberry", "tomato"
]

CATEGORIES = {
    "raw_produce": "Raw produce & perishables (fruits, vegetables, bread, dairy)",
    "packaged_solid": "Packaged food (snacks, noodles, biscuits, ready-to-eat)",
    "packaged_liquid": "Packaged beverages (juices, sodas, dairy drinks)",
    "packaged_exempt": "Packaged staples (oils, ghee, honey, dry fruits)",
}
PACKAGED_CATEGORIES = ("packaged_solid", "packaged_liquid", "packaged_exempt")

OPEN_FOOD_FACTS_URL = "https://world.openfoodfacts.org"
OPEN_FOOD_FACTS_SEARCH_URL = "https://search.openfoodfacts.org/search"
HTTP_USER_AGENT = "FoodGuardAI/1.0 (academic final-year project)"
HTTP_TIMEOUT_S = 8


@lru_cache(maxsize=1)
def load_thresholds() -> dict:
    with open(THRESHOLDS_PATH, encoding="utf-8") as f:
        return json.load(f)
