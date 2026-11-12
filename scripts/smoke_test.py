"""Quick smoke-test for all four FoodGuard pipelines.

Usage:
    python scripts/smoke_test.py

Sends one real request to each category using the uploaded test images.
Any 4xx/5xx response or Python exception is printed as a FAIL.
"""
import io, sys, traceback, json
from pathlib import Path
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ── Build a tiny synthetic test image ──────────────────────────────────────
def _make_produce_img():
    """Red circle (tomato-ish) on white."""
    img = Image.new("RGB", (300, 300), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.ellipse((50, 50, 250, 250), fill=(210, 40, 40))
    return img

def _make_label_img():
    """White image with black 'Nutrition Facts' text to pass the text-box guard."""
    from PIL import ImageFont
    img = Image.new("RGB", (500, 700), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    rows = [
        "Nutrition Information",
        "Serving size 25g",
        "Energy 565kJ  2260kJ",
        "Protein 1.5g  5.9g",
        "Fat - Total 8.5g  33.9g",
        "  - Saturated 3.9g  15.5g",
        "  - Trans 0.0g",
        "Carbohydrate 13.3g  53.1g",
        "  - Sugars 1.0g  3.9g",
        "Dietary Fibre 0.7g  2.7g",
        "Sodium 193mg  770mg",
    ]
    y = 20
    for row in rows:
        draw.text((20, y), row, fill=(0, 0, 0))
        y += 55
    return img

def _img_to_bytes(img: Image.Image, fmt="JPEG") -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()

# ── Drive Flask test client ─────────────────────────────────────────────────
import app as flask_app          # imports the Flask application

client = flask_app.app.test_client()

TESTS = [
    {
        "name": "raw_produce / tomato (synthetic)",
        "endpoint": "/api/analyze",
        "data": {"category": "raw_produce"},
        "image_key": "images",
        "image": _make_produce_img(),
    },
    {
        "name": "packaged_solid / chips label (synthetic)",
        "endpoint": "/api/analyze",
        "data": {"category": "packaged_solid"},
        "image_key": "images",
        "image": _make_label_img(),
    },
    {
        "name": "packaged_liquid / drink label (synthetic)",
        "endpoint": "/api/analyze",
        "data": {"category": "packaged_liquid"},
        "image_key": "images",
        "image": _make_label_img(),
    },
    {
        "name": "packaged_exempt / oil label (synthetic)",
        "endpoint": "/api/analyze",
        "data": {"category": "packaged_exempt"},
        "image_key": "images",
        "image": _make_label_img(),
    },
    {
        "name": "health endpoint",
        "endpoint": "/api/health",
        "data": None,
        "image_key": None,
        "image": None,
    },
]

passed = failed = 0
for test in TESTS:
    try:
        if test["data"] is None:
            resp = client.get(test["endpoint"])
        else:
            img_bytes = _img_to_bytes(test["image"])
            data = dict(test["data"])
            resp = client.post(
                test["endpoint"],
                data={**data, test["image_key"]: (io.BytesIO(img_bytes), "test.jpg")},
                content_type="multipart/form-data",
            )
        body = resp.get_json() or {}
        status = resp.status_code
        verdict = body.get("verdict", {}).get("status") or body.get("status") or body.get("error") or "?"
        if 200 <= status < 300:
            print(f"  ✅ PASS  {test['name']:50s}  HTTP {status}  verdict={verdict}")
            passed += 1
        else:
            print(f"  ❌ FAIL  {test['name']:50s}  HTTP {status}  body={json.dumps(body)[:200]}")
            failed += 1
    except Exception:
        print(f"  ❌ FAIL  {test['name']:50s}  EXCEPTION:")
        traceback.print_exc()
        failed += 1

print(f"\n{'='*60}")
print(f"Results: {passed} passed, {failed} failed")
sys.exit(0 if failed == 0 else 1)
