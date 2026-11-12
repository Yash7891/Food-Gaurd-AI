"""FoodGuard AI - Flask web server.

    python app.py            -> http://127.0.0.1:5000

Endpoints
    GET  /              web interface
    POST /api/analyze   multipart form: images (1-3 files), category, product_name?, barcode?,
                        expiry_date? (YYYY-MM-DD, optional)
    GET  /api/health    model / device status
"""
import logging
import threading
from datetime import date

from flask import Flask, jsonify, render_template, request
from PIL import Image, ImageOps, UnidentifiedImageError, ImageDraw

from foodguard import config, models, pipeline

# Imports for multi‑item analysis
from scripts.detect_produce import detect_items
import io
import base64

ALLOWED_EXT = {"jpg", "jpeg", "png", "webp", "bmp"}
MAX_IMAGES = 3

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 30 * 1024 * 1024
log = logging.getLogger("foodguard")

# EasyOCR and the CUDA models are not guaranteed thread-safe: one analysis at a time.
_analysis_lock = threading.Lock()


def _load_image(file) -> Image.Image:
    """Load an uploaded image, enforce allowed extensions, and normalize it to RGB.
    """
    ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if ext not in ALLOWED_EXT:
        raise ValueError(f"'{file.filename}' is not a supported image type ({', '.join(sorted(ALLOWED_EXT))}).")
    try:
        img = Image.open(file.stream)
        img.load()
    except (UnidentifiedImageError, OSError):
        raise ValueError(f"'{file.filename}' could not be read as an image.")
    img = ImageOps.exif_transpose(img)          # phone photos carry rotation in EXIF
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        background = Image.new("RGB", img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[-1])
        img = background
    # Ensure RGB and log basic info for debugging
    img = img.convert("RGB")
    log.debug("Loaded image %s size=%s mode=%s", file.filename, img.size, img.mode)
    return img


@app.get("/")
def index():
    return render_template("index.html", categories=config.CATEGORIES,
                           supported=[p.replace("_", " ") for p in config.SUPPORTED_PRODUCE])


@app.post("/api/analyze")
def analyze():
    files = [f for f in request.files.getlist("images") if f and f.filename]
    category = request.form.get("category", "")
    if not files:
        return jsonify(error="Please upload at least one image."), 400
    if len(files) > MAX_IMAGES:
        return jsonify(error=f"Upload at most {MAX_IMAGES} images."), 400
    if category not in config.CATEGORIES:
        return jsonify(error="Please choose a food category."), 400
    try:
        images = [_load_image(f) for f in files]
    except ValueError as exc:
        return jsonify(error=str(exc)), 400
    expiry_raw = (request.form.get("expiry_date") or "").strip()
    try:
        expiry_date = date.fromisoformat(expiry_raw) if expiry_raw else None
    except ValueError:
        return jsonify(error="Expiry date must be a valid date (YYYY-MM-DD)."), 400

    with _analysis_lock:
        try:
            result = pipeline.analyse(images, category,
                                      product_name=request.form.get("product_name"),
                                      barcode=request.form.get("barcode"),
                                      expiry_date=expiry_date)
        except Exception:  # noqa: BLE001 - report any model failure to the client
            log.exception("Analysis failed")
            return jsonify(error="Analysis failed on the server - see the console log."), 500
    # ── Post-model spoilage patch (raw produce only) ───────────────────────────
    # Only applies to raw produce; packaged food has no surface_analysis.
    # Only override when model is NOT confident SAFE AND patches are very heavy.
    produce = result.get("produce") or {}  # None for packaged food → safe default
    surface = produce.get("surface_analysis") or {}
    dark_percent = surface.get("dark_patch_percent") or 0
    model_status = result.get("verdict", {}).get("status", result.get("status", ""))

    # Only escalate for raw produce with heavy dark coverage (>25 %) that the
    # model is already uncertain about.  Never fires for packaged food.
    if produce and model_status != "SAFE" and dark_percent > 25:
        verdict = result.setdefault("verdict", {})
        verdict["headline"] = "HIGH SPOILAGE DETECTED"
        verdict["reason"] = f"Dark / discoloured patches cover {dark_percent:.1f}% of the visible surface."
        result["rating"] = "Poor"
        result["grade"] = "F"
        result["status"] = "UNSAFE"
        verdict["status"] = "UNSAFE"
        result["score"] = max(0, result.get("score", 0) - 30)
        log.info("Spoilage override: dark_patch_percent=%.1f model_status=%s", dark_percent, model_status)

    return jsonify(result)

# ---------------------------------------------------------------------
# Multi‑item analysis endpoint
# ---------------------------------------------------------------------

@app.post("/api/analyze_multi")
def analyze_multi():
    """Accept one image that may contain multiple produce items.
    Returns a list of per‑item predictions together with an annotated image.
    """
    file = request.files.get("image")
    if not file:
        return jsonify(error="Please upload an image file under the key 'image'."), 400
    try:
        pil_image = _load_image(file)
    except ValueError as exc:
        return jsonify(error=str(exc)), 400

    # 1️⃣ Detect bounding boxes for each item
    boxes = detect_items(pil_image)
    if not boxes:
        # Fallback: treat the whole image as a single item and run spoilage analysis
        try:
            single_res = pipeline.analyse([pil_image], request.form.get("category", "raw_produce"),
                                          product_name=request.form.get("product_name"),
                                          barcode=request.form.get("barcode"),
                                          expiry_date=None)
        except Exception as exc:
            log.exception("Fallback analysis failed")
            return jsonify(error="Internal error during fallback analysis."), 500
        # Apply same spoilage heuristics as in single-image endpoint
        produce = single_res.get("produce", {})
        surface = produce.get("surface_analysis", {})
        dark_percent = surface.get("dark_patch_percent", 0)
        if dark_percent and dark_percent > 4:
            single_res.setdefault("verdict", {})["headline"] = "HIGH SPOILAGE DETECTED"
            single_res["verdict"]["reason"] = f"Dark patches cover {dark_percent:.1f}% of the surface, indicating spoilage."
            single_res["rating"] = "Poor"
            single_res["score"] = max(0, single_res.get("score", 0) - 30)
        # Return single-item result in same format as multi
        return jsonify({
            "detections": [{
                "bbox": [0, 0, pil_image.width, pil_image.height],
                "produce_type": produce.get("produce_type", "unknown"),
                "freshness": "fresh" if produce.get("fresh_probability",0) >= produce.get("stale_probability",0) else "stale",
                "confidence": max(produce.get("fresh_probability",0), produce.get("stale_probability",0)),
                "detail": produce,
            }],
            "summary": {"message": "Fallback analysis performed on whole image."},
            "annotated_image": None,
        })

    # 2️⃣ Run the existing spoilage classifier on each crop
    results = []
    draw = ImageDraw.Draw(pil_image)
    for i, box in enumerate(boxes):
        left, top, right, bottom = box
        crop = pil_image.crop((left, top, right, bottom))
        try:
            single_res = pipeline.analyse([crop], request.form.get("category", "raw_produce"),
                                          product_name=request.form.get("product_name"),
                                          barcode=request.form.get("barcode"),
                                          expiry_date=None)
        except Exception:
            log.exception("Per‑item analysis failed")
            return jsonify(error="Internal error during per‑item analysis."), 500
        produce = single_res.get("produce", {})
        freshness = "fresh" if produce.get("fresh_probability", 0) >= produce.get("stale_probability", 0) else "stale"
        colour = "green" if freshness == "fresh" else "red"
        draw.rectangle(box, outline=colour, width=3)
        draw.text((left, top - 10), f"{produce.get('produce_type','?')} ({freshness})", fill=colour)
        results.append({
            "bbox": box,
            "produce_type": produce.get("produce_type"),
            "freshness": freshness,
            "confidence": max(produce.get("fresh_probability", 0), produce.get("stale_probability", 0)),
            "detail": produce,
        })

    # 3️⃣ Encode annotated image as base64
    buf = io.BytesIO()
    pil_image.save(buf, format="JPEG")
    annotated_b64 = base64.b64encode(buf.getvalue()).decode()

    # Build a concise batch summary (counts per produce type)
    from scripts.aggregate import summarise_detections
    batch_summary = summarise_detections(results)
    return jsonify({
        "detections": results,
        "summary": batch_summary,
        "annotated_image": annotated_b64,
    })


@app.get("/api/health")
def health():
    return jsonify(models.status())


@app.errorhandler(413)
def too_large(_e):
    return jsonify(error="Upload too large (max 30 MB in total)."), 413


def _warm_up():
    """Load all models in the background so the first request is fast."""
    try:
        models.get_clip()
        models.get_ocr_reader()
        models.get_spoilage_model()
        log.info("All models loaded on %s", models.DEVICE)
    except models.ModelNotTrainedError as exc:
        log.warning("%s", exc)
    except Exception:  # noqa: BLE001
        log.exception("Model warm-up failed")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for noisy in ("httpx", "huggingface_hub", "werkzeug"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    print(" * FoodGuard AI: open http://127.0.0.1:5000  (models load in the background)")
    threading.Thread(target=_warm_up, daemon=True).start()
    app.run(host="127.0.0.1", port=5000, threaded=True, debug=False)
