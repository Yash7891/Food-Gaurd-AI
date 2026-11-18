# -------------------------------------------------
# download_datasets.py – works online *and* offline
# -------------------------------------------------
import os
import sys
import json
import zipfile
import shutil
from pathlib import Path

# tqdm is optional – if not installed we fall back to a simple loop
try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kw: x

# ------------------------------------------------------------------
# CONFIGURATION (you can adjust the URLs or add more datasets here)
# ------------------------------------------------------------------
DATA_ROOT   = Path(__file__).resolve().parents[1] / "data"
RAW_DIR     = DATA_ROOT / "raw"
SAMPLE_ZIP  = Path(__file__).resolve().parent / "sample_data" / "fruit_sample.zip"

# Example public URLs – replace with the ones you actually use
DATASETS = [
    {
        "name": "fruit_images_kaggle",
        "url":  "https://example.com/nonexistent/fruit_images.zip",  # placeholder – will fail when offline
        "filename": "fruit_images.zip"
    },
    # add more dictionaries here if you have other sources
]

# ------------------------------------------------------------------
# Helper: download a single file with a progress bar
# ------------------------------------------------------------------
def _download(url: str, dest: Path) -> bool:
    """Return True on success, False on any network error."""
    import requests
    try:
        with requests.get(url, stream=True, timeout=10) as r:
            r.raise_for_status()
            total = int(r.headers.get('content-length', 0))
            dest.parent.mkdir(parents=True, exist_ok=True)
            with open(dest, "wb") as f:
                for chunk in tqdm(r.iter_content(chunk_size=8192),
                                 total=total // 8192 + 1,
                                 unit='KB', desc=f"Downloading {dest.name}"):
                    if chunk:
                        f.write(chunk)
        return True
    except Exception as e:      # any network problem ends up here
        print(f"[WARN] Could not download {url}: {e}")
        return False

# ------------------------------------------------------------------
# Helper: unzip a .zip file into a target folder
# ------------------------------------------------------------------
def _unzip(zip_path: Path, target_dir: Path):
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(target_dir)

# ------------------------------------------------------------------
# Helper: make sure a tiny sample dataset exists (offline first‑run)
# ------------------------------------------------------------------
def ensure_sample_dataset():
    """
    If RAW_DIR is empty, unpack the small sample zip that lives inside
    the repository (sample_data/fruit_sample.zip). This guarantees that
    the rest of the pipeline always has at least a few images.
    """
    if any(RAW_DIR.iterdir()):          # already has data → nothing to do
        return

    print("[INFO] No raw data found locally – extracting bundled sample dataset...")
    try:
        _unzip(SAMPLE_ZIP, RAW_DIR)
        print(f"[INFO] Extracted sample data ({len(list(RAW_DIR.iterdir()))} files).")
    except Exception as exc:
        print(f"[ERROR] Failed to unpack sample dataset: {exc}")
        sys.exit(1)

# ------------------------------------------------------------------
# MAIN logic – try online download, fall back to offline mode
# ------------------------------------------------------------------
def main():
    # 1️⃣  Ensure the raw folder exists
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # 2️⃣  Try each remote dataset – stop after the first successful download
    downloaded_any = False
    for ds in DATASETS:
        zip_path = RAW_DIR / ds["filename"]
        print(f"[INFO] Attempting to download '{ds['name']}' ...")
        success = _download(ds["url"], zip_path)
        if success:
            print(f"[INFO] Unzipping {zip_path.name} …")
            _unzip(zip_path, RAW_DIR)
            # optional: keep the zip for later reuse – or delete it
            # zip_path.unlink()   # uncomment to delete the archive after unzip
            downloaded_any = True
            break   # we got data, no need to try other URLs

    # 3️⃣  If no download succeeded (offline or URL failed), use the sample
    if not downloaded_any:
        print("[WARN] Could not download any remote dataset – switching to offline mode.")
        ensure_sample_dataset()
    else:
        print("[INFO] Dataset downloaded and ready in:", RAW_DIR)

    # 4️⃣  (Optional) write a tiny JSON manifest for downstream scripts
    manifest = {
        "raw_dir": str(RAW_DIR),
        "source": "online" if downloaded_any else "offline-sample"
    }
    (DATA_ROOT / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2))

if __name__ == "__main__":
    main()
