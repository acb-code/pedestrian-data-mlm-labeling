# we want to:
# - load all the images
# - compute a stable hash for deterministic filename
# - convert to RGB, JPEG
# - resize to CLIP friendly baseline (512 px max dim)
# - save to data/dataset/images
# - record metadata to bootstrap JSONL rows

import os
import hashlib
import json
from pathlib import Path
from PIL import Image

# Resolve repo root relative to this file so paths work anywhere the repo is checked out
ROOT = Path(__file__).resolve().parent.parent

RAW_DIR = ROOT / "data/raw/images_original"
OUT_IMG_DIR = ROOT / "data/processed/images_hashed"
OUT_ANN = ROOT / "data/raw/annotations.jsonl"

OUT_IMG_DIR.mkdir(parents=True, exist_ok=True)

def hash_image(path):
    with open(path, "rb") as f:
        data = f.read()
    return hashlib.md5(data).hexdigest()[:16]  # short, stable hash

def process_image(path):
    h = hash_image(path)
    out_name = f"{h}.jpg"
    out_path = OUT_IMG_DIR / out_name

    img = Image.open(path).convert("RGB")
    img.thumbnail((512, 512))  # good default for CLIP, preserves aspect

    img.save(out_path, "JPEG", quality=95)
    return out_name

def main():
    with open(OUT_ANN, "w") as fout:
        for img_path in RAW_DIR.iterdir():
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue

            fn = process_image(img_path)
            entry = {
                "image": fn,
                "affordance_tags": [],
                "walkability": None,
                "risk_level": None,
                "surface": None,
                "nav_cue": None,
                "lighting": None,
                "density": None,
                "text_description": "",
                "source_file": img_path.name,
            }
            fout.write(json.dumps(entry) + "\n")

if __name__ == "__main__":
    main()
