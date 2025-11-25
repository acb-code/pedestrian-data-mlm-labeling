"""
Build a packaged dataset drop under data/package/:
- Native JSONL (lossless) with image size and optional split
- COCO-style JSON for detection/segmentation consumers
- Tabular index CSV
- Symlinks to processed images for easy shipping
"""

import argparse
import csv
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple, Any

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]

ANN_IN = ROOT / "data/cleaned/annotations_fully_cleaned.jsonl"
IMG_DIR = ROOT / "data/processed/images_hashed"

OUT_DIR = ROOT / "data/package"
OUT_NATIVE = OUT_DIR / "annotations_native.jsonl"
OUT_COCO = OUT_DIR / "annotations_coco.json"
OUT_INDEX = OUT_DIR / "index.csv"
OUT_IMG_DIR = OUT_DIR / "images"


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def load_rows() -> List[Dict[str, Any]]:
    if not ANN_IN.exists():
        raise FileNotFoundError(f"Missing cleaned dataset: {ANN_IN}")
    with open(ANN_IN) as f:
        return [json.loads(line) for line in f]


def image_size(img_name: str) -> Tuple[int, int]:
    path = IMG_DIR / img_name
    if not path.exists():
        raise FileNotFoundError(f"Image not found for {img_name}: {path}")
    with Image.open(path) as im:
        return im.width, im.height


def build_category_lookup(rows: List[Dict[str, Any]]) -> Dict[str, int]:
    labels = set()
    for r in rows:
        for obj in r.get("objects", []) or []:
            if obj.get("label"):
                labels.add(obj["label"])
        for seg in r.get("segments", []) or []:
            if seg.get("label"):
                labels.add(seg["label"])
    categories = sorted(labels)
    return {name: idx + 1 for idx, name in enumerate(categories)}


def to_coco_bbox(bbox: List[Any]) -> List[float]:
    """Convert [x1, y1, x2, y2] -> [x, y, w, h]."""
    if len(bbox) != 4:
        return []
    x1, y1, x2, y2 = bbox
    return [float(x1), float(y1), float(x2) - float(x1), float(y2) - float(y1)]


def polygon_to_segmentation(poly: List[List[Any]]) -> List[float]:
    flat = []
    for x, y in poly:
        flat.extend([float(x), float(y)])
    return flat


def shoelace_area(poly: List[List[float]]) -> float:
    # Basic polygon area; expects >=3 points
    if len(poly) < 3:
        return 0.0
    area = 0.0
    for i in range(len(poly)):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % len(poly)]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def assign_splits(rows: List[Dict[str, Any]], seed: int) -> List[str]:
    """Deterministic 80/10/10 split if enabled."""
    n = len(rows)
    order = list(range(n))
    random.Random(seed).shuffle(order)
    train_cut = int(n * 0.8)
    val_cut = int(n * 0.9)
    splits = ["train"] * n
    for idx in order[train_cut:val_cut]:
        splits[idx] = "val"
    for idx in order[val_cut:]:
        splits[idx] = "test"
    return splits


def ensure_symlink(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(src)


# ---------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------

def write_native(rows: List[Dict[str, Any]], sizes: Dict[str, Tuple[int, int]], splits: Dict[str, str]):
    OUT_NATIVE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_NATIVE, "w") as f:
        for r in rows:
            img = r["image"]
            r_out = dict(r)
            w, h = sizes[img]
            r_out["width"] = w
            r_out["height"] = h
            if splits:
                r_out["split"] = splits[img]
            f.write(json.dumps(r_out) + "\n")


def write_index(rows: List[Dict[str, Any]], sizes: Dict[str, Tuple[int, int]], splits: Dict[str, str]):
    headers = [
        "image",
        "source_file",
        "width",
        "height",
        "split",
        "walkability",
        "risk_level",
        "surface",
        "nav_cue",
        "lighting",
        "density",
        "affordance_tags",
    ]
    with open(OUT_INDEX, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()
        for r in rows:
            img = r["image"]
            w, h = sizes[img]
            writer.writerow({
                "image": img,
                "source_file": r.get("source_file", ""),
                "width": w,
                "height": h,
                "split": splits.get(img, "") if splits else "",
                "walkability": r.get("walkability"),
                "risk_level": r.get("risk_level", ""),
                "surface": r.get("surface", ""),
                "nav_cue": r.get("nav_cue", ""),
                "lighting": r.get("lighting", ""),
                "density": r.get("density", ""),
                "affordance_tags": "|".join(r.get("affordance_tags") or []),
            })


def write_coco(rows: List[Dict[str, Any]], sizes: Dict[str, Tuple[int, int]], cat_lookup: Dict[str, int], splits: Dict[str, str]):
    images_json = []
    annotations_json = []
    ann_id = 1

    for img_id, r in enumerate(rows, start=1):
        img = r["image"]
        w, h = sizes[img]
        images_json.append({
            "id": img_id,
            "file_name": img,
            "width": w,
            "height": h,
            "split": splits.get(img, "") if splits else "",
        })

        for obj in r.get("objects", []) or []:
            if not obj.get("label") or not obj.get("bbox"):
                continue
            cat_id = cat_lookup.get(obj["label"])
            if not cat_id:
                continue
            coco_bbox = to_coco_bbox(obj["bbox"])
            if not coco_bbox:
                continue
            area = coco_bbox[2] * coco_bbox[3]
            annotations_json.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": cat_id,
                "bbox": coco_bbox,
                "area": area,
                "iscrowd": 0,
                "segmentation": [],
            })
            ann_id += 1

        for seg in r.get("segments", []) or []:
            if not seg.get("label") or not seg.get("polygon"):
                continue
            cat_id = cat_lookup.get(seg["label"])
            if not cat_id:
                continue
            poly = seg["polygon"]
            if len(poly) < 3:
                continue
            segmentation = [polygon_to_segmentation(poly)]
            area = shoelace_area(poly)
            annotations_json.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": cat_id,
                "bbox": [],
                "area": area,
                "iscrowd": 0,
                "segmentation": segmentation,
            })
            ann_id += 1

    categories_json = [{"id": cid, "name": name} for name, cid in sorted(cat_lookup.items(), key=lambda x: x[1])]

    with open(OUT_COCO, "w") as f:
        json.dump({
            "images": images_json,
            "annotations": annotations_json,
            "categories": categories_json,
        }, f, indent=2)


def symlink_images(rows: List[Dict[str, Any]]):
    OUT_IMG_DIR.mkdir(parents=True, exist_ok=True)
    for r in rows:
        img = r["image"]
        src = IMG_DIR / img
        dst = OUT_IMG_DIR / img
        ensure_symlink(src, dst)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main(do_split: bool, seed: int, symlink: bool):
    print("Loading cleaned dataset...")
    rows = load_rows()

    print(f"Images dir: {IMG_DIR}")
    print(f"Total rows: {len(rows)}")

    print("Reading image sizes...")
    sizes = {r["image"]: image_size(r["image"]) for r in rows}

    splits = {}
    if do_split:
        split_labels = assign_splits(rows, seed)
        splits = {r["image"]: split_labels[i] for i, r in enumerate(rows)}
        print("Assigned deterministic 80/10/10 splits.")

    print("Building categories...")
    cat_lookup = build_category_lookup(rows)
    print(f"Found {len(cat_lookup)} categories.")

    print(f"Writing native JSONL → {OUT_NATIVE}")
    write_native(rows, sizes, splits)

    print(f"Writing index CSV → {OUT_INDEX}")
    write_index(rows, sizes, splits)

    print(f"Writing COCO JSON → {OUT_COCO}")
    write_coco(rows, sizes, cat_lookup, splits)

    if symlink:
        print(f"Symlinking images → {OUT_IMG_DIR}")
        symlink_images(rows)

    print("\nDone. Outputs:")
    print(f" - {OUT_NATIVE}")
    print(f" - {OUT_INDEX}")
    print(f" - {OUT_COCO}")
    if symlink:
        print(f" - {OUT_IMG_DIR} (symlinks)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build packaged dataset artifacts.")
    parser.add_argument("--split", action="store_true", help="Create deterministic 80/10/10 splits.")
    parser.add_argument("--seed", type=int, default=13, help="RNG seed for splits.")
    parser.add_argument("--no-symlink", action="store_true", help="Skip creating image symlinks.")
    args = parser.parse_args()

    main(do_split=args.split, seed=args.seed, symlink=not args.no_symlink)
