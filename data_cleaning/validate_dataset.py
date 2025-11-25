"""
Dataset Validation Script
-------------------------

Validates the MLM annotation dataset for:
- JSON schema completeness
- Missing required fields
- Type errors
- Range errors
- Bounding box validation
- Scene-graph relation validation
"""

import argparse
import json
from pathlib import Path
from collections import Counter

# ---------------------------------------------------------------------
# DEFAULT PATHS (can be overridden via CLI)
# ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_INPUT = ROOT / "data/llm/annotations_llm.jsonl"
DEFAULT_REPORT = ROOT / "data/cleaned/validation_report.txt"
IMAGES_DIR = ROOT / "data/processed/images_hashed"

# ---------------------------------------------------------------------
# REQUIRED FIELDS & SCHEMA EXPECTATIONS
# ---------------------------------------------------------------------

REQUIRED_FIELDS = {
    "image": str,
    "affordance_tags": list,
    "walkability": (int, float),
    "risk_level": str,
    "surface": str,
    "nav_cue": str,
    "lighting": str,
    "density": str,
    "text_description": str,
}

# ---------------------------------------------------------------------
# VALIDATION HELPERS
# ---------------------------------------------------------------------

def load_annotations(path: Path):
    rows = []
    with open(path) as f:
        for line_num, line in enumerate(f, 1):
            try:
                rows.append(json.loads(line))
            except Exception as e:
                print(f"[JSON ERROR] Line {line_num}: {e}")
    return rows


def validate_required_fields(r, errors):
    for key, expected_type in REQUIRED_FIELDS.items():
        if key not in r:
            errors.append((r.get("image"), f"Missing field: {key}"))
            continue

        # Check for null
        if r[key] is None:
            errors.append((r.get("image"), f"Field '{key}' is null"))
            continue

        # Check for type
        if not isinstance(r[key], expected_type):
            errors.append(
                (
                    r.get("image"),
                    f"Field '{key}' expected {expected_type}, got {type(r[key])}",
                )
            )


def validate_walkability(r, errors):
    w = r.get("walkability")
    if isinstance(w, (int, float)):
        if not (0.0 <= w <= 1.0):
            errors.append((r["image"], f"walkability out of range: {w}"))
    else:
        errors.append((r["image"], f"walkability invalid type: {w}"))


def validate_bounding_boxes(r, errors):
    objects = r.get("objects")
    if not objects:
        return

    img_name = r["image"]

    for obj in objects:
        bbox = obj.get("bbox")
        if not bbox or len(bbox) != 4:
            errors.append((img_name, f"Invalid bbox format: {bbox}"))
            continue

        x1, y1, x2, y2 = bbox
        if any(v < 0 for v in [x1, y1, x2, y2]):
            errors.append((img_name, f"Negative bbox coordinates: {bbox}"))

        if x2 < x1 or y2 < y1:
            errors.append((img_name, f"Malformed bbox coordinates (x2<x1): {bbox}"))


def validate_scene_graph(r, errors):
    sg = r.get("scene_graph")
    if not sg:
        return

    for rel in sg:
        if "-" not in rel:
            errors.append((r["image"], f"Invalid scene relation syntax: {rel}"))
            continue

        parts = rel.split("-", 2)
        if len(parts) != 3:
            errors.append((r["image"], f"Scene graph triple malformed: {rel}"))


# ---------------------------------------------------------------------
# MAIN VALIDATION
# ---------------------------------------------------------------------

def main(input_path: Path, report_path: Path):
    print("Loading annotations:", input_path)
    rows = load_annotations(input_path)

    errors = []

    print("Validating dataset...")
    for r in rows:
        validate_required_fields(r, errors)
        validate_walkability(r, errors)
        validate_bounding_boxes(r, errors)
        validate_scene_graph(r, errors)

    print("\nValidation Complete.")
    print(f"Total rows: {len(rows)}")
    print(f"Total issues: {len(errors)}")

    if errors:
        print("\nSample errors:")
        for img, msg in errors[:20]:
            print(f" - {img}: {msg}")
        if len(errors) > 20:
            print(f"... {len(errors) - 20} more errors not shown")

    # error type summary
    counts = Counter(msg for (_, msg) in errors)
    print("\nError summary:")
    for msg, c in counts.items():
        print(f"  {c:3} × {msg}")

    # write report
    with open(report_path, "w") as f:
        for img, msg in errors:
            f.write(f"{img}: {msg}\n")

    print(f"\nFull report saved to: {report_path}")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate dataset annotations.")
    parser.add_argument("--input", type=str, default=str(DEFAULT_INPUT))
    parser.add_argument("--report", type=str, default=str(DEFAULT_REPORT))

    args = parser.parse_args()
    main(Path(args.input), Path(args.report))
