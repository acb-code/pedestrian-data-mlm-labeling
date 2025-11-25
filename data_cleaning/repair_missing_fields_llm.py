"""
repair_missing_fields_llm.py
----------------------------

Automatically repairs missing or null fields in annotations using Gemini.
Operates on the output of clean_scene_graph.py or clean_scene_graph_llm.py.

Repairs missing:
- walkability               (float)
- risk_level                (category)
- surface                   (category)
- nav_cue                   (string)
- lighting                  (category)
- density                   (category)
- text_description          (string)

The script:
1. Loads annotations
2. Identifies rows with missing/null fields
3. Builds a batch LLM request containing image + existing fields
4. Requests only missing fields to minimize cost
5. Applies repairs
6. Saves cleaned dataset

Uses the processed JPEG-image bytes in data/processed/images_hashed/.
"""

import json
from pathlib import Path
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()
client = genai.Client()

ROOT = Path(__file__).resolve().parents[1]

# INPUT: scene-graph-cleaned annotations
IN_PATH = ROOT / "data/cleaned/annotations_cleaned_scene_graph_llm.jsonl"

# OUTPUT: fully repaired annotations
OUT_PATH = ROOT / "data/cleaned/annotations_fully_cleaned.jsonl"

IMG_DIR = ROOT / "data/processed/images_hashed"

REQUIRED = [
    "walkability",
    "risk_level",
    "surface",
    "nav_cue",
    "lighting",
    "density",
    "text_description",
]

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def load_annotations():
    rows = []
    with open(IN_PATH) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def load_image_bytes(path):
    with open(path, "rb") as f:
        return f.read()


def needs_repair(row):
    """Return list of required fields that are missing, null, or empty."""
    missing = []

    for key in REQUIRED:
        val = row.get(key)

        # Case 1: completely missing or null
        if val is None:
            missing.append(key)
            continue

        # Case 2: empty string or whitespace
        if isinstance(val, str) and val.strip() == "":
            missing.append(key)
            continue

        # Case 3: empty list (e.g. [])
        if isinstance(val, list) and len(val) == 0:
            missing.append(key)
            continue

    return missing


def build_prompt(row, missing_fields):
    known = {k: row.get(k) for k in REQUIRED if k not in missing_fields}

    return f"""
You are repairing ONLY the missing fields in a pedestrian navigation annotation.

Return ONLY JSON. No explanation.

CRITICAL RULES:
- Return values ONLY for the missing fields: {missing_fields}
- Do NOT return or modify any other fields.
- Never return null.
- Never return an empty string.
- Never return an empty list.
- Required types:
    - walkability: float between 0.0 and 1.0
    - affordance_tags: list[str]
    - risk_level/surface/nav_cue/lighting/density: str
    - text_description: str
- Do NOT return descriptive text in numeric fields.
- Do NOT convert walkability into a sentence.

Known (already correct) fields:
{json.dumps(known, indent=2)}
"""


def call_gemini(image_bytes, prompt, img_name):
    """Call Gemini to repair missing fields."""
    resp = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
            prompt,
        ],
        config=types.GenerateContentConfig(response_mime_type="application/json")
    )

    try:
        return json.loads(resp.text)
    except Exception:
        print(f"[ERROR] Invalid JSON for image {img_name}. Raw output:")
        print(resp.text)
        return None


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():
    print(f"Loading: {IN_PATH}")
    rows = load_annotations()

    to_fix = [(i, needs_repair(r)) for i, r in enumerate(rows)]
    to_fix = [(i, m) if m else None for i, m in to_fix]
    to_fix = [x for x in to_fix if x is not None]

    print(f"Images needing repair: {len(to_fix)}")

    for idx, missing in to_fix:
        row = rows[idx]
        img_name = row["image"]
        print(f"→ Repairing {img_name}, missing: {missing}")

        img_path = IMG_DIR / img_name
        if not img_path.exists():
            print(f"  !!! Missing image file: {img_path}")
            continue

        image_bytes = load_image_bytes(img_path)
        prompt = build_prompt(row, missing)

        repaired = call_gemini(image_bytes, prompt, img_name)
        if not repaired:
            print(f"  !!! Repair failed for {img_name}")
            continue

        # apply repaired values
        for k, v in repaired.items():
            # Enforce type safety
            if k == "walkability":
                try:
                    v = float(v)
                except:
                    print(f"  !!! Invalid walkability repair for {row['image']}: {v}")
                    continue

            if k in ["affordance_tags"] and not isinstance(v, list):
                print(f"  !!! Invalid tag list repair for {row['image']}: {v}")
                continue

            # Normal assignment
            row[k] = v
        
        # Second-pass check: ensure no fields remain empty
        remaining = needs_repair(row)
        if remaining:
            print(f"  !!! Warning: still missing after repair for {img_name}: {remaining}")
        
        # Optional fallback for text_description
        if "text_description" in remaining:
            row["text_description"] = (
                f"A pedestrian navigation scene with {row.get('surface', 'ground surface')}."
            )
            print(f"    → Applied fallback text_description for {img_name}")
        
        # detect wrong-type walkability explicitly
        if isinstance(row.get("walkability"), str):
            print(f"  !!! walkability wrong type for {row['image']}: {row['walkability']}")
            remaining.append("walkability")

    # Save final cleaned dataset
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUT_PATH, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    print(f"\nAll missing-field repairs complete.")
    print(f"Final dataset saved to: {OUT_PATH}")


if __name__ == "__main__":
    main()