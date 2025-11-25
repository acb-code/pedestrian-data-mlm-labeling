import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import yaml
from tqdm.auto import tqdm

from google import genai
from google.genai import types
from google.genai.errors import APIError, ServerError

from dotenv import load_dotenv
load_dotenv()
client = genai.Client()

# ------------------------------------------------------------------------------
# PATH SETUP
# ------------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent  # repo root

# Base dataset (bootstrap annotations created by import_images.py)
DATASET_PATH = ROOT / "data/raw/annotations.jsonl"

# Processed images used for LLM labeling
IMAGES_DIR = ROOT / "data/processed/images_hashed"

# Category definitions
CATEGORIES_PATH = ROOT / "image_annotating/categories.yaml"

# LLM output paths
OUT_PATH = ROOT / "data/llm/annotations_llm.jsonl"
OUT_TMP = ROOT / "data/llm/annotations_llm.jsonl.tmp"

# Raw LLM responses for debugging
RAW_RESP_DIR = ROOT / "data/llm/raw_llm_responses"
RAW_RESP_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------------------
# PROMPT
# ------------------------------------------------------------------------------

PROMPT_TEMPLATE = """
You are labeling affordances for pedestrian navigation scenes.

Your output MUST be a single valid JSON OBJECT that matches EXACTLY the schema below.
Every field MUST be present, MUST follow the type, and MUST NOT be null or empty.

Return ONLY this JSON object. No explanation, no prose.

{{
  "affordance_tags": list[str],        // Allowed values: {affordance_tags}
  "walkability": float,                // Strictly between 0.0 and 1.0
  "risk_level": str,                   // Allowed: {risk_level}
  "surface": str,                      // Allowed: {surface}
  "nav_cue": str,                      // Non-empty description
  "lighting": str,                     // Allowed: {lighting}
  "density": str,                      // Allowed: {density}
  "text_description": str,             // 1–3 sentences, concise and factual

  "objects": [
      {{
        "label": str,
        "bbox": [x_min, y_min, x_max, y_max],   // exactly 4 positive integers
        "confidence": float                     // between 0.0 and 1.0
      }},
      ...
  ],

  "segments": [
      {{
        "label": str,
        "polygon": [[x, y], [x, y], ...],       // >= 3 points; each x,y are integers
        "confidence": float                     // between 0.0 and 1.0
      }},
      ...
  ],

  "scene_graph": [
      "subject - relation - object",
      ...
  ],

  "affordance_conflicts": list[str]
}}

===========================
STRICT RULES — READ CAREFULLY
===========================

### 1. JSON STRUCTURE
- You MUST return **valid, parseable JSON**.
- Do NOT include comments, trailing commas, or text outside the JSON object.
- All fields must be present and populated.

### 2. CONTROLLED VOCABULARIES
Use ONLY allowed values for:
- affordance_tags: {affordance_tags}
- risk_level: {risk_level}
- surface: {surface}
- lighting: {lighting}
- density: {density}

No synonyms, no new categories.

### 3. WALKABILITY
- MUST be a float between 0.0 and 1.0.
- Never null or empty.

### 4. OBJECT DETECTION (CRITICAL)
Your bounding boxes MUST follow this exact format:

"bbox": [x_min, y_min, x_max, y_max]

Where:
- All 4 values are integers.
- x_min < x_max and y_min < y_max.
- No additional numbers. Exactly 4.
- Values must be consistent with the image size you are given.
- If uncertain about an object, **do not guess**: omit it.

### 5. SEGMENTATION
"polygon" MUST be:
- a list of [x, y] integer pairs
- at least 3 points
- closed implicitly (do not duplicate the first point)

### 6. SCENE GRAPH RULES
Each entry MUST be a triplet in exactly this format:

"subject - relation - object"

Where:
- subject and object are nouns or noun phrases (no underscores)
- relation is a verb or prepositional phrase (no underscores)
- DO NOT embed relations into labels
- Convert phrases like “sidewalk_covered_with_leaves” into:
  "sidewalk - covered with - leaves"

If uncertain, return [].

### 7. AFFORDANCE CONFLICTS
- List textual descriptions of conflicts (e.g., "turn_left blocked by car")
- If none, return [].

### 8. OUTPUT REQUIREMENTS
- All fields MUST be populated.
- No empty strings.
- No nulls.
- No extra keys.
- No speculative or contradictory content.
- Return ONLY a JSON object matching this schema.

Begin.
"""

REQUIRED_KEYS = [
    "affordance_tags",
    "walkability",
    "risk_level",
    "surface",
    "nav_cue",
    "lighting",
    "density",
    "text_description",
    "objects",
    "segments",
    "scene_graph",
    "affordance_conflicts",
]


# ------------------------------------------------------------------------------
# Helpers: IO & prompt
# ------------------------------------------------------------------------------

def load_image_bytes(path: Path) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def build_prompt(categories: Dict[str, Any]) -> str:
    return PROMPT_TEMPLATE.format(
        affordance_tags=categories["affordance_tags"],
        risk_level=categories["risk_level"],
        surface=categories["surface"],
        lighting=categories["lighting"],
        density=categories["density"],
    )


def save_raw_response(img_name: str, response_text: str) -> None:
    """Store the raw Gemini output (even if it's malformed JSON)."""
    out_file = RAW_RESP_DIR / f"{img_name}.json"
    with open(out_file, "w") as f:
        f.write(response_text)


# ------------------------------------------------------------------------------
# Helpers: validation & hallucination detection
# ------------------------------------------------------------------------------

def validate_output_schema(
    output: Dict[str, Any],
    categories: Dict[str, List[str]],
) -> Dict[str, Any]:
    """
    Check that the LLM output:
    - has required keys
    - uses only allowed category values
    - has correct types
    - uses walkability in [0, 1]

    Returns a dict:
        { "ok": bool, "errors": [...], "warnings": [...], "extra_keys": [...] }
    """
    errors: List[str] = []
    warnings: List[str] = []

    keys = set(output.keys())
    missing = [k for k in REQUIRED_KEYS if k not in keys]
    if missing:
        errors.append(f"Missing keys: {missing}")

    extra = list(keys - set(REQUIRED_KEYS))
    if extra:
        # This is where we flag possible hallucinated fields
        warnings.append(f"Extra keys present (possible hallucinations): {extra}")

    # Only validate further if the keys exist
    if "affordance_tags" in output:
        if not isinstance(output["affordance_tags"], list):
            errors.append("affordance_tags must be a list")
        else:
            allowed = set(categories["affordance_tags"])
            bad_tags = [t for t in output["affordance_tags"] if t not in allowed]
            if bad_tags:
                errors.append(f"Invalid affordance_tags: {bad_tags}")

    if "walkability" in output:
        try:
            w = float(output["walkability"])
            if not (0.0 <= w <= 1.0):
                errors.append(f"walkability out of range [0,1]: {w}")
        except Exception:
            errors.append(f"walkability not a float: {output['walkability']}")

    def _check_cat(key: str):
        if key not in output:
            return
        val = output[key]
        allowed = set(categories[key])
        if val not in allowed:
            errors.append(f"{key} has invalid value '{val}', allowed={sorted(allowed)}")

    for cat_key in ["risk_level", "surface", "lighting", "density"]:
        _check_cat(cat_key)

    if "text_description" in output and not isinstance(output["text_description"], str):
        errors.append("text_description must be a string")

    # soft checks on objects/segments/scene_graph/affordance_conflicts
    for list_key in ["objects", "segments", "scene_graph", "affordance_conflicts"]:
        if list_key in output and not isinstance(output[list_key], list):
            warnings.append(f"{list_key} should be a list")

    return {
        "ok": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "extra_keys": extra,
    }


# ------------------------------------------------------------------------------
# Helpers: embeddings & similarity between runs
# ------------------------------------------------------------------------------

def cosine_similarity(v1: List[float], v2: List[float]) -> Optional[float]:
    if not v1 or not v2 or len(v1) != len(v2):
        return None
    dot = sum(a * b for a, b in zip(v1, v2))
    n1 = math.sqrt(sum(a * a for a in v1))
    n2 = math.sqrt(sum(b * b for b in v2))
    if n1 == 0 or n2 == 0:
        return None
    return dot / (n1 * n2)


def get_text_embedding(text: str) -> Optional[List[float]]:
    """
    Use Gemini embeddings; keep simple (no extra config).
    This uses the text-embedding model; see docs if you want a different one.
    """
    if not text:
        return None

    try:
        resp = client.models.embed_content(
            model="gemini-embedding-001",
            contents=text,
        )
        # resp.embeddings is a list of ContentEmbedding
        emb = resp.embeddings[0].values
        return list(emb)
    except Exception as e:
        print(f"    ✗ Embedding error: {e}")
        return None


def compute_description_similarity(
    prev_desc: str, new_desc: str
) -> Optional[float]:
    prev_emb = get_text_embedding(prev_desc)
    new_emb = get_text_embedding(new_desc)
    if prev_emb is None or new_emb is None:
        return None
    return cosine_similarity(prev_emb, new_emb)


# ------------------------------------------------------------------------------
# Gemini call with retries
# ------------------------------------------------------------------------------

def call_gemini_with_retry(
    image_bytes: bytes,
    prompt: str,
    img_name: str,
    max_retries: int = 5,
) -> Optional[Dict[str, Any]]:
    """Call Gemini with retry, logging, backoff, and raw-response storage."""
    for attempt in range(1, max_retries + 1):
        try:
            print(f"  → Gemini call attempt {attempt} for {img_name}")

            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[
                    types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                    prompt,
                ],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json"
                ),
            )

            # Store raw response for debugging, even if JSON fails later.
            save_raw_response(img_name, response.text)

            return json.loads(response.text)

        except (ServerError, APIError) as e:
            print(f"    ✗ API error (attempt {attempt}): {type(e).__name__}: {e}")

            if attempt == max_retries:
                print("    !!! Max retries reached, skipping image.")
                return None

            delay = random.uniform(2.0, 5.0)
            print(f"    → Backing off for {delay:.1f}s...")
            time.sleep(delay)

        except json.JSONDecodeError:
            print(f"    ✗ JSON parse error for {img_name}")
            delay = random.uniform(1.0, 3.0)
            print(f"    → Sleeping {delay:.1f}s and retrying JSON mode...")
            time.sleep(delay)

        except Exception as e:
            print(f"    ✗ Unexpected error on attempt {attempt}: {e}")
            if attempt == max_retries:
                return None
            delay = random.uniform(2.0, 5.0)
            print(f"    → Backing off for {delay:.1f}s...")
            time.sleep(delay)

    return None


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------

def main(force: bool = False, compute_embed_sim: bool = False) -> None:
    print(f"Loading categories from: {CATEGORIES_PATH}")
    categories = yaml.safe_load(open(CATEGORIES_PATH))

    prompt = build_prompt(categories)

    print(f"Reading base dataset: {DATASET_PATH}")
    base_lines = list(open(DATASET_PATH))

    # ------------------------------------------------------------------
    # Load previous annotations (for resume and/or similarity checking)
    # ------------------------------------------------------------------
    previous_by_image: Dict[str, Dict[str, Any]] = {}
    if OUT_PATH.exists():
        print(f"Found existing {OUT_PATH}, loading for resume/comparison.")
        with open(OUT_PATH) as f:
            for line in f:
                obj = json.loads(line)
                previous_by_image[obj["image"]] = obj
        print(f"Loaded {len(previous_by_image)} previous entries.")
    else:
        print("No previous annotations file found.")

    # ------------------------------------------------------------------
    # Decide resume behavior
    # ------------------------------------------------------------------
    if force:
        print("Starting fresh with --force (re-labeling all images).")
        already_labeled: Dict[str, Dict[str, Any]] = {}
    else:
        print("Resume mode: will skip images already labeled.")
        already_labeled = previous_by_image

    # Open tmp file for atomic write
    fout = open(OUT_TMP, "w")

    # If resuming (not force), re-write previously labeled entries first
    if not force and previous_by_image:
        print("Rewriting existing annotations into tmp file...")
        for obj in previous_by_image.values():
            fout.write(json.dumps(obj) + "\n")

    print(f"\nTotal images in dataset: {len(base_lines)}\n")

    # ------------------------------------------------------------------
    # Process dataset
    # ------------------------------------------------------------------
    for idx, line in enumerate(tqdm(base_lines, desc="Annotating images"), start=1):
        entry = json.loads(line)
        img_name = entry["image"]

        print(f"\n[{idx}/{len(base_lines)}] {img_name}")

        # Skip already labeled in resume mode
        if not force and img_name in already_labeled:
            print("  → Already labeled — skipping.")
            continue

        img_path = IMAGES_DIR / img_name
        if not img_path.exists():
            print(f"  !!! Image not found: {img_path}, skipping.")
            continue

        image_bytes = load_image_bytes(img_path)

        llm_output = call_gemini_with_retry(
            image_bytes=image_bytes,
            prompt=prompt,
            img_name=img_name,
            max_retries=5,
        )

        if llm_output is None:
            print(f"  !!! Failed to label {img_name}, writing bare entry.")
            fout.write(json.dumps(entry) + "\n")
            continue

        # Validate and detect hallucinations
        validation = validate_output_schema(llm_output, categories)
        if not validation["ok"]:
            print(f"  !!! Validation errors for {img_name}:")
            for e in validation["errors"]:
                print(f"      - {e}")
        if validation["warnings"]:
            print(f"  ⚠  Warnings for {img_name}:")
            for w in validation["warnings"]:
                print(f"      - {w}")

        entry.update(llm_output)
        entry["_validation"] = validation

        # Optional: embedding similarity vs previous run (only meaningful if we had previous)
        if compute_embed_sim and img_name in previous_by_image:
            prev_desc = previous_by_image[img_name].get("text_description", "")
            new_desc = llm_output.get("text_description", "")
            print("  → Computing embedding similarity vs previous run...")
            sim = compute_description_similarity(prev_desc, new_desc)
            entry["_embedding_similarity_prev_run"] = sim
            print(f"    similarity={sim}")

        fout.write(json.dumps(entry) + "\n")

        # polite sleep for free tier
        delay = random.uniform(1.0, 2.5)
        print(f"  → Sleeping {delay:.1f}s...")
        time.sleep(delay)

    fout.close()

    # Atomic rename
    OUT_TMP.rename(OUT_PATH)
    print(f"\nAll done. Atomic write complete → {OUT_PATH}")


# ------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-labeling all images instead of resuming.",
    )
    parser.add_argument(
        "--compute-embedding-similarity",
        action="store_true",
        help="Compute embedding similarity between previous and new text_description (if previous exists).",
    )
    args = parser.parse_args()

    main(force=args.force, compute_embed_sim=args.compute_embedding_similarity)
