"""
repair_scene_graph_llm.py
-------------------------

Batch-fixes all `[UNFIXED] <relation>` scene-graph entries using a single Gemini API call.
This takes the partially-cleaned file produced by clean_scene_graph.py and returns a fully
cleaned scene-graph annotation set.

Steps:
1. Load cleaned scene graph annotations.
2. Extract all unique `[UNFIXED] <relation>` entries.
3. Send them to Gemini in a single batch prompt.
4. Replace them with corrected triple-form relations.
5. Write out a fully corrected annotations file.
"""

import json
from pathlib import Path
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()
client = genai.Client()

ROOT = Path(__file__).resolve().parents[1]

IN_PATH = ROOT / "data/cleaned/annotations_cleaned_scene_graph.jsonl"
OUT_PATH = ROOT / "data/cleaned/annotations_cleaned_scene_graph_llm.jsonl"

# ---------------------------------------------------------------------
# 1. Load dataset and collect UNFIXED relations
# ---------------------------------------------------------------------

def load_annotations():
    rows = []
    with open(IN_PATH) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def extract_unfixed(rows):
    unfixed = set()
    for r in rows:
        for rel in r.get("scene_graph", []):
            if rel.startswith("[UNFIXED]"):
                raw = rel.replace("[UNFIXED]", "").strip()
                unfixed.add(raw)
    return sorted(unfixed)


# ---------------------------------------------------------------------
# 2. Build batch LLM prompt
# ---------------------------------------------------------------------

def build_prompt(unfixed_list):
    items = "\n".join([f"- {rel}" for rel in unfixed_list])
    return f"""
You are a scene-graph normalization assistant. Convert each of the following scene relations
into a **triplet** strictly in this form:

    subject - relation - object

Rules:
- Use natural language for relation.
- No underscores.
- subject/object = simple nouns or noun phrases.
- relation = verb or spatial phrase.
- If relation refers to scene context ("in background", "on left"), use object = "scene".

Convert ALL of the following (preserve order):

{items}

Return JSON ONLY in this form:

{{
  "results": [
    "subject - relation - object",
    ...
  ]
}}
"""


# ---------------------------------------------------------------------
# 3. Batch LLM Call
# ---------------------------------------------------------------------

def call_gemini_batch(unfixed_list):
    prompt = build_prompt(unfixed_list)

    resp = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[prompt],
        config=types.GenerateContentConfig(
            response_mime_type="application/json"
        ),
    )

    try:
        data = json.loads(resp.text)
        return data["results"]
    except Exception:
        print("Error: Gemini returned invalid JSON. Raw output:")
        print(resp.text)
        raise


# ---------------------------------------------------------------------
# 4. Apply fixes back to dataset
# ---------------------------------------------------------------------

def apply_fixes(rows, unfixed_list, fixed_list):
    mapping = dict(zip(unfixed_list, fixed_list))
    out = []

    for row in rows:
        new_sg = []
        for rel in row.get("scene_graph", []):
            if rel.startswith("[UNFIXED]"):
                raw = rel.replace("[UNFIXED]", "").strip()
                new_sg.append(mapping.get(raw, rel))  # fallback to original
            else:
                new_sg.append(rel)
        row["scene_graph"] = new_sg
        out.append(row)

    return out


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():
    print(f"Loading: {IN_PATH}")
    rows = load_annotations()

    unfixed = extract_unfixed(rows)
    print(f"Found {len(unfixed)} unfixed relations.")

    if not unfixed:
        print("Nothing to fix. Writing passthrough file.")
        cleaned = rows
    else:
        print("Sending batch LLM correction request...")
        fixed = call_gemini_batch(unfixed)

        print("Applying fixes...")
        cleaned = apply_fixes(rows, unfixed, fixed)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        for r in cleaned:
            f.write(json.dumps(r) + "\n")

    print(f"\nDone.")
    print(f"Fully cleaned annotations written to: {OUT_PATH}")
    print(f"  Total fixed: {len(unfixed)}")


if __name__ == "__main__":
    main()
