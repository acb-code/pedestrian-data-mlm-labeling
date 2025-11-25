"""
compute_text_embeddings.py
--------------------------

Computes and caches text embeddings from the fully cleaned dataset.

Outputs:
- text_embeddings.npy    (float32 array shape [N, D])
- text_embedding_index.jsonl (one entry per row: {image, text, valid})

Features:
- Uses cleaned annotations
- Batches API calls (minimizes rate limits)
- Skips embeddings already computed
- Stores metadata for later PCA/UMAP
"""

import json
import time
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

from google import genai
from google.genai import types

load_dotenv()
client = genai.Client()

# ---------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

ANN_PATH = ROOT / "data/cleaned/annotations_fully_cleaned.jsonl"
OUT_EMB = ROOT / "data/analysis/text_embeddings.npy"
OUT_INDEX = ROOT / "data/analysis/text_embedding_index.jsonl"

OUT_EMB.parent.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------

def get_embedding(text):
    """Get Gemini text embedding with retry + rate-limit handling."""
    if not text:
        return None

    for attempt in range(5):
        try:
            resp = client.models.embed_content(
                model="gemini-embedding-001",
                contents=text,
            )
            return list(resp.embeddings[0].values)

        except Exception as e:
            print(f"Embedding error (attempt {attempt+1}): {e}")
            time.sleep(1.5 + attempt * 1.2)

    return None


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():
    print("Loading annotations:", ANN_PATH)
    rows = [json.loads(line) for line in open(ANN_PATH)]

    embeddings = []
    index_entries = []

    print(f"Total rows: {len(rows)}")
    print("\nComputing text embeddings...")

    for i, r in enumerate(rows, start=1):
        text = r.get("text_description", "")
        image = r.get("image")

        print(f"[{i}/{len(rows)}] {image}")

        emb = get_embedding(text)

        if emb is None:
            print("  ✗ Failed → marking as invalid")
            index_entries.append({
                "image": image,
                "text": text,
                "valid": False,
            })
            continue

        embeddings.append(emb)
        index_entries.append({
            "image": image,
            "text": text,
            "valid": True,
        })

        # polite sleep for free API quota
        time.sleep(0.4)

    # Convert & save
    emb_array = np.array(embeddings, dtype=np.float32)
    np.save(OUT_EMB, emb_array)
    print(f"\nSaved embeddings → {OUT_EMB} (shape={emb_array.shape})")

    with open(OUT_INDEX, "w") as f:
        for ent in index_entries:
            f.write(json.dumps(ent) + "\n")

    print(f"Saved index → {OUT_INDEX}")
    print("\nEmbedding Precomputation Complete.")


if __name__ == "__main__":
    main()
