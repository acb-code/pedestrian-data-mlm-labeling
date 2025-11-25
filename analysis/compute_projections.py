"""
compute_projections.py
-----------------------

Loads precomputed text embeddings and computes low-dimensional projections:

- PCA (2D)
- PCA (3D)
- UMAP (2D)

Outputs:
- projections.json
- pca_2d.npy
- pca_3d.npy
- umap_2d.npy

"""

import json
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
import umap


ROOT = Path(__file__).resolve().parents[1]

EMB_PATH = ROOT / "data/analysis/text_embeddings.npy"
INDEX_PATH = ROOT / "data/analysis/text_embedding_index.jsonl"

OUT_PROJ_JSON = ROOT / "data/analysis/projections.json"
OUT_PCA2 = ROOT / "data/analysis/pca_2d.npy"
OUT_PCA3 = ROOT / "data/analysis/pca_3d.npy"
OUT_UMAP2 = ROOT / "data/analysis/umap_2d.npy"


def load_embeddings():
    print("Loading embeddings:", EMB_PATH)
    emb = np.load(EMB_PATH)
    print(" → shape:", emb.shape)
    return emb


def load_index():
    rows = []
    with open(INDEX_PATH) as f:
        for line in f:
            rows.append(json.loads(line))
    print("Loaded index:", len(rows))
    return rows


def compute_pca(emb):
    print("\nComputing PCA...")
    pca2 = PCA(n_components=2).fit_transform(emb)
    pca3 = PCA(n_components=3).fit_transform(emb)
    print(" → PCA 2D:", pca2.shape)
    print(" → PCA 3D:", pca3.shape)
    return pca2, pca3


def compute_umap(emb):
    print("\nComputing UMAP...")
    reducer = umap.UMAP(n_components=2, random_state=42)
    coords = reducer.fit_transform(emb)
    print(" → UMAP 2D:", coords.shape)
    return coords


def main():
    print("=== Projection Computation ===")

    emb = load_embeddings()
    index_rows = load_index()

    if emb.shape[0] != len(index_rows):
        print(" !!! WARNING: Embedding count does not match index count !!!")
        print("Embeddings:", emb.shape[0], " Index:", len(index_rows))

    # --- Compute projections ---
    pca2, pca3 = compute_pca(emb)
    umap2 = compute_umap(emb)

    # --- Save ---
    np.save(OUT_PCA2, pca2)
    np.save(OUT_PCA3, pca3)
    np.save(OUT_UMAP2, umap2)

    # Combined JSON metadata file
    proj = {
        "count": len(index_rows),
        "images": [r["image"] for r in index_rows],
        "pca_2d_x": pca2[:, 0].tolist(),
        "pca_2d_y": pca2[:, 1].tolist(),
        "umap_2d_x": umap2[:, 0].tolist(),
        "umap_2d_y": umap2[:, 1].tolist(),
    }

    with open(OUT_PROJ_JSON, "w") as f:
        json.dump(proj, f, indent=2)

    print("\nSaved:")
    print(" -", OUT_PCA2)
    print(" -", OUT_PCA3)
    print(" -", OUT_UMAP2)
    print(" -", OUT_PROJ_JSON)
    print("\nProjection computation complete.")


if __name__ == "__main__":
    main()
