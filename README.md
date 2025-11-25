# pedestrian-data-mlm-labeling

LLM-assisted pipeline for labeling pedestrian navigation scenes. The project ingests raw street-level photos, generates rich multimodal annotations (objects, scene graphs, affordances, text), cleans them with rule-based and LLM passes, and produces analysis-ready embeddings and projections.

## What's inside
- End-to-end pipeline (`run_pipeline.sh`) from raw images → cleaned annotations → embeddings → PCA/UMAP projections.
- Image hashing/import to normalize formats and track provenance.
- Gemini-powered annotation (`image_annotating/run_mlm_labels.py`) with strict schemas and controlled vocabularies (`image_annotating/categories.yaml`).
- Rule-based + LLM repair for scene graphs and missing fields.
- Validation scripts to catch schema and geometry issues early.
- Embedding + projection scripts for quick visualization/analysis.
- Notebook (`analysis/bounding_box_analysis_v2.ipynb`) to inspect bounding boxes and cleaned data.

## Repository layout
- `run_pipeline.sh` — orchestrates the full flow.
- `image_annotating/` — import images, run primary MLM labeling, category definitions.
- `data_cleaning/` — validation, scene-graph cleaning, LLM-based repairs, missing-field fixes.
- `analysis/` — text embeddings, PCA/UMAP projections, bounding box exploration notebook.
- `data/` — raw inputs, intermediate LLM outputs, cleaned annotations, processed images, analysis artifacts.

## Prerequisites
- Python 3.12+
- [uv](https://github.com/astral-sh/uv) for env + script execution (`pip install uv` or follow uv docs)
- Gemini API key in environment: create `.env` with `GEMINI_API_KEY=...`
- Raw pedestrian images placed in `data/raw/images_original/` (JPEG/PNG)

## Setup
```bash
# from repo root
uv sync  # install deps from pyproject/uv.lock
```
Ensure `.env` contains `GEMINI_API_KEY`, or export it in your shell before running.

## Running the full pipeline
```bash
./run_pipeline.sh
```
Steps executed:
1. Import raw images → hashes + normalized JPEGs at `data/processed/images_hashed/`, bootstrap annotations at `data/raw/annotations.jsonl`.
2. LLM annotation → `data/llm/annotations_llm.jsonl` (+ raw responses in `data/llm/raw_llm_responses/`).
3. Initial validation → report at `data_cleaning/report_llm_raw.txt`.
4. Rule-based scene-graph cleaning → `data/cleaned/annotations_cleaned_scene_graph.jsonl`.
5. LLM scene-graph repair → `data/cleaned/annotations_cleaned_scene_graph_llm.jsonl`.
6. Missing-field repair → `data/cleaned/annotations_fully_cleaned.jsonl`.
7. Final validation → report at `data_cleaning/report_final.txt`.
8. Text embeddings (Gemini) → `data/analysis/text_embeddings.npy` + index.
9. PCA/UMAP projections → `data/analysis/projections.json` + `.npy` artifacts.

## Running individual stages
- Import & hash images: `uv run image_annotating/import_images.py`
- Primary labeling (resume-aware): `uv run image_annotating/run_mlm_labels.py --force`
- Validate any JSONL: `uv run data_cleaning/validate_dataset.py --input <path> --report <out>`
- Clean scene graphs (rules): `uv run data_cleaning/clean_scene_graph.py`
- Repair scene graphs (LLM): `uv run data_cleaning/repair_scene_graph_llm.py`
- Repair missing fields (LLM): `uv run data_cleaning/repair_missing_fields_llm.py`
- Compute embeddings: `uv run analysis/compute_text_embeddings.py`
- Compute PCA/UMAP: `uv run analysis/compute_projections.py`
- Build packaged dataset: `uv run packaging/build_packaged_dataset.py --split`

## Notebooks
- `analysis/bounding_box_analysis_v2.ipynb` — bounding-box QA/visualization. Run after pipeline stages; auto-detects repo root.
- `analysis/explore_pedestrian_affordances.ipynb` — explore annotations/affordances interactively with processed images.

## Inputs and outputs (key files)
- Input images: `data/raw/images_original/`
- Processed images: `data/processed/images_hashed/`
- LLM annotations (raw): `data/llm/annotations_llm.jsonl`
- Scene-graph cleaned: `data/cleaned/annotations_cleaned_scene_graph.jsonl`
- Scene-graph cleaned (LLM-fixed): `data/cleaned/annotations_cleaned_scene_graph_llm.jsonl`
- Fully cleaned dataset: `data/cleaned/annotations_fully_cleaned.jsonl`
- Validation reports: `data_cleaning/report_llm_raw.txt`, `data_cleaning/report_final.txt`
- Embeddings + projections: `data/analysis/*`
- Packaged dataset exports: `data/package/annotations_native.jsonl`, `data/package/annotations_coco.json`, `data/package/index.csv`, `data/package/images/` (symlinks)

## Notes
- LLM-powered steps (labeling, repairs, embeddings) incur API calls/costs; keep an eye on your quota.
- The pipeline is safe to rerun; intermediate files are overwritten via atomic writes where relevant.
