#!/bin/bash

set -e

ROOT="$(cd "$(dirname "$0")" && pwd)"

echo "============================="
echo "STEP 1 — Import raw images"
echo "============================="
(cd "$ROOT" && uv run image_annotating/import_images.py)

echo "============================="
echo "STEP 2 — LLM annotation"
echo "============================="
(cd "$ROOT" && uv run image_annotating/run_mlm_labels.py --force)

echo "============================="
echo "STEP 3 — Initial validation"
echo "============================="
(cd "$ROOT" && uv run data_cleaning/validate_dataset.py --input data/llm/annotations_llm.jsonl --report data_cleaning/report_llm_raw.txt) || true

echo "============================="
echo "STEP 4 — Rule-based scene-graph cleaning"
echo "============================="
(cd "$ROOT" && uv run data_cleaning/clean_scene_graph.py)

echo "============================="
echo "STEP 5 — LLM scene-graph repair"
echo "============================="
(cd "$ROOT" && uv run data_cleaning/repair_scene_graph_llm.py)

echo "============================="
echo "STEP 6 — Missing-field repair"
echo "============================="
(cd "$ROOT" && uv run data_cleaning/repair_missing_fields_llm.py)

echo "============================="
echo "STEP 7 — Final validation"
echo "============================="
(cd "$ROOT" && uv run data_cleaning/validate_dataset.py --input data/cleaned/annotations_fully_cleaned.jsonl --report data_cleaning/report_final.txt)

echo "============================="
echo "STEP 8 — Compute text embeddings"
echo "============================="
(cd "$ROOT" && uv run analysis/compute_text_embeddings.py)

echo "============================="
echo "STEP 9 — Compute PCA & UMAP projections"
echo "============================="
(cd "$ROOT" && uv run analysis/compute_projections.py)

echo "============================="
echo "ALL DONE 🎉"
echo "============================="
