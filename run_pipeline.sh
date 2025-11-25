#!/bin/bash

set -e

echo "============================="
echo "STEP 1 — Import raw images"
echo "============================="
uv run mlm/image_annotating/import_images.py

echo "============================="
echo "STEP 2 — LLM annotation"
echo "============================="
uv run mlm/image_annotating/run_mlm_labels.py --force

echo "============================="
echo "STEP 3 — Initial validation"
echo "============================="
uv run mlm/data_cleaning/validate_dataset.py --input mlm/data/llm/annotations_llm.jsonl --report mlm/data_cleaning/report_llm_raw.txt || true

echo "============================="
echo "STEP 4 — Rule-based scene-graph cleaning"
echo "============================="
uv run mlm/data_cleaning/clean_scene_graph.py

echo "============================="
echo "STEP 5 — LLM scene-graph repair"
echo "============================="
uv run mlm/data_cleaning/repair_scene_graph_llm.py

echo "============================="
echo "STEP 6 — Missing-field repair"
echo "============================="
uv run mlm/data_cleaning/repair_missing_fields_llm.py

echo "============================="
echo "STEP 7 — Final validation"
echo "============================="
uv run mlm/data_cleaning/validate_dataset.py --input mlm/data/cleaned/annotations_fully_cleaned.jsonl --report mlm/data_cleaning/report_final.txt

echo "============================="
echo "STEP 8 — Compute text embeddings"
echo "============================="
uv run mlm/analysis/compute_text_embeddings.py

echo "============================="
echo "STEP 9 — Compute PCA & UMAP projections"
echo "============================="
uv run mlm/analysis/compute_projections.py

echo "============================="
echo "ALL DONE 🎉"
echo "============================="
