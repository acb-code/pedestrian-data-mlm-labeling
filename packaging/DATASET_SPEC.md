# Packaged Dataset

Artifacts written to `data/package/` by `packaging/build_packaged_dataset.py`:

- `annotations_native.jsonl` — lossless per-image records, includes width/height and optional split.
- `annotations_coco.json` — COCO-style detection/segmentation export (images/categories/annotations).
- `index.csv` — tabular view for quick inspection/joins.
- `images/` — symlinks to `data/processed/images_hashed/` (can be disabled with `--no-symlink`).

## Running

```bash
uv run packaging/build_packaged_dataset.py --split   # adds deterministic 80/10/10 splits
```

Flags:
- `--split` — enable splits (80/10/10, seeded)
- `--seed <int>` — seed for splits (default 13)
- `--no-symlink` — skip creating `data/package/images` symlinks

Inputs:
- Cleaned dataset: `data/cleaned/annotations_fully_cleaned.jsonl`
- Images: `data/processed/images_hashed/`

Outputs land in `data/package/` (created if missing).

## Native JSONL schema (per row)

- `image` (str): hashed filename (e.g., `abcd1234ef567890.jpg`)
- `source_file` (str, optional): original filename
- `width`, `height` (int): from processed JPEG
- `split` (str, optional): `train`/`val`/`test` when `--split` is used
- `affordance_tags` (list[str])
- `walkability` (float in [0,1])
- `risk_level`, `surface`, `nav_cue`, `lighting`, `density` (str)
- `text_description` (str)
- `objects` (list[object]): each with `label`, `bbox` = `[x1, y1, x2, y2]`, `confidence`
- `segments` (list[segment]): each with `label`, `polygon` = `[[x, y], ...]`, `confidence`
- `scene_graph` (list[str]): `"subject - relation - object"`
- `affordance_conflicts` (list[str])

## COCO export notes

- `images`: `{id, file_name, width, height, split?}`
- `categories`: derived from unique labels in `objects`/`segments`, sorted, 1-based `id`.
- `annotations`:
  - For `objects`: `bbox` converted to `[x, y, w, h]`, `area` from bbox, empty `segmentation`.
  - For `segments`: `segmentation` flattened polygon, `area` via shoelace, empty `bbox`.
  - Common: `id`, `image_id`, `category_id`, `iscrowd` = 0.
- Scene graphs and affordances stay in the native JSONL / index.csv (COCO does not support them).

## DVC & syncing (coming next)

Planned tracking:
- `data/package/` (primary packaged outputs)
- Optionally `data/processed/images_hashed/` if you want the source image dir tracked too.

Remote: Google Drive (to be configured during DVC setup). We'll walk through `dvc init`, remote config, and `dvc add` together before pushing.
