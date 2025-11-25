"""
Scene-Graph Cleaning Script
---------------------------

Fixes malformed scene_graph relations in annotations_llm.jsonl by converting
various syntaxes into the required triple format:

    "subject - relation - object"

Cleans patterns like:
- sidewalk_covered_with_leaves
- buildings-in_background
- road-sloped
- natural text: "sidewalk next to building"
"""

import re
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

IN_PATH = ROOT / "data/llm/annotations_llm.jsonl"
OUT_PATH = ROOT / "data/cleaned/annotations_cleaned_scene_graph.jsonl"

# ---------------------------------------------------------------------
# Core cleaning helpers
# ---------------------------------------------------------------------

def split_underscore_relation(rel: str):
    """
    Convert forms like:
        sidewalk_covered_with_leaves
        traffic_light-regulating_intersection
    into triplets.
    """
    parts = re.split(r"[_\-]+", rel)
    if len(parts) < 3:
        return None

    subject = parts[0]
    object_ = parts[-1]
    relation_terms = parts[1:-1]
    relation = " ".join(relation_terms)

    return f"{subject} - {relation} - {object_}"


def split_two_part_hyphen(rel: str):
    """
    Fix things like:
        buildings-in_background
        road-sloped
        bridge_structure-background
    """
    parts = rel.split("-", 1)
    if len(parts) != 2:
        return None

    subject = parts[0]
    relation = parts[1].replace("_", " ")
    object_ = "scene"  # default when no explicit object exists

    return f"{subject} - {relation} - {object_}"


def split_natural_language(rel: str):
    """
    Fix natural-language relations:
        road in front of curb
        curb next to sidewalk
        concrete planter contains bush
    """
    tokens = rel.split()
    if len(tokens) < 3:
        return None

    # heuristic: subject = first word
    subject = tokens[0]
    object_ = tokens[-1]
    relation = " ".join(tokens[1:-1])

    return f"{subject} - {relation} - {object_}"


def clean_relation(rel: str):
    """Return a cleaned triple version of the relation or None."""
    rel = rel.strip()

    # Case 1: Already correct triple
    if rel.count("-") == 2:
        return rel

    # Case 2: underscore-based pattern
    if "_" in rel and rel.count("_") >= 2:
        out = split_underscore_relation(rel)
        if out:
            return out

    # Case 3: hyphen, but only 1 hyphen
    if rel.count("-") == 1:
        out = split_two_part_hyphen(rel)
        if out:
            return out

    # Case 4: natural-language fallback
    out = split_natural_language(rel)
    if out:
        return out

    return None


# ---------------------------------------------------------------------
# Main cleaning
# ---------------------------------------------------------------------

def main():
    print(f"Loading: {IN_PATH}")
    cleaned = []
    fixed_count = 0
    failed_count = 0

    with open(IN_PATH) as f:
        for line in f:
            entry = json.loads(line)

            sg = entry.get("scene_graph", [])
            new_sg = []

            for rel in sg:
                cleaned_rel = clean_relation(rel)

                if cleaned_rel:
                    if cleaned_rel != rel:
                        fixed_count += 1
                    new_sg.append(cleaned_rel)
                else:
                    failed_count += 1
                    # Keep original but mark it as problematic
                    new_sg.append(f"[UNFIXED] {rel}")

            entry["scene_graph"] = new_sg
            cleaned.append(entry)

    # Write out cleaned file
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        for row in cleaned:
            f.write(json.dumps(row) + "\n")

    print("\nDone.")
    print(f"Cleaned scene_graph triples written to: {OUT_PATH}")
    print(f"  Fixed:   {fixed_count}")
    print(f"  Unfixed: {failed_count}")


if __name__ == "__main__":
    main()
