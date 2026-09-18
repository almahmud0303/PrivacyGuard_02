"""Create consistent BIO annotations and deterministic sequence splits."""
from __future__ import annotations

import csv
import json
import random
import re
from pathlib import Path

from src.redection.entity_detector import detect_entities

ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = ROOT / "data" / "raw" / "pii_dataset.csv"
ANNOTATION_PATH = ROOT / "data" / "annotations" / "bio_labels.json"
PROCESSED_PATH = ROOT / "data" / "processed" / "bio_dataset.json"
TOKEN_PATTERN = re.compile(r"\S+")
SEQUENCE_TYPES = {"PERSON", "EMAIL", "PHONE", "NID", "LOCATION"}


def annotate_sentence(text: str) -> dict[str, list[str]]:
    matches = list(TOKEN_PATTERN.finditer(text))
    entities = [entity for entity in detect_entities(text) if entity["type"] in SEQUENCE_TYPES]
    labels = []
    previous: tuple[int, str] | None = None
    for token in matches:
        entity_index = next((index for index, entity in enumerate(entities)
            if token.start() < entity["end"] and token.end() > entity["start"]), None)
        if entity_index is None:
            labels.append("O"); previous = None
            continue
        kind = entities[entity_index]["type"]
        prefix = "I" if previous == (entity_index, kind) else "B"
        labels.append(f"{prefix}-{kind}")
        previous = (entity_index, kind)
    return {"tokens": [match.group() for match in matches], "labels": labels}


def build_annotations() -> list[dict[str, list[str]]]:
    with INPUT_PATH.open(encoding="utf-8", newline="") as handle:
        texts = [row["text"] for row in csv.DictReader(handle)]
    return [annotate_sentence(text) for text in texts]


def _write_json(path: Path, data: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    annotations = build_annotations()
    _write_json(ANNOTATION_PATH, annotations)
    _write_json(PROCESSED_PATH, annotations)

    shuffled = annotations.copy()
    random.Random(42).shuffle(shuffled)
    train_end = int(len(shuffled) * .8)
    validation_end = int(len(shuffled) * .9)
    _write_json(ROOT / "data" / "processed" / "train.json", shuffled[:train_end])
    _write_json(ROOT / "data" / "processed" / "validation.json", shuffled[train_end:validation_end])
    _write_json(ROOT / "data" / "processed" / "test.json", shuffled[validation_end:])
    print(f"Annotated {len(annotations)} records (train={train_end}, validation={validation_end-train_end}, test={len(shuffled)-validation_end})")


if __name__ == "__main__":
    main()
