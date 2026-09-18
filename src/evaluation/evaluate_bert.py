"""Evaluate the saved BERT checkpoint on the held-out NER test split."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForTokenClassification
from src.models.transformer.bert_dataset import BERTDataset
from src.models.transformer.train import evaluate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    model_path = root / "models_saved" / "bert_pii"
    dataset = BERTDataset(root / "data" / "processed" / "test.json",
                          model_name=str(model_path), max_length=256)
    loader = DataLoader(dataset, batch_size=args.batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForTokenClassification.from_pretrained(model_path, local_files_only=True).to(device)
    scores = evaluate(model, loader, device)
    output = root / "results" / "bert_test_metrics.json"
    output.write_text(json.dumps(scores, indent=2), encoding="utf-8")
    print(json.dumps(scores, indent=2)); print(f"Saved: {output}")


if __name__ == "__main__":
    main()
