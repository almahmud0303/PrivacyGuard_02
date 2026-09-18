"""Fine-tune multilingual BERT for model-only PII token classification."""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import torch
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from .bert_dataset import BERTDataset
from .bert_model import MODEL_NAME, create_model
from .labels import LABELS, label2id


def metrics(predictions: list[int], targets: list[int]) -> dict[str, float]:
    outside = label2id["O"]
    tp = sum(p == y and y != outside for p, y in zip(predictions, targets))
    fp = sum(p != y and p != outside for p, y in zip(predictions, targets))
    fn = sum(p != y and y != outside for p, y in zip(predictions, targets))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = sum(p == y for p, y in zip(predictions, targets)) / max(len(targets), 1)
    return {"accuracy": accuracy, "entity_precision": precision, "entity_recall": recall, "entity_f1": f1}


def evaluate(model, loader, device) -> dict[str, float]:
    model.eval(); predictions, targets, confidences = [], [], []
    with torch.inference_mode():
        for batch in loader:
            labels = batch["labels"].to(device)
            logits = model(input_ids=batch["input_ids"].to(device),
                           attention_mask=batch["attention_mask"].to(device)).logits
            probabilities = logits.softmax(dim=-1)
            confidence, predicted = probabilities.max(dim=-1)
            mask = labels != -100
            predictions.extend(predicted[mask].cpu().tolist())
            targets.extend(labels[mask].cpu().tolist())
            confidences.extend(confidence[mask].cpu().tolist())
    outside = label2id["O"]
    best = metrics(predictions, targets); best_threshold = 0.0
    for threshold in (0.0, .3, .4, .5, .6, .7, .75, .8, .85, .9, .95):
        filtered = [outside if pred != outside and conf < threshold else pred
                    for pred, conf in zip(predictions, confidences)]
        candidate = metrics(filtered, targets)
        if candidate["entity_f1"] > best["entity_f1"]:
            best, best_threshold = candidate, threshold
    best["recommended_threshold"] = best_threshold
    return best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    root = Path(__file__).resolve().parents[3]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_data = BERTDataset(root / "data" / "processed" / "train.json", model_name=args.model_name, max_length=256)
    validation_data = BERTDataset(root / "data" / "processed" / "validation.json",
                                  model_name=args.model_name, max_length=256)
    if args.max_samples > 0:
        train_data.data = train_data.data[:args.max_samples]
        validation_data.data = validation_data.data[:max(100, args.max_samples // 10)]
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    validation_loader = DataLoader(validation_data, batch_size=args.batch_size)

    counts = Counter(label for item in train_data.data for label in item["labels"])
    total = sum(counts.values())
    weights = []
    for label in LABELS:
        count = counts.get(label, 0)
        weight = (total / (len(LABELS) * count)) ** .5 if count else 0.0
        weights.append(min(weight, 5.0))
    weights[label2id["O"]] = min(weights[label2id["O"]], .35)
    criterion = CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float, device=device), ignore_index=-100)

    model = create_model(args.model_name).to(device)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=.01)
    update_steps = max(1, (len(train_loader) * args.epochs) // args.gradient_accumulation)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(update_steps * .1), update_steps)
    save_path = root / "models_saved" / "bert_pii"
    result_path = root / "results" / "bert_metrics.json"
    save_path.mkdir(parents=True, exist_ok=True); result_path.parent.mkdir(parents=True, exist_ok=True)

    best_f1, history = -1.0, []
    for epoch in range(args.epochs):
        model.train(); optimizer.zero_grad(); total_loss = 0.0
        for step, batch in enumerate(train_loader, start=1):
            labels = batch["labels"].to(device)
            logits = model(input_ids=batch["input_ids"].to(device),
                           attention_mask=batch["attention_mask"].to(device)).logits
            loss = criterion(logits.view(-1, len(LABELS)), labels.view(-1)) / args.gradient_accumulation
            loss.backward(); total_loss += loss.item() * args.gradient_accumulation
            if step % args.gradient_accumulation == 0 or step == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step(); scheduler.step(); optimizer.zero_grad()
            if step == 1 or step % 100 == 0:
                print(f"Epoch {epoch + 1}/{args.epochs} batch {step}/{len(train_loader)} loss={loss.item() * args.gradient_accumulation:.4f}")

        scores = evaluate(model, validation_loader, device)
        scores.update(epoch=epoch + 1, train_loss=total_loss / max(len(train_loader), 1))
        history.append(scores); print(json.dumps(scores, indent=2))
        if scores["entity_f1"] > best_f1:
            best_f1 = scores["entity_f1"]
            model.save_pretrained(save_path)
            train_data.tokenizer.save_pretrained(save_path)
            (save_path / "inference_config.json").write_text(json.dumps({
                "confidence_threshold": scores["recommended_threshold"],
                "validation_entity_f1": scores["entity_f1"],
                "epoch": epoch + 1,
            }, indent=2), encoding="utf-8")
            print(f"Saved best checkpoint with entity F1={best_f1:.4f}")
        result_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

    print(f"Training complete. Best validation entity F1: {best_f1:.4f}")
    print(f"Model: {save_path}\nMetrics: {result_path}")


if __name__ == "__main__":
    main()
