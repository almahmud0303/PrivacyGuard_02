from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models.transformer.labels import LABELS

from .dataset import PIIDataset
from .model import BiLSTM_PII


def _label_weights(dataset: PIIDataset) -> torch.Tensor:
    """Create bounded inverse-frequency weights without over-amplifying rare labels."""
    counts = torch.zeros(len(LABELS), dtype=torch.float)
    for item in dataset.data:
        for label in item["labels"][:dataset.max_len]:
            counts[dataset.label2idx[label]] += 1

    present = counts > 0
    weights = torch.zeros_like(counts)
    weights[present] = torch.sqrt(counts[present].sum() / (present.sum() * counts[present]))
    weights = weights.clamp(max=5.0)
    # O is useful context but should not dominate entity learning.
    weights[dataset.label2idx["O"]] = min(float(weights[dataset.label2idx["O"]]), 0.35)
    return weights


def _entity_metrics(probabilities: torch.Tensor, targets: torch.Tensor, threshold: float) -> dict[str, float]:
    confidence, predictions = probabilities.max(dim=-1)
    predictions = predictions.clone()
    predictions[confidence < threshold] = 0
    valid = targets != -100
    predictions, targets = predictions[valid], targets[valid]

    predicted_entity = predictions != 0
    true_entity = targets != 0
    true_positive = ((predictions == targets) & true_entity).sum().item()
    false_positive = (predicted_entity & ((predictions != targets) | ~true_entity)).sum().item()
    false_negative = (true_entity & (predictions != targets)).sum().item()
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    accuracy = (predictions == targets).float().mean().item() if len(targets) else 0.0
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


def _validate(model, loader, criterion, device) -> tuple[float, torch.Tensor, torch.Tensor]:
    model.eval()
    losses = []
    all_probabilities = []
    all_targets = []
    with torch.inference_mode():
        for tokens, labels in loader:
            tokens, labels = tokens.to(device), labels.to(device)
            lengths = (tokens != 0).sum(dim=1)
            logits = model(tokens, lengths=lengths)
            losses.append(criterion(logits.flatten(0, 1), labels.flatten()).item())
            all_probabilities.append(logits.softmax(dim=-1).cpu())
            all_targets.append(labels.cpu())
    return (
        sum(losses) / max(len(losses), 1),
        torch.cat(all_probabilities),
        torch.cat(all_targets),
    )


def _best_threshold(probabilities: torch.Tensor, targets: torch.Tensor) -> tuple[float, dict[str, float]]:
    # Below 0.50, an argmax label is not a useful confidence claim. Starting at
    # 0.50 also avoids selecting 0.0 when every candidate ties on easy data.
    candidates = [value / 100 for value in range(50, 96, 5)]
    scored = [(threshold, _entity_metrics(probabilities, targets, threshold)) for threshold in candidates]
    return max(scored, key=lambda item: (item[1]["f1"], item[1]["recall"]))


def train(args: argparse.Namespace) -> dict:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    project_root = Path(__file__).resolve().parents[3]
    data_dir = project_root / "data" / "processed"
    model_dir = project_root / "models_saved"
    results_dir = project_root / "results"
    model_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = PIIDataset(
        data_dir / "train.json",
        max_len=args.max_length,
        min_frequency=args.min_frequency,
        word_dropout=args.word_dropout,
        normalize_tokens=True,
    )
    validation_path = data_dir / "validation.json"
    validation_dataset = PIIDataset(
        validation_path,
        max_len=args.max_length,
        word2idx=train_dataset.word2idx,
        min_frequency=args.min_frequency,
        normalize_tokens=True,
    )

    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )

    model = BiLSTM_PII(
        vocab_size=len(train_dataset.word2idx),
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_labels=len(LABELS),
        dropout=args.dropout,
    ).to(device)
    weights = _label_weights(train_dataset).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=-100, weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=1)

    best_f1 = -1.0
    best_validation_loss = float("inf")
    best_epoch = 0
    best_threshold = 0.80
    best_metrics: dict[str, float] = {}
    history = []
    epochs_without_improvement = 0
    model_path = model_dir / "bilstm_pii.pt"

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for tokens, labels in train_loader:
            tokens, labels = tokens.to(device), labels.to(device)
            lengths = (tokens != 0).sum(dim=1)
            optimizer.zero_grad()
            output = model(tokens, lengths=lengths)
            loss = criterion(output.flatten(0, 1), labels.flatten())
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        validation_loss, probabilities, targets = _validate(
            model, validation_loader, criterion, device
        )
        threshold, metrics = _best_threshold(probabilities, targets)
        scheduler.step(metrics["f1"])
        epoch_result = {
            "epoch": epoch + 1,
            "train_loss": total_loss / max(len(train_loader), 1),
            "validation_loss": validation_loss,
            "threshold": threshold,
            **metrics,
        }
        history.append(epoch_result)
        print(
            f"Epoch {epoch + 1}/{args.epochs} "
            f"train_loss={epoch_result['train_loss']:.4f} "
            f"val_loss={validation_loss:.4f} val_f1={metrics['f1']:.4f} "
            f"threshold={threshold:.2f}"
        )

        f1_improved = metrics["f1"] > best_f1 + 1e-6
        checkpoint_improved = f1_improved or (
            abs(metrics["f1"] - best_f1) <= 1e-6
            and validation_loss < best_validation_loss
        )
        if checkpoint_improved:
            best_f1 = max(best_f1, metrics["f1"])
            best_validation_loss = validation_loss
            best_epoch = epoch + 1
            best_threshold = threshold
            best_metrics = metrics
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "word2idx": train_dataset.word2idx,
                    "labels": LABELS,
                    "max_len": train_dataset.max_len,
                    "normalize_tokens": train_dataset.normalize_tokens,
                    "min_frequency": train_dataset.min_frequency,
                    "word_dropout": train_dataset.word_dropout,
                    "embedding_dim": args.embedding_dim,
                    "hidden_dim": args.hidden_dim,
                    "dropout": args.dropout,
                    "recommended_threshold": best_threshold,
                    "validation_metrics": best_metrics,
                },
                model_path,
            )
        if f1_improved:
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"Early stopping after {epoch + 1} epochs")
                break

    (model_dir / "bilstm_labels.json").write_text(
        json.dumps(LABELS, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    inference_config = {
        "confidence_threshold": best_threshold,
        "max_length": args.max_length,
        "window_overlap": min(args.window_overlap, args.max_length - 1),
    }
    (model_dir / "bilstm_inference_config.json").write_text(
        json.dumps(inference_config, indent=2), encoding="utf-8"
    )
    summary = {
        "device": device,
        "best_epoch": best_epoch,
        "best_threshold": best_threshold,
        "best_validation_metrics": best_metrics,
        "train_samples": len(train_dataset),
        "validation_samples": len(validation_dataset),
        "vocabulary_size": len(train_dataset.word2idx),
        "max_length": args.max_length,
        "history": history,
    }
    (results_dir / "bilstm_metrics.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"Best model saved to {model_path}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the BiLSTM PII tagger")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--window-overlap", type=int, default=32)
    parser.add_argument("--min-frequency", type=int, default=2)
    parser.add_argument("--word-dropout", type=float, default=0.08)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--embedding-dim", type=int, default=100)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--patience", type=int, default=3)
    args = parser.parse_args()
    if args.max_length < 8:
        parser.error("--max-length must be at least 8")
    if not 0 <= args.window_overlap < args.max_length:
        parser.error("--window-overlap must be between 0 and max-length - 1")
    if not 0 <= args.word_dropout < 1 or not 0 <= args.dropout < 1:
        parser.error("dropout values must be in [0, 1)")
    return args


if __name__ == "__main__":
    train(parse_args())
