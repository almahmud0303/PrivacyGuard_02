"""Unified, failure-safe inference for every PrivacyGuard model."""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

from src.redection.privacy_firewall import privacy_guard
from src.redection.entity_detector import detect_entities
from src.redection.redactor import redact_text
from src.redection.risk_score import calculate_risk, calculate_total_risk

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models_saved"


def available_models() -> dict[str, dict[str, str | bool]]:
    return {
        "Pattern detector": {"ready": True, "kind": "entity"},
        "BERT": {"ready": (MODEL_DIR / "bert_pii" / "model.safetensors").is_file(), "kind": "entity"},
        "BiLSTM": {"ready": (MODEL_DIR / "bilstm_pii.pt").is_file(), "kind": "entity"},
        "Logistic Regression": {"ready": (MODEL_DIR / "logistic_regression.joblib").is_file(), "kind": "classifier"},
        "Naive Bayes": {"ready": (MODEL_DIR / "naive_bayes.joblib").is_file(), "kind": "classifier"},
        "SVM": {"ready": (MODEL_DIR / "svm.joblib").is_file(), "kind": "classifier"},
    }


def _finish(text: str, entities: list[dict], threshold: float, model: str, note: str = "") -> dict:
    cleaned = []
    seen = set()
    for entity in sorted(entities, key=lambda e: (e["start"], e["end"])):
        key = (entity["start"], entity["end"], entity["type"])
        if key in seen or entity["type"] == "O":
            continue
        seen.add(key)
        risk = calculate_risk(entity["type"])
        entity.update(risk=risk["level"], score=risk["score"])
        cleaned.append(entity)
    return {"original": text, "entities": cleaned, "entity_count": len(cleaned),
            "overall_risk": calculate_total_risk(cleaned), "safe_text": redact_text(text, cleaned, threshold),
            "model": model, "note": note}


def _with_safety_patterns(text: str, model_entities: list[dict]) -> list[dict]:
    """Keep deterministic structured PII protection around experimental models."""
    patterns = detect_entities(text)
    protected_spans = [(item["start"], item["end"]) for item in patterns]
    extras = [item for item in model_entities if not any(
        item["start"] < end and item["end"] > start for start, end in protected_spans)]
    return patterns + extras


@lru_cache(maxsize=1)
def _bert_components():
    import torch
    from transformers import AutoModelForTokenClassification, AutoTokenizer
    path = MODEL_DIR / "bert_pii"
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(path, local_files_only=True)
    model.eval()
    return tokenizer, model, torch


def _predict_bert(text: str) -> list[dict]:
    tokenizer, model, torch = _bert_components()
    encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, return_offsets_mapping=True)
    offsets = encoded.pop("offset_mapping")[0].tolist()
    with torch.inference_mode():
        probabilities = model(**encoded).logits.softmax(dim=-1)[0]
    ids = probabilities.argmax(dim=-1).tolist()
    labels = model.config.id2label
    pieces = []
    for index, ((start, end), label_id) in enumerate(zip(offsets, ids)):
        label = labels.get(label_id, labels.get(str(label_id), "O"))
        if start == end or label == "O":
            continue
        pieces.append({"entity": text[start:end], "type": label.removeprefix("B-").removeprefix("I-"),
                       "start": start, "end": end, "confidence": float(probabilities[index, label_id]), "source": "bert"})
    # Join adjacent word pieces belonging to the same entity type.
    merged = []
    for item in pieces:
        if merged and item["type"] == merged[-1]["type"] and item["start"] <= merged[-1]["end"] + 1:
            merged[-1]["end"] = item["end"]
            merged[-1]["entity"] = text[merged[-1]["start"]:item["end"]]
            merged[-1]["confidence"] = min(merged[-1]["confidence"], item["confidence"])
        else:
            merged.append(item)
    return merged


@lru_cache(maxsize=1)
def _lstm_components():
    import torch
    from src.models.lstm.dataset import PIIDataset
    from src.models.lstm.model import BiLSTM_PII
    dataset = PIIDataset(ROOT / "data" / "annotations" / "bio_labels.json")
    model = BiLSTM_PII(len(dataset.word2idx))
    model.load_state_dict(torch.load(MODEL_DIR / "bilstm_pii.pt", map_location="cpu", weights_only=True))
    model.eval()
    return dataset, model, torch


def _predict_lstm(text: str) -> list[dict]:
    dataset, model, torch = _lstm_components()
    matches = list(re.finditer(r"\S+", text))[:dataset.max_len]
    ids = [dataset.word2idx.get(m.group(), 1) for m in matches]
    padded = ids + [0] * (dataset.max_len - len(ids))
    with torch.inference_mode():
        probs = model(torch.tensor([padded])).softmax(dim=-1)[0]
    labels = ["O", "B-PERSON", "B-EMAIL", "B-PHONE", "B-NID", "B-LOCATION"]
    entities = []
    for index, match in enumerate(matches):
        label_id = int(probs[index].argmax())
        if label_id:
            entities.append({"entity": match.group(), "type": labels[label_id].removeprefix("B-"),
                "start": match.start(), "end": match.end(), "confidence": float(probs[index, label_id]), "source": "bilstm"})
    return entities


@lru_cache(maxsize=3)
def _classical_model(filename: str):
    import joblib
    return joblib.load(MODEL_DIR / filename)


def analyze(text: str, model_name: str, threshold: float = .80, fallback: bool = True) -> dict:
    """Analyze text without allowing a model failure to crash the caller."""
    try:
        if model_name == "Pattern detector":
            result = privacy_guard(text, threshold); result.update(model=model_name, note="")
            return result
        if model_name == "BERT":
            return _finish(text, _with_safety_patterns(text, _predict_bert(text)), threshold, model_name,
                           "Structured PII patterns remain enabled as a privacy safety layer.")
        if model_name == "BiLSTM":
            return _finish(text, _with_safety_patterns(text, _predict_lstm(text)), threshold, model_name,
                           "Structured PII patterns remain enabled as a privacy safety layer.")
        files = {"Logistic Regression": "logistic_regression.joblib", "Naive Bayes": "naive_bayes.joblib", "SVM": "svm.joblib"}
        classifier = _classical_model(files[model_name])
        prediction = int(classifier.predict([text])[0])
        result = privacy_guard(text, threshold)
        result.update(model=model_name, classification="PII" if prediction else "SAFE",
                      note="Classical models classify the whole text; pattern spans are used for safe redaction.")
        return result
    except Exception as exc:
        if not fallback:
            raise
        result = privacy_guard(text, threshold)
        result.update(model="Pattern detector", requested_model=model_name,
                      note=f"{model_name} could not run ({type(exc).__name__}). Used the safe pattern detector instead.")
        return result
