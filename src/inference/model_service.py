"""Unified, failure-safe inference for every PrivacyGuard model."""
from __future__ import annotations

import re
import json
from typing import NamedTuple
from functools import lru_cache
from pathlib import Path

from src.redection.privacy_firewall import privacy_guard
from src.redection.redactor import redact_text
from src.redection.risk_score import calculate_risk, calculate_total_risk

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models_saved"


class TokenSpan(NamedTuple):
    value: str
    start: int
    end: int


def _lexical_spans(text: str) -> list[TokenSpan]:
    """Tokenize consistently without classifying any entity type."""
    spans = []
    edge_punctuation = "\"'()[]{}<>,;:!?।"
    for match in re.finditer(r"\S+", text):
        value, start, end = match.group(), match.start(), match.end()
        while value and value[0] in edge_punctuation:
            spans.append(TokenSpan(value[0], start, start + 1)); value, start = value[1:], start + 1
        trailing = []
        while value and (value[-1] in edge_punctuation or value[-1] == "."):
            trailing.append(TokenSpan(value[-1], end - 1, end)); value, end = value[:-1], end - 1
        if value:
            spans.append(TokenSpan(value, start, end))
        spans.extend(reversed(trailing))
    return spans


def available_models() -> dict[str, dict[str, str | bool | float]]:
    from src.models.transformer.labels import LABELS
    bert_ready = False
    config_path = MODEL_DIR / "bert_pii" / "config.json"
    if config_path.is_file():
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
            configured = [config["id2label"][str(index)] for index in range(len(config["id2label"]))]
            bert_ready = configured == LABELS
        except (KeyError, ValueError, TypeError, json.JSONDecodeError):
            pass
    bilstm_ready = False
    bilstm_labels = MODEL_DIR / "bilstm_labels.json"
    if (MODEL_DIR / "bilstm_pii.pt").is_file() and bilstm_labels.is_file():
        try:
            bilstm_ready = json.loads(bilstm_labels.read_text(encoding="utf-8")) == LABELS
        except (ValueError, TypeError, json.JSONDecodeError):
            pass
    bert_threshold = .80
    bert_inference_config = MODEL_DIR / "bert_pii" / "inference_config.json"
    if bert_inference_config.is_file():
        try: bert_threshold = float(json.loads(bert_inference_config.read_text(encoding="utf-8"))["confidence_threshold"])
        except (KeyError, ValueError, TypeError, json.JSONDecodeError): pass
    bilstm_threshold = .80
    bilstm_inference_config = MODEL_DIR / "bilstm_inference_config.json"
    if bilstm_inference_config.is_file():
        try: bilstm_threshold = float(json.loads(bilstm_inference_config.read_text(encoding="utf-8"))["confidence_threshold"])
        except (KeyError, ValueError, TypeError, json.JSONDecodeError): pass
    return {
        "Pattern detector": {"ready": True, "kind": "entity", "threshold": .80},
        "BERT": {"ready": bert_ready, "kind": "entity", "threshold": bert_threshold},
        "BiLSTM": {"ready": bilstm_ready, "kind": "entity", "threshold": bilstm_threshold},
        "Logistic Regression": {"ready": (MODEL_DIR / "logistic_regression.joblib").is_file(), "kind": "classifier"},
        "Logistic Regression (Scratch)": {
            "ready": (MODEL_DIR / "logistic_regression_scratch.joblib").is_file(),
            "kind": "classifier",
        },
        "Naive Bayes": {"ready": (MODEL_DIR / "naive_bayes.joblib").is_file(), "kind": "classifier"},
        "Naive Bayes (Scratch)": {
            "ready": (MODEL_DIR / "naive_bayes_scratch.joblib").is_file(),
            "kind": "classifier",
        },
        "SVM": {"ready": (MODEL_DIR / "svm.joblib").is_file(), "kind": "classifier"},
        "SVM (Scratch)": {
            "ready": (MODEL_DIR / "svm_scratch.joblib").is_file(),
            "kind": "classifier",
        },
    }


def _finish(text: str, entities: list[dict], threshold: float, model: str, note: str = "") -> dict:
    cleaned = []
    seen = set()
    for entity in sorted(entities, key=lambda e: (e["start"], e["end"])):
        if float(entity.get("confidence", 1.0)) < threshold:
            continue
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


@lru_cache(maxsize=1)
def _bert_components():
    import torch
    from transformers import AutoModelForTokenClassification, AutoTokenizer
    path = MODEL_DIR / "bert_pii"
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(path, local_files_only=True)
    from src.models.transformer.labels import LABELS
    configured = [model.config.id2label[index] for index in range(model.config.num_labels)]
    if configured != LABELS:
        raise RuntimeError("BERT checkpoint uses the old label schema; retrain it")
    model.eval()
    return tokenizer, model, torch


def _predict_bert(text: str) -> list[dict]:
    tokenizer, model, torch = _bert_components()
    token_spans = _lexical_spans(text)
    if not token_spans:
        return []
    encoded = tokenizer([span.value for span in token_spans], is_split_into_words=True,
                        return_tensors="pt", truncation=True, max_length=512)
    word_ids = encoded.word_ids()
    with torch.inference_mode():
        probabilities = model(**encoded).logits.softmax(dim=-1)[0]
    ids = probabilities.argmax(dim=-1).tolist()
    labels = model.config.id2label
    pieces = []
    seen_words = set()
    for index, (label_id, word_id) in enumerate(zip(ids, word_ids)):
        if word_id is None or word_id in seen_words:
            continue
        seen_words.add(word_id)
        start, end = token_spans[word_id].start, token_spans[word_id].end
        label = labels.get(label_id, labels.get(str(label_id), "O"))
        if start == end or label == "O":
            continue
        if label.startswith("B-") and not any(char.isalnum() for char in text[start:end]):
            continue
        confidence = float(probabilities[index, label_id])
        pieces.append({"entity": text[start:end], "type": label.removeprefix("B-").removeprefix("I-"), "bio": label[:1],
                       "start": start, "end": end, "confidence": confidence,
                       "confidence_sum": confidence, "token_count": 1, "source": "bert"})
    # Join adjacent word pieces belonging to the same entity type.
    merged = []
    for item in pieces:
        if merged and item["bio"] == "I" and item["type"] == merged[-1]["type"] and item["start"] <= merged[-1]["end"] + 1:
            merged[-1]["end"] = item["end"]
            merged[-1]["entity"] = text[merged[-1]["start"]:item["end"]]
            merged[-1]["confidence_sum"] += item["confidence"]
            merged[-1]["token_count"] += 1
            merged[-1]["confidence"] = merged[-1]["confidence_sum"] / merged[-1]["token_count"]
        else:
            merged.append(item)
    for item in merged:
        item.pop("bio", None)
        item.pop("confidence_sum", None)
        item.pop("token_count", None)
    return merged


@lru_cache(maxsize=1)
def _lstm_components():
    import torch
    from src.models.lstm.dataset import PIIDataset
    from src.models.lstm.model import BiLSTM_PII
    from src.models.transformer.labels import LABELS
    labels_file = MODEL_DIR / "bilstm_labels.json"
    if not labels_file.is_file() or json.loads(labels_file.read_text(encoding="utf-8")) != LABELS:
        raise RuntimeError("BiLSTM checkpoint uses the old label schema; retrain it")
    checkpoint = torch.load(MODEL_DIR / "bilstm_pii.pt", map_location="cpu", weights_only=True)
    if "state_dict" not in checkpoint or checkpoint.get("labels") != LABELS:
        raise RuntimeError("BiLSTM checkpoint has no compatible vocabulary metadata; retrain it")
    dataset = PIIDataset(
        ROOT / "data" / "processed" / "train.json",
        max_len=checkpoint.get("max_len", 32),
        word2idx=checkpoint["word2idx"],
        min_frequency=checkpoint.get("min_frequency", 1),
        normalize_tokens=checkpoint.get("normalize_tokens", False),
    )
    inference_config = MODEL_DIR / "bilstm_inference_config.json"
    overlap = min(16, max(dataset.max_len - 1, 0))
    if inference_config.is_file():
        try:
            overlap = int(json.loads(inference_config.read_text(encoding="utf-8"))["window_overlap"])
        except (KeyError, ValueError, TypeError, json.JSONDecodeError):
            pass
    dataset.window_overlap = min(max(overlap, 0), max(dataset.max_len - 1, 0))
    model = BiLSTM_PII(
        len(dataset.word2idx),
        embedding_dim=checkpoint.get("embedding_dim", 100),
        hidden_dim=checkpoint.get("hidden_dim", 128),
        num_labels=len(LABELS),
        dropout=checkpoint.get("dropout", 0.2),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return dataset, model, torch


def _predict_lstm(text: str) -> list[dict]:
    dataset, model, torch = _lstm_components()
    from src.models.transformer.labels import LABELS
    spans = _lexical_spans(text)
    if not spans:
        return []

    # Overlapping windows preserve bidirectional context at boundaries and let
    # the same checkpoint process text of any practical length.
    max_len = dataset.max_len
    overlap = getattr(dataset, "window_overlap", min(16, max_len - 1))
    stride = max(max_len - overlap, 1)
    starts = list(range(0, len(spans), stride))
    windows = [spans[start:start + max_len] for start in starts]
    best_context = [-1] * len(spans)
    best_probabilities = [None] * len(spans)

    with torch.inference_mode():
        for batch_start in range(0, len(windows), 64):
            batch_windows = windows[batch_start:batch_start + 64]
            lengths = torch.tensor([len(window) for window in batch_windows], dtype=torch.long)
            token_rows = []
            for window in batch_windows:
                ids = [dataset.token_to_id(span.value) for span in window]
                token_rows.append(ids + [0] * (max_len - len(ids)))
            probabilities = model(
                torch.tensor(token_rows, dtype=torch.long), lengths=lengths
            ).softmax(dim=-1)
            for local_batch_index, window in enumerate(batch_windows):
                global_start = starts[batch_start + local_batch_index]
                for local_index in range(len(window)):
                    global_index = global_start + local_index
                    context = min(local_index + 1, len(window) - local_index)
                    if context > best_context[global_index]:
                        best_context[global_index] = context
                        best_probabilities[global_index] = probabilities[local_batch_index, local_index]

    entities = []
    for span, probabilities in zip(spans, best_probabilities):
        label_id = int(probabilities.argmax())
        label = LABELS[label_id]
        if label == "O":
            continue
        if label.startswith("B-") and not any(character.isalnum() for character in span.value):
            continue
        kind = label.removeprefix("B-").removeprefix("I-")
        confidence = float(probabilities[label_id])
        if (label.startswith("I-") and entities and entities[-1]["type"] == kind
                and span.start <= entities[-1]["end"] + 1):
            entities[-1]["end"] = span.end
            entities[-1]["entity"] = text[entities[-1]["start"]:span.end]
            entities[-1]["confidence"] = min(entities[-1]["confidence"], confidence)
        else:
            entities.append({"entity": span.value, "type": kind, "start": span.start,
                             "end": span.end, "confidence": confidence, "source": "bilstm"})
    return entities


@lru_cache(maxsize=6)
def _classical_model(filename: str):
    import joblib
    return joblib.load(MODEL_DIR / filename)


def analyze(text: str, model_name: str, threshold: float = .80, fallback: bool = True) -> dict:
    """Analyze text without allowing a model failure to crash the caller."""
    try:
        if model_name == "Pattern detector":
            result = privacy_guard(text, threshold); result.update(model=model_name, note="")
            return result
        status = available_models().get(model_name)
        if status is None:
            raise ValueError(f"Unknown model: {model_name}")
        if not status["ready"]:
            raise RuntimeError(f"{model_name} artifact is missing or uses an outdated label schema")
        if model_name == "BERT":
            return _finish(text, _predict_bert(text), threshold, model_name,
                           "Model-only neural inference; no rule-based entities were added.")
        if model_name == "BiLSTM":
            return _finish(text, _predict_lstm(text), threshold, model_name,
                           "Model-only neural inference; no rule-based entities were added.")
        files = {
            "Logistic Regression": "logistic_regression.joblib",
            "Logistic Regression (Scratch)": "logistic_regression_scratch.joblib",
            "Naive Bayes": "naive_bayes.joblib",
            "Naive Bayes (Scratch)": "naive_bayes_scratch.joblib",
            "SVM": "svm.joblib",
            "SVM (Scratch)": "svm_scratch.joblib",
        }
        artifact = _classical_model(files[model_name])
        if model_name.endswith("(Scratch)"):
            if not isinstance(artifact, dict) or not {"vectorizer", "classifier"} <= artifact.keys():
                raise RuntimeError(f"{model_name} artifact is invalid; retrain it")
            features = artifact["vectorizer"].transform([text])
            prediction = int(artifact["classifier"].predict(features)[0])
        else:
            prediction = int(artifact.predict([text])[0])
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
