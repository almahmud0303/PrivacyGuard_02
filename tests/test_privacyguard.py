from src.redection.privacy_firewall import privacy_guard
from src.inference.model_service import analyze, available_models

def test_detects_and_redacts_multiple_entities():
    result = privacy_guard("Call 01712345678 or mail me@example.com. NID 1234567890")
    assert {item["type"] for item in result["entities"]} == {"PHONE", "EMAIL", "NID"}
    assert result["safe_text"] == "Call [PHONE_NUMBER] or mail [EMAIL]. NID [NID]"
    assert result["overall_risk"] == "CRITICAL"

def test_bengali_context_preserved():
    assert privacy_guard("আমার ফোন 01898765432")["safe_text"] == "আমার ফোন [PHONE_NUMBER]"

def test_invalid_credit_card_is_not_redacted():
    assert privacy_guard("Reference 1111 1111 1111 1111")["entity_count"] == 0

def test_untrained_model_falls_back_without_crashing():
    result = analyze("Call 01712345678", "Unavailable test model")
    assert result["safe_text"] == "Call [PHONE_NUMBER]"
    assert result["requested_model"] == "Unavailable test model"

def test_model_registry_contains_all_ui_models():
    assert set(available_models()) == {"Pattern detector", "BERT", "BiLSTM",
        "Logistic Regression", "Naive Bayes", "SVM"}

def test_multiword_person_location_and_nid_are_single_entities():
    result = privacy_guard("My name is Rysul aman nirob and I live in Dhaka Bangladesh. My NID is 1234567890.")
    assert [(item["entity"], item["type"]) for item in result["entities"]] == [
        ("Rysul aman nirob", "PERSON"), ("Dhaka Bangladesh", "LOCATION"), ("1234567890", "NID")]
    assert result["safe_text"] == "My name is [PERSON] and I live in [LOCATION]. My NID is [NID]."

def test_bengali_person_location_and_nid():
    result = privacy_guard("আমার নাম রাইসুল আমান নিরব এবং আমি থাকি ঢাকা। আমার এনআইডি ১২৩৪৫৬৭৮৯০")
    assert [(item["entity"], item["type"]) for item in result["entities"]] == [
        ("রাইসুল আমান নিরব", "PERSON"), ("ঢাকা", "LOCATION"), ("১২৩৪৫৬৭৮৯০", "NID")]

def test_bio_annotation_uses_inside_labels_for_multiword_entities():
    from src.dataset.annotation import annotate_sentence
    annotated = annotate_sentence("My name is Rysul aman nirob and I live in New York")
    assert annotated["labels"] == ["O", "O", "O", "B-PERSON", "I-PERSON", "I-PERSON",
        "O", "O", "O", "O", "B-LOCATION", "I-LOCATION"]

def test_person_subject_before_lives_is_grouped():
    result = privacy_guard("Rysul aman nirob lives in New York")
    assert [(item["entity"], item["type"]) for item in result["entities"]] == [
        ("Rysul aman nirob", "PERSON"), ("New York", "LOCATION")]

def test_name_is_and_pronoun_introductions_are_detected():
    result = privacy_guard("name is al mahmud. he is nirob")
    assert [(item["entity"], item["type"]) for item in result["entities"]] == [
        ("al mahmud", "PERSON"), ("nirob", "PERSON")]
    assert result["safe_text"] == "name is [PERSON]. he is [PERSON]"

def test_direct_ner_labels_cover_all_sensitive_types():
    from src.dataset.generate_ner_dataset import render, entity
    from src.models.transformer.labels import ENTITY_TYPES, MULTI_TOKEN_TYPES, label2id
    record = render([entity(kind, f"private {kind.lower()}" if kind in MULTI_TOKEN_TYPES else kind.lower())
                     for kind in ENTITY_TYPES])
    assert len(record["tokens"]) == len(record["labels"])
    assert set(record["labels"]).issubset(label2id)
    assert {label.removeprefix("B-") for label in record["labels"] if label.startswith("B-")} == set(ENTITY_TYPES)

def test_successful_bert_path_does_not_use_pattern_detector(monkeypatch):
    import src.inference.model_service as service
    monkeypatch.setattr(service, "available_models", lambda: {"BERT": {"ready": True, "kind": "entity"}})
    monkeypatch.setattr(service, "_predict_bert", lambda text: [{"entity": "al mahmud", "type": "PERSON",
        "start": 6, "end": 15, "confidence": .99, "source": "bert"}])
    monkeypatch.setattr(service, "privacy_guard", lambda *args: (_ for _ in ()).throw(AssertionError("fallback called")))
    result = service.analyze("he is al mahmud", "BERT")
    assert result["model"] == "BERT"
    assert result["safe_text"] == "he is [PERSON]"

def test_inference_tokenization_keeps_sensitive_values_whole():
    from src.inference.model_service import _lexical_spans
    text = r"Email sarah.ahmed\@gmail.com, phone +8801712345678. IDs EMP-45892 and CSE-2019-1025."
    values = [span.value for span in _lexical_spans(text)]
    assert r"sarah.ahmed\@gmail.com" in values
    assert "+8801712345678" in values
    assert "EMP-45892" in values
    assert "CSE-2019-1025" in values
