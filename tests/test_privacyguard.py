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
    result = analyze("Call 01712345678", "Logistic Regression")
    assert result["safe_text"] == "Call [PHONE_NUMBER]"
    assert result["requested_model"] == "Logistic Regression"

def test_model_registry_contains_all_ui_models():
    assert set(available_models()) == {"Pattern detector", "BERT", "BiLSTM",
        "Logistic Regression", "Naive Bayes", "SVM"}
