"""Public PrivacyGuard inference interface."""
from .entity_detector import detect_entities
from .redactor import redact_text
from .risk_score import calculate_risk, calculate_total_risk

def privacy_guard(text: str, confidence_threshold: float = .80) -> dict:
    entities = detect_entities(text)
    for entity in entities:
        risk = calculate_risk(entity["type"])
        entity.update(risk=risk["level"], score=risk["score"])
    return {"original": text, "entities": entities, "entity_count": len(entities),
            "overall_risk": calculate_total_risk(entities),
            "safe_text": redact_text(text, entities, confidence_threshold)}
