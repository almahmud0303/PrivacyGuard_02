"""Span-safe PII redaction."""
REPLACEMENT_MAP = {"PERSON": "[PERSON]", "EMAIL": "[EMAIL]", "PHONE": "[PHONE_NUMBER]",
    "NID": "[NID]", "LOCATION": "[LOCATION]", "ACCOUNT": "[ACCOUNT_NUMBER]",
    "CREDIT_CARD": "[CREDIT_CARD]", "IP_ADDRESS": "[IP_ADDRESS]", "DATE": "[DATE]",
    "PASSPORT": "[PASSPORT]", "ORGANIZATION": "[ORGANIZATION]", "ADDRESS": "[ADDRESS]",
    "EMPLOYEE_ID": "[EMPLOYEE_ID]", "MEDICAL_ID": "[MEDICAL_ID]",
    "STUDENT_ID": "[STUDENT_ID]", "HEALTH_CONDITION": "[HEALTH_CONDITION]",
    "OCCUPATION": "[OCCUPATION]", "EDUCATION": "[EDUCATION]"}

def should_redact(confidence: float, threshold: float = .80) -> bool:
    return confidence >= threshold

def redact_text(text: str, entities: list[dict], threshold: float = .80) -> str:
    selected = [e for e in entities if should_redact(float(e.get("confidence", 1)), threshold)]
    if all("start" in e and "end" in e for e in selected):
        output = text
        for entity in sorted(selected, key=lambda e: e["start"], reverse=True):
            marker = REPLACEMENT_MAP.get(entity["type"], "[PRIVATE_DATA]")
            output = output[:entity["start"]] + marker + output[entity["end"]:]
        return output
    output = text
    for entity in sorted(selected, key=lambda e: len(e["entity"]), reverse=True):
        output = output.replace(entity["entity"], REPLACEMENT_MAP.get(entity["type"], "[PRIVATE_DATA]"))
    return output
