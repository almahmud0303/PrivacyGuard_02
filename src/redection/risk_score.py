RISK_LEVELS = {"LOCATION": {"level": "LOW", "score": 1}, "PERSON": {"level": "MEDIUM", "score": 2},
    "DATE": {"level": "MEDIUM", "score": 2}, "PHONE": {"level": "HIGH", "score": 3},
    "EMAIL": {"level": "HIGH", "score": 3}, "IP_ADDRESS": {"level": "HIGH", "score": 3},
    "ACCOUNT": {"level": "CRITICAL", "score": 4}, "NID": {"level": "CRITICAL", "score": 4},
    "CREDIT_CARD": {"level": "CRITICAL", "score": 4},
    "PASSPORT": {"level": "CRITICAL", "score": 4},
    "ORGANIZATION": {"level": "MEDIUM", "score": 2},
    "ADDRESS": {"level": "HIGH", "score": 3},
    "EMPLOYEE_ID": {"level": "HIGH", "score": 3},
    "MEDICAL_ID": {"level": "CRITICAL", "score": 4},
    "STUDENT_ID": {"level": "HIGH", "score": 3},
    "HEALTH_CONDITION": {"level": "CRITICAL", "score": 4},
    "OCCUPATION": {"level": "MEDIUM", "score": 2},
    "EDUCATION": {"level": "MEDIUM", "score": 2}}

def calculate_risk(entity_type: str) -> dict:
    return RISK_LEVELS.get(entity_type, {"level": "UNKNOWN", "score": 0}).copy()

def calculate_total_risk(results: list[dict]) -> str:
    maximum = max((int(item.get("score", 0)) for item in results), default=0)
    return {0: "LOW", 1: "LOW", 2: "MEDIUM", 3: "HIGH", 4: "CRITICAL"}[min(maximum, 4)]
