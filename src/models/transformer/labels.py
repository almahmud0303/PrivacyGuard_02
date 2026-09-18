ENTITY_TYPES = ["PERSON", "LOCATION", "ORGANIZATION", "ADDRESS", "PHONE", "EMAIL", "NID",
                "CREDIT_CARD", "ACCOUNT", "EMPLOYEE_ID", "MEDICAL_ID", "STUDENT_ID",
                "IP_ADDRESS", "DATE", "PASSPORT", "HEALTH_CONDITION", "OCCUPATION", "EDUCATION"]
MULTI_TOKEN_TYPES = {"PERSON", "LOCATION", "ORGANIZATION", "ADDRESS", "CREDIT_CARD", "DATE",
                     "HEALTH_CONDITION", "OCCUPATION", "EDUCATION"}
LABELS = ["O"]
for kind in ENTITY_TYPES:
    LABELS.append(f"B-{kind}")
    if kind in MULTI_TOKEN_TYPES:
        LABELS.append(f"I-{kind}")
label2id = {label: index for index, label in enumerate(LABELS)}
id2label = {index: label for index, label in enumerate(LABELS)}
