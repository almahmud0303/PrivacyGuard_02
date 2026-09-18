"""PII entity detection with deterministic multilingual patterns."""
from __future__ import annotations
import re
from typing import Any

PATTERNS = (
    ("EMAIL", re.compile(r"(?<![\w.@])[\w.+-]+@[\w-]+(?:\.[\w-]+)+", re.UNICODE), .99),
    ("CREDIT_CARD", re.compile(r"(?<!\d)(?:\d[ -]?){15,19}(?!\d)"), .96),
    ("PHONE", re.compile(r"(?<!\d)(?:\+?88[- ]?)?01[3-9](?:[- ]?\d){8}(?!\d)"), .99),
    ("NID", re.compile(r"(?<!\d)(?:\d{10}|\d{13}|\d{17})(?!\d)"), .95),
    ("IP_ADDRESS", re.compile(r"\b(?:25[0-5]|2[0-4]\d|1?\d?\d)(?:\.(?:25[0-5]|2[0-4]\d|1?\d?\d)){3}\b"), .98),
)

def _luhn_valid(value: str) -> bool:
    digits = [int(c) for c in value if c.isdigit()]
    if not 13 <= len(digits) <= 19:
        return False
    total = 0
    for i, digit in enumerate(reversed(digits)):
        if i % 2:
            digit = digit * 2 - (9 if digit * 2 > 9 else 0)
        total += digit
    return total % 10 == 0

def detect_entities(text: str) -> list[dict[str, Any]]:
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    candidates = []
    for kind, pattern, confidence in PATTERNS:
        for match in pattern.finditer(text):
            if kind == "CREDIT_CARD" and not _luhn_valid(match.group()):
                continue
            candidates.append({"entity": match.group(), "type": kind, "start": match.start(),
                               "end": match.end(), "confidence": confidence, "source": "pattern"})
    candidates.sort(key=lambda e: (-e["confidence"], -(e["end"] - e["start"])))
    accepted = []
    for item in candidates:
        if not any(item["start"] < old["end"] and item["end"] > old["start"] for old in accepted):
            accepted.append(item)
    return sorted(accepted, key=lambda e: e["start"])
