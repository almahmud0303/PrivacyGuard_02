"""High-precision structured and contextual multilingual PII detection."""
from __future__ import annotations

import re
import unicodedata
from typing import Any

WORD = re.compile(r"[^\s,;.!?।:]+", re.UNICODE)
PHONE = re.compile(r"(?<!\d)(?:(?:\+?88|\+?৮৮)[- ]?)?(?:01|০১)[3-9৩-৯](?:[- ]?\d){8}(?!\d)")
STRUCTURED_PATTERNS = (
    ("EMAIL", re.compile(r"(?<![\w.@])[\w.+-]+@[\w-]+(?:\.[\w-]+)+", re.UNICODE), .99),
    ("CREDIT_CARD", re.compile(r"(?<!\d)(?:\d[ -]?){15,19}(?!\d)"), .96),
    ("PHONE", PHONE, .99),
    ("NID", re.compile(r"(?<!\d)(?:\d{10}|\d{13}|\d{17})(?!\d)"), .97),
    ("IP_ADDRESS", re.compile(r"\b(?:25[0-5]|2[0-4]\d|1?\d?\d)(?:\.(?:25[0-5]|2[0-4]\d|1?\d?\d)){3}\b"), .98),
)

PERSON_CUE = re.compile(
    r"(?:\b(?:my|his|her|their)\s+name\s+is\b|\bname\s+is\b|"
    r"\b(?:he|she)\s+is\b|\bthis\s+is\b|\bname\s*[:=-]|"
    r"\bamar\s+nam(?:e)?(?:\s+holo)?\b|\btar\s+nam(?:e)?(?:\s+holo)?\b|"
    r"(?:আমার|তার)\s+নাম(?:\s+হলো)?|তিনি\s+হলেন)\s*",
    re.IGNORECASE | re.UNICODE,
)
LOCATION_CUE = re.compile(
    r"(?:\bi\s+live\s+(?:at|in)\b|\blives?\s+in\b|\baddress\s*(?:is|[:=-])|"
    r"\bami\s+thaki\b|\bfrom\b|আমি\s+থাকি|আমার\s+ঠিকানা(?:\s+হলো)?)\s*",
    re.IGNORECASE | re.UNICODE,
)
STOP_WORDS = {
    "and", "or", "but", "with", "phone", "mobile", "email", "nid", "address",
    "lives", "live", "from", "amar", "ami", "my", "is", "at", "in", "এবং",
    "ফোন", "ইমেইল", "এনআইডি", "ঠিকানা",
}
KNOWN_LOCATIONS = (
    "Dhaka", "Chattogram", "Chittagong", "Rajshahi", "Khulna", "Barishal", "Barisal",
    "Sylhet", "Rangpur", "Mymensingh", "Comilla", "Cumilla", "Gazipur", "Narayanganj",
    "Bangladesh", "New York", "United States", "USA", "ঢাকা", "চট্টগ্রাম", "রাজশাহী",
    "খুলনা", "বরিশাল", "সিলেট", "রংপুর", "ময়মনসিংহ", "বাংলাদেশ",
)


def _ascii_digits(value: str) -> str:
    return "".join(str(int(char)) if char.isdigit() else char for char in value)


def _is_word(value: str) -> bool:
    return bool(value) and all(unicodedata.category(char)[0] in {"L", "M"} or char in "-'" for char in value)


def _luhn_valid(value: str) -> bool:
    digits = [int(char) for char in _ascii_digits(value) if char.isdigit()]
    if not 13 <= len(digits) <= 19:
        return False
    total = 0
    for index, digit in enumerate(reversed(digits)):
        if index % 2:
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit
    return total % 10 == 0


def _context_entities(text: str, cue: re.Pattern[str], kind: str, maximum: int) -> list[dict[str, Any]]:
    entities = []
    for cue_match in cue.finditer(text):
        words = []
        for match in WORD.finditer(text, cue_match.end()):
            # Do not jump across punctuation or a new clause.
            gap = text[cue_match.end() if not words else words[-1].end():match.start()]
            if re.search(r"[,;.!?।\n]", gap):
                break
            if not _is_word(match.group()):
                break
            if match.group().casefold() in STOP_WORDS:
                break
            words.append(match)
            if len(words) == maximum:
                break
        if words:
            start, end = words[0].start(), words[-1].end()
            entities.append({"entity": text[start:end], "type": kind, "start": start, "end": end,
                             "confidence": .92, "source": "context"})
    return entities


def _known_locations(text: str) -> list[dict[str, Any]]:
    result = []
    for value in sorted(KNOWN_LOCATIONS, key=len, reverse=True):
        pattern = re.compile(rf"(?<![^\W\d_]){re.escape(value)}(?![^\W\d_])", re.IGNORECASE | re.UNICODE)
        for match in pattern.finditer(text):
            result.append({"entity": match.group(), "type": "LOCATION", "start": match.start(), "end": match.end(),
                           "confidence": .90, "source": "gazetteer"})
    return result


def _subject_people(text: str) -> list[dict[str, Any]]:
    """Recognize a multi-word subject in constructions such as 'John Smith lives in'."""
    result = []
    pattern = re.compile(r"(?:^|(?<=[.!?।])\s)(?P<name>.+?)\s+lives?\s+in\b", re.IGNORECASE | re.UNICODE)
    for match in pattern.finditer(text):
        value = match.group("name").strip()
        words = [item for item in WORD.finditer(value) if _is_word(item.group())]
        if 2 <= len(words) <= 4 and " ".join(item.group() for item in words) == value:
            start = match.start("name")
            result.append({"entity": value, "type": "PERSON", "start": start, "end": start + len(value),
                           "confidence": .91, "source": "context"})
    return result


def _whole_text_name(text: str) -> list[dict[str, Any]]:
    stripped = text.strip(" \t\r\n.,!?।")
    words = [match for match in WORD.finditer(stripped) if _is_word(match.group())]
    if 2 <= len(words) <= 4 and " ".join(m.group() for m in words) == stripped:
        lowered = {m.group().casefold() for m in words}
        if not lowered.intersection(STOP_WORDS):
            start = text.find(stripped)
            return [{"entity": stripped, "type": "PERSON", "start": start, "end": start + len(stripped),
                     "confidence": .85, "source": "name-phrase"}]
    return []


def detect_entities(text: str) -> list[dict[str, Any]]:
    """Return ordered, non-overlapping entities with exact character offsets."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    candidates: list[dict[str, Any]] = []
    for kind, pattern, confidence in STRUCTURED_PATTERNS:
        for match in pattern.finditer(text):
            if kind == "CREDIT_CARD" and not _luhn_valid(match.group()):
                continue
            candidates.append({"entity": match.group(), "type": kind, "start": match.start(), "end": match.end(),
                               "confidence": confidence, "source": "pattern"})
    candidates.extend(_context_entities(text, PERSON_CUE, "PERSON", 4))
    candidates.extend(_subject_people(text))
    candidates.extend(_context_entities(text, LOCATION_CUE, "LOCATION", 5))
    candidates.extend(_known_locations(text))
    if not candidates:
        candidates.extend(_whole_text_name(text))

    # Prefer structured patterns, then contextual spans, then longer candidates.
    source_rank = {"pattern": 3, "context": 2, "gazetteer": 1, "name-phrase": 0}
    candidates.sort(key=lambda e: (-source_rank[e["source"]], -e["confidence"], -(e["end"] - e["start"])))
    accepted: list[dict[str, Any]] = []
    for item in candidates:
        if not any(item["start"] < old["end"] and item["end"] > old["start"] for old in accepted):
            accepted.append(item)
    return sorted(accepted, key=lambda e: e["start"])
