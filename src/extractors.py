"""Rule-based extractors that approximate the cheapest possible "generator"
for LoCoMo. No LLM. Purpose: quantify how much of LoCoMo's generation gap
is recoverable by trivial post-processing.

Implemented:
  - extract_session_date(text): parse '... on D Month, YYYY' → 'D Month YYYY'
  - extract_dates_from_text(text): scan free text for date patterns
  - extract_temporal_kitchen_sink(session_dt, body): combine session
    date, surrounding session dates, and all in-text date mentions
    into one whitespace-joined prediction string.
"""
from __future__ import annotations

import re

# Matches the LoCoMo session timestamp format like '2:21 pm on 8 May, 2023'
SESSION_DATE_RE = re.compile(
    r"on\s+(\d{1,2})\s+([A-Z][a-z]+),?\s+(\d{4})",
    re.IGNORECASE,
)

# In-body date patterns. We're explicit about the small set of forms that
# match LoCoMo gold answers; we are NOT trying to be a general date parser.
_MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December"
INBODY_DMY = re.compile(rf"\b(\d{{1,2}})\s+({_MONTHS})\s+(\d{{4}})\b", re.IGNORECASE)
INBODY_MDY = re.compile(rf"\b({_MONTHS})\s+(\d{{1,2}}),?\s+(\d{{4}})\b", re.IGNORECASE)
INBODY_MY  = re.compile(rf"\b({_MONTHS})\s+(\d{{4}})\b", re.IGNORECASE)
INBODY_DM  = re.compile(rf"\b(\d{{1,2}})\s+({_MONTHS})\b", re.IGNORECASE)
INBODY_YEAR = re.compile(r"\b(20\d{2})\b")


def extract_session_date(timestamp: str) -> str:
    """Return 'D Month YYYY' from a LoCoMo session timestamp, or '' if no match."""
    if not timestamp:
        return ""
    m = SESSION_DATE_RE.search(timestamp)
    if not m:
        return ""
    day, month, year = m.group(1), m.group(2), m.group(3)
    month = month.capitalize()
    return f"{day} {month} {year}"


def extract_dates_from_text(text: str) -> list[str]:
    """Scan free text for any date-like phrases. Returns canonicalized
    'D Month YYYY' / 'Month YYYY' / 'YYYY' strings, deduped, in document order."""
    if not text:
        return []
    found: list[str] = []
    seen: set[str] = set()
    def add(s: str):
        if s and s not in seen:
            found.append(s)
            seen.add(s)
    for m in INBODY_DMY.finditer(text):
        add(f"{m.group(1)} {m.group(2).capitalize()} {m.group(3)}")
    for m in INBODY_MDY.finditer(text):
        add(f"{m.group(2)} {m.group(1).capitalize()} {m.group(3)}")
    for m in INBODY_MY.finditer(text):
        add(f"{m.group(1).capitalize()} {m.group(2)}")
    for m in INBODY_DM.finditer(text):
        add(f"{m.group(1)} {m.group(2).capitalize()}")
    for m in INBODY_YEAR.finditer(text):
        add(m.group(1))
    return found


def extract_temporal_kitchen_sink(session_dt: str, body: str) -> str:
    """Combine session-date + all in-body date mentions into one prediction string.

    This is the "smarter regex" extractor used in run 014. Still no LLM.
    Whitespace-joined so LoCoMo's word-F1 metric can find token overlap.
    """
    parts: list[str] = []
    sd = extract_session_date(session_dt)
    if sd:
        parts.append(sd)
    parts.extend(extract_dates_from_text(body))
    return " ".join(parts)
