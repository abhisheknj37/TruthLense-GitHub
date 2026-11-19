from __future__ import annotations

import re
from collections import Counter
from typing import Any

URL_RE = re.compile(r"https?://[^\s]+", re.I)
DATE_RE = re.compile(r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{2}-\d{2})\b")
TIME_RE = re.compile(r"\b(?:[01]?\d|2[0-3]):[0-5]\d(?:\s?[AP]M)?\b", re.I)
CLAIM_TERMS = {
    "urgent_language": ("urgent", "breaking", "share immediately", "forward this", "viral"),
    "certainty_language": ("100% real", "confirmed", "undeniable", "proof", "definitely"),
    "misinformation_cues": ("doctors hate", "media won't show", "mainstream media", "secretly recorded"),
}


def analyze_text_context(text: str) -> dict[str, Any]:
    """Extract text review cues; does not decide whether a claim is true."""
    clean = (text or "").strip()
    if not clean:
        return {"status": "not_provided", "note": "No text, caption, transcript or claim was supplied."}
    lower = clean.lower()
    cues = []
    for code, terms in CLAIM_TERMS.items():
        found = [term for term in terms if term in lower]
        if found:
            cues.append({"type": code, "matched_terms": found, "interpretation": "A language pattern to review, not evidence that the claim is false."})
    words = re.findall(r"[\w']+", lower)
    return {
        "status": "review_cues_extracted",
        "character_count": len(clean),
        "word_count": len(words),
        "urls": URL_RE.findall(clean)[:30],
        "dates": DATE_RE.findall(clean)[:30],
        "times": TIME_RE.findall(clean)[:30],
        "language_cues": cues,
        "repeated_terms": [{"term": term, "count": count} for term, count in Counter(words).most_common(8) if count > 2],
        "claim_text": clean[:5000],
        "limitations": ["Rule-based triage only; not fact-checking or a trained misinformation detector.", "Dates, URLs and language cues must be checked against reliable primary sources.", "No claim is automatically labelled true or false."],
    }
