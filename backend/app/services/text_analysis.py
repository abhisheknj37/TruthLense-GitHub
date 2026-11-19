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


_TOKEN_RE = re.compile(r"[a-z0-9]{3,}", re.I)
_STOPWORDS = {
    "about", "after", "again", "against", "also", "because", "before", "being",
    "between", "could", "does", "doing", "each", "from", "have", "into", "more",
    "most", "other", "over", "same", "some", "such", "than", "that", "their",
    "them", "there", "these", "they", "this", "those", "through", "under", "very",
    "what", "when", "where", "which", "while", "with", "would", "your",
}


def compare_text_sources(sources: list[dict[str, Any]]) -> dict[str, Any]:
    """Compare OCR, transcript and user-supplied text by lexical overlap only."""
    prepared = []
    for index, source in enumerate(sources or []):
        raw = str(source.get("text") or "").strip()
        if not raw:
            continue
        tokens = {
            token.lower() for token in _TOKEN_RE.findall(raw)
            if token.lower() not in _STOPWORDS
        }
        prepared.append({
            "source_id": str(source.get("source_id") or f"text-{index + 1}"),
            "source_type": str(source.get("source_type") or "text"),
            "label": str(source.get("label") or f"Text source {index + 1}"),
            "filename": source.get("filename"),
            "character_count": len(raw),
            "word_count": len(_TOKEN_RE.findall(raw)),
            "text_preview": raw[:6000],
            "tokens": tokens,
        })

    comparisons = []
    for index, left in enumerate(prepared):
        for right in prepared[index + 1:]:
            shared = left["tokens"] & right["tokens"]
            union = left["tokens"] | right["tokens"]
            score = len(shared) / len(union) if union else 0.0
            comparisons.append({
                "left_source_id": left["source_id"],
                "left_label": left["label"],
                "right_source_id": right["source_id"],
                "right_label": right["label"],
                "jaccard_similarity": round(score, 4),
                "shared_terms": sorted(shared)[:20],
                "left_only_terms": sorted(left["tokens"] - right["tokens"])[:12],
                "right_only_terms": sorted(right["tokens"] - left["tokens"])[:12],
                "interpretation": (
                    "High lexical overlap" if score >= 0.5
                    else "Moderate lexical overlap" if score >= 0.2
                    else "Low lexical overlap"
                ),
            })

    public_sources = [
        {key: value for key, value in source.items() if key != "tokens"}
        for source in prepared
    ]
    return {
        "status": "completed" if len(prepared) >= 2 else "insufficient_sources",
        "method": "Case-insensitive token-set Jaccard overlap after common-word removal",
        "comparison_count": len(comparisons),
        "source_count": len(public_sources),
        "sources": public_sources,
        "comparisons": comparisons,
        "limitation": (
            "Lexical overlap highlights shared and different wording only. It does not detect semantic contradiction, "
            "establish that two sources describe the same event, or verify whether a claim is true."
        ),
    }
