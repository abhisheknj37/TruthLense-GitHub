from __future__ import annotations

import re
from typing import Any

import httpx


def extract_claims(text: str, limit: int = 8) -> list[str]:
    claims = []
    for sentence in re.split(r"(?<=[.!?])\s+", text.strip()):
        sentence = re.sub(r"\s+", " ", sentence).strip()
        if len(sentence) >= 35 and re.search(r"\b(is|are|was|were|has|have|causes|increases|decreases|found|shows|reported|according)\b", sentence, re.I):
            claims.append(sentence[:500])
        if len(claims) >= limit:
            break
    return claims


async def search_scholarly_sources(claim: str, limit: int = 5) -> dict[str, Any]:
    """Find related scholarly records from OpenAlex; this is not fact adjudication."""
    params = {"search": claim[:350], "per_page": limit, "select": "id,doi,title,publication_year,primary_location"}
    try:
        async with httpx.AsyncClient(timeout=12, headers={"User-Agent": "TruthLens/0.1 (research evidence lookup)"}) as client:
            response = await client.get("https://api.openalex.org/works", params=params)
            response.raise_for_status()
            payload = response.json()
        sources = []
        for item in payload.get("results", []):
            location = item.get("primary_location") or {}
            source = location.get("source") or {}
            sources.append({"title": item.get("title"), "year": item.get("publication_year"), "doi": item.get("doi"), "url": item.get("doi") or item.get("id"), "venue": source.get("display_name"), "openalex_id": item.get("id")})
        return {"status": "sources_retrieved", "provider": "OpenAlex", "query": claim, "candidate_sources": sources, "note": "Related scholarly records are not proof that a claim is supported; inspect original papers and relevant passages."}
    except Exception as exc:
        return {"status": "source_lookup_unavailable", "provider": "OpenAlex", "error": type(exc).__name__, "candidate_sources": [], "note": "External source retrieval failed; no claim verdict was produced."}


async def verify_claims(text: str) -> dict[str, Any]:
    results = []
    for claim in extract_claims(text):
        lookup = await search_scholarly_sources(claim)
        results.append({"claim": claim, **lookup, "verdict": "needs_human_review", "confidence": None})
    return {"status": "completed" if results else "no_claims_extracted", "method": "sentence-level claim extraction and OpenAlex scholarly source discovery", "claims_reviewed": len(results), "results": results, "important_limit": "No claim is labelled true or false. Candidate-source similarity is not evidence entailment or fact-checking."}
