from __future__ import annotations

import asyncio
import re
from typing import Any

import httpx

from app.services.semantic_review import compare_semantic_sources


def extract_claims(text: str, limit: int = 8) -> list[str]:
    claims = []
    for sentence in re.split(r"(?<=[.!?])\s+", text.strip()):
        sentence = re.sub(r"\s+", " ", sentence).strip()
        if len(sentence) >= 35 and re.search(r"\b(is|are|was|were|has|have|causes|increases|decreases|found|shows|reported|according)\b", sentence, re.I):
            claims.append(sentence[:500])
        if len(claims) >= limit:
            break
    return claims


def _abstract_from_inverted_index(index: Any, max_words: int = 650) -> str:
    if not isinstance(index, dict):
        return ""
    positions = [
        int(position)
        for values in index.values()
        if isinstance(values, list)
        for position in values
        if isinstance(position, int) and 0 <= position < max_words
    ]
    if not positions:
        return ""
    words = [""] * (max(positions) + 1)
    for token, token_positions in index.items():
        if not isinstance(token_positions, list):
            continue
        for position in token_positions:
            if isinstance(position, int) and 0 <= position < len(words):
                words[position] = str(token)
    return " ".join(word for word in words if word).strip()[:5000]


async def search_scholarly_sources(claim: str, limit: int = 5) -> dict[str, Any]:
    """Retrieve related scholarly candidates and available abstracts, not fact adjudication."""
    params = {
        "search": claim[:350],
        "per_page": max(1, min(limit, 5)),
        "select": "id,doi,title,publication_year,primary_location,abstract_inverted_index",
    }
    try:
        async with httpx.AsyncClient(timeout=12, headers={"User-Agent": "TruthLens/0.1 (research evidence lookup)"}) as client:
            response = await client.get("https://api.openalex.org/works", params=params)
            response.raise_for_status()
            payload = response.json()
        sources = []
        for item in payload.get("results", []):
            location = item.get("primary_location") or {}
            source = location.get("source") or {}
            abstract = _abstract_from_inverted_index(item.get("abstract_inverted_index"))
            sources.append({
                "title": item.get("title"),
                "year": item.get("publication_year"),
                "doi": item.get("doi"),
                "url": item.get("doi") or item.get("id"),
                "venue": source.get("display_name"),
                "openalex_id": item.get("id"),
                "abstract_available": bool(abstract),
                "abstract_excerpt": abstract[:1800] if abstract else None,
            })
        return {
            "status": "sources_retrieved",
            "provider": "OpenAlex",
            "query": claim,
            "candidate_sources": sources,
            "note": "Related scholarly records are candidates only. Abstract text may be incomplete or unavailable; inspect the original paper and relevant passages before using it as evidence.",
        }
    except Exception as exc:
        return {
            "status": "source_lookup_unavailable",
            "provider": "OpenAlex",
            "error": type(exc).__name__,
            "candidate_sources": [],
            "note": "External source retrieval failed; no claim verdict was produced.",
        }

async def verify_claims(text: str) -> dict[str, Any]:
    claims = extract_claims(text)
    if not claims:
        return {
            "status": "no_claims_extracted",
            "method": "sentence-level claim extraction and OpenAlex scholarly candidate discovery",
            "claims_reviewed": 0,
            "results": [],
            "important_limit": "No claim is labelled true or false. Candidate-source similarity is not evidence entailment or fact-checking.",
        }

    semaphore = asyncio.Semaphore(3)

    async def retrieve(claim: str):
        async with semaphore:
            return await search_scholarly_sources(claim)

    lookups = await asyncio.gather(*(retrieve(claim) for claim in claims))
    results = []
    for claim, lookup in zip(claims, lookups):
        abstracts = [
            item for item in lookup.get("candidate_sources", [])
            if item.get("abstract_excerpt")
        ]
        semantic = {
            "status": "not_run",
            "comparisons": [],
            "note": "No abstract passages were available for semantic review.",
        }
        if abstracts:
            sources = [{
                "source_id": "investigated-claim",
                "label": "Investigator-supplied claim",
                "source_type": "claim",
                "text": claim,
            }]
            sources.extend({
                "source_id": str(item.get("openalex_id") or item.get("doi") or index),
                "label": str(item.get("title") or "Scholarly candidate"),
                "source_type": "scholarly_abstract",
                "text": str(item["abstract_excerpt"]),
            } for index, item in enumerate(abstracts))
            semantic = await asyncio.to_thread(compare_semantic_sources, sources)
        results.append({
            "claim": claim,
            **lookup,
            "semantic_review": semantic,
            "verdict": "needs_human_review",
            "confidence": None,
        })
    return {
        "status": "completed",
        "method": "sentence-level claim extraction, OpenAlex candidate discovery and optional bidirectional NLI against available abstracts",
        "claims_reviewed": len(results),
        "results": results,
        "important_limit": (
            "No claim is labelled true or false. OpenAlex relevance and NLI entailment/contradiction candidates "
            "are not fact verification. Review full original papers, dates, attribution and source context."
        ),
    }
