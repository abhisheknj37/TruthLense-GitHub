import asyncio

from app.services import claim_verification


def test_abstract_reconstruction_preserves_word_positions():
    result = claim_verification._abstract_from_inverted_index({
        "The": [0], "bridge": [1], "reopened": [2], "Monday": [3]
    })
    assert result == "The bridge reopened Monday"


def test_verify_claims_adds_semantic_candidate_review_without_truth_verdict(monkeypatch):
    async def fake_search(claim):
        return {
            "status": "sources_retrieved",
            "provider": "OpenAlex",
            "candidate_sources": [{
                "title": "Bridge inspection report",
                "openalex_id": "W123",
                "doi": "https://doi.org/10.example",
                "abstract_available": True,
                "abstract_excerpt": "The bridge was closed on Monday after an inspection.",
            }],
        }

    def fake_semantic(sources):
        assert sources[0]["source_type"] == "claim"
        assert sources[1]["source_type"] == "scholarly_abstract"
        return {
            "status": "completed",
            "comparisons": [{"review_label": "contradiction_candidate"}],
            "limitation": "Candidate only.",
        }

    async def run_in_thread(func, *args):
        return func(*args)

    monkeypatch.setattr(claim_verification, "search_scholarly_sources", fake_search)
    monkeypatch.setattr(claim_verification, "compare_semantic_sources", fake_semantic)
    monkeypatch.setattr(claim_verification.asyncio, "to_thread", run_in_thread)

    text = "The bridge was open on Monday after the inspection had finished."
    result = asyncio.run(claim_verification.verify_claims(text))
    assert result["claims_reviewed"] == 1
    item = result["results"][0]
    assert item["semantic_review"]["comparisons"][0]["review_label"] == "contradiction_candidate"
    assert item["verdict"] == "needs_human_review"
    assert item["confidence"] is None
    assert "No claim is labelled true or false" in result["important_limit"]


def test_verify_claims_does_not_claim_semantic_support_when_abstract_is_missing(monkeypatch):
    async def fake_search(claim):
        return {"status": "sources_retrieved", "candidate_sources": [{"title": "Candidate"}]}

    async def run_in_thread(func, *args):
        return func(*args)

    monkeypatch.setattr(claim_verification, "search_scholarly_sources", fake_search)
    monkeypatch.setattr(claim_verification.asyncio, "to_thread", run_in_thread)
    result = asyncio.run(
        claim_verification.verify_claims(
            "The city council approved a public transport plan on Monday."
        )
    )
    assert result["results"][0]["semantic_review"]["status"] == "not_run"
    assert result["results"][0]["verdict"] == "needs_human_review"
