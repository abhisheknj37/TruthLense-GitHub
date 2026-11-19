from app.services.correlation import correlate_text_evidence


def test_text_correlation_reports_overlap_without_truth_claim():
    assets = [
        {
            "asset_id": "a1",
            "filename": "image.png",
            "text_extraction": {"text": "The city council approved a new public transport plan."},
        }
    ]
    result = correlate_text_evidence(assets, "The city council approved a new public transport plan yesterday.")
    assert result[0]["type"] == "text_evidence_overlap"
    assert result[0]["status"] == "lexical_overlap_found"
    assert "does not establish" in result[0]["detail"]
    assert 0 <= result[0]["jaccard_overlap"] <= 1


def test_text_correlation_explains_insufficient_sources():
    result = correlate_text_evidence([], "")
    assert result[0]["status"] == "insufficient_text_sources"


def test_text_correlation_does_not_call_low_overlap_a_contradiction():
    assets = [
        {"filename": "one.txt", "text_extraction": {"text": "The weather is sunny today."}},
        {"filename": "two.txt", "text_extraction": {"text": "A new scientific discovery was announced."}},
    ]
    result = correlate_text_evidence(assets)
    assert result[0]["status"] == "low_lexical_overlap"
    assert "not proof of contradiction" in result[0]["detail"]
