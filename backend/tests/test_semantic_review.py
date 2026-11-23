from app.services import semantic_review


def test_semantic_review_is_explicitly_optional_when_not_configured(monkeypatch):
    monkeypatch.delenv("TRUTHLENS_NLI_MODEL", raising=False)
    result = semantic_review.compare_semantic_sources([
        {"source_id": "a", "label": "Claim", "source_type": "claim", "text": "A bridge opened on Monday."},
        {"source_id": "b", "label": "Transcript", "source_type": "transcript", "text": "A bridge closed on Monday."},
    ])
    assert result["status"] == "not_configured"
    assert result["comparisons"] == []


def test_semantic_review_labels_bidirectional_candidates_without_fact_verdict(monkeypatch):
    class FakeModel:
        config = type("Config", (), {
            "id2label": {0: "contradiction", 1: "entailment", 2: "neutral"}
        })()

    monkeypatch.setenv("TRUTHLENS_NLI_MODEL", "fixture-nli")
    monkeypatch.setattr(semantic_review, "_load_model", lambda name: (object(), FakeModel()))
    monkeypatch.setattr(
        semantic_review,
        "_score_pairs",
        lambda tokenizer, model, pairs: [
            {"contradiction": 0.91, "entailment": 0.03, "neutral": 0.06}
            if index % 2 == 0 else
            {"contradiction": 0.88, "entailment": 0.04, "neutral": 0.08}
            for index, _ in enumerate(pairs)
        ],
    )
    result = semantic_review.compare_semantic_sources([
        {"source_id": "a", "label": "Claim", "source_type": "claim", "text": "The bridge opened after repairs."},
        {"source_id": "b", "label": "Transcript", "source_type": "transcript", "text": "The bridge remained closed after repairs."},
    ])
    assert result["status"] == "completed"
    assert result["comparisons"][0]["review_label"] == "contradiction_candidate"
    assert result["comparisons"][0]["score_kind"] == "uncalibrated_nli_softmax"
    assert "not verified facts" in result["limitation"]
