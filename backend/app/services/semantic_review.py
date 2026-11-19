from __future__ import annotations

import os
import re
import threading
from typing import Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

_LOCK = threading.Lock()
_MODEL_NAME: str | None = None
_TOKENIZER = None
_MODEL = None
_LOAD_ERROR: str | None = None


def _sentences(text: str, limit: int = 6) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", (text or "").strip())
    return [part.strip()[:700] for part in parts if len(part.strip()) >= 12][:limit]


def _load_model(model_name: str) -> tuple[Any, Any]:
    global _MODEL_NAME, _TOKENIZER, _MODEL, _LOAD_ERROR
    with _LOCK:
        if _MODEL_NAME == model_name and _MODEL is not None:
            return _TOKENIZER, _MODEL
        if _MODEL_NAME == model_name and _LOAD_ERROR:
            raise RuntimeError(_LOAD_ERROR)
        _MODEL_NAME, _TOKENIZER, _MODEL, _LOAD_ERROR = model_name, None, None, None
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
            model.eval()
            _TOKENIZER, _MODEL = tokenizer, model
            return tokenizer, model
        except Exception as exc:
            _LOAD_ERROR = f"{type(exc).__name__}: {str(exc)[:180]}"
            raise RuntimeError(_LOAD_ERROR) from exc


def _label_indices(model) -> dict[str, int]:
    result = {}
    for raw_index, raw_label in model.config.id2label.items():
        label = re.sub(r"[^a-z]", "", str(raw_label).lower())
        if label in {"entailment", "contradiction", "neutral"}:
            result[label] = int(raw_index)
    if set(result) != {"entailment", "contradiction", "neutral"}:
        raise ValueError("Configured NLI checkpoint must expose entailment, contradiction and neutral labels.")
    return result


def _score_pairs(tokenizer, model, pairs: list[tuple[str, str]]) -> list[dict[str, float]]:
    labels = _label_indices(model)
    output = []
    for start in range(0, len(pairs), 8):
        batch = pairs[start:start + 8]
        inputs = tokenizer(
            [item[0] for item in batch],
            [item[1] for item in batch],
            padding=True, truncation=True, max_length=384, return_tensors="pt",
        )
        with torch.inference_mode():
            logits = model(**inputs).logits.float()
            probabilities = torch.softmax(logits, dim=-1).cpu().tolist()
        for row in probabilities:
            output.append({
                name: round(float(row[index]), 5)
                for name, index in labels.items()
            })
    return output


def compare_semantic_sources(sources: list[dict[str, Any]]) -> dict[str, Any]:
    """Optional local NLI review; reports model candidates, never truth verdicts."""
    model_name = os.getenv("TRUTHLENS_NLI_MODEL", "").strip()
    if not model_name:
        return {
            "status": "not_configured",
            "model": None,
            "comparisons": [],
            "note": "Set TRUTHLENS_NLI_MODEL to enable optional local entailment/contradiction review. Lexical overlap remains available.",
        }
    prepared = []
    for source in sources[:8]:
        text = str(source.get("text") or "").strip()
        sentences = _sentences(text)
        if sentences:
            prepared.append({
                "source_id": str(source.get("source_id") or ""),
                "label": str(source.get("label") or "Evidence source"),
                "source_type": str(source.get("source_type") or "text"),
                "sentences": sentences,
            })
    if len(prepared) < 2:
        return {
            "status": "insufficient_sources",
            "model": model_name,
            "comparisons": [],
            "note": "At least two text-bearing sources are required for semantic comparison.",
        }

    candidate_pairs: list[dict[str, Any]] = []
    for index, left in enumerate(prepared):
        for right in prepared[index + 1:]:
            for left_sentence in left["sentences"]:
                for right_sentence in right["sentences"]:
                    if len(candidate_pairs) >= 32:
                        break
                    candidate_pairs.append({
                        "left": left, "right": right,
                        "left_sentence": left_sentence,
                        "right_sentence": right_sentence,
                    })
                if len(candidate_pairs) >= 32:
                    break
            if len(candidate_pairs) >= 32:
                break
        if len(candidate_pairs) >= 32:
            break
    try:
        tokenizer, model = _load_model(model_name)
        directed = []
        for item in candidate_pairs:
            directed.append((item["left_sentence"], item["right_sentence"]))
            directed.append((item["right_sentence"], item["left_sentence"]))
        scores = _score_pairs(tokenizer, model, directed)
    except Exception as exc:
        return {
            "status": "unavailable",
            "model": model_name,
            "comparisons": [],
            "error": str(exc)[:220],
            "note": "The configured NLI model could not produce semantic scores. No semantic conclusion was inferred.",
        }

    comparisons = []
    for index, item in enumerate(candidate_pairs):
        forward, reverse = scores[index * 2:index * 2 + 2]
        contradiction = max(forward["contradiction"], reverse["contradiction"])
        mutual_entailment = min(forward["entailment"], reverse["entailment"])
        if contradiction >= 0.70 and contradiction > max(forward["entailment"], reverse["entailment"]):
            label = "contradiction_candidate"
        elif mutual_entailment >= 0.70:
            label = "mutual_entailment_candidate"
        elif max(forward["entailment"], reverse["entailment"]) >= 0.70:
            label = "one_way_entailment_candidate"
        else:
            label = "neutral_or_uncertain"
        comparisons.append({
            "source_a_id": item["left"]["source_id"],
            "source_a": item["left"]["label"],
            "source_a_kind": item["left"]["source_type"],
            "source_b_id": item["right"]["source_id"],
            "source_b": item["right"]["label"],
            "source_b_kind": item["right"]["source_type"],
            "sentence_a": item["left_sentence"],
            "sentence_b": item["right_sentence"],
            "a_to_b": forward,
            "b_to_a": reverse,
            "review_label": label,
            "score_kind": "uncalibrated_nli_softmax",
        })
    rank = {
        "contradiction_candidate": 0,
        "one_way_entailment_candidate": 1,
        "mutual_entailment_candidate": 2,
        "neutral_or_uncertain": 3,
    }
    comparisons.sort(key=lambda item: (rank[item["review_label"]], -max(item["a_to_b"]["contradiction"], item["b_to_a"]["contradiction"])))
    return {
        "status": "completed",
        "model": model_name,
        "method": "Bidirectional sentence-pair natural language inference",
        "comparison_count": len(comparisons),
        "comparisons": comparisons,
        "thresholds": {"candidate_label_threshold": 0.70},
        "limitation": (
            "NLI labels are model-generated review candidates, not verified facts. Softmax scores are uncalibrated; "
            "negation, dates, attribution, OCR errors, quoted speech and domain shift can cause errors. "
            "An investigator must inspect the original source and context before accepting any candidate."
        ),
    }
