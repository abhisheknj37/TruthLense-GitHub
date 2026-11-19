from __future__ import annotations

import re
from typing import Any

_STOPWORDS = {
    "the", "and", "for", "that", "with", "this", "from", "have", "has",
    "was", "were", "are", "you", "your", "they", "their", "then", "than",
    "into", "about", "after", "before", "when", "where", "what", "which",
}


def _tokens(value: str) -> set[str]:
    return {
        token for token in re.findall(r"[a-z0-9]{3,}", (value or "").lower())
        if token not in _STOPWORDS
    }


def _asset_text(asset: dict[str, Any]) -> str:
    extracted = asset.get("text_extraction") or {}
    pieces = [str(extracted.get("text") or "")]
    pieces.extend(str(frame.get("text") or "") for frame in extracted.get("frames", []))
    pieces.extend(str(chunk.get("text") or "") for chunk in extracted.get("chunks", []))
    return " ".join(piece for piece in pieces if piece).strip()


def correlate_modalities(assets: list[dict[str, Any]], paired_media: bool) -> list[dict[str, str]]:
    """Compare separately supplied modality indicators without inventing a fused probability."""
    if not paired_media:
        return [{
            "type": "cross_modal_review",
            "status": "not_assessed",
            "detail": "Cross-modal comparison was not run because the user did not confirm that the uploaded files belong to the same recording.",
        }]

    video_items = [item for item in assets if item.get("media_type") == "video"]
    audio_items = [item for item in assets if item.get("media_type") == "audio"]
    if not video_items or not audio_items:
        return [{
            "type": "cross_modal_review",
            "status": "insufficient_modalities",
            "detail": "Paired-media review requires at least one video and one separate audio file in the same case.",
        }]

    notes = []
    for video in video_items:
        video_result = video.get("video_analysis") or {}
        video_score = (video_result.get("scores") or {}).get("fake_median_across_face_frames")
        for audio in audio_items:
            audio_result = audio.get("model_assessment") or {}
            audio_score = (audio_result.get("scores") or {}).get("fake_median_across_windows")
            if video_score is None or audio_score is None:
                status = "inconclusive"
                detail = "At least one modality lacks a usable model score, so no signal comparison is possible."
            elif video_score >= 0.65 and audio_score >= 0.65:
                status = "model_signals_agree_fake_direction"
                detail = "Both separate model pipelines produced a fake-direction indicator. This is not independent proof and does not establish synchronization."
            elif video_score <= 0.35 and audio_score <= 0.35:
                status = "model_signals_agree_real_direction"
                detail = "Both separate model pipelines produced a real-direction indicator. This does not prove authenticity."
            elif (video_score >= 0.65 and audio_score <= 0.35) or (video_score <= 0.35 and audio_score >= 0.65):
                status = "modality_disagreement"
                detail = "The separate audio and video model indicators disagree. Keep the case inconclusive and review the source files."
            else:
                status = "inconclusive"
                detail = "At least one modality is within the ambiguous score band."
            notes.append({
                "type": "paired_audio_video_model_signal_comparison",
                "status": status,
                "detail": detail + " Scores are not calibrated probabilities.",
            })
    return notes


def correlate_text_evidence(
    assets: list[dict[str, Any]], context_text: str = ""
) -> list[dict[str, Any]]:
    """Compare extracted text across evidence files and user context using lexical overlap only."""
    sources: list[tuple[str, str, str]] = []
    if context_text.strip():
        sources.append(("user_context", "User-supplied context", context_text.strip()))
    for asset in assets:
        text = _asset_text(asset)
        if text:
            sources.append((
                str(asset.get("asset_id") or asset.get("filename") or "evidence"),
                str(asset.get("filename") or "Evidence file"),
                text,
            ))

    if len(sources) < 2:
        return [{
            "type": "text_evidence_correlation",
            "status": "insufficient_text_sources",
            "detail": "At least two text-bearing sources are required. OCR and transcription availability may depend on optional local tools.",
        }]

    findings: list[dict[str, Any]] = []
    for index, (source_id, source_name, source_text) in enumerate(sources):
        left = _tokens(source_text)
        for other_id, other_name, other_text in sources[index + 1:]:
            right = _tokens(other_text)
            union = left | right
            overlap = len(left & right) / len(union) if union else 0.0
            findings.append({
                "type": "text_evidence_overlap",
                "status": "lexical_overlap_found" if overlap >= 0.15 else "low_lexical_overlap",
                "source_a": source_name,
                "source_b": other_name,
                "source_a_id": source_id,
                "source_b_id": other_id,
                "jaccard_overlap": round(overlap, 4),
                "shared_terms": sorted(left & right)[:30],
                "detail": (
                    "Text sources share vocabulary. This does not establish that their claims agree or are true."
                    if overlap >= 0.15 else
                    "Little vocabulary is shared. This is not proof of contradiction; paraphrases, OCR errors and different topics can reduce overlap."
                ),
                "method": "case-insensitive token Jaccard overlap; no semantic entailment model",
            })
    return findings
