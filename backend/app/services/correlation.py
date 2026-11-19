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
    sources: list[tuple[str, str, str, str]] = []
    if context_text.strip():
        sources.append(("user_context", "Investigator-supplied claim / context", context_text.strip(), "claim"))
    for asset in assets:
        text = _asset_text(asset)
        if text:
            media_type = str(asset.get("media_type") or "evidence")
            source_kind = {"audio": "transcript", "video": "caption_or_video_ocr", "image": "image_ocr"}.get(media_type, "evidence_text")
            sources.append((
                str(asset.get("asset_id") or asset.get("filename") or "evidence"),
                str(asset.get("filename") or "Evidence file"),
                text,
                source_kind,
            ))

    if len(sources) < 2:
        return [{
            "type": "text_evidence_correlation",
            "status": "insufficient_text_sources",
            "detail": "At least two text-bearing sources are required. OCR and transcription availability may depend on optional local tools.",
        }]

    findings: list[dict[str, Any]] = []
    for index, (source_id, source_name, source_text, source_kind) in enumerate(sources):
        left = _tokens(source_text)
        for other_id, other_name, other_text, other_kind in sources[index + 1:]:
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
                "source_a_kind": source_kind,
                "source_b_kind": other_kind,
                "comparison_scope": "claim_vs_media_text" if "claim" in (source_kind, other_kind) else "transcript_caption_or_ocr_pair",
                "jaccard_overlap": round(overlap, 4),
                "shared_terms": sorted(left & right)[:30],
                "detail": (
                    "Text sources share vocabulary. This does not establish that their claims agree or are true."
                    if overlap >= 0.15 else
                    "Little vocabulary is shared. This does not establish contradiction or falsehood; paraphrases, OCR errors and different topics can reduce overlap."
                ),
                "method": "case-insensitive token Jaccard overlap; no semantic entailment model",
                "shared_term_count": len(left & right),
                "source_a_preview": source_text[:500],
                "source_b_preview": other_text[:500],
            })
    return findings



def correlate_timed_model_signals(video_asset: dict[str, Any], audio_asset: dict[str, Any]) -> dict[str, Any]:
    """Compare relative-time score profiles; this is not synchronization or a fused detector."""
    video = video_asset.get("video_analysis") or {}
    audio = audio_asset.get("model_assessment") or {}
    frames = [
        row for row in video.get("frames", [])
        if row.get("status") == "model_assessed"
        and isinstance(row.get("timestamp_seconds"), (int, float))
        and isinstance((row.get("scores") or {}).get("fake_median_across_detected_faces"), (int, float))
    ]
    segments = [
        row for row in audio.get("segments", [])
        if isinstance(row.get("start_seconds"), (int, float))
        and isinstance(row.get("end_seconds"), (int, float))
        and isinstance(row.get("fake_model_score"), (int, float))
    ]
    if len(frames) < 3 or len(segments) < 3:
        return {
            "type": "relative_timeline_model_signal_association",
            "status": "insufficient_temporal_samples",
            "video_samples": len(frames),
            "audio_samples": len(segments),
            "note": "At least three timestamped model samples from each modality are required.",
        }

    video_times = np.asarray([float(item["timestamp_seconds"]) for item in frames], dtype=np.float64)
    video_scores = np.asarray([float(item["scores"]["fake_median_across_detected_faces"]) for item in frames], dtype=np.float64)
    audio_times = np.asarray([(float(item["start_seconds"]) + float(item["end_seconds"])) / 2 for item in segments], dtype=np.float64)
    audio_scores = np.asarray([float(item["fake_model_score"]) for item in segments], dtype=np.float64)
    video_duration = max(float(video_times.max()), 1e-6)
    audio_duration = max(float(audio_times.max()), max(float(item["end_seconds"]) for item in segments), 1e-6)
    # Relative progress aligns the starts/ends for exploratory profile comparison; it does not estimate sync offset.
    phases = np.linspace(0.0, 1.0, num=min(24, max(len(video_times), len(audio_times))))
    video_profile = np.interp(phases, video_times / video_duration, video_scores)
    audio_profile = np.interp(phases, audio_times / audio_duration, audio_scores)
    if float(np.std(video_profile)) < 1e-8 or float(np.std(audio_profile)) < 1e-8:
        association = None
    else:
        association = float(np.corrcoef(video_profile, audio_profile)[0, 1])
    return {
        "type": "relative_timeline_model_signal_association",
        "status": "completed",
        "method": "Pearson association of linearly interpolated fake-direction model scores over normalized clip progress",
        "video_samples": int(len(frames)),
        "audio_samples": int(len(segments)),
        "comparison_points": int(len(phases)),
        "relative_score_pearson_r": round(association, 4) if association is not None else None,
        "median_absolute_score_difference": round(float(np.median(np.abs(video_profile - audio_profile))), 5),
        "interpretation": "Descriptive association between two different model-score timelines. Similarity or divergence does not establish authenticity, synchronization, source identity or manipulation.",
        "limitation": "The comparison normalizes each file to relative progress and assumes the supplied files represent the same recording. It does not estimate a real-world time offset or phoneme/viseme correspondence.",
    }
