from __future__ import annotations

from typing import Any


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
