from pathlib import Path

import cv2
import numpy as np

from app.services.advanced_forensics import temporal_video_forensics, syncnet_lipsync, verify_c2pa
from app.services.claim_verification import extract_claims
from app.services.text_analysis import analyze_text_context


def test_temporal_video_forensics_reports_review_timestamps(tmp_path: Path):
    path = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (160, 120))
    for index in range(24):
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        x = index * 3 % 120
        cv2.rectangle(frame, (x, 30), (x + 25, 70), (255, 255, 255), -1)
        writer.write(frame)
    writer.release()
    result = temporal_video_forensics(path)
    assert result["status"] == "completed"
    assert "scene_change_timestamps_seconds" in result
    assert "trained_model_assessment" in result


def test_claim_extraction_is_bounded_and_conservative():
    claims = extract_claims("A study found that this intervention increases outcomes by 20 percent. This is a short note.")
    assert len(claims) == 1


def test_optional_integrations_report_unconfigured_state(monkeypatch, tmp_path):
    monkeypatch.delenv("TRUTHLENS_SYNCNET_WEIGHTS", raising=False)
    assert syncnet_lipsync(tmp_path / "missing.mp4")["status"] == "model_weights_missing"
    c2pa = verify_c2pa(tmp_path / "missing.mp4")
    assert c2pa["status"] in {"validator_not_installed", "unsupported_format"}


def test_text_triage_never_returns_truth_verdict():
    result = analyze_text_context("Breaking: share immediately. The report was published on 2026-11-19.")
    assert result["status"] == "review_cues_extracted"
    assert "claim_verification" not in result
    assert result["language_cues"]

def test_optional_integration_status_is_explicit_and_does_not_reveal_paths(monkeypatch, tmp_path):
    from app.services.optional_integrations import get_optional_integration_status

    monkeypatch.setenv("TRUTHLENS_TEMPORAL_MODEL", str(tmp_path / "missing.pt"))
    monkeypatch.setenv("TRUTHLENS_SYNCNET_WEIGHTS", str(tmp_path / "missing.model"))
    monkeypatch.setenv("TRUTHLENS_TESSERACT_CMD", str(tmp_path / "tesseract.exe"))
    monkeypatch.setenv("TRUTHLENS_ASR_MODEL", "")
    status = get_optional_integration_status()

    expected = {
        "temporal_video_model", "face_consistency", "syncnet", "c2pa",
        "ocr", "speech_transcription", "claim_research", "score_evaluation",
        "evidence_ledger",
    }
    assert expected.issubset(status)
    assert status["temporal_video_model"]["status"] == "configured_file_missing"
    assert status["syncnet"]["status"] == "needs_setup"
    assert "tesseract.exe" not in str(status)
    assert status["speech_transcription"]["status"] == "not_configured"

