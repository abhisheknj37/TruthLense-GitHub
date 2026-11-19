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
    assert c2pa["status"] == "evidence_file_missing"
    assert "No provenance conclusion" in c2pa["note"]


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
    monkeypatch.setenv("TRUTHLENS_FACE_SIMILARITY_REVIEW_THRESHOLD", "2.0")
    status = get_optional_integration_status()

    expected = {
        "temporal_video_model", "face_consistency", "syncnet", "c2pa",
        "ocr", "speech_transcription", "claim_research", "score_evaluation",
        "evidence_ledger",
    }
    assert expected.issubset(status)
    assert status["temporal_video_model"]["status"] == "configured_file_missing"
    assert status["syncnet"]["status"] == "needs_setup"
    assert str(tmp_path) not in str(status)
    assert "database" not in status["evidence_ledger"]
    assert "database_name" in status["evidence_ledger"]
    assert status["speech_transcription"]["status"] == "not_configured"
    assert status["face_consistency"]["threshold_config_valid"] is False
    assert status["face_consistency"]["review_threshold"] == 0.35

def test_c2pa_missing_manifest_is_not_reported_as_verification_error(monkeypatch, tmp_path):
    import sys
    import types
    import app.services.advanced_forensics as forensic

    class ManifestNotFound(Exception):
        pass

    class Reader:
        def __init__(self, mime, stream):
            pass

        def __enter__(self):
            raise ManifestNotFound("no JUMBF data found")

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(forensic.importlib.util, "find_spec", lambda name: object() if name == "c2pa" else None)
    monkeypatch.setitem(sys.modules, "c2pa", types.SimpleNamespace(Reader=Reader))
    image = tmp_path / "plain.jpg"
    image.write_bytes(b"not-a-real-jpeg-but-sufficient-for-reader-stub")

    result = forensic.verify_c2pa(image)

    assert result["status"] == "manifest_not_present"
    assert "does not indicate manipulation" in result["note"]
    assert "verification_error" != result["status"]


def test_c2pa_unexpected_reader_error_remains_verification_error(monkeypatch, tmp_path):
    import sys
    import types
    import app.services.advanced_forensics as forensic

    class Reader:
        def __init__(self, mime, stream):
            pass

        def __enter__(self):
            raise RuntimeError("native validator crashed")

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(forensic.importlib.util, "find_spec", lambda name: object() if name == "c2pa" else None)
    monkeypatch.setitem(sys.modules, "c2pa", types.SimpleNamespace(Reader=Reader))
    image = tmp_path / "plain.jpg"
    image.write_bytes(b"test")

    result = forensic.verify_c2pa(image)

    assert result["status"] == "verification_error"
    assert "native validator crashed" in result["detail"]


def test_temporal_torchscript_output_is_not_assumed_to_be_probability(tmp_path, monkeypatch):
    import torch
    import numpy as np
    from app.services import advanced_forensics, media

    class ScalarModel(torch.nn.Module):
        def forward(self, value):
            return value.mean().reshape(1)

    model_path = tmp_path / "temporal.pt"
    traced = torch.jit.trace(ScalarModel(), torch.ones((1, 8, 3, 224, 224)))
    traced.save(str(model_path))
    frame = np.full((16, 16, 3), 255, dtype=np.uint8)
    monkeypatch.setenv("TRUTHLENS_TEMPORAL_MODEL", str(model_path))
    monkeypatch.setattr(
        media, "sample_video_frames",
        lambda path, limit: {"frame_count": 8, "sampled_frames": [
            {"frame": frame, "frame_index": index, "timestamp_seconds": float(index)}
            for index in range(8)
        ]},
    )
    result = advanced_forensics._run_temporal_torchscript(tmp_path / "clip.mp4")
    assert result["status"] == "model_assessed"
    assert result["raw_model_output"] == 1.0
    assert result["score_semantics"] == "unverified_scalar_output"
    assert result["calibration_status"] == "not_calibrated"
    assert "not a probability" in result["note"]


def test_temporal_torchscript_rejects_multi_value_outputs(tmp_path, monkeypatch):
    import torch
    import numpy as np
    from app.services import advanced_forensics, media

    class MultiOutput(torch.nn.Module):
        def forward(self, value):
            return torch.stack((value.mean(), value.mean() + 1))

    model_path = tmp_path / "temporal-multiclass.pt"
    traced = torch.jit.trace(MultiOutput(), torch.ones((1, 8, 3, 224, 224)))
    traced.save(str(model_path))
    frame = np.full((16, 16, 3), 255, dtype=np.uint8)
    monkeypatch.setenv("TRUTHLENS_TEMPORAL_MODEL", str(model_path))
    monkeypatch.setattr(
        media, "sample_video_frames",
        lambda path, limit: {"frame_count": 8, "sampled_frames": [
            {"frame": frame, "frame_index": index, "timestamp_seconds": float(index)}
            for index in range(8)
        ]},
    )
    result = advanced_forensics._run_temporal_torchscript(tmp_path / "clip.mp4")
    assert result["status"] == "unsupported_model_output"
    assert result["output_shape"] == [2]
    assert "No class score was selected" in result["note"]
