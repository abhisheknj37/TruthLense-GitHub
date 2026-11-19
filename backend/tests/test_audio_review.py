from pathlib import Path
from types import SimpleNamespace

import numpy as np
import soundfile as sf

from app.services.audio_review import audio_review_timeline


def test_audio_review_returns_waveform_and_discloses_limits(tmp_path: Path):
    sample_rate = 16000
    seconds = 2
    time = np.arange(sample_rate * seconds) / sample_rate
    audio = (0.2 * np.sin(2 * np.pi * 440 * time)).astype(np.float32)
    path = tmp_path / "tone.wav"
    sf.write(path, audio, sample_rate)
    result = audio_review_timeline(path)
    assert result["status"] == "completed"
    assert result["duration_seconds"] == 2.0
    assert result["waveform"]
    assert "not identify synthetic speech" in result["interpretation"]



def test_audio_model_exposes_limited_synthetic_voice_indicator(monkeypatch):
    import app.detectors.audio as audio_module
    import torch

    class FakeModel:
        def __call__(self, **kwargs):
            return SimpleNamespace(logits=torch.tensor([[0.1, 2.0]]))

    class FakeProcessor:
        def __call__(self, clip, **kwargs):
            return {"input_values": torch.zeros((1, len(clip)))}

    detector = audio_module.AudioDeepfakeDetector()
    detector.model = FakeModel()
    detector.processor = FakeProcessor()
    detector.fake_index = 1
    detector.real_index = 0
    monkeypatch.setattr(
        audio_module.librosa,
        "load",
        lambda *args, **kwargs: (
            np.ones(audio_module.SAMPLE_RATE * 3, dtype=np.float32),
            audio_module.SAMPLE_RATE,
        ),
    )

    result = detector.analyze(Path("fixture.wav"))

    assert result["status"] == "model_assessed"
    review = result["voice_synthesis_review"]
    assert review["status"] == "model_indicator_available"
    assert review["indicator"] == "fake_signal"
    assert review["windows_assessed"] == 1
    assert review["model"] == audio_module.MODEL_ID
    assert "does not identify a speaker" in review["note"]
    assert "prove voice cloning" in review["note"]
