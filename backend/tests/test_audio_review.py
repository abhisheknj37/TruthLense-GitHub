from pathlib import Path

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
