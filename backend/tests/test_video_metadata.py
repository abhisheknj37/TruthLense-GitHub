from pathlib import Path

import numpy as np

from app.services import media


class FakeCapture:
    def __init__(self, frames, reported_count=-152954252944508352, fps=16.583):
        self.frames = list(frames)
        self.position = 0
        self.reported_count = reported_count
        self.fps = fps
        self.released = False

    def isOpened(self):
        return True

    def get(self, prop):
        if prop == media.cv2.CAP_PROP_FRAME_COUNT:
            return self.reported_count
        if prop == media.cv2.CAP_PROP_FPS:
            return self.fps
        if prop == media.cv2.CAP_PROP_FRAME_WIDTH:
            return 16
        if prop == media.cv2.CAP_PROP_FRAME_HEIGHT:
            return 12
        return 0

    def read(self):
        if self.position >= len(self.frames):
            return False, None
        frame = self.frames[self.position]
        self.position += 1
        return True, frame.copy()

    def release(self):
        self.released = True


def test_inspect_video_uses_two_pass_even_temporal_coverage(monkeypatch, tmp_path):
    frames = [np.full((12, 16, 3), value, dtype=np.uint8) for value in range(10)]
    captures = [FakeCapture(frames), FakeCapture(frames)]
    monkeypatch.setattr(media.cv2, "VideoCapture", lambda *_: captures.pop(0))

    metadata, observations, samples = media.inspect_video(Path(tmp_path / "sample.webm"), sample_limit=4)

    assert metadata["frame_count"] == 10
    assert metadata["sampled_frame_count"] == 4
    assert metadata["duration_seconds"] == round(10 / 16.583, 3)
    assert [item["frame_index"] for item in samples] == [0, 3, 6, 9]
    assert metadata["sampling_method"] == "two-pass exact temporal stratification"
    assert "10 frames" in observations[0]["detail"]
    assert len(captures) == 0


def test_inspect_video_reports_zero_frames_without_negative_metadata(monkeypatch, tmp_path):
    capture = FakeCapture([])
    monkeypatch.setattr(media.cv2, "VideoCapture", lambda *_: capture)

    metadata, observations, samples = media.inspect_video(Path(tmp_path / "empty.webm"))

    assert metadata["frame_count"] == 0
    assert metadata["duration_seconds"] is None
    assert metadata["sampled_frame_count"] == 0
    assert samples == []
    assert observations[0]["severity"] == "warning"
    assert capture.released
