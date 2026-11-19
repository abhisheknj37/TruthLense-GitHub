from pathlib import Path

import pytest

from app.services.media import MediaError, inspect_media, media_type_for


def test_plain_text_evidence_is_imported_as_text_without_execution(tmp_path):
    source = tmp_path / "caption.txt"
    source.write_text("The council announced a new bridge on Monday.", encoding="utf-8")
    result = inspect_media(source)
    assert media_type_for(source) == "text"
    assert result["media_type"] == "text"
    assert result["text_extraction"]["status"] == "completed"
    assert result["text_extraction"]["text"] == "The council announced a new bridge on Monday."
    assert result["metadata"]["word_count"] == 8


def test_plain_text_accepts_utf8_bom(tmp_path):
    source = tmp_path / "notes.txt"
    source.write_bytes(b"\xef\xbb\xbfVerified source note")
    result = inspect_media(source)
    assert result["text_extraction"]["text"] == "Verified source note"


def test_plain_text_rejects_invalid_encoding_and_empty_files(tmp_path):
    invalid = tmp_path / "bad.txt"
    invalid.write_bytes(b"\xff\xfe\x00")
    with pytest.raises(MediaError, match="UTF-8"):
        inspect_media(invalid)
    empty = tmp_path / "empty.md"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(MediaError, match="empty"):
        inspect_media(empty)


def test_plain_text_size_is_bounded(tmp_path):
    source = tmp_path / "large.txt"
    source.write_bytes(b"x" * (1024 * 1024 + 1))
    with pytest.raises(MediaError, match="1 MB"):
        inspect_media(source)
