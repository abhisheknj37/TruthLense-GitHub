from types import SimpleNamespace

import pytest

from app.services.ocr import _configure_tesseract


def fake_pytesseract(installed):
    wrapper = SimpleNamespace(tesseract_cmd=None)
    module = SimpleNamespace(pytesseract=wrapper)
    module.get_tesseract_version = lambda: "5.4.0"
    module.get_languages = lambda config="": installed
    return module


def test_configure_tesseract_accepts_installed_requested_languages(monkeypatch):
    monkeypatch.setenv("TRUTHLENS_OCR_LANGUAGES", "eng+hin")
    module = fake_pytesseract({"eng", "hin", "osd"})

    assert _configure_tesseract(module) == "eng+hin"


def test_configure_tesseract_reports_missing_language_data(monkeypatch):
    monkeypatch.setenv("TRUTHLENS_OCR_LANGUAGES", "eng+hin")
    module = fake_pytesseract({"eng", "osd"})

    with pytest.raises(RuntimeError, match="hin"):
        _configure_tesseract(module)


def test_configure_tesseract_uses_configured_windows_executable(monkeypatch, tmp_path):
    executable = tmp_path / "tesseract.exe"
    monkeypatch.setenv("TRUTHLENS_TESSERACT_CMD", str(executable))
    monkeypatch.setenv("TRUTHLENS_OCR_LANGUAGES", "eng")
    module = fake_pytesseract({"eng"})

    assert _configure_tesseract(module) == "eng"
    assert module.pytesseract.tesseract_cmd == str(executable)
