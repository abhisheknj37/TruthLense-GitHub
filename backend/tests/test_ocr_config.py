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

def test_ocr_confidence_is_labeled_as_recognition_quality_not_authenticity():
    from app.services.ocr import _ocr_note

    assert _ocr_note(30)[0] == "low"
    assert _ocr_note(55)[0] == "moderate"
    assert _ocr_note(85)[0] == "high"
    assert _ocr_note(None)[0] == "unavailable"
    assert "authentic" in _ocr_note(85)[1]


def test_prepare_image_returns_preprocessing_metadata():
    from PIL import Image
    from app.services.ocr import _prepare_image

    image = Image.new("RGB", (320, 180), "white")
    prepared, metadata = _prepare_image(image)

    assert prepared.mode == "L"
    assert prepared.width >= 320
    assert prepared.height >= 180
    assert "deskew_correction_degrees" in metadata
    assert "CLAHE contrast" in metadata["preprocessing"]
