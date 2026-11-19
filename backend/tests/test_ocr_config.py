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


def test_prepare_image_variants_returns_contrast_and_thresholded_images():
    from PIL import Image
    from app.services.ocr import _prepare_image_variants

    image = Image.new("RGB", (320, 180), "white")
    variants, metadata = _prepare_image_variants(image)

    assert [name for name, _ in variants] == ["contrast", "otsu", "adaptive"]
    assert all(item.mode == "L" for _, item in variants)
    assert all(item.width >= 320 and item.height >= 180 for _, item in variants)
    assert "deskew_correction_degrees" in metadata
    assert "CLAHE contrast" in metadata["preprocessing"]
    assert "adaptive threshold" in metadata["preprocessing"]

def test_low_confidence_ocr_is_preserved_but_not_eligible_for_claim_analysis():
    from app.services.ocr import _ocr_analysis_gate

    gate = _ocr_analysis_gate("Be ae a ao aad 5", 33.1, 6)

    assert gate["analysis_eligible"] is False
    assert gate["analysis_text"] == ""
    assert gate["analysis_gate"]["minimum_mean_word_confidence"] == 45


def test_clearer_ocr_can_pass_quality_gate_without_being_authenticity_verdict():
    from app.services.ocr import _ocr_analysis_gate

    gate = _ocr_analysis_gate("The meeting starts at noon tomorrow.", 82.0, 7)

    assert gate["analysis_eligible"] is True
    assert gate["analysis_text"] == "The meeting starts at noon tomorrow."
