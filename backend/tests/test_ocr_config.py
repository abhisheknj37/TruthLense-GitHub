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

def test_text_region_detection_returns_no_regions_for_blank_image():
    from PIL import Image
    from app.services.ocr import _detect_text_regions

    image = Image.new("L", (900, 500), "white")
    assert _detect_text_regions(image) == []


def test_text_region_detection_finds_text_like_block():
    import cv2
    import numpy as np
    from PIL import Image
    from app.services.ocr import _detect_text_regions

    pixels = np.full((500, 900), 255, dtype=np.uint8)
    for y in (120, 165, 210):
        cv2.putText(pixels, "TRUTHLENS EVIDENCE REVIEW", (80, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, 0, 2, cv2.LINE_AA)
    regions = _detect_text_regions(Image.fromarray(pixels))
    assert regions
    assert all(x2 > x1 and y2 > y1 for x1, y1, x2, y2 in regions)


def test_character_error_rate_matches_ground_truth_reference():
    from app.services.ocr import _normalized_edit_distance

    result = _normalized_edit_distance("hello world", "hello word")
    assert result["status"] == "scored"
    assert result["character_errors"] == 1
    assert result["reference_characters"] == 11
    assert result["cer"] == round(1 / 11, 4)


def test_character_error_rate_rejects_empty_reference():
    from app.services.ocr import _normalized_edit_distance

    result = _normalized_edit_distance("   ", "some OCR text")
    assert result["status"] == "invalid_reference"
    assert result["cer"] is None


def test_rapidocr_missing_dependency_is_reported_without_crashing(monkeypatch):
    import builtins
    from PIL import Image
    from app.services.ocr import _rapidocr_extract

    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "rapidocr_onnxruntime":
            raise ImportError("optional dependency not installed")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    result = _rapidocr_extract(Image.new("RGB", (100, 50), "white"))
    assert result["status"] == "engine_not_installed"
    assert result["text"] == ""

def test_ocr_quality_gate_excludes_symbol_heavy_gibberish():
    from app.services.ocr import _ocr_analysis_gate

    gate = _ocr_analysis_gate("oe LS N a a com WY ee i eft an mies Y Ay NS", 82.0, 12)

    assert gate["analysis_eligible"] is False
    assert gate["analysis_text"] == ""
    assert gate["analysis_gate"]["observed_meaningful_token_count"] < 2


def test_ocr_quality_gate_keeps_clear_sentence():
    from app.services.ocr import _ocr_analysis_gate

    text = "The meeting starts at noon tomorrow."
    gate = _ocr_analysis_gate(text, 82.0, 7)

    assert gate["analysis_eligible"] is True
    assert gate["analysis_text"] == text
    assert gate["analysis_gate"]["observed_meaningful_token_count"] >= 2
