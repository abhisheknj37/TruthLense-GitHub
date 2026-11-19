from pathlib import Path

from docx import Document
from app.services.documents import extract_document, DocumentError
from app.services.media import inspect_media, media_type_for


def test_text_and_markdown_documents_are_extracted(tmp_path: Path):
    path = tmp_path / "notes.txt"
    path.write_text("Breaking claim published on 2026-11-19", encoding="utf-8")
    result = extract_document(path)
    assert result["status"] == "text_extracted"
    assert "Breaking claim" in result["text"]


def test_csv_document_is_extracted_as_rows(tmp_path: Path):
    path = tmp_path / "evidence.csv"
    path.write_text("name,value\nsource,example", encoding="utf-8")
    result = extract_document(path)
    assert "name | value" in result["text"]
    assert "source | example" in result["text"]


def test_docx_document_is_extracted(tmp_path: Path):
    path = tmp_path / "evidence.docx"
    document = Document()
    document.add_paragraph("Document evidence paragraph")
    document.save(path)
    assert "Document evidence paragraph" in extract_document(path)["text"]


def test_empty_document_returns_explicit_state(tmp_path: Path):
    path = tmp_path / "empty.txt"
    path.write_text("", encoding="utf-8")
    assert extract_document(path)["status"] == "no_extractable_text"


def test_document_media_is_supported_by_inspector(tmp_path: Path):
    path = tmp_path / "notes.md"
    path.write_text("# Evidence", encoding="utf-8")
    assert media_type_for(path) == "document"
    assert inspect_media(path)["media_type"] == "document"


def test_unsupported_document_extension_is_rejected(tmp_path: Path):
    path = tmp_path / "evidence.rtf"
    path.write_text("not supported", encoding="utf-8")
    try:
        extract_document(path)
    except DocumentError:
        return
    raise AssertionError("Unsupported document type should be rejected")
