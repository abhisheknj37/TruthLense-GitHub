from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Any

from docx import Document
from pypdf import PdfReader

MAX_EXTRACTED_CHARS = 100_000


class DocumentError(ValueError):
    pass


def extract_document(path: Path) -> dict[str, Any]:
    """Extract reviewable text from supported user-supplied document evidence."""
    suffix = path.suffix.lower()
    try:
        if suffix in {".txt", ".md", ".csv"}:
            text = path.read_text(encoding="utf-8-sig", errors="replace")
            if suffix == ".csv":
                rows = list(csv.reader(io.StringIO(text)))
                text = "\n".join(" | ".join(row) for row in rows)
            pages = []
        elif suffix == ".pdf":
            reader = PdfReader(str(path))
            if reader.is_encrypted:
                raise DocumentError("Encrypted PDFs are not supported. Export an unlocked copy for review.")
            pages = []
            for index, page in enumerate(reader.pages[:200]):
                page_text = page.extract_text() or ""
                if page_text.strip():
                    pages.append({"page": index + 1, "text": page_text[:MAX_EXTRACTED_CHARS]})
            text = "\n\n".join(item["text"] for item in pages)
        elif suffix == ".docx":
            document = Document(str(path))
            paragraphs = [paragraph.text for paragraph in document.paragraphs if paragraph.text.strip()]
            tables = ["\n".join(" | ".join(cell.text for cell in row.cells) for row in table.rows)
                      for table in document.tables]
            text = "\n".join(paragraphs + tables)
            pages = []
        else:
            raise DocumentError(f"Unsupported document type: {suffix}")
    except DocumentError:
        raise
    except Exception as exc:
        raise DocumentError(f"Could not extract text from this document: {exc}") from exc

    original_length = len(text)
    text = text[:MAX_EXTRACTED_CHARS]
    if not text.strip():
        return {
            "status": "no_extractable_text",
            "text": "",
            "pages": pages,
            "character_count": 0,
            "note": "No selectable text was found. Scanned PDFs need OCR before their content can be reviewed.",
        }
    return {
        "status": "text_extracted",
        "text": text,
        "pages": pages,
        "character_count": len(text),
        "truncated": original_length > MAX_EXTRACTED_CHARS,
        "note": "Extracted text is supplied evidence, not independently verified. PDF extraction may omit scanned images, layout, or text in graphics.",
    }
