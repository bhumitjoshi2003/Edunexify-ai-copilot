"""
services/document_parser.py — extracts plain text from an uploaded Knowledge
Base document. Dispatches purely on file extension; Spring Boot already
validated the extension/size before calling us.

Returns per-page text (1-indexed page number, page text) rather than one flat
string. Two reasons: (1) so routers/knowledge_base.py can chunk within page
boundaries instead of across them, keeping each chunk's page attribution exact
rather than approximate, and (2) so citations can point to a real page number.
DOCX/TXT/MD have no native pagination, so they come back as a single
(None, text) "page" — page number stays null for those chunks.
"""
import io

from docx import Document
from pypdf import PdfReader


class UnsupportedDocumentType(Exception):
    pass


def extract_pages(filename: str, content: bytes) -> list[tuple[int | None, str]]:
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    if ext == "pdf":
        return _extract_pdf_pages(content)
    if ext == "docx":
        return [(None, _extract_docx(content))]
    if ext in ("txt", "md"):
        return [(None, content.decode("utf-8", errors="ignore"))]

    raise UnsupportedDocumentType(f"Unsupported file type '.{ext}'. Use PDF, DOCX, TXT, or MD.")


def _extract_pdf_pages(content: bytes) -> list[tuple[int | None, str]]:
    reader = PdfReader(io.BytesIO(content))
    return [(i + 1, page.extract_text() or "") for i, page in enumerate(reader.pages)]


def _extract_docx(content: bytes) -> str:
    doc = Document(io.BytesIO(content))
    return "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())
