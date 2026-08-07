"""
services/document_parser.py — extracts plain text from an uploaded Knowledge
Base document. Dispatches purely on file extension; Spring Boot already
validated the extension/size before calling us.
"""
import io

from docx import Document
from pypdf import PdfReader


class UnsupportedDocumentType(Exception):
    pass


def extract_text(filename: str, content: bytes) -> str:
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    if ext == "pdf":
        return _extract_pdf(content)
    if ext == "docx":
        return _extract_docx(content)
    if ext in ("txt", "md"):
        return content.decode("utf-8", errors="ignore")

    raise UnsupportedDocumentType(f"Unsupported file type '.{ext}'. Use PDF, DOCX, TXT, or MD.")


def _extract_pdf(content: bytes) -> str:
    reader = PdfReader(io.BytesIO(content))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages)


def _extract_docx(content: bytes) -> str:
    doc = Document(io.BytesIO(content))
    return "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())
