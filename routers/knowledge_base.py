"""
routers/knowledge_base.py — document processing endpoint for the Knowledge
Base (RAG) feature.

Spring Boot owns all persistence (KnowledgeDocument/KnowledgeChunk tables,
pgvector) — this service is stateless here too, same as chat.py. Spring
uploads the raw file to us once (auth via the same X-Internal-Secret shared
secret used by /chat/stream), we extract text, chunk it, and embed each
chunk, then hand the results straight back for Spring to persist. We never
touch Postgres.
"""
from fastapi import APIRouter, File, Header, HTTPException, UploadFile

from config import settings
from services.chunker import chunk_text
from services.document_parser import UnsupportedDocumentType, extract_text
from services.embeddings import embed_texts

router = APIRouter()


@router.post("/kb/process-document")
async def process_document(
    file: UploadFile = File(...),
    x_internal_secret: str = Header(alias="X-Internal-Secret"),
) -> dict:
    if x_internal_secret != settings.internal_secret:
        raise HTTPException(status_code=401, detail="Unauthorized")

    content = await file.read()

    try:
        text = extract_text(file.filename or "", content)
    except UnsupportedDocumentType as e:
        return {"status": "failed", "error": str(e)}
    except Exception as e:
        return {"status": "failed", "error": f"Could not parse document: {e}"}

    if not text.strip():
        return {"status": "failed", "error": "No extractable text found in this document."}

    chunks = chunk_text(text)
    if not chunks:
        return {"status": "failed", "error": "No extractable text found in this document."}

    try:
        embeddings = await embed_texts(chunks)
    except Exception as e:
        return {"status": "failed", "error": f"Embedding failed: {e}"}

    return {
        "status": "ready",
        "chunks": [
            {"chunkIndex": i, "text": chunk, "embedding": embedding}
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings))
        ],
    }
