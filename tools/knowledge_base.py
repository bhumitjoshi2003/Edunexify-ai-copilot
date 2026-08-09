"""
tools/knowledge_base.py — RAG retrieval tool for the AI copilot.

Embeds the user's question in-process, then asks Spring Boot for the closest
matching chunks from that school's uploaded policy/handbook documents.
Same trust pattern as every other tool here (see tools/attendance.py's module
docstring): we forward the user's own accessToken cookie, so Spring's normal
@PreAuthorize + schoolId scoping applies unchanged — this tool has no way to
reach another school's documents.
"""
import re

import httpx

from config import settings
from schemas.chat import UserContext
from services.embeddings import embed_texts

_TOP_K = 3


def _clean_title(title: str) -> str:
    """Admin-entered titles sometimes default to a raw filename (e.g.
    "Leave_Policy_2026") — present them more readably in citations without
    touching the stored title shown as-is in the admin Knowledge Base list."""
    return re.sub(r"[_-]+", " ", title).strip()


async def search_knowledge_base(user: UserContext, access_token: str, query: str) -> dict:
    try:
        [query_embedding] = await embed_texts([query])
    except Exception as e:
        return {"error": f"Could not search the knowledge base right now: {e}"}

    url = f"{settings.spring_boot_url}/api/knowledge-base/search"

    async with httpx.AsyncClient() as client:
        response = await client.post(
            url,
            json={"embedding": query_embedding, "topK": _TOP_K},
            cookies={"accessToken": access_token},
            timeout=15.0,
        )

    if response.status_code != 200:
        return {"error": f"Spring Boot returned {response.status_code}: {response.text}"}

    # Spring already excludes anything below the relevance floor (see
    # KnowledgeSearchRepository.search()) — an empty list here genuinely means
    # nothing in the knowledge base is a good enough match, not just "the top
    # result happened to be weak".
    raw_results = response.json().get("results", [])
    if not raw_results:
        return {
            "found": False,
            "message": "No sufficiently relevant content found in the school's knowledge base for this question.",
        }

    results = [
        {
            "chunkText": r["chunkText"],
            "documentTitle": _clean_title(r["documentTitle"]),
            "documentId": r["documentId"],
            "pageNumber": r.get("pageNumber"),
            "similarity": round(r["similarity"], 3),
        }
        for r in raw_results
    ]

    return {"found": True, "results": results}
