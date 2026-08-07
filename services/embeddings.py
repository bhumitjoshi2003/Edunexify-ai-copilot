"""
services/embeddings.py — wraps OpenAI's embeddings API for the Knowledge Base
(RAG) feature. A separate client from the DeepSeek one in routers/chat.py —
DeepSeek doesn't serve embeddings, so this always points at the real OpenAI API.
"""
from openai import AsyncOpenAI

from config import settings

_client: AsyncOpenAI | None = None


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        if not settings.openai_api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set — required for the Knowledge Base (RAG) feature."
            )
        _client = AsyncOpenAI(api_key=settings.openai_api_key)
    return _client


async def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embeds a batch of strings in a single API call. Order of the returned
    vectors matches the order of `texts`."""
    if not texts:
        return []
    response = await _get_client().embeddings.create(
        model=settings.embedding_model,
        input=texts,
    )
    return [item.embedding for item in response.data]
