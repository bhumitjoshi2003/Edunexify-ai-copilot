"""
memory.py — short-term conversation memory backed by Redis.

Scope: every conversation is keyed by schoolId + userId + conversationId, so
memory never crosses tenants or users, and a user's different chat sessions
(different conversationId) never bleed into each other.

Only the user-visible turns (the user's message + the assistant's final reply)
are stored — the intermediate tool-call / tool-result messages produced during
a single turn's tool-calling loop are never persisted. They're only meaningful
within that one request, and skipping them keeps stored history compact and
keeps every stored message a plain {role, content} pair that's always valid to
replay back to the model (no dangling tool_call_id references across turns).

History is trimmed to the last `conversation_max_messages` entries and given a
sliding TTL (refreshed on every write), so an idle conversation expires on its
own — no cleanup job needed.

If Redis is unreachable, memory is skipped entirely (logged, not raised) —
the assistant still answers, it just won't have prior turns for that request.
"""
import json
import logging

import redis.asyncio as redis

from config import settings

logger = logging.getLogger(__name__)

_redis: redis.Redis | None = None


def _client() -> redis.Redis:
    global _redis
    if _redis is None:
        _redis = redis.from_url(settings.redis_url, decode_responses=True)
    return _redis


async def close() -> None:
    """Called on app shutdown (see main.py) to close the connection pool cleanly."""
    global _redis
    if _redis is not None:
        await _redis.aclose()
        _redis = None


def _key(school_id: int, user_id: str, conversation_id: str) -> str:
    return f"aiconv:{school_id}:{user_id}:{conversation_id}"


async def get_history(school_id: int, user_id: str, conversation_id: str) -> list[dict]:
    """Returns the stored [{role, content}, ...] turns, oldest first. [] on miss or Redis error."""
    try:
        raw = await _client().get(_key(school_id, user_id, conversation_id))
    except Exception as e:
        logger.warning("Redis unavailable, continuing without conversation history: %s", e)
        return []

    if not raw:
        return []
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return []


async def append_turn(
    school_id: int,
    user_id: str,
    conversation_id: str,
    history: list[dict],
    user_message: str,
    assistant_reply: str,
) -> None:
    """Appends this turn to `history` (as already fetched by get_history), trims, and re-stores with a refreshed TTL."""
    updated = history + [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": assistant_reply},
    ]
    updated = updated[-settings.conversation_max_messages:]

    try:
        await _client().set(
            _key(school_id, user_id, conversation_id),
            json.dumps(updated),
            ex=settings.conversation_ttl_seconds,
        )
    except Exception as e:
        logger.warning("Failed to persist conversation turn to Redis: %s", e)
