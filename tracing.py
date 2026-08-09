"""
tracing.py — lightweight per-turn tracing for the AI Copilot. Two outputs per
completed chat turn, both best-effort and never able to slow down or break the
user's response:

  1. A structured JSON line to stdout (captured by Docker's existing json-file
     log driver) — synchronous, sub-millisecond, immediately grep/jq-able for
     "what just happened."
  2. A fire-and-forget POST to Spring's /api/ai/trace, forwarding the user's own
     accessToken cookie (same trust pattern as every tool call — see
     tools/knowledge_base.py), fired via asyncio.create_task() AFTER the caller
     has already finished sending the response — genuinely zero added perceived
     latency. Failures here are logged and swallowed, never raised.

ai_trace_event (the Postgres table this feeds) is an internal ops-debugging
table with a short retention window — see AiTraceCleanupScheduler on the Spring
side — because userMessage/finalReply can contain student-specific detail.

Never included in either output: accessToken, X-Internal-Secret, or raw tool
result payloads — tool_calls entries are name/latency/success/coarse-summary
only, since several tools (e.g. get_school_low_attendance_students) return
*other* students' names/data in their real payload.
"""
import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field

import httpx

from config import settings
from schemas.chat import UserContext

logger = logging.getLogger("ai.trace")


def summarize_tool_result(result: dict) -> dict:
    """Coarse, non-identifying signal only — counts and flags, never names,
    numbers, or any other field value that could be someone's actual data."""
    if not isinstance(result, dict):
        return {}
    if result.get("error"):
        return {"error": True}
    summary: dict = {}
    for key, value in result.items():
        if isinstance(value, list):
            summary[f"{key}Count"] = len(value)
        elif isinstance(value, bool):
            summary[key] = value
    return summary


@dataclass
class Trace:
    user: UserContext
    conversation_id: str
    user_message: str
    access_token: str  # only ever used to forward to Spring's trace endpoint below, never logged itself

    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    tool_calls: list = field(default_factory=list)
    rag_results: list = field(default_factory=list)
    deepseek_calls: list = field(default_factory=list)
    final_reply: str = ""
    errored: bool = False
    error_message: str | None = None
    interrupted: bool = False

    _start: float = field(default_factory=time.monotonic)

    def record_tool_call(self, name: str, latency_ms: float, success: bool,
                          error: str | None = None, result_summary: dict | None = None) -> None:
        self.tool_calls.append({
            "name": name,
            "latencyMs": round(latency_ms, 1),
            "success": success,
            "error": error,
            "resultSummary": result_summary,
        })

    def record_rag_results(self, results: list[dict] | None) -> None:
        for r in results or []:
            self.rag_results.append({
                "documentId": r.get("documentId"),
                "documentTitle": r.get("documentTitle"),
                "pageNumber": r.get("pageNumber"),
                "similarity": r.get("similarity"),
                # Policy document text, not student data — a longer preview is fine here
                # and is exactly what's needed to debug a grounding/hallucination issue.
                "chunkPreview": (r.get("chunkText") or "")[:200],
            })

    def record_deepseek_call(self, latency_ms: float, usage, finish_reason: str | None) -> None:
        entry: dict = {"latencyMs": round(latency_ms, 1), "finishReason": finish_reason}
        if usage is not None:
            entry["promptTokens"] = getattr(usage, "prompt_tokens", None)
            entry["completionTokens"] = getattr(usage, "completion_tokens", None)
            entry["totalTokens"] = getattr(usage, "total_tokens", None)
        self.deepseek_calls.append(entry)

    def _token_totals(self) -> dict:
        prompt = sum(c.get("promptTokens") or 0 for c in self.deepseek_calls)
        completion = sum(c.get("completionTokens") or 0 for c in self.deepseek_calls)
        total = sum(c.get("totalTokens") or 0 for c in self.deepseek_calls)
        return {
            "promptTokens": prompt or None,
            "completionTokens": completion or None,
            "totalTokens": total or None,
        }

    def to_dict(self) -> dict:
        return {
            "traceId": self.trace_id,
            "schoolId": self.user.schoolId,
            "userId": self.user.userId,
            "role": self.user.role,
            "conversationId": self.conversation_id,
            "userMessage": self.user_message,
            "finalReply": self.final_reply,
            "toolsCalled": self.tool_calls,
            "ragResults": self.rag_results,
            **self._token_totals(),
            "totalLatencyMs": round((time.monotonic() - self._start) * 1000, 1),
            "errored": self.errored,
            "errorMessage": self.error_message,
            "interrupted": self.interrupted,
        }

    async def finish(self) -> None:
        """Call exactly once, after the response has already been fully sent/streamed
        to the client. Never raises — a tracing failure must never affect the chat turn."""
        data = self.to_dict()

        try:
            logger.info(json.dumps(data, default=str))
        except Exception:
            logger.warning("Failed to serialize trace %s for logging", self.trace_id, exc_info=True)

        asyncio.create_task(_ship_to_spring(self.trace_id, self.access_token, data))


async def _ship_to_spring(trace_id: str, access_token: str, data: dict) -> None:
    # schoolId/userId/role are re-derived by Spring from the verified JWT — see
    # AiTraceService — so they're dropped here rather than trusted from this payload.
    payload = {k: v for k, v in data.items() if k not in ("schoolId", "userId", "role")}
    try:
        async with httpx.AsyncClient() as client:
            await client.post(
                f"{settings.spring_boot_url}/api/ai/trace",
                json=payload,
                cookies={"accessToken": access_token},
                timeout=5.0,
            )
    except Exception as e:
        logger.warning("Failed to ship trace %s to Spring: %s", trace_id, e)
