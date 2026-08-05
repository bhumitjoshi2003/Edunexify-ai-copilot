"""
routers/chat.py — AI orchestration layer using Groq's inference API.

Groq's SDK is OpenAI-compatible — same interface, faster inference.
The tool-calling protocol is identical to OpenAI's:

  finish_reason == "tool_calls"   → Groq wants to call one or more tools
  finish_reason == "stop"         → Groq produced the final text answer

Tools are defined as {"type": "function", "function": {"name", "description", "parameters"}}
and tool results are returned as {"role": "tool", "tool_call_id": ..., "content": ...}.
"""
import json
from datetime import date

from fastapi import APIRouter, Header, HTTPException
from openai import OpenAI, APIStatusError

import memory
from config import settings
from schemas.chat import ChatRequest, ChatResponse, UserContext
from tools.attendance import get_attendance_summary, current_academic_session
from tools.fees import get_fee_summary
from tools.results import get_results_summary

router = APIRouter()

# DeepSeek via the OpenAI-compatible endpoint — same tool-calling format,
# no changes needed to TOOLS definitions or message handling.
_client = OpenAI(
    api_key=settings.deepseek_api_key,
    base_url="https://api.deepseek.com",
)

# ─── Tool definitions (OpenAI format) ─────────────────────────────────────────
# OpenAI wraps each tool in {"type": "function", "function": {...}}.
# The "parameters" field is standard JSON Schema — same content as Anthropic's
# "input_schema", just a different wrapper key name.

TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "get_fee_summary",
            "description": (
                "Fetch the student's fee payment status for an academic session. "
                "Call this ONLY when the user asks about fees, pending payments, paid months, "
                "outstanding amounts, or their fee summary. "
                "Returns which months are paid, which are unpaid, and total amount paid."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "session": {
                        "type": "string",
                        "description": (
                            "Academic session in YYYY-YYYY format, e.g. '2026-2027'. "
                            "If omitted, the current session is used."
                        ),
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_results_summary",
            "description": (
                "Fetch the student's exam results and marks for an academic session. "
                "Call this when the user asks about: their marks, exam results, performance, "
                "scores, grades, rank, which subject they did best/worst in, or their result summary. "
                "Returns per-exam breakdowns with subject-wise marks, ranks, class averages, "
                "and cross-exam highlights (best and weakest subject overall)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "session": {
                        "type": "string",
                        "description": (
                            "Academic session in YYYY-YYYY format, e.g. '2026-2027'. "
                            "If omitted, the current session is used."
                        ),
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_attendance_summary",
            "description": (
                "Fetch the student's attendance summary from the school database. "
                "Call this ONLY when the user is asking about their attendance, absences, presence, or attendance percentage. "
                "Use type='year' for a full academic year overview (includes month-by-month breakdown). "
                "Use type='month' for a specific calendar month."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "type": {
                        "type": "string",
                        "enum": ["year", "month"],
                        "description": "Summary period. 'year' = full academic year. 'month' = single calendar month.",
                    },
                    "session": {
                        "type": "string",
                        "description": (
                            "Academic session in YYYY-YYYY format, e.g. '2026-2027'. "
                            "Only used when type='year'. If omitted, the current session is used."
                        ),
                    },
                    "month": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 12,
                        "description": "Calendar month (1–12). Required when type='month'.",
                    },
                    "year": {
                        "type": "integer",
                        "description": "Calendar year, e.g. 2026. Required when type='month'.",
                    },
                },
                "required": ["type"],
            },
        },
    }
]


def _build_system_prompt(user: UserContext) -> str:
    today = date.today().isoformat()
    session = current_academic_session()

    return f"""You are Edunexify AI Copilot — a helpful assistant embedded in the Edunexify school management platform.

Logged-in user:
- Name: {user.name or user.userId}
- Role: {user.role}
- Class: {user.className or "N/A"}

Today's date: {today}
Current academic session: {session}

## Available tools
- get_fee_summary — use when the student asks about fees, pending payments, paid months, or outstanding amounts.
- get_attendance_summary — use when the student asks about attendance, absences, or attendance percentage.
- get_results_summary — use when the student asks about marks, exam results, scores, grades, rank, or performance.

You may call multiple tools in a single turn if the user asks about more than one topic (e.g. "show me my attendance and results").

## Tool usage rules
Only call a tool when the user explicitly asks for their school data.
Do NOT call any tool for:
- Greetings or small talk ("hi", "hello", "how are you")
- Questions about your identity or capabilities ("who are you", "what can you help me with")
- General questions that do not require the student's personal data

## When showing fee data
- State how many months are paid and how many are pending.
- If there are pending months, list them by name on one line (e.g. "April, May, June").
- Show total amount paid so far. If the tool also returns a total pending/due amount, show that too.
- If all months are paid, congratulate the student briefly.
- The tool response includes correct calendar month names already — use them directly. Do NOT re-interpret month numbers.

## When showing attendance data
- Show: days present, days absent, total working days, and attendance percentage.
- If attendance is below 75%, clearly state how many more days they need to attend to reach 75%.

## When showing exam results
- Lead with the latest exam: overall percentage, grade, and rank.
- List subject-wise marks (obtained / max, percentage) for that exam.
- Highlight their strongest and weakest subject.
- If multiple exams exist, briefly note the trend.
- If the tool returns bestSubjectOverall / weakestSubjectOverall, include them.
- Rank = null means insufficient data to rank; do not say "rank 0".
- If the tool returns noResultsYet=true, tell the student no exam results have been published yet — do NOT say "unable to fetch".
- If the tool returns an error field starting with "Spring Boot returned", say results could not be loaded and share the error code. Do not fabricate data.

## General behaviour
- Never guess or fabricate numbers — only report what tools return.
- If a tool returns an error or a "message" field (no data yet), relay that clearly.
- When the user asks about multiple topics, call all relevant tools and present results in sections.
- Be concise and direct. Do NOT end responses with "Is there anything else you'd like to know?" or similar filler.
- Use emojis sparingly — only where they add meaning, not as decoration. Never use a cheerful emoji (😊 😄) when reporting a problem like unpaid fees or low attendance."""


async def _execute_tool(
    tool_name: str,
    tool_input: dict,
    user: UserContext,
    access_token: str,
) -> dict:
    """Dispatch a tool call by name to the corresponding Python function."""
    if tool_name == "get_fee_summary":
        return await get_fee_summary(
            user=user,
            access_token=access_token,
            session=tool_input.get("session"),
        )

    if tool_name == "get_attendance_summary":
        return await get_attendance_summary(
            user=user,
            access_token=access_token,
            type=tool_input["type"],
            session=tool_input.get("session"),
            month=tool_input.get("month"),
            year=tool_input.get("year"),
        )

    if tool_name == "get_results_summary":
        return await get_results_summary(
            user=user,
            access_token=access_token,
            session=tool_input.get("session"),
        )

    return {"error": f"Unknown tool '{tool_name}'."}


# ─── Main endpoint ────────────────────────────────────────────────────────────

@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    x_internal_secret: str = Header(alias="X-Internal-Secret"),
) -> ChatResponse:
    if x_internal_secret != settings.internal_secret:
        raise HTTPException(status_code=401, detail="Unauthorized")

    # Short-term memory: prior turns for this schoolId+userId+conversationId,
    # oldest first, already trimmed to the last N messages by memory.py.
    history = await memory.get_history(
        request.user.schoolId, request.user.userId, request.conversationId
    )

    # OpenAI format: system message is the first entry in the messages list,
    # not a separate top-level parameter like in Anthropic's API.
    messages: list[dict] = [
        {"role": "system", "content": _build_system_prompt(request.user)},
        *history,
        {"role": "user", "content": request.message},
    ]

    # ─── Tool-calling loop ────────────────────────────────────────────────────
    reply_text: str | None = None
    try:
        for _ in range(5):  # Safety cap — prevents infinite loops
            response = _client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
            )

            choice = response.choices[0]

            if choice.finish_reason == "stop":
                reply_text = choice.message.content or "I couldn't generate a response."
                break

            elif choice.finish_reason == "tool_calls":
                # Step 1: Append the assistant message (with tool_calls) to history.
                messages.append(choice.message)

                # Step 2: Execute each tool call and append its result.
                for tool_call in choice.message.tool_calls:
                    tool_input = json.loads(tool_call.function.arguments) or {}

                    tool_output = await _execute_tool(
                        tool_call.function.name,
                        tool_input,
                        request.user,
                        request.accessToken,
                    )

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(tool_output),
                    })

            else:
                # Unexpected finish_reason (e.g. "length") — bail out gracefully.
                break

    except APIStatusError as e:
        if e.status_code == 402:
            raise HTTPException(status_code=503, detail="AI service is temporarily unavailable. Please try again later.")
        if e.status_code == 429:
            raise HTTPException(status_code=429, detail="AI service is rate-limited. Please wait a moment and try again.")
        raise HTTPException(status_code=502, detail=f"AI provider error ({e.status_code}).")

    if reply_text is None:
        reply_text = "I wasn't able to complete your request. Please try again."

    # Persist only the user-visible turn (not the intermediate tool-call
    # messages above) — see memory.py for why.
    await memory.append_turn(
        request.user.schoolId,
        request.user.userId,
        request.conversationId,
        history,
        request.message,
        reply_text,
    )

    return ChatResponse(reply=reply_text)
