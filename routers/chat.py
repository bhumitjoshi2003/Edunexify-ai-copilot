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
from tools.teacher_attendance import get_class_attendance_summary, get_low_attendance_students
from tools.teacher_results import get_class_performance_summary
from tools.admin_dashboard import get_school_overview, get_class_attendance_comparison
from tools.admin_attendance import get_school_low_attendance_students
from tools.admin_fees import get_fee_defaulters
from tools.admin_results import get_school_performance_summary

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

STUDENT_TOOLS: list[dict] = [
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

# ─── Teacher tools ─────────────────────────────────────────────────────────────
# className is NEVER a tool argument — every tool below scopes to user.className,
# which Spring Boot resolved from the teacher's own classTeacher field (see
# AiProxyController.resolveClassName). The underlying endpoints re-check this
# server-side too, so there's no argument the model could pass to widen access.

TEACHER_TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "get_class_attendance_summary",
            "description": (
                "Fetch an attendance overview for the teacher's own class: class average attendance, "
                "how many students are below 75%, and the highest/lowest attending students. "
                "Call this when the teacher asks how their class's attendance is doing overall. "
                "Use type='year' for a full academic session. Use type='month' for one calendar month."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "type": {
                        "type": "string",
                        "enum": ["year", "month"],
                        "description": "Summary period. 'year' = full academic session. 'month' = single calendar month.",
                    },
                    "session": {
                        "type": "string",
                        "description": "Academic session in YYYY-YYYY format. Only used when type='year'. Defaults to the current session.",
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
    },
    {
        "type": "function",
        "function": {
            "name": "get_low_attendance_students",
            "description": (
                "List the specific students in the teacher's class whose attendance is below a threshold "
                "(75% by default). Call this when the teacher asks which students have low attendance, "
                "which students are missing too many days, or which students need attention on attendance."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "threshold": {
                        "type": "number",
                        "description": "Attendance percentage cutoff. Students below this are returned. Defaults to 75.",
                    },
                    "session": {
                        "type": "string",
                        "description": "Academic session in YYYY-YYYY format. Defaults to the current session.",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_class_performance_summary",
            "description": (
                "Fetch exam performance for the teacher's own class: class average, top performers, "
                "students needing attention, and per-subject class averages (which subjects students are "
                "struggling with). Call this when the teacher asks how their class performed in an exam, "
                "which students need academic attention, or which subjects students are weak in. "
                "Defaults to the most recently created exam in the current session if no exam name is given."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "session": {
                        "type": "string",
                        "description": "Academic session in YYYY-YYYY format. Defaults to the current session.",
                    },
                    "examName": {
                        "type": "string",
                        "description": "Specific exam name to fetch (e.g. 'Mid Term', 'Final Exam'). If omitted, the latest exam is used.",
                    },
                },
                "required": [],
            },
        },
    },
]

# ─── Admin tools ───────────────────────────────────────────────────────────────
# None of these take a className/studentId argument that widens scope beyond
# the caller's own school — every underlying endpoint derives schoolId from
# the JWT server-side (SecurityUtil), same as every other tool in this file.
# className here (get_fee_defaulters) narrows an already-school-scoped query,
# it never crosses into another school.

ADMIN_TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "get_school_overview",
            "description": (
                "Fetch a top-level school snapshot: total students, total teachers, fees collected "
                "this month, count of overdue students, today's attendance rate, and pending leave requests. "
                "Call this when the admin asks for an overall school summary or a general status check."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_class_attendance_comparison",
            "description": (
                "Fetch this month's attendance rate for every class in the school, for comparison. "
                "Call this when the admin asks which classes have the best/worst attendance, or wants "
                "a class-by-class attendance breakdown."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_school_low_attendance_students",
            "description": (
                "List individual students across the WHOLE school whose attendance is below a threshold "
                "(75% by default), with their class. Call this when the admin asks which specific students "
                "have low attendance or need attention on attendance, school-wide (not just one class)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "threshold": {
                        "type": "number",
                        "description": "Attendance percentage cutoff. Students below this are returned. Defaults to 75.",
                    },
                    "session": {
                        "type": "string",
                        "description": "Academic session in YYYY-YYYY format. Defaults to the current session.",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_fee_defaulters",
            "description": (
                "Fetch students with pending/overdue fees: total count, total amount due, a class-wise "
                "breakdown, and the most overdue students. Call this when the admin asks how many students "
                "have pending fees, which students are fee defaulters, or wants fee collection insights."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "session": {
                        "type": "string",
                        "description": "Academic session in YYYY-YYYY format. Defaults to the current session.",
                    },
                    "className": {
                        "type": "string",
                        "description": "Restrict to one class, e.g. '10'. Omit for the whole school.",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_school_performance_summary",
            "description": (
                "Fetch each class's performance in its own latest exam: school average, best/worst "
                "performing class, a full class-by-class breakdown, and the weakest subjects school-wide. "
                "Call this when the admin asks which class performed best/worst, wants an academic "
                "performance summary, or asks which subjects students are struggling with school-wide."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "session": {
                        "type": "string",
                        "description": "Academic session in YYYY-YYYY format. Defaults to the current session.",
                    },
                },
                "required": [],
            },
        },
    },
]


_STUDENT_PROMPT_BODY = """## Available tools
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

_TEACHER_PROMPT_BODY = """## Available tools
- get_class_attendance_summary — use when the teacher asks how their class's attendance is doing overall (averages, how many students are below 75%).
- get_low_attendance_students — use when the teacher asks WHICH students have low/poor attendance, are missing too many days, or need attention on attendance.
- get_class_performance_summary — use when the teacher asks how their class performed in an exam, which students need academic attention, or which subjects students are struggling with.

All three tools are automatically scoped to the teacher's own assigned class — there is no className argument, so you never need to ask the teacher which class they mean.

You may call multiple tools in a single turn if the user asks about more than one topic (e.g. "how's my class doing overall" could warrant both attendance and performance).

## Tool usage rules
Only call a tool when the teacher explicitly asks about their class's attendance or academic performance.
Do NOT call any tool for:
- Greetings or small talk ("hi", "hello", "how are you")
- Questions about your identity or capabilities ("who are you", "what can you help me with")
- General questions that do not require class data
If the user asks something ambiguous like "which students need attention", consider calling both get_low_attendance_students and get_class_performance_summary unless context clearly points to just one (e.g. they already said "attendance" or "marks").

## When showing class attendance data
- Lead with the class average attendance percentage and how many students are below 75%.
- Name the lowest-attendance students specifically when relevant, not just the count.
- If the tool returns an "error" field (e.g. not a class teacher, access denied), relay that message plainly — do not guess at attendance numbers.

## When showing low-attendance students
- List each student by name with their attendance percentage, lowest first.
- If studentsBelowThreshold is 0, say clearly that no students are below the threshold — that's good news, don't imply a problem exists.

## When showing class performance data
- Lead with the exam name and class average percentage.
- Name the students in studentsNeedingAttention specifically, not just "some students".
- List subjectPerformance from weakest to strongest — call out the weakest subject explicitly since that's usually the most actionable insight.
- If noExamsYet is true, say clearly that no exams are configured yet — do NOT say "unable to fetch".
- If the tool returns an "availableExams" list (exam name not found), tell the teacher which exam names actually exist instead of failing silently.

## General behaviour
- Never guess or fabricate numbers, student names, or subjects — only report what tools return.
- If a tool returns an error or a "message" field (no data yet), relay that clearly.
- When the user asks about multiple topics, call all relevant tools and present results in sections.
- Be concise and direct. Do NOT end responses with "Is there anything else you'd like to know?" or similar filler.
- Use emojis sparingly — only where they add meaning, not as decoration. Never use a cheerful emoji (😊 😄) when reporting a problem like low attendance or weak performance."""

_ADMIN_PROMPT_BODY = """## Available tools
- get_school_overview — use for an overall school summary or general status check (students, teachers, fees this month, overdue count, today's attendance, pending leaves).
- get_class_attendance_comparison — use when the admin asks which classes have the best/worst attendance, or wants a class-by-class breakdown (this month).
- get_school_low_attendance_students — use when the admin asks WHICH specific students (not just which classes) have low attendance, school-wide.
- get_fee_defaulters — use when the admin asks about pending fees, how many students owe money, or fee defaulters.
- get_school_performance_summary — use when the admin asks which class performed best/worst in its latest exam, wants an academic performance summary, or asks which subjects students are struggling with school-wide.

For broad questions (e.g. "give me an overall school summary"), call multiple relevant tools in the same turn and present the results in sections — don't limit yourself to one.
For ambiguous requests like "which students/classes need attention", consider calling get_school_low_attendance_students, get_fee_defaulters, and get_school_performance_summary together unless the phrasing clearly points to just one domain.

## Tool usage rules
Only call a tool when the admin explicitly asks about school data.
Do NOT call any tool for:
- Greetings or small talk ("hi", "hello", "how are you")
- Questions about your identity or capabilities ("who are you", "what can you help me with")
- General questions that do not require school data

## When showing school overview data
- Lead with student/teacher counts and today's attendance rate.
- Call out overdue students and pending leaves as action items, not just numbers.

## When showing class attendance comparison
- Name the lowest and highest attendance classes specifically, with their percentages.
- Mention this reflects the current calendar month only, not the full session, if the admin's question implied a longer period.

## When showing low-attendance students
- List students by name with their class and attendance percentage, lowest first.
- If truncated is true, mention there are more below the threshold than shown (studentsBelowThreshold gives the real total).
- If studentsBelowThreshold is 0, say clearly that's good news — don't imply a problem.

## When showing fee defaulters
- Lead with total defaulter count and total amount due.
- Break down by class if byClass has more than one entry.
- Name the most overdue students specifically from mostOverdue.

## When showing school performance data
- Name the best and worst performing class specifically, with their exam name and percentage — note if they're different exams.
- List weakestSubjectsSchoolWide explicitly; that's usually the most actionable insight.
- If noResultsYet is true, say clearly no exam results are available yet — do NOT say "unable to fetch".
- classesWithNoExamYet > 0 means some classes were skipped because they have no exam configured yet — mention this if relevant rather than implying full school coverage.

## General behaviour
- Never guess or fabricate numbers, student names, class names, or subjects — only report what tools return.
- If a tool returns an "error" field, relay that message plainly rather than guessing at data.
- Be concise and direct. Do NOT end responses with "Is there anything else you'd like to know?" or similar filler.
- Use emojis sparingly — only where they add meaning, not as decoration. Never use a cheerful emoji (😊 😄) when reporting a problem like overdue fees, low attendance, or weak performance."""

_DEFAULT_PROMPT_BODY = """No specific data tools are available for your role yet. Answer general questions about Edunexify only — do not claim to fetch live data."""


def _build_system_prompt(user: UserContext) -> str:
    today = date.today().isoformat()
    session = current_academic_session()

    body = {
        "STUDENT": _STUDENT_PROMPT_BODY,
        "TEACHER": _TEACHER_PROMPT_BODY,
        "ADMIN": _ADMIN_PROMPT_BODY,
    }.get(user.role, _DEFAULT_PROMPT_BODY)

    return f"""You are Edunexify AI Copilot — a helpful assistant embedded in the Edunexify school management platform.

Logged-in user:
- Name: {user.name or user.userId}
- Role: {user.role}
- Class: {user.className or "N/A"}

Today's date: {today}
Current academic session: {session}

{body}"""


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

    if tool_name == "get_class_attendance_summary":
        return await get_class_attendance_summary(
            user=user,
            access_token=access_token,
            type=tool_input["type"],
            session=tool_input.get("session"),
            month=tool_input.get("month"),
            year=tool_input.get("year"),
        )

    if tool_name == "get_low_attendance_students":
        return await get_low_attendance_students(
            user=user,
            access_token=access_token,
            threshold=tool_input.get("threshold", 75.0),
            session=tool_input.get("session"),
        )

    if tool_name == "get_class_performance_summary":
        return await get_class_performance_summary(
            user=user,
            access_token=access_token,
            session=tool_input.get("session"),
            examName=tool_input.get("examName"),
        )

    if tool_name == "get_school_overview":
        return await get_school_overview(user=user, access_token=access_token)

    if tool_name == "get_class_attendance_comparison":
        return await get_class_attendance_comparison(user=user, access_token=access_token)

    if tool_name == "get_school_low_attendance_students":
        return await get_school_low_attendance_students(
            user=user,
            access_token=access_token,
            threshold=tool_input.get("threshold", 75.0),
            session=tool_input.get("session"),
        )

    if tool_name == "get_fee_defaulters":
        return await get_fee_defaulters(
            user=user,
            access_token=access_token,
            session=tool_input.get("session"),
            className=tool_input.get("className"),
        )

    if tool_name == "get_school_performance_summary":
        return await get_school_performance_summary(
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

    # Only offer the tools relevant to this role — DeepSeek can't call a tool
    # it was never given, so this is the first line of scoping (Spring Boot's
    # own per-endpoint checks are what actually enforce it, see tools/*.py).
    role_tools = {
        "STUDENT": STUDENT_TOOLS,
        "TEACHER": TEACHER_TOOLS,
        "ADMIN": ADMIN_TOOLS,
    }.get(request.user.role, [])

    # ─── Tool-calling loop ────────────────────────────────────────────────────
    reply_text: str | None = None
    try:
        for _ in range(5):  # Safety cap — prevents infinite loops
            completion_kwargs: dict = {"model": "deepseek-chat", "messages": messages}
            if role_tools:
                completion_kwargs["tools"] = role_tools
                completion_kwargs["tool_choice"] = "auto"

            response = _client.chat.completions.create(**completion_kwargs)

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
