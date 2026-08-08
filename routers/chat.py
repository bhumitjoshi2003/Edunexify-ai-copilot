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
from collections.abc import AsyncGenerator
from datetime import date

from fastapi import APIRouter, Header, HTTPException, Request
from fastapi.responses import StreamingResponse
from openai import APIStatusError, AsyncOpenAI, OpenAI

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
from tools.admin_results import get_school_performance_summary, get_class_exam_results
from tools.knowledge_base import search_knowledge_base

router = APIRouter()

# DeepSeek via the OpenAI-compatible endpoint — same tool-calling format,
# no changes needed to TOOLS definitions or message handling.
_client = OpenAI(
    api_key=settings.deepseek_api_key,
    base_url="https://api.deepseek.com",
)

# Async client for /chat/stream only. Streaming with the sync client would mean
# `for chunk in stream:` blocks the event loop for the whole response — fine for
# a single request, bad for a server handling concurrent chats. AsyncOpenAI's
# `async for` yields control back between chunks like everything else here.
_async_client = AsyncOpenAI(
    api_key=settings.deepseek_api_key,
    base_url="https://api.deepseek.com",
)

# Streamed content is held back until it exceeds this length with no tool call
# having started yet — see _stream_chat_response. Comfortably longer than a
# short "let me check that" preamble (observed at 40-55 chars in testing) so a
# real preamble never accidentally crosses it and leaks before a tool call.
_STREAM_HOLD_BACK_CHARS = 120

# ─── Tool definitions (OpenAI format) ─────────────────────────────────────────
# OpenAI wraps each tool in {"type": "function", "function": {...}}.
# The "parameters" field is standard JSON Schema — same content as Anthropic's
# "input_schema", just a different wrapper key name.

# Shared across all three roles — every role can ask about school policies/handbooks
# (see tools/knowledge_base.py). Retrieval itself is schoolId-scoped server-side by
# Spring Boot, same as every other tool, so there's no extra scoping needed here.
_SEARCH_KNOWLEDGE_BASE_TOOL: dict = {
    "type": "function",
    "function": {
        "name": "search_knowledge_base",
        "description": (
            "Search the school's uploaded policy documents and handbooks (attendance policy, "
            "leave policy, fee policy, exam guidelines, student handbook, code of conduct, etc.) "
            "for content relevant to the user's question. Call this when the user asks about a "
            "school POLICY, RULE, or GUIDELINE — e.g. 'what is the leave policy', 'how many sick "
            "leaves am I allowed', 'what are the exam rules', 'what does the handbook say about "
            "uniforms'. Do NOT use this for the user's own live data (actual attendance/fees/marks) "
            "— use the other tools for that."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The user's question, or a short natural-language description of what to search for.",
                },
            },
            "required": ["query"],
        },
    },
}

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
    },
    _SEARCH_KNOWLEDGE_BASE_TOOL,
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
    _SEARCH_KNOWLEDGE_BASE_TOOL,
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
                "Fetch and compare attendance rate across every class in the school. "
                "Call this when the admin asks which classes have the best/worst attendance, or wants "
                "a class-by-class attendance breakdown. "
                "Use type='year' (the default) for the full academic session — use this unless the admin "
                "specifically says 'this month'. Use type='month' ONLY when they explicitly ask about the "
                "current calendar month specifically; that mode can report 0% for every class if attendance "
                "simply hasn't been marked yet this month, which is NOT the same as a real 0% attendance rate — "
                "the tool flags this explicitly when it happens, do not report it as poor attendance."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "type": {
                        "type": "string",
                        "enum": ["year", "month"],
                        "description": "'year' = full academic session (default, use this for general questions). 'month' = current calendar month only.",
                    },
                    "session": {
                        "type": "string",
                        "description": "Academic session in YYYY-YYYY format. Only used when type='year'. Defaults to the current session.",
                    },
                },
                "required": [],
            },
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
                "performing class, a full class-by-class breakdown, weakest subjects school-wide, AND a "
                "school-wide top 5 / bottom 5 individual student leaderboard (schoolWideTopPerformers / "
                "schoolWideNeedingAttention) built from those same per-student results. "
                "Call this when the admin asks which class performed best/worst, wants an academic "
                "performance summary, asks which subjects students are struggling with school-wide, OR asks "
                "which STUDENT is weakest/strongest academically WITHOUT naming a specific class. "
                "If the admin names a specific class (e.g. 'who scored lowest in Class 2's exam'), use "
                "get_class_exam_results instead — that gives the full ranked list for that one class, this "
                "tool only gives the top/bottom 5 across the whole school."
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
    {
        "type": "function",
        "function": {
            "name": "get_class_exam_results",
            "description": (
                "Fetch the FULL per-student ranked results for one named class's exam: top scorer, lowest "
                "scorer, every student's percentage and rank, and subject averages for that class. "
                "This is the ONLY tool that can answer 'who scored lowest/highest in class X's exam' — "
                "no other tool returns an individual student's exam score. Requires className. "
                "Defaults to that class's most recently created exam if examName is omitted."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "className": {
                        "type": "string",
                        "description": "The class to fetch results for, e.g. '10' or 'Class 2'. Required.",
                    },
                    "session": {
                        "type": "string",
                        "description": "Academic session in YYYY-YYYY format. Defaults to the current session.",
                    },
                    "examName": {
                        "type": "string",
                        "description": "Specific exam name (e.g. 'Final Year Exam'). If omitted, the latest exam for that class is used.",
                    },
                },
                "required": ["className"],
            },
        },
    },
    _SEARCH_KNOWLEDGE_BASE_TOOL,
]


_STUDENT_PROMPT_BODY = """## Available tools
- get_fee_summary — use when the student asks about fees, pending payments, paid months, or outstanding amounts.
- get_attendance_summary — use when the student asks about attendance, absences, or attendance percentage.
- get_results_summary — use when the student asks about marks, exam results, scores, grades, rank, or performance.
- search_knowledge_base — use when the student asks about a school POLICY, RULE, or GUIDELINE (e.g. leave policy, attendance policy, exam rules, handbook, uniform rules) rather than their own live data.

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

## When answering from the knowledge base (search_knowledge_base)
- Treat the returned chunks as the ONLY source for policy content — never add typical/generic school-policy knowledge from general training, even if it sounds plausible or is commonly true elsewhere.
- Preserve the exact modal strength of the source wording: "may" is not "must" or "is required"; "should" is not "must"; "can" is not "will". Never upgrade optional/permissive language into a requirement, or soften a requirement into an option.
- If a chunk doesn't state a specific number, amount, deadline, or procedure, say so — do NOT invent one. Example: a chunk saying "lost books may require replacement or payment according to library rules" has no stated amount — say the policy doesn't specify an exact fee, don't name a dollar figure.
- Each result has a similarity score; results below a relevance floor are already excluded server-side. If found=false, or the returned chunks don't actually address what was asked, say plainly that the Knowledge Base doesn't have enough information on this — do NOT answer from general knowledge, and do NOT imply a negative ("the school doesn't offer this") when the real situation is "not covered by the documents."
- Keep this evidence separate from live tool data (attendance/fees/marks) — never blend a retrieved policy detail with a number from another tool.

## General behaviour
- Never guess or fabricate numbers — only report what tools return.
- Only use numbers from a tool call made THIS turn — never reuse or repurpose a number mentioned in an earlier reply for a new, different question (e.g. an attendance percentage is not an exam score). Call the relevant tool again if you need current data.
- If a tool returns an error or a "message" field (no data yet), relay that clearly.
- When the user asks about multiple topics, call all relevant tools and present results in sections.
- Use emojis sparingly — only where they add meaning, not as decoration. Never use a cheerful emoji (😊 😄) when reporting a problem like unpaid fees or low attendance."""

_TEACHER_PROMPT_BODY = """## Available tools
- get_class_attendance_summary — use when the teacher asks how their class's attendance is doing overall (averages, how many students are below 75%).
- get_low_attendance_students — use when the teacher asks WHICH students have low/poor attendance, are missing too many days, or need attention on attendance.
- get_class_performance_summary — use when the teacher asks how their class performed in an exam, which students need academic attention, or which subjects students are struggling with.
- search_knowledge_base — use when the teacher asks about a school POLICY, RULE, or GUIDELINE (e.g. leave policy, attendance policy, exam guidelines, handbook) rather than live class data.

The three class-data tools are automatically scoped to the teacher's own assigned class — there is no className argument, so you never need to ask the teacher which class they mean. search_knowledge_base is school-wide, not class-scoped.

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

## When answering from the knowledge base (search_knowledge_base)
- Treat the returned chunks as the ONLY source for policy content — never add typical/generic school-policy knowledge from general training, even if it sounds plausible or is commonly true elsewhere.
- Preserve the exact modal strength of the source wording: "may" is not "must" or "is required"; "should" is not "must"; "can" is not "will". Never upgrade optional/permissive language into a requirement, or soften a requirement into an option.
- If a chunk doesn't state a specific number, amount, deadline, or procedure, say so — do NOT invent one. Example: a chunk saying "lost books may require replacement or payment according to library rules" has no stated amount — say the policy doesn't specify an exact fee, don't name a dollar figure.
- Each result has a similarity score; results below a relevance floor are already excluded server-side. If found=false, or the returned chunks don't actually address what was asked, say plainly that the Knowledge Base doesn't have enough information on this — do NOT answer from general knowledge, and do NOT imply a negative ("the school doesn't offer this") when the real situation is "not covered by the documents."
- Keep this evidence separate from live tool data (attendance/fees/marks) — never blend a retrieved policy detail with a number from another tool.

## General behaviour
- Never guess or fabricate numbers, student names, or subjects — only report what tools return.
- Only use numbers from a tool call made THIS turn — never reuse or repurpose a number mentioned in an earlier reply for a new, different question (e.g. an attendance percentage is not an exam score). Call the relevant tool again if you need current data.
- If a tool returns an error or a "message" field (no data yet), relay that clearly.
- When the user asks about multiple topics, call all relevant tools and present results in sections.
- Use emojis sparingly — only where they add meaning, not as decoration. Never use a cheerful emoji (😊 😄) when reporting a problem like low attendance or weak performance."""

_ADMIN_PROMPT_BODY = """## Available tools
- get_school_overview — use for an overall school summary or general status check (students, teachers, fees this month, overdue count, today's attendance, pending leaves).
- get_class_attendance_comparison — use when the admin asks which classes have the best/worst attendance, or wants a class-by-class breakdown. Defaults to the full session; only pass type='month' if they explicitly say "this month".
- get_school_low_attendance_students — use when the admin asks WHICH specific students (not just which classes) have low attendance, school-wide. Returns attendance PERCENTAGES only — never exam data.
- get_fee_defaulters — use when the admin asks about pending fees, how many students owe money, or fee defaulters.
- get_school_performance_summary — use for class-level exam performance (best/worst class, weakest subjects) AND for "which student is weakest/strongest academically" when no specific class is named (see schoolWideTopPerformers / schoolWideNeedingAttention).
- get_class_exam_results — the ONLY tool with individual exam scores for a named class (top scorer, lowest scorer, full ranked list). Use whenever the admin names a specific class and asks about individual student performance in it (e.g. "who scored lowest in Class 2's exam").
- search_knowledge_base — use when the admin asks about a school POLICY, RULE, or GUIDELINE (e.g. leave policy, attendance policy, fee policy, exam guidelines, handbook) rather than live structured data.

For broad questions (e.g. "give me an overall school summary"), call multiple relevant tools in the same turn and present the results in sections — don't limit yourself to one.
For ambiguous requests like "which students/classes need attention", consider calling get_school_low_attendance_students, get_fee_defaulters, and get_school_performance_summary together unless the phrasing clearly points to just one domain.

## Tool usage rules
Only call a tool when the admin explicitly asks about school data.
Do NOT call any tool for:
- Greetings or small talk ("hi", "hello", "how are you")
- Questions about your identity or capabilities ("who are you", "what can you help me with")
- General questions that do not require school data

## CRITICAL — do not confuse metrics or reuse old numbers
These are DIFFERENT metrics from DIFFERENT tools — never substitute one for another, even when the numbers look plausible:
- Attendance percentage (get_school_low_attendance_students, get_class_attendance_comparison) is NOT an exam score. A student flagged for low attendance did not "score" that percentage on any exam.
- If asked about an individual student's exam score, rank, or academic standing, you MUST call get_class_exam_results (if a class is named) or use get_school_performance_summary's schoolWideTopPerformers/schoolWideNeedingAttention (if not). There is no other legitimate source for this.
- Only use numbers that came from a tool call made THIS turn (or earlier in the same turn's tool-calling loop). Never reuse or repurpose a number mentioned in an earlier reply in the conversation history to answer a new, different question — call the relevant tool again instead.
- If no available tool can answer what's being asked, say so plainly. Do not estimate, infer, or present a guess as if it were fetched data.

## When showing school overview data
- Lead with student/teacher counts and today's attendance rate.
- Call out overdue students and pending leaves as action items, not just numbers.

## When showing class attendance comparison
- Name the lowest and highest attendance classes specifically, with their percentages.
- State the period the data covers (the tool's "period" field) — full session by default, or "current calendar month" if type='month' was used.
- If classesWithNoAttendanceRecordedYet is non-empty, say clearly that those classes have no attendance marked yet for that period — do NOT report them as having 0% attendance, that's a different thing.

## When showing individual class exam results (get_class_exam_results)
- State topScorer and lowestScorer by name and percentage — these come directly from the tool, do not calculate or guess them yourself.
- Mention studentsWithNoMarksEntered if non-empty, so the admin knows those students aren't reflected in the ranking yet.
- List subjectPerformance from weakest to strongest for that class.

## When showing school performance data (get_school_performance_summary)
- Name the best and worst performing class specifically, with their exam name and percentage — note if they're different exams.
- List weakestSubjectsSchoolWide explicitly; that's usually the most actionable insight.
- When asked which student is weakest/strongest school-wide, use schoolWideTopPerformers / schoolWideNeedingAttention and relay the tool's "note" field (different classes may be on different exams, so this is an approximation).
- If noResultsYet is true, say clearly no exam results are available yet — do NOT say "unable to fetch".
- classesWithNoExamYet > 0 means some classes were skipped because they have no exam configured yet — mention this if relevant rather than implying full school coverage.

## When showing low-attendance students
- List students by name with their class and attendance percentage, lowest first.
- If truncated is true, mention there are more below the threshold than shown (studentsBelowThreshold gives the real total).
- If studentsBelowThreshold is 0, say clearly that's good news — don't imply a problem.

## When showing fee defaulters
- Lead with total defaulter count and total amount due.
- Break down by class if byClass has more than one entry.
- Name the most overdue students specifically from mostOverdue.

## When answering from the knowledge base (search_knowledge_base)
- Treat the returned chunks as the ONLY source for policy content — never add typical/generic school-policy knowledge from general training, even if it sounds plausible or is commonly true elsewhere.
- Preserve the exact modal strength of the source wording: "may" is not "must" or "is required"; "should" is not "must"; "can" is not "will". Never upgrade optional/permissive language into a requirement, or soften a requirement into an option.
- If a chunk doesn't state a specific number, amount, deadline, or procedure, say so — do NOT invent one. Example: a chunk saying "lost books may require replacement or payment according to library rules" has no stated amount — say the policy doesn't specify an exact fee, don't name a dollar figure.
- Each result has a similarity score; results below a relevance floor are already excluded server-side. If found=false, or the returned chunks don't actually address what was asked, say plainly that the Knowledge Base doesn't have enough information on this — do NOT answer from general knowledge, and do NOT imply a negative ("the school doesn't offer this") when the real situation is "not covered by the documents."
- Keep this evidence separate from live tool data (attendance/fees/marks) — never blend a retrieved policy detail with a number from another tool.

## General behaviour
- Never guess or fabricate numbers, student names, class names, or subjects — only report what tools return.
- If a tool returns an "error" field, relay that message plainly rather than guessing at data.
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

## Responses are streamed live to the user
Your replies are streamed to the chat bubble as you generate them — the user sees text appear as you write it.
When you decide to call a tool, call it directly with NO preceding text — do not write "Let me check that",
"I'll look that up", "One moment", or any other narration before or instead of a tool call. Any text you produce
in the same turn as a tool call is shown to the user immediately and cannot be taken back, so only ever produce
text there if it's part of your genuine final answer. Write your actual response only in the turn after your
tool results come back, once you have real data to report.

## Response style
You are a professional school-admin assistant, not a chatbot demo — write like a knowledgeable staff member,
not an AI narrating its own process.
- Answer the user's exact question first. Lead with the answer, not a preamble.
- Default to concise: roughly 2-3 sentences, unless the question genuinely needs more (a multi-subject
  breakdown, a list of several students, several policy points) or the user explicitly asks for more detail.
- Never narrate what you did to get the answer — don't say "Based on the Knowledge Base", "According to the
  policy documents", "I searched...", "Let me check that for you", or similar. Just state the answer; the
  citation (see below) handles attribution, you don't need to say it in prose too.
- Don't tack on unprompted offers like "Would you like me to check X?" or "Let me know if you need anything
  else" — only suggest a next step when it's the genuinely obvious continuation of what they just asked.

## Citations
Cite a source ONLY when your answer draws on search_knowledge_base — never for attendance/fee/exam/other live
tool data, and never one citation per data source when an answer combines both. Put it on its own line at the
very end of the answer, nothing else on that line:
📄 <document title>[ · Page N]
Use the document title and page number exactly as the tool gives them (omit the page part if none was given).
Never mention similarity scores, chunk numbers/text, "results", or any other retrieval/tool-internal detail —
those exist for you to reason with, not for the user to see.

## Attendance vs. leave
"Absent" (attendance data) and "on approved leave" are not the same thing — a student can be marked absent
without ever submitting a leave request, and a submitted request is not automatically approved. Never describe
attendance-absence numbers as "leave" or as "approved leave" unless you actually have leave-specific data
confirming a request and its approval status. If asked about leave status/approval and you have no leave data
available, say you don't have that information rather than inferring it from attendance.

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
        return await get_class_attendance_comparison(
            user=user,
            access_token=access_token,
            type=tool_input.get("type", "year"),
            session=tool_input.get("session"),
        )

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

    if tool_name == "get_class_exam_results":
        return await get_class_exam_results(
            user=user,
            access_token=access_token,
            className=tool_input["className"],
            session=tool_input.get("session"),
            examName=tool_input.get("examName"),
        )

    if tool_name == "search_knowledge_base":
        return await search_knowledge_base(
            user=user,
            access_token=access_token,
            query=tool_input["query"],
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
            # Lower than the ~1.0 default: this app's answers are grounded in tool/RAG
            # data, not creative writing, and a lower temperature measurably reduces
            # the model embellishing retrieved text with plausible-sounding but
            # unsupported specifics (see the Knowledge Base grounding fixes).
            completion_kwargs: dict = {"model": "deepseek-chat", "messages": messages, "temperature": 0.3}
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


# ─── Streaming endpoint ─────────────────────────────────────────────────────
# Same orchestration as /chat (same system prompt, same role-scoped tools, same
# _execute_tool dispatch, same memory contract) — the only difference is HOW the
# final answer reaches the caller. Every call in the loop below is made with
# stream=True. A turn where the model wants to call a tool produces no (or
# empty) content deltas — DeepSeek only emits real text once it's done calling
# tools — so simply forwarding content deltas as they arrive naturally streams
# ONLY the final user-facing answer; tool-calling turns stay entirely internal
# without any special-casing. Tool-call deltas (name/arguments fragments) are
# accumulated but never written to the response stream.


async def _stream_chat_response(request: ChatRequest, http_request: Request) -> AsyncGenerator[str, None]:
    history = await memory.get_history(
        request.user.schoolId, request.user.userId, request.conversationId
    )

    messages: list[dict] = [
        {"role": "system", "content": _build_system_prompt(request.user)},
        *history,
        {"role": "user", "content": request.message},
    ]

    role_tools = {
        "STUDENT": STUDENT_TOOLS,
        "TEACHER": TEACHER_TOOLS,
        "ADMIN": ADMIN_TOOLS,
    }.get(request.user.role, [])

    full_reply = ""
    errored = False
    interrupted = False  # user hit "stop" — client gone, no point doing more work

    try:
        for _ in range(5):  # Safety cap — prevents infinite loops, same as /chat
            completion_kwargs: dict = {
                "model": "deepseek-chat", "messages": messages, "stream": True, "temperature": 0.3,
            }
            if role_tools:
                completion_kwargs["tools"] = role_tools
                completion_kwargs["tool_choice"] = "auto"

            stream = await _async_client.chat.completions.create(**completion_kwargs)

            turn_content = ""       # everything produced this turn, for context reconstruction below
            held_back = ""          # content not yet released to the client
            committed = False       # True once we've decided this turn is safe to stream live
            saw_tool_call = False
            # index -> accumulated {id, name, arguments} fragments for this turn's tool call(s)
            tool_calls_acc: dict[int, dict] = {}
            finish_reason: str | None = None

            async for chunk in stream:
                if await http_request.is_disconnected():
                    # User hit "stop" (or the tab/connection just died). Bytes we yield
                    # from here on have no one to reach — stop pulling more tokens from
                    # DeepSeek immediately rather than burning the rest of the turn.
                    interrupted = True
                    break

                choice = chunk.choices[0]
                if choice.finish_reason:
                    finish_reason = choice.finish_reason

                delta = choice.delta

                if delta.tool_calls:
                    saw_tool_call = True
                    for tc_delta in delta.tool_calls:
                        entry = tool_calls_acc.setdefault(tc_delta.index, {"id": None, "name": "", "arguments": ""})
                        if tc_delta.id:
                            entry["id"] = tc_delta.id
                        if tc_delta.function:
                            if tc_delta.function.name:
                                entry["name"] += tc_delta.function.name
                            if tc_delta.function.arguments:
                                entry["arguments"] += tc_delta.function.arguments

                if delta.content:
                    turn_content += delta.content

                    if saw_tool_call:
                        # A tool call already started this turn — the prompt asks the model not to
                        # narrate before calling a tool, but it isn't always obeyed (observed in
                        # testing), and bytes already sent over HTTP can't be recalled. So: hold back
                        # by default (below) and only ever release content once we're confident no
                        # tool call is coming. Once one HAS arrived, anything further is discarded too.
                        continue

                    if committed:
                        full_reply += delta.content
                        yield delta.content
                    else:
                        held_back += delta.content
                        # Past a short preamble's length with still no tool call in sight — safe to
                        # start streaming live from here. Flushes the held-back text as one small
                        # catch-up burst, then every following chunk streams individually.
                        if len(held_back) > _STREAM_HOLD_BACK_CHARS:
                            committed = True
                            full_reply += held_back
                            yield held_back
                            held_back = ""

            if interrupted:
                await stream.close()  # release the DeepSeek connection immediately, don't let it keep generating
                break

            if not saw_tool_call and held_back:
                # Turn ended (short final answer) without ever crossing the threshold — flush the rest.
                full_reply += held_back
                yield held_back

            if finish_reason == "tool_calls":
                ordered_calls = [tool_calls_acc[i] for i in sorted(tool_calls_acc)]
                messages.append({
                    "role": "assistant",
                    "content": turn_content or None,
                    "tool_calls": [
                        {
                            "id": c["id"],
                            "type": "function",
                            "function": {"name": c["name"], "arguments": c["arguments"]},
                        }
                        for c in ordered_calls
                    ],
                })

                for c in ordered_calls:
                    tool_input = json.loads(c["arguments"]) if c["arguments"] else {}
                    tool_output = await _execute_tool(c["name"], tool_input, request.user, request.accessToken)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": c["id"],
                        "content": json.dumps(tool_output),
                    })
                continue  # next iteration streams the follow-up turn

            break  # finish_reason == "stop" (or unexpected) — done

    except APIStatusError as e:
        errored = True
        if e.status_code == 402 or e.status_code == 429:
            msg = "\n\n⚠️ AI Copilot is temporarily unavailable. Please try again later."
        else:
            msg = f"\n\n⚠️ AI provider error ({e.status_code}). Please try again."
        full_reply += msg
        yield msg
    except Exception as e:
        errored = True
        msg = "\n\n⚠️ AI Copilot is temporarily unavailable. Please try again later."
        full_reply += msg
        yield msg

    if not full_reply and not interrupted:
        full_reply = "I wasn't able to complete your request. Please try again."
        yield full_reply

    # Only persist a clean, complete reply — never a partial/errored/interrupted one, so
    # a broken or stopped turn doesn't pollute the next turn's memory with a garbled reply.
    if not errored and not interrupted:
        await memory.append_turn(
            request.user.schoolId,
            request.user.userId,
            request.conversationId,
            history,
            request.message,
            full_reply,
        )


@router.post("/chat/stream")
async def chat_stream(
    request: ChatRequest,
    http_request: Request,
    x_internal_secret: str = Header(alias="X-Internal-Secret"),
) -> StreamingResponse:
    if x_internal_secret != settings.internal_secret:
        raise HTTPException(status_code=401, detail="Unauthorized")

    return StreamingResponse(_stream_chat_response(request, http_request), media_type="text/plain; charset=utf-8")
