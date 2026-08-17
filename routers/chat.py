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
import time
from collections.abc import AsyncGenerator
from datetime import date

from fastapi import APIRouter, Header, HTTPException, Request
from fastapi.responses import StreamingResponse
from openai import APIStatusError, AsyncOpenAI, OpenAI

import memory
from config import settings
from tracing import Trace, summarize_tool_result
from schemas.chat import ChatRequest, ChatResponse, UserContext
from academic_calendar import current_academic_session
from tools.attendance import get_attendance_summary
from tools.fees import get_fee_summary
from tools.results import get_results_summary
from tools.teacher_attendance import (
    get_class_attendance_summary,
    get_consecutively_absent_students,
    get_low_attendance_students,
)
from tools.teacher_attendance_workflows import start_teacher_attendance_reminder_workflow
from tools.teacher_results import get_class_performance_summary
from tools.admin_dashboard import get_school_overview, get_class_attendance_comparison
from tools.admin_attendance import get_school_low_attendance_students
from tools.admin_staff_attendance import get_staff_attendance_by_date, get_staff_attendance_month_summary
from tools.admin_fees import get_fee_defaulters
from tools.admin_fee_workflows import start_fee_reminder_workflow
from tools.admin_attendance_workflows import start_attendance_reminder_workflow
from tools.admin_results import get_school_performance_summary, get_class_exam_results
from tools.leave_requests import get_pending_leave_requests, get_leave_request_details
from tools.leave_workflows import start_leave_decision_workflow
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
            "name": "get_consecutively_absent_students",
            "description": (
                "List students in the teacher's class who have been absent for several school days "
                "IN A ROW, with how many consecutive days and the exact dates. Call this for "
                "questions about RECENT or ONGOING absence — 'who has been absent the last 3 days', "
                "'which students haven't shown up this week', 'anyone missing several days in a "
                "row'. This is NOT the same as get_low_attendance_students: that one finds students "
                "whose cumulative percentage is low for the whole year, this one finds students who "
                "have stopped attending recently (who may still have a good overall percentage). "
                "Read-only — it never sends anything."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "minDays": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 60,
                        "description": "How many consecutive school days of absence to look for. Defaults to 3.",
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
    {
        "type": "function",
        "function": {
            "name": "start_teacher_attendance_reminder_workflow",
            "description": (
                "Starts the low-attendance warning review-and-approval workflow for the "
                "teacher's own class — call this when the teacher asks to SEND, WARN, REMIND, "
                "or NOTIFY parents of low-attendance students (an action), never for a question "
                "about who has low attendance (use get_low_attendance_students for that). This "
                "does NOT send any email itself: it prepares a batch and shows the teacher an "
                "approval card; sending only happens if the teacher clicks Approve on that card. "
                "There is no way to skip this review step, and there is no parameter to choose "
                "recipients or a class — the workflow always fetches the full, authoritative "
                "list of qualifying students in the teacher's own class itself. "
                "Set criterion='CONSECUTIVE_ABSENCE' when the teacher is talking about students "
                "absent several days IN A ROW ('absent the last 3 days', 'missing all week'); "
                "leave it as 'BELOW_THRESHOLD' when they mean a cumulative percentage ('below "
                "70%', 'poor attendance this year')."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "session": {
                        "type": "string",
                        "description": "Academic session in YYYY-YYYY format. Defaults to the current session.",
                    },
                    "criterion": {
                        "type": "string",
                        "enum": ["BELOW_THRESHOLD", "CONSECUTIVE_ABSENCE"],
                        "description": (
                            "How to pick students. 'BELOW_THRESHOLD' = cumulative attendance under "
                            "`threshold`%. 'CONSECUTIVE_ABSENCE' = absent the last `minConsecutiveDays` "
                            "school days in a row. Defaults to BELOW_THRESHOLD."
                        ),
                    },
                    "threshold": {
                        "type": "number",
                        "description": "Attendance percentage cutoff, used only with criterion='BELOW_THRESHOLD'. Defaults to 75.",
                    },
                    "minConsecutiveDays": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 60,
                        "description": "Consecutive absent school days required, used only with criterion='CONSECUTIVE_ABSENCE'. Defaults to 3.",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_pending_leave_requests",
            "description": (
                "List leave requests still awaiting a decision, oldest first, each with its "
                "leaveId, student, date and reason. Call this when asked who has applied for "
                "leave, what leave is pending, or to review leave requests. READ-ONLY — it never "
                "approves or rejects anything. Always call this (or get_leave_request_details) "
                "before starting a decision, so you act on real leaveIds rather than guessing."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "studentId": {"type": "string", "description": "Restrict to one student's requests. Omit for all."},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 100, "description": "Max requests to return. Defaults to 25."},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_leave_request_details",
            "description": (
                "Fetch specific leave requests by their leaveId, showing each one's CURRENT "
                "status (PENDING, APPROVED or REJECTED). Use this to confirm what a request is "
                "before acting on it, or to check what happened to one. READ-ONLY."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "leaveIds": {"type": "array", "items": {"type": "integer"}, "description": "The leaveId values to look up."},
                },
                "required": ["leaveIds"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "start_leave_decision_workflow",
            "description": (
                "Starts the review-and-approval workflow for APPROVING or REJECTING specific "
                "leave requests — call this when the user asks to approve, reject, grant, deny or "
                "decline leave, including for requests discussed earlier in the conversation "
                "(e.g. 'approve the first two', 'reject Riley\'s'). This does NOT change anything "
                "itself: it prepares the decision and shows an approval card; the change only "
                "happens if the user clicks Approve on that card. "
                "You MUST pass real leaveId values obtained from get_pending_leave_requests or "
                "get_leave_request_details in this conversation — never invent, guess, or "
                "increment an id. If you do not have real ids yet, call a read tool first."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "decision": {"type": "string", "enum": ["APPROVED", "REJECTED"], "description": "What to do with the listed leave requests."},
                    "leaveIds": {"type": "array", "items": {"type": "integer"}, "description": "leaveId values to act on, from a read tool in this conversation."},
                },
                "required": ["decision", "leaveIds"],
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
            "name": "get_staff_attendance_by_date",
            "description": (
                "Fetch every teacher's attendance STATUS for one date (default: today), grouped "
                "into presentTeachers, lateTeachers, absentTeachers, halfDayTeachers, and "
                "onLeaveTeachers, each with names. This is STAFF/TEACHER attendance (did teachers "
                "check in), never student attendance. Call this when the admin asks who is absent "
                "today, who was late today, which teachers are on leave today, or asks about staff "
                "attendance for a specific date."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {
                        "type": "string",
                        "description": "Date in YYYY-MM-DD format. Omit for today (resolved in the school's own timezone).",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_staff_attendance_month_summary",
            "description": (
                "Fetch whole-school STAFF/TEACHER attendance totals for one month (present/late/"
                "absent/half-day/on-leave day counts, on-time percentage), plus a ranked list of "
                "teachers with the most absences that month. Call this when the admin asks for a "
                "staff attendance summary for a month, or asks which teachers have the most "
                "absences/lates. Defaults to the current month if not specified."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "month": {"type": "integer", "minimum": 1, "maximum": 12, "description": "1-12. Defaults to the current month."},
                    "year": {"type": "integer", "description": "e.g. 2026. Defaults to the current year."},
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
            "name": "start_fee_reminder_workflow",
            "description": (
                "Starts the fee reminder review-and-approval workflow — call this when the admin "
                "asks to SEND, REMIND, or NOTIFY fee defaulters (an action), never for a question "
                "about defaulters (use get_fee_defaulters for that). This does NOT send any email "
                "itself: it prepares a batch and shows the admin an approval card; sending only "
                "happens if the admin clicks Approve on that card. There is no way to skip this "
                "review step, and there is no parameter to choose recipients — the workflow always "
                "fetches the full, authoritative list of current fee defaulters itself."
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
                        "description": "Restrict the batch to one class, e.g. '10'. Omit for the whole school.",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "start_attendance_reminder_workflow",
            "description": (
                "Starts the low-attendance warning review-and-approval workflow — call this when "
                "the admin asks to SEND, WARN, REMIND, or NOTIFY parents of students with low "
                "attendance (an action), never for a question about who has low attendance (use "
                "get_school_low_attendance_students for that). This does NOT send any email "
                "itself: it prepares a batch and shows the admin an approval card; sending only "
                "happens if the admin clicks Approve on that card. There is no way to skip this "
                "review step, and there is no parameter to choose recipients — the workflow always "
                "fetches the full, authoritative list of currently-below-threshold students itself."
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
                        "description": "Restrict the batch to one class, e.g. '10'. Omit for the whole school.",
                    },
                    "threshold": {
                        "type": "number",
                        "description": "Attendance percentage cutoff. Students below this are included. Defaults to 75.",
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
    {
        "type": "function",
        "function": {
            "name": "get_pending_leave_requests",
            "description": (
                "List leave requests still awaiting a decision, oldest first, each with its "
                "leaveId, student, date and reason. Call this when asked who has applied for "
                "leave, what leave is pending, or to review leave requests. READ-ONLY — it never "
                "approves or rejects anything. Always call this (or get_leave_request_details) "
                "before starting a decision, so you act on real leaveIds rather than guessing."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "studentId": {"type": "string", "description": "Restrict to one student's requests. Omit for all."},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 100, "description": "Max requests to return. Defaults to 25."},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_leave_request_details",
            "description": (
                "Fetch specific leave requests by their leaveId, showing each one's CURRENT "
                "status (PENDING, APPROVED or REJECTED). Use this to confirm what a request is "
                "before acting on it, or to check what happened to one. READ-ONLY."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "leaveIds": {"type": "array", "items": {"type": "integer"}, "description": "The leaveId values to look up."},
                },
                "required": ["leaveIds"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "start_leave_decision_workflow",
            "description": (
                "Starts the review-and-approval workflow for APPROVING or REJECTING specific "
                "leave requests — call this when the user asks to approve, reject, grant, deny or "
                "decline leave, including for requests discussed earlier in the conversation "
                "(e.g. 'approve the first two', 'reject Riley\'s'). This does NOT change anything "
                "itself: it prepares the decision and shows an approval card; the change only "
                "happens if the user clicks Approve on that card. "
                "You MUST pass real leaveId values obtained from get_pending_leave_requests or "
                "get_leave_request_details in this conversation — never invent, guess, or "
                "increment an id. If you do not have real ids yet, call a read tool first."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "decision": {"type": "string", "enum": ["APPROVED", "REJECTED"], "description": "What to do with the listed leave requests."},
                    "leaveIds": {"type": "array", "items": {"type": "integer"}, "description": "leaveId values to act on, from a read tool in this conversation."},
                },
                "required": ["decision", "leaveIds"],
            },
        },
    },
    _SEARCH_KNOWLEDGE_BASE_TOOL,
]

# Defense-in-depth for _execute_tool: derived (never hand-duplicated) from the same
# STUDENT_TOOLS/TEACHER_TOOLS/ADMIN_TOOLS lists used to build each role's schema — so it can't
# drift out of sync with what's actually offered to the model. Without this, the only thing
# stopping a role from executing a tool meant for a different role was the model never being
# handed that tool's schema; there was no server-side check in the dispatch layer itself.
_TOOL_NAMES_BY_ROLE: dict[str, set[str]] = {
    "STUDENT": {t["function"]["name"] for t in STUDENT_TOOLS},
    "TEACHER": {t["function"]["name"] for t in TEACHER_TOOLS},
    "ADMIN": {t["function"]["name"] for t in ADMIN_TOOLS},
}


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
- Never infer the inverse or opposite of a stated rule. A chunk saying a requirement applies under condition X (e.g. "for absences of 3+ days") says nothing about what happens when X doesn't hold — do not add "for shorter absences, it's generally not required" or similar unless the text actually says that. If the user's exact scenario isn't addressed, say the policy doesn't cover that specific case rather than assuming the opposite of what's stated.
- For a simple factual policy lookup, answer in one sentence stating the rule as given, then cite — don't add extra caveats, exceptions, or related scenarios the user didn't ask about.

## General behaviour
- Never guess or fabricate numbers — only report what tools return.
- Only use numbers from a tool call made THIS turn — never reuse or repurpose a number mentioned in an earlier reply for a new, different question (e.g. an attendance percentage is not an exam score). Call the relevant tool again if you need current data.
- If a tool returns an error or a "message" field (no data yet), relay that clearly.
- When the user asks about multiple topics, call all relevant tools and present results in sections.
- Use emojis sparingly — only where they add meaning, not as decoration. Never use a cheerful emoji (😊 😄) when reporting a problem like unpaid fees or low attendance."""

_TEACHER_PROMPT_BODY = """## Available tools
- get_class_attendance_summary — use when the teacher asks how their class's attendance is doing overall (averages, how many students are below 75%).
- get_low_attendance_students — use when the teacher asks WHICH students have low/poor CUMULATIVE attendance for the year (an INFORMATION request — never sends anything).
- get_consecutively_absent_students — use when the teacher asks about RECENT, ONGOING absence: "absent the last 3 days", "hasn't shown up this week", "missing several days in a row" (an INFORMATION request — never sends anything).
- start_teacher_attendance_reminder_workflow — use when the teacher asks to SEND, WARN, REMIND, or NOTIFY parents about EITHER pattern (an ACTION request). See "Attendance reminders — information vs. action" below before choosing between these.
- get_class_performance_summary — use when the teacher asks how their class performed in an exam, which students need academic attention, or which subjects students are struggling with.
- get_pending_leave_requests / get_leave_request_details — use when the teacher asks who has applied for leave or what a request says (INFORMATION, never changes anything).
- start_leave_decision_workflow — use when the teacher asks to APPROVE or REJECT leave requests (an ACTION). Requires real leaveIds from a read tool first.
- search_knowledge_base — use when the teacher asks about a school POLICY, RULE, or GUIDELINE (e.g. leave policy, attendance policy, exam guidelines, handbook) rather than live class data.

All class-data tools are automatically scoped to the teacher's own assigned class — there is no className argument, so you never need to ask the teacher which class they mean. search_knowledge_base is school-wide, not class-scoped.

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

## When showing consecutively-absent students
- Lead with how many students, and name each one with their consecutiveAbsentDays — that number is the point of the question.
- The days are SCHOOL days the class actually had attendance marked, not calendar days; if you mention dates, use the absentDates the tool returned rather than counting back from today yourself.
- If studentsFound is 0, say plainly that nobody has been absent that many days in a row.
- Their attendancePercentage is included for context — do not present it as the reason they were flagged.

## When showing class performance data
- Lead with the exam name and class average percentage.
- Name the students in studentsNeedingAttention specifically, not just "some students".
- List subjectPerformance from weakest to strongest — call out the weakest subject explicitly since that's usually the most actionable insight.
- If noExamsYet is true, say clearly that no exams are configured yet — do NOT say "unable to fetch".
- If the tool returns an "availableExams" list (exam name not found), tell the teacher which exam names actually exist instead of failing silently.

## Two different attendance patterns — pick the right one
These answer different questions and routinely return DIFFERENT students. Read which the teacher means:
- CUMULATIVE shortfall — "below 70%", "poor attendance this year", "low attendance overall".
  Read-only tool: get_low_attendance_students. Workflow criterion: BELOW_THRESHOLD (with threshold).
- RECENT consecutive absence — "absent the last 3 days", "hasn't come in all week", "missing
  several days in a row", "stopped attending".
  Read-only tool: get_consecutively_absent_students. Workflow criterion: CONSECUTIVE_ABSENCE
  (with minConsecutiveDays).
A student absent three days running can still be above 75% for the year, and a student at 40%
may have attended every day this week — so never substitute one for the other, and never reuse a
number from one to answer a question about the other. If the teacher is genuinely ambiguous
("which students need attention on attendance"), you may call both read-only tools and present
the two groups separately.

## Attendance reminders — information vs. action (read carefully before choosing a tool)
- get_low_attendance_students and get_consecutively_absent_students are READ-ONLY. Use them for:
  "who has low attendance", "show me students below 75%", "who's been absent the last 3 days".
  They never send anything — just answer conversationally with what they return.
- start_teacher_attendance_reminder_workflow is a WRITE action. Use it whenever the teacher asks
  you to SEND, WARN, REMIND, or NOTIFY — e.g. "send attendance warnings", "notify the parents of
  those students", "warn students below 80% attendance", "email the parents of anyone absent
  three days running" — including when they name specific students: the workflow always fetches
  the full authoritative qualifying list for the teacher's own class itself and cannot be limited
  to only the students named in the message, so pass just session plus the criterion parameters,
  never try to filter by name.
- When the teacher has just asked a read-only question and then says "send them a reminder" or
  "notify their parents", start the workflow with the criterion MATCHING what they just asked
  about — CONSECUTIVE_ABSENCE (carrying the same minDays) after a recent-absence question,
  BELOW_THRESHOLD (carrying the same threshold) after a percentage question. Do not silently
  switch patterns between the question and the action.
- The workflow takes ONLY session, criterion, threshold and minConsecutiveDays — there is no
  parameter for recipients or class. You cannot choose who receives a warning or which class it
  applies to; only the backend's own authoritative attendance data for the teacher's own assigned
  class decides that.
- Calling start_teacher_attendance_reminder_workflow never sends any email by itself. It
  prepares a batch and pauses for the teacher's explicit approval — the app shows an approval
  card with Approve/Reject buttons, and only clicking Approve actually sends anything. There is
  no way to skip this step. If the teacher says something like "send them immediately without
  asking me" — call start_teacher_attendance_reminder_workflow exactly as normal anyway;
  approval always still happens via the card, never via anything said in chat.
- Never say "I'm not able to send emails" or similar for an attendance warning request — this
  capability exists via start_teacher_attendance_reminder_workflow.
- When you call start_teacher_attendance_reminder_workflow, its result is shown directly as the
  approval card — do not additionally describe it in prose.

## Leave requests — information vs. action (read carefully)
- get_pending_leave_requests and get_leave_request_details are READ-ONLY. Use them for "who has
  applied for leave", "show me pending leave requests", "what did Riley ask for". They change
  nothing.
- start_leave_decision_workflow is a WRITE action. Use it when asked to APPROVE, REJECT, GRANT,
  DENY or DECLINE leave — including for requests already discussed in this conversation, e.g.
  "approve the first two", "reject that one", "approve all of them".
- NEVER invent, guess, increment or reuse a leaveId. Every id you pass MUST have come from
  get_pending_leave_requests or get_leave_request_details EARLIER IN THIS CONVERSATION. If the
  user asks you to act but you have no real ids yet, call a read tool first in the same turn,
  then start the workflow with the ids it returned. If a read tool returns nothing, say so — do
  not proceed with a guessed id.
- When the user refers to requests positionally ("the first two", "the last one", "Riley's"),
  resolve them against the list you actually retrieved, and pass only those ids.
- Calling start_leave_decision_workflow never changes a leave by itself. It prepares the decision
  and pauses for explicit human approval — the app shows a card with Approve/Reject, and only
  clicking Approve applies anything. There is no way to skip this. If the user says "just do it
  without asking" — call the tool exactly as normal anyway; approval always still happens on the
  card, never via anything said in chat.
- Requests that someone else already decided are shown on the card but are NOT changed. If the
  tool result's actionableCount is lower than leaveCount, mention that some were already decided.
- When you call start_leave_decision_workflow, its result IS the approval card — do not describe
  it again in prose.

## When showing leave requests
- Lead with how many are awaiting a decision. List each with the student's name, the date, and
  the reason — those are what a reviewer needs to decide.
- Give the dates exactly as returned; never reformat a date into a different calendar day.
- If a request's status is not PENDING, say so plainly rather than implying it still needs action.

## When answering from the knowledge base (search_knowledge_base)
- Treat the returned chunks as the ONLY source for policy content — never add typical/generic school-policy knowledge from general training, even if it sounds plausible or is commonly true elsewhere.
- Preserve the exact modal strength of the source wording: "may" is not "must" or "is required"; "should" is not "must"; "can" is not "will". Never upgrade optional/permissive language into a requirement, or soften a requirement into an option.
- If a chunk doesn't state a specific number, amount, deadline, or procedure, say so — do NOT invent one. Example: a chunk saying "lost books may require replacement or payment according to library rules" has no stated amount — say the policy doesn't specify an exact fee, don't name a dollar figure.
- Each result has a similarity score; results below a relevance floor are already excluded server-side. If found=false, or the returned chunks don't actually address what was asked, say plainly that the Knowledge Base doesn't have enough information on this — do NOT answer from general knowledge, and do NOT imply a negative ("the school doesn't offer this") when the real situation is "not covered by the documents."
- Keep this evidence separate from live tool data (attendance/fees/marks) — never blend a retrieved policy detail with a number from another tool.
- Never infer the inverse or opposite of a stated rule. A chunk saying a requirement applies under condition X (e.g. "for absences of 3+ days") says nothing about what happens when X doesn't hold — do not add "for shorter absences, it's generally not required" or similar unless the text actually says that. If the user's exact scenario isn't addressed, say the policy doesn't cover that specific case rather than assuming the opposite of what's stated.
- For a simple factual policy lookup, answer in one sentence stating the rule as given, then cite — don't add extra caveats, exceptions, or related scenarios the user didn't ask about.

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
- start_attendance_reminder_workflow — use when the admin asks to SEND, WARN, REMIND, or NOTIFY parents of low-attendance students (an ACTION request). See "Attendance reminders — information vs. action" below before choosing between these two.
- get_staff_attendance_by_date — use when the admin asks who is absent/late/on leave TODAY (or on a specific date), for STAFF/TEACHERS (not students). READ-ONLY.
- get_staff_attendance_month_summary — use when the admin asks for a staff attendance summary for a month, or which teachers have the most absences/lates this month. READ-ONLY.
- get_fee_defaulters — use when the admin asks to SEE, COUNT, or LIST fee defaulters (an INFORMATION request — never sends anything).
- start_fee_reminder_workflow — use when the admin asks to SEND, REMIND, or NOTIFY fee defaulters (an ACTION request). See "Fee reminders — information vs. action" below before choosing between these two.
- get_school_performance_summary — use for class-level exam performance (best/worst class, weakest subjects) AND for "which student is weakest/strongest academically" when no specific class is named (see schoolWideTopPerformers / schoolWideNeedingAttention).
- get_class_exam_results — the ONLY tool with individual exam scores for a named class (top scorer, lowest scorer, full ranked list). Use whenever the admin names a specific class and asks about individual student performance in it (e.g. "who scored lowest in Class 2's exam").
- get_pending_leave_requests / get_leave_request_details — use when the admin asks who has applied for leave or what a request says (INFORMATION, never changes anything).
- start_leave_decision_workflow — use when the admin asks to APPROVE or REJECT leave requests (an ACTION). Requires real leaveIds from a read tool first.
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

## When showing staff/teacher attendance data (get_staff_attendance_by_date, get_staff_attendance_month_summary)
- This is STAFF/TEACHER attendance — completely separate from student attendance
  (get_school_low_attendance_students, get_class_attendance_comparison) and from student LEAVE
  (get_pending_leave_requests). Never mix these up or answer a staff-attendance question using
  student data or vice versa.
- get_staff_attendance_by_date: answer "who is absent" from absentTeachers, "who was late" from
  lateTeachers, "who is on leave" from onLeaveTeachers — list them by name. If the "note" field is
  present, relay it plainly (an empty result on a given date usually means it's a non-working day,
  not that everyone was present) — do NOT say "no one was absent" or "everyone was present" in
  that case.
- get_staff_attendance_month_summary: lead with the school-wide totals (presentDays, lateDays,
  absentDays, halfDayDays, onLeaveDays, onTimePercentage) if asked for a general summary. For
  "which teachers have the most absences", list teachersWithMostAbsences by name with their
  absentDays count — if a teacher has 0 absences don't call special attention to them. If
  truncated is true, mention there are more teachers below the ones shown.
- Both tools are READ-ONLY — they never mark, change, or approve anything. There is currently no
  action tool for staff attendance (no send/notify workflow yet) — if asked to notify or warn
  about staff attendance, say that capability isn't available yet rather than implying it happened.

## When showing fee defaulters
- Lead with total defaulter count. If totalAmountDue is a number, include it. If totalAmountDue
  is null/missing, say the total amount isn't available (fee structure not configured for that
  class/session) — do NOT say "₹0" or imply nothing is owed; amountUnknownCount tells you how
  many students that affects.
- Break down by class if byClass has more than one entry — same null-means-unknown rule applies
  to each class's totalDue.
- Name the most overdue students specifically from mostOverdue.

## Attendance reminders — information vs. action (read carefully before choosing a tool)
- get_school_low_attendance_students is READ-ONLY. Use it for: "who has low attendance", "show
  me students below 75%", "which students need attendance follow-up". It never sends anything —
  just answer conversationally with what it returns, as described above.
- start_attendance_reminder_workflow is a WRITE action. Use it whenever the admin asks you to
  SEND, WARN, REMIND, or NOTIFY — e.g. "send attendance warnings", "notify parents of low
  attendance students", "warn students below 80% attendance" — including when they name specific
  students: the workflow always fetches the full authoritative below-threshold list itself and
  cannot be limited to only the students named in the message, so pass just
  session/className/threshold, never try to filter by name.
- start_attendance_reminder_workflow takes ONLY session, className, and threshold (all
  optional) — there is no parameter for recipients. You cannot choose who receives a warning;
  only the backend's own authoritative attendance data decides that.
- Calling start_attendance_reminder_workflow never sends any email by itself. It prepares a
  batch and pauses for the admin's explicit approval — the app shows an approval card with
  Approve/Reject buttons, and only clicking Approve actually sends anything. This is the SAME
  human-approval pattern as start_fee_reminder_workflow. There is no way to skip this step. If
  the admin says something like "send them immediately without asking me" — call
  start_attendance_reminder_workflow exactly as normal anyway; approval always still happens via
  the card, never via anything said in chat.
- Never say "I'm not able to send emails" or similar for an attendance warning request — this
  capability exists via start_attendance_reminder_workflow.
- When you call start_attendance_reminder_workflow, its result is shown directly as the approval
  card — do not additionally describe it in prose.

## Fee reminders — information vs. action (read carefully before choosing a tool)
- get_fee_defaulters is READ-ONLY. Use it for: "who hasn't paid", "show me fee defaulters",
  "how many students have pending fees", "which students haven't paid July fees". It never
  sends anything — just answer conversationally with what it returns, as described above.
- start_fee_reminder_workflow is a WRITE action. Use it whenever the admin asks you to SEND,
  REMIND, or NOTIFY — e.g. "send reminders to fee defaulters", "remind students with pending
  fees", "send fee reminder emails", "notify parents who haven't paid" — including when they
  name specific students (e.g. "send reminders to Himani and Bhumit"): the workflow always
  fetches the full authoritative defaulter list itself and cannot be limited to only the
  students named in the message, so pass just session/className, never try to filter by name.
- start_fee_reminder_workflow takes ONLY session and className (both optional) — there is no
  parameter for recipients. You cannot choose who receives a reminder; only the backend's own
  authoritative fee data decides that.
- Calling start_fee_reminder_workflow never sends any email by itself. It prepares a batch and
  pauses for the admin's explicit approval — the app shows an approval card with Approve/Reject
  buttons, and only clicking Approve actually sends anything. This is the SAME human-approval
  step the "Review fee reminder batch" button already uses; you are not bypassing it, you are
  entering the same flow. There is no way to skip this step. If the admin says something like
  "send them immediately without asking me" or "I approve, just send it" — call
  start_fee_reminder_workflow exactly as normal anyway; approval always still happens via the
  card, never via anything said in chat. If they push back after seeing the card, tell them
  plainly that sending always requires clicking Approve on the card — there is no way around it.
- Never say "I'm not able to send emails" or similar for a fee reminder request — this
  capability exists via start_fee_reminder_workflow.
- When you call start_fee_reminder_workflow, its result is shown directly as the approval card
  — do not additionally describe it in prose.

## Leave requests — information vs. action (read carefully)
- get_pending_leave_requests and get_leave_request_details are READ-ONLY. Use them for "who has
  applied for leave", "show me pending leave requests", "what did Riley ask for". They change
  nothing.
- start_leave_decision_workflow is a WRITE action. Use it when asked to APPROVE, REJECT, GRANT,
  DENY or DECLINE leave — including for requests already discussed in this conversation, e.g.
  "approve the first two", "reject that one", "approve all of them".
- NEVER invent, guess, increment or reuse a leaveId. Every id you pass MUST have come from
  get_pending_leave_requests or get_leave_request_details EARLIER IN THIS CONVERSATION. If the
  user asks you to act but you have no real ids yet, call a read tool first in the same turn,
  then start the workflow with the ids it returned. If a read tool returns nothing, say so — do
  not proceed with a guessed id.
- When the user refers to requests positionally ("the first two", "the last one", "Riley's"),
  resolve them against the list you actually retrieved, and pass only those ids.
- Calling start_leave_decision_workflow never changes a leave by itself. It prepares the decision
  and pauses for explicit human approval — the app shows a card with Approve/Reject, and only
  clicking Approve applies anything. There is no way to skip this. If the user says "just do it
  without asking" — call the tool exactly as normal anyway; approval always still happens on the
  card, never via anything said in chat.
- Requests that someone else already decided are shown on the card but are NOT changed. If the
  tool result's actionableCount is lower than leaveCount, mention that some were already decided.
- When you call start_leave_decision_workflow, its result IS the approval card — do not describe
  it again in prose.

## When showing leave requests
- Lead with how many are awaiting a decision. List each with the student's name, the date, and
  the reason — those are what a reviewer needs to decide.
- Give the dates exactly as returned; never reformat a date into a different calendar day.
- If a request's status is not PENDING, say so plainly rather than implying it still needs action.

## When answering from the knowledge base (search_knowledge_base)
- Treat the returned chunks as the ONLY source for policy content — never add typical/generic school-policy knowledge from general training, even if it sounds plausible or is commonly true elsewhere.
- Preserve the exact modal strength of the source wording: "may" is not "must" or "is required"; "should" is not "must"; "can" is not "will". Never upgrade optional/permissive language into a requirement, or soften a requirement into an option.
- If a chunk doesn't state a specific number, amount, deadline, or procedure, say so — do NOT invent one. Example: a chunk saying "lost books may require replacement or payment according to library rules" has no stated amount — say the policy doesn't specify an exact fee, don't name a dollar figure.
- Each result has a similarity score; results below a relevance floor are already excluded server-side. If found=false, or the returned chunks don't actually address what was asked, say plainly that the Knowledge Base doesn't have enough information on this — do NOT answer from general knowledge, and do NOT imply a negative ("the school doesn't offer this") when the real situation is "not covered by the documents."
- Keep this evidence separate from live tool data (attendance/fees/marks) — never blend a retrieved policy detail with a number from another tool.
- Never infer the inverse or opposite of a stated rule. A chunk saying a requirement applies under condition X (e.g. "for absences of 3+ days") says nothing about what happens when X doesn't hold — do not add "for shorter absences, it's generally not required" or similar unless the text actually says that. If the user's exact scenario isn't addressed, say the policy doesn't cover that specific case rather than assuming the opposite of what's stated.
- For a simple factual policy lookup, answer in one sentence stating the rule as given, then cite — don't add extra caveats, exceptions, or related scenarios the user didn't ask about.

## General behaviour
- Never guess or fabricate numbers, student names, class names, or subjects — only report what tools return.
- If a tool returns an "error" field, relay that message plainly rather than guessing at data.
- Use emojis sparingly — only where they add meaning, not as decoration. Never use a cheerful emoji (😊 😄) when reporting a problem like overdue fees, low attendance, or weak performance."""

_DEFAULT_PROMPT_BODY = """No specific data tools are available for your role yet. Answer general questions about Edunexify only — do not claim to fetch live data."""


async def _build_system_prompt(user: UserContext, access_token: str) -> str:
    today = date.today().isoformat()
    session = await current_academic_session(access_token)

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
    allowed = _TOOL_NAMES_BY_ROLE.get(user.role, set())
    if tool_name not in allowed:
        return {"error": f"Tool '{tool_name}' is not available for role '{user.role}'."}

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

    if tool_name == "get_consecutively_absent_students":
        return await get_consecutively_absent_students(
            user=user,
            access_token=access_token,
            minDays=tool_input.get("minDays", 3),
            session=tool_input.get("session"),
        )

    if tool_name == "start_teacher_attendance_reminder_workflow":
        return await start_teacher_attendance_reminder_workflow(
            user=user,
            access_token=access_token,
            session=tool_input.get("session"),
            threshold=tool_input.get("threshold", 75.0),
            criterion=tool_input.get("criterion", "BELOW_THRESHOLD"),
            minConsecutiveDays=tool_input.get("minConsecutiveDays"),
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

    if tool_name == "start_attendance_reminder_workflow":
        return await start_attendance_reminder_workflow(
            user=user,
            access_token=access_token,
            session=tool_input.get("session"),
            className=tool_input.get("className"),
            threshold=tool_input.get("threshold", 75.0),
        )

    if tool_name == "get_staff_attendance_by_date":
        return await get_staff_attendance_by_date(
            user=user,
            access_token=access_token,
            date=tool_input.get("date"),
        )

    if tool_name == "get_staff_attendance_month_summary":
        return await get_staff_attendance_month_summary(
            user=user,
            access_token=access_token,
            month=tool_input.get("month"),
            year=tool_input.get("year"),
        )

    if tool_name == "get_fee_defaulters":
        return await get_fee_defaulters(
            user=user,
            access_token=access_token,
            session=tool_input.get("session"),
            className=tool_input.get("className"),
        )

    if tool_name == "start_fee_reminder_workflow":
        return await start_fee_reminder_workflow(
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

    if tool_name == "get_pending_leave_requests":
        return await get_pending_leave_requests(
            user=user,
            access_token=access_token,
            studentId=tool_input.get("studentId"),
            limit=tool_input.get("limit", 25),
        )

    if tool_name == "get_leave_request_details":
        return await get_leave_request_details(
            user=user,
            access_token=access_token,
            leaveIds=tool_input.get("leaveIds") or [],
        )

    if tool_name == "start_leave_decision_workflow":
        return await start_leave_decision_workflow(
            user=user,
            access_token=access_token,
            decision=tool_input.get("decision", ""),
            leaveIds=tool_input.get("leaveIds") or [],
        )

    if tool_name == "search_knowledge_base":
        return await search_knowledge_base(
            user=user,
            access_token=access_token,
            query=tool_input["query"],
        )

    return {"error": f"Unknown tool '{tool_name}'."}


async def _execute_tool_traced(
    tool_name: str,
    tool_input: dict,
    user: UserContext,
    access_token: str,
    trace: Trace,
) -> dict:
    """Wraps _execute_tool with timing + trace recording, used by both /chat and
    /chat/stream's tool-calling loops so this instrumentation lives in one place."""
    start = time.monotonic()
    try:
        result = await _execute_tool(tool_name, tool_input, user, access_token)
        error = result.get("error") if isinstance(result, dict) else None
    except Exception as e:
        result = {"error": str(e)}
        error = str(e)
    latency_ms = (time.monotonic() - start) * 1000

    trace.record_tool_call(
        tool_name, latency_ms, success=(error is None), error=error,
        result_summary=summarize_tool_result(result),
    )
    if tool_name == "search_knowledge_base" and isinstance(result, dict):
        trace.record_rag_results(result.get("results"))

    return result


# Tools that don't just answer a question — they start a LangGraph workflow (see
# tools/admin_fee_workflows.py, tools/admin_attendance_workflows.py). When one of these is
# called and actually produces a workflow, both /chat and /chat/stream short-circuit: skip
# the usual follow-up turn where the model would draft a sentence describing the result, and
# return the tool's own result directly as structured data instead, so Angular can render the
# matching approval card. This is what keeps "the AI must never autonomously send anything"
# true by construction — the model's role ends at deciding whether to call this tool; nothing
# here, or afterward, ever calls the approve endpoint.
# Which card each workflow-starting tool produces. This used to be inferred by sniffing for
# disjoint fields in the result (defaulterCount vs a top-level className vs neither), which was
# already subtle at three workflows and became genuinely fragile at four — the leave result also
# carries a top-level className, so the old ordering would have misrouted it to the teacher
# attendance card. The triggering tool name is known at both capture sites, so the mapping is now
# explicit and the result carries its own kind.
_WORKFLOW_TOOL_KINDS = {
    "start_fee_reminder_workflow": "fee_reminder_workflow",
    "start_attendance_reminder_workflow": "attendance_reminder_workflow",
    "start_teacher_attendance_reminder_workflow": "teacher_attendance_reminder_workflow",
    "start_leave_decision_workflow": "leave_decision_workflow",
}
_WORKFLOW_STARTING_TOOLS = set(_WORKFLOW_TOOL_KINDS)


def _as_workflow_result(tool_name: str, tool_output) -> dict | None:
    if tool_name not in _WORKFLOW_TOOL_KINDS:
        return None
    if not isinstance(tool_output, dict) or not tool_output.get("workflowId"):
        return None  # tool errored (e.g. access denied, Spring unreachable) — let the model explain it normally
    return {**tool_output, "kind": _WORKFLOW_TOOL_KINDS[tool_name]}


def _workflow_kind(result: dict) -> str:
    return result.get("kind") or "attendance_reminder_workflow"


def _workflow_memory_note(result: dict) -> str:
    # An empty batch (no_defaulters / no_low_attendance_students) routes straight to finish()
    # without ever pausing at the approval interrupt — confirmed live during E2E testing, where
    # the unconditional "awaiting approval" wording was saved into conversation memory even
    # though there was nothing to approve, which could mislead a follow-up turn (e.g. an admin
    # asking "did that send?" after a batch that never had anyone to send to).
    kind = _workflow_kind(result)

    if kind == "leave_decision_workflow":
        if result.get("status") == "no_actionable_requests":
            return ("[Looked up those leave requests — none of them are still awaiting a decision, "
                    "so no approval card was created.]")
        verb = "approve" if result.get("decision") == "APPROVED" else "reject"
        return (f"[Prepared a request to {verb} {result.get('actionableCount', 0)} leave request(s) "
                "— awaiting approval in the approval card. Nothing has changed yet.]")

    if kind == "fee_reminder_workflow":
        count = result.get("defaulterCount", 0)
        if result.get("status") == "no_defaulters":
            return "[Checked for fee defaulters — none found, so no reminder batch was started.]"
        return f"[Started a fee reminder review for {count} student(s) — awaiting the admin's approval in the approval card.]"

    count = result.get("studentCount", 0)
    if kind == "teacher_attendance_reminder_workflow":
        if result.get("status") == "no_low_attendance_students":
            return "[Checked for low-attendance students in your class — none found, so no reminder batch was started.]"
        return f"[Started an attendance reminder review for {count} student(s) in your class — awaiting your approval in the approval card.]"

    if result.get("status") == "no_low_attendance_students":
        return "[Checked for low-attendance students — none found, so no reminder batch was started.]"
    return f"[Started an attendance reminder review for {count} student(s) — awaiting the admin's approval in the approval card.]"


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
        {"role": "system", "content": await _build_system_prompt(request.user, request.accessToken)},
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

    trace = Trace(
        user=request.user,
        conversation_id=request.conversationId,
        user_message=request.message,
        access_token=request.accessToken,
    )

    # ─── Tool-calling loop ────────────────────────────────────────────────────
    reply_text: str | None = None
    workflow_result: dict | None = None
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

            _deepseek_start = time.monotonic()
            response = _client.chat.completions.create(**completion_kwargs)
            trace.record_deepseek_call(
                (time.monotonic() - _deepseek_start) * 1000,
                response.usage,
                response.choices[0].finish_reason,
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

                    tool_output = await _execute_tool_traced(
                        tool_call.function.name,
                        tool_input,
                        request.user,
                        request.accessToken,
                        trace,
                    )

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(tool_output),
                    })

                    workflow_result = workflow_result or _as_workflow_result(tool_call.function.name, tool_output)

                if workflow_result is not None:
                    break  # skip the usual follow-up turn — see _WORKFLOW_STARTING_TOOLS above

            else:
                # Unexpected finish_reason (e.g. "length") — bail out gracefully.
                break

    except APIStatusError as e:
        trace.errored = True
        trace.error_message = f"APIStatusError({e.status_code})"
        trace.final_reply = ""
        await trace.finish()
        if e.status_code == 402:
            raise HTTPException(status_code=503, detail="AI service is temporarily unavailable. Please try again later.")
        if e.status_code == 429:
            raise HTTPException(status_code=429, detail="AI service is rate-limited. Please wait a moment and try again.")
        raise HTTPException(status_code=502, detail=f"AI provider error ({e.status_code}).")

    if workflow_result is not None:
        reply_text = _workflow_memory_note(workflow_result)
        trace.workflow_id = workflow_result.get("workflowId")
        trace.final_reply = reply_text
        await trace.finish()
        await memory.append_turn(
            request.user.schoolId, request.user.userId, request.conversationId,
            history, request.message, reply_text,
        )
        return ChatResponse(reply=reply_text, workflow=workflow_result)

    if reply_text is None:
        reply_text = "I wasn't able to complete your request. Please try again."

    trace.final_reply = reply_text
    await trace.finish()

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
        {"role": "system", "content": await _build_system_prompt(request.user, request.accessToken)},
        *history,
        {"role": "user", "content": request.message},
    ]

    role_tools = {
        "STUDENT": STUDENT_TOOLS,
        "TEACHER": TEACHER_TOOLS,
        "ADMIN": ADMIN_TOOLS,
    }.get(request.user.role, [])

    trace = Trace(
        user=request.user,
        conversation_id=request.conversationId,
        user_message=request.message,
        access_token=request.accessToken,
    )

    full_reply = ""
    errored = False
    interrupted = False  # user hit "stop" — client gone, no point doing more work
    workflow_result: dict | None = None

    try:
        for _ in range(5):  # Safety cap — prevents infinite loops, same as /chat
            completion_kwargs: dict = {
                "model": "deepseek-chat", "messages": messages, "stream": True, "temperature": 0.3,
                "stream_options": {"include_usage": True},
            }
            if role_tools:
                completion_kwargs["tools"] = role_tools
                completion_kwargs["tool_choice"] = "auto"

            _deepseek_start = time.monotonic()
            deepseek_usage = None
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

                if chunk.usage:
                    deepseek_usage = chunk.usage

                if not chunk.choices:
                    # The extra usage-only final chunk (stream_options.include_usage) has
                    # no choices at all — nothing else to process for it.
                    continue

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

            trace.record_deepseek_call((time.monotonic() - _deepseek_start) * 1000, deepseek_usage, finish_reason)

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
                    tool_output = await _execute_tool_traced(
                        c["name"], tool_input, request.user, request.accessToken, trace,
                    )
                    messages.append({
                        "role": "tool",
                        "tool_call_id": c["id"],
                        "content": json.dumps(tool_output),
                    })
                    workflow_result = workflow_result or _as_workflow_result(c["name"], tool_output)

                if workflow_result is not None:
                    break  # skip the follow-up turn — yielded as structured data below instead
                continue  # next iteration streams the follow-up turn

            break  # finish_reason == "stop" (or unexpected) — done

    except APIStatusError as e:
        errored = True
        trace.error_message = f"APIStatusError({e.status_code})"
        if e.status_code == 402 or e.status_code == 429:
            msg = "\n\n⚠️ AI Copilot is temporarily unavailable. Please try again later."
        else:
            msg = f"\n\n⚠️ AI provider error ({e.status_code}). Please try again."
        full_reply += msg
        yield msg
    except Exception as e:
        errored = True
        trace.error_message = str(e)
        msg = "\n\n⚠️ AI Copilot is temporarily unavailable. Please try again later."
        full_reply += msg
        yield msg

    if workflow_result is not None and not interrupted:
        # The entire remaining body IS the structured payload — no text was streamed for
        # this turn (any held-back preamble was already discarded once the tool call
        # started, same as every other tool call). Angular's stream-completion handler
        # recognizes this by attempting to parse the full accumulated body as JSON and
        # checking `kind`, falling back to plain text if that fails — see
        # ai-copilot.component.ts. This is the entire "protocol extension": no sentinels,
        # no header/trailer tricks, the transport (chunked text/plain) is unchanged.
        full_reply = json.dumps({"kind": _workflow_kind(workflow_result), **workflow_result})
        yield full_reply
    elif not full_reply and not interrupted:
        full_reply = "I wasn't able to complete your request. Please try again."
        yield full_reply

    # Only persist a clean, complete reply — never a partial/errored/interrupted one, so
    # a broken or stopped turn doesn't pollute the next turn's memory with a garbled reply.
    # A workflow start persists a short synthetic note instead of the raw JSON payload, so
    # future turns' context reconstruction reads naturally rather than replaying a JSON blob.
    if workflow_result is not None and not interrupted:
        await memory.append_turn(
            request.user.schoolId, request.user.userId, request.conversationId,
            history, request.message, _workflow_memory_note(workflow_result),
        )
    elif not errored and not interrupted:
        await memory.append_turn(
            request.user.schoolId,
            request.user.userId,
            request.conversationId,
            history,
            request.message,
            full_reply,
        )

    # Fired only after every chunk has already been yielded above — never adds to
    # perceived streaming latency (finish() itself is fire-and-forget besides).
    if workflow_result is not None:
        trace.workflow_id = workflow_result.get("workflowId")
        trace.final_reply = _workflow_memory_note(workflow_result)  # readable in trace logs, not the raw JSON
    else:
        trace.final_reply = full_reply
    trace.errored = errored
    trace.interrupted = interrupted
    await trace.finish()


@router.post("/chat/stream")
async def chat_stream(
    request: ChatRequest,
    http_request: Request,
    x_internal_secret: str = Header(alias="X-Internal-Secret"),
) -> StreamingResponse:
    if x_internal_secret != settings.internal_secret:
        raise HTTPException(status_code=401, detail="Unauthorized")

    return StreamingResponse(_stream_chat_response(request, http_request), media_type="text/plain; charset=utf-8")
