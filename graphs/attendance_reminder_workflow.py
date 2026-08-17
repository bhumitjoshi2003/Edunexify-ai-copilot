"""
graphs/attendance_reminder_workflow.py — the Low-Attendance Warning human-in-the-loop workflow.

    load_low_attendance_students -> prepare_reminder_draft -> request_admin_approval (interrupt)
        -[resume: approved]-> send_reminders -> finish
        -[resume: rejected]-> finish
    (load error, or student_count == 0) -> finish / handle_failure
    (send_reminders hard error, not a partial failure) -> handle_failure -> finish

Direct structural mirror of graphs/fee_reminder_workflow.py — same interrupt()/Command(resume=...)
mechanism, same reasons (see that file's module docstring for the full rationale, not repeated
here). The two workflows are deliberately kept as separate graphs/files rather than parameterized
into one, matching how tools/admin_fees.py and tools/admin_attendance.py are already separate
files for separate domains — a shared "batch reminder" abstraction would need to paper over the
different data shapes (totalDue vs attendancePercentage) for no real benefit at this size.

Same critical rule as the fee workflow: request_admin_approval does nothing but call interrupt()
and return its result — no side effects, because a resumed node re-executes its entire body.

The access token is threaded through via config["configurable"]["access_token"] on every
invocation, never through state — see fee_reminder_workflow.py's docstring for why.
"""
import logging
from datetime import datetime, timezone

import httpx
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from openai import AsyncOpenAI

from checkpointer import RedisCheckpointSaver
from config import settings
from graphs.state import (
    AttendanceReminderWorkflowState,
    AttendanceWorkflowStatus,
    LowAttendanceStudentLite,
    SendOutcome,
    SendResult,
)

logger = logging.getLogger(__name__)

_async_client = AsyncOpenAI(api_key=settings.deepseek_api_key, base_url="https://api.deepseek.com")

# Same reasoning as fee_reminder_workflow.py's MAX_OUTCOMES_STORED.
MAX_OUTCOMES_STORED = 50


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _touch(state: AttendanceReminderWorkflowState) -> dict:
    return {"updated_at": _now_iso()}


# ─── Nodes ─────────────────────────────────────────────────────────────────────

async def load_low_attendance_students(state: AttendanceReminderWorkflowState, config: RunnableConfig) -> dict:
    access_token = config["configurable"]["access_token"]

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{settings.spring_boot_url}/api/attendance/summary/school",
                params={"type": "year", "session": state.session},
                cookies={"accessToken": access_token},
                timeout=15.0,
            )
    except httpx.HTTPError as e:
        logger.warning("load_low_attendance_students: Spring call failed: %s", e)
        return {"error": f"Could not reach attendance data: {e}", **_touch(state)}

    if response.status_code == 403:
        return {"error": "Access denied fetching attendance data.", **_touch(state)}
    if response.status_code != 200:
        return {"error": f"Spring Boot returned {response.status_code} loading attendance summary.", **_touch(state)}

    raw = response.json()
    students = [
        LowAttendanceStudentLite(
            studentId=s["studentId"],
            studentName=s.get("studentName") or s["studentId"],
            className=s.get("className") or "",
            attendancePercentage=s["attendancePercentage"],
            daysPresent=s.get("daysPresent", 0),
            daysAbsent=s.get("daysAbsent", 0),
            totalWorkingDays=s.get("totalWorkingDays", 0),
        )
        for s in raw
        if s["attendancePercentage"] < state.threshold
        and (state.class_name is None or s.get("className") == state.class_name)
    ]
    students.sort(key=lambda s: s.attendancePercentage)

    return {
        "students": students,
        "student_count": len(students),
        **_touch(state),
    }


def route_after_load(state: AttendanceReminderWorkflowState) -> str:
    if state.error:
        return "handle_failure"
    if state.student_count == 0:
        return "no_low_attendance_students"
    return "prepare_reminder_draft"


async def prepare_reminder_draft(state: AttendanceReminderWorkflowState, config: RunnableConfig) -> dict:
    fallback = (
        f"{state.student_count} student(s) are below {state.threshold:.0f}% attendance"
        f" for session {state.session}"
        + (f" (class {state.class_name})" if state.class_name else "")
        + ". Review and approve to send attendance warning emails using the standard template."
    )

    # Real per-student percentages, not just names — a prior version of this prompt asked
    # the model to "mention example names with their attendance percentage" while only ever
    # supplying bare names, and it reliably invented a plausible-looking (but wrong) number
    # to fill the gap (confirmed live during E2E testing: reported "74.0%" for a student who
    # was actually at 30.0%). Giving it the real figure removes the reason to guess.
    sample_examples = "; ".join(f"{s.studentName} ({s.attendancePercentage:.1f}%)" for s in state.students[:5])
    prompt = (
        "Write a short, professional 2-3 sentence summary for a SCHOOL ADMIN reviewing a batch "
        "of low-attendance warning emails before sending. This text is shown only to the admin, "
        "never to parents. State how many students are affected and the attendance threshold "
        "used. You may mention a couple of example students by name — if you state a specific "
        "attendance percentage for a named student, it MUST be exactly the figure given for "
        "them in exampleStudents below; never estimate, round to a different value, or invent "
        "one. End by inviting the admin to approve or reject sending.\n\n"
        f"studentCount: {state.student_count}\n"
        f"threshold: {state.threshold}\n"
        f"session: {state.session}\n"
        f"className: {state.class_name or 'all classes'}\n"
        f"exampleStudents (name and their real attendance percentage): {sample_examples}"
    )

    try:
        response = await _async_client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=200,
            timeout=15.0,
        )
        draft = (response.choices[0].message.content or "").strip() or fallback
    except Exception as e:  # cosmetic text only — never block the workflow on a DeepSeek hiccup
        logger.warning("prepare_reminder_draft: DeepSeek call failed, using fallback: %s", e)
        draft = fallback

    return {"draft_summary": draft, **_touch(state)}


async def request_admin_approval(state: AttendanceReminderWorkflowState) -> dict:
    # Nothing but the interrupt call belongs here — see module docstring for why.
    decision = interrupt({
        "workflowId": state.workflow_id,
        "studentCount": state.student_count,
        "threshold": state.threshold,
        "draftSummary": state.draft_summary,
        "session": state.session,
        "className": state.class_name,
    })
    return {"approval_decision": decision}


def route_after_approval(state: AttendanceReminderWorkflowState) -> str:
    if state.approval_decision == "approved":
        return "send_reminders"
    if state.approval_decision == "rejected":
        return "rejected"
    # Defensive only — routers/workflows_attendance_reminders.py validates the decision
    # value before ever invoking resume, so this should be unreachable.
    return "handle_failure"


async def send_reminders(state: AttendanceReminderWorkflowState, config: RunnableConfig) -> dict:
    access_token = config["configurable"]["access_token"]
    student_ids = [s.studentId for s in state.students]

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{settings.spring_boot_url}/api/ai/workflows/attendance-reminders/{state.workflow_id}/dispatch",
                json={"studentIds": student_ids, "session": state.session},
                cookies={"accessToken": access_token},
                timeout=60.0,  # a real bulk email loop — give it real headroom
            )
    except httpx.HTTPError as e:
        logger.warning("send_reminders: Spring dispatch call failed: %s", e)
        return {
            "send_result": SendResult(error=f"Could not reach Spring to send reminders: {e}"),
            **_touch(state),
        }

    if response.status_code != 200:
        return {
            "send_result": SendResult(error=f"Spring Boot returned {response.status_code} dispatching reminders."),
            **_touch(state),
        }

    body = response.json()
    outcomes = [
        SendOutcome(studentId=o["studentId"], status=o["status"])
        for o in (body.get("outcomes") or [])[:MAX_OUTCOMES_STORED]
    ]
    return {
        "send_result": SendResult(
            sentCount=body.get("sentCount", 0),
            failedCount=body.get("failedCount", 0),
            outcomes=outcomes,
        ),
        **_touch(state),
    }


def route_after_send(state: AttendanceReminderWorkflowState) -> str:
    if state.send_result and state.send_result.error:
        return "handle_failure"
    return "finish"


async def handle_failure(state: AttendanceReminderWorkflowState) -> dict:
    error = state.error or (state.send_result.error if state.send_result else None) or "Unknown error"
    return {"status": AttendanceWorkflowStatus.FAILED, "error": error, **_touch(state)}


async def finish(state: AttendanceReminderWorkflowState) -> dict:
    if state.status == AttendanceWorkflowStatus.FAILED:
        return {}  # handle_failure already set the terminal status
    if state.approval_decision == "rejected":
        return {"status": AttendanceWorkflowStatus.REJECTED, **_touch(state)}
    if state.student_count == 0:
        return {"status": AttendanceWorkflowStatus.NO_LOW_ATTENDANCE_STUDENTS, **_touch(state)}
    if state.send_result:
        # Three-way, not two-way: failedCount > 0 alone conflates "some sent, some failed"
        # with "every single send failed" — same fix as fee_reminder_workflow.py's finish(),
        # mirroring AiAttendanceWorkflowController.dispatch()'s sent/failed logic on the
        # Spring side for the exact same distinction.
        if state.send_result.failedCount == 0:
            status = AttendanceWorkflowStatus.SENT
        elif state.send_result.sentCount == 0:
            status = AttendanceWorkflowStatus.FAILED
        else:
            status = AttendanceWorkflowStatus.PARTIALLY_SENT
        return {"status": status, **_touch(state)}
    return {}


# ─── Graph assembly ────────────────────────────────────────────────────────────

def _build_graph():
    builder = StateGraph(AttendanceReminderWorkflowState)

    builder.add_node("load_low_attendance_students", load_low_attendance_students)
    builder.add_node("prepare_reminder_draft", prepare_reminder_draft)
    builder.add_node("request_admin_approval", request_admin_approval)
    builder.add_node("send_reminders", send_reminders)
    builder.add_node("handle_failure", handle_failure)
    builder.add_node("finish", finish)

    builder.add_edge(START, "load_low_attendance_students")
    builder.add_conditional_edges("load_low_attendance_students", route_after_load, {
        "handle_failure": "handle_failure",
        "no_low_attendance_students": "finish",
        "prepare_reminder_draft": "prepare_reminder_draft",
    })
    builder.add_edge("prepare_reminder_draft", "request_admin_approval")
    builder.add_conditional_edges("request_admin_approval", route_after_approval, {
        "send_reminders": "send_reminders",
        "rejected": "finish",
        "handle_failure": "handle_failure",
    })
    builder.add_conditional_edges("send_reminders", route_after_send, {
        "handle_failure": "handle_failure",
        "finish": "finish",
    })
    builder.add_edge("handle_failure", "finish")
    builder.add_edge("finish", END)

    return builder


# Explicitly allowlist this module's custom Enum/Pydantic types for checkpoint
# (de)serialization — see fee_reminder_workflow.py's identical block for why.
_serde = JsonPlusSerializer(
    allowed_msgpack_modules=[
        ("graphs.state", "AttendanceWorkflowStatus"),
        ("graphs.state", "LowAttendanceStudentLite"),
        ("graphs.state", "SendOutcome"),
        ("graphs.state", "SendResult"),
        ("graphs.state", "AttendanceReminderWorkflowState"),
    ],
)
_checkpointer = RedisCheckpointSaver(ttl_seconds=settings.workflow_checkpoint_ttl_seconds, serde=_serde)
graph = _build_graph().compile(checkpointer=_checkpointer)
