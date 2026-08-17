"""
graphs/teacher_attendance_reminder_workflow.py — the teacher-scoped Low-Attendance Warning
human-in-the-loop workflow.

    load_low_attendance_students -> prepare_reminder_draft -> request_teacher_approval (interrupt)
        -[resume: approved]-> send_reminders -> finish
        -[resume: rejected]-> finish
    (load error, or student_count == 0) -> finish / handle_failure
    (send_reminders hard error, not a partial failure) -> handle_failure -> finish

Direct structural mirror of graphs/attendance_reminder_workflow.py (the ADMIN, school-wide
variant) — same interrupt()/Command(resume=...) mechanism, same three-way terminal-status
logic, same hallucination-resistant draft prompt. See that file's and
graphs/fee_reminder_workflow.py's module docstrings for the full rationale, not repeated here.

The one load-bearing difference: state.class_name is REQUIRED (never optional, never None) and
load_low_attendance_students calls the TEACHER-permitted endpoint
GET /api/attendance/summary/class/{className} — the same endpoint tools/teacher_attendance.py's
_fetch_class_attendance already uses — instead of the admin's school-wide
GET /api/attendance/summary/school. className is never taken from anywhere the LLM could
influence: it is resolved server-side by Spring (AiTeacherAttendanceWorkflowController.start,
mirroring AiProxyController.resolveClassName) before this graph is ever invoked, and Spring's own
endpoint independently re-verifies the requesting teacher still owns that class on every call —
the same defense-in-depth pattern already relied on throughout tools/teacher_attendance.py.

Same critical rule as the other workflows: request_teacher_approval does nothing but call
interrupt() and return its result — no side effects, because a resumed node re-executes its
entire body.

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
    AttendanceCriterion,
    AttendanceWorkflowStatus,
    LowAttendanceStudentLite,
    SendOutcome,
    SendResult,
    TeacherAttendanceReminderWorkflowState,
)

logger = logging.getLogger(__name__)

_async_client = AsyncOpenAI(api_key=settings.deepseek_api_key, base_url="https://api.deepseek.com")

# Same reasoning as fee_reminder_workflow.py's MAX_OUTCOMES_STORED.
MAX_OUTCOMES_STORED = 50


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _touch(state: TeacherAttendanceReminderWorkflowState) -> dict:
    return {"updated_at": _now_iso()}


# ─── Nodes ─────────────────────────────────────────────────────────────────────

async def load_low_attendance_students(state: TeacherAttendanceReminderWorkflowState, config: RunnableConfig) -> dict:
    """Loads the batch's students according to state.criterion.

    Both branches hit a class-scoped Spring endpoint that independently re-verifies the caller
    teaches this class, and neither takes className from anywhere the model could influence —
    it comes from state, which Spring itself populated from the teacher's own classTeacher field.
    The two differ only in *which* question they ask of the same attendance data.
    """
    access_token = config["configurable"]["access_token"]
    consecutive = state.criterion == AttendanceCriterion.CONSECUTIVE_ABSENCE

    if consecutive:
        url = f"{settings.spring_boot_url}/api/attendance/consecutive-absentees/class/{state.class_name}"
        params = {"minDays": state.min_consecutive_days or 3, "session": state.session}
    else:
        url = f"{settings.spring_boot_url}/api/attendance/summary/class/{state.class_name}"
        params = {"type": "year", "session": state.session}

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, params=params, cookies={"accessToken": access_token}, timeout=15.0)
    except httpx.HTTPError as e:
        logger.warning("load_low_attendance_students (teacher): Spring call failed: %s", e)
        return {"error": f"Could not reach attendance data: {e}", **_touch(state)}

    if response.status_code == 403:
        return {"error": "Access denied. You can only send reminders for your own assigned class.", **_touch(state)}
    if response.status_code != 200:
        return {"error": f"Spring Boot returned {response.status_code} loading attendance summary.", **_touch(state)}

    raw = response.json()

    def _lite(s: dict) -> LowAttendanceStudentLite:
        return LowAttendanceStudentLite(
            studentId=s["studentId"],
            studentName=s.get("studentName") or s["studentId"],
            className=s.get("className") or state.class_name,
            attendancePercentage=s.get("attendancePercentage", 0.0),
            daysPresent=s.get("daysPresent", 0),
            daysAbsent=s.get("daysAbsent", 0),
            totalWorkingDays=s.get("totalWorkingDays", 0),
            consecutiveAbsentDays=s.get("consecutiveAbsentDays"),
            absentDates=s.get("absentDates"),
        )

    if consecutive:
        # Spring already applied the streak filter and ordered by longest streak first — the
        # selection itself is never re-derived here, so the teacher's approval card and the email
        # Spring later sends can never disagree about who qualified.
        students = [_lite(s) for s in raw]
    else:
        students = [_lite(s) for s in raw if s["attendancePercentage"] < state.threshold]
        students.sort(key=lambda s: s.attendancePercentage)

    return {
        "students": students,
        "student_count": len(students),
        **_touch(state),
    }


def route_after_load(state: TeacherAttendanceReminderWorkflowState) -> str:
    if state.error:
        return "handle_failure"
    if state.student_count == 0:
        return "no_low_attendance_students"
    return "prepare_reminder_draft"


async def prepare_reminder_draft(state: TeacherAttendanceReminderWorkflowState, config: RunnableConfig) -> dict:
    consecutive = state.criterion == AttendanceCriterion.CONSECUTIVE_ABSENCE
    min_days = state.min_consecutive_days or 3

    if consecutive:
        fallback = (
            f"{state.student_count} student(s) in class {state.class_name} have been absent for "
            f"{min_days} or more consecutive school days. Review and approve to send attendance "
            "warning emails using the standard template."
        )
        criterion_line = f"selectionCriterion: absent for at least {min_days} consecutive school days"
        # Each student's REAL streak, so the model never has to guess a number to fill the gap.
        sample_examples = "; ".join(
            f"{s.studentName} ({s.consecutiveAbsentDays} consecutive days)" for s in state.students[:5]
        )
        examples_label = "exampleStudents (name and their real consecutive-absence count)"
        figure_rule = (
            "if you state a number of consecutive absent days for a named student, it MUST be "
            "exactly the figure given for them in exampleStudents below"
        )
    else:
        fallback = (
            f"{state.student_count} student(s) in class {state.class_name} are below "
            f"{state.threshold:.0f}% attendance for session {state.session}. Review and approve to "
            "send attendance warning emails using the standard template."
        )
        criterion_line = f"selectionCriterion: cumulative attendance below {state.threshold}%"
        sample_examples = "; ".join(f"{s.studentName} ({s.attendancePercentage:.1f}%)" for s in state.students[:5])
        examples_label = "exampleStudents (name and their real attendance percentage)"
        figure_rule = (
            "if you state a specific attendance percentage for a named student, it MUST be "
            "exactly the figure given for them in exampleStudents below"
        )

    # Real per-student figures, not just names — same hallucination fix as the admin workflow's
    # prompt (see graphs/attendance_reminder_workflow.py's docstring for the confirmed live
    # failure mode this avoids).
    prompt = (
        "Write a short, professional 2-3 sentence summary for a TEACHER reviewing a batch of "
        "attendance warning emails before sending to their own class's parents. This text is "
        "shown only to the teacher, never to parents. State how many students are affected and "
        f"the criterion used. You may mention a couple of example students by name — {figure_rule}; "
        "never estimate, round to a different value, or invent one. End by inviting the teacher "
        "to approve or reject sending.\n\n"
        f"studentCount: {state.student_count}\n"
        f"{criterion_line}\n"
        f"session: {state.session}\n"
        f"className: {state.class_name}\n"
        f"{examples_label}: {sample_examples}"
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
        logger.warning("prepare_reminder_draft (teacher): DeepSeek call failed, using fallback: %s", e)
        draft = fallback

    return {"draft_summary": draft, **_touch(state)}


async def request_teacher_approval(state: TeacherAttendanceReminderWorkflowState) -> dict:
    # Nothing but the interrupt call belongs here — see module docstring for why.
    decision = interrupt({
        "workflowId": state.workflow_id,
        "studentCount": state.student_count,
        "threshold": state.threshold,
        "criterion": state.criterion.value,
        "minConsecutiveDays": state.min_consecutive_days,
        "draftSummary": state.draft_summary,
        "session": state.session,
        "className": state.class_name,
    })
    return {"approval_decision": decision}


def route_after_approval(state: TeacherAttendanceReminderWorkflowState) -> str:
    if state.approval_decision == "approved":
        return "send_reminders"
    if state.approval_decision == "rejected":
        return "rejected"
    # Defensive only — routers/workflows_teacher_attendance_reminders.py validates the
    # decision value before ever invoking resume, so this should be unreachable.
    return "handle_failure"


async def send_reminders(state: TeacherAttendanceReminderWorkflowState, config: RunnableConfig) -> dict:
    access_token = config["configurable"]["access_token"]
    student_ids = [s.studentId for s in state.students]

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{settings.spring_boot_url}/api/ai/workflows/teacher-attendance-reminders/{state.workflow_id}/dispatch",
                json={"studentIds": student_ids, "session": state.session},
                cookies={"accessToken": access_token},
                timeout=60.0,  # a real bulk email loop — give it real headroom
            )
    except httpx.HTTPError as e:
        logger.warning("send_reminders (teacher): Spring dispatch call failed: %s", e)
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


def route_after_send(state: TeacherAttendanceReminderWorkflowState) -> str:
    if state.send_result and state.send_result.error:
        return "handle_failure"
    return "finish"


async def handle_failure(state: TeacherAttendanceReminderWorkflowState) -> dict:
    error = state.error or (state.send_result.error if state.send_result else None) or "Unknown error"
    return {"status": AttendanceWorkflowStatus.FAILED, "error": error, **_touch(state)}


async def finish(state: TeacherAttendanceReminderWorkflowState) -> dict:
    if state.status == AttendanceWorkflowStatus.FAILED:
        return {}  # handle_failure already set the terminal status
    if state.approval_decision == "rejected":
        return {"status": AttendanceWorkflowStatus.REJECTED, **_touch(state)}
    if state.student_count == 0:
        return {"status": AttendanceWorkflowStatus.NO_LOW_ATTENDANCE_STUDENTS, **_touch(state)}
    if state.send_result:
        # Three-way, not two-way — same fix as the admin/fee workflows' finish(), mirroring
        # AiTeacherAttendanceWorkflowController.dispatch()'s sent/failed logic on the Spring side.
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
    builder = StateGraph(TeacherAttendanceReminderWorkflowState)

    builder.add_node("load_low_attendance_students", load_low_attendance_students)
    builder.add_node("prepare_reminder_draft", prepare_reminder_draft)
    builder.add_node("request_teacher_approval", request_teacher_approval)
    builder.add_node("send_reminders", send_reminders)
    builder.add_node("handle_failure", handle_failure)
    builder.add_node("finish", finish)

    builder.add_edge(START, "load_low_attendance_students")
    builder.add_conditional_edges("load_low_attendance_students", route_after_load, {
        "handle_failure": "handle_failure",
        "no_low_attendance_students": "finish",
        "prepare_reminder_draft": "prepare_reminder_draft",
    })
    builder.add_edge("prepare_reminder_draft", "request_teacher_approval")
    builder.add_conditional_edges("request_teacher_approval", route_after_approval, {
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
        ("graphs.state", "AttendanceCriterion"),
        ("graphs.state", "AttendanceWorkflowStatus"),
        ("graphs.state", "LowAttendanceStudentLite"),
        ("graphs.state", "SendOutcome"),
        ("graphs.state", "SendResult"),
        ("graphs.state", "TeacherAttendanceReminderWorkflowState"),
    ],
)
_checkpointer = RedisCheckpointSaver(ttl_seconds=settings.workflow_checkpoint_ttl_seconds, serde=_serde)
graph = _build_graph().compile(checkpointer=_checkpointer)
