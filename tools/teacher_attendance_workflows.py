"""
tools/teacher_attendance_workflows.py — lets TEACHER chat requests enter the teacher-scoped
Low-Attendance Warning LangGraph workflow (graphs/teacher_attendance_reminder_workflow.py)
instead of only describing it.

Direct structural mirror of tools/admin_attendance_workflows.py — a thin httpx wrapper around
Spring's POST /api/ai/workflows/teacher-attendance-reminders. See that file's module docstring
for the full rationale (why this goes through Spring rather than the graph/checkpointer
directly) — not repeated here.

No className parameter here on purpose, same reasoning as every tool in tools/teacher_attendance.py:
Spring resolves className itself, server-side, from the teacher's own classTeacher field (see
AiTeacherAttendanceWorkflowController.start) — there is no argument the model could pass that
would widen this beyond the caller's own class.

No recipient parameter either, same reasoning as tools/admin_attendance_workflows.py: the model
can never hand this tool a list of student IDs/names, so there is no path from "the LLM decided
who to email" to an actual recipient list.
"""
import httpx

from academic_calendar import current_academic_session
from config import settings
from schemas.chat import UserContext


async def start_teacher_attendance_reminder_workflow(
    user: UserContext,
    access_token: str,
    session: str | None = None,
    threshold: float = 75.0,
    criterion: str = "BELOW_THRESHOLD",
    minConsecutiveDays: int | None = None,
) -> dict:
    payload: dict = {
        "session": session or await current_academic_session(access_token),
        "threshold": threshold,
        "criterion": criterion,
        "minConsecutiveDays": minConsecutiveDays,
    }

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{settings.spring_boot_url}/api/ai/workflows/teacher-attendance-reminders",
                json=payload,
                cookies={"accessToken": access_token},
                timeout=30.0,  # this call runs the graph up to the interrupt — not instant
            )
    except httpx.HTTPError as e:
        return {"error": f"Could not start the attendance reminder review: {e}"}

    if response.status_code == 403:
        return {"error": "Access denied. You must be assigned as a class teacher to send attendance reminders."}
    if response.status_code != 200:
        return {"error": f"Spring Boot returned {response.status_code} starting the attendance reminder workflow: {response.text[:300]}"}

    return response.json()
