"""
tools/admin_attendance_workflows.py — lets ADMIN chat requests enter the Low-Attendance Warning
LangGraph workflow (graphs/attendance_reminder_workflow.py) instead of only describing it.

Direct structural mirror of tools/admin_fee_workflows.py — a thin httpx wrapper around Spring's
POST /api/ai/workflows/attendance-reminders, the EXACT SAME endpoint the Angular quick-action
button calls. See that file's module docstring for the full rationale (why this goes through
Spring rather than the graph/checkpointer directly, why there's exactly one code path with two
callers) — not repeated here.

No recipient parameter exists here on purpose, same reasoning as the fee workflow: the model can
never hand this tool a list of student IDs/names, so there is no path from "the LLM decided who
to email" to an actual recipient list. See routers/chat.py's system prompt for the corresponding
instruction.
"""
import httpx

from academic_calendar import current_academic_session
from config import settings
from schemas.chat import UserContext


async def start_attendance_reminder_workflow(
    user: UserContext,
    access_token: str,
    session: str | None = None,
    className: str | None = None,
    threshold: float = 75.0,
) -> dict:
    payload: dict = {
        "session": session or await current_academic_session(access_token),
        "threshold": threshold,
    }
    if className:
        payload["className"] = className

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{settings.spring_boot_url}/api/ai/workflows/attendance-reminders",
                json=payload,
                cookies={"accessToken": access_token},
                timeout=30.0,  # this call runs the graph up to the interrupt — not instant
            )
    except httpx.HTTPError as e:
        return {"error": f"Could not start the attendance reminder review: {e}"}

    if response.status_code == 403:
        return {"error": "Access denied. Only admins can start an attendance reminder review."}
    if response.status_code != 200:
        return {"error": f"Spring Boot returned {response.status_code} starting the attendance reminder workflow: {response.text[:300]}"}

    return response.json()
