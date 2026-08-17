"""
tools/leave_workflows.py — lets ADMIN/TEACHER chat requests enter the Leave Decision LangGraph
workflow (graphs/leave_decision_workflow.py) instead of only describing it.

Thin httpx wrapper around Spring's POST /api/ai/workflows/leave-decisions, matching
tools/teacher_attendance_workflows.py. See that file for why this goes through Spring rather than
the graph directly.

UNLIKE the reminder workflow tools, this one DOES take a recipient-shaped argument — leaveIds —
because deciding leave is inherently about specific requests. That removes the reminder
workflows' structural guarantee that the model cannot choose targets, so the compensating
controls live server-side: Spring re-resolves every id within the caller's school (and the
teacher's own class), drops anything not still PENDING, and applies nothing until a human
approves the card. The ids sent from here are a proposal, not an instruction.

There is deliberately no className parameter: for a teacher, Spring derives it from their own
classTeacher field, so there is no argument the model could pass to widen scope.
"""
import httpx

from config import settings
from schemas.chat import UserContext


async def start_leave_decision_workflow(
    user: UserContext,
    access_token: str,
    decision: str,
    leaveIds: list[int],
) -> dict:
    normalized = (decision or "").strip().upper()
    if normalized in {"APPROVE", "APPROVED"}:
        normalized = "APPROVED"
    elif normalized in {"REJECT", "REJECTED", "DENY", "DENIED"}:
        normalized = "REJECTED"
    else:
        return {"error": "decision must be APPROVED or REJECTED."}

    if not leaveIds:
        return {"error": "No leave requests were identified. Look them up first, then say which ones to act on."}

    payload = {"decision": normalized, "leaveIds": list(dict.fromkeys(int(i) for i in leaveIds))}

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{settings.spring_boot_url}/api/ai/workflows/leave-decisions",
                json=payload,
                cookies={"accessToken": access_token},
                timeout=30.0,  # runs the graph up to the interrupt — not instant
            )
    except httpx.HTTPError as e:
        return {"error": f"Could not start the leave decision review: {e}"}

    if response.status_code == 403:
        try:
            detail = response.json().get("error")
        except Exception:
            detail = None
        return {"error": detail or "Access denied. You do not have permission to decide leave requests."}
    if response.status_code != 200:
        return {"error": f"Spring Boot returned {response.status_code} starting the leave decision workflow: {response.text[:300]}"}

    return response.json()
