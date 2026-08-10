"""
tools/admin_fee_workflows.py — lets ADMIN chat requests enter the existing Fee Defaulter
Reminder LangGraph workflow (graphs/fee_reminder_workflow.py) instead of only describing it.

Deliberately NOT a call into the graph/checkpointer directly. This is a thin httpx wrapper
around Spring's POST /api/ai/workflows/fee-reminders — the EXACT SAME endpoint the Angular
quick-action button calls (AiWorkflowController.start()). That endpoint is what creates the
ai_fee_reminder_batch row (tenant-isolation + idempotency anchor) before ever calling the
graph — calling the graph's start_workflow() directly from here would skip that row entirely
and silently break the approve/reject endpoints' tenant check. Going through Spring here means
there is exactly ONE code path that starts this workflow, with two callers (Angular directly,
and this tool acting as a second authenticated caller on the admin's behalf) — not a second
implementation.

No recipient parameter exists here on purpose (only session/className, both optional filters
on Spring's own authoritative defaulter query) — the model can never hand this tool a list of
student IDs/names, so there is no path from "the LLM decided who to email" to an actual
recipient list. See routers/chat.py's system prompt for the corresponding instruction.
"""
import httpx

from config import settings
from schemas.chat import UserContext


async def start_fee_reminder_workflow(
    user: UserContext,
    access_token: str,
    session: str | None = None,
    className: str | None = None,
) -> dict:
    from tools.attendance import current_academic_session  # local import avoids a module cycle

    payload: dict = {"session": session or current_academic_session()}
    if className:
        payload["className"] = className

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{settings.spring_boot_url}/api/ai/workflows/fee-reminders",
                json=payload,
                cookies={"accessToken": access_token},
                timeout=30.0,  # this call runs the graph up to the interrupt — not instant
            )
    except httpx.HTTPError as e:
        return {"error": f"Could not start the fee reminder review: {e}"}

    if response.status_code == 403:
        return {"error": "Access denied. Only admins can start a fee reminder review."}
    if response.status_code != 200:
        return {"error": f"Spring Boot returned {response.status_code} starting the fee reminder workflow: {response.text[:300]}"}

    return response.json()
