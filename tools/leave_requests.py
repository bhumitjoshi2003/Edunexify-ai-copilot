"""
tools/leave_requests.py — read-only leave tools for ADMIN and TEACHER.

Endpoint used: GET /api/leaves/for-review
  - Without ids: the PENDING review queue, oldest first.
  - With ids: those specific requests at their CURRENT status, so an already-decided request is
    reported honestly rather than vanishing.

Spring scopes every response to the caller's school, and — on this endpoint specifically —
confines a TEACHER to their own class regardless of what className is sent. These tools therefore
never take a className argument from the model, the same guarantee tools/teacher_attendance.py
relies on for user.className.

These are strictly READ tools. Nothing here can change a leave request; that requires
tools/leave_workflows.py, which pauses for human approval.
"""
import httpx

from config import settings
from schemas.chat import UserContext


async def _fetch(access_token: str, params: dict) -> list | dict:
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{settings.spring_boot_url}/api/leaves/for-review",
            params=params,
            cookies={"accessToken": access_token},
            timeout=10.0,
        )
    if response.status_code == 403:
        return {"error": "Access denied. You can only view leave requests for your own class."}
    if response.status_code != 200:
        return {"error": f"Spring Boot returned {response.status_code}: {response.text[:200]}"}
    return response.json()


def _slim(r: dict) -> dict:
    return {
        "leaveId": r["id"],
        "studentId": r.get("studentId"),
        "studentName": r.get("studentName"),
        "className": r.get("className"),
        "leaveDate": r.get("leaveDate"),
        "reason": r.get("reason"),
        "status": r.get("status"),
    }


async def get_pending_leave_requests(
    user: UserContext,
    access_token: str,
    studentId: str | None = None,
    limit: int = 25,
) -> dict:
    """The queue of leave requests still awaiting a decision, oldest first."""
    params: dict = {"limit": max(1, min(limit, 100))}
    if studentId:
        params["studentId"] = studentId

    raw = await _fetch(access_token, params)
    if isinstance(raw, dict):
        return raw  # error dict

    scope = user.className if user.role == "TEACHER" else "the whole school"
    if not raw:
        return {
            "scope": scope,
            "pendingCount": 0,
            "message": "There are no leave requests awaiting a decision.",
        }

    return {
        "scope": scope,
        "pendingCount": len(raw),
        # leaveId is what the decision workflow acts on — always surface it so a follow-up
        # like "approve the first two" can be resolved to real requests.
        "leaveRequests": [_slim(r) for r in raw],
    }


async def get_leave_request_details(
    user: UserContext,
    access_token: str,
    leaveIds: list[int],
) -> dict:
    """Full detail for specific leave requests, at their current status."""
    if not leaveIds:
        return {"error": "At least one leaveId is required."}

    raw = await _fetch(access_token, {"ids": [str(i) for i in leaveIds]})
    if isinstance(raw, dict):
        return raw  # error dict

    found = [_slim(r) for r in raw]
    missing = sorted(set(leaveIds) - {r["leaveId"] for r in found})
    result: dict = {"leaveRequests": found}
    if missing:
        # Either they don't exist, or they're outside this caller's school/class. Reported
        # rather than silently dropped so the model never claims to have found them.
        result["notFoundOrOutOfScope"] = missing
    return result
